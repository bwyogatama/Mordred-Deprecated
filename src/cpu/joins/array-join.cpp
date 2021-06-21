#include "types.h"
#include "generator.h"
//#include "affinity.h"
#include "barrier.h"
#include "cpu_mapping.h"
#include "rdtsc.h"
#include <pthread.h>
#include <nmmintrin.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cstdio>

#include "utils/cpu_utils.h"

using namespace std;

#define CACHE_LINE_SIZE 64

#ifndef BARRIER_ARRIVE
/** barrier wait macro */
#define BARRIER_ARRIVE(B,RV)                            \
    RV = pthread_barrier_wait(B);                       \
    if(RV !=0 && RV != PTHREAD_BARRIER_SERIAL_THREAD){  \
        printf("Couldn't wait on barrier\n");           \
        exit(EXIT_FAILURE);                             \
    }
#endif

/** checks malloc() result */
#ifndef MALLOC_CHECK
#define MALLOC_CHECK(M)                                                 \
    if(!M){                                                             \
        printf("[ERROR] MALLOC_CHECK: %s : %d\n", __FILE__, __LINE__);  \
        perror(": malloc() failed!\n");                                 \
        exit(EXIT_FAILURE);                                             \
    }
#endif

static void *
alloc_aligned(size_t size)
{
    void * ret;
    int rv;
    rv = posix_memalign((void**)&ret, CACHE_LINE_SIZE, size);

    if (rv) {
        perror("alloc_aligned() failed: out of memory");
        return 0;
    }

    return ret;
}

struct thr_arg_t {
  int         tid;
  uint64_t      matches;
  uint64_t      checksum;
  value_t *       array;
  tuple_t *       relR;
  tuple_t *     relS;
  uint64_t        numR;
  uint64_t        numS;
  uint64_t      totalR;
  pthread_barrier_t * barrier;
  struct timeval *  start;
  struct timeval *  end;
};
typedef struct thr_arg_t thr_arg_t;

template<bool is_checksum>
void *
array_join_thread(void * args)
{
  int rv;
  uint64_t matches = 0;
  uint64_t checksum = 0;
  uint64_t i = 0;

  thr_arg_t *arg = (thr_arg_t*)(args);
  const tuple_t * relR = arg->relR;
  const tuple_t * relS = arg->relS;
  value_t * array = arg->array;
  const uint64_t numR = arg->numR;
  const uint64_t numS = arg->numS;
  const uint64_t totalR = arg->totalR;

#ifdef ALGO_TIME
  if (arg->tid == 0)
    gettimeofday(arg->start, NULL);
  BARRIER_ARRIVE(arg->barrier, rv);
#endif
/*
  const uint64_t prefetch_offset = 4;
  const uint64_t prefetch_bound = numR - 4;

  for (i = 0; i < prefetch_bound; ++i)
  {
    array[relR[i].key] = relR[i].payload;
    __builtin_prefetch(array+relR[i+prefetch_offset].key, 1, 1);
  }

  for (i = prefetch_bound; i < numR; ++i)
    array[relR[i].key] = relR[i].payload;
  */

#ifdef ARRAY_PREFETCH
  uint64_t bound = numR - 1;
  for (i = 0; i < bound; )
  {
    array[relR[i].key] = relR[i].payload;
    __builtin_prefetch(array+relR[++i].key, 1, 1);
  }
  array[relR[bound].key] = relR[bound].payload;


  BARRIER_ARRIVE(arg->barrier, rv);

  bound = numS - 1;

  for (i = 0; i < bound; )
  {
    if (relS[i].key > 0 && relS[i].key <= numR && array[relS[i].key] != 0)
    {
      ++matches;
      if (is_checksum)
        checksum += relS[i].payload + array[relS[i].key];
    }
    __builtin_prefetch(array+relS[++i].key, 0, 1);
  }
  if (relS[i].key > 0 && relS[i].key <= totalR && array[relS[bound].key] != 0)
  {
    ++matches;
    if (is_checksum)
      checksum += relS[bound].payload + array[relS[bound].key];
  }
#else
  for (i = 0; i < numR; ++i)
    array[relR[i].key] = relR[i].payload;

  BARRIER_ARRIVE(arg->barrier, rv);

#ifndef NO_TIMING
    if(arg->tid ==0 ) {
        timeval build;
        gettimeofday(&build, NULL);
        printf("time to build: %lf\n",diff_usec(arg->start,&build));
    }
#endif

  for (i = 0; i < numS; ++i)
  {
    if (relS[i].key > 0 && relS[i].key <= totalR && array[relS[i].key] != 0)
    {
      ++matches;
      if (is_checksum)
        checksum += relS[i].payload + array[relS[i].key];
    }
  }
#endif
  arg->matches = matches;
  arg->checksum = checksum;

#ifdef ALGO_TIME
  BARRIER_ARRIVE(arg->barrier, rv);
  if (arg->tid == 0)
    gettimeofday(arg->end, NULL);
#endif
  return 0;
}

/*
 * AJ: Array join
 * relation R should be dense primary key
 */
template<bool is_checksum>
join_result_t
AJ(relation_t * R, relation_t * S, int nThreads)
{
  int rv;
  pthread_t tid[nThreads];
  pthread_attr_t attr;
  pthread_barrier_t barrier;
  cpu_set_t set;
  thr_arg_t args[nThreads];
  uint64_t checksum = 0;
  uint64_t matches = 0;
  uint64_t numR_per_thr = 0;
  uint64_t numS_per_thr = 0;
  uint64_t offsetR = 0;
  uint64_t offsetS = 0;
  struct timeval start, end;

  const uint64_t numR = R->num_tuples;
  const uint64_t numS = S->num_tuples;

  value_t *array = (value_t*)malloc((numR+1) * sizeof(value_t));
  MALLOC_CHECK(array);

  // numa_localize_for_array(array, numR+1, nThreads);

    rv = pthread_barrier_init(&barrier, NULL, nThreads);
    if(rv != 0){
        printf("[ERROR] Couldn't create the barrier\n");
        exit(EXIT_FAILURE);
    }

  pthread_attr_init(&attr);

  numR_per_thr = numR / nThreads;
  numS_per_thr = numS / nThreads;

  for (int i = 0; i < nThreads; ++i)
  {
    int cpu_idx = get_cpu_id(i);

    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

    args[i].array = array;

    args[i].relR = R->tuples + offsetR;
    args[i].relS = S->tuples + offsetS;

    args[i].numR = (i == nThreads-1) ? (numR-numR_per_thr*i):numR_per_thr;
    args[i].numS = (i == nThreads-1) ? (numS-numS_per_thr*i):numS_per_thr;
    args[i].totalR = numR;

    args[i].tid = i;
    args[i].barrier = &barrier;

    args[i].start = &start;
    args[i].end = &end;

    offsetR += numR_per_thr;
    offsetS += numS_per_thr;

    rv= pthread_create(&tid[i], &attr, array_join_thread<is_checksum>, (void*)&args[i]);
    if (rv)
    {
      printf("[ERROR] return code from pthread_create() is %d\n", rv);
      exit(1);
    }
  }

  for (int i = 0; i < nThreads; ++i)
    pthread_join(tid[i], NULL);

  for (int i = 0; i < nThreads; ++i)
  {
    matches += args[i].matches;
    checksum += args[i].checksum;
  }

  free(array);

  uint64_t time_usec = (end.tv_usec + end.tv_sec * 1000000LLU)
             - (start.tv_usec + start.tv_sec * 1000000LLU);
  join_result_t res = {matches, checksum, time_usec,0,0};
  return res;
}

template join_result_t AJ<false>(relation_t * R, relation_t * S, int nThreads);
template join_result_t AJ<true>(relation_t * R, relation_t * S, int nThreads);


int main(int argc, char** argv) {
  int nthreads = 8;
  int r_size   = 16 * (1 << 20);
  int s_size   = 256 * (1 << 20);

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", s_size);
  args.GetCmdLineArgument("d", r_size);
  //args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
      "[--n=<num fact>] "
      "[--d=<num dim>] "
      "[--t=<num trials>] "
      "\n", argv[0]);
    exit(0);
  }

  relation_t relR;
  relation_t relS;

  cout << "RSize " << r_size << " SSize " << s_size << endl;

  relR.tuples = (tuple_t*) alloc_aligned(r_size * sizeof(tuple_t));
  relS.tuples = (tuple_t*) alloc_aligned(s_size * sizeof(tuple_t));

  create_relation_pk(&relR, r_size);

  create_relation_fk(&relS, s_size, r_size);


  join_result_t res = AJ<true>(&relR, &relS, nthreads);
  cout << "Checksum: " << res.checksum << endl;

  return 0;
}


