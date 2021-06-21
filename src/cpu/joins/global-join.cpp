#include <math.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "types.h"
#include "generator.h"

#include "utils/cpu_utils.h"

using namespace tbb;
using namespace std;

#define CACHE_LINE_SIZE 64

void *mamalloc(size_t size)
{
  void *ptr = NULL;
  return posix_memalign(&ptr, 64, size) ? NULL : ptr;
}

int *align(const int* p)
{
  while (63 & (size_t) p) p++;
  return (int*) p;
}

int log_2(size_t n)
{
  size_t b = 1;
  int p = 0;
  while (b < n) {
    b += b;
    p++;
  }
  assert(b == n);
  return p;
}

typedef struct {
  uint32_t key;
  uint32_t val;
} bucket_t;

// TODO: Make this multi-threaded
float build(relation_t * rel, bucket_t *table,
           int num_slots)
{
  chrono::high_resolution_clock::time_point start, finish;
  int batch_size = min(1<<22, (int)rel->num_tuples / 8);

  start = chrono::high_resolution_clock::now();

  parallel_for(blocked_range<size_t>(0, rel->num_tuples, batch_size), [&](auto range) {
    size_t i;
    for (i = range.begin(); i != range.end(); ++i) {
      tuple_t& tuple = rel->tuples[i];
      size_t h = (tuple.key & (num_slots - 1));

      while (table[h].key != 0)
        h = (h + 1) & (num_slots - 1);

      table[h].key = tuple.key;
      table[h].val = tuple.payload;
    }
  });

  finish = chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

inline void store(uint32_t *p, uint32_t v)
{
#ifndef __MIC
  _mm_stream_si32((int*) p, v);
#else
  *p = v;
#endif
}

// TODO: tuple_t and bucket_t are the same
float probe(relation_t* rel, const bucket_t *table, int num_slots, uint64_t& matches, uint64_t& checksum)
{
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = min(1<<22, (int)rel->num_tuples / 8);
  const uint32_t invalid_key = 0;
  tbb::atomic<uint64_t> full_matches = 0;
  tbb::atomic<uint64_t> full_checksum = 0;

  start = chrono::high_resolution_clock::now();

  parallel_for(blocked_range<size_t>(0, rel->num_tuples, batch_size), [&](auto range) {
    size_t i;
    uint64_t local_matches = 0;
    uint64_t local_checksum = 0;
    const uint64_t *table_64 = (const uint64_t*) table;
    for (i = range.begin(); i != range.end(); ++i) {
      tuple_t tuple = rel->tuples[i];
      size_t h = (uint32_t) (tuple.key & (num_slots - 1));
      //size_t h = HASH(key, num_slots);
      uint64_t tab = table_64[h];
      if (tuple.key == (uint32_t) tab) {
        local_checksum += tuple.payload + (tab >>  32);
        local_matches += 1;
      } else while (invalid_key != (uint32_t) tab) {
        h = (h + 1) & (num_slots - 1);
        tab = table_64[h];
        if (tuple.key == (uint32_t) tab) {
          local_matches++;
          local_checksum += tuple.payload + (tab >> 32);
          break;
        }
      }
    }
    full_matches.fetch_and_add(local_matches);
    full_checksum.fetch_and_add(local_checksum);
  });

  finish = chrono::high_resolution_clock::now();

  matches = full_matches.fetch_and_add(0);
  checksum = full_checksum.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;

  return diff.count() * 1000;
}

inline __m256i _mm256_packlo_epi32(__m256i x, __m256i y)
{
  __m256 a = _mm256_castsi256_ps(x);
  __m256 b = _mm256_castsi256_ps(y);
  __m256 c = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));
  __m256i z = _mm256_castps_si256(c);
  return _mm256_permute4x64_epi64(z, _MM_SHUFFLE(3,1,2,0));
}

inline __m256i _mm256_packhi_epi32(__m256i x, __m256i y)
{
  __m256 a = _mm256_castsi256_ps(x);
  __m256 b = _mm256_castsi256_ps(y);
  __m256 c = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3,1,3,1));
  __m256i z = _mm256_castps_si256(c);
  return _mm256_permute4x64_epi64(z, _MM_SHUFFLE(3,1,2,0));
}

const uint64_t perm[256] = {0x0706050403020100ull,
     0x0007060504030201ull, 0x0107060504030200ull, 0x0001070605040302ull,
     0x0207060504030100ull, 0x0002070605040301ull, 0x0102070605040300ull,
     0x0001020706050403ull, 0x0307060504020100ull, 0x0003070605040201ull,
     0x0103070605040200ull, 0x0001030706050402ull, 0x0203070605040100ull,
     0x0002030706050401ull, 0x0102030706050400ull, 0x0001020307060504ull,
     0x0407060503020100ull, 0x0004070605030201ull, 0x0104070605030200ull,
     0x0001040706050302ull, 0x0204070605030100ull, 0x0002040706050301ull,
     0x0102040706050300ull, 0x0001020407060503ull, 0x0304070605020100ull,
     0x0003040706050201ull, 0x0103040706050200ull, 0x0001030407060502ull,
     0x0203040706050100ull, 0x0002030407060501ull, 0x0102030407060500ull,
     0x0001020304070605ull, 0x0507060403020100ull, 0x0005070604030201ull,
     0x0105070604030200ull, 0x0001050706040302ull, 0x0205070604030100ull,
     0x0002050706040301ull, 0x0102050706040300ull, 0x0001020507060403ull,
     0x0305070604020100ull, 0x0003050706040201ull, 0x0103050706040200ull,
     0x0001030507060402ull, 0x0203050706040100ull, 0x0002030507060401ull,
     0x0102030507060400ull, 0x0001020305070604ull, 0x0405070603020100ull,
     0x0004050706030201ull, 0x0104050706030200ull, 0x0001040507060302ull,
     0x0204050706030100ull, 0x0002040507060301ull, 0x0102040507060300ull,
     0x0001020405070603ull, 0x0304050706020100ull, 0x0003040507060201ull,
     0x0103040507060200ull, 0x0001030405070602ull, 0x0203040507060100ull,
     0x0002030405070601ull, 0x0102030405070600ull, 0x0001020304050706ull,
     0x0607050403020100ull, 0x0006070504030201ull, 0x0106070504030200ull,
     0x0001060705040302ull, 0x0206070504030100ull, 0x0002060705040301ull,
     0x0102060705040300ull, 0x0001020607050403ull, 0x0306070504020100ull,
     0x0003060705040201ull, 0x0103060705040200ull, 0x0001030607050402ull,
     0x0203060705040100ull, 0x0002030607050401ull, 0x0102030607050400ull,
     0x0001020306070504ull, 0x0406070503020100ull, 0x0004060705030201ull,
     0x0104060705030200ull, 0x0001040607050302ull, 0x0204060705030100ull,
     0x0002040607050301ull, 0x0102040607050300ull, 0x0001020406070503ull,
     0x0304060705020100ull, 0x0003040607050201ull, 0x0103040607050200ull,
     0x0001030406070502ull, 0x0203040607050100ull, 0x0002030406070501ull,
     0x0102030406070500ull, 0x0001020304060705ull, 0x0506070403020100ull,
     0x0005060704030201ull, 0x0105060704030200ull, 0x0001050607040302ull,
     0x0205060704030100ull, 0x0002050607040301ull, 0x0102050607040300ull,
     0x0001020506070403ull, 0x0305060704020100ull, 0x0003050607040201ull,
     0x0103050607040200ull, 0x0001030506070402ull, 0x0203050607040100ull,
     0x0002030506070401ull, 0x0102030506070400ull, 0x0001020305060704ull,
     0x0405060703020100ull, 0x0004050607030201ull, 0x0104050607030200ull,
     0x0001040506070302ull, 0x0204050607030100ull, 0x0002040506070301ull,
     0x0102040506070300ull, 0x0001020405060703ull, 0x0304050607020100ull,
     0x0003040506070201ull, 0x0103040506070200ull, 0x0001030405060702ull,
     0x0203040506070100ull, 0x0002030405060701ull, 0x0102030405060700ull,
     0x0001020304050607ull, 0x0706050403020100ull, 0x0007060504030201ull,
     0x0107060504030200ull, 0x0001070605040302ull, 0x0207060504030100ull,
     0x0002070605040301ull, 0x0102070605040300ull, 0x0001020706050403ull,
     0x0307060504020100ull, 0x0003070605040201ull, 0x0103070605040200ull,
     0x0001030706050402ull, 0x0203070605040100ull, 0x0002030706050401ull,
     0x0102030706050400ull, 0x0001020307060504ull, 0x0407060503020100ull,
     0x0004070605030201ull, 0x0104070605030200ull, 0x0001040706050302ull,
     0x0204070605030100ull, 0x0002040706050301ull, 0x0102040706050300ull,
     0x0001020407060503ull, 0x0304070605020100ull, 0x0003040706050201ull,
     0x0103040706050200ull, 0x0001030407060502ull, 0x0203040706050100ull,
     0x0002030407060501ull, 0x0102030407060500ull, 0x0001020304070605ull,
     0x0507060403020100ull, 0x0005070604030201ull, 0x0105070604030200ull,
     0x0001050706040302ull, 0x0205070604030100ull, 0x0002050706040301ull,
     0x0102050706040300ull, 0x0001020507060403ull, 0x0305070604020100ull,
     0x0003050706040201ull, 0x0103050706040200ull, 0x0001030507060402ull,
     0x0203050706040100ull, 0x0002030507060401ull, 0x0102030507060400ull,
     0x0001020305070604ull, 0x0405070603020100ull, 0x0004050706030201ull,
     0x0104050706030200ull, 0x0001040507060302ull, 0x0204050706030100ull,
     0x0002040507060301ull, 0x0102040507060300ull, 0x0001020405070603ull,
     0x0304050706020100ull, 0x0003040507060201ull, 0x0103040507060200ull,
     0x0001030405070602ull, 0x0203040507060100ull, 0x0002030405070601ull,
     0x0102030405070600ull, 0x0001020304050706ull, 0x0607050403020100ull,
     0x0006070504030201ull, 0x0106070504030200ull, 0x0001060705040302ull,
     0x0206070504030100ull, 0x0002060705040301ull, 0x0102060705040300ull,
     0x0001020607050403ull, 0x0306070504020100ull, 0x0003060705040201ull,
     0x0103060705040200ull, 0x0001030607050402ull, 0x0203060705040100ull,
     0x0002030607050401ull, 0x0102030607050400ull, 0x0001020306070504ull,
     0x0406070503020100ull, 0x0004060705030201ull, 0x0104060705030200ull,
     0x0001040607050302ull, 0x0204060705030100ull, 0x0002040607050301ull,
     0x0102040607050300ull, 0x0001020406070503ull, 0x0304060705020100ull,
     0x0003040607050201ull, 0x0103040607050200ull, 0x0001030406070502ull,
     0x0203040607050100ull, 0x0002030406070501ull, 0x0102030406070500ull,
     0x0001020304060705ull, 0x0506070403020100ull, 0x0005060704030201ull,
     0x0105060704030200ull, 0x0001050607040302ull, 0x0205060704030100ull,
     0x0002050607040301ull, 0x0102050607040300ull, 0x0001020506070403ull,
     0x0305060704020100ull, 0x0003050607040201ull, 0x0103050607040200ull,
     0x0001030506070402ull, 0x0203050607040100ull, 0x0002030506070401ull,
     0x0102030506070400ull, 0x0001020305060704ull, 0x0405060703020100ull,
     0x0004050607030201ull, 0x0104050607030200ull, 0x0001040506070302ull,
     0x0204050607030100ull, 0x0002040506070301ull, 0x0102040506070300ull,
     0x0001020405060703ull, 0x0304050607020100ull, 0x0003040506070201ull,
     0x0103040506070200ull, 0x0001030405060702ull, 0x0203040506070100ull,
     0x0002030405060701ull, 0x0102030405060700ull, 0x0001020304050607ull};

// Print Register
void preg(__m256i reg) {
  uint32_t check_vals[8];
  _mm256_store_si256((__m256i*)check_vals, reg);
  for (int i=0; i<8; i++) cout << check_vals[i] << " ";
  cout << endl;
}

float probe_ver(relation_t* rel, const bucket_t *table, int num_slots, uint64_t& matches, uint64_t& checksum)
{
  const uint32_t invalid_key = 0;
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = 1<<21;
  tbb::atomic<uint64_t> full_matches = 0;
  tbb::atomic<uint64_t> full_checksum = 0;

  start = chrono::high_resolution_clock::now();

  parallel_for(blocked_range<size_t>(0, rel->num_tuples, batch_size), [&](auto range) {
    const long long *tuples_64 = (const long long*) (rel->tuples + range.begin());

    size_t i = 0, o = 0, b = 0, partition_size = range.end() - range.begin();
    assert(sizeof(table[0]) == 8);

    const size_t buckets = num_slots;
    const __m256i empty = _mm256_set1_epi32(invalid_key);
    const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
    const __m256i mask_1 = _mm256_set1_epi32(1);
    uint64_t local_matches = 0;
    uint64_t local_checksum = 0;

    const long long *table_64 = (const long long*) table;

    const size_t buf_size = 128;
    //int buf_space[buf_size + 8 + 15];
    //int *buf = align((void*) buf_space);
    __m256i key, val, off, inv = _mm256_set1_epi32(-1);

    // TODO: Simplified version
    while (i + 8 <= partition_size) {
      //cout << i << " " << size << endl;
      // load new items (mask out reloads)
      __m256i first4 = _mm256_load_si256((const __m256i*) &tuples_64[i]);
      __m256i next4 = _mm256_load_si256((const __m256i*) &tuples_64[i+4]);

      key = _mm256_packlo_epi32(first4, next4);
      val = _mm256_packhi_epi32(first4, next4);

      __m256i h = _mm256_and_si256(key, buckets_minus_1);

      // gather
      __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
      h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
      __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
      __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
      __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
      //cout << "Tab Key "; preg(tab_key);
      //cout << "Tab Val "; preg(tab_val);

      // update count & sum
      __m256i out = _mm256_cmpeq_epi32(tab_key, key);

      // load permutation masks
      size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));

      //cout << "Num Matched " << i << endl;

      // permutation for output
      tab_val = _mm256_and_si256(tab_val, out);
      __m256i c2 = _mm256_and_si256(val, out);
      tab_val = _mm256_add_epi32(tab_val, c2);

      uint32_t check_vals[8];
      _mm256_store_si256((__m256i*)check_vals, tab_val);
      unsigned long long local_sum = 0;
      for (size_t i=0; i<8; i++) { local_sum += check_vals[i]; }

      //cout << "Check Add "; preg(tab_val);
      //cout << "Check Before "; preg(checksum);
      local_matches += _mm_popcnt_u64(k);
      local_checksum += local_sum;
      i += 8;

      //cout << "Check After "; preg(checksum);
    }

    full_matches.fetch_and_add(local_matches);
    full_checksum.fetch_and_add(local_checksum);
  });

  finish = chrono::high_resolution_clock::now();

  matches = full_matches.fetch_and_add(0);
  checksum = full_checksum.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

struct JoinResult {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
  uint64_t matches;
  uint64_t checksum;
};

float parallel_zero(int *arr, int size) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();
  const __m256i zero = _mm256_set1_epi32(0);
  int batch_size = 1<<21;

  parallel_for(blocked_range<size_t>(0, size, batch_size), [&](auto range) {
    for (size_t i = range.begin(); i < range.end(); i += 8) {
      _mm256_stream_si256((__m256i*) &arr[i], zero);
    }
  });
  finish = chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

JoinResult joinScalar(relation_t * relR, relation_t * relS) {
  bucket_t* hash_table = NULL;
  int num_slots = (int) relR->num_tuples;
  float time_build, time_probe, time_memset, time_memset2;
  uint64_t matches, checksum;

  hash_table = (bucket_t*) _mm_malloc(sizeof(bucket_t) * num_slots, 256);

  time_memset = parallel_zero((int*)hash_table, num_slots * 2);

  time_build = build(relR, hash_table, num_slots);

  time_probe = probe(relS, hash_table, num_slots, matches, checksum);

  JoinResult t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset, matches, checksum};
  return t;
}

JoinResult joinSIMD(relation_t * relR, relation_t * relS) {
  bucket_t* hash_table = NULL;
  int num_slots = (int) relR->num_tuples;
  float time_build, time_probe, time_memset, time_memset2;
  uint64_t matches, checksum;

  hash_table = (bucket_t*) _mm_malloc(sizeof(bucket_t) * num_slots, 256);

  time_memset = parallel_zero((int*)hash_table, num_slots * 2);

  time_build = build(relR, hash_table, num_slots);

  time_probe = probe_ver(relS, hash_table, num_slots, matches, checksum);

  JoinResult t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset, matches, checksum};

  return t;
}

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

int main(int argc, char** argv)
{
  int nthreads = 8;
  int r_size   = 16 * (1 << 20);
  int s_size   = 256 * (1 << 20);
  int num_trials = 1;

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

  relR.tuples = (tuple_t*) alloc_aligned(r_size * sizeof(tuple_t));
  relS.tuples = (tuple_t*) alloc_aligned(s_size * sizeof(tuple_t));

  create_relation_pk(&relR, r_size);

  create_relation_fk(&relS, s_size, r_size);

  for (int t = 0; t < num_trials; t++) {
    JoinResult time_simd = joinSIMD(&relR, &relS);
    JoinResult time_scalar = joinScalar(&relR, &relS);

    cout<< "{"
      << "\"num_dim\":" << relR.num_tuples
      << ",\"num_fact\":" << relS.num_tuples
      << ",\"time_build_scalar\":" << time_scalar.time_build
      << ",\"time_probe_scalar\":" << time_scalar.time_probe
      << ",\"matches_scalar\":" << time_scalar.matches
      << ",\"checksum_scalar\":" << time_scalar.checksum
      << ",\"time_build_simd\":" << time_simd.time_build
      << ",\"time_probe_simd\":" << time_simd.time_probe
      << ",\"matches_simd\":" << time_simd.matches
      << ",\"matches_simd\":" << time_simd.checksum
      << "}" << endl;
  }

  return 0;
}

