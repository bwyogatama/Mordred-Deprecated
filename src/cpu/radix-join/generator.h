#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <sched.h>              /* CPU_ZERO, CPU_SET */
#include <pthread.h>            /* pthread_attr_setaffinity_np */
#include <stdio.h>              /* perror */
#include <stdlib.h>             /* RAND_MAX */
#include <math.h>               /* fmod, pow */
#include <time.h>               /* time() */
#include <unistd.h>             /* getpagesize() */
#include <string.h>             /* memcpy() */
#include <iostream>
using namespace std;

/* return a random number in range [0,N] */
#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RAND_RANGE48(N,STATE) ((double)nrand48(STATE)/((double)RAND_MAX+1)*(N))


static int seeded = 0;
static unsigned int seedValue;

void
seed_generator(unsigned int seed)
{
    srand(seed);
    seedValue = seed;
    seeded = 1;
}

/** Check wheter seeded, if not seed the generator with current time */
static void
check_seed()
{
    if(!seeded) {
        seedValue = time(NULL);
        srand(seedValue);
        seeded = 1;
    }
}


/**
 * Shuffle tuples of the relation using Knuth shuffle.
 *
 * @param relation
 */
void
knuth_shuffle(relation_t * relation)
{
    int i;
    for (i = relation->num_tuples - 1; i > 0; i--) {
        int64_t  j              = RAND_RANGE(i);
        intkey_t tmp            = relation->tuples[i].key;
        relation->tuples[i].key = relation->tuples[j].key;
        relation->tuples[j].key = tmp;
    }
}

void
knuth_shuffle48(relation_t * relation, unsigned short * state)
{
    int i;
    for (i = relation->num_tuples - 1; i > 0; i--) {
        int64_t  j              = RAND_RANGE48(i, state);
        intkey_t tmp            = relation->tuples[i].key;
        relation->tuples[i].key = relation->tuples[j].key;
        relation->tuples[j].key = tmp;
    }
}

void
random_unique_gen(relation_t *rel)
{
    uint64_t i;
    uint64_t num = rel->num_tuples;
    rel->ratio_holes = 1;

    for (i = 0; i < num; ++i) {
        rel->tuples[i].key = i+1;
        rel->tuples[i].payload= i+1;
    }

    /* randomly shuffle elements */
    knuth_shuffle(rel);
}

/**
 * Generate unique tuple IDs with Knuth shuffling
 * relation must have been allocated
 */
void
random_unique_gen_with_holes(relation_t *rel)
{
    uint64_t i;
    uint64_t step = rel->ratio_holes;
    uint64_t num = rel->num_tuples;
  uint64_t val = 1;

    for (i = 0; i < num; ++i) {
        rel->tuples[i].key = val;
        rel->tuples[i].payload= val;
    val += step;
    }

    /* randomly shuffle elements */
    knuth_shuffle(rel);
}

/** NaN masks */
static int64_t NaNExpMask = (0x7FFL << 52U);
static int64_t NaNlowbitclear = ~(1L << 52U);

/** Avoid NaN in values, since tuples are treated as double for AVX instructions */
void
avoid_NaN(int64_t * data)
{
    /* avoid NaN in values */
    int64_t * bitpattern = (int64_t *) data;
    if(((*bitpattern) & NaNExpMask) == NaNExpMask){
        *bitpattern &= NaNlowbitclear;
    }
}


/**
 * Generate tuple IDs -> random distribution
 * relation must have been allocated
 */
void
random_gen(relation_t *rel, const int64_t maxid)
{
    uint64_t i;

    for (i = 0; i < rel->num_tuples; i++) {
        rel->tuples[i].key = RAND_RANGE(maxid);
        rel->tuples[i].payload = rel->num_tuples - i;

        /* avoid NaN in values */
        avoid_NaN((int64_t*)&(rel->tuples[i]));
    }
}

int
create_relation_pk(relation_t *relation, int64_t num_tuples)
{
    check_seed();

    relation->num_tuples = num_tuples;

    if (!relation->tuples) {
        perror("memory must be allocated first");
        return -1;
    }

    random_unique_gen(relation);

#ifdef PERSIST_RELATIONS
    write_relation(relation, "R.tbl");
#endif

    return 0;
}

int
create_relation_fk(relation_t *relation, int64_t num_tuples, const int64_t maxid)
{
  int32_t i, iters;
  int64_t remainder;
  relation_t tmp;

  check_seed();

  relation->num_tuples = num_tuples;

  if (!relation->tuples) {
    perror("memory must be allocated first");
    return -1;
  }

  /* alternative generation method */
  iters = num_tuples / maxid;
  for(i = 0; i < iters; i++){
    tmp.num_tuples = maxid;
    tmp.tuples = relation->tuples + maxid * i;
    random_unique_gen(&tmp);
  }

  /* if num_tuples is not an exact multiple of maxid */
  remainder = num_tuples % maxid;
  if(remainder > 0) {
    tmp.num_tuples = remainder;
    tmp.tuples = relation->tuples + maxid * iters;
    random_unique_gen(&tmp);
  }

#ifdef PERSIST_RELATIONS
  write_relation(relation, "S.tbl");
#endif

  return 0;
}

#endif
