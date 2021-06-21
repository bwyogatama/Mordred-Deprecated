#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <inttypes.h>

/**
 * @defgroup Types Common Types
 * Common type definitions used by all join implementations.
 * @{
 */

#ifdef KEY_8B /* 64-bit key/payload, 16B tuples */
typedef uint64_t intkey_t;
typedef uint64_t value_t;
#else /* 32-bit key/payload, 8B tuples */
typedef uint32_t intkey_t;
typedef uint32_t value_t;
#endif

#if !defined PRId64
# undef PRId64
# define PRId64 "lld"
#endif

#if !defined PRIu64
# undef PRIu64
# define PRIu64 "llu"
#endif


typedef struct tuple_t    tuple_t;
typedef struct relation_t relation_t;
typedef struct result_t result_t;
typedef struct threadresult_t threadresult_t;
typedef struct joinconfig_t joinconfig_t;
typedef struct join_result_t join_result_t;

/**
 * Type definition for a tuple, depending on KEY_8B a tuple can be 16B or 8B
 * @note this layout is chosen as a work-around for AVX double operations.
 */
struct tuple_t {
    value_t  payload;
    intkey_t key;
};


/**
 * Type definition for a relation.
 * It consists of an array of tuples and a size of the relation.
 */
struct relation_t {
  tuple_t * tuples;
  uint64_t  num_tuples;
  int ratio_holes;
};


/** Holds the join results of a thread */
struct threadresult_t {
    int64_t  nresults;
    void *   results;
    uint32_t threadid;
};

/** Type definition for join results. */
struct result_t {
    int64_t          totalresults;
    threadresult_t * resultlist;
    int              nthreads;
};

struct join_result_t {
  uint64_t matches;
  uint64_t checksum;
  uint64_t time_usec;
  uint64_t part_usec;
  uint64_t join_usec;
};

/**
 * Various NUMA shuffling strategies for data shuffling phase of join
 * algorithms as also described by NUMA-aware data shuffling paper [CIDR'13].
 *
 * NUMA_SHUFFLE_RANDOM, NUMA_SHUFFLE_RING, NUMA_SHUFFLE_NEXT
 */
enum numa_strategy_t {RANDOM, RING, NEXT};

/** Join configuration parameters. */
struct joinconfig_t {
    int NTHREADS;
    int PARTFANOUT;
    int SCALARSORT;
    int SCALARMERGE;
    int MWAYMERGEBUFFERSIZE;
    enum numa_strategy_t NUMASTRATEGY;
};

typedef struct __attribute__((aligned(64))) {
    unsigned int thread_num;
    unsigned long num_elements;
    unsigned long int idx_i;
    unsigned long* histogram;
    tuple_t* data;
} chunk;

typedef struct __attribute__((aligned(64))) {
  unsigned int thread_num;
    long long int start;
    long long int end;
    long long int idx_i;
    long long int idx_j;
} info_per_thread;

#ifndef MAXLOADFACTOR
#define MAXLOADFACTOR 1.1
#endif
/** @} */

#endif /* TYPES_H */