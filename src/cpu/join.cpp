#include <math.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/generator.h"
#include "utils/cpu_utils.h"

using namespace tbb;
using namespace std;

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
float build(const uint32_t *keys, const uint32_t *vals, size_t size, bucket_t *table,
           int num_slots)
{
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = 1<<22;

  start = chrono::high_resolution_clock::now();
  parallel_for(blocked_range<size_t>(0, size, batch_size), [&](auto range) {
    size_t i;
    for (i = range.begin(); i != range.end(); ++i) {
      uint32_t key = keys[i];
      uint32_t val = vals[i];
      size_t h = (key & (num_slots - 1));
      //while (table[h].key != invalid_key)
        //h = (h + 1) & (buckets - 1);
      table[h].key = key;
      table[h].val = val;
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

float probe(uint32_t *keys, uint32_t *vals, size_t size,
              const bucket_t *table, int num_slots)
{
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = 1<<29;
  const uint32_t invalid_key = 0;
  tbb::atomic<long long> full_checksum;
  full_checksum = 0;

  start = chrono::high_resolution_clock::now();

  parallel_for(blocked_range<size_t>(0, size, batch_size), [&](auto range) {
    size_t i;
    long long checksum = 0;
    const uint64_t *table_64 = (const uint64_t*) table;
    for (i = range.begin(); i != range.end(); ++i) {
      uint32_t key = keys[i];
      uint32_t val = vals[i];
      size_t h = (uint32_t) (key & (num_slots - 1));
      //size_t h = HASH(key, num_slots);
      uint64_t tab = table_64[h];
      if (key == (uint32_t) tab) {
        checksum += val + (tab >>  32);
        //cout << "K V1 V2 " << key << " " << val << " " << (tab >> 32) << endl;
      } else while (invalid_key != (uint32_t) tab) {
        h = (h + 1) & (num_slots - 1);
        tab = table_64[h];
        if (key == (uint32_t) tab) {
          checksum += val + (tab >> 32);
          break;
        }
      }
    }
    full_checksum.fetch_and_add(checksum);
  });

  finish = chrono::high_resolution_clock::now();

  long long res_checksum = full_checksum.fetch_and_add(0);
  cout << "Checksum Scalar " << res_checksum << endl;

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

// Size of group for prefetching
#define GROUP_PREFETCH_SIZE 16

float probe_prefetch(uint32_t *keys, uint32_t *vals, size_t size,
              const bucket_t *table, int num_slots)
{
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = 1<<29;
  const uint32_t invalid_key = 0;
  tbb::atomic<long long> full_checksum;
  full_checksum = 0;

  start = chrono::high_resolution_clock::now();

  parallel_for(blocked_range<size_t>(0, size, batch_size), [&](auto range) {
    size_t i, j;
    long long checksum = 0;
    const uint64_t *table_64 = (const uint64_t*) table;
    for (i = range.begin(); i != range.end(); i += GROUP_PREFETCH_SIZE) {
      uint32_t keys_arr[GROUP_PREFETCH_SIZE];
      uint32_t vals_arr[GROUP_PREFETCH_SIZE];
      uint32_t hash_arr[GROUP_PREFETCH_SIZE];
      for (j = 0; j < GROUP_PREFETCH_SIZE; j++) {
        keys_arr[j] = keys[i + j];
        vals_arr[j] = vals[i + j];
        hash_arr[j] = (uint32_t) (keys_arr[j] & (num_slots - 1));
        _mm_prefetch(&table_64[hash_arr[j]], 0);
      }

      for (j = 0; j < GROUP_PREFETCH_SIZE; j++) {
        uint64_t tab = table_64[hash_arr[j]];
        if (keys_arr[j] == (uint32_t) tab) {
          checksum += vals_arr[j] + (tab >>  32);
          //cout << "K V1 V2 " << key << " " << val << " " << (tab >> 32) << endl;
        } else while (invalid_key != (uint32_t) tab) {
          hash_arr[j] = (hash_arr[j] + 1) & (num_slots - 1);
          tab = table_64[hash_arr[j]];
          if (keys_arr[j] == (uint32_t) tab) {
            checksum += vals_arr[j] + (tab >> 32);
            break;
          }
        }
      }
    }
    full_checksum.fetch_and_add(checksum);
  });

  finish = chrono::high_resolution_clock::now();

  long long res_checksum = full_checksum.fetch_and_add(0);
  cout << "Checksum Prefetch " << res_checksum << endl;

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

float probe_ver(uint32_t *keys_arr, uint32_t *vals_arr, size_t size,
                 const bucket_t *table, int num_slots)
{
  const uint32_t invalid_key = 0;
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = 1<<21;
  tbb::atomic<unsigned long long> full_checksum;
  full_checksum = 0;

  start = chrono::high_resolution_clock::now();

  parallel_for(blocked_range<size_t>(0, size, batch_size), [&](auto range) {
    uint32_t *keys = keys_arr + range.begin();
    uint32_t *vals = vals_arr + range.begin();
    size_t i = 0, o = 0, b = 0, partition_size = range.end() - range.begin();
    assert(sizeof(table[0]) == 8);
    const size_t buckets = num_slots;
    //const __m128i shift = _mm_cvtsi32_si128(32 - log_2(buckets));
    //const __m256i factor = _mm256_set1_epi32(table_factor);
    const __m256i empty = _mm256_set1_epi32(invalid_key);
    //preg(empty);
    const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
    const __m256i mask_1 = _mm256_set1_epi32(1);
    __m256i checksum = _mm256_set1_epi32(0);
    //preg(checksum);
#if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600
    const long *table_64 = (const long*) table;
#else
    const long long *table_64 = (const long long*) table;
#endif
    const size_t buf_size = 128;
    //int buf_space[buf_size + 8 + 15];
    //int *buf = align((void*) buf_space);
    __m256i key, val, off, inv = _mm256_set1_epi32(-1);
    while (i + 8 <= partition_size) {

      //cout << i << " " << size << endl;
      // load new items (mask out reloads)
      __m256i new_key = _mm256_maskload_epi32((const int*) &keys[i], inv);
      key = _mm256_andnot_si256(inv, key); // Does not(inv) & key
      key = _mm256_or_si256(key, new_key);

      __m256i new_val = _mm256_maskload_epi32((const int*) &vals[i], inv);
      val = _mm256_andnot_si256(inv, val); // Does not(inv) & key
      val = _mm256_or_si256(val, new_val);

      //cout << "Key: "; preg(key);
      //cout << "Val: "; preg(val);

      // hash
      //__m256i h = _mm256_mullo_epi32(key, factor);
      off = _mm256_add_epi32(off, mask_1);
      off = _mm256_andnot_si256(inv, off);
      //h = _mm256_srl_epi32(h, shift);
      __m256i h = _mm256_add_epi32(key, off);
      h = _mm256_and_si256(h, buckets_minus_1);
      // gather
      __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
      h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
      __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
      __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
      //cout << "Tab Key "; preg(tab_key);
      __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
      //cout << "Tab Val "; preg(tab_val);
      // update count & sum
      inv = _mm256_cmpeq_epi32(tab_key, empty);
      __m256i out = _mm256_cmpeq_epi32(tab_key, key);
      inv = _mm256_or_si256(inv, out);
      // load permutation masks
      size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
      size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
      __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
      __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
      __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
      __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
      // permutation for invalid
      inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
      key = _mm256_permutevar8x32_epi32(key, perm_inv);
      off = _mm256_permutevar8x32_epi32(off, perm_inv);

      i += _mm_popcnt_u64(j);
      //cout << "Num Matched " << i << endl;

      // permutation for output
      tab_val = _mm256_and_si256(tab_val, out);
      __m256i c2 = _mm256_and_si256(val, out);
      tab_val = _mm256_add_epi32(tab_val, c2);
      //cout << "Check Add "; preg(tab_val);
      //cout << "Check Before "; preg(checksum);
      checksum = _mm256_add_epi32(checksum, tab_val);
      //cout << "Check After "; preg(checksum);

      val = _mm256_permutevar8x32_epi32(val, perm_inv);

      //tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
      //out = _mm256_permutevar8x32_epi32(out, perm_out);
      //_mm256_maskstore_epi32(&buf[b], out, tab_val);

/*      b += _mm_popcnt_u64(k);*/

      //// flush buffer
      //if (b > buf_size) {
        //size_t b_i = 0;
        //do {
          //__m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
          //_mm256_stream_si256((__m256i*) &vals[o], x);
          //b_i += 8;
          //o += 8;
        //} while (b_i != buf_size);
        //__m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
        //_mm256_store_si256((__m256i*) &buf[0], x);
        //b -= buf_size;
      /*}*/

    }

    // clean buffer
    //size_t b_i = 0;
    //while (b_i != b) _mm_stream_si32(&((int*) vals)[o++], buf[b_i++]);
    // extract last keys
    /*
    uint32_t l_keys[8];
    _mm256_storeu_si256((__m256i*) l_keys, key);
    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
    j = 8 - _mm_popcnt_u64(j);
    i += j;
    while (i != size) l_keys[j++] = keys[i++];
    // process last keys
    //const uint8_t s = 32 - log_2(buckets);
    for (i = 0 ; i != j ; ++i) {
      uint32_t k = l_keys[i];
      size_t h = ((uint32_t) (k & (num_slots - 1)));
      while (invalid_key != table[h].key) {
        if (k == table[h].key) {
          _mm_stream_si32(&((int*) vals)[o++], table[h].val);
          break;
        }
        h = (h + 1) & (buckets - 1);
      }
    }
    */
    uint32_t check_vals[8];
    _mm256_store_si256((__m256i*)check_vals, checksum);
    unsigned long long local_sum = 0;
    //cout << "Final Vals ";
    for (size_t i=0; i<8; i++) { local_sum += check_vals[i]; }
    //cout << endl;
    //cout << local_sum << endl;
    full_checksum.fetch_and_add(local_sum);
  });

  finish = chrono::high_resolution_clock::now();

  unsigned long long res_checksum = full_checksum.fetch_and_add(0);
  cout << "Checksum SIMD " << res_checksum << endl;

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
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

TimeKeeper joinScalar(uint32_t* dim_key, uint32_t* dim_val, uint32_t* fact_fkey, uint32_t* fact_val, int num_dim, int num_fact) {
  bucket_t* hash_table = NULL;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  hash_table = (bucket_t*) _mm_malloc(sizeof(bucket_t) * num_slots, 256);

  time_memset = parallel_zero((int*)hash_table, num_slots * 2);

  time_build = build(dim_key, dim_val, num_dim, hash_table, num_slots);

  time_probe = probe(fact_fkey, fact_val, num_fact, hash_table, num_slots);

  TimeKeeper t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset};
  return t;
}

TimeKeeper joinPrefetch(uint32_t* dim_key, uint32_t* dim_val, uint32_t* fact_fkey, uint32_t* fact_val, int num_dim, int num_fact) {
  bucket_t* hash_table = NULL;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  hash_table = (bucket_t*) _mm_malloc(sizeof(bucket_t) * num_slots, 256);

  time_memset = parallel_zero((int*)hash_table, num_slots * 2);

  time_build = build(dim_key, dim_val, num_dim, hash_table, num_slots);

  time_probe = probe_prefetch(fact_fkey, fact_val, num_fact, hash_table, num_slots);

  TimeKeeper t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset};
  return t;
}


TimeKeeper joinSIMD(uint32_t* dim_key, uint32_t* dim_val, uint32_t* fact_fkey, uint32_t* fact_val, int num_dim, int num_fact) {
  bucket_t* hash_table = NULL;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  hash_table = (bucket_t*) _mm_malloc(sizeof(bucket_t) * num_slots, 256);

  time_memset = parallel_zero((int*)hash_table, num_slots * 2);

  time_build = build(dim_key, dim_val, num_dim, hash_table, num_slots);

  time_probe = probe_ver(fact_fkey, fact_val, num_fact, hash_table, num_slots);

  TimeKeeper t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset};

  return t;
}

int main(int argc, char** argv)
{
  int num_fact       = 256 * 1<<20;
  int num_dim      = 64 * 1<<20;
  int num_trials     = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_fact);
  args.GetCmdLineArgument("d", num_dim);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
      "[--n=<num fact>] "
      "[--d=<num dim>] "
      "[--t=<num trials>] "
      "[--device=<device-id>] "
      "[--v] "
      "\n", argv[0]);
    exit(0);
  }

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  long long sum = 0;
  for (int i=0; i<num_fact; i++) sum += h_fact_val[i];
  //cout << "FSUM " << sum << endl;

  for (int i=0; i<num_dim; i++) sum += h_dim_val[i];
  //cout << "DSUM " << sum << endl;

  cout << "Done Generating" << endl;

  uint32_t *dim_key = (uint32_t*) h_dim_key;
  uint32_t *dim_val = (uint32_t*) h_dim_val;
  uint32_t *fact_fkey = (uint32_t*) h_fact_fkey;
  uint32_t *fact_val = (uint32_t*) h_fact_val;

/*  cout << "Original" << endl;*/
  //for (int i=0; i<8; i++) cout << fact_fkey[i] << " ";
  //cout << endl;
  //for (int i=0; i<8; i++) cout << fact_val[i] << " ";
  /*cout << endl;*/

  for (int t = 0; t < num_trials; t++) {
/*    TimeKeeper time_simd = joinSIMD(dim_key, dim_val, fact_fkey, fact_val, num_dim, num_fact);*/
    /*cout << "Done SIMD" << endl;*/

    TimeKeeper time_scalar = joinScalar(dim_key, dim_val, fact_fkey, fact_val, num_dim, num_fact);
    cout << "Done Scalar" << endl;

    TimeKeeper time_prefetch = joinPrefetch(dim_key, dim_val, fact_fkey, fact_val, num_dim, num_fact);
    cout << "Done Prefetch" << endl;

    cout<< "{"
      << "\"num_dim\":" << num_dim
      << ",\"num_fact\":" << num_fact
      << ",\"time_build_scalar\":" << time_scalar.time_build
      << ",\"time_probe_scalar\":" << time_scalar.time_probe
      << ",\"time_build_prefetch\":" << time_prefetch.time_build
      << ",\"time_probe_prefetch\":" << time_prefetch.time_probe
  /*    << ",\"time_build_simd\":" << time_simd.time_build*/
      /*<< ",\"time_probe_simd\":" << time_simd.time_probe*/
      << "}" << endl;
  }

  return 0;
}

