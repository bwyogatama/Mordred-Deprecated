#include <math.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <immintrin.h>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/cpu_utils.h"

using namespace std;
using namespace tbb;

#define BATCH_SIZE 2048

float selectIfCPU(float* keys, float* payload, float* keys_out, float* payload_out, int num_items,
    float keys_hi, float keys_lo, int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<int> counter;
  counter = 0;

  parallel_for(blocked_range<size_t>(0, num_items, BATCH_SIZE), [&](auto range) {
    int count = 0;
    for (size_t i = range.begin(); i < range.end(); i++) {
      count += (keys[i] >= keys_lo) && (keys[i] < keys_hi);
    }
    if (count > 0) {
      int offset = counter.fetch_and_add(count);
      for (size_t i = range.begin(); i < range.end(); i++) {
        if ((keys[i] >= keys_lo) && (keys[i] < keys_hi)) {
          keys_out[offset] = keys[i];
          payload_out[offset] = payload[i];
          offset++;
        }
      }
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  return (chrono::duration_cast<chrono::milliseconds>(finish-start)).count();
}

float selectPredCPU(float* keys, float* payload, float* keys_out, float* payload_out, int num_items,
    float keys_hi, float keys_lo, int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<uint> counter = 0;
  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    uint count = 0;
    for (size_t i = range.begin(); i < range.end(); i++) {
      count += (keys[i] >= keys_lo) && (keys[i] < keys_hi);
    }
    if (count > 0) {
      uint offset = counter.fetch_and_add(count);
      for (size_t i = range.begin(); i < range.end(); i++) {
        keys_out[offset] = keys[i];
        payload_out[offset] = payload[i];
        offset += (keys[i] >= keys_lo) && (keys[i] < keys_hi);
      }
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

float* align(float * p)
{
  while (63 & (size_t) p) p++;
  return p;
}

float selectSIMDCPU(float* keys, float* payload, float* keys_out, float* payload_out, int num_items,
    float keys_hi, float keys_lo, int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<int> counter;
  tbb::atomic<int> counter_tail;
  counter = 0;
  counter_tail = 0;
  float keys_tail[32*64];
  float payload_tail[32*64];
  const __m256 key_L = _mm256_set1_ps(keys_lo);
  const __m256 key_H = _mm256_set1_ps(keys_hi);
  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    int k = 0;
    const size_t buf_size = 128;
    float buf[(buf_size + 8) * 2 + 15];
    // todo: align
    float *keys_buf = align(buf);
    float *payload_buf = &keys_buf[buf_size + 8];
    for (size_t i = range.begin(); i < range.end(); i += 8) {
      __m256 key = _mm256_load_ps(&keys[i]);
      // generates bitmask - 1111 for true 0000 for false
      __m256 in_L = _mm256_cmp_ps(key_L, key, _CMP_LT_OS);
      __m256 in_H = _mm256_cmp_ps(key, key_H, _CMP_LT_OS);
      __m256 cmp = _mm256_and_ps(in_L, in_H);
      /* load key columns and evaluate predicates */
      if (!_mm256_testz_ps(cmp, cmp)) {
        __m256 val = _mm256_load_ps(&payload[i]);
        /* permute and store the input pointers */
        // The following algorithm is copied from https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        // gets 8 sign bits
        int m = _mm256_movemask_ps(cmp); //_mm256_castsi256_ps(cmp));
        uint64_t expanded_mask = _pdep_u64(m, 0x0101010101010101);  // unpack each bit to a byte
        expanded_mask *= 0xFF;   // ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte
        const uint64_t identity_indices = 0x0706050403020100;    // the identity shuffle for vpermps, packed to one index per byte
        uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);
        __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
        __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
        key = _mm256_permutevar8x32_ps(key, shufmask);
        val = _mm256_permutevar8x32_ps(val, shufmask);
        __m256i cmp_i = _mm256_cvtps_epi32(cmp);
        _mm256_maskstore_ps(&keys_buf[k], cmp_i, key);
        _mm256_maskstore_ps(&payload_buf[k], cmp_i, val);
        k += _mm_popcnt_u64(m);
        if (k > buf_size) {
          int offset = counter.fetch_and_add(buf_size);
          /* flush the buffer */
          int b;
          for (b = 0; b != buf_size; b += 8) {
            /* dereference column values and store */
            key = _mm256_load_ps(&keys_buf[b]);
            val = _mm256_load_ps(&payload_buf[b]);
            _mm256_stream_ps(&keys_out[offset + b], key);
            _mm256_stream_ps(&payload_out[offset + b], val);
          }
          /* move extra items to the start of the buffer */
          key = _mm256_load_ps(&keys_buf[b]);
          val = _mm256_load_ps(&payload_buf[b]);
          _mm256_store_ps(&keys_buf[0], key);
          _mm256_store_ps(&payload_buf[0], val);
          k -= buf_size;
        }
      }
    }
    // int offset = counter_tail.fetch_and_add(k);
    // TODO. copy of the tail of each local buffer
    //rid = _mm256_add_epi32(rid, mask_8);
    //for (int i=0; i<k; i++)
    int offset = counter.fetch_and_add((k >> 3) << 3);
    int b;
    for (b = 0; b < k; b += 8) {
      __m256 key = _mm256_load_ps(&keys_buf[b]);
      __m256 val = _mm256_load_ps(&payload_buf[b]);
      _mm256_stream_ps(&keys_out[offset + b], key);
      _mm256_stream_ps(&payload_out[offset + b], val);
    }
    offset = counter_tail.fetch_and_add(k - ((k >> 3) << 3));
    for (; b < k; b++) {
      keys_tail[offset + b - ((k >> 3) << 3)] = keys_buf[b];
      payload_tail[offset + b - ((k >> 3) << 3)] = payload_buf[b];
    }
  });

  //memcpy(&out[counter.fetch_and_add(0)], tail_buffer, counter_tail.fetch_and_add(0));
  finish = chrono::high_resolution_clock::now();
  num_selected = counter.fetch_and_add(0) + counter_tail.fetch_and_add(0);
  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(int argc, char** argv) {
    int num_items           = 1<<28;
    int num_trials          = 3;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("t", num_trials);

    // Print usage
    if (args.CheckCmdLineFlag("help")) {
        printf("%s "
            "[--n=<input items>] "
            "[--t=<num trials>] "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    float *h_keys, *h_payload, *h_keys_out, *h_payload_out;
    char *h_flags;

    h_keys = (float*) _mm_malloc(sizeof(float) * num_items, 256);
    h_payload = (float*) malloc(sizeof(float) * num_items);
    h_keys_out = (float*) _mm_malloc(sizeof(float) * num_items, 256);
    h_payload_out = (float*) _mm_malloc(sizeof(float) * num_items, 256);
    h_flags = (char*) malloc(sizeof(float) * num_items);

    parallel_for(blocked_range<size_t>(0, num_items, 1024 * 1024 + 1), [&](auto range) {
      unsigned int seed = range.begin();
      for (size_t i = range.begin(); i < range.end(); i++) {
        h_keys[i] = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);;
        h_payload[i] = h_keys[i];
        h_keys_out[i] = 0;
        h_payload_out[i] = 0;
        h_flags[i] = 0;
      }
    });

    for (int t = 0; t < num_trials; t++) {
        for (int i = 0; i <= 10; i++) {
            float selectivity = i/10.0;

            float time_if_cpu;
            float time_if_pred_cpu;
            float time_simd_cpu;
            int num_selected_if_cpu;
            int num_selected_simd_cpu;
            int num_selected_if_pred_cpu;

            time_if_cpu = selectIfCPU(h_keys, h_payload, h_keys_out, h_payload_out, num_items, selectivity, 0.0,
                num_selected_if_cpu);

            time_simd_cpu = selectSIMDCPU(h_keys, h_payload, h_keys_out, h_payload_out, num_items, selectivity, 0.0,
                num_selected_simd_cpu);

            time_if_pred_cpu = selectPredCPU(h_keys, h_payload, h_keys_out, h_payload_out, num_items, selectivity, 0.0,
                num_selected_if_pred_cpu);

            int s = num_selected_if_cpu;
            if (!(s == num_selected_simd_cpu &&
                s == num_selected_if_pred_cpu)) {
                cout << "Answers don't match. "
                     << "\n\tif_cpu: " << num_selected_if_cpu
                     << "\n\tsimd_cpu: " <<  num_selected_simd_cpu
                     << endl;
            }

            cout<< "{"
                << "\"selectivity\":" << selectivity
                << ",\"time_if_cpu\":" << time_if_cpu
                << ",\"time_simd_cpu\":" << time_simd_cpu
                << ",\"time_if_pred_cpu\":" << time_if_pred_cpu
                << ",\"num_selected\":" << num_selected_if_cpu
                << ",\"num_entries\":" << num_items
                << ",\"selectivity_real\":" << (((float) num_selected_if_cpu) / num_items)
                << "}" << endl;
        }
    }

    return 0;
}

