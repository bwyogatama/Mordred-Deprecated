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

float selectIfCPU(float* in, float* out, int num_items, float cutpoint,
    int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<int> counter;
  counter = 0;

  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    int count = 0;
    for (size_t i = range.begin(); i < range.end(); i++) {
      count += (in[i] < cutpoint);
    }
    int offset = counter.fetch_and_add(count);
    for (size_t i = range.begin(); i < range.end(); i++) {
      if (in[i] < cutpoint) {
        out[offset++] = in[i];
      }
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

float selectFlaggedCPU(float* in, float* val, float* out, char* flags,
    int num_items, float cutpoint, int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<uint> counter = 0;
  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    uint count = 0;
    for (size_t i = range.begin(); i < range.end(); i++) {
      flags[i] = (in[i] < cutpoint);
      count += flags[i];
    }
    uint offset = counter.fetch_and_add(count);
    for (size_t i = range.begin(); i < range.end(); i++) {
      if (flags[i]) {
        out[offset++] = val[i];
      }
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  return (chrono::duration_cast<chrono::milliseconds>(finish-start)).count();
}

float selectIfPredCPU(float* in, float* out, int num_items, float cutpoint,
    int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<uint> counter = 0;
  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    uint count = 0;
    for (size_t i = range.begin(); i < range.end(); i++) {
      count += (in[i] < cutpoint);
    }
    uint offset = counter.fetch_and_add(count);
    for (size_t i = range.begin(); i < range.end(); i++) {
      out[offset] = in[i];
      offset += (in[i] < cutpoint);
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

float selectFlaggedPredCPU(float* in, float* val, float* out, char* flags,
    int num_items, float cutpoint, int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<uint> counter = 0;
  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    uint count = 0;
    for (size_t i = range.begin(); i < range.end(); i++) {
      flags[i] = (in[i] < cutpoint);
      count += flags[i];
    }
    uint offset = counter.fetch_and_add(count);
    for (size_t i = range.begin(); i < range.end(); i++) {
      out[offset] = val[i];
      offset += flags[i];
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}


float selectSIMDCPU_nostream(float* in, float* out, int num_items, float cutpoint,
    int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<int> counter;
  counter = 0;

  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    int count = 0;
    #pragma simd
    for (size_t i = range.begin(); i < range.end(); i++) {
      count += (in[i] < cutpoint);
    }
    int offset = counter.fetch_and_add(count);
    const __m256 all_1 = _mm256_set1_ps(0xffff);
    const __m256 cps = _mm256_set1_ps(cutpoint);
    size_t o = range.begin();
    if (range.begin() % 8 != 0 || range.end() % 8 != 0)
        printf("TODO. input data does not align with AVX2\n");
    for (size_t i = range.begin(); i < range.end(); i+= 8) {
      __m256 invec = _mm256_load_ps(&in[i]);
      __m256 comp = _mm256_cmp_ps(invec, cps, _CMP_GT_OQ);
      if (!_mm256_testc_si256(comp, all_1)) {
        size_t m = _mm256_movemask_ps(comp);
        m ^= 255;
        do {
          size_t j = __tzcnt_u64(m);
          uint32_t k = in[i + j];
          out[offset++] = in[i];
          m &= m - 1;
        } while(m);
      }
    }
  });
  finish = chrono::high_resolution_clock::now();

  num_selected = counter.fetch_and_add(0);

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

float selectSIMDCPU(float* in, float* out, int num_items, float cutpoint,
    int& num_selected) {
  chrono::high_resolution_clock::time_point start, finish;
  const int batch_size = BATCH_SIZE;

  start = chrono::high_resolution_clock::now();
  tbb::atomic<int> counter;
  tbb::atomic<int> counter_tail;
  counter = 0;
  counter_tail = 0;
  float tail_buffer[32*64];
  __m256 mask = _mm256_set1_ps(cutpoint);
  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    int k = 0;
    int buf_size = 256 + 8;
    float buffer[buf_size] __attribute__ ((aligned (32)));
    for (size_t i = range.begin(); i < range.end(); i += 8) {
      __m256 key = _mm256_load_ps(&in[i]);
      // generates bitmask - 1111 for true 0000 for false
      __m256 cmp = _mm256_cmp_ps(key, mask, _CMP_LT_OS);
      /* load key columns and evaluate predicates */
      if (!_mm256_testz_ps(cmp, cmp)) {
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
        __m256 ptr = _mm256_permutevar8x32_ps(key, shufmask);
        __m256i cmp_i = _mm256_cvtps_epi32(cmp);
        _mm256_maskstore_ps(&buffer[k], cmp_i, ptr);
        k += _mm_popcnt_u64(m);
        if (k > buf_size - 8) {
          int offset = counter.fetch_and_add(buf_size - 8);
          /* flush the buffer */
          int b;
          for (b=0; b!=buf_size-8; b+=8) {
            /* dereference column values and store */
            __m256 ptr = _mm256_load_ps(&buffer[b]);
            _mm256_stream_ps(&out[offset + b], ptr);
          }
            /* move extra items to the start of the buffer */
          ptr = _mm256_load_ps(&buffer[b]);
          _mm256_store_ps(&buffer[0], ptr);
          k -= buf_size - 8;
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
      __m256 ptr = _mm256_load_ps(&buffer[b]);
      _mm256_stream_ps(&out[offset + b], ptr);
    }
    offset = counter_tail.fetch_and_add(k - ((k >> 3) << 3));
    //for (; b < k; b++) {
      //out[offset + b] = buffer[b];
    /*}*/
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

    float *h_in, *h_val, *h_out;
    char *h_flags;

    h_in = (float*) _mm_malloc(sizeof(float) * num_items, 256);
    h_val = (float*) malloc(sizeof(float) * num_items);
    h_out = (float*) _mm_malloc(sizeof(float) * num_items, 256);
    h_flags = (char*) malloc(sizeof(float) * num_items);

    parallel_for(blocked_range<size_t>(0, num_items, 32 * 1024), [&](auto range) {
      unsigned int seed = range.begin();
      for (size_t i = range.begin(); i < range.end(); i++) {
        h_in[i] = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);;
        h_val[i] = h_in[i];
        h_out[i] = 0;
        h_flags[i] = 0;
      }
    });

    for (int t = 0; t < num_trials; t++) {
        for (int i = 0; i <= 10; i++) {
            float selectivity = i/10.0;

            float time_flagged_cpu, time_if_cpu;
            float time_simd_cpu_no_stream, time_simd_cpu;
            float time_flagged_pred_cpu, time_if_pred_cpu;
            int num_selected_flagged_cpu, num_selected_if_cpu;
            int num_selected_simd_cpu_no_stream, num_selected_simd_cpu;
            int num_selected_flagged_pred_cpu, num_selected_if_pred_cpu;

            time_if_cpu = selectIfCPU(h_in, h_out, num_items, selectivity,
                num_selected_if_cpu);

            time_simd_cpu_no_stream = selectSIMDCPU_nostream(h_in, h_out, num_items, selectivity,
                num_selected_simd_cpu_no_stream);

            time_simd_cpu = selectSIMDCPU(h_in, h_out, num_items, selectivity,
                num_selected_simd_cpu);

            time_flagged_cpu = selectFlaggedCPU(h_in, h_val, h_out, h_flags, num_items, selectivity,
                num_selected_flagged_cpu);

            time_if_pred_cpu = selectIfPredCPU(h_in, h_out, num_items, selectivity,
                num_selected_if_pred_cpu);

            time_flagged_pred_cpu = selectFlaggedPredCPU(h_in, h_val, h_out, h_flags, num_items, selectivity,
                num_selected_flagged_pred_cpu);

            int s = num_selected_flagged_cpu;
            if (!(s == num_selected_flagged_cpu &&
                s == num_selected_if_cpu &&
                s == num_selected_simd_cpu &&
                s == num_selected_simd_cpu_no_stream &&
                s == num_selected_flagged_pred_cpu && s == num_selected_if_pred_cpu)) {
                cout << "Answers don't match. "
                     << "\n\tif_cpu: " << num_selected_if_cpu
                     << "\n\tsimd_cpu: " <<  num_selected_simd_cpu
                     << endl;
            }

            cout<< "{"
                << "\"selectivity\":" << selectivity
                << ",\"time_if_cpu\":" << time_if_cpu
                << ",\"time_simd_cpu\":" << time_simd_cpu
                << ",\"time_simd_cpu_no_stream\":" << time_simd_cpu_no_stream
                << ",\"time_flagged_cpu\":" << time_flagged_cpu
                << ",\"time_if_pred_cpu\":" << time_if_pred_cpu
                << ",\"time_flagged_pred_cpu\":" << time_flagged_pred_cpu
                << ",\"num_selected\":" << num_selected_flagged_cpu
                << ",\"num_entries\":" << num_items
                << ",\"selectivity_real\":" << (((float) num_selected_flagged_cpu) / num_items)
                << "}" << endl;
        }
    }

    return 0;
}

