#include <math.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/cpu_utils.h"

using namespace tbb;
using namespace std;

float projectCPU(float* in1, float* in2, float* out, int num_items) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();
  parallel_for(blocked_range<size_t>(0, num_items, 1024 * 1024), [&](auto range) {
    //int duration = range.end() - range.begin();
    //if (duration % 8 != 0) cout << "Mis" << endl;
    for (size_t i = range.begin(); i < range.end(); i += 8) {
      __m256 in1vec = _mm256_load_ps(&in1[i]);
      __m256 mul1 = _mm256_set_ps(2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0);
      __m256 in2vec = _mm256_load_ps(&in2[i]);
      __m256 mul2 = _mm256_set_ps(3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0);
      __m256 p1 = _mm256_mul_ps(in1vec, mul1);
      __m256 p2 = _mm256_mul_ps(in2vec, mul2);
      __m256 res = _mm256_add_ps(p1, p2);
      _mm256_stream_ps(&out[i], res);
      // Using store instead of stream increases runtime from 64 to 90
      // _mm256_store_ps(&out[i], res);
    }
  });

  finish = chrono::high_resolution_clock::now();
  uint time_proj = (chrono::duration_cast<chrono::microseconds>(finish-start)).count();
  return ((float) time_proj) / 1000;
}

float projectOldCPU(float* in1, float* in2, float* out, int num_items) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();
  parallel_for(blocked_range<size_t>(0, num_items, 1024 * 1024), [&](auto range) {
    for (size_t i = range.begin(); i < range.end(); i++) {
      out[i] = 2*in1[i] + 3*in2[i];
    }
  });
  finish = chrono::high_resolution_clock::now();
  uint time_proj = (chrono::duration_cast<chrono::microseconds>(finish-start)).count();
  return ((float) time_proj) / 1000;
}

float projectSigmoidCPU(float* in1, float* in2, float* out, int num_items) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();
  parallel_for(blocked_range<size_t>(0, num_items, 1024 * 1024), [&](auto range) {
    for (size_t i = range.begin(); i < range.end(); i += 8) {
      __m256 in1vec = _mm256_load_ps(&in1[i]);
      __m256 mul1 = _mm256_set_ps(-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0);
      __m256 in2vec = _mm256_load_ps(&in2[i]);
      __m256 mul2 = _mm256_set_ps(-3.0,-3.0,-3.0,-3.0,-3.0,-3.0,-3.0,-3.0);
      __m256 p1 = _mm256_mul_ps(in1vec, mul1);
      __m256 p2 = _mm256_mul_ps(in2vec, mul2);
      __m256 addres = _mm256_add_ps(p1, p2);
      __m256 expres = _mm256_exp_ps(addres);
      __m256 one = _mm256_set_ps(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0);
      __m256 addres2 = _mm256_add_ps(one,expres);
      __m256 res = _mm256_log_ps(_mm256_div_ps(one, addres2));
      _mm256_stream_ps(&out[i], res);
    }
  });
  finish = chrono::high_resolution_clock::now();
  uint time_proj = (chrono::duration_cast<chrono::microseconds>(finish-start)).count();
  return ((float) time_proj) / 1000;
}

float projectSigmoidOldCPU(float* in1, float* in2, float* out, int num_items) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();
  parallel_for(blocked_range<size_t>(0, num_items, 1024 * 1024), [&](auto range) {
    for (size_t i = range.begin(); i < range.end(); i += 1) {
      out[i] = 1.0f / (1.0f + expf(-2*in1[i] -3*in2[i]));
    }
  });
  finish = chrono::high_resolution_clock::now();
  uint time_proj = (chrono::duration_cast<chrono::microseconds>(finish-start)).count();
  return ((float) time_proj) / 1000;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_items           = 1<<28;
  int num_trials          = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--n=<input items>] "
          "[--t=<num trials>] "
          "[--device=<device-id>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  float *h_in1, *h_in2, *h_out;
  h_in1 = (float*) _mm_malloc(sizeof(float) * num_items, 256);
  h_in2 = (float*) _mm_malloc(sizeof(float) * num_items, 256);
  h_out = (float*) _mm_malloc(sizeof(float) * num_items, 256);

  srand(1231);

  parallel_for(blocked_range<size_t>(0, num_items, 32 * 1024), [&](auto range) {
    unsigned int seed = range.begin();
    for (size_t i = range.begin(); i < range.end(); i++) {
      h_in1[i] = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);;
      h_in2[i] = h_in1[i]; // static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
      h_out[i] = 0;
    }
  });

  float time_proj_cpu, time_proj_opt_cpu;
  float time_proj_sigmoid_cpu, time_proj_opt_sigmoid_cpu;

  // To get the right numbers
  // run sigmoid with clang
  // run sigmoid with icc
  for (int t = 0; t < num_trials; t++) {
    time_proj_cpu = projectOldCPU(h_in1, h_in2, h_out, num_items);
    time_proj_opt_cpu = projectCPU(h_in1, h_in2, h_out, num_items);
    time_proj_sigmoid_cpu = projectSigmoidOldCPU(h_in1, h_in2, h_out, num_items);
    time_proj_opt_sigmoid_cpu = projectSigmoidCPU(h_in1, h_in2, h_out, num_items);

    cout<< "{"
        << "\"time_proj_cpu\":" << time_proj_cpu
        << "\"time_proj_opt_cpu\":" << time_proj_opt_cpu
        << ",\"time_proj_sigmoid_cpu\":" << time_proj_sigmoid_cpu
        << ",\"time_proj_opt_sigmoid_cpu\":" << time_proj_opt_sigmoid_cpu
        << "}" << endl;
  }

  return 0;
}

