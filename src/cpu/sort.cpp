#include <iostream>
#include <math.h>
#include <chrono>
#include <cstdlib>

#include "tbb/parallel_sort.h"
#include "tbb/parallel_for.h"

using namespace tbb;
using namespace std;

float sortKeysCPU(float* key_buf, int num_items) {
  chrono::high_resolution_clock::time_point start =
    chrono::high_resolution_clock::now();
  parallel_sort(key_buf, key_buf + num_items);
  chrono::high_resolution_clock::time_point finish =
    chrono::high_resolution_clock::now();
  uint time_sort_float =
    (chrono::duration_cast<chrono::milliseconds>(finish-start)).count();
  return time_sort_float;
}

int main(int argc, char** argv)
{
    int num_items           = 1<<28;
    int num_trials          = 3;

    float *key_buf = new float[num_items];
    float *key_alt_buf = new float[num_items];
    uint  *value_buf = new uint[num_items];
    uint  *value_alt_buf = new uint[num_items];

    srand(1231);

    parallel_for(blocked_range<size_t>(0, num_items, 32 * 1024), [&](auto range) {
      for (size_t i = range.begin(); i < range.end(); i++) {
        key_buf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
        value_buf[i] = rand();
      }
    });

    memcpy(key_alt_buf, key_buf, sizeof(float) * num_items);
    memcpy(value_alt_buf, value_buf, sizeof(uint) * num_items);

    for (int i = 0; i < num_trials; i++) {
      float time_sort_cpu;
      time_sort_cpu = sortKeysCPU(key_buf, num_items);

      memcpy(key_buf, key_alt_buf, sizeof(float) * num_items);
      memcpy(value_buf, value_alt_buf, sizeof(uint) * num_items);

      cout<< "{"
          << "\"time_sort_cpu\":" << time_sort_cpu
          << "}" << endl;
    }

    return 0;
}
