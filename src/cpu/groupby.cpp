#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/cpu_utils.h"

using namespace tbb;
using namespace std;

float fullAggCPU(float* key_buf, int num_items) {
  chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
  float result = parallel_reduce(
      blocked_range<float*>(key_buf, key_buf + num_items),
      0.f,
      [](const blocked_range<float*>& r, float init)->float {
          for (float* a=r.begin(); a!=r.end(); ++a )
              init += *a;
          return init;
      },
      []( float x, float y )->float {
          return x+y;
      }
  );

  chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
  uint time_full_agg = (chrono::duration_cast<chrono::milliseconds>(finish-start)).count();
  return time_full_agg;
}

int main(int argc, char** argv)
{
    uint num_items          = 1 << 28;
    uint num_keys           = 1 << 24;
    int num_trials          = 3;
    bool full_agg           = true;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("k", num_keys);    
    args.GetCmdLineArgument("t", num_trials);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items>] "
            "[--n=<num keys>] "           
            "[--t=<num trials>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    uint *h_keys;
    float *h_value;
    h_keys = (uint*) malloc(sizeof(uint) * num_items);
    h_value = (float*) malloc(sizeof(float) * num_items);

    srand(1231);

    parallel_for(blocked_range<size_t>(0, num_items, 32 * 1024), [&](auto range) {
      for (size_t i = range.begin(); i < range.end(); i++) {
        h_keys[i] = rand() & (num_keys - 1) ;;
        h_value[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
      }
    });

    for (int i = 0; i < num_trials; i++) {
        // Full Aggregation.
        float time_full_agg_cpu;

        time_full_agg_cpu = fullAggCPU(h_value, num_items);
 
        cout<< "{"
            << "\"time_full_agg_cpu\":" << time_full_agg_cpu
            << "}" << endl;
    }

    free(h_keys);
    free(h_value);

    return 0;
}