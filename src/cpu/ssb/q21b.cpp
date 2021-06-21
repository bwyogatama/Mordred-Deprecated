// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <chrono>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/cpu_utils.h"
#include "ssb_utils.h"

#define BATCH_SIZE 2048

using namespace std;
using namespace tbb;

/*
Implementation of Q21
select sum(lo_extendedprice * lo_discount) as revenue from lineorder,date where lo_orderdate = d_datekey and d_year = 1993 and lo_discount>=1 and lo_discount<=3 and lo_quantity<25;
*/
float runQuery(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice, bool* selection_flags, int num_items) {
  const int batch_size = BATCH_SIZE;

  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();

  tbb::atomic<unsigned long long> revenue = 0;

  // Deal with remaining items in the last batch
  const int remaining_items = num_items % batch_size;
  const int num_batches = (remaining_items == 0) ? num_items / batch_size : num_items / batch_size + 1 ;
  const int last_batch_start = (num_batches-1) * batch_size;

  // d_year = 1993
  int batch_index = 1;

  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    if (batch_index < num_batches) {
      #pragma simd
      for (size_t i = range.begin(); i < range.end(); i++) {
        selection_flags[i] = (lo_orderdate[i] > 19930000 && lo_orderdate[i] < 19940000);
      }
    }

    batch_index++;
  });
  for (int i = last_batch_start; i < last_batch_start + remaining_items; i++) {
    selection_flags[i] = (lo_orderdate[i] > 19930000 && lo_orderdate[i] < 19940000);
  }

  // lo_quantity < 25
  batch_index = 1;

  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    if (batch_index < num_batches) {
      #pragma simd
      for (size_t i = range.begin(); i < range.end(); i++) {
        selection_flags[i] = selection_flags[i] && (lo_quantity[i] < 25);
      }
    }

    batch_index++;
  });
  for (int i = last_batch_start; i < last_batch_start + remaining_items; i++) {
    selection_flags[i] = selection_flags[i] && (lo_quantity[i] < 25);
  }

  // 1 <= lo_discount <= 3
  batch_index = 1;

  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    if (batch_index <  num_batches) {
      #pragma simd
      for (size_t i = range.begin(); i < range.end(); i++) {
        selection_flags[i] = selection_flags[i] && (lo_discount[i] >= 1 && lo_discount[i] <= 3);
      }
    }

    batch_index++;
  });
  for (int i = last_batch_start; i < last_batch_start + remaining_items; i++) {
    selection_flags[i] = selection_flags[i] && (lo_discount[i] >= 1 && lo_discount[i] <= 3);
  }

  // select sum(lo_extendedprice * lo_discount) as revenue
  batch_index = 1;

  parallel_for(blocked_range<size_t>(0, num_items, batch_size), [&](auto range) {
    unsigned long long local_revenue = 0;
    if (batch_index < num_batches) {
      for (size_t i = range.begin(); i < range.end(); i++) {
        if (selection_flags[i]) {
          local_revenue += (lo_extendedprice[i] * lo_discount[i]);
          printf("%d %d\n", lo_extendedprice[i],  lo_discount[i]);
        }
      }
    }

    batch_index++;
    revenue.fetch_and_add(local_revenue);
  });
  unsigned long long local_revenue = 0;
  for (int i = last_batch_start; i < last_batch_start + remaining_items; i++) {
    if (selection_flags[i]) {
      local_revenue += (lo_extendedprice[i] * lo_discount[i]);
      printf("%d %d\n", lo_extendedprice[i], lo_discount[i]);
    }
  }
  revenue.fetch_and_add(local_revenue);

  finish = chrono::high_resolution_clock::now();

  cout << "Revenue: " << revenue << endl;

  std::chrono::duration<double> diff = finish - start;
  return diff.count() * 1000;
}

/**
 * Main
 */
int main(int argc, char** argv) {
  int num_trials = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
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

  // Load in data
  int *lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);

  int *p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *p_brand1 = loadColumn<int>("p_brand1", P_LEN);
  int *p_category = loadColumn<int>("p_category", P_LEN);

  int *d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *d_year = loadColumn<int>("d_year", D_LEN);

  int *s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *s_region = loadColumn<int>("s_region", S_LEN);

  cout << "** LOADED DATA **" << endl;

  // For selection: Initally assume everything is selected
  bool *selection_flags = (bool*) malloc(sizeof(bool) * LO_LEN);
  for (size_t i = 0; i < LO_LEN; i++) {
    selection_flags[i] = true;
  }

  // Run trials
  for (int t = 0; t < num_trials; t++) {
    float time_query = runQuery(lo_orderdate,
                                lo_discount,
                                lo_quantity,
                                lo_extendedprice,
                                selection_flags,
                                LO_LEN);

    cout << "{" << "\"query\":1" << ",\"time_query\":" << time_query << "}" << endl;
  }

  return 0;
}
