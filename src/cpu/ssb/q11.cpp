// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <chrono>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/cpu_utils.h"
#include "ssb_utils.h"

using namespace std;
using namespace tbb;

/*
Implementation of Q11
select sum(lo_extendedprice * lo_discount) as revenue from lineorder,date where lo_orderdate = d_datekey and d_year = 1993 and lo_discount>=1 and lo_discount<=3 and lo_quantity<25;
*/
float runQuery(int* lo_orderdate, int* lo_discount, int* lo_quantity, int* lo_extendedprice, int num_items) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();

  tbb::atomic<unsigned long long> revenue = 0;

  parallel_for(blocked_range<size_t>(0, num_items, num_items/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned long long local_revenue = 0;
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        bool selection_flag = (lo_orderdate[i] > 19930000 && lo_orderdate[i] < 19940000);
        selection_flag = selection_flag && (lo_quantity[i] < 25);
        selection_flag = selection_flag && (lo_discount[i] >= 1 && lo_discount[i] <= 3);
        if (selection_flag) {
          local_revenue += selection_flag * (lo_extendedprice[i] * lo_discount[i]);
        }
      }
    }
    for (int i = end_batch; i < end; i++) {
      bool selection_flag = (lo_orderdate[i] > 19930000 && lo_orderdate[i] < 19940000);
      selection_flag = selection_flag && (lo_quantity[i] < 25);
      selection_flag = selection_flag && (lo_discount[i] >= 1 && lo_discount[i] <= 3);
      if (selection_flag) {
        local_revenue += selection_flag * (lo_extendedprice[i] * lo_discount[i]);
      }
    }
    revenue.fetch_and_add(local_revenue);
  });

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
  int *lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);

  cout << "** LOADED DATA **" << endl;

  // Run trials
  for (int t = 0; t < num_trials; t++) {
  //while (1) {
    float time_query = runQuery(lo_orderdate,
                                lo_discount,
                                lo_quantity,
                                lo_extendedprice,
                                LO_LEN);

    cout << "{" << "\"query\":11" << ",\"time_query\":" << time_query << "}" << endl;
    //int x;
    //cin >> x;
  }

  return 0;
}
