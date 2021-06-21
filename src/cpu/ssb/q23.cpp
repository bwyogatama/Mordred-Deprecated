// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <atomic>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "utils/cpu_utils.h"
#include "ssb_utils.h"

using namespace std;
using namespace tbb;

#define HASH_WM(X,Y,Z) ((X-Z) % Y)
#define HASH(X,Y) (X % Y)

struct slot {
  int year;
  int brand1;
  std::atomic<long long> revenue;
};

/*
Implementation of q23
select sum(lo_revenue),d_year,p_brand1 from lineorder,part,supplier,date where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand1 = 'MFGR#2239' and s_region = 'EUROPE' group by d_year,p_brand1 order by d_year,p_brand1;
*/

float runQuery(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* p_partkey, int* p_brand1, int* p_category, int p_len,
    int *d_datekey, int* d_year, int d_len,
    int *s_suppkey, int* s_region, int s_len) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();

  int d_val_len = 19981230 - 19920101 + 1;
  int d_val_min = 19920101;
  int *ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
  int *ht_p = (int*)malloc(2 * p_len * sizeof(int));
  int *ht_s = (int*)malloc(2 * s_len * sizeof(int));

  memset(ht_d, 0, 2 * d_val_len * sizeof(int));
  memset(ht_p, 0, 2 * p_len * sizeof(int));
  memset(ht_s, 0, 2 * s_len * sizeof(int));

  // Build hashtable d
  parallel_for(blocked_range<size_t>(0, d_len, d_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      int key = d_datekey[i];
      int val = d_year[i];
      int hash = HASH_WM(key, d_val_len, d_val_min);
      ht_d[hash << 1] = key;
      ht_d[(hash << 1) + 1] = val;
    }
  });

  // Build hashtable s
  parallel_for(blocked_range<size_t>(0, s_len, s_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (s_region[i] == 3) {
        int key = s_suppkey[i];
        //int val = d_year[i];
        int hash = HASH(key, s_len);
        ht_s[hash << 1] = key;
      }
    }
  });

  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, p_len, p_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (p_brand1[i] == 260) {
        int key = p_partkey[i];
        int hash = HASH(key, p_len);
				ht_p[hash << 1] = key;
        ht_p[(hash << 1)+1] = p_brand1[i];
      }
    }
  });

  int num_slots = ((1998-1992+1) * 1000);
  slot* res = new slot[num_slots];

  for (int i=0; i<num_slots; i++) {
    res[i].year = 0;
  }

  // Probe
  parallel_for(blocked_range<size_t>(0, lo_len, lo_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned long long local_revenue = 0;
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash, slot;
        hash = HASH(lo_partkey[i], p_len);
        long long p_slot = reinterpret_cast<long long*>(ht_p)[hash];
        if (p_slot  != 0) {
          int brand = p_slot >> 32;
          hash = HASH(lo_suppkey[i], s_len);
          slot = ht_s[hash << 1];
          if (slot != 0) {
            hash = HASH_WM(lo_orderdate[i], d_val_len, d_val_min);
            int year = ht_d[(hash << 1) + 1];
            hash = (brand * 7 +  (year - 1992)) % num_slots;
            res[hash].year = year;
            res[hash].brand1 = brand;
            res[hash].revenue += lo_revenue[i];
          }
        }
      }
    }
    for (int i = end_batch ; i < end; i++) {
      int hash, slot;
      hash = HASH(lo_partkey[i], p_len);
      long long p_slot = reinterpret_cast<long long*>(ht_p)[hash];
      if (p_slot  != 0) {
        int brand = p_slot >> 32;
        hash = HASH(lo_suppkey[i], s_len);
        slot = ht_s[hash << 1];
        if (slot != 0) {
          hash = HASH_WM(lo_orderdate[i], d_val_len, d_val_min);
          int year = ht_d[(hash << 1) + 1];
          hash = (brand * 7 +  (year - 1992)) % num_slots;
          res[hash].year = year;
          res[hash].brand1 = brand;
          res[hash].revenue += lo_revenue[i];
        }
      }
    }
  });

  finish = chrono::high_resolution_clock::now();

  cout << "Result:" << endl;

  int res_count = 0;
  for (int i=0; i<num_slots; i++) {
    if (res[i].year != 0) {
      cout << res[i].year << " " << res[i].brand1 << " " << res[i].revenue << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;

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
    float time_query;
    time_query = runQuery(
        lo_orderdate, lo_partkey, lo_suppkey, lo_revenue, LO_LEN,
        p_partkey, p_brand1, p_category, P_LEN,
        d_datekey, d_year, D_LEN,
        s_suppkey, s_region, S_LEN);
    cout<< "{"
        << "\"query\":23"
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}
