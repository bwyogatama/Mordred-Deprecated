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
  int c_nation;
  std::atomic<long long> revenue;
};

/*
Implementation of q41
select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit from lineorder,supplier,customer,part, date where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year,c_nation order by d_year,c_nation;
*/

float runQuery(int* lo_orderdate, int* lo_custkey, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int* lo_supplycost, int lo_len,
    int *d_datekey, int *d_year, int d_len,
    int *p_partkey, int *p_mfgr, int p_len,
    int *s_suppkey, int* s_region, int s_len,
    int *c_custkey, int* c_region, int* c_nation, int c_len) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();

  int d_val_len = 19981230 - 19920101 + 1;
  int d_val_min = 19920101;
  int *ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
  int *ht_s = (int*)malloc(2 * s_len * sizeof(int));
  int *ht_c = (int*)malloc(2 * c_len * sizeof(int));
  int *ht_p = (int*)malloc(2 * p_len * sizeof(int));

  memset(ht_d, 0, 2 * d_val_len * sizeof(int));
  memset(ht_s, 0, 2 * s_len * sizeof(int));
  memset(ht_c, 0, 2 * c_len * sizeof(int));
  memset(ht_p, 0, 2 * p_len * sizeof(int));

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

  // Build hashtable c
  parallel_for(blocked_range<size_t>(0, c_len, c_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (c_region[i] == 1) {
        int key = c_custkey[i];
        int val = c_nation[i];
        int hash = HASH(key, c_len);
        ht_c[hash << 1] = key;
        ht_c[(hash << 1) + 1] = val;
      }
    }
  });

  // Build hashtable s
  parallel_for(blocked_range<size_t>(0, s_len, s_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (s_region[i] == 1) {
        int key = s_suppkey[i];
        int hash = HASH(key, s_len);
        ht_s[hash << 1] = key;
      }
    }
  });


  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, p_len, p_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (p_mfgr[i] == 0 || p_mfgr[i] == 1) {
        int key = p_partkey[i];
        int hash = HASH(key, p_len);
        ht_p[hash << 1] = key;
      }
    }
  });

  int num_slots = ((1998-1992+1) * 25);
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
        int hash = HASH(lo_suppkey[i], s_len);
        int slot = ht_s[hash << 1];
        if (slot != 0) {
          hash = HASH(lo_custkey[i], c_len);
          slot = ht_c[hash << 1];
          if (slot != 0) {
            int c_nation = ht_c[(hash << 1) + 1];
            hash = HASH(lo_partkey[i], p_len);
            slot = ht_p[hash << 1];
            if (slot != 0) {
              hash = HASH_WM(lo_orderdate[i], d_val_len, d_val_min);
              slot = ht_d[hash << 1];
              int year = ht_d[(hash << 1) + 1];
              hash = (c_nation * 7 + (year - 1992)) % num_slots;
              res[hash].year = year;
              res[hash].c_nation = c_nation;
              res[hash].revenue += lo_revenue[i] - lo_supplycost[i];
            }
          }
        }
      }
    }
    for (int i = end_batch ; i < end; i++) {
      int hash = HASH(lo_suppkey[i], s_len);
      int slot = ht_s[hash << 1];
      if (slot != 0) {
        hash = HASH(lo_custkey[i], c_len);
        slot = ht_c[hash << 1];
        if (slot != 0) {
            int c_nation = ht_c[(hash << 1) + 1];
            hash = HASH(lo_partkey[i], p_len);
            slot = ht_p[hash << 1];
            if (slot != 0) {
              hash = HASH_WM(lo_orderdate[i], d_val_len, d_val_min);
              slot = ht_d[hash << 1];
              int year = ht_d[(hash << 1) + 1];
              hash = (c_nation * 7 + (year - 1992)) % num_slots;
              res[hash].year = year;
              res[hash].c_nation = c_nation;
              res[hash].revenue += lo_revenue[i] - lo_supplycost[i];
            }
        }
      }
    }
  });

  finish = chrono::high_resolution_clock::now();

  cout << "Result:" << endl;

  int res_count = 0;
  for (int i=0; i<num_slots; i++) {
    if (res[i].year != 0) {
      cout << res[i].year << " " << res[i].c_nation << " " << res[i].revenue << endl;
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
  int num_trials = 1;

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
  int *lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
  int *lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);

  int *d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *d_year = loadColumn<int>("d_year", D_LEN);
  int *d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

  int *s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *s_region = loadColumn<int>("s_region", S_LEN);

  int *p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *p_mfgr = loadColumn<int>("p_mfgr", P_LEN);

  int *c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *c_region = loadColumn<int>("c_region", C_LEN);
  int *c_nation = loadColumn<int>("c_nation", C_LEN);

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
        lo_orderdate, lo_custkey, lo_partkey, lo_suppkey, lo_revenue, lo_supplycost, LO_LEN,
        d_datekey, d_year, D_LEN,
        p_partkey, p_mfgr, P_LEN,
        s_suppkey, s_region, S_LEN,
        c_custkey, c_region, c_nation, C_LEN);
    cout<< "{"
        << "\"query\":41"
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}
