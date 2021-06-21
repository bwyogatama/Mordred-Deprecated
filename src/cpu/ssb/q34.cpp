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
  int s_city;
  int c_city;
  std::atomic<int> revenue;
};

/*
Implementation of q34
select c_city,s_city,d_year,sum(lo_revenue) as revenue from lineorder,customer,supplier,date where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city = 'UNITED KI1' or c_city = 'UNITED KI5') and (s_city = 'UNITED KI1' or s_city = 'UNITED KI5') and d_yearmonth = 'Dec1997' group by c_city,s_city,d_year order by d_year asc,revenue desc;
*/

float runQuery(int* lo_orderdate, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int *d_datekey, int* d_year, int *d_yearmonthnum, int d_len,
    int *s_suppkey, int* s_city, int s_len,
    int *c_custkey, int* c_city, int c_len) {
  chrono::high_resolution_clock::time_point start, finish;
  start = chrono::high_resolution_clock::now();

  int d_val_len = 19981230 - 19920101 + 1;
  int d_val_min = 19920101;
  int *ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
  int *ht_s = (int*)malloc(2 * s_len * sizeof(int));
  int *ht_c = (int*)malloc(2 * c_len * sizeof(int));

  memset(ht_d, 0, 2 * d_val_len * sizeof(int));
  memset(ht_s, 0, 2 * s_len * sizeof(int));
  memset(ht_c, 0, 2 * c_len * sizeof(int));

  // Build hashtable d
  parallel_for(blocked_range<size_t>(0, d_len, d_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (d_yearmonthnum[i] == 199712) {
        int key = d_datekey[i];
        int val = d_year[i];
        int hash = HASH_WM(key, d_val_len, d_val_min);
        ht_d[hash << 1] = key;
        ht_d[(hash << 1) + 1] = val;
      }
    }
  });

  // Build hashtable c
  parallel_for(blocked_range<size_t>(0, c_len, c_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (c_city[i] == 231 || c_city[i] == 235) {
        int key = c_custkey[i];
        int hash = HASH(key, c_len);
        ht_c[hash << 1] = key;
        ht_c[(hash << 1) + 1] = c_city[i];
      }
    }
  });

  // Build hashtable s
  parallel_for(blocked_range<size_t>(0, s_len, s_len/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (s_city[i] == 231 || s_city[i] == 235) {
        int key = s_suppkey[i];
        int hash = HASH(key, s_len);
        ht_s[hash << 1] = key;
        ht_s[(hash << 1)+1] = s_city[i];
      }
    }
  });

  int num_slots = ((1998-1992+1) * 250 * 250);
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
        long long s_slot = reinterpret_cast<long long*>(ht_s)[hash];
        if (s_slot != 0) {
          int s_city = s_slot >> 32;
          hash = HASH(lo_custkey[i], c_len);
          long long c_slot = reinterpret_cast<long long*>(ht_c)[hash];
          if (c_slot != 0) {
            int c_city = c_slot >> 32;
            hash = HASH_WM(lo_orderdate[i], d_val_len, d_val_min);
            long long d_slot = reinterpret_cast<long long*>(ht_d)[hash];
            if (d_slot != 0) {
              int year = d_slot >> 32;
              hash = (s_city * 250 * 7 + c_city * 7 + (year - 1992)) % num_slots;
              res[hash].year = year;
              res[hash].c_city = c_city;
              res[hash].s_city = s_city;
              res[hash].revenue += lo_revenue[i];
            }
          }
        }
      }
    }
    for (int i = end_batch ; i < end; i++) {
      int hash = HASH(lo_suppkey[i], s_len);
      int slot = ht_s[hash << 1];
      if (slot != 0) {
        int s_city = ht_s[(hash << 1) + 1];
        hash = HASH(lo_custkey[i], c_len);
        slot = ht_c[hash << 1];
        if (slot != 0) {
            int c_city = ht_c[(hash << 1) + 1];
            hash = HASH_WM(lo_orderdate[i], d_val_len, d_val_min);
            slot = ht_d[hash << 1];
            if (slot != 0) {
              int year = ht_d[(hash << 1) + 1];
              hash = (s_city * 250 * 7 + c_city * 7 + (year - 1992)) % num_slots;
              res[hash].year = year;
              res[hash].c_city = c_city;
              res[hash].s_city = s_city;
              res[hash].revenue += lo_revenue[i];
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
      cout << res[i].year << " " << res[i].c_city << " " << res[i].s_city << " " << res[i].revenue << endl;
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
  int *lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);

  int *d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *d_year = loadColumn<int>("d_year", D_LEN);
  int *d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

  int *s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *s_city = loadColumn<int>("s_city", S_LEN);

  int *c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *c_city = loadColumn<int>("c_city", C_LEN);

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
        lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, LO_LEN,
        d_datekey, d_year, d_yearmonthnum, D_LEN,
        s_suppkey, s_city, S_LEN,
        c_custkey, c_city, C_LEN);
    cout<< "{"
        << "\"query\":34"
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}
