#ifndef _QUERY_PROCESSING_H_
#define _QUERY_PROCESSING_H_

#include "QueryOptimizer.h"
#include "GPUProcessing3.h"
#include "tbb/tbb.h"

#include <chrono>
#include <atomic>
#include <unistd.h>
#include <iostream>
#include <stdio.h>

#define NUM_QUERIES 4

using namespace std;
using namespace tbb;

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

void probe_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int h_total, int start_offset, int* offset);

void probe_group_by_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int fact_len, int start_index);


class QueryProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;

  int min_key1, min_key2, min_key3, min_key4;
  int min_val1, min_val2, min_val3, min_val4, total_val;
  int unique_val1, unique_val2, unique_val3, unique_val4;
  int dim_len1, dim_len2, dim_len3, dim_len4;
  int *ht1, *ht2, *ht3, *ht4;
  int *d_ht1, *d_ht2, *d_ht3, *d_ht4;

  int* res;
  int* d_res;

  vector<int> query_freq;

  chrono::high_resolution_clock::time_point begin_time;

  QueryProcessing(CacheManager* _cm) {
    qo = new QueryOptimizer();
    cm = qo->cm;
    begin_time = chrono::high_resolution_clock::now();
    query_freq.resize(NUM_QUERIES);
  }

  int generate_rand_query() {
    return rand() % NUM_QUERIES;
  }

  void processQuery() {
    int query = generate_rand_query();
    qo->parseQuery(query);
    prepareQuery(query);
    updateStatsQuery(query);
    qo->clearVector();
    endQuery(query);

  }

  void prepareQuery(int query) {
    if (query == 0) {

      min_key1 = 19920101;
      min_key2 = 0;
      min_key3 = 0;
      min_key4 = 0;

      min_val1 = 1992;
      min_val2 = 0;
      min_val3 = 0;
      min_val4 = 0;

      unique_val1 = 1;
      unique_val2 = 0;
      unique_val3 = 0;
      unique_val4 = 0;

      dim_len1 = 19981230 - 19920101 + 1;
      dim_len2 = 0;
      dim_len3 = 0;
      dim_len4 = 0;

      total_val = (1998-1992+1);

      ht1 = (int*)malloc(2 * dim_len1 * sizeof(int));
      ht2 = NULL;
      ht3 = NULL;
      ht4 = NULL;

      memset(ht1, 0, 2 * dim_len1 * sizeof(int));

      g_allocator.DeviceAllocate((void**)&d_ht1, 2 * dim_len1 * sizeof(int));
      d_ht2 = NULL;
      d_ht3 = NULL;
      d_ht4 = NULL;

      cudaMemset(d_ht1, 0, 2 * dim_len1 * sizeof(int));

      int res_array_size = total_val * 6;
      res = new int[res_array_size];
      memset(res, 0, res_array_size * sizeof(int));
         
      g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
      cudaMemset(d_res, 0, res_array_size * sizeof(int));

    } else if (query == 1) {

      min_key1 = 0;
      min_key2 = 0;
      min_key3 = 19920101;
      min_key4 = 0;

      min_val1 = 0;
      min_val2 = 0;
      min_val3 = 1992;
      min_val4 = 0;

      unique_val1 = 7;
      unique_val2 = 0;
      unique_val3 = 1;
      unique_val4 = 0;

      dim_len1 = P_LEN;
      dim_len2 = S_LEN;
      dim_len3 = 19981230 - 19920101 + 1;
      dim_len4 = 0;

      total_val = ((1998-1992+1) * (5 * 5 * 40));

      ht1 = (int*)malloc(2 * dim_len1 * sizeof(int));
      ht2 = (int*)malloc(2 * dim_len2 * sizeof(int));
      ht3 = (int*)malloc(2 * dim_len3 * sizeof(int));
      ht4 = NULL;

      memset(ht1, 0, 2 * dim_len1 * sizeof(int));
      memset(ht2, 0, 2 * dim_len2 * sizeof(int));
      memset(ht3, 0, 2 * dim_len3 * sizeof(int));

      g_allocator.DeviceAllocate((void**)&d_ht1, 2 * dim_len1 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht2, 2 * dim_len2 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht3, 2 * dim_len3 * sizeof(int));
      d_ht4 = NULL;

      cudaMemset(d_ht1, 0, 2 * dim_len1 * sizeof(int));
      cudaMemset(d_ht2, 0, 2 * dim_len2 * sizeof(int));
      cudaMemset(d_ht3, 0, 2 * dim_len3 * sizeof(int));

      int res_array_size = total_val * 6;
      res = new int[res_array_size];
      memset(res, 0, res_array_size * sizeof(int));
         
      g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
      cudaMemset(d_res, 0, res_array_size * sizeof(int));

    } else if (query == 2) {

      min_key1 = 0;
      min_key2 = 0;
      min_key3 = 19920101;
      min_key4 = 0;

      min_val1 = 0;
      min_val2 = 0;
      min_val3 = 1992;
      min_val4 = 0;

      unique_val1 = 7;
      unique_val2 = 25 * 7;
      unique_val3 = 1;
      unique_val4 = 0;

      dim_len1 = C_LEN;
      dim_len2 = S_LEN;
      dim_len3 = 19981230 - 19920101 + 1;
      dim_len4 = 0;

      total_val = ((1998-1992+1) * 25 * 25);

      ht1 = (int*)malloc(2 * dim_len1 * sizeof(int));
      ht2 = (int*)malloc(2 * dim_len2 * sizeof(int));
      ht3 = (int*)malloc(2 * dim_len3 * sizeof(int));
      ht4 = NULL;

      memset(ht1, 0, 2 * dim_len1 * sizeof(int));
      memset(ht2, 0, 2 * dim_len2 * sizeof(int));
      memset(ht3, 0, 2 * dim_len3 * sizeof(int));

      g_allocator.DeviceAllocate((void**)&d_ht1, 2 * dim_len1 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht2, 2 * dim_len2 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht3, 2 * dim_len3 * sizeof(int));
      d_ht4 = NULL;

      cudaMemset(d_ht1, 0, 2 * dim_len1 * sizeof(int));
      cudaMemset(d_ht2, 0, 2 * dim_len2 * sizeof(int));
      cudaMemset(d_ht3, 0, 2 * dim_len3 * sizeof(int));

      int res_array_size = total_val * 6;
      res = new int[res_array_size];
      memset(res, 0, res_array_size * sizeof(int));
         
      g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
      cudaMemset(d_res, 0, res_array_size * sizeof(int));

    } else {

      min_key1 = 0;
      min_key2 = 0;
      min_key3 = 0;
      min_key4 = 19920101;

      min_val1 = 0;
      min_val2 = 0;
      min_val3 = 0;
      min_val4 = 1992;

      unique_val1 = 0;
      unique_val2 = 7;
      unique_val3 = 1;
      unique_val4 = 0;

      dim_len1 = P_LEN;
      dim_len2 = C_LEN;
      dim_len3 = S_LEN;
      dim_len4 = 19981230 - 19920101 + 1;

      total_val = ((1998-1992+1) * 25);

      ht1 = (int*)malloc(2 * dim_len1 * sizeof(int));
      ht2 = (int*)malloc(2 * dim_len2 * sizeof(int));
      ht3 = (int*)malloc(2 * dim_len3 * sizeof(int));
      ht4 = (int*)malloc(2 * dim_len4 * sizeof(int));

      memset(ht1, 0, 2 * dim_len1 * sizeof(int));
      memset(ht2, 0, 2 * dim_len2 * sizeof(int));
      memset(ht3, 0, 2 * dim_len3 * sizeof(int));
      memset(ht4, 0, 2 * dim_len4 * sizeof(int));

      g_allocator.DeviceAllocate((void**)&d_ht1, 2 * dim_len1 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht2, 2 * dim_len2 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht3, 2 * dim_len3 * sizeof(int));
      g_allocator.DeviceAllocate((void**)&d_ht4, 2 * dim_len4 * sizeof(int));

      cudaMemset(d_ht1, 0, 2 * dim_len1 * sizeof(int));
      cudaMemset(d_ht2, 0, 2 * dim_len2 * sizeof(int));
      cudaMemset(d_ht3, 0, 2 * dim_len3 * sizeof(int));
      cudaMemset(d_ht4, 0, 2 * dim_len4 * sizeof(int));

      int res_array_size = total_val * 6;
      res = new int[res_array_size];
      memset(res, 0, res_array_size * sizeof(int));
         
      g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
      cudaMemset(d_res, 0, res_array_size * sizeof(int));
    }
  }

  void endQuery(int query) {
    if (query == 0) {
      free(ht1);
      g_allocator.DeviceFree(d_ht1);

    } else if (query == 1) {
      free(ht1);
      free(ht2);
      free(ht3);

      g_allocator.DeviceFree(d_ht1);
      g_allocator.DeviceFree(d_ht2);
      g_allocator.DeviceFree(d_ht3);

    } else if (query == 2) {
      free(ht1);
      free(ht2);
      free(ht3);

      g_allocator.DeviceFree(d_ht1);
      g_allocator.DeviceFree(d_ht2);
      g_allocator.DeviceFree(d_ht3);

    } else {
      free(ht1);
      free(ht2);
      free(ht3);
      free(ht4);

      g_allocator.DeviceFree(d_ht1);
      g_allocator.DeviceFree(d_ht2);
      g_allocator.DeviceFree(d_ht3);
      g_allocator.DeviceFree(d_ht4);
    }

    delete res;
    g_allocator.DeviceFree(d_res);
  }

  void updateStatsQuery(int query) {
    chrono::high_resolution_clock::time_point cur_time = chrono::high_resolution_clock::now();
    chrono::duration<double> timestamp = cur_time - begin_time;
    query_freq[query]++;

    for (int i = 0; i < qo->querySelectColumn.size(); i++) {
      cm->updateColumnFrequency(qo->querySelectColumn[i]);
      cm->updateColumnTimestamp(qo->querySelectColumn[i], timestamp.count());
      cm->updateQueryFrequency(qo->querySelectColumn[i], query_freq[query]);
    }
    for (int i = 0; i < qo->queryBuildColumn.size(); i++) {
      cm->updateColumnFrequency(qo->queryBuildColumn[i]);
      cm->updateColumnTimestamp(qo->queryBuildColumn[i], timestamp.count());
      cm->updateQueryFrequency(qo->queryBuildColumn[i], query_freq[query]);
    }
    for (int i = 0; i < qo->queryProbeColumn.size(); i++) {
      cm->updateColumnFrequency(qo->queryProbeColumn[i]);
      cm->updateColumnTimestamp(qo->queryProbeColumn[i], timestamp.count());
      cm->updateQueryFrequency(qo->queryProbeColumn[i], query_freq[query]);
    }
    for (int i = 0; i < qo->queryGroupByColumn.size(); i++) {
      cm->updateColumnFrequency(qo->queryGroupByColumn[i]);
      cm->updateColumnTimestamp(qo->queryGroupByColumn[i], timestamp.count());
      cm->updateQueryFrequency(qo->queryGroupByColumn[i], query_freq[query]);
    }
    for (int i = 0; i < qo->queryAggrColumn.size(); i++) {
      cm->updateColumnFrequency(qo->queryAggrColumn[i]);
      cm->updateColumnTimestamp(qo->queryAggrColumn[i], timestamp.count());
      cm->updateQueryFrequency(qo->queryAggrColumn[i], query_freq[query]);
    }
  }

  void call_probe_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
    int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4,
    int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
    int h_total, int start_offset, int* offset) {

    probe_CPU(lo_off, dim_off1, dim_off2, dim_off3, dim_off4,
      dimkey_col1, dimkey_col2, dimkey_col3, dimkey_col4,
      ht1, dim_len1, ht2, dim_len2, ht3, dim_len3, ht4, dim_len4,
      min_key1, min_key2, min_key3, min_key4,
      h_lo_off, h_dim_off1, h_dim_off2, h_dim_off3, h_dim_off4,
      h_total, start_offset, offset);
  };

  void call_probe_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* dim_key4,
    int fact_len, int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
    int *total, int start_offset) {

    int tile_items = 128*4;

    probe_GPU<128,4><<<(fact_len + tile_items - 1)/tile_items, 128>>>(dim_key1, dim_key2, dim_key3, dim_key4,
      fact_len, ht1, dim_len1, ht2, dim_len2, ht3, dim_len3, ht4, dim_len4,
      min_key1, min_key2, min_key3, min_key4,
      lo_off, dim_off1, dim_off2, dim_off3, dim_off4, 
      total, start_offset);

  };

  void call_probe_GPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
      int* dimkey_idx1, int* dimkey_idx2, int* dimkey_idx3, int* dimkey_idx4, int* aggr_idx,
      int fact_len, int* out_lo_off, int* out_dim_off1, int* out_dim_off2, int* out_dim_off3, int* out_dim_off4, 
      int *total, int start_offset) {

    int tile_items = 128*4;

    probe_GPU2<128,4><<<(fact_len + tile_items - 1)/tile_items, 128>>>(
      lo_off, dim_off1, dim_off2, dim_off3, dim_off4,
      cm->gpuCache, dimkey_idx1, dimkey_idx2, dimkey_idx3, dimkey_idx4, aggr_idx,
      fact_len, ht1, dim_len1, ht2, dim_len2, ht3, dim_len3, ht4, dim_len4,
      min_key1, min_key2, min_key3, min_key4,
      out_lo_off, out_dim_off1, out_dim_off2, out_dim_off3, out_dim_off4, 
      total,start_offset);
  };

  void call_probe_group_by_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
    int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
    int* res, int fact_len, int start_index) {

    probe_group_by_CPU(lo_off, dim_off1, dim_off2, dim_off3, dim_off4,
      dimkey_col1, dimkey_col2, dimkey_col3, dimkey_col4, aggr_col,
      ht1, dim_len1, ht2, dim_len2, ht3, dim_len3, ht4, dim_len4, res,
      min_val1, unique_val1, min_val2, unique_val2, min_val3, unique_val3, min_val4, unique_val4, 
      total_val, min_key1, min_key2, min_key3, min_key4, fact_len, start_index);

  };

  void call_probe_group_by_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* dim_key4, int* aggr, 
    int fact_len, int* res) {

    int tile_items = 128*4;

    probe_group_by_GPU<128,4><<<(fact_len + tile_items - 1)/tile_items, 128>>>(
      dim_key1, dim_key2, dim_key3, dim_key4, aggr, 
      fact_len, ht1, dim_len1, ht2, dim_len2, ht3, dim_len3, ht4, dim_len4, res,
      min_val1, unique_val1, min_val2, unique_val2, min_val3, unique_val3, min_val4, unique_val4,
      total_val, min_key1, min_key2, min_key3, min_key4);

  };

  void call_probe_group_by_GPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
      int* dimkey_idx1, int* dimkey_idx2, int* dimkey_idx3, int* dimkey_idx4, int* aggr_idx,
      int fact_len, int* res) {

    int tile_items = 128*4;

    probe_group_by_GPU2<128,4><<<(fact_len + tile_items - 1)/tile_items, 128>>>(
      lo_off, dim_off1, dim_off2, dim_off3, dim_off4,
      cm->gpuCache, dimkey_idx1, dimkey_idx2, dimkey_idx3, dimkey_idx4, aggr_idx,
      fact_len, ht1, dim_len1, ht2, dim_len2, ht3, dim_len3, ht4, dim_len4, res,
      min_val1, unique_val1, min_val2, unique_val2, min_val3, unique_val3, min_val4, unique_val4,
      total_val, min_key1, min_key2, min_key3, min_key4);

  };

};

void probe_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int h_total, int start_offset, int* offset) {

  // Probe
  parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    int count = 0;
    assert(h_lo_off != NULL);
    int temp[5][end-start];
    //printf("start = %d end = %d\n", start, end);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot1, slot2, slot3, slot4;
        int lo_offset;
        if (lo_off != NULL) lo_offset = lo_off[start_offset + i];
        else lo_offset = start_offset + i;
        if (dimkey_col1 != NULL) {
          hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
          slot1 = ht1[(hash << 1) + 1];
        } else {
          slot1 = 1;
          if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;
        }
        if (slot1 != 0) {
          if (dimkey_col2 != NULL) {
            hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
            slot2 = ht2[(hash << 1) + 1];
          } else {
            slot2 = 1;
            if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;
          }
          if (slot2 != 0) {
            if (dimkey_col3 != NULL) {
              hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
              slot3 = ht3[(hash << 1) + 1];
            } else {
              slot3 = 1;
              if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;
            }
            if (slot3 != 0) {
              if (dimkey_col4 != NULL) {
                hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
                slot4 = ht4[(hash << 1) + 1];
              } else {
                slot4 = 1;
                if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;
              }
              if (slot4 != 0) {
                temp[0][count] = lo_offset;
                temp[1][count] = slot1-1;
                temp[2][count] = slot2-1;
                temp[3][count] = slot3-1;
                temp[4][count] = slot4-1;
                count++;
              }
            }
          }
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot1, slot2, slot3, slot4;
      int lo_offset;
      if (lo_off != NULL) lo_offset = lo_off[start_offset + i];
      else lo_offset = start_offset + i;
      if (dimkey_col1 != NULL) {
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot1 = ht1[(hash << 1) + 1];
      } else {
        slot1 = 1;
        if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;
      }
      if (slot1 != 0) {
        if (dimkey_col2 != NULL) {
          hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
          slot2 = ht2[(hash << 1) + 1];
        } else {
          slot2 = 1;
          if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;
        }
        if (slot2 != 0) {
          if (dimkey_col3 != NULL) {
            hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
            slot3 = ht3[(hash << 1) + 1];
          } else {
            slot3 = 1;
            if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;
          }
          if (slot3 != 0) {
            if (dimkey_col4 != NULL) {
              hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
              slot4 = ht4[(hash << 1) + 1];
            } else {
              slot4 = 1;
              if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;
            }
            if (slot4 != 0) {
              temp[0][count] = lo_offset;
              temp[1][count] = slot1-1;
              temp[2][count] = slot2-1;
              temp[3][count] = slot3-1;
              temp[4][count] = slot4-1;
              count++;
            }
          }
        }
      }
    }
    //printf("count = %d\n", count);
    int thread_off = __atomic_fetch_add(offset, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      h_lo_off[thread_off+i] = temp[0][i];
      if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  });
}

void probe_group_by_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int fact_len, int start_index) {

  // Probe
  parallel_for(blocked_range<size_t>(0, fact_len, fact_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1, dim_val2, dim_val3, dim_val4;
        int lo_offset;
        if (lo_off != NULL) lo_offset = lo_off[start_index + i];
        else lo_offset = start_index + i;
        if (dim_off1 == NULL && dimkey_col1 != NULL) {
          hash = HASH_WM(dimkey_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          dim_val1 = slot >> 32;
        } else if (dim_off1 != NULL && dimkey_col1 != NULL){
          slot = 1;
          dim_val1 = dimkey_col1[dim_off1[start_index + i]];
        } else {
          slot = 1;
          dim_val1 = 0;
        }
        if (slot != 0) {
          if (dim_off2 == NULL && dimkey_col2 != NULL) {
            hash = HASH_WM(dimkey_col2[lo_offset], dim_len2, min_key2);
            slot = reinterpret_cast<long long*>(ht2)[hash];
            dim_val2 = slot >> 32;
          } else if (dim_off2 != NULL && dimkey_col2 != NULL){
            slot = 1;
            dim_val2 = dimkey_col2[dim_off2[start_index + i]];
          } else {
            slot = 1;
            dim_val2 = 0;
          }
          if (slot != 0) {
            if (dim_off3 == NULL && dimkey_col3 != NULL) {
              hash = HASH_WM(dimkey_col3[lo_offset], dim_len3, min_key3);
              slot = reinterpret_cast<long long*>(ht3)[hash];
              dim_val3 = slot >> 32;
            } else if (dim_off3 != NULL && dimkey_col3 != NULL){
              slot = 1;
              dim_val3 = dimkey_col3[dim_off3[start_index + i]];
            } else {
              slot = 1;
              dim_val3 = 0;
            }
            if (slot != 0) {
              if (dim_off4 == NULL && dimkey_col4 != NULL) {
                hash = HASH_WM(dimkey_col4[lo_offset], dim_len4, min_key4);
                slot = reinterpret_cast<long long*>(ht4)[hash];
                dim_val4 = slot >> 32;
              } else if (dim_off4 != NULL && dimkey_col4 != NULL){
                slot = 1;
                dim_val4 = dimkey_col4[dim_off4[start_index + i]];
              } else {
                slot = 1;
                dim_val4 = 0;
              }
              if (slot != 0) {
                hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
                res[hash * 6] = dim_val1;
                res[hash * 6 + 1] = dim_val2;
                res[hash * 6 + 2] = dim_val3;
                res[hash * 6 + 3] = dim_val4;
                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[lo_offset]), __ATOMIC_RELAXED);
              }
            }
          }
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      int dim_val1, dim_val2, dim_val3, dim_val4;
      int lo_offset;
      if (lo_off != NULL) lo_offset = lo_off[start_index + i];
      else lo_offset = start_index + i;
      if (dim_off1 == NULL && dimkey_col1 != NULL) {
        hash = HASH_WM(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        dim_val1 = slot >> 32;
      } else if (dim_off1 != NULL && dimkey_col1 != NULL){
        slot = 1;
        dim_val1 = dimkey_col1[dim_off1[start_index + i]];
      } else {
        slot = 1;
        dim_val1 = 0;
      }
      if (slot != 0) {
        if (dim_off2 == NULL && dimkey_col2 != NULL) {
          hash = HASH_WM(dimkey_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          dim_val2 = slot >> 32;
        } else if (dim_off2 != NULL && dimkey_col2 != NULL){
          slot = 1;
          dim_val2 = dimkey_col2[dim_off2[start_index + i]];
        } else {
          slot = 1;
          dim_val2 = 0;
        }
        if (slot != 0) {
          if (dim_off3 == NULL && dimkey_col3 != NULL) {
            hash = HASH_WM(dimkey_col3[lo_offset], dim_len3, min_key3);
            slot = reinterpret_cast<long long*>(ht3)[hash];
            dim_val3 = slot >> 32;
          } else if (dim_off3 != NULL && dimkey_col3 != NULL){
            slot = 1;
            dim_val3 = dimkey_col3[dim_off3[start_index + i]];
          } else {
            slot = 1;
            dim_val3 = 0;
          }
          if (slot != 0) {
            if (dim_off4 == NULL && dimkey_col4 != NULL) {
              hash = HASH_WM(dimkey_col4[lo_offset], dim_len4, min_key4);
              slot = reinterpret_cast<long long*>(ht4)[hash];
              dim_val4 = slot >> 32;
            } else if (dim_off4 != NULL && dimkey_col4 != NULL){
              slot = 1;
              dim_val4 = dimkey_col4[dim_off4[start_index + i]];
            } else {
              slot = 1;
              dim_val4 = 0;
            }
            if (slot != 0) {
              hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
              res[hash * 6] = dim_val1;
              res[hash * 6 + 1] = dim_val2;
              res[hash * 6 + 2] = dim_val3;
              res[hash * 6 + 3] = dim_val4;
              __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[lo_offset]), __ATOMIC_RELAXED);
            }
          }
        }
      }
    }
  });

}

void build_CPU(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int isoffset) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      int key = dim_key[i];
      int hash = HASH_WM(key, num_slots, val_min);
      hash_table[hash << 1] = key;
      if (isoffset == 1) hash_table[(hash << 1) + 1] = i + 1;
      else if (isoffset == 0) hash_table[(hash << 1) + 1] = dim_val[i];
      else hash_table[(hash << 1) + 1] = 0;
    }
  });
}

void build_filter_CPU(int *filter_col, int compare, int *dim_key, int* dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int isoffset) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == compare) {
        int key = dim_key[i];
        int hash = HASH(key, num_slots, val_min);
        hash_table[hash << 1] = key;
        if (isoffset == 1) hash_table[(hash << 1) + 1] = i + 1;
        else if (isoffset == 0) hash_table[(hash << 1) + 1] = dim_val[i];
        else hash_table[(hash << 1) + 1] = 0;
      }
    }
  });
}

void runAggregationQ2CPU(int* lo_revenue, int* p_brand1, int* d_year, int* lo_off, int* part_off, int* date_off, int num_tuples, int* res, int num_slots) {
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int brand = p_brand1[part_off[i]];
        int year = d_year[date_off[i]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[lo_off[i]]), __ATOMIC_RELAXED);
      }
    }
    for (int i = end_batch ; i < end; i++) {
        int brand = p_brand1[part_off[i]];
        int year = d_year[date_off[i]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[lo_off[i]]), __ATOMIC_RELAXED);
    }
  });
}

#endif

// void probe_group_by_CPU(int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
//   int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
//   int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
//   int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int start_index) {

//   // Probe
//   parallel_for(blocked_range<size_t>(0, fact_len, fact_len/NUM_THREADS + 4), [&](auto range) {
//     int start = range.begin();
//     int end = range.end();
//     int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

//     for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
//       #pragma simd
//       for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
//         int hash;
//         long long slot;
//         int dim_val1, dim_val2, dim_val3, dim_val4;
//         if (dimkey_col1 != NULL) {
//           hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
//           slot = reinterpret_cast<long long*>(ht1)[hash];
//           dim_val1 = slot >> 32;
//         } else {
//           slot = 1;
//           dim_val1 = 0;
//         }
//         if (slot != 0) {
//           if (dimkey_col2 != NULL) {
//             hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
//             slot = reinterpret_cast<long long*>(ht2)[hash];
//             dim_val2 = slot >> 32;
//           } else {
//             slot = 1;
//             dim_val2 = 0;
//           }
//           if (slot != 0) {
//             if (dimkey_col3 != NULL) {
//               hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
//               slot = reinterpret_cast<long long*>(ht3)[hash];
//               dim_val3 = slot >> 32;
//             } else {
//               slot = 1;
//               dim_val3 = 0;
//             }
//             if (slot != 0) {
//               if (dimkey_col4 != NULL) {
//                 hash = HASH_WM(dimkey_col4[start_index + i], dim_len4, min_key4);
//                 slot = reinterpret_cast<long long*>(ht4)[hash];
//                 dim_val4 = slot >> 32;
//               } else {
//                 slot = 1;
//                 dim_val4 = 0;
//               }
//               if (slot != 0) {
//                 hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
//                 res[hash * 6] = dim_val1;
//                 res[hash * 6 + 1] = dim_val2;
//                 res[hash * 6 + 2] = dim_val3;
//                 res[hash * 6 + 3] = dim_val4;
//                 __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
//               }
//             }
//           }
//         }
//       }
//     }

//     for (int i = end_batch ; i < end; i++) {
//       int hash;
//       long long slot;
//       int dim_val1, dim_val2, dim_val3, dim_val4;
//       if (dimkey_col1 != NULL) {
//         hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
//         slot = reinterpret_cast<long long*>(ht1)[hash];
//         dim_val1 = slot >> 32;
//       } else {
//         slot = 1;
//         dim_val1 = 0;
//       }
//       if (slot != 0) {
//         if (dimkey_col2 != NULL) {
//           hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
//           slot = reinterpret_cast<long long*>(ht2)[hash];
//           dim_val2 = slot >> 32;
//         } else {
//           slot = 1;
//           dim_val2 = 0;
//         }
//         if (slot != 0) {
//           if (dimkey_col3 != NULL) {
//             hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
//             slot = reinterpret_cast<long long*>(ht3)[hash];
//             dim_val3 = slot >> 32;
//           } else {
//             slot = 1;
//             dim_val3 = 0;
//           }
//           if (slot != 0) {
//             if (dimkey_col4 != NULL) {
//               hash = HASH_WM(dimkey_col4[start_index + i], dim_len4, min_key4);
//               slot = reinterpret_cast<long long*>(ht4)[hash];
//               dim_val4 = slot >> 32;
//             } else {
//               slot = 1;
//               dim_val4 = 0;
//             }
//             if (slot != 0) {
//               hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
//               res[hash * 6] = dim_val1;
//               res[hash * 6 + 1] = dim_val2;
//               res[hash * 6 + 2] = dim_val3;
//               res[hash * 6 + 3] = dim_val4;
//               __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
//             }
//           }
//         }
//       }
//     }
//   });
// }