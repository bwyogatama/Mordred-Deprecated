#ifndef _QUERY_PROCESSING_H_
#define _QUERY_PROCESSING_H_

#include "QueryOptimizer.h"
#include "GPUProcessing4.h"
#include "CPUProcessing2.h"
// #include "tbb/tbb.h"

// #include <chrono>
// #include <atomic>
// #include <unistd.h>
// #include <iostream>
// #include <stdio.h>

#define NUM_QUERIES 4

// using namespace std;
// using namespace tbb;

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

class QueryProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;

  unordered_map<ColumnInfo*, int> min_key;
  unordered_map<ColumnInfo*, int> min_val;
  unordered_map<ColumnInfo*, int> unique_val;
  unordered_map<ColumnInfo*, int> dim_len;

  int total_val;

  unordered_map<ColumnInfo*, int*> ht_CPU;
  unordered_map<ColumnInfo*, int*> ht_GPU;
  unordered_map<ColumnInfo*, int*> col_idx;

  int *ht_p, *ht_c, *ht_s, *ht_d;
  int *d_ht_p, *d_ht_c, *d_ht_s, *d_ht_d;

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

  void runQuery();

  void prepareQuery(int query);

  void endQuery(int query);

  void updateStatsQuery(int query);

  void processQuery() {

    int query = generate_rand_query();
    qo->parseQuery(query);
    prepareQuery(query);
    updateStatsQuery(query);
    runQuery(query);
    endQuery(query);
  };

  void switch_device(int mode, int sg);

  void call_probe_CPU(int** h_off_col, int h_total, int sg);

  void call_probe_GPU(int** off_col, int d_total, int sg);

  void call_build_GPU(int dim_select);

  void call_build_CPU();

  void call_filter_GPU(int sg);

  void call_filter_CPU(int sg);
};

void
QueryProcessing::endQuery(int query) {
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
  free(h_dimkey_col);
  dimkey_col.clear();

  qo->clearVector();
}

void
QueryProcessing::updateStatsQuery(int query) {
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

void 
QueryProcessing::prepareQuery(int query) {
  if (query == 0) {

    min_key[cm->p_partkey] = 0;
    min_key[cm->c_custkey] = 0;
    min_key[cm->s_suppkey] = 0;
    min_key[cm->d_datekey] = 19920101;

    min_val[cm->p_partkey] = 0;
    min_val[cm->c_custkey] = 0;
    min_val[cm->s_suppkey] = 0;
    min_val[cm->d_datekey] = 1992;

    unique_val[cm->p_partkey] = 0;
    unique_val[cm->c_custkey] = 0;
    unique_val[cm->s_suppkey] = 0;
    unique_val[cm->d_datekey] = 1;

    dim_len[cm->p_partkey] = 0;
    dim_len[cm->c_custkey] = 0;
    dim_len[cm->s_suppkey] = 0;
    dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    total_val = (1998-1992+1);

    ht_CPU[cm->p_partkey] = NULL;
    ht_CPU[cm->c_custkey] = NULL;
    ht_CPU[cm->s_suppkey] = NULL;
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));

    g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int));
    d_ht_p = NULL;
    d_ht_c = NULL;
    d_ht_s = NULL;

    cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));

    ht_GPU[cm->p_partkey] = d_ht_p;
    ht_GPU[cm->c_custkey] = d_ht_c;
    ht_GPU[cm->s_suppkey] = d_ht_s;
    ht_GPU[cm->d_datekey] = d_ht_d;

    int res_array_size = total_val * 6;
    res = new int[res_array_size];
    memset(res, 0, res_array_size * sizeof(int));
       
    g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
    cudaMemset(d_res, 0, res_array_size * sizeof(int));

  } else if (query == 1) {

    min_key[cm->p_partkey] = 0;
    min_key[cm->c_custkey] = 0;
    min_key[cm->s_suppkey] = 0;
    min_key[cm->d_datekey] = 19920101;

    min_val[cm->p_partkey] = 0;
    min_val[cm->c_custkey] = 0;
    min_val[cm->s_suppkey] = 0;
    min_val[cm->d_datekey] = 1992;

    unique_val[cm->p_partkey] = 7;
    unique_val[cm->c_custkey] = 0;
    unique_val[cm->s_suppkey] = 0;
    unique_val[cm->d_datekey] = 1;

    dim_len[cm->p_partkey] = P_LEN;
    dim_len[cm->c_custkey] = 0;
    dim_len[cm->s_suppkey] = S_LEN;
    dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    total_val = ((1998-1992+1) * (5 * 5 * 40));

    ht_CPU[cm->p_partkey] = (int*)malloc(2 * dim_len[cm->p_partkey] * sizeof(int));
    ht_CPU[cm->c_custkey] = NULL;
    ht_CPU[cm->s_suppkey] = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_CPU[cm->p_partkey], 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    memset(ht_CPU[cm->s_suppkey], 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));

    g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * dim_len[cm->p_partkey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int));

    cudaMemset(d_ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));

    ht_GPU[cm->p_partkey] = d_ht_p;
    ht_GPU[cm->c_custkey] = d_ht_c;
    ht_GPU[cm->s_suppkey] = d_ht_s;
    ht_GPU[cm->d_datekey] = d_ht_d;

    int res_array_size = total_val * 6;
    res = new int[res_array_size];
    memset(res, 0, res_array_size * sizeof(int));
       
    g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
    cudaMemset(d_res, 0, res_array_size * sizeof(int));

  } else if (query == 2) {

    min_key[cm->p_partkey] = 0;
    min_key[cm->c_custkey] = 0;
    min_key[cm->s_suppkey] = 0;
    min_key[cm->d_datekey] = 19920101;

    min_val[cm->p_partkey] = 0;
    min_val[cm->c_custkey] = 0;
    min_val[cm->s_suppkey] = 0;
    min_val[cm->d_datekey] = 1992;

    unique_val[cm->p_partkey] = 0;
    unique_val[cm->c_custkey] = 7;
    unique_val[cm->s_suppkey] = 25 * 7;
    unique_val[cm->d_datekey] = 1;

    dim_len[cm->p_partkey] = 0;
    dim_len[cm->c_custkey] = C_LEN;
    dim_len[cm->s_suppkey] = S_LEN;
    dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    total_val = ((1998-1992+1) * 25 * 25);

    ht_CPU[cm->p_partkey] = NULL;
    ht_CPU[cm->c_custkey] = (int*)malloc(2 * dim_len[cm->c_custkey] * sizeof(int));
    ht_CPU[cm->s_suppkey] = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_CPU[cm->c_custkey], 0, 2 * dim_len[cm->c_custkey] * sizeof(int));
    memset(ht_CPU[cm->s_suppkey], 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));

    g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * dim_len[cm->c_custkey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int));

    cudaMemset(d_ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int));
    cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));

    ht_GPU[cm->p_partkey] = d_ht_p;
    ht_GPU[cm->c_custkey] = d_ht_c;
    ht_GPU[cm->s_suppkey] = d_ht_s;
    ht_GPU[cm->d_datekey] = d_ht_d;

    int res_array_size = total_val * 6;
    res = new int[res_array_size];
    memset(res, 0, res_array_size * sizeof(int));
       
    g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
    cudaMemset(d_res, 0, res_array_size * sizeof(int));

  } else {

    min_key[cm->p_partkey] = 0;
    min_key[cm->c_custkey] = 0;
    min_key[cm->s_suppkey] = 0;
    min_key[cm->d_datekey] = 19920101;

    min_val[cm->p_partkey] = 0;
    min_val[cm->c_custkey] = 0;
    min_val[cm->s_suppkey] = 0;
    min_val[cm->d_datekey] = 1992;

    unique_val[cm->p_partkey] = 0;
    unique_val[cm->c_custkey] = 7;
    unique_val[cm->s_suppkey] = 0;
    unique_val[cm->d_datekey] = 1;

    dim_len[cm->p_partkey] = P_LEN;
    dim_len[cm->c_custkey] = C_LEN;
    dim_len[cm->s_suppkey] = S_LEN;
    dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    total_val = ((1998-1992+1) * 25);

    ht_CPU[cm->p_partkey] = (int*)malloc(2 * dim_len[cm->p_partkey] * sizeof(int));
    ht_CPU[cm->c_custkey] = (int*)malloc(2 * dim_len[cm->c_custkey] * sizeof(int));
    ht_CPU[cm->s_suppkey] = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_CPU[cm->p_partkey], 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    memset(ht_CPU[cm->s_suppkey], 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    memset(ht_CPU[cm->c_custkey], 0, 2 * dim_len[cm->c_custkey] * sizeof(int));

    g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * dim_len[cm->p_partkey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int));
    g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * dim_len[cm->c_custkey] * sizeof(int));

    cudaMemset(d_ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    cudaMemset(d_ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int));

    ht_GPU[cm->p_partkey] = d_ht_p;
    ht_GPU[cm->c_custkey] = d_ht_c;
    ht_GPU[cm->s_suppkey] = d_ht_s;
    ht_GPU[cm->d_datekey] = d_ht_d;

    int res_array_size = total_val * 6;
    res = new int[res_array_size];
    memset(res, 0, res_array_size * sizeof(int));
       
    g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
    cudaMemset(d_res, 0, res_array_size * sizeof(int));
  }
}

void 
QueryProcessing::switch_device(int** off_col, int** h_off_col, int* d_total, int h_total, int sg, int mode) {
  if (mode == 0) { //CPU to GPU
    assert(h_off_col != NULL);
    assert(h_total > 0);

    off_col = (int**) malloc(5 * sizeof(int*));
    cudaMalloc((void**) &d_off_col, 5 * SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int));
    for (int i = 0; i < 5; i++) {
      off_col[i] = d_off_col + i * SEGMENT_SIZE * qo->segment_group_count[0][sg];
    }

    cudaMemcpy(d_total, &h_total, sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < 5; i++) {
      if (h_off_col[i] != NULL)
        cudaMemcpy(off_col[i], h_off_col[i], h_total * sizeof(int), cudaMemcpyHostToDevice);
      else
        off_col[i] = NULL;
    }
  } else { // GPU to CPU
    assert(off_col != NULL);

    h_off_col = (int**) malloc(5 * sizeof(int*));
    for (int i = 0; i < 5; i++) {
      h_off_col[i] = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int));
    }

    cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++) {
      if (off_col[i] != NULL)
        cudaMemcpy(h_off_col[i], off_col[i], h_total * sizeof(int), cudaMemcpyDeviceToHost);
      else
        h_off_col[i] = NULL;
    }
  }
  
}

void 
QueryProcessing::call_probe_GPU(int** off_col, int* d_total, int h_total, int sg, int join_so_far) {
  int **off_col_out;
  int *d_off_col_out, *out_total;
  int min_key[4], dim_len[4];
  int *ht[4], *dimkey_idx[4], *key_col[4];
  vector<ColumnInfo*> dimkey_col (4, NULL);
  int start_offset, idx_key;

  int tile_items = 128*4;

  off_col_out = (int**) malloc(5 * sizeof(int*));
  cudaMalloc((void**) &d_off_col_out, 5 * SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int));

  cudaMalloc((void **)&out_total, sizeof(int));
  cudaMemset(out_total, 0, sizeof(int));

  off_col_out[0] = d_off_col_out;
  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    assert(join_so_far + i < qo->join.size() - 1);
    assert(joinGPUPipelineCol[sg][i] != NULL);
    dimkey_col[join_so_far + i] = qo->joinGPUPipelineCol[sg][i];
    off_col_out[join_so_far + i + 1] = d_off_col_out + (join_so_far + i + 1) * SEGMENT_SIZE * qo->segment_group_count[0][sg];
  }

  for (int i = 0; i < 4; i++) {
    ColumnInfo* column = dimkey_col[i];
    if (column == NULL) {
      dimkey_idx[i] = NULL;
      ht[i] = NULL;
      min_key[i] = 0;
      dim_len[i] = 0;
    } else {
      if (col_idx[column] == NULL)
        cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice);
      assert(col_idx[column] != NULL);
      assert(ht_GPU[column] != NULL);
      dimkey_idx[i] = col_idx[column];
      ht[i] = ht_GPU[column];
      min_key[i] = min_key[column];
      dim_len[i] = dim_len[column];
    }
  }

  if (off_col == NULL) {

    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;

      for (int j = 0; j < 4; j++) {
        ColumnInfo* column = dimkey_col[j];
        if (column == NULL) key_col[j] = NULL;
        else {
          idx_key = cm->segment_list[column->column_id][segment_number];
          assert(idx_key >= 0);
          key_col[j] = cm->gpuCache + idx_key * SEGMENT_SIZE;            
        }
      }

      probe_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>(
        key_col[0], key_col[1], key_col[2], key_col[3], SEGMENT_SIZE, 
        ht[0], dim_len[0], ht[1], dim_len[1], ht[2], dim_len[2], ht[3], dim_len[3],
        min_key[0], min_key[1], min_key[2], min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        out_total, start_offset);

    }

  } else {

    probe_GPU2<128,4><<<(h_total + tile_items - 1)/tile_items, 128>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4],
      cm->gpuCache, dimkey_idx[0], dimkey_idx[1], dimkey_idx[2], dimkey_idx[3], h_total, 
      ht[0], dim_len[0], ht[1], dim_len[1], ht[2], dim_len[2], ht[3], dim_len[3],
      min_key[0], min_key[1], min_key[2], min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      out_total, 0, NULL);

  }

  if (off_col != NULL) {
    cudaFree(d_off_col);
    free(off_col);
  }

  d_off_col = d_off_col_out;
  off_col = off_col_out;
  d_total = out_total;

};

void 
QueryProcessing::call_probe_CPU(int** h_off_col, int* h_total, int sg, int join_so_far) {
  int **off_col_out;
  int min_key[4], dim_len[4];
  int *ht[4], *dimkey_idx[4], *key_col[4];
  int start_offset, out_total;
  vector<ColumnInfo*> dimkey_col (4, NULL);

  off_col_out = (int**) malloc(5 * sizeof(int*));
  off_col_out[0] = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)); //lo_off
  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    assert(join_so_far + i < qo->join.size() - 1);
    assert(joinCPUPipelineCol[sg][i] != NULL);
    dimkey_col[join_so_far + i] = qo->joinCPUPipelineCol[sg][i];
    key_col[join_so_far + i] = dimkey_col[join_so_far + i]->col_ptr;
    off_col_out[join_so_far + i + 1] = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)); //dim_off
  }

  for (int i = 0; i < 4; i++) {
    ColumnInfo* column = dimkey_col[i];
    if (column == NULL) {
      ht[i] = NULL;
      min_key[i] = 0;
      dim_len[i] = 0;
    } else {
      assert(ht_CPU[column] != NULL);;
      ht[i] = ht_CPU[column];
      min_key[i] = min_key[column];
      dim_len[i] = dim_len[column];
    }
  }

  out_total = 0;

  if (h_off_col == NULL) {
    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;

      probe_CPU(NULL, NULL, NULL, NULL, NULL,
        key_col[0], key_col[1], key_col[2], key_col[3], SEGMENT_SIZE,
        ht[0], dim_len[0], ht[1], dim_len[1], ht[2], dim_len[2], ht[3], dim_len[3],
        min_key[0], min_key[1], min_key[2], min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        NULL, start_offset, &out_total);

    }
  } else {

      probe_CPU(h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4],
        key_col[0], key_col[1], key_col[2], key_col[3], *h_total,
        ht[0], dim_len[0], ht[1], dim_len[1], ht[2], dim_len[2], ht[3], dim_len[3],
        min_key[0], min_key[1], min_key[2], min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        NULL, start_offset, &out_total);
  }

  h_off_col = off_col_out;
  *h_total = out_total;
};

void 
QueryProcessing::call_build_GPU(int** off_col, int* h_total, int dim_table) {
  int tile_items = 128*4;
  int* dimkey_idx;

  ColumnInfo* column = qo->join[dim_table].second;
  if (ht_GPU[column] != NULL) {

    if (col_idx[column] == NULL)
      cudaMemcpy(dimkey_idx, cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice);

    build_GPU2<128,4><<<(total[dim_table] + tile_items - 1)/tile_items, 128>>>(
      off_col[dim_table], cm->gpuCache, dimkey_idx, NULL, h_total[dim_table],
      ht_GPU[column], dim_len[column], min_key[column], 0, 1);

    cudaFree(off_col[dim_table]);
  }
};

void 
QueryProcessing::call_build_CPU(int** h_off_col, int* h_total, int dim_table) {

  ColumnInfo* column = qo->join[dim_table].second;
  if (ht_CPU[column] != NULL) {

    build_CPU(h_off_col[dim_table], column->col_ptr, NULL, h_total[dim_table], 
       ht_CPU[column], dim_len[column], min_key[column], 1);

    free(h_off_col[dim_table]);
  }
};

void
QueryProcessing::call_build_filter_GPU(int* off_col, int* d_total, int sg, int dim_table) {
  int tile_items = 128*4;
  int* off_col_out, *out_total;
  cudaMalloc((void**) &off_col_out, qo->segment_group_count[dim_table][sg] * SEGMENT_SIZE * sizeof(int));

  ColumnInfo* column = qo->select_build[join[dim_table].second][0];

  cudaMalloc((void **)&out_total, sizeof(int));
  cudaMemset(out_total, 0, sizeof(int));

  for (int i = 0; i < segment_group_count[dim_table][sg]; i++){

    int segment_number = qo->segment_group[dim_table][sg * column->total_segment + i];
    int start_offset = SEGMENT_SIZE * segment_number;
    int idx_key = cm->segment_list[column->column_id][segment_number];
    int* filter_col = cm->gpuCache + idx_key * SEGMENT_SIZE;

    filter_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>
      (filter_col, NULL, compare1, compare2, compare3, compare4, off_col_out, 
         out_total, start_offset, SEGMENT_SIZE, mode1, mode2);
  }

  off_col = off_col_out;
  d_total = out_total;
}

void
QueryProcessing::call_build_filter_CPU(int* h_off_col, int* h_total, int sg, int dim_table) {
  int* off_col_out = (int**) malloc(qo->segment_group_count[dim_table][sg] * SEGMENT_SIZE * sizeof(int*));

  int start_offset, out_total = 0;
  ColumnInfo* column = qo->select_build[join[dim_table].second][0];
  int* filter_col = column->col_ptr;

  for (int i = 0; i < segment_group_count[dim_table][sg]; i++){
    int segment_number = qo->segment_group[dim_table][sg * column->total_segment + i];
    int start_offset = SEGMENT_SIZE * segment_number;

    filter_CPU(NULL, filter_col, NULL, compare1, compare2, compare3, compare4, off_col_out, 
         &out_total, start_offset, SEGMENT_SIZE, mode1, mode2);
  }

  h_off_col = off_col_out;
  *h_total = out_total;
}

void
QueryProcessing::call_probe_filter_GPU(int* off_col, int* d_total, int h_total, int sg, int select_so_far) {
  int tile_items = 128*4;
  int *off_col_out, *out_total, *filter_idx;
  int *filter_col[2];
  vector<ColumnInfo*> filter (2, NULL);
  int start_offset, idx_fil;

  cudaMalloc((void**) &off_col_out, SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int));
  cudaMalloc((void **)&out_total, sizeof(int));
  cudaMemset(out_total, 0, sizeof(int));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    filter[select_so_far + i] = qo->selectGPUPipelineCol[sg][i];
  }

  if (off_col == NULL) {
    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;

      for (int j = 0; j < 2; j++) {
        if (filter[j] == NULL) filter_col[j] = NULL;
        else {
          idx_fil = cm->segment_list[filter[j]->column_id][segment_number];
          assert(idx_fil >= 0);
          filter_col[j] = cm->gpuCache + idx_fil * SEGMENT_SIZE;
        }
      }

      filter_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>
        (filter_col[0], filter_col[1], compare1, compare2, compare3, compare4, off_col_out, 
           out_total, start_offset, SEGMENT_SIZE, mode1, mode2);
    }
  } else {

    if (col_idx[filter[1]] == NULL)
      cudaMemcpy(col_idx[filter[1]], cm->segment_list[filter[1]->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice);
    filter_idx = col_idx[filter[1]];

    filter_GPU2<128,4><<<(h_total + tile_items - 1)/tile_items, 128>>>
        (off_col, filter_idx, compare1, compare2, off_col_out, 
           out_total, start_offset, h_total, mode2);
  }

  if (off_col != NULL) free(off_col);
  off_col = off_col_out;
  d_total = out_total;
}

void
QueryProcessing::call_probe_filter_CPU(int* h_off_col, int* h_total, int sg, int select_so_far) {
  int *off_col_out;
  int *filter_col[2];
  int start_offset;
  int out_total = 0;

  off_col_out = (int*) malloc (SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int));

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    filter_col[select_so_far + i] = qo->selectCPUPipelineCol[sg][i]->col_ptr;
  }

  if (h_off_col == NULL) {
    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;
      filter_CPU(NULL, filter_col[0], filter_col[1], compare1, compare2, compare3, compare4, off_col_out, 
           &out_total, start_offset, SEGMENT_SIZE, mode1, mode2);
    }
  } else {
    assert(filter_col[0] == NULL);
    assert(filter_col[1] != NULL);
    filter_CPU(h_off_col, filter_col[0], filter_col[1], compare1, compare2, compare3, compare4, off_col_out, 
         &out_total, start_offset, *h_total, mode1, mode2);
  }

  if (h_off_col != NULL) free(h_off_col);
  h_off_col = off_col_out;
  *h_total = out_total;
}


void
QueryProcessing::runQuery() {

  for (int sg = 0; sg < 64; sg++) {
    if (qo->segment_group_count[sg] > 0) { 

      call_build_GPU(off_col, d_total, sg);

      call_build_CPU(h_off_col, h_total, sg);

    }
  }

  for (int sg = 0; sg < 64; sg++) {

    int** h_off_col, **off_col;
    int*d_off_col, *d_total;
    int h_total;

    if (qo->segment_group_count[0][sg] > 0) { 

      call_probe_GPU(off_col, d_total, sg);

      switch_device(off_col, h_off_col, d_total, h_total, sg, 1);

      call_probe_CPU(h_off_col, h_total, sg);

    }
  }

}
#endif