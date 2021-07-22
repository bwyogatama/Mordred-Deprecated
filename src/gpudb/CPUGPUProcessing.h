#ifndef _CPUGPU_PROCESSING_H_
#define _CPUGPU_PROCESSING_H_

#include "QueryOptimizer.h"
#include "GPUProcessing2.h"
#include "CPUProcessing.h"

class QueryParams{
public:

  int query;
  
  unordered_map<ColumnInfo*, int> min_key;
  unordered_map<ColumnInfo*, int> min_val;
  unordered_map<ColumnInfo*, int> unique_val;
  unordered_map<ColumnInfo*, int> dim_len;

  unordered_map<ColumnInfo*, int*> ht_CPU;
  unordered_map<ColumnInfo*, int*> ht_GPU;

  unordered_map<ColumnInfo*, int> compare1;
  unordered_map<ColumnInfo*, int> compare2;
  unordered_map<ColumnInfo*, int> mode;

  unordered_map<ColumnInfo*, float> selectivity;

  int total_val, mode_group;

  int *ht_p, *ht_c, *ht_s, *ht_d;
  int *d_ht_p, *d_ht_c, *d_ht_s, *d_ht_d;

  int* res;
  int* d_res;

  QueryParams(int _query): query(_query) {};
};


class CPUGPUProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;

  int** col_idx;
  chrono::high_resolution_clock::time_point begin_time;

  CPUGPUProcessing(size_t cache_size, size_t _processing_size, size_t _pinned_memsize) {
    qo = new QueryOptimizer(cache_size, _processing_size, _pinned_memsize);
    cm = qo->cm;
    begin_time = chrono::high_resolution_clock::now();
    col_idx = new int*[cm->TOT_COLUMN]();
  }

  void resetCGP() {
    for (int i = 0; i < cm->TOT_COLUMN; i++) {
      col_idx[i] = NULL;
    }
  }

  void switch_device_fact(int** &off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table, cudaStream_t stream);

  void call_pfilter_probe_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

  void call_pfilter_probe_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

  void call_probe_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_probe_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_probe_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, cudaStream_t stream);

  void call_probe_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_pfilter_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);



  void switch_device_dim(int* &off_col, int* &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table, cudaStream_t stream);

  void call_bfilter_build_GPU(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream);

  void call_bfilter_build_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table);

  void call_build_GPU(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream);

  void call_build_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table);

  void call_bfilter_GPU(QueryParams* params, int* &d_off_col, int* &d_total, int* h_total, int sg, int table, cudaStream_t stream);

  void call_bfilter_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table);



  void call_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, cudaStream_t stream);

  void call_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total);

};

void 
CPUGPUProcessing::switch_device_fact(int** &off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table, cudaStream_t stream) {
  cudaEvent_t start, stop; 
  float time;
  cout << "Transfer size: " << *h_total << " sg: " << sg << endl;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 
  cudaEventRecord(start, 0);
  if (mode == 0) { //CPU to GPU
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    assert(*h_total > 0);

    off_col = new int*[cm->TOT_TABLE]();
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i]) 
        off_col[i] = cm->customCudaMalloc(*h_total);
    }
    
    CubDebugExit(cudaMemcpyAsync(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice, stream));
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL && (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])) {
        CubDebugExit(cudaMemcpyAsync(off_col[i], h_off_col[i], *h_total * sizeof(int), cudaMemcpyHostToDevice, stream));
      } else
        off_col[i] = NULL;
    }
    cudaStreamSynchronize(stream);
  } else { // GPU to CPU
    if (off_col == NULL) return;
    assert(off_col != NULL);

    h_off_col = new int*[cm->TOT_TABLE]();
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i]) 
        h_off_col[i] = cm->customCudaHostAlloc(*h_total);
    }

    CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL && (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])) {
        CubDebugExit(cudaMemcpyAsync(h_off_col[i], off_col[i], *h_total * sizeof(int), cudaMemcpyDeviceToHost, stream));
      } else
        h_off_col[i] = NULL;
    }
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "Transfer Time: " << time << " sg: " << sg << endl;
  
}

void 
CPUGPUProcessing::switch_device_dim(int* &d_off_col, int* &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table, cudaStream_t stream) {

  if (mode == 0) { //CPU to GPU
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    assert(*h_total > 0);

    d_off_col = cm->customCudaMalloc(*h_total);

    CubDebugExit(cudaMemcpyAsync(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice, stream));

    if (h_off_col != NULL) {
      CubDebugExit(cudaMemcpyAsync(d_off_col, h_off_col, *h_total * sizeof(int), cudaMemcpyHostToDevice, stream));
    } else
      d_off_col = NULL;

    cudaStreamSynchronize(stream);
  } else { // GPU to CPU
    if (d_off_col == NULL) return;
    assert(d_off_col != NULL);

    h_off_col = cm->customCudaHostAlloc(*h_total);

    CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));

    if (d_off_col != NULL) {
      CubDebugExit(cudaMemcpyAsync(h_off_col, d_off_col, *h_total * sizeof(int), cudaMemcpyDeviceToHost, stream));
    } else
      h_off_col = NULL;

    cudaStreamSynchronize(stream);
  }
  
}

void
CPUGPUProcessing::call_pfilter_probe_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream) {

  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  int *filter_idx[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_idx[2] = {}, *group_idx[4] = {};

  int tile_items = 128*4;

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    if (col_idx[column->column_id] == NULL) { //THIS IS A CRITICAL SECTION
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    filter_idx[select_so_far + i] = col_idx[column->column_id];
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
  }

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    aggr_idx[i] = col_idx[column->column_id];
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      if (col_idx[column->column_id] == NULL) {
        int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
        CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      }
      group_idx[column_key->table_id - 1] = col_idx[column->column_id];
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;                // Variable that will hold the time
  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  if (off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

    // filter_probe_group_by_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   NULL, NULL, NULL, NULL, NULL,
    //   cm->gpuCache, filter_idx[0], filter_idx[1], 
    //   _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    //   fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    //   aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3], params->mode_group,
    //   LEN , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    //   _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    //   params->total_val, params->d_res, 0, d_segment_group);

    filter_probe_group_by_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      aggr_idx[0], aggr_idx[1], params->mode_group,
      LEN , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->d_res, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

  } else {

    // filter_probe_group_by_GPU2<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   off_col[0], off_col[1], off_col[2], off_col[3], off_col[4],
    //   cm->gpuCache, filter_idx[0], filter_idx[1], 
    //   _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    //   fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    //   aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3], params->mode_group,
    //   *h_total , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    //   _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    //   params->total_val, params->d_res, 0, NULL);

    filter_probe_group_by_GPU3<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4],
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3], params->mode_group,
      *h_total , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->d_res);

    CHECK_ERROR_STREAM(stream);

  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  cout << "Kernel time: " << time << endl;

};

void
CPUGPUProcessing::call_pfilter_probe_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far) {

  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int *filter_col[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_col[2] = {}, *group_col[4] = {};

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    filter_col[select_so_far + i] = column->col_ptr;
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
  }

  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinCPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    fkey_col[table_id - 1] = column->col_ptr;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    ht[table_id - 1] = params->ht_CPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    aggr_col[i] = column->col_ptr;
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      group_col[column_key->table_id - 1] = column->col_ptr;
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;                // Variable that will hold the time
  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  if (h_off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    filter_probe_group_by_CPU(
      NULL, NULL, NULL, NULL, NULL,
      filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], 
      aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3], params->mode_group,
      LEN , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->res, 0, segment_group_ptr);
  } else {

    filter_probe_group_by_CPU(
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4],
      filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], 
      aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3], params->mode_group,
      *h_total , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->res, 0, NULL);
  }

  // parallel_for(blocked_range<size_t>(0, LEN, LEN/4096 + 4), [&](auto range) {
  //   int start = range.begin();
  //   int end = range.end();
  //   int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
  //   for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
  //     #pragma simd
  //     for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
  //       int hash = HASH(cm->h_lo_suppkey[i], S_LEN, 0);
  //       int slot = ht_s[hash << 1];
  //       if (slot != 0) {
  //         hash = HASH(cm->h_lo_partkey[i], P_LEN, 0);
  //         slot = ht_p[hash << 1];
  //         if (slot != 0) {
  //           int brand = ht_p[(hash << 1) + 1];
  //           brand = cm->h_p_brand1[brand - 1];
  //           hash = HASH(cm->h_lo_orderdate[i], params->dim_len[cm->d_datekey], params->min_key[cm->d_datekey]);
  //           int year = ht_d[(hash << 1) + 1];
  //           year = cm->h_d_year[year - 1];
  //           hash = (brand * 7 +  (year - 1992)) % params->total_val;
  //           params->res[hash * 6] = year;
  //           params->res[hash * 6 + 1] = brand;
  //           __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&params->res[hash * 6 + 4]), (long long)(cm->h_lo_revenue[i]), __ATOMIC_RELAXED);
  //         }
  //       }
  //       // int hash, slot, off;
  //       // hash = HASH(cm->h_lo_partkey[i], P_LEN, 0);
  //       // long long p_slot = reinterpret_cast<long long*>(ht_p)[hash];
  //       // if (p_slot  != 0) {
  //       //   int brand = p_slot >> 32;
  //       //   brand = cm->h_p_brand1[brand - 1];
  //       //   hash = HASH(cm->h_lo_suppkey[i], S_LEN, 0);
  //       //   slot = ht_s[hash << 1];
  //       //   if (slot != 0) {
  //       //     hash = HASH(cm->h_lo_orderdate[i], params->dim_len[cm->d_datekey], params->min_key[cm->d_datekey]);
  //       //     int year = ht_d[(hash << 1) + 1];
  //       //     year = cm->h_d_year[year - 1];
  //       //     hash = (brand * 7 +  (year - 1992)) % params->total_val;
  //       //     params->res[hash * 6] = year;
  //       //     params->res[hash * 6 + 1] = brand;
  //       //     __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&params->res[hash * 6 + 4]), (long long)(cm->h_lo_revenue[i]), __ATOMIC_RELAXED);
  //       //   }
  //       // }
  //     }
  //   }
  //   for (int i = end_batch ; i < end; i++) {
  //     int hash = HASH(cm->h_lo_suppkey[i], S_LEN, 0);
  //     int slot = ht_s[hash << 1];
  //     if (slot != 0) {
  //       hash = HASH(cm->h_lo_partkey[i], P_LEN, 0);
  //       slot = ht_p[hash << 1];
  //       if (slot != 0) {
  //         int brand = ht_p[(hash << 1) + 1];
  //         brand = cm->h_p_brand1[brand - 1];
  //         hash = HASH(cm->h_lo_orderdate[i], params->dim_len[cm->d_datekey], params->min_key[cm->d_datekey]);
  //         int year = ht_d[(hash << 1) + 1];
  //         year = cm->h_d_year[year - 1];
  //         hash = (brand * 7 +  (year - 1992)) % params->total_val;
  //         params->res[hash * 6] = year;
  //         params->res[hash * 6 + 1] = brand;
  //         __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&params->res[hash * 6 + 4]), (long long)(cm->h_lo_revenue[i]), __ATOMIC_RELAXED);
  //       }
  //     }
  //   }
  // });

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  cout << "Kernel time: " << time << endl;

};

void 
CPUGPUProcessing::call_pfilter_probe_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  int *filter_idx[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;
  //ColumnInfo* fkey[4] = {};

  int tile_items = 128*4;

  if(qo->joinGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize it to null

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    filter_idx[select_so_far + i] = col_idx[column->column_id];
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];

    output_selectivity *= params->selectivity[column];
  }

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    //fkey[table_id - 1] = column;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];

    output_selectivity *= params->selectivity[column];
  }

  if (off_col == NULL) {

    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaMalloc(output_estimate);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

    // filter_probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   NULL, NULL, NULL, NULL, NULL, cm->gpuCache, filter_idx[0], filter_idx[1], 
    //   _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    //   fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    //   LEN, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
    //   d_total, 0, d_segment_group);


    filter_probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      LEN, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

  } else {

    assert(*h_total > 0);

    output_estimate = *h_total * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaMalloc(output_estimate);
    }

    // filter_probe_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], 
    //   cm->gpuCache, filter_idx[0], filter_idx[1], 
    //   _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    //   fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3], 
    //   *h_total, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
    //   d_total, 0, NULL);

    filter_probe_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], 
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3], 
      *h_total, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total);

    CHECK_ERROR_STREAM(stream);

  }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

};

void 
CPUGPUProcessing::call_pfilter_probe_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int out_total = 0;
  int *filter_col[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;

  if(qo->joinCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to null

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    filter_col[select_so_far + i] = column->col_ptr;
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];

    output_selectivity *= params->selectivity[column];
  }

  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinCPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    fkey_col[table_id - 1] = column->col_ptr;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    ht[table_id - 1] = params->ht_CPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];

    output_selectivity *= params->selectivity[column];
  }

  if (h_off_col == NULL) {

    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaHostAlloc(output_estimate);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    filter_probe_CPU(NULL, NULL, NULL, NULL, NULL,
      filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN,
      ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      &out_total, 0, segment_group_ptr);

  } else {

    assert(*h_total > 0);

    output_estimate = *h_total * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaHostAlloc(output_estimate);
    }

    filter_probe_CPU(NULL, NULL, NULL, NULL, NULL,
      filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1],
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], *h_total,
      ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      &out_total, 0, NULL);

  }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

};

void
CPUGPUProcessing::call_probe_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream) {

  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_idx[2] = {}, *group_idx[4] = {};

  int tile_items = 128*4;

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    aggr_idx[i] = col_idx[column->column_id];
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      if (col_idx[column->column_id] == NULL) {
        int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
        CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      }
      group_idx[column_key->table_id - 1] = col_idx[column->column_id];
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  if (off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

    // fkey_idx[0] = cm->gpuCache + SEGMENT_SIZE * cm->segment_list[cm->lo_suppkey->column_id][0];
    // fkey_idx[2] = cm->gpuCache + SEGMENT_SIZE * cm->segment_list[cm->lo_partkey->column_id][0];
    // fkey_idx[3] = cm->gpuCache + SEGMENT_SIZE * cm->segment_list[cm->lo_orderdate->column_id][0];
    // aggr_idx[0] = cm->gpuCache + SEGMENT_SIZE * cm->segment_list[cm->lo_revenue->column_id][0];

    // probe_group_by_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   NULL, NULL, NULL, NULL, NULL,
    //   cm->gpuCache, fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    //   aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3], params->mode_group,
    //   LEN , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    //   _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    //   params->total_val, params->d_res, 0, d_segment_group);

    probe_group_by_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      aggr_idx[0], aggr_idx[1], params->mode_group,
      LEN , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->d_res, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

  } else {

    // probe_group_by_GPU2<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   off_col[0], off_col[1], off_col[2], off_col[3], off_col[4],
    //   cm->gpuCache, fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    //   aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3], params->mode_group,
    //   *h_total , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    //   _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    //   params->total_val, params->d_res, 0, NULL);

    probe_group_by_GPU3<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4],
      cm->gpuCache, fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3], params->mode_group,
      *h_total , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->d_res);

    CHECK_ERROR_STREAM(stream);

  }

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Kernel time: " << time << endl;

};

void
CPUGPUProcessing::call_probe_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg) {

  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_col[2] = {}, *group_col[4] = {};

  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinCPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    fkey_col[table_id - 1] = column->col_ptr;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    ht[table_id - 1] = params->ht_CPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    aggr_col[i] = column->col_ptr;
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      group_col[column_key->table_id - 1] = column->col_ptr;
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    probe_group_by_CPU(
      NULL, NULL, NULL, NULL, NULL,
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], 
      aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3], params->mode_group,
      LEN , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->res, 0, segment_group_ptr);
  } else {

    probe_group_by_CPU(
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4],
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], 
      aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3], params->mode_group,
      *h_total , ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      _min_val[0], _min_val[1], _min_val[2], _min_val[3],
      _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
      params->total_val, params->res, 0, NULL);
  }

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Kernel time: " << time << endl;

};

void 
CPUGPUProcessing::call_probe_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, cudaStream_t stream) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  float output_selectivity = 1.0;
  int output_estimate = 0;
  //ColumnInfo* fkey[4] = {};

  int tile_items = 128*4;

  if(qo->joinGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize it to null

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    //fkey[table_id - 1] = column;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];

    output_selectivity *= params->selectivity[column];
  }

  if (off_col == NULL) {

    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaMalloc(output_estimate);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

    // probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   NULL, NULL, NULL, NULL, NULL, cm->gpuCache,
    //   fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    //   LEN, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
    //   d_total, 0, d_segment_group);

    probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache,
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      LEN, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

  } else {

    assert(*h_total > 0);

    output_estimate = *h_total * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaMalloc(output_estimate);
    }

    // probe_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], cm->gpuCache, 
    //   fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3], 
    //   *h_total, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //   _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //   off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
    //   d_total, 0, NULL);

    probe_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], cm->gpuCache, 
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3], 
      *h_total, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total);

    CHECK_ERROR_STREAM(stream);

  }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

};

void 
CPUGPUProcessing::call_probe_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int out_total = 0;
  float output_selectivity = 1.0;
  int output_estimate = 0;

  if(qo->joinCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to null

  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinCPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    fkey_col[table_id - 1] = column->col_ptr;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    ht[table_id - 1] = params->ht_CPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];

    output_selectivity *= params->selectivity[column];
  }

  if (h_off_col == NULL) {

    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaHostAlloc(output_estimate);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    probe_CPU(NULL, NULL, NULL, NULL, NULL,
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN,
      ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      &out_total, 0, segment_group_ptr);

  } else {

    assert(*h_total > 0);

    output_estimate = *h_total * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaHostAlloc(output_estimate);
    }

    probe_CPU(h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4],
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], *h_total,
      ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      &out_total, 0, NULL);

  }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

};

void
CPUGPUProcessing::call_pfilter_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream) {
  int tile_items = 128*4;
  int **off_col_out;
  int *filter_idx[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;

  if (qo->selectGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to NULL

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    filter_idx[select_so_far + i] = col_idx[column->column_id];
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
    _mode[select_so_far + i] = params->mode[column];

    output_selectivity *= params->selectivity[column];
  }

  if (off_col == NULL) {

    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaMalloc(output_estimate);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

    // filter_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
    //   NULL, 
    //   cm->gpuCache, filter_idx[0], filter_idx[1], 
    //   _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
    //   off_col_out[0], d_total, LEN, 0, d_segment_group);

    filter_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], d_total, LEN, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

  } else {

    assert(*h_total > 0);

    output_estimate = *h_total * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaMalloc(output_estimate);
    }

    // filter_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>
    //   (off_col[0], 
    //   cm->gpuCache, filter_idx[0], filter_idx[1], 
    //   _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
    //   off_col_out[0], d_total,  *h_total, 0, NULL);

    filter_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>
      (off_col[0],
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], d_total,  *h_total);

    CHECK_ERROR_STREAM(stream);

  }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

}

void
CPUGPUProcessing::call_pfilter_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int **off_col_out;
  int *filter_col[2] = {};
  int out_total = 0;
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;

  if (qo->selectCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE](); //initialize to NULL

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    filter_col[select_so_far + i] = column->col_ptr;
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
    _mode[select_so_far + i] = params->mode[column];

    output_selectivity *= params->selectivity[column];
  }

  if (h_off_col == NULL) {

    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaHostAlloc(output_estimate);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    filter_CPU(NULL, filter_col[0], filter_col[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], &out_total, LEN,
      0, segment_group_ptr);

  } else {
    assert(filter_col[0] == NULL);
    assert(filter_col[1] != NULL);
    assert(*h_total > 0);

    output_estimate = *h_total * output_selectivity;

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i] || qo->joinGPUcheck[i])
        off_col_out[i] = cm->customCudaHostAlloc(output_estimate);
    }

    filter_CPU(h_off_col[0], filter_col[0], filter_col[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], &out_total, *h_total,
      0, NULL);

  }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

}

void 
CPUGPUProcessing::call_bfilter_build_GPU(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream) {
  int tile_items = 128*4;
  int* dimkey_idx, *group_idx, *filter_idx;
  ColumnInfo* column, *filter_col;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (params->ht_GPU[column] != NULL) {

    if (qo->groupby_build[column].size() > 0) {
      if (qo->groupGPUcheck) {
        ColumnInfo* group_col = qo->groupby_build[column][0];
        if (col_idx[group_col->column_id] == NULL) {
          int* desired = cm->customCudaMalloc(group_col->total_segment); int* expected = NULL;
          CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[group_col->column_id], group_col->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
          cudaStreamSynchronize(stream);
          __atomic_compare_exchange_n(&(col_idx[group_col->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        }
        group_idx = col_idx[group_col->column_id];
      } else group_idx = NULL;
    } else group_idx = NULL;

    if (qo->select_build[column].size() > 0) {
      filter_col = qo->select_build[column][0];
      if (col_idx[filter_col->column_id] == NULL) {
        int* desired = cm->customCudaMalloc(filter_col->total_segment); int* expected = NULL;
        //for (int x = 0; x < filter_col->total_segment; x++) cout << cm->segment_list[filter_col->column_id][x] << endl;
        CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[filter_col->column_id], filter_col->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        __atomic_compare_exchange_n(&(col_idx[filter_col->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      }
      filter_idx = col_idx[filter_col->column_id];
    } else filter_idx = NULL;

    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }

    dimkey_idx = col_idx[column->column_id];

    if (d_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      } 

      short* d_segment_group;
      d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
      CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

      // build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      //   NULL, cm->gpuCache, filter_idx, params->compare1[filter_col], params->compare2[filter_col], params->mode[filter_col],
      //   dimkey_idx, group_idx, LEN, 
      //   params->ht_GPU[column], params->dim_len[column], params->min_key[column],
      //   0, d_segment_group);

      build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, filter_idx, params->compare1[filter_col], params->compare2[filter_col], params->mode[filter_col],
        dimkey_idx, group_idx, LEN, 
        params->ht_GPU[column], params->dim_len[column], params->min_key[column],
        0, d_segment_group);

      CHECK_ERROR_STREAM(stream);

    } else {

      // build_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      //   d_off_col, cm->gpuCache, filter_idx, params->compare1[filter_col], params->compare2[filter_col], params->mode[filter_col],
      //   dimkey_idx, group_idx, *h_total,
      //   params->ht_GPU[column], params->dim_len[column], params->min_key[column], 
      //   0, NULL);

      build_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
        d_off_col, cm->gpuCache, filter_idx, params->compare1[filter_col], params->compare2[filter_col], params->mode[filter_col],
        dimkey_idx, group_idx, *h_total,
        params->ht_GPU[column], params->dim_len[column], params->min_key[column]);

      CHECK_ERROR_STREAM(stream);
    }
  }
};

void 
CPUGPUProcessing::call_bfilter_build_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* column, *filter_col;
  int* group_ptr, *filter_ptr;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (qo->groupby_build[column].size() > 0) {
    group_ptr = qo->groupby_build[column][0]->col_ptr;
  } else {
    group_ptr = NULL;
  }

  if (qo->select_build[column].size() > 0) {
    filter_col = qo->select_build[column][0];
    filter_ptr = filter_col->col_ptr;
  } else {
    filter_ptr = NULL;
  }

  if (params->ht_CPU[column] != NULL) {

    if (h_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

      build_CPU(NULL, filter_ptr, params->compare1[filter_col], params->compare2[filter_col], params->mode[filter_col], 
        column->col_ptr, group_ptr, LEN, params->ht_CPU[column], params->dim_len[column], params->min_key[column], 0, segment_group_ptr);

    } else {

      build_CPU(h_off_col, filter_ptr, params->compare1[filter_col], params->compare2[filter_col], params->mode[filter_col],
        column->col_ptr, group_ptr, *h_total, params->ht_CPU[column], params->dim_len[column], params->min_key[column], 0, NULL);

    }

  }
};

void 
CPUGPUProcessing::call_build_GPU(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream) {
  int tile_items = 128*4;
  int* dimkey_idx, *group_idx;
  ColumnInfo* column;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (params->ht_GPU[column] != NULL) {

    if (qo->groupby_build[column].size() > 0) {
      if (qo->groupGPUcheck) {
        ColumnInfo* group_col = qo->groupby_build[column][0];
        if (col_idx[group_col->column_id] == NULL) {
          int* desired = cm->customCudaMalloc(group_col->total_segment); int* expected = NULL;
          CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[group_col->column_id], group_col->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
          cudaStreamSynchronize(stream);
          __atomic_compare_exchange_n(&(col_idx[group_col->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        }
        group_idx = col_idx[group_col->column_id];
      } else group_idx = NULL;
    } else group_idx = NULL;

    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }

    dimkey_idx = col_idx[column->column_id];

    if (d_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* d_segment_group;
      d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
      CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

      // build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      //   NULL, cm->gpuCache, NULL, 0, 0, 0,
      //   dimkey_idx, group_idx, LEN, 
      //   params->ht_GPU[column], params->dim_len[column], params->min_key[column],
      //   0, d_segment_group);

      build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, NULL, 0, 0, 0,
        dimkey_idx, group_idx, LEN, 
        params->ht_GPU[column], params->dim_len[column], params->min_key[column],
        0, d_segment_group);

      CHECK_ERROR_STREAM(stream);

    } else {

      // build_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      //   d_off_col, cm->gpuCache, NULL, 0, 0, 0,
      //   dimkey_idx, group_idx, *h_total,
      //   params->ht_GPU[column], params->dim_len[column], params->min_key[column], 
      //   0, NULL);

      build_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
        d_off_col, cm->gpuCache, NULL, 0, 0, 0,
        dimkey_idx, group_idx, *h_total,
        params->ht_GPU[column], params->dim_len[column], params->min_key[column]);

      CHECK_ERROR_STREAM(stream);
    }
  }
};

void 
CPUGPUProcessing::call_build_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* column;
  int* group_ptr;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (qo->groupby_build[column].size() > 0) {
    group_ptr = qo->groupby_build[column][0]->col_ptr;
  } else {
    group_ptr = NULL;
  }

  if (params->ht_CPU[column] != NULL) {

    if (h_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

      build_CPU(NULL, NULL, 0, 0, 0, column->col_ptr, group_ptr, LEN, 
        params->ht_CPU[column], params->dim_len[column], params->min_key[column], 0, segment_group_ptr);

    } else {

      build_CPU(h_off_col, NULL, 0, 0, 0, column->col_ptr, group_ptr, *h_total, 
        params->ht_CPU[column], params->dim_len[column], params->min_key[column], 0, NULL);

    }
  }
};


void
CPUGPUProcessing::call_bfilter_GPU(QueryParams* params, int* &d_off_col, int* &d_total, int* h_total, int sg, int table, cudaStream_t stream) {

  ColumnInfo* temp;
  int tile_items = 128*4;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table){
      temp = qo->join[i].second; break;
    }
  }
  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];

  int output_estimate = qo->segment_group_count[table][sg] * SEGMENT_SIZE * params->selectivity[column];
  
  d_off_col = cm->customCudaMalloc(output_estimate);

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  int LEN;
  if (sg == qo->last_segment[table]) {
    LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
  } else { 
    LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
  }

  if (col_idx[column->column_id] == NULL) {
    int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
    CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
    cudaStreamSynchronize(stream);
    __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  int* filter_idx = col_idx[column->column_id];

  short* d_segment_group;
  d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
  short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
  CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));

  // filter_GPU2<128,4> <<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
  //   NULL, 
  //   cm->gpuCache, filter_idx, NULL, 
  //   params->compare1[column], params->compare2[column], 0, 0, params->mode[column], 0,
  //   d_off_col, d_total, LEN, 0, d_segment_group);

  filter_GPU2<128,4> <<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
    cm->gpuCache, filter_idx, NULL, 
    params->compare1[column], params->compare2[column], 0, 0, params->mode[column], 0,
    d_off_col, d_total, LEN, 0, d_segment_group);

  CHECK_ERROR_STREAM(stream);

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);
}

void
CPUGPUProcessing::call_bfilter_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* temp;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      temp = qo->join[i].second; break;
    }
  }

  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];
  int* filter_col = column->col_ptr;

  int output_estimate = qo->segment_group_count[table][sg] * SEGMENT_SIZE * params->selectivity[column];

  h_off_col = cm->customCudaHostAlloc(output_estimate);

  int LEN;
  if (sg == qo->last_segment[table]) {
    LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
  } else { 
    LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
  }

  short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

  filter_CPU(NULL, filter_col, NULL, 
    params->compare1[column], params->compare2[column], 0, 0, params->mode[column], 0,
    h_off_col, h_total, LEN,
    0, segment_group_ptr);

  cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

}

void
CPUGPUProcessing::call_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, cudaStream_t stream) {
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_idx[2] = {}, *group_idx[4] = {};
  int tile_items = 128 * 4;

  if (qo->groupby_probe[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    if (col_idx[column->column_id] == NULL) {
      int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      cudaStreamSynchronize(stream);
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    aggr_idx[i] = col_idx[column->column_id];
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      if (col_idx[column->column_id] == NULL) {
        int* desired = cm->customCudaMalloc(column->total_segment); int* expected = NULL;
        CubDebugExit(cudaMemcpyAsync(desired, cm->segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      }
      group_idx[column_key->table_id - 1] = col_idx[column->column_id];
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  groupByGPU<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
    off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], 
    cm->gpuCache, aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3], _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    params->total_val, *h_total, params->d_res, params->mode_group);

  CHECK_ERROR_STREAM(stream);

}

void
CPUGPUProcessing::call_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total) {
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_col[2] = {}, *group_col[4] = {};

  if (qo->groupby_probe[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    aggr_col[i] = column->col_ptr;
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      group_col[column_key->table_id - 1] = column->col_ptr;
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  groupByCPU(h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4], 
    aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3], _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    params->total_val, *h_total, params->res, params->mode_group);

}

#endif