#ifndef _CPUGPU_PROCESSING_H_
#define _CPUGPU_PROCESSING_H_

#include "QueryOptimizer.cuh"
#include "GPUProcessing.h"
#include "CPUProcessing.h"

class CPUGPUProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;

  int** col_idx;
  chrono::high_resolution_clock::time_point begin_time;
  bool verbose;

  CPUGPUProcessing(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize, bool _verbose) {
    qo = new QueryOptimizer(_cache_size, _ondemand_size, _processing_size, _pinned_memsize);
    cm = qo->cm;
    begin_time = chrono::high_resolution_clock::now();
    col_idx = new int*[cm->TOT_COLUMN]();
    verbose = _verbose;
  }

  ~CPUGPUProcessing() {
    delete[] col_idx;
    delete qo;
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

  void call_aggregation_GPU(QueryParams* params, int* &off_col, int* h_total, cudaStream_t stream);

  void call_aggregation_CPU(QueryParams* params, int* &h_off_col, int* h_total);

  void call_probe_aggr_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_probe_aggr_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_pfilter_probe_aggr_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_aggr_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

};

#endif