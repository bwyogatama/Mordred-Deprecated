#ifndef _CPUGPU_PROCESSING_H_
#define _CPUGPU_PROCESSING_H_

#include "QueryOptimizer.h"
#include "GPUProcessing.h"
#include "CPUProcessing.h"
#include "CPUProcessingHE.h"
#include "common.h"

#define OD_BATCH_SIZE 8

class CPUGPUProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;

  bool custom;
  bool skipping;

  int** col_idx;
  // int** od_col_idx;
  chrono::high_resolution_clock::time_point begin_time;
  bool verbose;

  double transfer_time_total;
  double cpu_time_total;
  double gpu_time_total;
  double malloc_time_total;

  double* transfer_time;
  double* cpu_time;
  double* gpu_time;
  double* malloc_time;

  unsigned long long* cpu_to_gpu;
  unsigned long long* gpu_to_cpu;

  unsigned long long cpu_to_gpu_total;
  unsigned long long gpu_to_cpu_total;

  double execution_total;
  double optimization_total;
  double merging_total;

  CPUGPUProcessing(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize, bool _verbose, bool _custom = true, bool _skipping = true, double alpha = 0.1);

  ~CPUGPUProcessing() {
    delete[] col_idx;
    // delete[] od_col_idx;
    delete[] transfer_time;
    delete[] cpu_time;
    delete[] gpu_time;
    delete[] malloc_time;

    delete[] cpu_to_gpu;
    delete[] gpu_to_cpu;
    delete qo;
  }

  void resetCGP() {
    for (int i = 0; i < cm->TOT_COLUMN; i++) {
      if (col_idx[i] != NULL && !custom) cudaFree(col_idx);
      col_idx[i] = NULL;
    }
    // for (int i = 0; i < cm->TOT_COLUMN; i++) {
    //   od_col_idx[i] = NULL;
    // }
  }

  void resetTime();

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



  void call_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_aggregation_GPU(QueryParams* params, int* &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_aggregation_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg);

  void call_probe_aggr_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_probe_aggr_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_pfilter_probe_aggr_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_aggr_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

  void copyColIdx();

  void call_pfilter_probe_aggr_OD(QueryParams* params, 
      ColumnInfo** filter, ColumnInfo** pkey, ColumnInfo** fkey, ColumnInfo** aggr,
      int sg, int batch, int batch_size, int total_batch,
      cudaStream_t stream);

  void call_probe_group_by_OD(QueryParams* params, ColumnInfo** pkey, ColumnInfo** fkey, ColumnInfo** aggr,
    int sg, int batch, int batch_size, int total_batch,
    cudaStream_t stream);

  void call_probe_GPUNP(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, cudaStream_t stream, ColumnInfo* column);

  void call_probe_CPUNP(QueryParams* params, int** &h_off_col, int* h_total, int sg, ColumnInfo* column);

  void call_pfilter_GPUNP(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, cudaStream_t stream, ColumnInfo* column);

  void call_pfilter_CPUNP(QueryParams* params, int** &h_off_col, int* h_total, int sg, ColumnInfo* column);



  void call_pfilter_probe_group_by_GPUHE(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_group_by_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

  void call_pfilter_probe_GPUHE(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

  void call_probe_group_by_GPUHE(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_probe_group_by_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_probe_GPUHE(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, cudaStream_t stream);

  void call_probe_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_pfilter_GPUHE(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

  void call_bfilter_build_GPUHE(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream);

  void call_bfilter_build_CPUHE(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table);

  void call_build_GPUHE(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream);

  void call_build_CPUHE(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table);

  void call_bfilter_GPUHE(QueryParams* params, int* &d_off_col, int* &d_total, int* h_total, int sg, int table, cudaStream_t stream);

  void call_bfilter_CPUHE(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table);

  void call_group_by_GPUHE(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_group_by_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_aggregation_GPUHE(QueryParams* params, int* &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_aggregation_CPUHE(QueryParams* params, int* &h_off_col, int* h_total, int sg);

  void call_probe_aggr_GPUHE(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream);

  void call_probe_aggr_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg);

  void call_pfilter_probe_aggr_GPUHE(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream);

  void call_pfilter_probe_aggr_CPUHE(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far);

};

#endif