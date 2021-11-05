#ifndef _CPU_PROCESSING_H_
#define _CPU_PROCESSING_H_

#include "common.h"
#include "KernelArgs.h"

#define BATCH_SIZE 256
#define NUM_THREADS 48
#define TASK_SIZE 1024 //! TASK_SIZE must be a factor of SEGMENT_SIZE and must be less than 20000

void filter_probe_CPU(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset, short* segment_group);

void filter_probe_CPU2(struct offsetCPU in_off, struct filterArgsCPU fargs, struct probeArgsCPU pargs,
  struct offsetCPU out_off, int num_tuples, int* total, int start_offset) ;

void probe_CPU(
  struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset, short* segment_group);

void probe_CPU2(struct offsetCPU in_off, struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset);

void probe_group_by_CPU(
  struct probeArgsCPU pargs,  struct groupbyArgsCPU gargs, int num_tuples, 
  int* res, int start_offset, short* segment_group);

void probe_group_by_CPU2(struct offsetCPU offset,
  struct probeArgsCPU pargs,  struct groupbyArgsCPU gargs, int num_tuples,
  int* res, int start_offset);

void filter_probe_group_by_CPU(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset, short* segment_group);

void filter_probe_group_by_CPU2(struct offsetCPU offset,
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset);

void build_CPU(struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table,
  int start_offset, short* segment_group);

void build_CPU2(int *dim_off, struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table,
  int start_offset);

void filter_CPU(struct filterArgsCPU fargs,
  int* out_off, int num_tuples, int* total,
  int start_offset, short* segment_group);

void filter_CPU2(int* off_col, struct filterArgsCPU fargs,
  int* out_off, int num_tuples, int* total,
  int start_offset);

void groupByCPU(struct offsetCPU offset, 
  struct groupbyArgsCPU gargs, int num_tuples, int* res);

void aggregationCPU(int* lo_off, 
  struct groupbyArgsCPU gargs, int num_tuples, int* res);

void probe_aggr_CPU(
  struct probeArgsCPU pargs, struct groupbyArgsCPU gargs, int num_tuples,
  int* res, int start_offset, short* segment_group);

void probe_aggr_CPU2(struct offsetCPU offset,
  struct probeArgsCPU pargs, struct groupbyArgsCPU gargs, 
  int num_tuples, int* res, int start_offset);

void filter_probe_aggr_CPU(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset, short* segment_group);

void filter_probe_aggr_CPU2(struct offsetCPU offset,
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset);

void merge(int* resCPU, int* resGPU, int num_tuples);

void build_CPU_minmax(struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table, int* min_global, int* max_global, 
  int start_offset, short* segment_group);

void build_CPU_minmax2(int *dim_off, struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table, int* min_global, int* max_global, 
  int start_offset);

#endif