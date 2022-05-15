#ifndef _CPU_PROCESSING_HE_
#define _CPU_PROCESSING_HE_

#include "common.h"
#include "KernelArgs.h"

#define BATCH_SIZE 256
#define NUM_THREADS 48
#define TASK_SIZE 1024 //! TASK_SIZE must be a factor of SEGMENT_SIZE and must be less than 20000

void filter_probe_CPUHE(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset, short* segment_group);

void filter_probe_CPU2HE(struct offsetCPU in_off, struct filterArgsCPU fargs, struct probeArgsCPU pargs,
  struct offsetCPU out_off, int num_tuples, int* total, int start_offset);

void probe_CPUHE(
  struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset, short* segment_group);

void probe_CPU2HE(struct offsetCPU in_off, struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset);

void probe_group_by_CPUHE(
  struct probeArgsCPU pargs,  struct groupbyArgsCPU gargs, int num_tuples, 
  int* res, int start_offset, short* segment_group);

void probe_group_by_CPU2HE(struct offsetCPU offset,
  struct probeArgsCPU pargs,  struct groupbyArgsCPU gargs, int num_tuples,
  int* res, int start_offset);

void filter_probe_group_by_CPUHE(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset, short* segment_group);

void filter_probe_group_by_CPU2HE(struct offsetCPU offset,
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset);

void build_CPUHE(struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table,
  int start_offset, short* segment_group);

void build_CPU2HE(int *dim_off, struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table,
  int start_offset);

void filter_CPUHE(struct filterArgsCPU fargs,
  int* out_off, int num_tuples, int* total,
  int start_offset, short* segment_group);

void filter_CPU2HE(int* off_col, struct filterArgsCPU fargs,
  int* out_off, int num_tuples, int* total,
  int start_offset);

void groupByCPUHE(struct offsetCPU offset, 
  struct groupbyArgsCPU gargs, int num_tuples, int* res);

void aggregationCPUHE(int* lo_off, 
  struct groupbyArgsCPU gargs, int num_tuples, int* res);

void probe_aggr_CPUHE(
  struct probeArgsCPU pargs, struct groupbyArgsCPU gargs, int num_tuples,
  int* res, int start_offset, short* segment_group);

void probe_aggr_CPU2HE(struct offsetCPU offset,
  struct probeArgsCPU pargs, struct groupbyArgsCPU gargs, 
  int num_tuples, int* res, int start_offset);

void filter_probe_aggr_CPUHE(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset, short* segment_group);

void filter_probe_aggr_CPU2HE(struct offsetCPU offset,
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset);

#endif