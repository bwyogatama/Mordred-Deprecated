#include "CPUGPUProcessing.h"
#include "CacheManager.h"
#include "QueryOptimizer.h"


CPUGPUProcessing::CPUGPUProcessing(size_t _cache_size, size_t _processing_size, size_t _pinned_memsize, bool _verbose, bool _custom, bool _skipping, double alpha) {
  custom = _custom;
  skipping = _skipping;
  if (custom) qo = new QueryOptimizer(_cache_size, _processing_size, _pinned_memsize, this);
  else qo = new QueryOptimizer(_cache_size, 0, 0, this);
  cm = qo->cm;
  begin_time = chrono::high_resolution_clock::now();
  col_idx = new int*[cm->TOT_COLUMN]();
  verbose = _verbose;
  cpu_time = new double[MAX_GROUPS];
  gpu_time = new double[MAX_GROUPS];
  transfer_time = new double[MAX_GROUPS];
  malloc_time = new double[MAX_GROUPS];

  cpu_to_gpu = new unsigned long long[MAX_GROUPS];
  gpu_to_cpu = new unsigned long long[MAX_GROUPS];

  resetTime();
}

void
CPUGPUProcessing::resetTime() {
  for (int sg = 0 ; sg < MAX_GROUPS; sg++) {
    cpu_time[sg] = 0;
    gpu_time[sg] = 0;
    transfer_time[sg] = 0;
    malloc_time[sg] = 0;

    cpu_to_gpu[sg] = 0;
    gpu_to_cpu[sg] = 0;
  }

  transfer_time_total = 0;
  gpu_time_total = 0;
  cpu_time_total = 0;
  malloc_time_total = 0;

  cpu_to_gpu_total = 0;
  gpu_to_cpu_total = 0;

  execution_total = 0;
  optimization_total = 0;
  merging_total = 0;

}

void 
CPUGPUProcessing::switch_device_fact(int** &off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table, cudaStream_t stream) {
  // chrono::high_resolution_clock::time_point st = chrono::high_resolution_clock::now();
  float time;
  SETUP_TIMING();

  if (mode == 0) {
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    // assert(*h_total > 0); // DONT BE SURPRISED IF WE REACHED THIS FOR 19980401-19980430 PREDICATES CAUSE THE RESULT IS 0
    assert(h_off_col[0] != NULL);
    off_col = new int*[cm->TOT_TABLE]();

    CubDebugExit(cudaMemcpyAsync(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice, stream));
    CubDebugExit(cudaStreamSynchronize(stream));
    cpu_to_gpu[sg] += (1 * sizeof(int));

    cudaEventRecord(start, 0);

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL) {
        if (!custom) CubDebugExit(cudaMalloc((void**) &off_col[i], *h_total * sizeof(int)));
        if (custom) off_col[i] = (int*) cm->customCudaMalloc<int>(*h_total);
      }
    }
  } else {
    if (off_col == NULL) return;
    assert(off_col != NULL);
    assert(off_col[0] != NULL);
    h_off_col = new int*[cm->TOT_TABLE]();

    CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CubDebugExit(cudaStreamSynchronize(stream));
    gpu_to_cpu[sg] += (1 * sizeof(int));

    cudaEventRecord(start, 0);

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL) {
        if (!custom) CubDebugExit(cudaHostAlloc((void**) &h_off_col[i], *h_total * sizeof(int), cudaHostAllocDefault));
        if (custom) h_off_col[i] = (int*) cm->customCudaHostAlloc<int>(*h_total);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  // cout << "sg: " << sg << " transfer malloc time: " << malloc_time[sg]<< endl;
  
  if (verbose) cout << "Transfer size: " << *h_total << " sg: " << sg << endl;

  cudaEventRecord(start, 0);

  if (mode == 0) { //CPU to GPU
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL) {
        // if (custom) off_col[i] = (int*) cm->customCudaMalloc<int>(*h_total);
        CubDebugExit(cudaMemcpyAsync(off_col[i], h_off_col[i], *h_total * sizeof(int), cudaMemcpyHostToDevice, stream));
        CubDebugExit(cudaStreamSynchronize(stream));
        if (!custom) cudaFreeHost(h_off_col[i]);
        cpu_to_gpu[sg] += (*h_total * sizeof(int));
      }
    }
    CubDebugExit(cudaStreamSynchronize(stream));
  } else { // GPU to CPU
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL) {
        // if (custom) h_off_col[i] = (int*) cm->customCudaHostAlloc<int>(*h_total);
        CubDebugExit(cudaMemcpyAsync(h_off_col[i], off_col[i], *h_total * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CubDebugExit(cudaStreamSynchronize(stream));
        if (!custom) cudaFree(off_col[i]);
        gpu_to_cpu[sg] += (*h_total * sizeof(int));
      }
    }
    CubDebugExit(cudaStreamSynchronize(stream));
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  if (verbose) cout << "Transfer Time: " << time << " sg: " << sg << endl;
  transfer_time[sg] += time;
  
}

void 
CPUGPUProcessing::switch_device_dim(int* &d_off_col, int* &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table, cudaStream_t stream) {

  float time;
  SETUP_TIMING();
  cudaEventRecord(start, 0);

  if (mode == 0) {
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    assert(*h_total > 0);

    CubDebugExit(cudaMemcpyAsync(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice, stream));
    CubDebugExit(cudaStreamSynchronize(stream));
    cpu_to_gpu[sg] += (1 * sizeof(int));

    if (!custom) CubDebugExit(cudaMalloc((void**) &d_off_col, *h_total * sizeof(int)));
    if (custom) d_off_col = (int*) cm->customCudaMalloc<int>(*h_total);
  } else {
    if (d_off_col == NULL) return;
    assert(d_off_col != NULL);
    assert(*h_total > 0);

    CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CubDebugExit(cudaStreamSynchronize(stream));
    gpu_to_cpu[sg] += (1 * sizeof(int));

    if (!custom) CubDebugExit(cudaHostAlloc((void**) &h_off_col, *h_total * sizeof(int), cudaHostAllocDefault));
    if (custom) h_off_col = (int*) cm->customCudaHostAlloc<int>(*h_total);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;

  if (verbose) cout << "Transfer size: " << *h_total << " sg: " << sg << endl;

  cudaEventRecord(start, 0);

  if (mode == 0) { //CPU to GPU
    // if (custom) d_off_col = (int*) cm->customCudaMalloc<int>(*h_total);

    if (h_off_col != NULL) {
      CubDebugExit(cudaMemcpyAsync(d_off_col, h_off_col, *h_total * sizeof(int), cudaMemcpyHostToDevice, stream));
      CubDebugExit(cudaStreamSynchronize(stream));
      cpu_to_gpu[sg] += (*h_total * sizeof(int));
      // if (!custom) cudaFreeHost(h_off_col); //TODO: UNCOMMENTING THIS WILL CAUSE SEGFAULT BECAUSE THERE IS A POSSIBILITY OF
      //FILTER CPU -> SWITCH -> BUILD GPU -> BUILD CPU (H_OFF_COL WILL BE USED AGAIN IN BUILD CPU)
    } else d_off_col = NULL;

    CubDebugExit(cudaStreamSynchronize(stream));

  } else { // GPU to CPU
    // if (custom) h_off_col = (int*) cm->customCudaHostAlloc<int>(*h_total);

    if (d_off_col != NULL) {
      CubDebugExit(cudaMemcpyAsync(h_off_col, d_off_col, *h_total * sizeof(int), cudaMemcpyDeviceToHost, stream));
      CubDebugExit(cudaStreamSynchronize(stream));
      gpu_to_cpu[sg] += (*h_total * sizeof(int));
      if (!custom) cudaFree(d_off_col);
    } else h_off_col = NULL;

    CubDebugExit(cudaStreamSynchronize(stream));
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  if (verbose) cout << "Transfer Time: " << time << " sg: " << sg << endl;
  transfer_time[sg] += time;
  
}

void 
CPUGPUProcessing::call_pfilter_probe_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  int *filter_idx[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;
  ColumnInfo* filter_col[2] = {};

  int tile_items = 128*4;

  if(qo->joinGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize it to null

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    filter_idx[select_so_far + i] = col_idx[column->column_id];
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
    filter_col[select_so_far + i] = column;

    output_selectivity *= params->selectivity[column];
  }

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];

    output_selectivity *= params->selectivity[column];
  }

  struct filterArgsGPU fargs = {
    filter_idx[0], filter_idx[1],
    _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    1, 1,
    (filter_col[0] != NULL) ? (params->map_filter_func_dev[filter_col[0]]) : (NULL), 
    (filter_col[1] != NULL) ? (params->map_filter_func_dev[filter_col[1]]) : (NULL)
  };

  struct probeArgsGPU pargs = {
    fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  if (off_col == NULL) {
    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinGPUcheck[i]) {
        if (!custom) CubDebugExit(cudaMalloc((void**) &off_col_out[i], output_estimate * sizeof(int)));
        if (custom) off_col_out[i] = (int*) cm->customCudaMalloc<int>(output_estimate);
      }
    }
  } else {
    assert(*h_total > 0);
    output_estimate = *h_total * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL || i == 0 || qo->joinGPUcheck[i]) {
        if (!custom) CubDebugExit(cudaMalloc((void**) &off_col_out[i], output_estimate * sizeof(int)));
        if (custom) off_col_out[i] = (int*) cm->customCudaMalloc<int>(output_estimate);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  if (off_col == NULL) {

    struct offsetGPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(cm->lo_orderdate->total_segment);
    else CubDebugExit(cudaMalloc((void**) &d_segment_group, cm->lo_orderdate->total_segment * sizeof(short)));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
    cpu_to_gpu[sg] += (qo->segment_group_count[0][sg] * sizeof(short));

    filter_probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, fargs, pargs, out_off, LEN, d_total, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

    if (!custom) cudaFree(d_segment_group);

  } else {

    assert(*h_total > 0);

    struct offsetGPU in_off = {
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4]
    };

    struct offsetGPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    filter_probe_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>( 
      cm->gpuCache, in_off, fargs, pargs, out_off, *h_total, d_total);

    CHECK_ERROR_STREAM(stream);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (off_col[i] != NULL) cudaFree(off_col[i]);
      }
    }

  }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CubDebugExit(cudaStreamSynchronize(stream));
  gpu_to_cpu[sg] += (1 * sizeof(int));

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg << endl;
  assert(*h_total <= output_estimate);
  // assert(*h_total > 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  gpu_time[sg] += time;

  if (verbose) cout << "Filter Probe Kernel time GPU: " << time << endl;

};

void 
CPUGPUProcessing::call_pfilter_probe_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int out_total = 0;
  ColumnInfo *filter_col[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;

  if(qo->joinCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to null

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    filter_col[select_so_far + i] = column;
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

  struct filterArgsCPU fargs = {
    (filter_col[0] != NULL) ? (filter_col[0]->col_ptr) : (NULL), 
    (filter_col[1] != NULL) ? (filter_col[1]->col_ptr) : (NULL),
    _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    1, 1, 
    (filter_col[0] != NULL) ? (params->map_filter_func_host[filter_col[0]]) : (NULL), 
    (filter_col[1] != NULL) ? (params->map_filter_func_host[filter_col[1]]) : (NULL)
  };

  struct probeArgsCPU pargs = {
    fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {
    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i]) {
        if (!custom) CubDebugExit(cudaHostAlloc((void**) &off_col_out[i], output_estimate * sizeof(int), cudaHostAllocDefault));
        if (custom) off_col_out[i] = (int*) cm->customCudaHostAlloc<int>(output_estimate);
      }
    }
  } else {
    assert(*h_total > 0);
    output_estimate = *h_total * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL || i == 0 || qo->joinCPUcheck[i]) {
        if (!custom) CubDebugExit(cudaHostAlloc((void**) &off_col_out[i], output_estimate * sizeof(int), cudaHostAllocDefault));
        if (custom) off_col_out[i] = (int*) cm->customCudaHostAlloc<int>(output_estimate);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {

    struct offsetCPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    filter_probe_CPU(
      fargs, pargs, out_off, LEN, &out_total, 0, segment_group_ptr);

  } else {

    assert(*h_total > 0);

    struct offsetCPU in_off = {
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4]
    };

    struct offsetCPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    filter_probe_CPU2(
      in_off, fargs, pargs, out_off, *h_total, &out_total, 0);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (h_off_col[i] != NULL) cudaFreeHost(h_off_col[i]);
      }
    }

  }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg << endl;
  assert(*h_total <= output_estimate);
  // assert(*h_total > 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  if (verbose) cout << "Filter Probe Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;

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
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    aggr_idx[i] = col_idx[column->column_id];
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      cm->indexTransfer(col_idx, column, stream, custom);
      cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
      group_idx[column_key->table_id - 1] = col_idx[column->column_id];
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  struct probeArgsGPU pargs = {
    fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  struct groupbyArgsGPU gargs = {
    aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    params->total_val, params->mode_group, params->d_group_func
  };

  float time;
  SETUP_TIMING();

  if (off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(cm->lo_orderdate->total_segment);
    else CubDebugExit(cudaMalloc((void**) &d_segment_group, cm->lo_orderdate->total_segment * sizeof(short)));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
    cpu_to_gpu[sg] += (qo->segment_group_count[0][sg] * sizeof(short));

    cudaEventRecord(start, 0);

    probe_group_by_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, pargs, gargs, LEN, params->d_res, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

    if (!custom) cudaFree(d_segment_group);

  } else {

    struct offsetGPU offset = {
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4]
    };

    cudaEventRecord(start, 0);
    
    probe_group_by_GPU3<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, offset, pargs, gargs, *h_total, params->d_res);

    CHECK_ERROR_STREAM(stream);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (off_col[i] != NULL) cudaFree(off_col[i]);
      }
    }

  }

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  if (verbose) cout << "Probe Group Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;

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

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
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

  struct probeArgsCPU pargs = {
    fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  struct groupbyArgsCPU gargs = {
    aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    params->total_val, params->mode_group, params->h_group_func
  };

  float time;
  SETUP_TIMING();
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    probe_group_by_CPU(pargs, gargs, LEN , params->res, 0, segment_group_ptr);
  } else {

    struct offsetCPU offset = {
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4]
    };

    probe_group_by_CPU2(offset, pargs, gargs, *h_total, params->res, 0);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (h_off_col[i] != NULL) cudaFreeHost(h_off_col[i]);
      }
    }
  }

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  if (verbose) cout << "Probe Group Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;

};

void 
CPUGPUProcessing::call_probe_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, cudaStream_t stream) {

  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  float output_selectivity = 1.0;
  int output_estimate = 0;

  int tile_items = 128*4;

  if(qo->joinGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize it to null

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
    output_selectivity *= params->selectivity[column];
  }

  struct probeArgsGPU pargs = {
    fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };


  float time;
  SETUP_TIMING();
  cudaEventRecord(start, 0);

  if (off_col == NULL) {
    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinGPUcheck[i]) {
        if (!custom) CubDebugExit(cudaMalloc((void**) &off_col_out[i], output_estimate * sizeof(int)));
        if (custom) off_col_out[i] = (int*) cm->customCudaMalloc<int>(output_estimate);
      }
    }
  } else {
    assert(*h_total > 0);
    output_estimate = *h_total * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL || i == 0 || qo->joinGPUcheck[i]) {
        if (!custom) CubDebugExit(cudaMalloc((void**) &off_col_out[i], output_estimate * sizeof(int)));
        if (custom) off_col_out[i] = (int*) cm->customCudaMalloc<int>(output_estimate);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  if (off_col == NULL) {

    struct offsetGPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(cm->lo_orderdate->total_segment);
    else CubDebugExit(cudaMalloc((void**) &d_segment_group, cm->lo_orderdate->total_segment * sizeof(short)));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
    cpu_to_gpu[sg] += (qo->segment_group_count[0][sg] * sizeof(short));

    probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, pargs, out_off, LEN, d_total, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

    if (!custom) cudaFree(d_segment_group);

  } else {

    assert(*h_total > 0);

      struct offsetGPU in_off = {
        off_col[0], off_col[1], off_col[2], off_col[3], off_col[4]
      };

      struct offsetGPU out_off = {
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
      };

      CHECK_ERROR_STREAM(stream);

      probe_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, in_off, pargs, out_off, *h_total, d_total);

      CHECK_ERROR_STREAM(stream);   

      if (!custom) {
        for (int i = 0; i < cm->TOT_TABLE; i++) {
          if (off_col[i] != NULL) cudaFree(off_col[i]);
        }
      }   
    // }

  }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CubDebugExit(cudaStreamSynchronize(stream));
  gpu_to_cpu[sg] += (1 * sizeof(int));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  // assert(*h_total > 0);

  if (verbose) cout << "Probe Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;

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

  struct probeArgsCPU pargs = {
    fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  float time;
  SETUP_TIMING();
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {
    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (i == 0 || qo->joinCPUcheck[i]) {
        if (!custom) CubDebugExit(cudaHostAlloc((void**) &off_col_out[i], output_estimate * sizeof(int), cudaHostAllocDefault));
        if (custom) off_col_out[i] = (int*) cm->customCudaHostAlloc<int>(output_estimate);
      }
    }
  } else {
    assert(*h_total > 0);
    output_estimate = *h_total * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL || i == 0 || qo->joinCPUcheck[i]) {
        if (!custom) CubDebugExit(cudaHostAlloc((void**) &off_col_out[i], output_estimate * sizeof(int), cudaHostAllocDefault));
        if (custom) off_col_out[i] = (int*) cm->customCudaHostAlloc<int>(output_estimate);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {

    struct offsetCPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    probe_CPU(pargs, out_off, LEN, &out_total, 0, segment_group_ptr);

  } else {

    assert(*h_total > 0);

    struct offsetCPU in_off = {
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4]
    };

    struct offsetCPU out_off = {
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4]
    };

    probe_CPU2(in_off, pargs, out_off, *h_total, &out_total, 0);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (h_off_col[i] != NULL) cudaFreeHost(h_off_col[i]);
      }
    }

  }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  // assert(*h_total > 0);

  if (verbose) cout << "Probe Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;
};

//WONT WORK IF JOIN HAPPEN BEFORE FILTER (ONLY WRITE OUTPUT AS A SINGLE COLUMN OFF_COL_OUT[0])
void
CPUGPUProcessing::call_pfilter_GPU(QueryParams* params, int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far, cudaStream_t stream) {
  int tile_items = 128*4;
  int **off_col_out;
  int *filter_idx[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;
  ColumnInfo* filter_col[2] = {};

  if (qo->selectGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to NULL

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    filter_idx[select_so_far + i] = col_idx[column->column_id];
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
    _mode[select_so_far + i] = params->mode[column];
    filter_col[select_so_far + i] = column;

    output_selectivity *= params->selectivity[column];
  }

  struct filterArgsGPU fargs = {
    filter_idx[0], filter_idx[1],
    _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    _mode[0], _mode[1], 
    (filter_col[0] != NULL) ? (params->map_filter_func_dev[filter_col[0]]) : (NULL), 
    (filter_col[1] != NULL) ? (params->map_filter_func_dev[filter_col[1]]) : (NULL)
  };

  float time;
  SETUP_TIMING();
  cudaEventRecord(start, 0);

  if (off_col == NULL) {
    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;
    if (!custom) CubDebugExit(cudaMalloc((void**) &off_col_out[0], output_estimate * sizeof(int)));
    if (custom) off_col_out[0] = (int*) cm->customCudaMalloc<int>(output_estimate);
  } else {
    assert(*h_total > 0);
    output_estimate = *h_total * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL || i == 0) {
        if (!custom) CubDebugExit(cudaMalloc((void**) &off_col_out[i], output_estimate * sizeof(int)));
        if (custom) off_col_out[i] = (int*) cm->customCudaMalloc<int>(output_estimate);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  if (off_col == NULL) {

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(cm->lo_orderdate->total_segment);
    else CubDebugExit(cudaMalloc((void**) &d_segment_group, cm->lo_orderdate->total_segment * sizeof(short)));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
    cpu_to_gpu[sg] += (qo->segment_group_count[0][sg] * sizeof(short));

    filter_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, fargs, off_col_out[0], LEN, d_total, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

    if (!custom) cudaFree(d_segment_group);

  } else {

    assert(*h_total > 0);

    filter_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>
      (cm->gpuCache, off_col[0], fargs, off_col_out[0], *h_total, d_total);

    CHECK_ERROR_STREAM(stream);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (off_col[i] != NULL) cudaFree(off_col[i]);
      }
    }

  }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CubDebugExit(cudaStreamSynchronize(stream));
  gpu_to_cpu[sg] += (1 * sizeof(int));

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Filter Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;

}

//WONT WORK IF JOIN HAPPEN BEFORE FILTER (ONLY WRITE OUTPUT AS A SINGLE COLUMN OFF_COL_OUT[0])
void
CPUGPUProcessing::call_pfilter_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int **off_col_out;
  ColumnInfo *filter_col[2] = {};
  int out_total = 0;
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};
  float output_selectivity = 1.0;
  int output_estimate = 0;

  if (qo->selectCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE](); //initialize to NULL

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    filter_col[select_so_far + i] = column;
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
    _mode[select_so_far + i] = params->mode[column];
    output_selectivity *= params->selectivity[column];
  }

  struct filterArgsCPU fargs = {
    (filter_col[0] != NULL) ? (filter_col[0]->col_ptr) : (NULL), 
    (filter_col[1] != NULL) ? (filter_col[1]->col_ptr) : (NULL),
    _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    _mode[0], _mode[1], 
    (filter_col[0] != NULL) ? (params->map_filter_func_host[filter_col[0]]) : (NULL), 
    (filter_col[1] != NULL) ? (params->map_filter_func_host[filter_col[1]]) : (NULL)
  };

  float time;
  SETUP_TIMING();
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {
    output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;
    if (!custom) CubDebugExit(cudaHostAlloc((void**) &off_col_out[0], output_estimate * sizeof(int), cudaHostAllocDefault));
    if (custom) off_col_out[0] = (int*) cm->customCudaHostAlloc<int>(output_estimate);
  } else {
    assert(filter_col[0] == NULL);
    assert(filter_col[1] != NULL);
    assert(*h_total > 0);
    output_estimate = *h_total * output_selectivity;
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL || i == 0) {
        if (!custom) CubDebugExit(cudaHostAlloc((void**) &off_col_out[i], output_estimate * sizeof(int), cudaHostAllocDefault));
        if (custom) off_col_out[i] = (int*) cm->customCudaHostAlloc<int>(output_estimate);
      }
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  if (h_off_col == NULL) {

    // output_estimate = SEGMENT_SIZE * qo->segment_group_count[0][sg] * output_selectivity;

    // if (custom) off_col_out[0] = (int*) cm->customCudaHostAlloc<int>(output_estimate);

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    filter_CPU(fargs, off_col_out[0], LEN, &out_total, 0, segment_group_ptr);

  } else {
    assert(filter_col[0] == NULL);
    assert(filter_col[1] != NULL);
    assert(*h_total > 0);

    filter_CPU2(h_off_col[0], fargs, off_col_out[0], *h_total, &out_total, 0);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (h_off_col[i] != NULL) cudaFreeHost(h_off_col[i]);
      }
    }

  }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Filter Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;

}

void 
CPUGPUProcessing::call_bfilter_build_GPU(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream) {
  int tile_items = 128*4;
  int* dimkey_idx, *group_idx = NULL, *filter_idx = NULL;
  ColumnInfo* column, *filter_col;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (params->ht_GPU[column] != NULL) {

    if (qo->groupby_build.size() > 0 && qo->groupby_build[column].size() > 0) {
      if (qo->groupGPUcheck) {
        ColumnInfo* group_col = qo->groupby_build[column][0];
        cm->indexTransfer(col_idx, group_col, stream, custom);
        cpu_to_gpu[sg] += (group_col->total_segment * sizeof(int));
        group_idx = col_idx[group_col->column_id];
      }
    }

    if (qo->select_build[column].size() > 0) {
      filter_col = qo->select_build[column][0];
      cm->indexTransfer(col_idx, filter_col, stream, custom);
      cpu_to_gpu[sg] += (filter_col->total_segment * sizeof(int));
      filter_idx = col_idx[filter_col->column_id];
    }

    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));

    dimkey_idx = col_idx[column->column_id];

    struct filterArgsGPU fargs = {
      filter_idx, NULL,
      params->compare1[filter_col], params->compare2[filter_col], 0, 0,
      params->mode[filter_col], 0,
      (filter_col != NULL) ? (params->map_filter_func_dev[filter_col]) : (NULL), NULL
    };

    struct buildArgsGPU bargs = {
      dimkey_idx, group_idx,
      params->dim_len[column], params->min_key[column]
    };

    SETUP_TIMING();
    float time;
    cudaEventRecord(start, 0);

    if (d_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      } 

      short* d_segment_group;
      // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
      if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(column->total_segment);
      else CubDebugExit(cudaMalloc((void**) &d_segment_group, column->total_segment * sizeof(short)));
      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
      CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
      cpu_to_gpu[sg] += (qo->segment_group_count[table][sg] * sizeof(short));

      build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, fargs, bargs, LEN, params->ht_GPU[column], 0, d_segment_group);

      CHECK_ERROR_STREAM(stream);

      if (!custom) cudaFree(d_segment_group);

    } else {

      build_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, d_off_col, fargs, bargs, *h_total, params->ht_GPU[column]);

      CHECK_ERROR_STREAM(stream);

      if (!custom) cudaFree(d_off_col);
        
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (verbose) cout << "Filter Build Kernel time GPU: " << time << endl;
    gpu_time[sg] += time;
  }
};

void 
CPUGPUProcessing::call_bfilter_build_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* column, *filter_col;
  int* group_ptr = NULL, *filter_ptr = NULL;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (qo->groupby_build.size() > 0 && qo->groupby_build[column].size() > 0) {
    group_ptr = qo->groupby_build[column][0]->col_ptr;
  }

  if (qo->select_build[column].size() > 0) {
    filter_col = qo->select_build[column][0];
    filter_ptr = filter_col->col_ptr;
  }

  struct filterArgsCPU fargs = {
    filter_ptr, NULL,
    params->compare1[filter_col], params->compare2[filter_col], 0, 0,
    params->mode[filter_col], 0, 
    (filter_col != NULL) ? (params->map_filter_func_host[filter_col]) : (NULL), NULL
  };

  struct buildArgsCPU bargs = {
    column->col_ptr, group_ptr,
    params->dim_len[column], params->min_key[column]
  };

  if (params->ht_CPU[column] != NULL) {

    SETUP_TIMING();
    float time;
    cudaEventRecord(start, 0);

    if (h_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

      build_CPU(fargs, bargs, LEN, params->ht_CPU[column], 0, segment_group_ptr);

    } else {

      build_CPU2(h_off_col, fargs, bargs, *h_total, params->ht_CPU[column], 0);

      if (!custom) cudaFreeHost(h_off_col);

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (verbose) cout << "Filter Build Kernel time CPU: " << time << endl;
    cpu_time[sg] += time;

  }
};

void 
CPUGPUProcessing::call_build_GPU(QueryParams* params, int* &d_off_col, int* h_total, int sg, int table, cudaStream_t stream) {
  int tile_items = 128*4;
  int* dimkey_idx, *group_idx = NULL;
  ColumnInfo* column;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (params->ht_GPU[column] != NULL) {

    if (qo->groupby_build.size() > 0 && qo->groupby_build[column].size() > 0) {
      if (qo->groupGPUcheck) {
        ColumnInfo* group_col = qo->groupby_build[column][0];
        cm->indexTransfer(col_idx, group_col, stream, custom);
        cpu_to_gpu[sg] += (group_col->total_segment * sizeof(int));
        group_idx = col_idx[group_col->column_id];
      }
    }

    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));

    dimkey_idx = col_idx[column->column_id];

    struct filterArgsGPU fargs = {NULL, NULL, 0, 0, 0, 0, 0, 0, NULL, NULL};

    struct buildArgsGPU bargs = {
      dimkey_idx, group_idx,
      params->dim_len[column], params->min_key[column]
    };

    SETUP_TIMING();
    float time;
    cudaEventRecord(start, 0);

    if (d_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* d_segment_group;
      // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
      if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(column->total_segment);
      else CubDebugExit(cudaMalloc((void**) &d_segment_group, column->total_segment * sizeof(short)));
      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
      CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
      cpu_to_gpu[sg] += (qo->segment_group_count[table][sg] * sizeof(short));

      build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, fargs, bargs, LEN, params->ht_GPU[column], 0, d_segment_group);

      CHECK_ERROR_STREAM(stream);

      if (!custom) cudaFree(d_segment_group);

    } else {

      build_GPU3<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
        cm->gpuCache, d_off_col, fargs, bargs, *h_total, params->ht_GPU[column]);

      CHECK_ERROR_STREAM(stream);

      if (!custom) cudaFree(d_off_col);

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (verbose) cout << "Build Kernel time GPU: " << time << endl;
    gpu_time[sg] += time;
  }
};

void 
CPUGPUProcessing::call_build_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* column;
  int* group_ptr = NULL;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (qo->groupby_build.size() > 0 && qo->groupby_build[column].size() > 0) {
    group_ptr = qo->groupby_build[column][0]->col_ptr;
  }

  struct filterArgsCPU fargs = {NULL, NULL, 0, 0, 0, 0, 0, 0, NULL, NULL};

  struct buildArgsCPU bargs = {
    column->col_ptr, group_ptr,
    params->dim_len[column], params->min_key[column]
  };

  if (params->ht_CPU[column] != NULL) {

    SETUP_TIMING();
    float time;
    cudaEventRecord(start, 0);

    if (h_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

      build_CPU(fargs, bargs, LEN, params->ht_CPU[column], 0, segment_group_ptr);

    } else {

      build_CPU2(h_off_col, fargs, bargs, *h_total, params->ht_CPU[column], 0);

      if (!custom) cudaFreeHost(h_off_col);

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (verbose) cout << "Build Kernel time CPU: " << time << endl;
    cpu_time[sg] += time;
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

  // assert(qo->select_build[temp].size() > 0);
  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];

  int output_estimate = qo->segment_group_count[table][sg] * SEGMENT_SIZE * params->selectivity[column];

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);
  
  // d_off_col = (int*) cm->customCudaMalloc<int>(output_estimate);
  if (custom) d_off_col = (int*) cm->customCudaMalloc<int>(output_estimate);
  else CubDebugExit(cudaMalloc((void**) &d_off_col, output_estimate * sizeof(int)));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  CubDebugExit(cudaMemsetAsync(d_total, 0, sizeof(int), stream));

  int LEN;
  if (sg == qo->last_segment[table]) {
    LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
  } else { 
    LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
  }

  cm->indexTransfer(col_idx, column, stream, custom);
  cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
  int* filter_idx = col_idx[column->column_id];

  struct filterArgsGPU fargs = {
    filter_idx, NULL,
    params->compare1[column], params->compare2[column], 0, 0,
    params->mode[column], 0, params->map_filter_func_dev[column], NULL
  };

  short* d_segment_group;
  // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
  if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(column->total_segment);
  else CubDebugExit(cudaMalloc((void**) &d_segment_group, column->total_segment * sizeof(short)));
  short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
  CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
  cpu_to_gpu[sg] += (qo->segment_group_count[table][sg] * sizeof(short));

  filter_GPU2<128,4> <<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
    cm->gpuCache, fargs, d_off_col, LEN, d_total, 0, d_segment_group);

  CHECK_ERROR_STREAM(stream);

  CubDebugExit(cudaMemcpyAsync(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CubDebugExit(cudaStreamSynchronize(stream));
  gpu_to_cpu[sg] += (1 * sizeof(int));

  if (!custom) cudaFree(d_segment_group);

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Filter Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;
};

void
CPUGPUProcessing::call_bfilter_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* temp;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      temp = qo->join[i].second; break;
    }
  }

  // assert(qo->select_build[temp].size() > 0);
  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];
  int* filter_col = column->col_ptr;

  int output_estimate = qo->segment_group_count[table][sg] * SEGMENT_SIZE * params->selectivity[column];

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  if (custom) h_off_col = (int*) cm->customCudaHostAlloc<int>(output_estimate);
  else CubDebugExit(cudaHostAlloc((void**) &h_off_col, output_estimate * sizeof(int), cudaHostAllocDefault));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  malloc_time[sg] += time;
  cudaEventRecord(start, 0);

  int LEN;
  if (sg == qo->last_segment[table]) {
    LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
  } else { 
    LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
  }

  struct filterArgsCPU fargs = {
    filter_col, NULL,
    params->compare1[column], params->compare2[column], 0, 0,
    params->mode[column], 0, params->map_filter_func_host[column], NULL
  };

  short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

  filter_CPU(fargs, h_off_col, LEN, h_total, 0, segment_group_ptr);

  if (verbose) cout << "h_total: " << *h_total << " output_estimate: " << output_estimate << " sg: " << sg  << endl;
  assert(*h_total <= output_estimate);
  assert(*h_total > 0);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Filter Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;

};

void
CPUGPUProcessing::call_group_by_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream) {
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_idx[2] = {}, *group_idx[4] = {};
  int tile_items = 128 * 4;

  if (qo->aggregation[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    aggr_idx[i] = col_idx[column->column_id];
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      cm->indexTransfer(col_idx, column, stream, custom);
      cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
      group_idx[column_key->table_id - 1] = col_idx[column->column_id];
      _min_val[column_key->table_id - 1] = params->min_val[column_key];
      _unique_val[column_key->table_id - 1] = params->unique_val[column_key];
    }
  }

  struct groupbyArgsGPU gargs = {
    aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    params->total_val, params->mode_group, params->d_group_func
  };

  struct offsetGPU offset = {
    off_col[0], off_col[1], off_col[2], off_col[3], off_col[4]
  };

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  if (*h_total > 0) {
    groupByGPU<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, offset, gargs, *h_total, params->d_res);
  }

  CHECK_ERROR_STREAM(stream);

  if (!custom) {
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL) cudaFree(off_col[i]);
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Group Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;
};

void
CPUGPUProcessing::call_group_by_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg) {
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_col[2] = {}, *group_col[4] = {};

  if (qo->aggregation[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
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

  struct groupbyArgsCPU gargs = {
    aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3],
    _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    params->total_val, params->mode_group, params->h_group_func
  };

  struct offsetCPU offset = {
    h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4], 
  };

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  if (*h_total > 0) groupByCPU(offset, gargs, *h_total, params->res);

  if (!custom) {
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL) cudaFreeHost(h_off_col[i]);
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Group Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;

};

void
CPUGPUProcessing::call_aggregation_GPU(QueryParams* params, int* &off_col, int* h_total, int sg, cudaStream_t stream) {

  int *aggr_idx[2] = {};
  int tile_items = 128 * 4;

  if (qo->aggregation[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    aggr_idx[i] = col_idx[column->column_id];
  }

  struct groupbyArgsGPU gargs = {
    aggr_idx[0], aggr_idx[1], NULL, NULL, NULL, NULL,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, params->mode_group, params->d_group_func
  };

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  if (*h_total > 0) {
    aggregationGPU<128,4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
    cm->gpuCache, off_col, gargs, *h_total, params->d_res);
  }

  CHECK_ERROR_STREAM(stream);

  if (!custom) cudaFree(off_col);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Aggr Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;
};

void 
CPUGPUProcessing::call_aggregation_CPU(QueryParams* params, int* &h_off_col, int* h_total, int sg) {
  int *aggr_col[2] = {};

  if (qo->aggregation[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    aggr_col[i] = column->col_ptr;
  }

  struct groupbyArgsCPU gargs = {
    aggr_col[0], aggr_col[1], NULL, NULL, NULL, NULL,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, params->mode_group, params->h_group_func
  };

  SETUP_TIMING();
  float time;
  cudaEventRecord(start, 0);

  // assert(h_off_col != NULL);

  if (*h_total > 0) aggregationCPU(h_off_col, gargs, *h_total, params->res);

  if (!custom) cudaFree(h_off_col);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Aggr Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;
};

void 
CPUGPUProcessing::call_probe_aggr_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, cudaStream_t stream) {
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  int *aggr_idx[2] = {};

  int tile_items = 128*4;

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    aggr_idx[i] = col_idx[column->column_id];
  }

  struct probeArgsGPU pargs = {
    fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  struct groupbyArgsGPU gargs = {
    aggr_idx[0], aggr_idx[1], NULL, NULL, NULL, NULL,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, params->mode_group, params->d_group_func
  };

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
    // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(cm->lo_orderdate->total_segment);
    else CubDebugExit(cudaMalloc((void**) &d_segment_group, cm->lo_orderdate->total_segment * sizeof(short)));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
    cpu_to_gpu[sg] += (qo->segment_group_count[0][sg] * sizeof(short));

    probe_aggr_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, pargs, gargs, LEN, params->d_res, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

    if (!custom) cudaFree(d_segment_group);

  } else {

    struct offsetGPU offset = {
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4]
    };

    probe_aggr_GPU3<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, offset, pargs, gargs, *h_total, params->d_res);

    CHECK_ERROR_STREAM(stream);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (off_col[i] != NULL) cudaFree(off_col[i]);
      }
    }

  }

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  if (verbose) cout << "Probe Aggr Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;
};

void 
CPUGPUProcessing::call_probe_aggr_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg) {

  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int *aggr_col[2] = {};

  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinCPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    fkey_col[table_id - 1] = column->col_ptr;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    ht[table_id - 1] = params->ht_CPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    aggr_col[i] = column->col_ptr;
  }

  struct probeArgsCPU pargs = {
    fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  struct groupbyArgsCPU gargs = {
    aggr_col[0], aggr_col[1], NULL, NULL, NULL, NULL,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, params->mode_group, params->h_group_func
  };

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

    probe_aggr_CPU(pargs, gargs, LEN, params->res, 0, segment_group_ptr);
  } else {

    struct offsetCPU offset = {
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4]
    };

    probe_aggr_CPU2(offset, pargs, gargs, *h_total, params->res, 0);
  }

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  if (verbose) cout << "Probe Aggr Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;

};

void
CPUGPUProcessing::call_pfilter_probe_aggr_GPU(QueryParams* params, int** &off_col, int* h_total, int sg, int select_so_far, cudaStream_t stream) {

  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}; //initialize it to null
  int *filter_idx[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  int *aggr_idx[2] = {};
  ColumnInfo* filter_col[2] = {};

  int tile_items = 128*4;

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    filter_idx[select_so_far + i] = col_idx[column->column_id];
    _compare1[select_so_far + i] = params->compare1[column];
    _compare2[select_so_far + i] = params->compare2[column];
    filter_col[select_so_far + i] = column;
  }

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    int table_id = qo->fkey_pkey[column]->table_id;
    ColumnInfo* pkey = qo->fkey_pkey[column];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    assert(col_idx[column->column_id] != NULL);
    fkey_idx[table_id - 1] = col_idx[column->column_id];
    ht[table_id - 1] = params->ht_GPU[pkey];
    _min_key[table_id - 1] = params->min_key[pkey];
    _dim_len[table_id - 1] = params->dim_len[pkey];
  }

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    cm->indexTransfer(col_idx, column, stream, custom);
    cpu_to_gpu[sg] += (column->total_segment * sizeof(int));
    aggr_idx[i] = col_idx[column->column_id];
  }

  struct filterArgsGPU fargs = {
    filter_idx[0], filter_idx[1],
    _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    1, 1,
    (filter_col[0] != NULL) ? (params->map_filter_func_dev[filter_col[0]]) : (NULL), 
    (filter_col[1] != NULL) ? (params->map_filter_func_dev[filter_col[1]]) : (NULL)
  };

  struct probeArgsGPU pargs = {
    fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  struct groupbyArgsGPU gargs = {
    aggr_idx[0], aggr_idx[1], NULL, NULL, NULL, NULL,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, params->mode_group, params->d_group_func
  };

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
    // d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    if (custom) d_segment_group = (short*) cm->customCudaMalloc<short>(cm->lo_orderdate->total_segment);
    else CubDebugExit(cudaMalloc((void**) &d_segment_group, cm->lo_orderdate->total_segment * sizeof(short)));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpyAsync(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice, stream));
    cpu_to_gpu[sg] += (qo->segment_group_count[0][sg] * sizeof(short));

    filter_probe_aggr_GPU2<128, 4><<<(LEN + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, fargs, pargs, gargs, LEN, params->d_res, 0, d_segment_group);

    CHECK_ERROR_STREAM(stream);

    if (!custom) cudaFree(d_segment_group);

  } else {

    struct offsetGPU offset = {
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4]
    };

    filter_probe_aggr_GPU3<128, 4><<<(*h_total + tile_items - 1)/tile_items, 128, 0, stream>>>(
      cm->gpuCache, offset, fargs, pargs, gargs, *h_total, params->d_res);

    CHECK_ERROR_STREAM(stream);

    if (!custom) {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        if (off_col[i] != NULL) cudaFree(off_col[i]);
      }
    }

  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Filter Probe Aggr Kernel time GPU: " << time << endl;
  gpu_time[sg] += time;

};

void 
CPUGPUProcessing::call_pfilter_probe_aggr_CPU(QueryParams* params, int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  ColumnInfo* filter_col[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0};
  int *aggr_col[2] = {};

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    if (select_so_far == qo->select_probe[cm->lo_orderdate].size()) break;
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    filter_col[select_so_far + i] = column;
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

  for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->aggregation[cm->lo_orderdate][i];
    aggr_col[i] = column->col_ptr;
  }

  struct filterArgsCPU fargs = {
    (filter_col[0] != NULL) ? (filter_col[0]->col_ptr) : (NULL), 
    (filter_col[1] != NULL) ? (filter_col[1]->col_ptr) : (NULL),
    _compare1[0], _compare2[0], _compare1[1], _compare2[1],
    1, 1,
    (filter_col[0] != NULL) ? (params->map_filter_func_host[filter_col[0]]) : (NULL), 
    (filter_col[1] != NULL) ? (params->map_filter_func_host[filter_col[1]]) : (NULL)
  };

  struct probeArgsCPU pargs = {
    fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3],
    ht[0], ht[1], ht[2], ht[3], 
    _dim_len[0], _dim_len[1], _dim_len[2], _dim_len[3],
    _min_key[0], _min_key[1], _min_key[2], _min_key[3]
  };

  struct groupbyArgsCPU gargs = {
    aggr_col[0], aggr_col[1], NULL, NULL, NULL, NULL,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, params->mode_group, params->h_group_func
  };

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

    filter_probe_aggr_CPU(fargs, pargs, gargs, LEN, params->res, 0, segment_group_ptr);
  } else {

    struct offsetCPU offset = {
      h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4]
    };

    filter_probe_aggr_CPU2(offset, fargs, pargs, gargs, *h_total, params->res, 0);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  if (verbose) cout << "Filter Probe Aggr Kernel time CPU: " << time << endl;
  cpu_time[sg] += time;
};