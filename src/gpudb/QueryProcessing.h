#ifndef _QUERY_PROCESSING_H_
#define _QUERY_PROCESSING_H_

#include "QueryOptimizer.h"
#include "GPUProcessing.h"
#include "CPUProcessing.h"

#define NUM_QUERIES 4

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

  int total_val, mode_group;

  unordered_map<ColumnInfo*, int*> ht_CPU;
  unordered_map<ColumnInfo*, int*> ht_GPU;
  unordered_map<ColumnInfo*, int*> col_idx;

  int *ht_p, *ht_c, *ht_s, *ht_d;
  int *d_ht_p, *d_ht_c, *d_ht_s, *d_ht_d;

  int* res;
  int* d_res;

  vector<uint64_t> query_freq;
  unordered_map<ColumnInfo*, int> compare1;
  unordered_map<ColumnInfo*, int> compare2;
  unordered_map<ColumnInfo*, int> mode;

  chrono::high_resolution_clock::time_point begin_time;

  QueryProcessing(size_t cache_size, size_t _processing_size) {
    qo = new QueryOptimizer(cache_size, _processing_size);
    cm = qo->cm;
    begin_time = chrono::high_resolution_clock::now();
    query_freq.resize(NUM_QUERIES);
  }

  int generate_rand_query() {
    return rand() % NUM_QUERIES;
  }

  void runQuery(int query);

  void prepareQuery(int query);

  void endQuery(int query);

  void updateStatsQuery(int query);

  void processQuery(int query);


  void switch_device_fact(int** &off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table);

  void call_probe_GPU(int** &off_col, int* &d_total, int* h_total, int sg);

  void call_probe_CPU(int** &h_off_col, int* h_total, int sg);

  void call_probe_filter_GPU(int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far);

  void call_probe_filter_CPU(int** &h_off_col, int* h_total, int sg, int select_so_far);


  void switch_device_dim(int* &d_off_col, int* &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table);

  void call_build_GPU(int* &d_off_col, int* h_total, int sg, int table);

  void call_build_CPU(int* &h_off_col, int* h_total, int sg, int table);

  void call_build_filter_GPU(int* &d_off_col, int* &d_total, int* h_total, int sg, int table);

  void call_build_filter_CPU(int* &h_off_col, int* h_total, int sg, int table);


  void call_group_by_GPU(int** &off_col, int* h_total);

  void call_group_by_CPU(int** &h_off_col, int* h_total);

};

void 
QueryProcessing::switch_device_fact(int** &off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table) {
  if (mode == 0) { //CPU to GPU
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    assert(*h_total > 0);

    off_col = new int*[cm->TOT_TABLE]();
    //CubDebugExit(cudaMalloc((void**) &d_off_col, cm->TOT_TABLE * SEGMENT_SIZE * qo->segment_group_count[table][sg] * sizeof(int)));
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      //off_col[i] = d_off_col + i * SEGMENT_SIZE * qo->segment_group_count[table][sg];
      off_col[i] = cm->customCudaMalloc(SEGMENT_SIZE * qo->segment_group_count[table][sg]);
    }
    
    CubDebugExit(cudaMemcpy(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice));
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (h_off_col[i] != NULL) {
        CubDebugExit(cudaMemcpy(off_col[i], h_off_col[i], *h_total * sizeof(int), cudaMemcpyHostToDevice));
      } else
        off_col[i] = NULL;
    }
  } else { // GPU to CPU
    if (off_col == NULL) return;
    assert(off_col != NULL);

    h_off_col = new int*[cm->TOT_TABLE]();
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      //h_off_col[i] = new int[SEGMENT_SIZE * qo->segment_group_count[table][sg]];
      h_off_col[i] = cm->customMalloc(SEGMENT_SIZE * qo->segment_group_count[table][sg]);
    }

    CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      if (off_col[i] != NULL) {
        CubDebugExit(cudaMemcpy(h_off_col[i], off_col[i], *h_total * sizeof(int), cudaMemcpyDeviceToHost));
      } else
        h_off_col[i] = NULL;
    }
  }
  
}

void 
QueryProcessing::switch_device_dim(int* &d_off_col, int* &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table) {

  if (mode == 0) { //CPU to GPU
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    assert(*h_total > 0);

    //CubDebugExit(cudaMalloc((void**) &d_off_col, SEGMENT_SIZE * qo->segment_group_count[table][sg] * sizeof(int)));
    d_off_col = cm->customCudaMalloc(SEGMENT_SIZE * qo->segment_group_count[table][sg]);

    CubDebugExit(cudaMemcpy(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice));

    if (h_off_col != NULL) {
      CubDebugExit(cudaMemcpy(d_off_col, h_off_col, *h_total * sizeof(int), cudaMemcpyHostToDevice));
    } else
      d_off_col = NULL;

  } else { // GPU to CPU
    if (d_off_col == NULL) return;
    assert(d_off_col != NULL);

    //h_off_col = new int[SEGMENT_SIZE * qo->segment_group_count[table][sg]]; //initialize it to null
    h_off_col = cm->customMalloc(SEGMENT_SIZE * qo->segment_group_count[table][sg]);

    CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

    if (d_off_col != NULL) {
      CubDebugExit(cudaMemcpy(h_off_col, d_off_col, *h_total * sizeof(int), cudaMemcpyDeviceToHost));
    } else
      h_off_col = NULL;

  }
  
}

void 
QueryProcessing::call_probe_GPU(int** &off_col, int* &d_total, int* h_total, int sg) {
  int **off_col_out;
  //int *d_off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}, *fkey_col[4] = {}; //initialize it to null
  ColumnInfo* fkey[4] = {};

  int tile_items = 128*4;

  if(qo->joinGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize it to null
  //CubDebugExit(cudaMalloc((void**) &d_off_col_out, cm->TOT_TABLE * SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)));

  CubDebugExit(cudaMemset(d_total, 0, sizeof(int)));

  for (int i = 0; i < cm->TOT_TABLE; i++) {
    //off_col_out[i] = d_off_col_out + i * SEGMENT_SIZE * qo->segment_group_count[0][sg];
    off_col_out[i] = cm->customCudaMalloc(SEGMENT_SIZE * qo->segment_group_count[0][sg]);
  }

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    assert(column != NULL);
    int table_id = qo->fkey_pkey[column]->table_id;
    assert(table_id > 0);
    fkey[table_id - 1] = column;

    ColumnInfo* pkey = qo->fkey_pkey[column];
    if (col_idx.find(column) == col_idx.end()) {
      //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
      col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
      CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
    }
    assert(col_idx[column] != NULL);
    assert(ht_GPU[pkey] != NULL);
    fkey_idx[table_id - 1] = col_idx[column];
    ht[table_id - 1] = ht_GPU[pkey];
    _min_key[table_id - 1] = min_key[pkey];
    _dim_len[table_id - 1] = dim_len[pkey];
  }

  if (off_col == NULL) {

    // PROBE GPU 1

    // int start_offset = 0, idx_fkey, LEN;
    // for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

    //   int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
    //   start_offset = segment_number * SEGMENT_SIZE;

    //   for (int j = 0; j < 4; j++) {
    //     ColumnInfo* column = fkey[j];
    //     if (column == NULL) fkey_col[j] = NULL;
    //     else {
    //       idx_fkey = cm->segment_list[column->column_id][segment_number];
    //       assert(idx_fkey >= 0);
    //       fkey_col[j] = cm->gpuCache + idx_fkey * SEGMENT_SIZE;
    //     }
    //   }

    //   if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0) 
    //     LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
    //   else 
    //     LEN = SEGMENT_SIZE;

    //   probe_GPU<128,4><<<(LEN + tile_items - 1)/tile_items, 128>>>(
    //     NULL, NULL, 0, 0, 0, 0,
    //     fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN, 
    //     ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //     _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //     off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
    //     d_total, start_offset);

    //   CHECK_ERROR();

    // }

    //PROBE GPU 2

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpy(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice));

    probe_GPU2<128,4><<<(LEN+ tile_items - 1)/tile_items, 128>>>(
      NULL, NULL, NULL, NULL, NULL, cm->gpuCache,
      NULL, NULL, 0, 0, 0, 0,
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3],
      LEN, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total, 0, d_segment_group);

    CHECK_ERROR();

  } else {

    for (int i = 0; i < cm->TOT_TABLE; i++) assert(off_col[i] != NULL);

    assert(*h_total > 0);

    probe_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], cm->gpuCache, 
      NULL, NULL, 0, 0, 0, 0,
      fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3], 
      *h_total, ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total, 0, NULL);
    CHECK_ERROR();

  }

  // if (off_col != NULL) {
  //   assert(d_off_col != NULL);
  //   assert(off_col != NULL);
  //   d_off_col = NULL;
  //   //CubDebugExit(cudaFree(d_off_col));
  //   //free(off_col);
  //   off_col = NULL;
  //   //delete[] off_col;
  // }

  //d_off_col = d_off_col_out;

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

};

void 
QueryProcessing::call_probe_CPU(int** &h_off_col, int* h_total, int sg) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int out_total = 0;

  if(qo->joinCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to null

  for (int i = 0; i < cm->TOT_TABLE; i++) {
    off_col_out[i] = cm->customMalloc(SEGMENT_SIZE * qo->segment_group_count[0][sg]);
    //off_col_out[i] = new int[SEGMENT_SIZE * qo->segment_group_count[0][sg]];
  }

  for (int i = 0; i < qo->joinCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinCPUPipelineCol[sg][i];
    assert(column != NULL);
    int table_id = qo->fkey_pkey[column]->table_id;
    assert(table_id > 0);
    fkey_col[table_id - 1] = column->col_ptr;

    ColumnInfo* pkey = qo->fkey_pkey[column];
    assert(ht_CPU[pkey] != NULL);
    ht[table_id - 1] = ht_CPU[pkey];
    _min_key[table_id - 1] = min_key[pkey];
    _dim_len[table_id - 1] = dim_len[pkey];
  }

  if (h_off_col == NULL) {

    for (int i = 0; i < cm->TOT_TABLE; i++) {
      assert(off_col_out[i] != NULL);
    }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

    probe_CPU(NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, 0, 0, 0, 0,
      fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN,
      ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      &out_total, 0, segment_group_ptr);

    // int start_offset = 0, LEN;

    // for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

    //   int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
    //   start_offset = segment_number * SEGMENT_SIZE;

    //   if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0)
    //     LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
    //   else
    //     LEN = SEGMENT_SIZE;

    //   probe_CPU(NULL, NULL, NULL, NULL, NULL,
    //     NULL, NULL, 0, 0, 0, 0,
    //     fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN,
    //     ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
    //     _min_key[0], _min_key[1], _min_key[2], _min_key[3],
    //     off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
    //     &out_total, start_offset, NULL);

    // }

  } else {
      for (int i = 0; i < cm->TOT_TABLE; i++) {
        assert(h_off_col[i] != NULL);
        assert(off_col_out[i] != NULL);
      }

      assert(*h_total > 0);

      probe_CPU(h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4],
        NULL, NULL, 0, 0, 0, 0,
        fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], *h_total,
        ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
        _min_key[0], _min_key[1], _min_key[2], _min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        &out_total, 0, NULL);

  }

  // if (h_off_col != NULL) {
  //   for (int i = 0; i < cm->TOT_TABLE; i++)
  //     h_off_col[i] = NULL;
  //     //if (h_off_col[i] != NULL) delete[] h_off_col[i]; //free(h_off_col[i]);
  //   //free(h_off_col);
  //   h_off_col = NULL;
  //   //delete[] h_off_col;
  // }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;
};

void 
QueryProcessing::call_build_GPU(int* &d_off_col, int* h_total, int sg, int table) {
  int tile_items = 128*4;
  int* dimkey_idx, *key_col;

  ColumnInfo* column;
  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (ht_GPU[column] != NULL) {

    if (d_off_col == NULL) {

      // int start_offset, idx, LEN;
      // for (int i = 0; i < column->total_segment; i++) {

      //   start_offset = i * SEGMENT_SIZE;

      //   if (i == column->total_segment-1 && column->LEN % SEGMENT_SIZE != 0)
      //     LEN = column->LEN % SEGMENT_SIZE;
      //   else
      //     LEN = SEGMENT_SIZE;

      //   idx = cm->segment_list[column->column_id][i];
      //   assert(idx >= 0);
      //   key_col = cm->gpuCache + idx * SEGMENT_SIZE;

      //   build_GPU<128, 4><<<(LEN + tile_items - 1)/tile_items, 128>>>
      //     (NULL, 0, 0, 0, 
      //       key_col, NULL, LEN, 
      //       ht_GPU[column], dim_len[column], min_key[column], 1, start_offset);

      //   CHECK_ERROR();  

      // }

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      if (col_idx.find(column) == col_idx.end()) {
        //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
        col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
        CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
      }

      dimkey_idx = col_idx[column];

      short* d_segment_group;
      //CubDebugExit(cudaMalloc((void**) &d_segment_group, column->total_segment * sizeof(short)));
      d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
      CubDebugExit(cudaMemcpy(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice));

      build_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128>>>(
        NULL, cm->gpuCache, NULL, 0, 0, 0,
        dimkey_idx, NULL, LEN, 
        ht_GPU[column], dim_len[column], min_key[column],
        1, 0, d_segment_group); 
      CHECK_ERROR();

    } else {

      if (col_idx.find(column) == col_idx.end()) {
        //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
        col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
        CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
      }

      dimkey_idx = col_idx[column];

      build_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>(
        d_off_col, cm->gpuCache, NULL, 0, 0, 0,
        dimkey_idx, NULL, *h_total,
        ht_GPU[column], dim_len[column], min_key[column], 
        1, 0, NULL);

      CHECK_ERROR();

      //cudaFree(d_off_col);

    }
  }
};

void 
QueryProcessing::call_build_CPU(int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* column;
  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (ht_CPU[column] != NULL) {

    if (h_off_col == NULL) {

      int LEN;
      if (sg == qo->last_segment[table]) {
        LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
      } else { 
        LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
      }

      short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

      build_CPU(NULL, NULL, 0, 0, 0, column->col_ptr, NULL, LEN, 
        ht_CPU[column], dim_len[column], min_key[column], 1, 0, segment_group_ptr);

    } else {

      build_CPU(h_off_col, NULL, 0, 0, 0, column->col_ptr, NULL, *h_total, 
        ht_CPU[column], dim_len[column], min_key[column], 1, 0, NULL);

      //free(h_off_col);
    }

  }
};

void
QueryProcessing::call_build_filter_GPU(int* &d_off_col, int* &d_total, int* h_total, int sg, int table) {

  ColumnInfo* temp;
  int tile_items = 128*4;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table){
      temp = qo->join[i].second; break;
    }
  }
  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];
  
  //CubDebugExit(cudaMalloc((void**) &d_off_col, qo->segment_group_count[table][sg] * SEGMENT_SIZE * sizeof(int)));
  d_off_col = cm->customCudaMalloc(qo->segment_group_count[table][sg] * SEGMENT_SIZE);

  CubDebugExit(cudaMemset(d_total, 0, sizeof(int)));

  // int LEN;
  // for (int i = 0; i < qo->segment_group_count[table][sg]; i++){

  //   int segment_number = qo->segment_group[table][sg * column->total_segment + i];
  //   int start_offset = SEGMENT_SIZE * segment_number;
  //   int idx_key = cm->segment_list[column->column_id][segment_number];
  //   int* filter_col = cm->gpuCache + idx_key * SEGMENT_SIZE;

  //   if (segment_number == column->total_segment-1 && column->LEN % SEGMENT_SIZE != 0)
  //     LEN = column->LEN % SEGMENT_SIZE;
  //   else
  //     LEN = SEGMENT_SIZE;

  //   filter_GPU<128,4> <<<(LEN + tile_items - 1)/tile_items, 128>>> (
  //     filter_col, NULL, 
  //     compare1[column], compare2[column], 0, 0, mode[column], 0, 
  //     d_off_col, d_total, LEN, start_offset);
  //   CHECK_ERROR();
  // }

  int LEN;
  if (sg == qo->last_segment[table]) {
    LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
  } else { 
    LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
  }

  if (col_idx.find(column) == col_idx.end()) {
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
    col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
    CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
  }
  int* filter_idx = col_idx[column];

  short* d_segment_group;
  //CubDebugExit(cudaMalloc((void**) &d_segment_group, column->total_segment * sizeof(short)));
  d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(column->total_segment));
  short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);
  CubDebugExit(cudaMemcpy(d_segment_group, segment_group_ptr, qo->segment_group_count[table][sg] * sizeof(short), cudaMemcpyHostToDevice));

  filter_GPU2<128,4> <<<(LEN + tile_items - 1)/tile_items, 128>>>(
    NULL, 
    cm->gpuCache, filter_idx, NULL, 
    compare1[column], compare2[column], 0, 0, mode[column], 0,
    d_off_col, d_total, LEN, 0, d_segment_group);

  CHECK_ERROR();

  CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

}

void
QueryProcessing::call_build_filter_CPU(int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* temp;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      temp = qo->join[i].second; break;
    }
  }

  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];
  int* filter_col = column->col_ptr;

  //h_off_col = new int[qo->segment_group_count[table][sg] * SEGMENT_SIZE];
  h_off_col = cm->customMalloc(qo->segment_group_count[table][sg] * SEGMENT_SIZE);

  // int start_offset, LEN;

  // for (int i = 0; i < qo->segment_group_count[table][sg]; i++){
  //   int segment_number = qo->segment_group[table][sg * column->total_segment + i];
  //   start_offset = SEGMENT_SIZE * segment_number;

  //   if (segment_number == column->total_segment-1 && column->LEN % SEGMENT_SIZE != 0)
  //     LEN = column->LEN % SEGMENT_SIZE;
  //   else
  //     LEN = SEGMENT_SIZE;

  //   filter_CPU(NULL, filter_col, NULL, 
  //     compare1[column], compare2[column], 0, 0, mode[column], 0,
  //     h_off_col, h_total, LEN, 
  //     start_offset, NULL);
  // }

  int LEN;
  if (sg == qo->last_segment[table]) {
    LEN = (qo->segment_group_count[table][sg] - 1) * SEGMENT_SIZE + column->LEN % SEGMENT_SIZE;
  } else { 
    LEN = qo->segment_group_count[table][sg] * SEGMENT_SIZE;
  }

  short* segment_group_ptr = qo->segment_group[table] + (sg * column->total_segment);

  filter_CPU(NULL, filter_col, NULL, 
    compare1[column], compare2[column], 0, 0, mode[column], 0,
    h_off_col, h_total, LEN,
    0, segment_group_ptr);

}

void
QueryProcessing::call_probe_filter_GPU(int** &off_col, int* &d_total, int* h_total, int sg, int select_so_far) {
  int tile_items = 128*4;
  int **off_col_out;
  //int *d_off_col_out;
  int *filter_col[2] = {}, *filter_idx[2] = {};
  ColumnInfo *filter[2] = {};
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};

  if (qo->selectGPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to NULL
  //CubDebugExit(cudaMalloc((void**) &d_off_col_out, cm->TOT_TABLE * SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)));
  for (int i = 0; i < cm->TOT_TABLE; i++) {
    //off_col_out[i] = d_off_col_out + i * SEGMENT_SIZE * qo->segment_group_count[0][sg];
    off_col_out[i] = cm->customCudaMalloc(SEGMENT_SIZE * qo->segment_group_count[0][sg]);
  }

  CubDebugExit(cudaMemset(d_total, 0, sizeof(int)));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    assert(select_so_far + i < qo->select_probe[cm->lo_orderdate].size());
    assert(column != NULL);
    filter[select_so_far + i] = column;
    if (col_idx.find(column) == col_idx.end()) {
      //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[filter[1]], cm->cache_total_seg * sizeof(int)));
      col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
      CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
    }
    filter_idx[select_so_far + i] = col_idx[column];
    _compare1[select_so_far + i] = compare1[column];
    _compare2[select_so_far + i] = compare2[column];
    _mode[select_so_far + i] = mode[column];
  }

  if (off_col == NULL) {
    assert(filter[0] != NULL);

    // int start_offset, idx_fil, LEN;
    // for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

    //   int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
    //   start_offset = segment_number * SEGMENT_SIZE;

    //   for (int j = 0; j < 2; j++) {
    //     if (filter[j] == NULL) filter_col[j] = NULL;
    //     else {
    //       idx_fil = cm->segment_list[filter[j]->column_id][segment_number];
    //       assert(idx_fil >= 0);
    //       filter_col[j] = cm->gpuCache + idx_fil * SEGMENT_SIZE;
    //     }
    //   }

    //   if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0)
    //     LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
    //   else
    //     LEN = SEGMENT_SIZE;

    //   filter_GPU<128,4><<<(LEN + tile_items - 1)/tile_items, 128>>>
    //     (filter_col[0], filter_col[1], 
    //       _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1], 
    //       off_col_out[0], d_total, LEN, start_offset);
    //   CHECK_ERROR();
    // }

    int LEN;
    if (sg == qo->last_segment[0]) {
      LEN = (qo->segment_group_count[0][sg] - 1) * SEGMENT_SIZE + cm->lo_orderdate->LEN % SEGMENT_SIZE;
    } else { 
      LEN = qo->segment_group_count[0][sg] * SEGMENT_SIZE;
    }

    short* d_segment_group;
    d_segment_group = reinterpret_cast<short*>(cm->customCudaMalloc(cm->lo_orderdate->total_segment));
    short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);
    CubDebugExit(cudaMemcpy(d_segment_group, segment_group_ptr, qo->segment_group_count[0][sg] * sizeof(short), cudaMemcpyHostToDevice));

    filter_GPU2<128,4><<<(LEN + tile_items - 1)/tile_items, 128>>>(
      NULL, 
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], d_total, LEN, 0, d_segment_group);

    CHECK_ERROR();


  } else {

    assert(off_col[0] != NULL);

    filter_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>
      (off_col[0], 
      cm->gpuCache, filter_idx[0], filter_idx[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], d_total,  *h_total, 0, NULL);

    CHECK_ERROR();
  }

  // if (off_col != NULL) {
  //   assert(d_off_col != NULL);
  //   //free(off_col);
  //   //delete[] off_col;
  //   off_col = NULL;
  //   //CubDebugExit(cudaFree(d_off_col));
  //   d_off_col = NULL;
  // }

  off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    off_col[i] = off_col_out[i];

  //d_off_col = d_off_col_out;
  CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
}

void
QueryProcessing::call_probe_filter_CPU(int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int **off_col_out;
  int *filter_col[2] = {};
  int out_total = 0;
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};

  if (qo->selectCPUPipelineCol[sg].size() == 0) return;

  off_col_out = new int*[cm->TOT_TABLE](); //initialize to NULL
  for (int i = 0; i < cm->TOT_TABLE; i++) {
    //off_col_out[i] = new int[SEGMENT_SIZE * qo->segment_group_count[0][sg]];
    off_col_out[i] = cm->customMalloc(SEGMENT_SIZE * qo->segment_group_count[0][sg]);
  }

  for (int i = 0; i < qo->selectCPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->selectCPUPipelineCol[sg][i];
    assert(select_so_far + i < qo->select_probe[cm->lo_orderdate].size());
    assert(column != NULL);
    filter_col[select_so_far + i] = column->col_ptr;
    _compare1[select_so_far + i] = compare1[column];
    _compare2[select_so_far + i] = compare2[column];
    _mode[select_so_far + i] = mode[column];

  }

  if (h_off_col == NULL) {
    // int start_offset, LEN;
    // for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

    //   int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
    //   start_offset = segment_number * SEGMENT_SIZE;

    //   if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0)
    //     LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
    //   else
    //     LEN = SEGMENT_SIZE;

    //   filter_CPU(NULL, filter_col[0], filter_col[1], 
    //     _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
    //     off_col_out[0], &out_total, LEN,
    //     start_offset, NULL);
    // }

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

    filter_CPU(h_off_col[0], filter_col[0], filter_col[1], 
      _compare1[0], _compare2[0], _compare1[1], _compare2[1], _mode[0], _mode[1],
      off_col_out[0], &out_total, *h_total,
      0, NULL);
  }

  // if (h_off_col != NULL) {
  //   for (int i = 0; i < cm->TOT_TABLE; i++)
  //     h_off_col[i] = NULL;
  //     //if (h_off_col[i] != NULL) delete[] h_off_col[i]; //free(h_off_col[i]);
  //   //free(h_off_col);
  //   h_off_col = NULL;
  //   //delete[] h_off_col;
  // }

  h_off_col = off_col_out;

  for (int i = 0; i < cm->TOT_TABLE; i++)
    h_off_col[i] = off_col_out[i];

  *h_total = out_total;
}

void
QueryProcessing::call_group_by_GPU(int** &off_col, int* h_total) {
  int _min_val[4] = {0}, _unique_val[4] = {0};
  int *aggr_idx[2] = {}, *group_idx[4] = {};
  int tile_items = 128 * 4;

  if (qo->groupby_probe[cm->lo_orderdate].size() == 0) return;

  for (int i = 0; i < qo->groupby_probe[cm->lo_orderdate].size(); i++) {
    ColumnInfo* column = qo->groupby_probe[cm->lo_orderdate][i];
    if (col_idx.find(column) == col_idx.end()) {
      //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
      col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
      CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
    }
    aggr_idx[i] = col_idx[column];
  }

  unordered_map<ColumnInfo*, vector<ColumnInfo*>>::iterator it;
  for (it = qo->groupby_build.begin(); it != qo->groupby_build.end(); it++) {
    if (it->second.size() > 0) {
      ColumnInfo* column = it->second[0];
      ColumnInfo* column_key = it->first;
      if (col_idx.find(column) == col_idx.end()) {
        //CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
        col_idx[column] = cm->customCudaMalloc(cm->cache_total_seg);
        CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
      }
      group_idx[column_key->table_id - 1] = col_idx[column];
      _min_val[column_key->table_id - 1] = min_val[column_key];
      _unique_val[column_key->table_id - 1] = unique_val[column_key];
    }
  }

  groupByGPU<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>(
    off_col[0], off_col[1], off_col[2], off_col[3], off_col[4], 
    cm->gpuCache, aggr_idx[0], aggr_idx[1], group_idx[0], group_idx[1], group_idx[2], group_idx[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3], _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    total_val, *h_total, d_res, mode_group);

  CHECK_ERROR();

  //CubDebugExit(cudaMemcpy(res, d_res, total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));

}

void
QueryProcessing::call_group_by_CPU(int** &h_off_col, int* h_total) {
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
      _min_val[column_key->table_id - 1] = min_val[column_key];
      _unique_val[column_key->table_id - 1] = unique_val[column_key];
    }
  }

  groupByCPU(h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4], 
    aggr_col[0], aggr_col[1], group_col[0], group_col[1], group_col[2], group_col[3],
    _min_val[0], _min_val[1], _min_val[2], _min_val[3], _unique_val[0], _unique_val[1], _unique_val[2], _unique_val[3],
    total_val, *h_total, res, mode_group);

}

void
QueryProcessing::runQuery(int query) {

  chrono::high_resolution_clock::time_point st, finish;
  chrono::duration<double> diff;
  
  st = chrono::high_resolution_clock::now();

  for (int i = 0; i < qo->join.size(); i++) {

    for (int sg = 0; sg < 2; sg++) {

      int *h_off_col = NULL, *d_off_col = NULL;

      //cout << qo->join[i].second->column_name << endl;

      if (qo->segment_group_count[qo->join[i].second->table_id][sg] > 0) {

        int h_total = 0, *d_total = NULL;
        d_total = cm->customCudaMalloc(1);

        printf("sg = %d\n", sg);

        if (sg == 0) {
          call_build_filter_CPU(h_off_col, &h_total, sg, qo->join[i].second->table_id);

          call_build_CPU(h_off_col, &h_total, sg, qo->join[i].second->table_id);

          if (qo->joinGPUcheck[i]) {
            switch_device_dim(d_off_col, h_off_col, d_total, &h_total, sg, 0, qo->join[i].second->table_id);
            call_build_GPU(d_off_col, &h_total, sg, qo->join[i].second->table_id);
          }

          //delete[] h_off_col;
          //h_off_col = NULL;

        } else {
          call_build_filter_GPU(d_off_col, d_total, &h_total, sg, qo->join[i].second->table_id);

          call_build_GPU(d_off_col, &h_total, sg, qo->join[i].second->table_id);

          switch_device_dim(d_off_col, h_off_col, d_total, &h_total, sg, 1, qo->join[i].second->table_id);
          call_build_CPU(h_off_col, &h_total, sg, qo->join[i].second->table_id);

          //CubDebugExit(cudaFree(d_off_col));
          //d_off_col = NULL;
          
        }

      }

      //CubDebugExit(cudaFree(d_total));
    }
  }

  

  for (int sg = 0; sg < 64; sg++) {

    int** h_off_col = NULL, **off_col = NULL;

    if (qo->segment_group_count[0][sg] > 0) {

      int*d_total = NULL;
      int h_total = 0;
      
      d_total = cm->customCudaMalloc(1);

      printf("sg = %d\n", sg);

      call_probe_filter_CPU(h_off_col, &h_total, sg, 0);

      if (qo->selectCPUPipelineCol[sg].size() > 0 && (qo->joinGPUPipelineCol[sg].size() > 0 || qo->selectGPUPipelineCol[sg].size() > 0 || qo->groupbyGPUPipelineCol[sg].size() > 0)) 
        switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 0, 0);

      call_probe_filter_GPU(off_col, d_total, &h_total, sg, qo->selectCPUPipelineCol[sg].size());

      call_probe_GPU(off_col, d_total, &h_total, sg);

      //printf("%d\n", h_total);

      if ((qo->selectGPUPipelineCol[sg].size() > 0 || qo->joinGPUPipelineCol[sg].size() > 0) && (qo->joinCPUPipelineCol[sg].size() > 0 || qo->groupbyCPUPipelineCol[sg].size() > 0))
        switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 1, 0);

      call_probe_CPU(h_off_col, &h_total, sg);

      if (qo->groupGPUcheck) {
        if (qo->groupbyGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() > 0)
            switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 0, 0);
          call_group_by_GPU(off_col, &h_total);
        } else {
          call_group_by_CPU(h_off_col, &h_total);
        }
      } else {
        call_group_by_CPU(h_off_col, &h_total);
      }

      // call_probe_filter_GPU(off_col, d_total, &h_total, sg, 0);

      // if (qo->selectGPUPipelineCol[sg].size() > 0 && (qo->joinCPUPipelineCol[sg].size() > 0 || qo->selectCPUPipelineCol[sg].size() > 0 || qo->groupbyCPUPipelineCol[sg].size() > 0))
      // switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 1, 0);

      // call_probe_filter_CPU(h_off_col, &h_total, sg, qo->selectGPUPipelineCol[sg].size());

      // call_probe_CPU(h_off_col, &h_total, sg);

      // if ((qo->selectCPUPipelineCol[sg].size() > 0 || qo->joinCPUPipelineCol[sg].size() > 0) && (qo->joinGPUPipelineCol[sg].size() > 0 || qo->groupbyGPUPipelineCol[sg].size() > 0))
      //   switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 0, 0);

      // call_probe_GPU(off_col, d_total, &h_total, sg);

      // if (qo->groupGPUcheck) {
      //   if (qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //     call_group_by_GPU(off_col, &h_total);
      //   } else {
      //     if (qo->joinGPUPipelineCol[sg].size() > 0)
      //       switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 1, 0);
      //     call_group_by_CPU(h_off_col, &h_total);
      //   }
      // } else {
      //   if (qo->joinGPUPipelineCol[sg].size() > 0)
      //     switch_device_fact(off_col, h_off_col, d_total, &h_total, sg, 1, 0);
      //   call_group_by_CPU(h_off_col, &h_total);
      // }

      // for (int i = 0; i < h_total2; i++) {
      //   bool found = false;
      //   for (int j = 0; j < h_total; j++) {
      //     if (h_off_col2[0][i] == h_off_col[0][j]) {
      //       found = true;
      //       break;
      //     }
      //   }
      //   if (!found)
      //     printf("error %d\n", h_off_col2[0][i]);
      // }

      // if (h_off_col != NULL) {
      //   for (int i = 0; i < cm->TOT_TABLE; i++) h_off_col[i] = NULL; //delete[] h_off_col[i];
      //   h_off_col = NULL; // delete[] h_off_col;
      // }
      //if (off_col != NULL) delete[] off_col;
      //off_col = NULL;
      //if (d_off_col != NULL) CubDebugExit(cudaFree(d_ofd_off_col = NULL;

      //CubDebugExit(cudaFree(d_total));

    }

  }

  // int* resGPU = new int [total_val * 6]();
  int* resGPU = cm->customMalloc(total_val * 6);
  CubDebugExit(cudaMemcpy(resGPU, d_res, total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));

  merge(res, resGPU, total_val);
  //delete[] resGPU;

  finish = chrono::high_resolution_clock::now();

  diff = finish - st;

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i<total_val; i++) {
    if (res[6*i+4] != 0) {
      cout << res[6*i] << " " << res[6*i+1] << " " << res[6*i+2] << " " << res[6*i+3] << " " << reinterpret_cast<unsigned long long*>(&res[6*i+4])[0]  << endl;
      res_count++;
    }
  }
  cout << "Res count = " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

}

void 
QueryProcessing::processQuery(int query) {

  qo->parseQuery(query);
  updateStatsQuery(query);

  prepareQuery(query);

  runQuery(query);

  endQuery(query);

};

void
QueryProcessing::endQuery(int query) {
  min_key.clear();
  min_val.clear();
  unique_val.clear();
  dim_len.clear();

  // if (ht_p != NULL) {
  //   free(ht_p); ht_p = NULL;
  // }
  // if (ht_c != NULL) {
  //   free(ht_c); ht_c = NULL;
  // }
  // if (ht_s != NULL) {
  //   free(ht_s); ht_s = NULL;
  // }
  // if (ht_d != NULL) {
  //   free(ht_d); ht_d = NULL;
  // }
  // if (d_ht_p != NULL) {
  //   CubDebugExit(g_allocator.DeviceFree(d_ht_p)); d_ht_p = NULL;
  // }
  // if (d_ht_c != NULL) {
  //   CubDebugExit(g_allocator.DeviceFree(d_ht_c)); d_ht_c = NULL;
  // }
  // if (d_ht_s != NULL) {
  //   CubDebugExit(g_allocator.DeviceFree(d_ht_s)); d_ht_s = NULL;
  // }
  // if (d_ht_d != NULL) {
  //   CubDebugExit(g_allocator.DeviceFree(d_ht_d)); d_ht_d = NULL;
  // }

  ht_p = NULL;
  ht_c = NULL;
  ht_s = NULL;
  ht_d = NULL;
  d_ht_p = NULL;
  d_ht_c = NULL;
  d_ht_s = NULL;
  d_ht_d = NULL;
  
  // unordered_map<ColumnInfo*, int*>::iterator it;
  // for (it = col_idx.begin(); it != col_idx.end(); it++) {
  //   //CubDebugExit(g_allocator.DeviceFree(it->second));
  //   it->second = NULL;
  // }

  ht_CPU.clear();
  ht_GPU.clear();
  col_idx.clear();

  compare1.clear();
  compare2.clear();
  mode.clear();

  //delete[] res;
  res = NULL;
  //CubDebugExit(g_allocator.DeviceFree(d_res));
  d_res = NULL;

  qo->clearVector();

  cm->gpuPointer = 0;
  cm->cpuPointer = 0;
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

    mode[cm->d_year] = 0;
    compare1[cm->d_year] = 1993;
    mode[cm->lo_discount] = 1;
    compare1[cm->lo_discount] = 1;
    compare2[cm->lo_discount] = 3;
    mode[cm->lo_quantity] = 3;
    compare1[cm->lo_quantity] = 25;
    mode_group = 2;

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

    total_val = 1;

    ht_p = NULL;
    ht_c = NULL;
    ht_s = NULL;
    //ht_d = (int*) malloc (2 * dim_len[cm->d_datekey] * sizeof(int));
    ht_d = cm->customMalloc(2 * dim_len[cm->d_datekey]);

    memset(ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));

    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    d_ht_d = cm->customCudaMalloc(2 * dim_len[cm->d_datekey]);
    d_ht_p = NULL;
    d_ht_c = NULL;
    d_ht_s = NULL;

    CubDebugExit(cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 1) {

    mode[cm->s_region] = 0;
    compare1[cm->s_region] = 1;
    mode[cm->p_category] = 0;
    compare1[cm->p_category] = 1;
    mode_group = 0;

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

    //ht_p = (int*)malloc(2 * dim_len[cm->p_partkey] * sizeof(int));
    ht_p = cm->customMalloc(2 * dim_len[cm->p_partkey]);
    ht_c = NULL;
    //ht_s = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_s = cm->customMalloc(2 * dim_len[cm->s_suppkey]);
    //ht_d = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));
    ht_d = cm->customMalloc(2 * dim_len[cm->d_datekey]);

    memset(ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    memset(ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));

    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    d_ht_p = cm->customCudaMalloc(2 * dim_len[cm->p_partkey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    d_ht_s = cm->customCudaMalloc(2 * dim_len[cm->s_suppkey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    d_ht_d = cm->customCudaMalloc(2 * dim_len[cm->d_datekey]);
    d_ht_c = NULL;

    CubDebugExit(cudaMemset(d_ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 2) {

    mode[cm->c_region] = 0;
    compare1[cm->c_region] = 2;
    mode[cm->s_region] = 0;
    compare1[cm->s_region] = 2;
    mode[cm->d_year] = 1;
    compare1[cm->d_year] = 1992;
    compare2[cm->d_year] = 1997;
    mode_group = 0;

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

    ht_p = NULL;
    //ht_c = (int*)malloc(2 * dim_len[cm->c_custkey] * sizeof(int));
    ht_c = cm->customMalloc(2 * dim_len[cm->c_custkey]);
    //ht_s = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_s = cm->customMalloc(2 * dim_len[cm->s_suppkey]);
    //ht_d = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));
    ht_d = cm->customMalloc(2 * dim_len[cm->d_datekey]);

    memset(ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int));
    memset(ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));

    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * dim_len[cm->c_custkey] * sizeof(int)));
    d_ht_c = cm->customCudaMalloc(2 * dim_len[cm->c_custkey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    d_ht_s = cm->customCudaMalloc(2 * dim_len[cm->s_suppkey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    d_ht_d = cm->customCudaMalloc(2 * dim_len[cm->d_datekey]);
    d_ht_p = NULL;

    CubDebugExit(cudaMemset(d_ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 3) {

    mode[cm->c_region] = 0;
    compare1[cm->c_region] = 1;
    mode[cm->s_region] = 0;
    compare1[cm->s_region] = 1;
    mode[cm->p_mfgr] = 2;
    compare1[cm->p_mfgr] = 0;
    compare2[cm->p_mfgr] = 1;
    mode_group = 1;

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

    //ht_p = (int*)malloc(2 * dim_len[cm->p_partkey] * sizeof(int));
    ht_p = cm->customMalloc(2 * dim_len[cm->p_partkey]);
    //ht_c = (int*)malloc(2 * dim_len[cm->c_custkey] * sizeof(int));
    ht_c = cm->customMalloc(2 * dim_len[cm->c_custkey]);
    //ht_s = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_s = cm->customMalloc(2 * dim_len[cm->s_suppkey]);
    //ht_d = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));
    ht_d = cm->customMalloc(2 * dim_len[cm->d_datekey]);

    memset(ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    memset(ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    memset(ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int));

    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    d_ht_p = cm->customCudaMalloc(2 * dim_len[cm->p_partkey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    d_ht_s = cm->customCudaMalloc(2 * dim_len[cm->s_suppkey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    d_ht_d = cm->customCudaMalloc(2 * dim_len[cm->d_datekey]);
    //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * dim_len[cm->c_custkey] * sizeof(int)));
    d_ht_c = cm->customCudaMalloc(2 * dim_len[cm->c_custkey]);

    CubDebugExit(cudaMemset(d_ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int)));

  }

  ht_GPU[cm->p_partkey] = d_ht_p;
  ht_GPU[cm->c_custkey] = d_ht_c;
  ht_GPU[cm->s_suppkey] = d_ht_s;
  ht_GPU[cm->d_datekey] = d_ht_d;

  ht_CPU[cm->p_partkey] = ht_p;
  ht_CPU[cm->c_custkey] = ht_c;
  ht_CPU[cm->s_suppkey] = ht_s;
  ht_CPU[cm->d_datekey] = ht_d;

  int res_array_size = total_val * 6;
  //res = new int[res_array_size];
  res = cm->customMalloc(res_array_size);
  memset(res, 0, res_array_size * sizeof(int));
     
  //CubDebugExit(g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int)));
  d_res = cm->customCudaMalloc(res_array_size);
  CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
}
#endif