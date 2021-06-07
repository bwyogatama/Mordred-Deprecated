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

  vector<int> query_freq;
  unordered_map<ColumnInfo*, int> compare1;
  unordered_map<ColumnInfo*, int> compare2;
  unordered_map<ColumnInfo*, int> mode;

  chrono::high_resolution_clock::time_point begin_time;

  QueryProcessing() {
    qo = new QueryOptimizer();
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

  void processQuery() {
    int query = 3;
    qo->parseQuery(query);
    prepareQuery(query);
    updateStatsQuery(query);
    runQuery(query);
    endQuery(query);
  };

  void switch_device_fact(int** &off_col, int* &d_off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table);

  void call_probe_GPU(int** &off_col, int* &d_off_col, int* &d_total, int* h_total, int sg);

  void call_probe_CPU(int** &h_off_col, int* h_total, int sg);

  void call_probe_filter_GPU(int** &off_col, int* &d_off_col, int* &d_total, int* h_total, int sg, int select_so_far);

  void call_probe_filter_CPU(int** &h_off_col, int* h_total, int sg, int select_so_far);


  void switch_device_dim(int* &d_off_col, int* &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table);

  void call_build_GPU(int* &d_off_col, int* h_total, int table);

  void call_build_CPU(int* &h_off_col, int* h_total, int table);

  void call_build_filter_GPU(int* &d_off_col, int* &d_total, int* h_total, int sg, int table);

  void call_build_filter_CPU(int* &h_off_col, int* h_total, int sg, int table);


  void call_group_by_GPU(int** &off_col, int* h_total);

  void call_group_by_CPU(int** &h_off_col, int* h_total);
};

void
QueryProcessing::endQuery(int query) {
  min_key.clear();
  min_val.clear();
  unique_val.clear();
  dim_len.clear();

  if (ht_p != NULL) free(ht_p);
  if (ht_c != NULL) free(ht_c);
  if (ht_s != NULL) free(ht_s);
  if (ht_d != NULL) free(ht_d);
  if (d_ht_p != NULL) g_allocator.DeviceFree(d_ht_p);
  if (d_ht_c != NULL) g_allocator.DeviceFree(d_ht_c);
  if (d_ht_s != NULL) g_allocator.DeviceFree(d_ht_s);
  if (d_ht_d != NULL) g_allocator.DeviceFree(d_ht_d);
  
  unordered_map<ColumnInfo*, int*>::iterator it;
  for (it = col_idx.begin(); it != col_idx.end(); it++) {
    g_allocator.DeviceFree(it->second);
  }

  ht_CPU.clear();
  ht_GPU.clear();
  col_idx.clear();

  delete[] res;
  g_allocator.DeviceFree(d_res);

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
QueryProcessing::switch_device_fact(int** &off_col, int* &d_off_col, int** &h_off_col, int* &d_total, int* h_total, int sg, int mode, int table) {
  if (mode == 0) { //CPU to GPU
    if (h_off_col == NULL) return;
    assert(h_off_col != NULL);
    assert(*h_total > 0);

    //off_col = (int**) malloc(cm->TOT_TABLE * sizeof(int*));
    off_col = new int*[cm->TOT_TABLE]();
    CubDebugExit(cudaMalloc((void**) &d_off_col, cm->TOT_TABLE * SEGMENT_SIZE * qo->segment_group_count[table][sg] * sizeof(int)));
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      off_col[i] = d_off_col + i * SEGMENT_SIZE * qo->segment_group_count[table][sg];
    }
    
    CubDebugExit(cudaMalloc((void**) &d_total, sizeof(int)));
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

    //h_off_col = (int**) malloc(cm->TOT_TABLE * sizeof(int*));
    h_off_col = new int*[cm->TOT_TABLE]();
    for (int i = 0; i < cm->TOT_TABLE; i++) {
      //h_off_col[i] = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[table][sg] * sizeof(int));
      h_off_col[i] = new int[SEGMENT_SIZE * qo->segment_group_count[table][sg]];
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

    CubDebugExit(cudaMalloc((void**) &d_off_col, SEGMENT_SIZE * qo->segment_group_count[table][sg] * sizeof(int)));

    CubDebugExit(cudaMalloc((void**) &d_total, sizeof(int)));
    CubDebugExit(cudaMemcpy(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice));

    if (h_off_col != NULL) {
      CubDebugExit(cudaMemcpy(d_off_col, h_off_col, *h_total * sizeof(int), cudaMemcpyHostToDevice));
    } else
      d_off_col = NULL;

  } else { // GPU to CPU
    if (d_off_col == NULL) return;
    assert(d_off_col != NULL);

    //h_off_col = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[table][sg] * sizeof(int));
    h_off_col = new int[SEGMENT_SIZE * qo->segment_group_count[table][sg]]; //initialize it to null

    CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

    if (d_off_col != NULL) {
      CubDebugExit(cudaMemcpy(h_off_col, d_off_col, *h_total * sizeof(int), cudaMemcpyDeviceToHost));
    } else
      h_off_col = NULL;


  }
  
}

void 
QueryProcessing::call_probe_GPU(int** &off_col, int* &d_off_col, int* &d_total, int* h_total, int sg) {
  int **off_col_out;
  int *d_off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_idx[4] = {}, *fkey_col[4] = {}; //initialize it to null
  ColumnInfo* fkey[4] = {};
  int start_offset, idx_fkey, LEN;

  int tile_items = 128*4;

  if(qo->joinGPUPipelineCol[sg].size() == 0) return;

  //off_col_out = (int**) malloc(cm->TOT_TABLE * sizeof(int*));
  off_col_out = new int*[cm->TOT_TABLE] (); //initialize it to null
  CubDebugExit(cudaMalloc((void**) &d_off_col_out, cm->TOT_TABLE * SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)));

  CubDebugExit(cudaMalloc((void **)&d_total, sizeof(int)));
  CubDebugExit(cudaMemset(d_total, 0, sizeof(int)));

  for (int i = 0; i < cm->TOT_TABLE; i++) {
    off_col_out[i] = d_off_col_out + i * SEGMENT_SIZE * qo->segment_group_count[0][sg];
  }

  for (int i = 0; i < qo->joinGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->joinGPUPipelineCol[sg][i];
    assert(column != NULL);
    int table_id = qo->fkey_pkey[column]->table_id;
    assert(table_id > 0);
    fkey[table_id - 1] = column;

    ColumnInfo* pkey = qo->fkey_pkey[column];
    if (col_idx.find(column) == col_idx.end()) {
      CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
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

    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;

      //printf("segment number = %d\n", segment_number);

      for (int j = 0; j < 4; j++) {
        ColumnInfo* column = fkey[j];
        if (column == NULL) fkey_col[j] = NULL;
        else {
          //cout << column->column_name << endl;
          idx_fkey = cm->segment_list[column->column_id][segment_number];
          //printf("%d\n", idx_fkey);
          assert(idx_fkey >= 0);
          fkey_col[j] = cm->gpuCache + idx_fkey * SEGMENT_SIZE;
        }
      }

      if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0) 
        LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
      else 
        LEN = SEGMENT_SIZE;

      probe_GPU<128,4><<<(LEN + tile_items - 1)/tile_items, 128>>>(
        fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN, 
        ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
        _min_key[0], _min_key[1], _min_key[2], _min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        d_total, start_offset);

      CHECK_ERROR();

    }

  } else {

    for (int i = 0; i < cm->TOT_TABLE; i++) assert(off_col[i] != NULL);

    assert(*h_total > 0);

    probe_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>(
      off_col[0], off_col[1], off_col[2], off_col[3], off_col[4],
      cm->gpuCache, fkey_idx[0], fkey_idx[1], fkey_idx[2], fkey_idx[3], *h_total, 
      ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
      _min_key[0], _min_key[1], _min_key[2], _min_key[3],
      off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
      d_total, 0, NULL);
    CHECK_ERROR();

  }

  if (off_col != NULL) {
    assert(d_off_col != NULL);
    assert(off_col != NULL);
    cudaFree(d_off_col);
    //free(off_col);
    delete[] off_col;
  }

  d_off_col = d_off_col_out;
  off_col = off_col_out;
  CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

};

void 
QueryProcessing::call_probe_CPU(int** &h_off_col, int* h_total, int sg) {
  int **off_col_out;
  int _min_key[4] = {0}, _dim_len[4] = {0};
  int *ht[4] = {}, *fkey_col[4] = {};
  int start_offset, out_total, LEN;

  if(qo->joinCPUPipelineCol[sg].size() == 0) return;

  //off_col_out = (int**) malloc(cm->TOT_TABLE * sizeof(int*));
  //off_col_out[0] = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)); //lo_off
  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to null

  for (int i = 0; i < cm->TOT_TABLE; i++) {
    off_col_out[i] = new int[SEGMENT_SIZE * qo->segment_group_count[0][sg]];
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

  out_total = 0;

  if (h_off_col == NULL) {
    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;

      if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0)
        LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
      else
        LEN = SEGMENT_SIZE;

      probe_CPU(NULL, NULL, NULL, NULL, NULL,
        fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], LEN,
        ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
        _min_key[0], _min_key[1], _min_key[2], _min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        start_offset, &out_total, NULL);

    }
  } else {

      probe_CPU(h_off_col[0], h_off_col[1], h_off_col[2], h_off_col[3], h_off_col[4],
        fkey_col[0], fkey_col[1], fkey_col[2], fkey_col[3], *h_total,
        ht[0], _dim_len[0], ht[1], _dim_len[1], ht[2], _dim_len[2], ht[3], _dim_len[3],
        _min_key[0], _min_key[1], _min_key[2], _min_key[3],
        off_col_out[0], off_col_out[1], off_col_out[2], off_col_out[3], off_col_out[4],
        start_offset, &out_total, NULL);
  }

  if (h_off_col != NULL) {
    for (int i = 0; i < cm->TOT_TABLE; i++)
      if (h_off_col[i] != NULL) delete[] h_off_col[i]; //free(h_off_col[i]);
    //free(h_off_col);
    delete[] h_off_col;
  }

  h_off_col = off_col_out;
  *h_total = out_total;
};

void 
QueryProcessing::call_build_GPU(int* &d_off_col, int* h_total, int table) {
  int tile_items = 128*4;
  int start_offset, idx, LEN;
  int* dimkey_idx, *key_col;

  ColumnInfo* column;
  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table)
      column = qo->join[i].second;
  }

  if (ht_GPU[column] != NULL) {

    if (d_off_col == NULL) {

      for (int i = 0; i < column->total_segment; i++) {

        start_offset = i * SEGMENT_SIZE;

        if (i == column->total_segment-1 && column->LEN % SEGMENT_SIZE != 0)
          LEN = column->LEN % SEGMENT_SIZE;
        else
          LEN = SEGMENT_SIZE;

        idx = cm->segment_list[column->column_id][i];
        assert(idx >= 0);
        key_col = cm->gpuCache + idx * SEGMENT_SIZE;

        build_GPU<128, 4><<<(LEN + tile_items - 1)/tile_items, 128>>>
          (key_col, NULL, LEN, ht_GPU[column], 
            dim_len[column], min_key[column], start_offset, 1);

        CHECK_ERROR();  

      }   

    } else {

      if (col_idx.find(column) == col_idx.end()) {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
        CubDebugExit(cudaMemcpy(col_idx[column], cm->segment_list[column->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
      }

      dimkey_idx = col_idx[column];

      build_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>(
        d_off_col, cm->gpuCache, dimkey_idx, NULL, *h_total,
        ht_GPU[column], dim_len[column], min_key[column], 0, 1);

      CHECK_ERROR();

      //cudaFree(d_off_col);

    }
  }
};

void 
QueryProcessing::call_build_CPU(int* &h_off_col, int* h_total, int table) {

  ColumnInfo* column;
  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table) {
      column = qo->join[i].second; break;
    }
  }

  if (ht_CPU[column] != NULL) {

    if (h_off_col == NULL) {

      build_CPU(NULL, column->col_ptr, NULL, column->LEN, 
       ht_CPU[column], dim_len[column], min_key[column], 1);

    } else {

      build_CPU(h_off_col, column->col_ptr, NULL, *h_total, 
         ht_CPU[column], dim_len[column], min_key[column], 1);

      //free(h_off_col);
    }

  }
};

void
QueryProcessing::call_build_filter_GPU(int* &d_off_col, int* &d_total, int* h_total, int sg, int table) {

  ColumnInfo* temp;
  int LEN;
  int tile_items = 128*4;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table){
      temp = qo->join[i].second; break;
    }
  }
  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];
  
  CubDebugExit(cudaMalloc((void**) &d_off_col, qo->segment_group_count[table][sg] * SEGMENT_SIZE * sizeof(int)));

  CubDebugExit(cudaMalloc((void **)&d_total, sizeof(int)));
  CubDebugExit(cudaMemset(d_total, 0, sizeof(int)));

  for (int i = 0; i < qo->segment_group_count[table][sg]; i++){

    int segment_number = qo->segment_group[table][sg * column->total_segment + i];
    int start_offset = SEGMENT_SIZE * segment_number;
    int idx_key = cm->segment_list[column->column_id][segment_number];
    int* filter_col = cm->gpuCache + idx_key * SEGMENT_SIZE;

    if (segment_number == column->total_segment-1 && column->LEN % SEGMENT_SIZE != 0)
      LEN = column->LEN % SEGMENT_SIZE;
    else
      LEN = SEGMENT_SIZE;

    filter_GPU<128,4> <<<(LEN + tile_items - 1)/tile_items, 128>>> (filter_col, NULL, 
      compare1[column], compare2[column], 0, 0, d_off_col, 
         d_total, start_offset, LEN, mode[column], 0);
    CHECK_ERROR();
  }
  CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
  printf("date total = %d\n", *h_total);

}

void
QueryProcessing::call_build_filter_CPU(int* &h_off_col, int* h_total, int sg, int table) {

  ColumnInfo* temp;
  int LEN;

  for (int i = 0; i < qo->join.size(); i++) {
    if (qo->join[i].second->table_id == table)
      temp = qo->join[i].second;
  }

  if (qo->select_build[temp].size() == 0) return;

  ColumnInfo* column = qo->select_build[temp][0];
  int* filter_col = column->col_ptr;

  //h_off_col = (int*) malloc(qo->segment_group_count[table][sg] * SEGMENT_SIZE * sizeof(int));
  h_off_col = new int[qo->segment_group_count[table][sg] * SEGMENT_SIZE];

  int start_offset;

  for (int i = 0; i < qo->segment_group_count[table][sg]; i++){
    int segment_number = qo->segment_group[table][sg * column->total_segment + i];
    start_offset = SEGMENT_SIZE * segment_number;

    if (segment_number == column->total_segment-1 && column->LEN % SEGMENT_SIZE != 0)
      LEN = column->LEN % SEGMENT_SIZE;
    else
      LEN = SEGMENT_SIZE;

    filter_CPU(NULL, filter_col, NULL, compare1[column], compare2[column], 0, 0, h_off_col, 
         h_total, start_offset, LEN, mode[column], 0);
  }

}

void
QueryProcessing::call_probe_filter_GPU(int** &off_col, int* &d_off_col, int* &d_total, int* h_total, int sg, int select_so_far) {
  int tile_items = 128*4;
  int **off_col_out, *filter_idx, *d_off_col_out;
  int *filter_col[2] = {};
  ColumnInfo *filter[2] = {};
  int start_offset, idx_fil, LEN;
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};

  if (qo->selectGPUPipelineCol[sg].size() == 0) return;

  //off_col_out = (int**) malloc(cm->TOT_TABLE * sizeof(int*));
  off_col_out = new int*[cm->TOT_TABLE] (); //initialize to NULL
  CubDebugExit(cudaMalloc((void**) &d_off_col_out, cm->TOT_TABLE * SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)));
  for (int i = 0; i < cm->TOT_TABLE; i++) {
    off_col_out[i] = d_off_col_out + i * SEGMENT_SIZE * qo->segment_group_count[0][sg];
  }

  CubDebugExit(cudaMalloc((void **)&d_total, sizeof(int)));
  CubDebugExit(cudaMemset(d_total, 0, sizeof(int)));

  for (int i = 0; i < qo->selectGPUPipelineCol[sg].size(); i++) {
    ColumnInfo* column = qo->selectGPUPipelineCol[sg][i];
    assert(select_so_far + i < qo->select_probe[cm->lo_orderdate].size());
    assert(column != NULL);
    filter[select_so_far + i] = column;
    _compare1[select_so_far + i] = compare1[column];
    _compare2[select_so_far + i] = compare2[column];
    _mode[select_so_far + i] = mode[column];
  }

  if (off_col == NULL) {
    assert(filter[0] != NULL);

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

      if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0)
        LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
      else
        LEN = SEGMENT_SIZE;

      filter_GPU<128,4><<<(LEN + tile_items - 1)/tile_items, 128>>>
        (filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1], off_col_out[0], 
           d_total, start_offset, LEN, _mode[0], _mode[1]);
      CHECK_ERROR();
    }
  } else {

    assert(filter[1] != NULL);
    assert(off_col[0] != NULL);

    if (col_idx.find(filter[1]) == col_idx.end()) {
      CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[filter[1]], cm->cache_total_seg * sizeof(int)));
      CubDebugExit(cudaMemcpy(col_idx[filter[1]], cm->segment_list[filter[1]->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice));
    }

    filter_idx = col_idx[filter[1]];

    assert(filter_idx != NULL);

    cout << filter[1]->column_name << endl;

    filter_GPU2<128,4><<<(*h_total + tile_items - 1)/tile_items, 128>>>
        (off_col[0], cm->gpuCache, filter_idx, _compare1[1], _compare2[1], off_col_out[0], 
           d_total, start_offset, *h_total, _mode[1]);
    CHECK_ERROR();
  }

  if (off_col != NULL) {
    assert(d_off_col != NULL);
    //free(off_col);
    delete[] off_col;
    cudaFree(d_off_col);
  }

  off_col = off_col_out;
  d_off_col = d_off_col_out;
  CubDebugExit(cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
}

void
QueryProcessing::call_probe_filter_CPU(int** &h_off_col, int* h_total, int sg, int select_so_far) {
  int **off_col_out;
  int *filter_col[2] = {};
  int start_offset, LEN;
  int out_total = 0;
  int _compare1[2] = {0}, _compare2[2] = {0}, _mode[2] = {0};

  if (qo->selectCPUPipelineCol[sg].size() == 0) return;

  //off_col_out = (int**) malloc(cm->TOT_TABLE * sizeof(int*));
  //off_col_out[0] = (int*) malloc(SEGMENT_SIZE * qo->segment_group_count[0][sg] * sizeof(int)); //lo_off
  off_col_out = new int*[cm->TOT_TABLE](); //initialize to NULL
  for (int i = 0; i < cm->TOT_TABLE; i++) {
    off_col_out[i] = new int[SEGMENT_SIZE * qo->segment_group_count[0][sg]];
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

    for (int i = 0; i < qo->segment_group_count[0][sg]; i++) {

      int segment_number = qo->segment_group[0][sg * cm->lo_orderdate->total_segment + i];
      start_offset = segment_number * SEGMENT_SIZE;

      if (segment_number == cm->lo_orderdate->total_segment-1 && cm->lo_orderdate->LEN % SEGMENT_SIZE != 0)
        LEN = cm->lo_orderdate->LEN % SEGMENT_SIZE;
      else
        LEN = SEGMENT_SIZE;

      filter_CPU(NULL, filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1], off_col_out[0], 
           &out_total, start_offset, LEN, _mode[0], _mode[1]);
    }

  } else {
    assert(filter_col[0] == NULL);
    assert(filter_col[1] != NULL);
    filter_CPU(h_off_col[0], filter_col[0], filter_col[1], _compare1[0], _compare2[0], _compare1[1], _compare2[1], off_col_out[0], 
         &out_total, start_offset, *h_total, _mode[0], _mode[1]);
  }

  if (h_off_col != NULL) {
    if (h_off_col[0] != NULL) delete[] h_off_col[0]; //free(h_off_col[0]);
    //free(h_off_col);
    delete[] h_off_col;
  }

  h_off_col = off_col_out;
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
      CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
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
        CubDebugExit(g_allocator.DeviceAllocate((void**)&col_idx[column], cm->cache_total_seg * sizeof(int)));
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

  CubDebugExit(cudaMemcpy(res, d_res, total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));
  //TODO: don't forget to merge after this !!!!!!

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

  //TODO: don't forget to merge after this !!!!!!

}

void
QueryProcessing::runQuery(int query) {
  for (int i = 0; i < qo->join.size(); i++) {

    for (int sg = 0; sg < 2; sg++) {

      int *h_off_col = NULL, *d_off_col = NULL, *d_total = NULL;
      int h_total = 0;

      //cout << qo->join[i].second->column_name << endl;

      if (qo->segment_group_count[qo->join[i].second->table_id][sg] > 0) {

        if (sg == 0) {
          call_build_filter_CPU(h_off_col, &h_total, sg, qo->join[i].second->table_id);

          call_build_CPU(h_off_col, &h_total, qo->join[i].second->table_id);

          if (qo->joinGPUcheck[i]) {
            switch_device_dim(d_off_col, h_off_col, d_total, &h_total, sg, 0, qo->join[i].second->table_id);
            call_build_GPU(d_off_col, &h_total, qo->join[i].second->table_id);
          }

          delete[] h_off_col;

        } else { 
          call_build_filter_GPU(d_off_col, d_total, &h_total, sg, qo->join[i].second->table_id);

          call_build_GPU(d_off_col, &h_total, qo->join[i].second->table_id);

          switch_device_dim(d_off_col, h_off_col, d_total, &h_total, sg, 1, qo->join[i].second->table_id);
          call_build_CPU(h_off_col, &h_total, qo->join[i].second->table_id);

          cudaFree(d_off_col);
          
        }

      }
    }
  }

  for (int sg = 0; sg < 64; sg++) {

    int** h_off_col = NULL, **off_col = NULL;
    int* d_off_col = NULL, *d_total = NULL, h_total = 0;
    //CubDebugExit(cudaMalloc((void**) &d_total, sizeof(int)));

    if (qo->segment_group_count[0][sg] > 0) {

      printf("%d\n", sg);

      call_probe_filter_CPU(h_off_col, &h_total, sg, 0);

      switch_device_fact(off_col, d_off_col, h_off_col, d_total, &h_total, sg, 0, 0);

      call_probe_filter_GPU(off_col, d_off_col, d_total, &h_total, sg, qo->selectCPUPipelineCol[sg].size());

      call_probe_GPU(off_col, d_off_col, d_total, &h_total, sg);

      switch_device_fact(off_col, d_off_col, h_off_col, d_total, &h_total, sg, 1, 0);

      call_probe_CPU(h_off_col, &h_total, sg);

      // call_probe_CPU(h_off_col, &h_total, sg);

      // switch_device_fact(off_col, d_off_col, h_off_col, d_total, &h_total, sg, 0, 0);

      // call_probe_GPU(off_col, d_off_col, d_total, &h_total, sg);

      if (qo->groupGPUcheck) {
        if (qo->groupbyGPUPipelineCol[sg].size() > 0) {
          //switch_device_fact(off_col, d_off_col, h_off_col, d_total, &h_total, sg, 0, 0);
          call_group_by_GPU(off_col, &h_total);
        } else {
          switch_device_fact(off_col, d_off_col, h_off_col, d_total, &h_total, sg, 1, 0);
          call_group_by_CPU(h_off_col, &h_total);
        }
      } else {
        switch_device_fact(off_col, d_off_col, h_off_col, d_total, &h_total, sg, 1, 0);
        call_group_by_CPU(h_off_col, &h_total);
      }

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

      if (h_off_col != NULL) {
        for (int i = 0; i < cm->TOT_TABLE; i++) delete[] h_off_col[i];
        delete[] h_off_col;
      }
      if (off_col != NULL) delete[] off_col;
      if (d_off_col != NULL) cudaFree(d_off_col);

    }
  }

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i<total_val; i++) {
    if (res[6*i+4] != 0) {
      cout << res[6*i] << " " << res[6*i+1] << " " << res[6*i+2] << " " << res[6*i+3] << " " << reinterpret_cast<unsigned long long*>(&res[6*i+4])[0]  << endl;
      res_count++;
    }
  }
  printf("Res count = %d\n", res_count);

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

    total_val = (1998-1992+1);

    ht_CPU[cm->p_partkey] = NULL;
    ht_CPU[cm->c_custkey] = NULL;
    ht_CPU[cm->s_suppkey] = NULL;
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));
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

    ht_CPU[cm->p_partkey] = (int*)malloc(2 * dim_len[cm->p_partkey] * sizeof(int));
    ht_CPU[cm->c_custkey] = NULL;
    ht_CPU[cm->s_suppkey] = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_CPU[cm->p_partkey], 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    memset(ht_CPU[cm->s_suppkey], 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));

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

    ht_CPU[cm->p_partkey] = NULL;
    ht_CPU[cm->c_custkey] = (int*)malloc(2 * dim_len[cm->c_custkey] * sizeof(int));
    ht_CPU[cm->s_suppkey] = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_CPU[cm->c_custkey], 0, 2 * dim_len[cm->c_custkey] * sizeof(int));
    memset(ht_CPU[cm->s_suppkey], 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * dim_len[cm->c_custkey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));

    CubDebugExit(cudaMemset(d_ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int)));

  } else {

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

    ht_CPU[cm->p_partkey] = (int*)malloc(2 * dim_len[cm->p_partkey] * sizeof(int));
    ht_CPU[cm->c_custkey] = (int*)malloc(2 * dim_len[cm->c_custkey] * sizeof(int));
    ht_CPU[cm->s_suppkey] = (int*)malloc(2 * dim_len[cm->s_suppkey] * sizeof(int));
    ht_CPU[cm->d_datekey] = (int*)malloc(2 * dim_len[cm->d_datekey] * sizeof(int));

    memset(ht_CPU[cm->d_datekey], 0, 2 * dim_len[cm->d_datekey] * sizeof(int));
    memset(ht_CPU[cm->p_partkey], 0, 2 * dim_len[cm->p_partkey] * sizeof(int));
    memset(ht_CPU[cm->s_suppkey], 0, 2 * dim_len[cm->s_suppkey] * sizeof(int));
    memset(ht_CPU[cm->c_custkey], 0, 2 * dim_len[cm->c_custkey] * sizeof(int));

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_ht_c, 2 * dim_len[cm->c_custkey] * sizeof(int)));

    CubDebugExit(cudaMemset(d_ht_p, 0, 2 * dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_s, 0, 2 * dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_d, 0, 2 * dim_len[cm->d_datekey] * sizeof(int)));
    CubDebugExit(cudaMemset(d_ht_c, 0, 2 * dim_len[cm->c_custkey] * sizeof(int)));

  }

  ht_GPU[cm->p_partkey] = d_ht_p;
  ht_GPU[cm->c_custkey] = d_ht_c;
  ht_GPU[cm->s_suppkey] = d_ht_s;
  ht_GPU[cm->d_datekey] = d_ht_d;

  int res_array_size = total_val * 6;
  res = new int[res_array_size];
  memset(res, 0, res_array_size * sizeof(int));
     
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int)));
  CubDebugExit(cudaMemset(d_res, 0, res_array_size * sizeof(int)));
}
#endif