#include "QueryProcessing.h"

#define CUB_STDERR

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    printf("CUDA error: %s\n", cudaGetErrorString(error)); \
    exit(-1); \
  } \
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

tbb::task_scheduler_init init(1); // Use the default number of threads.

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main () {

  CacheManager* cm = new CacheManager(1000000000, 25);

  cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 6000);
  cm->cacheColumnSegmentInGPU(cm->lo_partkey, 6000);
  cm->cacheColumnSegmentInGPU(cm->lo_suppkey, 6000);
  cm->cacheColumnSegmentInGPU(cm->lo_revenue, 6000);
  cm->cacheColumnSegmentInGPU(cm->d_datekey, 3);
  cm->cacheColumnSegmentInGPU(cm->d_year, 3);
  cm->cacheColumnSegmentInGPU(cm->p_partkey, 200);
  cm->cacheColumnSegmentInGPU(cm->p_category, 200);
  cm->cacheColumnSegmentInGPU(cm->p_brand1, 200);
  cm->cacheColumnSegmentInGPU(cm->s_suppkey, 2);
  cm->cacheColumnSegmentInGPU(cm->s_region, 2);

  cm->constructListSegmentInGPU(cm->s_suppkey);
  cm->constructListSegmentInGPU(cm->s_region);
  cm->constructListSegmentInGPU(cm->p_partkey);
  cm->constructListSegmentInGPU(cm->p_category);
  cm->constructListSegmentInGPU(cm->p_brand1);
  cm->constructListSegmentInGPU(cm->d_datekey);
  cm->constructListSegmentInGPU(cm->d_year);
  cm->constructListSegmentInGPU(cm->lo_suppkey);
  cm->constructListSegmentInGPU(cm->lo_partkey);
  cm->constructListSegmentInGPU(cm->lo_orderdate);
  cm->constructListSegmentInGPU(cm->lo_revenue);

  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  int d_val_len = 19981230 - 19920101 + 1;

  int *h_ht_p = (int*)malloc(2 * P_LEN * sizeof(int));

  memset(h_ht_p, 0, 2 * P_LEN * sizeof(int));

  int *d_ht_d, *d_ht_s;
  g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * d_val_len * sizeof(int));
  g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * S_LEN * sizeof(int));

  cudaMemset(d_ht_d, 0, 2 * d_val_len * sizeof(int));
  cudaMemset(d_ht_s, 0, 2 * S_LEN * sizeof(int));

  for (int i = 0; i < 2; i++) {
    int idx_key = cm->segment_list[cm->s_suppkey->column_id][i];
    int idx_filter = cm->segment_list[cm->s_region->column_id][i];
    int* filter_col = cm->gpuCache + idx_filter * SEGMENT_SIZE;
    int* dim_key = cm->gpuCache + idx_key * SEGMENT_SIZE;
    int segment_number = i;
    //build_filter_GPU<<<((S_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(filter_col, 1, dim_key, NULL, S_LEN % SEGMENT_SIZE, d_ht_s, S_LEN, 0, segment_number, 2);
    build_filter_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(filter_col, 1, dim_key, NULL, SEGMENT_SIZE, d_ht_s, S_LEN, 0, segment_number, 1);
  }

  build_filter_CPU(cm->h_p_category, 1, cm->h_p_partkey, NULL, P_LEN, h_ht_p, P_LEN, 0, 1);

  for (int i = 0; i < 3; i++) {
    int idx_key = cm->segment_list[cm->d_datekey->column_id][i];
    int* dim_key = cm->gpuCache + idx_key * SEGMENT_SIZE;
    int segment_number = i;
    if (i == 2)
      build_GPU<<<((D_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(dim_key, NULL, D_LEN % SEGMENT_SIZE, d_ht_d, d_val_len, 19920101, segment_number, 1);
    else
      build_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(dim_key, NULL, SEGMENT_SIZE, d_ht_d, d_val_len, 19920101, segment_number, 1);
  }

  int *h_lo_off = new int[LO_LEN];
  int *h_supp_off = new int[LO_LEN];
  int *h_part_off = new int[LO_LEN];
  int *h_date_off = new int[LO_LEN];

  int *lo_off = new int[LO_LEN];
  int *supp_off = new int[LO_LEN];
  int *part_off = new int[LO_LEN];
  int *date_off = new int[LO_LEN];

  int *d_lo_off, *d_supp_off, *d_part_off, *d_date_off;
  g_allocator.DeviceAllocate((void**)&d_lo_off, LO_LEN * sizeof(int));
  g_allocator.DeviceAllocate((void**)&d_supp_off, LO_LEN * sizeof(int));
  g_allocator.DeviceAllocate((void**)&d_part_off, LO_LEN * sizeof(int));
  g_allocator.DeviceAllocate((void**)&d_date_off, LO_LEN * sizeof(int));

  int *d_res;
  int res_size = ((1998-1992+1) * (5 * 5 * 40));
  int res_array_size = res_size * 6;
  g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
  cudaMemset(d_res, 0, res_array_size * sizeof(int));

  int offset = 0;
  int start_index = 0;

  int *total;
  int h_total = 0;
  //cudaMalloc((void **)&total, 1 * sizeof(int));
  g_allocator.DeviceAllocate((void**)&total, sizeof(int));
  cudaMemset(total, 0, sizeof(int));

  for (int i = 0; i < 6000; i++) {

    start_index = h_total;

    int tile_items = 128*4;
    int idx_key1 = cm->segment_list[cm->lo_suppkey->column_id][i];
    int idx_key2 = cm->segment_list[cm->lo_orderdate->column_id][i];
    int start_offset = i * SEGMENT_SIZE;

    int* dim_key1 = cm->gpuCache + idx_key1 * SEGMENT_SIZE;
    int* dim_key2 = cm->gpuCache + idx_key2 * SEGMENT_SIZE;

    probe_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>> 
    (dim_key1, dim_key2, NULL, NULL,
      d_ht_s, S_LEN, d_ht_d, d_val_len, NULL, 0, NULL, 0, 0, 19920101, 0, 0, 
      d_lo_off, d_supp_off, d_date_off, NULL, NULL,
      SEGMENT_SIZE, total, start_offset);

    cudaMemcpy(&h_total, total, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(lo_off + start_index, d_lo_off + start_index , (h_total - start_index) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(supp_off + start_index, d_supp_off + start_index , (h_total - start_index) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(date_off + start_index, d_date_off + start_index , (h_total - start_index) * sizeof(int), cudaMemcpyDeviceToHost);

    // probe_2_CPU(h_t_table, cm->h_lo_partkey, h_ht_p, h_t_table_res, (h_total - start_index), P_LEN, start_index, 0, &offset);

    probe_CPU(lo_off, supp_off, date_off, NULL, NULL,
      NULL, NULL, cm->h_lo_partkey, NULL,
      NULL, 0, NULL, 0, h_ht_p, P_LEN, NULL, 0,
      0, 0, 0, 0,
      h_lo_off, h_supp_off, h_date_off, h_part_off, NULL, 
      (h_total - start_index), start_index, &offset);


    // for (int j = 0; j < offset; j++) {
    //   if (h_t_table_res[j << 2] != 0) {
    //     printf("%d %d %d %d\n", h_t_table_res[j << 2], h_t_table_res[(j << 2) + 1], h_t_table_res[(j << 2) + 2], h_t_table_res[(j << 2) + 3]);
    //   }
    // }
  }

  // int*d_lo_idx, *d_p_idx, *d_d_idx;
  // g_allocator.DeviceAllocate((void**)&d_lo_idx, cm->cache_total_seg * sizeof(int));
  // g_allocator.DeviceAllocate((void**)&d_p_idx, cm->cache_total_seg * sizeof(int));
  // g_allocator.DeviceAllocate((void**)&d_d_idx, cm->cache_total_seg * sizeof(int));

  printf("total = %d\n", offset);

  // cudaMemcpy(d_lo_off, h_lo_off, offset * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_supp_off, h_supp_off, offset * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_part_off, h_part_off, offset * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_date_off, h_date_off, offset * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_lo_idx, cm->segment_list[cm->lo_revenue->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_p_idx, cm->segment_list[cm->p_brand1->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_d_idx, cm->segment_list[cm->d_year->column_id], cm->cache_total_seg * sizeof(int), cudaMemcpyHostToDevice);

  int* res = new int[res_array_size];
  memset(res, 0, res_array_size * sizeof(int));

  runAggregationQ2CPU(cm->h_lo_revenue, cm->h_p_brand1, cm->h_d_year, h_lo_off, h_part_off, h_date_off, offset, res, res_size);

  // runAggregationQ2GPU<<<(offset + 128 - 1)/128, 128>>>(cm->gpuCache, d_lo_idx, d_p_idx, d_d_idx, d_t_table, offset, d_res, res_size);

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;
  // cudaMemcpy(res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost);

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i<res_size; i++) {
    if (res[6*i+1] != 0) {
      cout << res[6*i+1] << " " << res[6*i+2] << " " << reinterpret_cast<unsigned long long*>(&res[6*i+4])[0]  << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete cm;
  delete lo_off;
  delete supp_off;
  delete part_off;
  delete date_off;
  delete h_lo_off;
  delete h_supp_off;
  delete h_part_off;
  delete h_date_off;
  delete h_ht_p;
  delete res;

  g_allocator.DeviceFree(d_ht_s);
  g_allocator.DeviceFree(d_ht_d);
  g_allocator.DeviceFree(d_lo_off);
  g_allocator.DeviceFree(d_supp_off);
  g_allocator.DeviceFree(d_part_off);
  g_allocator.DeviceFree(d_date_off);
  // g_allocator.DeviceFree(d_lo_idx);
  // g_allocator.DeviceFree(d_p_idx);
  // g_allocator.DeviceFree(d_d_idx);

  return 0;
}



// void probe_2_CPU(int* h_t_table, int* dimkey_col1, int* ht1, int h_total, int dim_len1, int start_offset, int min_key1) {

//   // Probe
//   parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
//     int start = range.begin();
//     int end = range.end();
//     int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

//     for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
//       #pragma simd
//       for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
//         int hash;
//         int slot;
//         int lo_offset = h_t_table[((start_offset + i) << 2)];
//         hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
//         slot = ht1[hash << 1];
//         if (slot != 0) {
//           int dim_offset1 = ht1[(hash << 1) + 1];
//           h_t_table[((start_offset + i) << 2) + 3] = dim_offset1;
//         } else {
//           h_t_table[((start_offset + i) << 2)] = 0;
//           h_t_table[((start_offset + i) << 2) + 1] = 0;
//           h_t_table[((start_offset + i) << 2) + 2] = 0;
//           h_t_table[((start_offset + i) << 2) + 3] = 0;
//         }
//       }
//     }

//     for (int i = end_batch ; i < end; i++) {
//       int hash;
//       int slot;
//       int lo_offset = h_t_table[((start_offset + i) << 2)];
//       hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
//       slot = ht1[hash << 1];
//       if (slot != 0) {
//         int dim_offset1 = ht1[(hash << 1) + 1];
//         h_t_table[((start_offset + i) << 2) + 3] = dim_offset1;
//       } else {
//         h_t_table[((start_offset + i) << 2)] = 0;
//         h_t_table[((start_offset + i) << 2) + 1] = 0;
//         h_t_table[((start_offset + i) << 2) + 2] = 0;
//         h_t_table[((start_offset + i) << 2) + 3] = 0;
//       }
//     }
//   });