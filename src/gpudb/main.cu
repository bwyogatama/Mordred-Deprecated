#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main () {

	CacheManager* cm = new CacheManager(1000000000, 25);

  cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 60);
  cm->cacheColumnSegmentInGPU(cm->lo_partkey, 60);
  cm->cacheColumnSegmentInGPU(cm->lo_suppkey, 60);
  cm->cacheColumnSegmentInGPU(cm->lo_revenue, 60);
  cm->cacheColumnSegmentInGPU(cm->d_datekey, 1);
  cm->cacheColumnSegmentInGPU(cm->d_year, 1);
  cm->cacheColumnSegmentInGPU(cm->p_partkey, 1);
  cm->cacheColumnSegmentInGPU(cm->p_category, 1);
  cm->cacheColumnSegmentInGPU(cm->p_brand1, 1);
  cm->cacheColumnSegmentInGPU(cm->s_suppkey, 1);
  cm->cacheColumnSegmentInGPU(cm->s_region, 1);

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

  for (int trial = 0; trial < 3; trial++) {

    chrono::high_resolution_clock::time_point st, finish, bCPU1, bGPU1, bCPU2, bGPU2, pCPU1, pCPU2, pGPU1, pGPU2, tr1, tr2;
    st = chrono::high_resolution_clock::now();

    bCPU1 = chrono::high_resolution_clock::now();

  	int d_val_len = 19981230 - 19920101 + 1;

  	int *h_ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
  	int *h_ht_p = (int*)malloc(2 * P_LEN * sizeof(int));
  	int *h_ht_s = (int*)malloc(2 * S_LEN * sizeof(int));

  	memset(h_ht_d, 0, 2 * d_val_len * sizeof(int));
  	memset(h_ht_p, 0, 2 * P_LEN * sizeof(int));
  	memset(h_ht_s, 0, 2 * S_LEN * sizeof(int));

    build_filter_CPU(cm->h_s_region, 1, cm->h_s_suppkey, NULL, S_LEN, h_ht_s, S_LEN, 0, 2);

    build_filter_CPU(cm->h_p_category, 1, cm->h_p_partkey, cm->h_p_brand1, P_LEN, h_ht_p, P_LEN, 0, 0);

    build_CPU(cm->h_d_datekey, cm->h_d_year, D_LEN, h_ht_d, d_val_len, 19920101, 0);

    bCPU2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildtimeCPU = bCPU2 - bCPU1;

    bGPU1 = chrono::high_resolution_clock::now();

  	int *d_ht_d, *d_ht_p, *d_ht_s;
  	g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * d_val_len * sizeof(int));
  	g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * P_LEN * sizeof(int));
  	g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * S_LEN * sizeof(int));

  	cudaMemset(d_ht_d, 0, 2 * d_val_len * sizeof(int));
  	cudaMemset(d_ht_p, 0, 2 * P_LEN * sizeof(int));
  	cudaMemset(d_ht_s, 0, 2 * S_LEN * sizeof(int));

  	for (int i = 0; i < 1; i++) {
  		int idx_key = cm->segment_list[cm->s_suppkey->column_id][i];
  		int idx_filter = cm->segment_list[cm->s_region->column_id][i];
      int* dim_key = cm->gpuCache + idx_key * SEGMENT_SIZE;
      int* filter_col = cm->gpuCache + idx_filter * SEGMENT_SIZE;
      int segment_number = i;
  		build_filter_GPU<<<((S_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(filter_col, 1, dim_key, NULL, S_LEN % SEGMENT_SIZE, d_ht_s, S_LEN, 0, segment_number, 2);
      //build_filter_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(filter_col, 1, dim_key, NULL, SEGMENT_SIZE, d_ht_s, S_LEN, 0, segment_number, 2);
  	}

  	for (int i = 0; i < 1; i++) {
  		int idx_key = cm->segment_list[cm->p_partkey->column_id][i];
  		int idx_filter = cm->segment_list[cm->p_category->column_id][i];
  		int idx_value = cm->segment_list[cm->p_brand1->column_id][i];
      int* filter_col = cm->gpuCache + idx_filter * SEGMENT_SIZE;
      int* dim_key = cm->gpuCache + idx_key * SEGMENT_SIZE;
      int* dim_val = cm->gpuCache + idx_value * SEGMENT_SIZE;
      int segment_number = i;
  		build_filter_GPU<<<((P_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(filter_col, 1, dim_key, dim_val, P_LEN % SEGMENT_SIZE, d_ht_p, P_LEN, 0, segment_number, 0);
  	}

  	for (int i = 0; i < 1; i++) {
  		int idx_key = cm->segment_list[cm->d_datekey->column_id][i];
  		int idx_value = cm->segment_list[cm->d_year->column_id][i];
      int* dim_key = cm->gpuCache + idx_key * SEGMENT_SIZE;
      int* dim_val = cm->gpuCache + idx_value * SEGMENT_SIZE;
      int segment_number = i;
      if (i == 0)
        build_GPU<<<((D_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(dim_key, dim_val, D_LEN % SEGMENT_SIZE, d_ht_d, d_val_len, 19920101, segment_number, 0);
      else
        build_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(dim_key, dim_val, SEGMENT_SIZE, d_ht_d, d_val_len, 19920101, segment_number, 0);
    }

    bGPU2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildtimeGPU = bGPU2 - bGPU1;

    pGPU1 = chrono::high_resolution_clock::now();

    int *d_res;
    int res_size = ((1998-1992+1) * (5 * 5 * 40));
    int res_array_size = res_size * 6;

    int* res = new int[res_array_size];
    memset(res, 0, res_array_size * sizeof(int));
       
    g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
    cudaMemset(d_res, 0, res_array_size * sizeof(int));

    for (int i = 0; i < 40; i++) {
     	int tile_items = 128*4;
     	int idx_key1 = cm->segment_list[cm->lo_suppkey->column_id][i];
     	int idx_key2 = cm->segment_list[cm->lo_partkey->column_id][i];
     	int idx_key3 = cm->segment_list[cm->lo_orderdate->column_id][i];
     	int idx_aggr = cm->segment_list[cm->lo_revenue->column_id][i];

      int* dim_key1 = cm->gpuCache + idx_key1 * SEGMENT_SIZE;
      int* dim_key2 = cm->gpuCache + idx_key2 * SEGMENT_SIZE;
      int* dim_key3 = cm->gpuCache + idx_key3 * SEGMENT_SIZE;
      int* aggr = cm->gpuCache + idx_aggr * SEGMENT_SIZE;

    	probe_group_by_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>
      (dim_key1, dim_key2, dim_key3, NULL, aggr, 
  			SEGMENT_SIZE, d_ht_s, S_LEN, d_ht_p, P_LEN, d_ht_d, d_val_len, NULL, 0, d_res,
  			0, 1, 0, 7, 1992, 1, 0, 0, res_size, 0, 0, 19920101, 0);
    }

    pGPU2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> probetimeGPU = pGPU2 - pGPU1;

    tr1 = chrono::high_resolution_clock::now();

    cudaMemcpy(res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost);

    tr2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> transfertime = tr2 - tr1;

    pCPU1 = chrono::high_resolution_clock::now();

    int start_index = 40000000;
    int CPU_len = 19986214;

    probe_group_by_CPU(cm->h_lo_suppkey, cm->h_lo_partkey, cm->h_lo_orderdate, NULL, cm->h_lo_revenue,
      CPU_len, h_ht_s, S_LEN, h_ht_p, P_LEN, h_ht_d, d_val_len, NULL, 0, res,
      0, 0, 0, 7, 1992, 1, 0, 0, res_size,
      0, 0, 19920101, 0, start_index);

    pCPU2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> probetimeCPU = pCPU2 - pCPU1;

    finish = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = finish - st;

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
    cout << "Build CPU Time Taken Total: " << buildtimeCPU.count() * 1000 << endl;
    cout << "Build GPU Time Taken Total: " << buildtimeGPU.count() * 1000 << endl;
    cout << "Probe CPU Time Taken Total: " << probetimeCPU.count() * 1000 << endl;
    cout << "Probe GPU Time Taken Total: " << probetimeGPU.count() * 1000 << endl;
    cout << "Transfer Time Taken Total: " << transfertime.count() * 1000 << endl;

    delete h_ht_p;
    delete h_ht_s;
    delete h_ht_d;
    delete res;

    g_allocator.DeviceFree(d_ht_p);
    g_allocator.DeviceFree(d_ht_s);
    g_allocator.DeviceFree(d_ht_d);
    g_allocator.DeviceFree(d_res);

  }

  delete cm;

	return 0;
}