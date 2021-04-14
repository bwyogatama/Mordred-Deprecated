#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main () {

	CacheManager* cm = new CacheManager(1000000000, 25);

	cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 6);
	cm->cacheColumnSegmentInGPU(cm->lo_partkey, 6);
	cm->cacheColumnSegmentInGPU(cm->lo_suppkey, 6);
	cm->cacheColumnSegmentInGPU(cm->lo_revenue, 6);
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

    chrono::high_resolution_clock::time_point st, finish;
    st = chrono::high_resolution_clock::now();

  	int d_val_len = 19981230 - 19920101 + 1;

  	int *h_ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
  	int *h_ht_p = (int*)malloc(2 * P_LEN * sizeof(int));
  	int *h_ht_s = (int*)malloc(2 * S_LEN * sizeof(int));

  	memset(h_ht_d, 0, 2 * d_val_len * sizeof(int));
  	memset(h_ht_p, 0, 2 * P_LEN * sizeof(int));
  	memset(h_ht_s, 0, 2 * S_LEN * sizeof(int));

    int res_size = ((1998-1992+1) * (5 * 5 * 40));
    int res_array_size = res_size * 6;
       
    int* res = new int[res_array_size];
    memset(res, 0, res_array_size * sizeof(int));

    int start_index = 0;
    int CPU_len = 6000000;

    build_filter_CPU(cm->h_s_region, 1, cm->h_s_suppkey, NULL, S_LEN, h_ht_s, S_LEN, 0, 2);

    build_filter_CPU(cm->h_p_category, 1, cm->h_p_partkey, cm->h_p_brand1, P_LEN, h_ht_p, P_LEN, 0, 0);

    build_CPU(cm->h_d_datekey, cm->h_d_year, D_LEN, h_ht_d, d_val_len, 19920101, 0);

    probe_group_by_CPU(cm->h_lo_suppkey, cm->h_lo_partkey, cm->h_lo_orderdate, NULL, cm->h_lo_revenue,
      CPU_len, h_ht_s, S_LEN, h_ht_p, P_LEN, h_ht_d, d_val_len, NULL, 0, res,
      0, 0, 0, 7, 1992, 1, 0, 0, res_size,
      0, 0, 19920101, 0, start_index);

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

    delete h_ht_p;
    delete h_ht_s;
    delete h_ht_d;
    delete res;

  }

  delete cm;

	return 0;
}