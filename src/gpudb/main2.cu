#include "QueryOptimizer.h"

#include <chrono>
#include <atomic>
/*#include <cub/device/device_select.cuh>*/
#include <cub/cub.cuh>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

using namespace std;
using namespace cub;
using namespace tbb;

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_3_GPU(int* gpuCache, int idx_key1, int idx_key2, int idx_key3, 
  int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3,
  int min_key1, int min_key2, int min_key3, int* t_table, int *total, int start_offset) {

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;    // Current tile index
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockLoadInt::TempStorage load_items;
    typename BlockScanInt::TempStorage scan;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int dim_offset1[ITEMS_PER_THREAD];
  int dim_offset2[ITEMS_PER_THREAD];
  int dim_offset3[ITEMS_PER_THREAD];
  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int num_tiles = (fact_len + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = fact_len - tile_offset;
    is_last_tile = true;
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    selection_flags[ITEM] = 1;
  }

  int* dim_key1 = gpuCache + idx_key1 * SEGMENT_SIZE;
  int* dim_key2 = gpuCache + idx_key2 * SEGMENT_SIZE;
  int* dim_key3 = gpuCache + idx_key3 * SEGMENT_SIZE;

  __syncthreads();

  /********************
    Not the last tile
    ******************/
  if (!is_last_tile) {
    BlockLoadInt(temp_storage.load_items).Load(dim_key1 + tile_offset, items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH(items[ITEM], dim_len1, min_key1);
      if (selection_flags[ITEM]) {
        int slot = ht1[(hash << 1) + 1];
        if (slot != 0) {
          dim_offset1[ITEM] = slot;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(dim_key2 + tile_offset, items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH(items[ITEM], dim_len2, min_key2);
      if (selection_flags[ITEM]) {
        int slot = ht2[(hash << 1) + 1];
        if (slot != 0) {
          // printf("brand %d\n", items[ITEM]);
          // cudaDeviceSynchronize();
          dim_offset2[ITEM] = slot;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(dim_key3 + tile_offset, items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH(items[ITEM], dim_len3, min_key3); //19920101
      if (selection_flags[ITEM]) {
        int slot = ht3[(hash << 1) + 1];
        if (slot != 0) {
          // printf("item %d\n", items[ITEM]);
          // printf("ID %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM);
          // cudaDeviceSynchronize();
          t_count++;
          dim_offset3[ITEM] = slot;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }

  }
  else {
    BlockLoadInt(temp_storage.load_items).Load(dim_key1 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    /*
     * Join with supplier table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int hash = HASH(items[ITEM], dim_len1, min_key1);
        if (selection_flags[ITEM]) {
          int slot = ht1[(hash << 1) + 1];
          if (slot != 0) {
            dim_offset1[ITEM] = slot;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }

    __syncthreads();

      BlockLoadInt(temp_storage.load_items).Load(dim_key2 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    /*
     * Join with part table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int hash = HASH(items[ITEM], dim_len2, min_key2);
        if (selection_flags[ITEM]) {
          int slot = ht2[(hash << 1) + 1];
          if (slot != 0) {
            // printf("brand %d\n", items[ITEM]);
            // cudaDeviceSynchronize();
            dim_offset2[ITEM] = slot;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(dim_key3 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    /*
     * Join with date table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int hash = HASH(items[ITEM], dim_len3, min_key3); //19920101
        if (selection_flags[ITEM]) {
          int slot = ht3[(hash << 1) + 1];
          if (slot != 0) {
            //printf("item %d\n", items[ITEM]);
            // printf("ID %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM);
            // cudaDeviceSynchronize();
            t_count++;
            dim_offset3[ITEM] = slot;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }
  }

  //Barrier
  __syncthreads();

  // TODO: need to check logic for offset
  BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
  if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
      block_off = atomicAdd(total, t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
  } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

  __syncthreads();

  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        t_table[offset << 2] = dim_offset1[ITEM];
        t_table[(offset << 2) + 1] = dim_offset2[ITEM];
        t_table[(offset << 2) + 2] = dim_offset3[ITEM];
        t_table[(offset << 2) + 3] = start_offset + blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
      }
    }
  }
}

__global__
void build_offset_GPU(int *gpuCache, int idx_key, int segment_number, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
    int value = segment_number * SEGMENT_SIZE + offset;
    int hash = HASH(key, num_slots, val_min);
    atomicCAS(&hash_table[hash << 1], 0, key);
    hash_table[(hash << 1) + 1] = value;
  }
}

__global__
void build_filter_offset_GPU(int *gpuCache, int idx_filter, int compare, int idx_key, int segment_number, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (gpuCache[idx_filter * SEGMENT_SIZE + offset] == compare) {
      int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
      int value = segment_number * SEGMENT_SIZE + offset;
      int hash = HASH(key, num_slots, val_min);
      atomicCAS(&hash_table[hash << 1], 0, key);
      hash_table[(hash << 1) + 1] = value;
    }
  }
}

void runAggregationQ2CPU(int* lo_revenue, int* p_brand1, int* d_year, int* t_table, int t_table_len, int* res, int num_slots) {
  parallel_for(blocked_range<size_t>(0, t_table_len, t_table_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int brand = p_brand1[t_table[(i << 2) + 1]];
        int year = d_year[t_table[(i << 2) + 2]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[t_table[(i << 2) + 3]]), __ATOMIC_RELAXED);
      }
    }
    for (int i = end_batch ; i < end; i++) {
        int brand = p_brand1[t_table[(i << 2)+ 1]];
        int year = d_year[t_table[(i << 2) + 2]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[t_table[(i << 2) + 3]]), __ATOMIC_RELAXED);
    }
  });
}

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

	int *h_ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
	int *h_ht_p = (int*)malloc(2 * P_LEN * sizeof(int));
	int *h_ht_s = (int*)malloc(2 * S_LEN * sizeof(int));

	memset(h_ht_d, 0, 2 * d_val_len * sizeof(int));
	memset(h_ht_p, 0, 2 * P_LEN * sizeof(int));
	memset(h_ht_s, 0, 2 * S_LEN * sizeof(int));

	int *d_ht_d, *d_ht_p, *d_ht_s;
	g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * d_val_len * sizeof(int));
	g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * P_LEN * sizeof(int));
	g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * S_LEN * sizeof(int));

	cudaMemset(d_ht_d, 0, 2 * d_val_len * sizeof(int));
	cudaMemset(d_ht_p, 0, 2 * P_LEN * sizeof(int));
	cudaMemset(d_ht_s, 0, 2 * S_LEN * sizeof(int));

	for (int i = 0; i < 2; i++) {
		int idx_key = cm->segment_list[cm->s_suppkey->column_id][i];
		int idx_filter = cm->segment_list[cm->s_region->column_id][i];
    int segment_number = i;
		build_filter_offset_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_filter, 1, idx_key, segment_number, SEGMENT_SIZE, d_ht_s, S_LEN, 0);
	}

	for (int i = 0; i < 200; i++) {
		int idx_key = cm->segment_list[cm->p_partkey->column_id][i];
		int idx_filter = cm->segment_list[cm->p_category->column_id][i];
    int segment_number = i;
		build_filter_offset_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_filter, 1, idx_key, segment_number, SEGMENT_SIZE, d_ht_p, P_LEN, 0);
	}

	for (int i = 0; i < 3; i++) {
		if (i == 2) {
			int idx_key = cm->segment_list[cm->d_datekey->column_id][i];
			int segment_number = i;
			build_offset_GPU<<<((D_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(cm->gpuCache, idx_key, segment_number, D_LEN % SEGMENT_SIZE, d_ht_d, d_val_len, 19920101);
		} else {
			int idx_key = cm->segment_list[cm->d_datekey->column_id][i];
			int segment_number = i;
			build_offset_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_key, segment_number, SEGMENT_SIZE, d_ht_d, d_val_len, 19920101);
		}
	}

  int t_len = LO_LEN * 4; // l0_len*4 is the max entries required
  int *t_table;
  
  g_allocator.DeviceAllocate((void**)&t_table, t_len * sizeof(int));

  int *total;
  int h_total;
  cudaMalloc((void **)&total, sizeof(int));
  cudaMemset(total, 0, sizeof(int));

  for (int i = 0; i < 3; i++) {
    int tile_items = 128*4;
    int idx_key1 = cm->segment_list[cm->lo_suppkey->column_id][i];
    int idx_key2 = cm->segment_list[cm->lo_partkey->column_id][i];
    int idx_key3 = cm->segment_list[cm->lo_orderdate->column_id][i];
    int start_offset = i * SEGMENT_SIZE;

    probe_3_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>(cm->gpuCache, idx_key1, idx_key2, idx_key3, 
      SEGMENT_SIZE, d_ht_s, S_LEN, d_ht_p, P_LEN, d_ht_d, d_val_len,
      0, 0, 19920101, t_table, total, start_offset);
  }

  cudaMemcpy(&h_total, total, sizeof(int), cudaMemcpyDeviceToHost);

  t_len = h_total * 4;
  int *h_t_table = new int[t_len];

  cudaMemcpy(h_t_table, t_table, t_len * sizeof(int), cudaMemcpyDeviceToHost);

  printf("total = %d\n", h_total);

  // for (int j = 0; j < h_total; j++) {
  //   printf("%d %d %d %d\n", h_t_table[j << 2], h_t_table[(j << 2) + 1], h_t_table[(j << 2) + 2], h_t_table[(j << 2) + 3]);
  // }

  int res_size = (1998-1992+1) * 5 * 5 * 40;
  int res_array_size = res_size * 6;
  int* res = new int[res_array_size];

  memset(res, 0, res_array_size * sizeof(int));

  runAggregationQ2CPU(cm->h_lo_revenue, cm->h_p_brand1, cm->h_d_year, h_t_table, h_total, res, (1998-1992+1) * 5 * 5 * 40);

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

	delete cm;

	return 0;
}