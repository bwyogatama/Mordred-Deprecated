#include "QueryOptimizer.h"

#include <chrono>
#include <atomic>
#include <unistd.h>
/*#include <cub/device/device_select.cuh>*/
#include <cub/cub.cuh>
#include "tbb/tbb.h"
// #include "tbb/parallel_for.h"
// #include "tbb/parallel_scan.h"
// #include "tbb/task_arena.h"
// #include "tbb/tbb_thread.h"
// #include "tbb/task.h"
// #include "tbb/task_scheduler_init.h"
// #include "tbb/partitioner.h"

using namespace std;
using namespace cub;
using namespace tbb;

//tbb::task_scheduler_init init(48); // Use the default number of threads.

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_2_GPU(int* gpuCache, int idx_key1, int idx_key2, 
  int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2,
  int min_key1, int min_key2, int* t_table, int *total, int start_offset) {

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
          // printf("item %d\n", items[ITEM]);
          // printf("ID %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM);
          // cudaDeviceSynchronize();
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
      int hash = HASH(items[ITEM], dim_len2, min_key2); //19920101
      if (selection_flags[ITEM]) {
        int slot = ht2[(hash << 1) + 1];
        if (slot != 0) {
          t_count++;
          dim_offset2[ITEM] = slot;
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
            // printf("item %d\n", items[ITEM]);
            // printf("ID %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM);
            // cudaDeviceSynchronize();
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
     * Join with date table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int hash = HASH(items[ITEM], dim_len2, min_key2); //19920101
        if (selection_flags[ITEM]) {
          int slot = ht2[(hash << 1) + 1]; //TODO fix this
          if (slot != 0) {
            t_count++;
            dim_offset2[ITEM] = slot;
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
        int offset = block_off + c_t_count++; // block offset can be out of order, does not have to match block id order
        t_table[(offset << 2)] = start_offset + blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
        t_table[(offset << 2) + 1] = dim_offset1[ITEM];
        t_table[(offset << 2) + 2] = dim_offset2[ITEM];
        t_table[(offset << 2) + 3] = 0;
      }
    }
  }
}

void probe_2_CPU(int* h_t_table, int* dimkey_col1, int* ht1, int h_total, int dim_len1, int start_offset, int min_key1) {

  // Probe
  parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot;
        int lo_offset = h_t_table[((start_offset + i) << 2)];
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = ht1[hash << 1];
        if (slot != 0) {
          int dim_offset1 = ht1[(hash << 1) + 1];
          h_t_table[((start_offset + i) << 2) + 3] = dim_offset1;
        } else {
          h_t_table[((start_offset + i) << 2)] = 0;
          h_t_table[((start_offset + i) << 2) + 1] = 0;
          h_t_table[((start_offset + i) << 2) + 2] = 0;
          h_t_table[((start_offset + i) << 2) + 3] = 0;
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot;
      int lo_offset = h_t_table[((start_offset + i) << 2)];
      hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
      slot = ht1[hash << 1];
      if (slot != 0) {
        int dim_offset1 = ht1[(hash << 1) + 1];
        h_t_table[((start_offset + i) << 2) + 3] = dim_offset1;
      } else {
        h_t_table[((start_offset + i) << 2)] = 0;
        h_t_table[((start_offset + i) << 2) + 1] = 0;
        h_t_table[((start_offset + i) << 2) + 2] = 0;
        h_t_table[((start_offset + i) << 2) + 3] = 0;
      }
    }
  });
}

void probe_2_CPU2(int* h_t_table, int* dimkey_col1, int* ht1, int* h_t_table_res, int h_total, int dim_len1, int start_offset, int min_key1, int* offset) {

  // Probe
  parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    int count = 0;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot;
        int lo_offset = h_t_table[((start_offset + i) << 2)];
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = ht1[hash << 1];
        if (slot != 0) {
          count++;
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot;
      int lo_offset = h_t_table[((start_offset + i) << 2)];
      hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
      slot = ht1[hash << 1];
      if (slot != 0) {
        count++;
      }
    }
    //printf("count = %d\n", count);
    int thread_off = __atomic_fetch_add(offset, count, __ATOMIC_RELAXED);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot;
        int lo_offset = h_t_table[((start_offset + i) << 2)];
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = ht1[hash << 1];
        if (slot != 0) {
          int dim_offset1 = ht1[(hash << 1) + 1];
          int dim_offset2 = h_t_table[((start_offset + i) << 2) + 1];
          int dim_offset3 = h_t_table[((start_offset + i) << 2) + 2];
          h_t_table_res[(thread_off << 2)] = lo_offset;
          h_t_table_res[(thread_off << 2) + 1] = dim_offset1;
          h_t_table_res[(thread_off << 2) + 2] = dim_offset2;
          h_t_table_res[(thread_off << 2) + 3] = dim_offset3;
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot;
      int lo_offset = h_t_table[((start_offset + i) << 2)];
      hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
      slot = ht1[hash << 1];
      if (slot != 0) {
        int dim_offset1 = ht1[(hash << 1) + 1];
        int dim_offset2 = h_t_table[((start_offset + i) << 2) + 1];
        int dim_offset3 = h_t_table[((start_offset + i) << 2) + 2];
        h_t_table_res[(thread_off << 2)] = lo_offset;
        h_t_table_res[(thread_off << 2) + 1] = dim_offset1;
        h_t_table_res[(thread_off << 2) + 2] = dim_offset2;
        h_t_table_res[(thread_off << 2) + 3] = dim_offset3;
      }
    }
  });
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

void build_filter_offset_CPU(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots, int val_min) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == 1) {
        int key = dim_key[i];
        int hash = HASH(key, num_slots, val_min);
        hash_table[hash << 1] = key;
        hash_table[(hash << 1) + 1] = i;
      }
    }
  });
}

__global__
void runAggregationQ2GPU(int* lo_col, int* p_col, int* d_col, int* d_t_table, int num_tuples, int* res, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < num_tuples) {
    int revenue = lo_col[d_t_table[(offset << 2)]];
    int year = d_col[d_t_table[(offset << 2) + 2]];
    int brand = p_col[d_t_table[(offset << 2) + 3]];

    int hash = (brand * 7 + (year - 1992)) % num_slots;

    res[hash * 6] = 0;
    res[hash * 6 + 1] = brand;
    res[hash * 6 + 2] = year;
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(revenue));

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
        if (t_table[i << 2] != 0) {
          int brand = p_brand1[t_table[(i << 2) + 3]];
          int year = d_year[t_table[(i << 2) + 2]];
          int hash = (brand * 7 + (year - 1992)) % num_slots;
          res[hash * 6 + 1] = brand;
          res[hash * 6 + 2] = year;
          __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[t_table[i << 2]]), __ATOMIC_RELAXED);
        }
      }
    }
    for (int i = end_batch ; i < end; i++) {
      if (t_table[i << 2] != 0) {
        int brand = p_brand1[t_table[(i << 2) + 3]];
        int year = d_year[t_table[(i << 2) + 2]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[t_table[i << 2]]), __ATOMIC_RELAXED);
      }
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
    int segment_number = i;
		build_filter_offset_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_filter, 1, idx_key, segment_number, SEGMENT_SIZE, d_ht_s, S_LEN, 0);
	}

  build_filter_offset_CPU(cm->h_p_category, cm->h_p_partkey, P_LEN, h_ht_p, P_LEN, 0);

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

  int t_len = LO_LEN * 4; // lo_len*4 is the max entries required
  int *t_table;
  int *h_t_table = new int[t_len];
  int *h_t_table_res = new int[t_len];
  g_allocator.DeviceAllocate((void**)&t_table, t_len * sizeof(int));

  int *total;
  int h_total = 0;
  cudaMalloc((void **)&total, sizeof(int));
  cudaMemset(total, 0, sizeof(int));

  int *d_res;
  int res_size = ((1998-1992+1) * (5 * 5 * 40));
  int res_array_size = res_size * 6;
     
  g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
  cudaMemset(d_res, 0, res_array_size * sizeof(int));

  int* offset = (int*) malloc(sizeof(int));
  *offset = 0;

  int start_index = 0;

  for (int i = 0; i < 6000; i++) {

    start_index = h_total;

    int tile_items = 128*4;
    int idx_key1 = cm->segment_list[cm->lo_suppkey->column_id][i];
    int idx_key2 = cm->segment_list[cm->lo_orderdate->column_id][i];
    int start_offset = i * SEGMENT_SIZE;

    probe_2_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>(cm->gpuCache, idx_key1, idx_key2, 
      SEGMENT_SIZE, d_ht_s, S_LEN, d_ht_d, d_val_len,
      0, 19920101, t_table, total, start_offset);

    //cudaDeviceSynchronize();

    cudaMemcpy(&h_total, total, sizeof(int), cudaMemcpyDeviceToHost);

    t_len = (h_total - start_index) * 4;

    cudaMemcpy(h_t_table + start_index * 4, t_table + start_index * 4, t_len * sizeof(int), cudaMemcpyDeviceToHost);

    probe_2_CPU2(h_t_table, cm->h_lo_partkey, h_ht_p, h_t_table_res, (h_total - start_index), P_LEN, start_index, 0, offset);
    //probe_2_CPU(h_t_table, cm->h_lo_partkey, h_ht_p, (h_total - start_index), P_LEN, start_index, 0);

    //printf("h_total = %d\n", h_total);
    //printf("total = %d\n", *offset);
    //printf("start_offset = %d\n", start_index);

    // for (int j = 0; j < h_total; j++) {
    //   if (h_t_table[j << 2] != 0) {
    //     printf("%d %d %d %d\n", h_t_table[j << 2], h_t_table[(j << 2) + 1], h_t_table[(j << 2) + 2], h_t_table[(j << 2) + 3]);
    //   }
    // }

    /*int lo_idx = cm->segment_list[cm->lo_revenue->column_id][i];
    int p_idx = cm->segment_list[cm->p_brand1->column_id][i];
    int d_idx = cm->segment_list[cm->d_year->column_id][i];

    int* lo_col = cm->gpuCache + lo_idx * SEGMENT_SIZE;
    int* p_col = cm->gpuCache + p_idx * SEGMENT_SIZE;
    int* d_col = cm->gpuCache + d_idx * SEGMENT_SIZE;

    runAggregationQ2GPU<<<((h_total - start_index) + 128 - 1)/128, 128>>>(lo_col, p_col, d_col, d_t_table, h_total - start_index, d_res, (1998-1992+1) * 5 * 5 * 40);

    cudaDeviceSynchronize();

    g_allocator.DeviceFree(d_t_table);*/
  }

  int* d_t_table;
  g_allocator.DeviceAllocate((void**)&d_t_table, *offset * 4 * sizeof(int));

  cudaMemcpy(d_t_table, h_t_table_res, *offset * 4 * sizeof(int), cudaMemcpyHostToDevice);



  int* res = new int[res_array_size];
  memset(res, 0, res_array_size * sizeof(int));

  runAggregationQ2CPU(cm->h_lo_revenue, cm->h_p_brand1, cm->h_d_year, h_t_table, h_total, res, (1998-1992+1) * 5 * 5 * 40);


  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;
  //cudaMemcpy(res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost);

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

  printf("hi\n");

  delete h_t_table;
  delete h_ht_p;

  g_allocator.DeviceFree(d_ht_s);
  g_allocator.DeviceFree(d_ht_d);
  g_allocator.DeviceFree(t_table);

	return 0;
}