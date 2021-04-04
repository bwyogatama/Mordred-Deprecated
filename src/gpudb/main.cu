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

#define HASH_WM(X,Y,Z) ((X-Z) % Y)

bool g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_3_GPU(int* gpuCache, int idx_key1, int idx_key2, int idx_key3, int idx_aggr, 
	int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* res,
	int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int total_val,
	int min_key1, int min_key2, int min_key3) {

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;    // Current tile index
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockLoadInt::TempStorage load_items;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int aggrval[ITEMS_PER_THREAD];

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
  int* aggr = gpuCache + idx_aggr * SEGMENT_SIZE;

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
      int hash = HASH_WM(items[ITEM], dim_len1, min_key1);
      if (selection_flags[ITEM]) {
				uint64_t slot = *reinterpret_cast<uint64_t*>(&ht1[hash << 1]);
				if (slot != 0) {
          // printf("ID %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, items[ITEM]);
          // cudaDeviceSynchronize();
					groupval1[ITEM] = (slot >> 32);
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
      int hash = HASH_WM(items[ITEM], dim_len2, min_key2);
      if (selection_flags[ITEM]) {
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht2[hash << 1]);
        if (slot != 0) {
          // printf("ID %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, items[ITEM]);
          // cudaDeviceSynchronize();
          groupval2[ITEM] = (slot >> 32);
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
      int hash = HASH_WM(items[ITEM], dim_len3, min_key3); //19920101
      if (selection_flags[ITEM]) {
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht3[hash << 1]);
        if (slot != 0) {
          // printf("item %d\n", items[ITEM]);
          // printf("ID2 %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, items[ITEM]);
          // cudaDeviceSynchronize();
          groupval3[ITEM] = (slot >> 32);
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(aggr + tile_offset, aggrval);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      if (selection_flags[ITEM]) {
        // printf("groupval2 %d\n", groupval2[ITEM]);
      	//int hash = (groupval1[ITEM] * 1 + groupval2[ITEM] * 7 +  (groupval3[ITEM] - 1992)) % ((1998-1992+1) * (5*5*40)); //!
        int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3)) % total_val; //!
        // printf("ID %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, aggrval[ITEM]);
        // cudaDeviceSynchronize();
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval[ITEM]));
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
      if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      	int hash = HASH_WM(items[ITEM], dim_len1, min_key1);
    		if (selection_flags[ITEM]) {
    			uint64_t slot = *reinterpret_cast<uint64_t*>(&ht1[hash << 1]);
    			if (slot != 0) {
    				groupval1[ITEM] = (slot >> 32);
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
      	  int hash = HASH_WM(items[ITEM], dim_len2, min_key2);
	      if (selection_flags[ITEM]) {
	        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht2[hash << 1]);
	        if (slot != 0) {
            // printf("ID %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, items[ITEM]);
            // cudaDeviceSynchronize();
	          groupval2[ITEM] = (slot >> 32);
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
      if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
	      int hash = HASH_WM(items[ITEM], dim_len3, min_key3); //19920101
	      if (selection_flags[ITEM]) {
	        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht3[hash << 1]);
	        if (slot != 0) {
            // printf("ID2 %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, items[ITEM]);
            // cudaDeviceSynchronize();
	          groupval3[ITEM] = (slot >> 32);
	        } else {
	          selection_flags[ITEM] = 0;
	        }
	      }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(aggr + tile_offset, aggrval, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
	      if (selection_flags[ITEM]) {
          // cudaDeviceSynchronize();
	      	//int hash = (groupval1[ITEM] * 1 + groupval2[ITEM] * 7 +  (groupval3[ITEM] - 1992)) % ((1998-1992+1) * (5*5*40)); //!
	        int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3)) % total_val; //!
          // printf("ID %d %d %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM, hash, aggrval[ITEM]);
          // cudaDeviceSynchronize();
	        res[hash * 6] = groupval1[ITEM];
	        res[hash * 6 + 1] = groupval2[ITEM];
	        res[hash * 6 + 2] = groupval3[ITEM];
	        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval[ITEM]));
	      }
      }
    }
  }
}

void probe_group_by_3_CPU(int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* aggr_col,
  int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int total_val,
  int min_key1, int min_key2, int min_key3, int start_index) {

  // Probe
  parallel_for(blocked_range<size_t>(0, fact_len, fact_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        //long long slot;
        int slot;
        hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
        //slot = reinterpret_cast<long long*>(ht1)[hash << 1];
        slot = ht1[hash << 1];
        if (slot != 0) {
          //int dim_val1 = slot >> 32;
          int dim_val1 = ht1[(hash << 1) + 1];
          hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
          //slot = reinterpret_cast<long long*>(ht2)[hash << 1];
          slot = ht2[hash << 1];
          if (slot != 0) {
            //int dim_val2 = slot >> 32;
            int dim_val2 = ht2[(hash << 1) + 1];
            hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
            //slot = reinterpret_cast<long long*>(ht3)[hash << 1];
            slot = ht3[hash << 1];
            if (slot != 0) {
              //int dim_val3 = slot >> 32;
              int dim_val3 = ht3[(hash << 1) + 1];
              hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3)) % total_val;
              res[hash * 6] = dim_val1;
              res[hash * 6 + 1] = dim_val2;
              res[hash * 6 + 2] = dim_val3;
              __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
            }
          }
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      //long long slot;
      int slot;
      hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
      //slot = reinterpret_cast<long long*>(ht1)[hash << 1];
      slot = ht1[hash << 1];
      if (slot != 0) {
        //int dim_val1 = slot >> 32;
        int dim_val1 = ht1[(hash << 1) + 1];
        hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
        //slot = reinterpret_cast<long long*>(ht2)[hash << 1];
        slot = ht2[hash << 1];
        if (slot != 0) {
          //int dim_val2 = slot >> 32;
          int dim_val2 = ht2[(hash << 1) + 1];
          hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
          //slot = reinterpret_cast<long long*>(ht3)[hash << 1];
          slot = ht3[hash << 1];
          if (slot != 0) {
            //int dim_val3 = slot >> 32;
            int dim_val3 = ht3[(hash << 1) + 1];
            hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3)) % total_val;
            res[hash * 6] = dim_val1;
            res[hash * 6 + 1] = dim_val2;
            res[hash * 6 + 2] = dim_val3;
            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
          }
        }
      }
    }
  });

}

__global__
void build_filter_GPU(int *gpuCache, int idx_filter, int compare, int idx_key, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
  	if (gpuCache[idx_filter * SEGMENT_SIZE + offset] == compare) {
			int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
			int hash = HASH_WM(key, num_slots, val_min);
			atomicCAS(&hash_table[hash << 1], 0, key);
  	}
  }
}

/*__global__
void build_filter_GPU2(int* filter_col, int compare, int* key_col, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset] == compare) {
      int key = key_col[offset];
      int hash = HASH_WM(key, num_slots, val_min);
      atomicCAS(&hash_table[hash << 1], 0, key);
    }
  }
}*/

__global__
void build_value_GPU(int *gpuCache, int idx_key, int idx_value, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
		int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
		int value = gpuCache[idx_value * SEGMENT_SIZE + offset];
		int hash = HASH_WM(key, num_slots, val_min);
		atomicCAS(&hash_table[hash << 1], 0, key);
		hash_table[(hash << 1) + 1] = value;
  }
}

__global__
void build_filter_value_GPU(int *gpuCache, int idx_filter, int compare, int idx_key, int idx_value, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
  	if (gpuCache[idx_filter * SEGMENT_SIZE + offset] == compare) {
			int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
			int value = gpuCache[idx_value * SEGMENT_SIZE + offset];
			int hash = HASH_WM(key, num_slots, val_min);
			atomicCAS(&hash_table[hash << 1], 0, key);
			hash_table[(hash << 1) + 1] = value;
  	}
  }
}

/*__global__
void build_filter_value_GPU2(int *filter_col, int compare, int* key_col, int* val_col, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset] == compare) {
      int key = key_col[offset];
      int value = val_col[offset];
      int hash = HASH_WM(key, num_slots, val_min);
      atomicCAS(&hash_table[hash << 1], 0, key);
      hash_table[(hash << 1) + 1] = value;
    }
  }
}*/

void build_filter_CPU(int *filter_col, int compare, int *dim_key, int num_tuples, int *hash_table, int num_slots, int val_min) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == 1) {
        int key = dim_key[i];
        int hash = HASH_WM(key, num_slots, val_min);
        hash_table[hash << 1] = key;
      }
    }
  });
}

void build_filter_value_CPU(int *filter_col, int compare, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == 1) {
        int key = dim_key[i];
        int val = dim_val[i];
        int hash = HASH_WM(key, num_slots, val_min);
        hash_table[hash << 1] = key;
        hash_table[(hash << 1) + 1] = val;
      }
    }
  });
}

void build_value_CPU(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      int key = dim_key[i];
      int val = dim_val[i];
      int hash = HASH_WM(key, num_slots, val_min);
      hash_table[hash << 1] = key;
      hash_table[(hash << 1) + 1] = val;
    }
  });
}


int main () {

	CacheManager* cm = new CacheManager(1000000000, 25);

	cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 4000);
	cm->cacheColumnSegmentInGPU(cm->lo_partkey, 4000);
	cm->cacheColumnSegmentInGPU(cm->lo_suppkey, 4000);
	cm->cacheColumnSegmentInGPU(cm->lo_revenue, 4000);
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
    //int* filter_col = cm->gpuCache + idx_filter * SEGMENT_SIZE;
    //int* key_col = cm->gpuCache + idx_key * SEGMENT_SIZE;
		build_filter_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_filter, 1, idx_key, SEGMENT_SIZE, d_ht_s, S_LEN, 0);
    //build_filter_GPU2<<<(SEGMENT_SIZE + 127)/128, 128>>>(filter_col, 1, key_col, SEGMENT_SIZE, d_ht_s, S_LEN, 0);
	}

	/*int idx_key = cm->segment_list[cm->s_suppkey->column_id][0];
	cudaMemcpy(h_ht_s, cm->gpuCache + idx_key * SEGMENT_SIZE, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 1000; i++) {
		printf("%d\n", h_ht_s[i]);
	}*/
	for (int i = 0; i < 200; i++) {
		int idx_key = cm->segment_list[cm->p_partkey->column_id][i];
		int idx_filter = cm->segment_list[cm->p_category->column_id][i];
		int idx_value = cm->segment_list[cm->p_brand1->column_id][i];
    //int* filter_col = cm->gpuCache + idx_filter * SEGMENT_SIZE;
    //int* key_col = cm->gpuCache + idx_key * SEGMENT_SIZE;
    //int* val_col = cm->gpuCache + idx_value * SEGMENT_SIZE;
		build_filter_value_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_filter, 1, idx_key, idx_value, SEGMENT_SIZE, d_ht_p, P_LEN, 0);
    //build_filter_value_GPU2<<<(SEGMENT_SIZE + 127)/128, 128>>>(filter_col, 1, key_col, val_col, SEGMENT_SIZE, d_ht_p, P_LEN, 0);
		//cudaMemcpy(h_ht_p + i * SEGMENT_SIZE, cm->gpuCache + idx_value * SEGMENT_SIZE, 1000 * sizeof(int), cudaMemcpyDeviceToHost);
	}


	/*cudaMemcpy(h_ht_p, d_ht_p, 2 * P_LEN * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < P_LEN; i++) {
		//printf("i = %d\n", i << 1);
		if (h_ht_p[i << 1] != 0) {
			printf("%d\n", h_ht_p[i << 1]);
			printf("%d\n", h_ht_p[(i << 1) + 1]);
		}
	}*/

	for (int i = 0; i < 3; i++) {
		if (i == 2) {
			int idx_key = cm->segment_list[cm->d_datekey->column_id][i];
			int idx_value = cm->segment_list[cm->d_year->column_id][i];
			build_value_GPU<<<((D_LEN % SEGMENT_SIZE) + 127)/128, 128>>>(cm->gpuCache, idx_key, idx_value, D_LEN % SEGMENT_SIZE, d_ht_d, d_val_len, 19920101);
		} else {
			int idx_key = cm->segment_list[cm->d_datekey->column_id][i];
			int idx_value = cm->segment_list[cm->d_year->column_id][i];
			build_value_GPU<<<(SEGMENT_SIZE + 127)/128, 128>>>(cm->gpuCache, idx_key, idx_value, SEGMENT_SIZE, d_ht_d, d_val_len, 19920101);
		}
	}

  int *d_res;
  int res_size = ((1998-1992+1) * (5 * 5 * 40));
  int res_array_size = res_size * 6;
     
  g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));
  cudaMemset(d_res, 0, res_array_size * sizeof(int));

  /*int *lo = (int*)malloc(3000 * sizeof(int));
  int idx_aggr = cm->segment_list[cm->lo_revenue->column_id][2];
  printf("idx_aggr %d\n", idx_aggr);
  cudaMemcpy(lo, cm->gpuCache + idx_aggr * SEGMENT_SIZE, 1000 * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 1000; i++) {
    printf("%d %d\n", i, cm->h_lo_revenue[i]);
  }*/

  for (int i = 0; i < 4000; i++) {
   	int tile_items = 128*4;
   	int idx_key1 = cm->segment_list[cm->lo_suppkey->column_id][i];
   	int idx_key2 = cm->segment_list[cm->lo_partkey->column_id][i];
   	int idx_key3 = cm->segment_list[cm->lo_orderdate->column_id][i];
   	int idx_aggr = cm->segment_list[cm->lo_revenue->column_id][i];

  	probe_group_by_3_GPU<128,4><<<(SEGMENT_SIZE + tile_items - 1)/tile_items, 128>>>(cm->gpuCache, idx_key1, idx_key2, idx_key3, idx_aggr, 
			SEGMENT_SIZE, d_ht_s, S_LEN, d_ht_p, P_LEN, d_ht_d, d_val_len, d_res,
			0, 1, 0, 7, 1992, (1998-1992+1) * (5*5*40),
			0, 0, 19920101);
  }

  int* res = new int[res_array_size];
  memset(res, 0, res_array_size * sizeof(int));
  cudaMemcpy(res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost);
  
  build_filter_CPU(cm->h_s_region, 1, cm->h_s_suppkey, S_LEN, h_ht_s, S_LEN, 0);

  build_filter_value_CPU(cm->h_p_category, 1, cm->h_p_partkey, cm->h_p_brand1, P_LEN, h_ht_p, P_LEN, 0);

  build_value_CPU(cm->h_d_datekey, cm->h_d_year, D_LEN, h_ht_d, d_val_len, 19920101);

  int CPU_len = 2000000;
  int start_index = 4000000;

  probe_group_by_3_CPU(cm->h_lo_suppkey, cm->h_lo_partkey, cm->h_lo_orderdate, cm->h_lo_revenue,
    CPU_len, h_ht_s, S_LEN, h_ht_p, P_LEN, h_ht_d, d_val_len, res,
    0, 1, 0, 7, 1992, (1998-1992+1) * (5*5*40),
    0, 0, 19920101, start_index);

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