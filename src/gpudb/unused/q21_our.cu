// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <chrono>
#include <atomic>

#include <cuda.h>
#include <cub/util_allocator.cuh>
/*#include <cub/device/device_select.cuh>*/
#include <cub/cub.cuh>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

#include "cub/test/test_util.h"
#include "utils/gpu_utils.h"

#include "ssb_utils.h"

using namespace std;
using namespace cub;
using namespace tbb;


/**
 * Globals, constants and typedefs
 */
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<typename T>
T* loadToGPU(T* src, int numEntries, CachingDeviceAllocator& g_allocator) {
  T* dest;
  g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries);
  cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice);
  return dest;
}

#define HASH_WM(X,Y,Z) ((X-Z) % Y)
#define HASH(X,Y) (X % Y)

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    printf("CUDA error: %s\n", cudaGetErrorString(error)); \
    exit(-1); \
  } \
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* res) {

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
  int brand[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int num_tiles = (lo_len + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
    is_last_tile = true;
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    selection_flags[ITEM] = 1;
  }

  /********************
    Not the last tile
    ******************/
  if (!is_last_tile) {
    BlockLoadInt(temp_storage.load_items).Load(lo_suppkey + tile_offset, items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH(items[ITEM], s_len); // hash of lo_suppkey
      int slot = ht_s[hash];
      if (slot != 0) {
        selection_flags[ITEM] = 1;
      } else {
        selection_flags[ITEM] = 0;
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(lo_partkey + tile_offset, items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH(items[ITEM], p_len);
      if (selection_flags[ITEM]) {
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht_p[hash << 1]);
        if (slot != 0) {
          brand[ITEM] = (slot >> 32);
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(lo_orderdate + tile_offset, items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH_WM(items[ITEM], d_len, 19920101);
      if (selection_flags[ITEM]) {
        year[ITEM] = ht_d[(hash << 1) + 1];
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(lo_revenue + tile_offset, revenue);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      if (selection_flags[ITEM]) {
        int hash = (brand[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * (5*5*40));
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = brand[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM]));
      }
    }
  }
  else {
    BlockLoadInt(temp_storage.load_items).Load(lo_suppkey + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    /*
     * Join with supplier table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      int hash = HASH(items[ITEM], s_len);
      if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
        int slot = ht_s[hash];
        if (slot != 0) {
          selection_flags[ITEM] = 1;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }

    __syncthreads();

      BlockLoadInt(temp_storage.load_items).Load(lo_partkey + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    /*
     * Join with part table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        if (selection_flags[ITEM]) {
          int hash = HASH(items[ITEM], p_len);
          uint64_t slot = *reinterpret_cast<uint64_t*>(&ht_p[hash << 1]);
          if (slot != 0) {
            brand[ITEM] = (slot >> 32);
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(lo_orderdate + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    /*
     * Join with date table.
     */
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int hash = HASH_WM(items[ITEM], d_len, 19920101);

      if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
        if (selection_flags[ITEM]) {
          year[ITEM] = ht_d[(hash << 1) + 1];
        }
      }
    }

    __syncthreads();

      BlockLoadInt(temp_storage.load_items).Load(lo_revenue + tile_offset, revenue, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
        if (selection_flags[ITEM]) {
          int hash = (brand[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * (5*5*40));
          res[hash * 4] = year[ITEM];
          res[hash * 4 + 1] = brand[ITEM];
          atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM]));
        }
      }
    }
  }
}

__global__
void build_hashtable_s(bool *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset]) {
      int key = dim_key[offset];
      int hash = HASH(key, num_slots);
      atomicCAS(&hash_table[hash], 0, key);
    }
  }
}

__global__
void build_hashtable_p(bool *filter_col, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset]) {
      int key = dim_key[offset];
      int val = dim_val[offset];
      int hash = HASH(key, num_slots);

      atomicCAS(&hash_table[hash << 1], 0, key);
      hash_table[(hash << 1) + 1] = val;
    }
  }
}

__global__
void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int key = dim_key[offset];
    int val = dim_val[offset];
    int hash = HASH_WM(key, num_slots, val_min);

    atomicCAS(&hash_table[hash << 1], 0, key);
    hash_table[(hash << 1) + 1] = val;
  }
}

void build_hashtable_p_CPU(int *filter_col, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, bool* sb_p) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == 1) {
        sb_p[i] = true;
        int key = dim_key[i];
        int val = dim_val[i];
        int hash = HASH(key, num_slots);
        hash_table[hash << 1] = key;
        hash_table[(hash << 1) + 1] = val;
      }
    }
  });
}

void build_hashtable_s_CPU(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots, bool* sb_s) {
  // Build hashtable s
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == 1) {
        sb_s[i] = true;
        int key = dim_key[i];
        //int val = d_year[i];
        int hash = HASH(key, num_slots);
        hash_table[hash << 1] = key;
      }
    }
  });
}

void build_hashtable_d_CPU(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  // Build hashtable d
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

void probeCPU(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int CPU_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_val_len, int d_val_min,
    int* res, int num_slots, int start_index) {

  // Probe
  parallel_for(blocked_range<size_t>(0, CPU_len, CPU_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned long long local_revenue = 0;
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash, slot;
        hash = HASH(lo_partkey[start_index + i], p_len);
        long long p_slot = reinterpret_cast<long long*>(ht_p)[hash];
        if (p_slot  != 0) {
          int brand = p_slot >> 32;
          hash = HASH(lo_suppkey[start_index + i], s_len);
          slot = ht_s[hash << 1];
          if (slot != 0) {
            hash = HASH_WM(lo_orderdate[start_index + i], d_val_len, d_val_min);
            int year = ht_d[(hash << 1) + 1];
            hash = (brand * 7 +  (year - 1992)) % num_slots;
            res[hash * 4] = year;
            res[hash * 4 + 1] = brand;
            //res[hash * 4 + 2] += lo_revenue[start_index + i];
			__atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(lo_revenue[i + start_index]), __ATOMIC_RELAXED);
          }
        }
      }
    }
    for (int i = end_batch ; i < end; i++) {
      int hash = HASH(lo_suppkey[start_index + i], s_len);
      int slot = ht_s[hash << 1];
      if (slot != 0) {
        hash = HASH(lo_partkey[start_index + i], p_len);
        slot = ht_p[hash << 1];
        if (slot != 0) {
          int brand = ht_p[(hash<<1) + 1];
          hash = HASH_WM(lo_orderdate[start_index + i], d_val_len, d_val_min);
          int year = ht_d[(hash << 1) + 1];
          hash = (brand * 7 +  (year - 1992)) % ((1998-1992+1) * 1000);
          res[hash * 4] = year;
          res[hash * 4 + 1] = brand;
          //res[hash * 4 + 2] += lo_revenue[start_index + i];
		  __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(lo_revenue[i + start_index]), __ATOMIC_RELAXED);
        }
      }
    }
  });

}


/*
Implementation of q21
select sum(lo_revenue),d_year,p_brand1 from lineorder,part,supplier,date where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year,p_brand1 order by d_year,p_brand1;
*/

float runQuery(
    int* h_lo_orderdate, int* h_lo_partkey, int* h_lo_suppkey, int* h_lo_revenue, int lo_len,
    int* h_p_partkey, int* h_p_brand1, int* h_p_category, int p_len,
    int* h_d_datekey, int* h_d_year, int d_len,
    int* h_s_suppkey, int* h_s_region, int s_len,
    int* d_lo_orderdate, int* d_lo_partkey, int* d_lo_suppkey, int* d_lo_revenue, int GPU_len,
    int* d_p_partkey, int* d_p_brand1,
    int* d_d_datekey, int* d_d_year,
    int* d_s_suppkey,    
    CachingDeviceAllocator&  g_allocator) {
  
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  /**************************/
  // RUN SELECTION IN CPU

  bool *sb_p = new bool[p_len];
  bool *sb_s = new bool[s_len];

  memset(sb_p, 0, p_len * sizeof(bool));
  memset(sb_s, 0, s_len * sizeof(bool));

  int d_val_len = 19981230 - 19920101 + 1;
  int d_val_min = 19920101;

  int *h_ht_d = (int*)malloc(2 * d_val_len * sizeof(int));
  int *h_ht_p = (int*)malloc(2 * p_len * sizeof(int));
  int *h_ht_s = (int*)malloc(2 * s_len * sizeof(int));

  memset(h_ht_d, 0, 2 * d_val_len * sizeof(int));
  memset(h_ht_p, 0, 2 * p_len * sizeof(int));
  memset(h_ht_s, 0, 2 * s_len * sizeof(int));

  build_hashtable_d_CPU(h_d_datekey, h_d_year, d_len, h_ht_d, d_val_len, d_val_min);
  
  build_hashtable_s_CPU(h_s_region, h_s_suppkey, s_len, h_ht_s, s_len, sb_s);

  build_hashtable_p_CPU(h_p_category, h_p_partkey, h_p_brand1, p_len, h_ht_p, p_len, sb_p);

  bool *d_sb_p, *d_sb_s;
  g_allocator.DeviceAllocate((void**)&d_sb_p, p_len * sizeof(bool));
  g_allocator.DeviceAllocate((void**)&d_sb_s, s_len * sizeof(bool));

  // Copy the selection output to GPU
  cudaMemcpy(d_sb_p, sb_p, p_len * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sb_s, sb_s, s_len * sizeof(bool), cudaMemcpyHostToDevice);

  /**************************/
  // BUILD HASH TABLES IN GPU
  
  int *d_ht_d, *d_ht_p, *d_ht_s;
  g_allocator.DeviceAllocate((void**)&d_ht_d, 2 * d_val_len * sizeof(int));
  g_allocator.DeviceAllocate((void**)&d_ht_p, 2 * p_len * sizeof(int));
  g_allocator.DeviceAllocate((void**)&d_ht_s, 2 * s_len * sizeof(int));

  cudaMemset(d_ht_d, 0, 2 * d_val_len * sizeof(int));
  cudaMemset(d_ht_p, 0, 2 * p_len * sizeof(int));
  cudaMemset(d_ht_s, 0, 2 * s_len * sizeof(int));
 
  build_hashtable_s<<<(s_len + 127)/128, 128>>>(d_sb_s, d_s_suppkey, s_len, d_ht_s, s_len);

  build_hashtable_p<<<(p_len + 127)/128, 128>>>(d_sb_p, d_p_partkey, d_p_brand1, p_len, d_ht_p, p_len);

  build_hashtable_d<<<(d_len + 127)/128, 128>>>(d_d_datekey, d_d_year, d_len, d_ht_d, d_val_len, d_val_min);

  /**************************/
  // PROBE HASH TABLE IN GPU

   
  int *d_res;
  int res_size = ((1998-1992+1) * (5 * 5 * 40));
  int res_array_size = res_size * 4;
     
  g_allocator.DeviceAllocate((void**)&d_res, res_array_size * sizeof(int));

  cudaMemset(d_res, 0, res_array_size * sizeof(int));

  int tile_items = 128*4;
  probe<128,4><<<(GPU_len + tile_items - 1)/tile_items, 128>>>(d_lo_orderdate,
    d_lo_partkey, d_lo_suppkey, d_lo_revenue, GPU_len, d_ht_s, s_len, d_ht_p, p_len, d_ht_d, d_val_len, d_res);

  int* res = new int[res_array_size];
  memset(res, 0, res_array_size * sizeof(int));
  cudaMemcpy(res, d_res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost);

  /**************************/
  // PROBE HASH TABLE IN CPU

  int CPU_len = lo_len - GPU_len;

  probeCPU(h_lo_orderdate, h_lo_partkey, h_lo_suppkey, h_lo_revenue, CPU_len, 
    h_ht_s, s_len, h_ht_p, p_len, h_ht_d, d_val_len, d_val_min, res, res_array_size, GPU_len);

  // CALCULATE TIME

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop); 

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i<res_size; i++) {
    if (res[4*i] != 0) {
      cout << res[4*i] << " " << res[4*i + 1] << " " << reinterpret_cast<unsigned long long*>(&res[4*i + 2])[0]  << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete[] res;
  delete[] sb_s;
  delete[] sb_p;

  CLEANUP(d_res);
  CLEANUP(d_ht_d);
  CLEANUP(d_ht_p);
  CLEANUP(d_ht_s);
  CLEANUP(d_sb_s);
  CLEANUP(d_sb_p);

  return time_query;

}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_trials = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--t=<num trials>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  // Initialize device
  args.DeviceInit();

  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);

  int *h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *h_p_brand1 = loadColumn<int>("p_brand1", P_LEN);
  int *h_p_category = loadColumn<int>("p_category", P_LEN);

  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_region = loadColumn<int>("s_region", S_LEN);

  /*
   * Choose columns to be loaded to the GPU.
   * In this implementation, we only load
   * those columns that are required for join.
   */


  int GPU_LEN = 3001171;

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, GPU_LEN, g_allocator);
  int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, GPU_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, GPU_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, GPU_LEN, g_allocator);

  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);

  int *d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_p_brand1 = loadToGPU<int>(h_p_brand1, P_LEN, g_allocator);

  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(
        h_lo_orderdate, h_lo_partkey, h_lo_suppkey, h_lo_revenue, LO_LEN,
        h_p_partkey, h_p_brand1, h_p_category, P_LEN,
        h_d_datekey, h_d_year, D_LEN,
        h_s_suppkey, h_s_region, S_LEN,
        d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, GPU_LEN,
        d_p_partkey, d_p_brand1,
        d_d_datekey, d_d_year,
        d_s_suppkey,
        g_allocator);
    cout<< "{"
        << "\"query\":21"
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}