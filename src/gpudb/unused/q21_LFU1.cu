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

struct slot {
  int year;
  int brand1;
  std::atomic<long long> revenue;
};

/**
 * Globals, constants and typedefs
 */
bool g_verbose = false;  // Whether to display input/output to console
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
__global__ void probe(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* t_table,
    int *total) {

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
  int p_offset[ITEMS_PER_THREAD];
  int s_offset[ITEMS_PER_THREAD];
  int d_offset[ITEMS_PER_THREAD];
  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0;
  __shared__ int block_off;

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
        s_offset[ITEM] = slot - 1;
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
        int slot = ht_p[hash];
        if (slot != 0) {
          p_offset[ITEM] = slot - 1;
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
        int slot = ht_d[hash];
        if (slot != 0) {
	        t_count++; // TODO: check this. count of items that have selection_flag = 1
          d_offset[ITEM] = slot - 1;
        } else {
          selection_flags[ITEM] = 0;
        }
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
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int slot = ht_s[hash];
        if (slot != 0) {
          s_offset[ITEM] = slot - 1;
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
          int slot = ht_p[hash];
          if (slot != 0) {
            p_offset[ITEM] = slot - 1;
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

      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        if (selection_flags[ITEM]) {
          int slot = ht_d[hash];
          if (slot != 0) {
            t_count++;
            d_offset[ITEM] = slot - 1;
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
  BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count);
  if(threadIdx.x == blockDim.x - 1) {
      block_off = atomicAdd(total, t_count+c_t_count);
  }

  __syncthreads();

  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        t_table[offset << 2] = s_offset[ITEM];
        t_table[(offset << 2) + 1] = p_offset[ITEM];
        t_table[(offset << 2) + 2] = d_offset[ITEM];
        t_table[(offset << 2) + 3] = blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
      }
    }
  }
}

__global__
void build_hashtable_s(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset] == 1) {
      int hash = HASH(dim_key[offset], num_slots);
      atomicCAS(&hash_table[hash], 0, offset+1); // TODO: why do we need atomicCAS?
    }
  }
}

__global__
void build_hashtable_p(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset] == 1) {
      int hash = HASH(dim_key[offset], num_slots);
      atomicCAS(&hash_table[hash], 0, offset+1); // TODO: what about offset of 0?? Right now the 0th entry is not selected, so we are safe
    }
  }
}

__global__
void build_hashtable_d(int *dim_key, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int hash = HASH_WM(dim_key[offset], num_slots, val_min);
    atomicCAS(&hash_table[hash], 0, offset+1);
  }
}

void runAggregationCPU(int* lo_revenue, int* p_brand1, int* d_year, int* t_table, int lo_len, slot* res, int num_slots) {
  parallel_for(blocked_range<size_t>(0, lo_len, lo_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        //__int128 slot = *reinterpret_cast<__int128*> (&t_table[i << 2]);
        //if (t_table[(i << 2) + 2] != 0) {
          int brand = p_brand1[t_table[(i << 2) + 1]];
          int year = d_year[t_table[(i << 2) + 2]];
          int hash = (brand * 7 + (year - 1992)) % num_slots;
          res[hash].year = year;
          res[hash].brand1 = brand;
          res[hash].revenue += lo_revenue[t_table[(i << 2) + 3]];
        //}
      }
    }
    for (int i = end_batch ; i < end; i++) {
      //__int128 slot = *reinterpret_cast<__int128*> (&t_table[i << 2]);
      //if (t_table[i << 2] != 0) {
        int brand = p_brand1[t_table[(i << 2)+ 1]];
        int year = d_year[t_table[(i << 2) + 2]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash].year = year;
        res[hash].brand1 = brand;
        res[hash].revenue += lo_revenue[t_table[(i << 2) + 3]];
      //}
    }
  });
}


/*
Implementation of q21
select sum(lo_revenue),d_year,p_brand1 from lineorder,part,supplier,date where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year,p_brand1 order by d_year,p_brand1;
*/

float runQuery(
    int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int lo_len,
    int* p_partkey, int* p_brand1, int* p_category, int p_len,
    int *d_datekey, int* d_year, int d_len,
    int *s_suppkey, int* s_region, int s_len,
    CachingDeviceAllocator&  g_allocator) {
  
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;

  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  /**************************/
  // BUILD HASH TABLES IN GPU

  int *ht_d, *ht_p, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  int d_val_min = 19920101;
  g_allocator.DeviceAllocate((void**)&ht_d, d_val_len * sizeof(int)); // TODO: do we need 2*d_val_len?
  g_allocator.DeviceAllocate((void**)&ht_p, p_len * sizeof(int));
  g_allocator.DeviceAllocate((void**)&ht_s, s_len * sizeof(int));

  cudaMemset(ht_d, 0, d_val_len * sizeof(int));
  cudaMemset(ht_p, 0, p_len * sizeof(int));
  cudaMemset(ht_s, 0, s_len * sizeof(int));
 
  build_hashtable_s<<<(s_len + 127)/128, 128>>>(s_region, s_suppkey, s_len, ht_s, s_len);

  build_hashtable_p<<<(p_len + 127)/128, 128>>>(p_category, p_partkey, p_len, ht_p, p_len);

  build_hashtable_d<<<(d_len + 127)/128, 128>>>(d_datekey, d_len, ht_d, d_val_len, d_val_min);

  /**************************/
  // PROBE HASH TABLE IN GPU

  int t_len = lo_len * 4; // l0_len*4 is the max entries required
  int *t_table;
  
  g_allocator.DeviceAllocate((void**)&t_table, t_len * sizeof(int));
  //cudaMemset(t_table, 0, t_len * sizeof(int));

  int tile_items = 128*4;
  int *total;
  int h_total;
  cudaMalloc((void **)&total, sizeof(int));

  probe<128,4><<<(lo_len + tile_items - 1)/tile_items, 128>>>(lo_orderdate,
          lo_partkey, lo_suppkey, lo_len, ht_s, s_len, ht_p, p_len, ht_d, d_val_len, t_table, total);

  cudaMemcpy(&h_total, total, sizeof(int), cudaMemcpyDeviceToHost);

  t_len = h_total * 4;
  int *h_t_table = new int[t_len];

  //memset(h_t_table, 0, t_len * sizeof(int));
  cudaMemcpy(h_t_table, t_table, t_len * sizeof(int), cudaMemcpyDeviceToHost);

  int num_slots = (1998-1992+1) * 1000; //((1998-1992+1) * 1000); TODO: need to figure this out
  slot* res = new slot[num_slots];

  memset(res, 0, sizeof(slot)*num_slots); // TODO: this can save around 10ms but res.revenue?
  /**************************/
  // RUN AGGREGATION IN CPU
  runAggregationCPU(lo_revenue, p_brand1, d_year, h_t_table, h_total, res, num_slots);

  // CALCULATE TIME

  finish = chrono::high_resolution_clock::now();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop); 

  std::chrono::duration<double> diff = finish - st;

  cout << "Result:" << endl;

  int res_count = 0;
  for (int i=0; i<num_slots; i++) {
    if (res[i].year != 0) {
      cout << res[i].year << " " << res[i].brand1 << " " << res[i].revenue << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete[] res;
  delete[] h_t_table;

  CLEANUP(t_table);
  CLEANUP(ht_d);
  CLEANUP(ht_p);
  CLEANUP(ht_s);

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
  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_p_partkey = loadToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_p_category = loadToGPU<int>(h_p_category, P_LEN, g_allocator);
  int *d_s_region = loadToGPU<int>(h_s_region, S_LEN, g_allocator);

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(
        d_lo_orderdate, d_lo_partkey, d_lo_suppkey, h_lo_revenue, LO_LEN,
        d_p_partkey, h_p_brand1, d_p_category, P_LEN,
        d_d_datekey, h_d_year, D_LEN,
        d_s_suppkey, d_s_region, S_LEN,
        g_allocator);
    cout<< "{"
        << "\"query\":21"
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}
