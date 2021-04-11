#include <cub/cub.cuh>
#include "tbb/tbb.h"

using namespace std;
using namespace cub;
using namespace tbb;

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