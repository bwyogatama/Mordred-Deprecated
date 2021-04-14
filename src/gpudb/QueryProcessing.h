#include "QueryOptimizer.h"
#include <cub/cub.cuh>
#include "tbb/tbb.h"

#include <chrono>
#include <atomic>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>

using namespace std;
using namespace cub;
using namespace tbb;

#define HASH_WM(X,Y,Z) ((X-Z) % Y)

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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* dim_key4,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int fact_len, int *total, int start_offset) {

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
  int dim_offset4[ITEMS_PER_THREAD];
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

  __syncthreads();

  /********************
    Not the last tile
    ******************/
  if (!is_last_tile) {
    if (dim_key1 != NULL) {
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
            dim_offset1[ITEM] = slot - 1;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }

    __syncthreads();

    if (dim_key2 != NULL) {
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
            dim_offset2[ITEM] = slot - 1;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }

    __syncthreads();

    if (dim_key3 != NULL) {
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
            dim_offset3[ITEM] = slot - 1;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    }

    __syncthreads();

    if (dim_key4 != NULL) {
      BlockLoadInt(temp_storage.load_items).Load(dim_key4 + tile_offset, items);

      // Barrier for smem reuse
      __syncthreads();

      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        // Out-of-bounds items are selection_flags
        int hash = HASH(items[ITEM], dim_len4, min_key4); //19920101
        if (selection_flags[ITEM]) {
          int slot = ht4[(hash << 1) + 1];
          if (slot != 0) {
            t_count++;
            dim_offset4[ITEM] = slot - 1;
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
          if (selection_flags[ITEM]) {
            t_count++;
          }
        }
      }
    }

  }
  else {

    if (dim_key1 != NULL) {
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
              dim_offset1[ITEM] = slot - 1;
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    }

    __syncthreads();

    if (dim_key2 != NULL) {
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
              dim_offset2[ITEM] = slot - 1;
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    }

    __syncthreads();

    if (dim_key3 != NULL) {
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
              dim_offset3[ITEM] = slot - 1;
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    }

    __syncthreads();

    if (dim_key4 != NULL) {
      BlockLoadInt(temp_storage.load_items).Load(dim_key4 + tile_offset, items, num_tile_items);

      // Barrier for smem reuse
      __syncthreads();

      /*
       * Join with date table.
       */
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
          int hash = HASH(items[ITEM], dim_len4, min_key4); //19920101
          if (selection_flags[ITEM]) {
            int slot = ht4[(hash << 1) + 1];
            if (slot != 0) {
              t_count++;
              dim_offset4[ITEM] = slot - 1;
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
          if (selection_flags[ITEM]) {
            t_count++;
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
        lo_off[offset] = start_offset + blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
        if (dim_off1 != NULL) dim_off1[offset] = dim_offset1[ITEM];
        if (dim_off2 != NULL) dim_off2[offset] = dim_offset2[ITEM];
        if (dim_off3 != NULL) dim_off3[offset] = dim_offset3[ITEM];
        if (dim_off4 != NULL) dim_off4[offset] = dim_offset4[ITEM];
      }
    }
  }
}

void probe_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int h_total, int start_offset, int* offset) {

  // Probe
  parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    int count = 0;
    assert(lo_off != NULL);
    assert(h_lo_off != NULL);
    int temp[5][end-start];
    //printf("start = %d end = %d\n", start, end);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot1, slot2, slot3, slot4;
        int lo_offset = lo_off[start_offset + i];
        if (dimkey_col1 != NULL) {
          hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
          slot1 = ht1[(hash << 1) + 1];
        } else {
          slot1 = 1;
          if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;
        }
        if (slot1 != 0) {
          if (dimkey_col2 != NULL) {
            hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
            slot2 = ht2[(hash << 1) + 1];
          } else {
            slot2 = 1;
            if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;
          }
          if (slot2 != 0) {
            if (dimkey_col3 != NULL) {
              hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
              slot3 = ht3[(hash << 1) + 1];
            } else {
              slot3 = 1;
              if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;
            }
            if (slot3 != 0) {
              if (dimkey_col4 != NULL) {
                hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
                slot4 = ht4[(hash << 1) + 1];
              } else {
                slot4 = 1;
                if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;
              }
              if (slot4 != 0) {
                temp[0][count] = lo_offset;
                temp[1][count] = slot1-1;
                temp[2][count] = slot2-1;
                temp[3][count] = slot3-1;
                temp[4][count] = slot4-1;
                count++;
              }
            }
          }
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot1, slot2, slot3, slot4;
      int lo_offset = lo_off[start_offset + i];
      if (dimkey_col1 != NULL) {
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot1 = ht1[(hash << 1) + 1];
      } else {
        slot1 = 1;
        if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;
      }
      if (slot1 != 0) {
        if (dimkey_col2 != NULL) {
          hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
          slot2 = ht2[(hash << 1) + 1];
        } else {
          slot2 = 1;
          if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;
        }
        if (slot2 != 0) {
          if (dimkey_col3 != NULL) {
            hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
            slot3 = ht3[(hash << 1) + 1];
          } else {
            slot3 = 1;
            if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;
          }
          if (slot3 != 0) {
            if (dimkey_col4 != NULL) {
              hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
              slot4 = ht4[(hash << 1) + 1];
            } else {
              slot4 = 1;
              if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;
            }
            if (slot4 != 0) {
              temp[0][count] = lo_offset;
              temp[1][count] = slot1-1;
              temp[2][count] = slot2-1;
              temp[3][count] = slot3-1;
              temp[4][count] = slot4-1;
              count++;
            }
          }
        }
      }
    }
    //printf("count = %d\n", count);
    int thread_off = __atomic_fetch_add(offset, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      h_lo_off[thread_off+i] = temp[0][i];
      if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  });
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* dim_key4, int* aggr, 
  int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4,
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4) {

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
  int groupval4[ITEMS_PER_THREAD];
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

  __syncthreads();

  /********************
    Not the last tile
    ******************/
  if (!is_last_tile) {

    if (dim_key1 != NULL) {
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
            groupval1[ITEM] = (slot >> 32);
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }

    }
    else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        groupval1[ITEM] = 0;
      }
    }

    __syncthreads();

    if (dim_key2 != NULL) {
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
            groupval2[ITEM] = (slot >> 32);
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        groupval2[ITEM] = 0;
      }
    }

    __syncthreads();

    if (dim_key3 != NULL) {
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
            groupval3[ITEM] = (slot >> 32);
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        groupval3[ITEM] = 0;
      }
    }

    __syncthreads();

    if (dim_key4 != NULL) {
      BlockLoadInt(temp_storage.load_items).Load(dim_key4 + tile_offset, items);

      // Barrier for smem reuse
      __syncthreads();

      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        // Out-of-bounds items are selection_flags
        int hash = HASH_WM(items[ITEM], dim_len4, min_key4); //19920101
        if (selection_flags[ITEM]) {
          uint64_t slot = *reinterpret_cast<uint64_t*>(&ht4[hash << 1]);
          if (slot != 0) {
            groupval4[ITEM] = (slot >> 32);
          } else {
            selection_flags[ITEM] = 0;
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        groupval4[ITEM] = 0;
      }
    }

    __syncthreads();

    BlockLoadInt(temp_storage.load_items).Load(aggr + tile_offset, aggrval);

    // Barrier for smem reuse
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3) * unique_val3 + (groupval4[ITEM] - min_val4) * unique_val4) % total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval[ITEM]));
      }
    }
  }
  else {

    if (dim_key1 != NULL) {
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
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
          groupval1[ITEM] = 0;
        }
      }
    }

    __syncthreads();

    if (dim_key2 != NULL) {
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
              groupval2[ITEM] = (slot >> 32);
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
          groupval2[ITEM] = 0;
        }
      }
    }

    __syncthreads();

    if (dim_key3 != NULL) {
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
              groupval3[ITEM] = (slot >> 32);
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
          groupval3[ITEM] = 0;
        }
      }  
    }

    __syncthreads();

    if (dim_key4 != NULL) {
      BlockLoadInt(temp_storage.load_items).Load(dim_key4 + tile_offset, items, num_tile_items);

      // Barrier for smem reuse
      __syncthreads();

      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
          // Out-of-bounds items are selection_flags
          int hash = HASH_WM(items[ITEM], dim_len4, min_key4); //19920101
          if (selection_flags[ITEM]) {
            uint64_t slot = *reinterpret_cast<uint64_t*>(&ht4[hash << 1]);
            if (slot != 0) {
              groupval4[ITEM] = (slot >> 32);
            } else {
              selection_flags[ITEM] = 0;
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
          groupval4[ITEM] = 0;
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
          int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3) * unique_val3 + (groupval4[ITEM] - min_val4) * unique_val4) % total_val; //!
          res[hash * 6] = groupval1[ITEM];
          res[hash * 6 + 1] = groupval2[ITEM];
          res[hash * 6 + 2] = groupval3[ITEM];
          res[hash * 6 + 3] = groupval4[ITEM];
          atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval[ITEM]));
        }
      }
    }
  }
}

void probe_group_by_CPU(int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
  int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int start_index) {

  // Probe
  parallel_for(blocked_range<size_t>(0, fact_len, fact_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1, dim_val2, dim_val3, dim_val4;
        if (dimkey_col1 != NULL) {
          hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          dim_val1 = slot >> 32;
        } else {
          slot = 1;
          dim_val1 = 0;
        }
        if (slot != 0) {
          if (dimkey_col2 != NULL) {
            hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
            slot = reinterpret_cast<long long*>(ht2)[hash];
            dim_val2 = slot >> 32;
          } else {
            slot = 1;
            dim_val2 = 0;
          }
          if (slot != 0) {
            if (dimkey_col3 != NULL) {
              hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
              slot = reinterpret_cast<long long*>(ht3)[hash];
              dim_val3 = slot >> 32;
            } else {
              slot = 1;
              dim_val3 = 0;
            }
            if (slot != 0) {
              if (dimkey_col4 != NULL) {
                hash = HASH_WM(dimkey_col4[start_index + i], dim_len4, min_key4);
                slot = reinterpret_cast<long long*>(ht4)[hash];
                dim_val4 = slot >> 32;
              } else {
                slot = 1;
                dim_val4 = 0;
              }
              if (slot != 0) {
                hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
                res[hash * 6] = dim_val1;
                res[hash * 6 + 1] = dim_val2;
                res[hash * 6 + 2] = dim_val3;
                res[hash * 6 + 3] = dim_val4;
                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
              }
            }
          }
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      int dim_val1, dim_val2, dim_val3, dim_val4;
      if (dimkey_col1 != NULL) {
        hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        dim_val1 = slot >> 32;
      } else {
        slot = 1;
        dim_val1 = 0;
      }
      if (slot != 0) {
        if (dimkey_col2 != NULL) {
          hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          dim_val2 = slot >> 32;
        } else {
          slot = 1;
          dim_val2 = 0;
        }
        if (slot != 0) {
          if (dimkey_col3 != NULL) {
            hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
            slot = reinterpret_cast<long long*>(ht3)[hash];
            dim_val3 = slot >> 32;
          } else {
            slot = 1;
            dim_val3 = 0;
          }
          if (slot != 0) {
            if (dimkey_col4 != NULL) {
              hash = HASH_WM(dimkey_col4[start_index + i], dim_len4, min_key4);
              slot = reinterpret_cast<long long*>(ht4)[hash];
              dim_val4 = slot >> 32;
            } else {
              slot = 1;
              dim_val4 = 0;
            }
            if (slot != 0) {
              hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
              res[hash * 6] = dim_val1;
              res[hash * 6 + 1] = dim_val2;
              res[hash * 6 + 2] = dim_val3;
              res[hash * 6 + 3] = dim_val4;
              __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
            }
          }
        }
      }
    }
  });
}

void probe_group_by_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int fact_len, int start_index) {

  // Probe
  parallel_for(blocked_range<size_t>(0, fact_len, fact_len/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1, dim_val2, dim_val3, dim_val4;
        int lo_offset = lo_off[start_index + i];
        if (dim_off1 == NULL && dimkey_col1 != NULL) {
          hash = HASH_WM(dimkey_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          dim_val1 = slot >> 32;
        } else if (dim_off1 != NULL && dimkey_col1 != NULL){
          slot = 1;
          dim_val1 = dimkey_col1[dim_off1[start_index + i]];
        } else {
          slot = 1;
          dim_val1 = 0;
        }
        if (slot != 0) {
          if (dim_off2 == NULL && dimkey_col2 != NULL) {
            hash = HASH_WM(dimkey_col2[lo_offset], dim_len2, min_key2);
            slot = reinterpret_cast<long long*>(ht2)[hash];
            dim_val2 = slot >> 32;
          } else if (dim_off2 != NULL && dimkey_col2 != NULL){
            slot = 1;
            dim_val2 = dimkey_col2[dim_off2[start_index + i]];
          } else {
            slot = 1;
            dim_val2 = 0;
          }
          if (slot != 0) {
            if (dim_off3 == NULL && dimkey_col3 != NULL) {
              hash = HASH_WM(dimkey_col3[lo_offset], dim_len3, min_key3);
              slot = reinterpret_cast<long long*>(ht3)[hash];
              dim_val3 = slot >> 32;
            } else if (dim_off3 != NULL && dimkey_col3 != NULL){
              slot = 1;
              dim_val3 = dimkey_col3[dim_off3[start_index + i]];
            } else {
              slot = 1;
              dim_val3 = 0;
            }
            if (slot != 0) {
              if (dim_off4 == NULL && dimkey_col4 != NULL) {
                hash = HASH_WM(dimkey_col4[lo_offset], dim_len4, min_key4);
                slot = reinterpret_cast<long long*>(ht4)[hash];
                dim_val4 = slot >> 32;
              } else if (dim_off4 != NULL && dimkey_col4 != NULL){
                slot = 1;
                dim_val4 = dimkey_col4[dim_off4[start_index + i]];
              } else {
                slot = 1;
                dim_val4 = 0;
              }
              if (slot != 0) {
                hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
                res[hash * 6] = dim_val1;
                res[hash * 6 + 1] = dim_val2;
                res[hash * 6 + 2] = dim_val3;
                res[hash * 6 + 3] = dim_val4;
                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[lo_offset]), __ATOMIC_RELAXED);
              }
            }
          }
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      int dim_val1, dim_val2, dim_val3, dim_val4;
      int lo_offset = lo_off[start_index + i];
      if (dim_off1 == NULL && dimkey_col1 != NULL) {
        hash = HASH_WM(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        dim_val1 = slot >> 32;
      } else if (dim_off1 != NULL && dimkey_col1 != NULL){
        slot = 1;
        dim_val1 = dimkey_col1[dim_off1[start_index + i]];
      } else {
        slot = 1;
        dim_val1 = 0;
      }
      if (slot != 0) {
        if (dim_off2 == NULL && dimkey_col2 != NULL) {
          hash = HASH_WM(dimkey_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          dim_val2 = slot >> 32;
        } else if (dim_off2 != NULL && dimkey_col2 != NULL){
          slot = 1;
          dim_val2 = dimkey_col2[dim_off2[start_index + i]];
        } else {
          slot = 1;
          dim_val2 = 0;
        }
        if (slot != 0) {
          if (dim_off3 == NULL && dimkey_col3 != NULL) {
            hash = HASH_WM(dimkey_col3[lo_offset], dim_len3, min_key3);
            slot = reinterpret_cast<long long*>(ht3)[hash];
            dim_val3 = slot >> 32;
          } else if (dim_off3 != NULL && dimkey_col3 != NULL){
            slot = 1;
            dim_val3 = dimkey_col3[dim_off3[start_index + i]];
          } else {
            slot = 1;
            dim_val3 = 0;
          }
          if (slot != 0) {
            if (dim_off4 == NULL && dimkey_col4 != NULL) {
              hash = HASH_WM(dimkey_col4[lo_offset], dim_len4, min_key4);
              slot = reinterpret_cast<long long*>(ht4)[hash];
              dim_val4 = slot >> 32;
            } else if (dim_off4 != NULL && dimkey_col4 != NULL){
              slot = 1;
              dim_val4 = dimkey_col4[dim_off4[start_index + i]];
            } else {
              slot = 1;
              dim_val4 = 0;
            }
            if (slot != 0) {
              hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
              res[hash * 6] = dim_val1;
              res[hash * 6 + 1] = dim_val2;
              res[hash * 6 + 2] = dim_val3;
              res[hash * 6 + 3] = dim_val4;
              __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[lo_offset]), __ATOMIC_RELAXED);
            }
          }
        }
      }
    }
  });

}

__global__
void build_GPU(int* dim_key, int* dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int segment_number, int isoffset) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int key = dim_key[offset];
    int value;
    if (isoffset == 1) value = segment_number * SEGMENT_SIZE + offset + 1;
    else if (isoffset == 0) value = dim_val[offset];
    else value = 0;
    int hash = HASH(key, num_slots, val_min);
    atomicCAS(&hash_table[hash << 1], 0, key);
    hash_table[(hash << 1) + 1] = value;
  }
}

__global__
void build_filter_GPU(int* filter_col, int compare, int* dim_key, int* dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int segment_number, int isoffset) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (filter_col[offset] == compare) {
      int key = dim_key[offset];
      int value;
      if (isoffset == 1) value = segment_number * SEGMENT_SIZE + offset + 1;
      else if (isoffset == 0) value = dim_val[offset];
      else value = 0;
      int hash = HASH(key, num_slots, val_min);
      atomicCAS(&hash_table[hash << 1], 0, key);
      hash_table[(hash << 1) + 1] = value;
    }
  }
}

void build_CPU(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int isoffset) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      int key = dim_key[i];
      int hash = HASH_WM(key, num_slots, val_min);
      hash_table[hash << 1] = key;
      if (isoffset == 1) hash_table[(hash << 1) + 1] = i + 1;
      else if (isoffset == 0) hash_table[(hash << 1) + 1] = dim_val[i];
      else hash_table[(hash << 1) + 1] = 0;
    }
  });
}

void build_filter_CPU(int *filter_col, int compare, int *dim_key, int* dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int isoffset) {
  // Build hashtable p
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    for (int i = range.begin(); i < range.end(); i++) {
      if (filter_col[i] == compare) {
        int key = dim_key[i];
        int hash = HASH(key, num_slots, val_min);
        hash_table[hash << 1] = key;
        if (isoffset == 1) hash_table[(hash << 1) + 1] = i + 1;
        else if (isoffset == 0) hash_table[(hash << 1) + 1] = dim_val[i];
        else hash_table[(hash << 1) + 1] = 0;
      }
    }
  });
}

__global__
void runAggregationQ2GPU(int* gpuCache, int* lo_idx, int* p_idx, int* d_idx, int* lo_off, int* part_off, int* date_off, int num_tuples, int* res, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < num_tuples) {
    int revenue_idx = lo_off[offset];
    int brand_idx = part_off[offset];
    int year_idx = date_off[offset];

    int revenue_seg = lo_idx[revenue_idx / SEGMENT_SIZE];
    int brand_seg = p_idx[brand_idx / SEGMENT_SIZE];
    int year_seg = d_idx[year_idx / SEGMENT_SIZE];

    int revenue = gpuCache[revenue_seg * SEGMENT_SIZE + (revenue_idx % SEGMENT_SIZE)];
    int brand = gpuCache[brand_seg * SEGMENT_SIZE + (brand_idx % SEGMENT_SIZE)];
    int year = gpuCache[year_seg * SEGMENT_SIZE + (year_idx % SEGMENT_SIZE)];

    int hash = (brand * 7 + (year - 1992)) % num_slots;

    res[hash * 6] = 0;
    res[hash * 6 + 1] = brand;
    res[hash * 6 + 2] = year;
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(revenue));

  }
}

void runAggregationQ2CPU(int* lo_revenue, int* p_brand1, int* d_year, int* lo_off, int* part_off, int* date_off, int num_tuples, int* res, int num_slots) {
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int brand = p_brand1[part_off[i]];
        int year = d_year[date_off[i]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[lo_off[i]]), __ATOMIC_RELAXED);
      }
    }
    for (int i = end_batch ; i < end; i++) {
        int brand = p_brand1[part_off[i]];
        int year = d_year[date_off[i]];
        int hash = (brand * 7 + (year - 1992)) % num_slots;
        res[hash * 6 + 1] = brand;
        res[hash * 6 + 2] = year;
        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[lo_off[i]]), __ATOMIC_RELAXED);
    }
  });
}

// void probe_group_by_CPU(int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
//   int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
//   int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
//   int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int start_index) {

//   // Probe
//   parallel_for(blocked_range<size_t>(0, fact_len, fact_len/NUM_THREADS + 4), [&](auto range) {
//     int start = range.begin();
//     int end = range.end();
//     int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

//     for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
//       #pragma simd
//       for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
//         int hash;
//         int slot;
//         int dim_val1, dim_val2, dim_val3, dim_val4;
//         if (dimkey_col1 != NULL) {
//           hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
//           slot = ht1[hash << 1];
//           dim_val1 = ht1[(hash << 1) + 1];
//         } else {
//           slot = 1;
//           dim_val1 = 0;
//         }
//         if (slot != 0) {
//           if (dimkey_col2 != NULL) {
//             hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
//             slot = ht2[hash << 1];
//             dim_val2 = ht2[(hash << 1) + 1];
//           } else {
//             slot = 1;
//             dim_val2 = 0;
//           }
//           if (slot != 0) {
//             if (dimkey_col3 != NULL) {
//               hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
//               slot = ht3[hash << 1];
//               dim_val3 = ht3[(hash << 1) + 1];
//             } else {
//               slot = 1;
//               dim_val3 = 0;
//             }
//             if (slot != 0) {
//               if (dimkey_col4 != NULL) {
//                 hash = HASH_WM(dimkey_col4[start_index + i], dim_len4, min_key4);
//                 slot = ht4[hash << 1];
//                 dim_val4 = ht4[(hash << 1) + 1];
//               } else {
//                 slot = 1;
//                 dim_val4 = 0;
//               }
//               if (slot != 0) {
//                 hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
//                 res[hash * 6] = dim_val1;
//                 res[hash * 6 + 1] = dim_val2;
//                 res[hash * 6 + 2] = dim_val3;
//                 res[hash * 6 + 3] = dim_val4;
//                 __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
//               }
//             }
//           }
//         }
//       }
//     }

//     for (int i = end_batch ; i < end; i++) {
//       int hash;
//       int slot;
//       int dim_val1, dim_val2, dim_val3, dim_val4;
//       if (dimkey_col1 != NULL) {
//         hash = HASH_WM(dimkey_col1[start_index + i], dim_len1, min_key1);
//         slot = ht1[hash << 1];
//         dim_val1 = ht1[(hash << 1) + 1];
//       } else {
//         slot = 1;
//         dim_val1 = 0;
//       }
//       if (slot != 0) {
//         if (dimkey_col2 != NULL) {
//           hash = HASH_WM(dimkey_col2[start_index + i], dim_len2, min_key2);
//           slot = ht2[hash << 1];
//           dim_val2 = ht2[(hash << 1) + 1];
//         } else {
//           slot = 1;
//           dim_val2 = 0;
//         }
//         if (slot != 0) {
//           if (dimkey_col3 != NULL) {
//             hash = HASH_WM(dimkey_col3[start_index + i], dim_len3, min_key3);
//             slot = ht3[hash << 1];
//             dim_val3 = ht3[(hash << 1) + 1];
//           } else {
//             slot = 1;
//             dim_val3 = 0;
//           }
//           if (slot != 0) {
//             if (dimkey_col4 != NULL) {
//               hash = HASH_WM(dimkey_col4[start_index + i], dim_len4, min_key4);
//               slot = ht4[hash << 1];
//               dim_val4 = ht4[(hash << 1) + 1];
//             } else {
//               slot = 1;
//               dim_val4 = 0;
//             }
//             if (slot != 0) {
//               hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
//               res[hash * 6] = dim_val1;
//               res[hash * 6 + 1] = dim_val2;
//               res[hash * 6 + 2] = dim_val3;
//               res[hash * 6 + 3] = dim_val4;
//               __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[i + start_index]), __ATOMIC_RELAXED);
//             }
//           }
//         }
//       }
//     }
//   });

// }