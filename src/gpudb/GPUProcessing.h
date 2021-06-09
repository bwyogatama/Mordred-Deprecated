#ifndef _GPU_PROCESSING_H_
#define _GPU_PROCESSING_H_

#include <cub/cub.cuh>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>

#include "crystal/crystal.cuh"

using namespace cub;

#define CUB_STDERR

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    gpuErrchk(error); \
    exit(-1); \
  } \
}

#define cudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }

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
__device__ __forceinline__ void BlockProbeGPU(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  cudaAssert(ht != NULL);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int hash = HASH(items[ITEM], ht_len, keys_min);
      if (selection_flags[ITEM]) {  
        int slot = ht[(hash << 1) + 1];
        if (slot != 0) {
          offset[ITEM] = slot - 1;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

// template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
// __device__ __forceinline__ void BlockProbeGPU2(
//     int tid,
//     int  (&items)[ITEMS_PER_THREAD],
//     int  (&items_lo)[ITEMS_PER_THREAD],
//     int  (&offset)[ITEMS_PER_THREAD],
//     int  (&selection_flags)[ITEMS_PER_THREAD],
//     int* gpuCache,
//     int* dimkey_idx,
//     int* lo_off,
//     int* ht,
//     int ht_len,
//     int keys_min,
//     int num_items
//     ) {
//   cudaAssert(ht != NULL);
//   #pragma unroll
//   for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
//   {
//     if (tid + (ITEM * BLOCK_THREADS) < num_items) {
//       if (lo_off != NULL) {
//         cudaAssert(dimkey_idx != NULL);
//         int dimkey_seg = dimkey_idx[items_lo[ITEM] / SEGMENT_SIZE];
//         items[ITEM] = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_lo[ITEM] % SEGMENT_SIZE)];
//       }

//       // Out-of-bounds items are selection_flags
//       int hash = HASH(items[ITEM], ht_len, keys_min);
//       if (selection_flags[ITEM]) {
//         int slot = ht[(hash << 1) + 1];
//         if (slot != 0) {
//           offset[ITEM] = slot - 1;
//         } else {
//           selection_flags[ITEM] = 0;
//         }
//       }
//     }
//   }
// }

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGroupByGPU(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  cudaAssert(ht != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int hash = HASH(items[ITEM], ht_len, keys_min);
      if (selection_flags[ITEM]) {
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
        if (slot != 0) {
          res[ITEM] = (slot >> 32);
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}


// template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
// __device__ __forceinline__ void BlockProbeGroupByGPU2(
//     int tid,
//     int  (&items)[ITEMS_PER_THREAD],
//     int  (&items_lo)[ITEMS_PER_THREAD],
//     int  (&res)[ITEMS_PER_THREAD],
//     int  (&selection_flags)[ITEMS_PER_THREAD],
//     int* gpuCache,
//     int* dimkey_idx,
//     int* lo_off,
//     int* ht,
//     int ht_len,
//     int keys_min,
//     int num_items
//     ) {
//   cudaAssert(ht != NULL);
//   #pragma unroll
//   for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
//   {
//     if (tid + (ITEM * BLOCK_THREADS) < num_items) {
//       if (lo_off != NULL) {
//         cudaAssert(dimkey_idx != NULL);
//         int dimkey_seg = dimkey_idx[items_lo[ITEM] / SEGMENT_SIZE];
//         items[ITEM] = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_lo[ITEM] % SEGMENT_SIZE)];
//       }

//       // Out-of-bounds items are selection_flags
//       int hash = HASH(items[ITEM], ht_len, keys_min);
//       if (selection_flags[ITEM]) {
//         uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
//         if (slot != 0) {
//           res[ITEM] = (slot >> 32);
//           //printf("groupval1 = %d\n", groupval1[ITEM]);
//         } else {
//           selection_flags[ITEM] = 0;
//         }
//       }
//     }
//   }
// }

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPassThroughOffset(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        offset[ITEM] = items[ITEM];
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetGroupValZero(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&groupval)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) { // CUB use blocked arrangement
      if (selection_flags[ITEM]) {
        groupval[ITEM] = 0;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadFilteredOffset(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* col_idx,
    int num_items
    ) {

  cudaAssert(col_idx != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int seg = col_idx[items_off[ITEM] / SEGMENT_SIZE];
        items[ITEM] = gpuCache[seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPU(
    int tid,
    int blockid,
    int start_offset,
    int  (&keys)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  cudaAssert(ht != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {  // use stripe arrangement since we are using Crystal Blockload instead of CUB
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);
        int old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
        ht[(hash << 1) + 1] = start_offset + (blockid * tile_size) + (tid + ITEM * BLOCK_THREADS) + 1;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPU2(
    int tid,
    int  (&items)[ITEMS_PER_THREAD], //equal to items
    int* gpuCache,
    int* dimkey_idx,
    int* dimval_idx,
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {
  cudaAssert(ht != NULL);
  cudaAssert(dimkey_idx != NULL);
  cudaAssert(dimval_idx != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int dimkey_seg = dimkey_idx[items[ITEM] / SEGMENT_SIZE];
      int dimval_seg = dimval_idx[items[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items[ITEM] % SEGMENT_SIZE)];
      int val = gpuCache[dimval_seg * SEGMENT_SIZE + (items[ITEM] % SEGMENT_SIZE)];

      // Out-of-bounds items are selection_flags
      int hash = HASH(key, ht_len, keys_min);
      atomicCAS(&ht[hash << 1], 0, key);
      ht[(hash << 1) + 1] = val;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPU2(
    int tid,
    int  (&items)[ITEMS_PER_THREAD], //equal to items
    int* gpuCache,
    int* dimkey_idx,
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {
  cudaAssert(ht != NULL);
  cudaAssert(dimkey_idx != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int dimkey_seg = dimkey_idx[items[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items[ITEM] % SEGMENT_SIZE)];

      // Out-of-bounds items are selection_flags
      int hash = HASH(key, ht_len, keys_min);
      atomicCAS(&ht[hash << 1], 0, key);
      ht[(hash << 1) + 1] = items[ITEM] + 1;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadOffsetGPU(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&items)[ITEMS_PER_THREAD], //equal to items
    int* gpuCache,
    int* col_idx,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int col_seg = col_idx[items_off[ITEM] / SEGMENT_SIZE];
      items[ITEM] = gpuCache[col_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* dim_key4,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int *total, int start_offset) {

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  // CUB use blocked arrangement
  //typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    //typename BlockLoadInt::TempStorage load_items;
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

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  __syncthreads();

  if (dim_key1 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key1 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
  }

  __syncthreads();

  if (dim_key2 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key2 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
  }

  __syncthreads();

  if (dim_key3 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key3 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
  }

  __syncthreads();

  if (dim_key4 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key4 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  }

  //Barrier
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      if(selection_flags[ITEM]) t_count++;
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

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        cudaAssert(lo_off != NULL);
        lo_off[offset] = start_offset + blockIdx.x * tile_size + threadIdx.x + ITEM * BLOCK_THREADS;
        if (dim_off1 != NULL) dim_off1[offset] = dim_offset1[ITEM];
        if (dim_off2 != NULL) dim_off2[offset] = dim_offset2[ITEM];
        if (dim_off3 != NULL) dim_off3[offset] = dim_offset3[ITEM];
        if (dim_off4 != NULL) dim_off4[offset] = dim_offset4[ITEM];
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_GPU2(
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* gpuCache, int* dimkey_idx1, int* dimkey_idx2, int* dimkey_idx3, int* dimkey_idx4,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* out_lo_off, int* out_dim_off1, int* out_dim_off2, int* out_dim_off3, int* out_dim_off4, 
  int *total, int start_offset, int* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;    // Current tile index
  int tile_offset = blockIdx.x * tile_size;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_index;
  if (segment_group == NULL)
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
  else {
    int idx = tile_offset / SEGMENT_SIZE;
    segment_index = segment_group[idx];
    start_offset = segment_index * SEGMENT_SIZE;
    tile_idx = blockIdx.x % tiles_per_segment;
  }
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    //typename BlockLoadInt::TempStorage load_items;
    typename BlockScanInt::TempStorage scan;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int dim_offset1[ITEMS_PER_THREAD];
  int dim_offset2[ITEMS_PER_THREAD];
  int dim_offset3[ITEMS_PER_THREAD];
  int dim_offset4[ITEMS_PER_THREAD];
  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  __syncthreads();

  //if (lo_off != NULL) BlockLoadInt(temp_storage.load_items).Load(lo_off + tile_offset, items_lo, num_tile_items);
  if (lo_off != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_lo, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  if (dimkey_idx1 != NULL) {

    if (lo_off == NULL) {
      cudaAssert(dimkey_idx1 != NULL);
      int dimkey_seg1 = dimkey_idx1[segment_index];
      int* ptr = gpuCache + dimkey_seg1 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx1, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, dim_offset1, selection_flags, gpuCache, dimkey_idx1, lo_off, ht1, dim_len1, min_key1, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);

  } else if (dim_off1 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockPassThroughOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (dimkey_idx2 != NULL) {

    if (lo_off == NULL) {
      int dimkey_seg2 = dimkey_idx2[segment_index];
      int* ptr = gpuCache + dimkey_seg2 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx2, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, dim_offset2, selection_flags, gpuCache, dimkey_idx2, lo_off, ht2, dim_len2, min_key2, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);

  } else if (dim_off2 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockPassThroughOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (dimkey_idx3 != NULL) {

    if (lo_off == NULL) {
      int dimkey_seg3 = dimkey_idx3[segment_index];
      int* ptr = gpuCache + dimkey_seg3 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx3, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, dim_offset3, selection_flags, gpuCache, dimkey_idx3, lo_off, ht3, dim_len3, min_key3, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);

  } else if (dim_off3 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockPassThroughOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (dimkey_idx4 != NULL) {

    if (lo_off == NULL) {
      int dimkey_seg4 = dimkey_idx4[segment_index];
      int* ptr = gpuCache + dimkey_seg4 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx4, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, dim_offset4, selection_flags, gpuCache, dimkey_idx4, lo_off, ht4, dim_len4, min_key4, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);

  } else if (dim_off4 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockPassThroughOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  //Barrier
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      if(selection_flags[ITEM]) t_count++;
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

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        if (lo_off != NULL) out_lo_off[offset] = items_lo[ITEM];
        else out_lo_off[offset] = start_offset + tile_idx * tile_size + threadIdx.x + ITEM * BLOCK_THREADS;
        if (out_dim_off1 != NULL) out_dim_off1[offset] = dim_offset1[ITEM];
        if (out_dim_off2 != NULL) out_dim_off2[offset] = dim_offset2[ITEM];
        if (out_dim_off3 != NULL) out_dim_off3[offset] = dim_offset3[ITEM];
        if (out_dim_off4 != NULL) out_dim_off4[offset] = dim_offset4[ITEM];
        //printf("%d %d %d %d\n", dim_offset1[ITEM], dim_offset2[ITEM], dim_offset3[ITEM], dim_offset4[ITEM]);
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* dim_key4, int* aggr, 
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4,
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4) {

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Allocate shared memory for BlockLoad
  // __shared__ union TempStorage
  // {
  //   typename BlockLoadInt::TempStorage load_items;
  // } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];
  int aggrval[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  __syncthreads();

  if (dim_key1 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key1 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);

  } else {

    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, num_tile_items);
  }

  __syncthreads();

  if (dim_key2 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key2 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);

  } else {
    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, num_tile_items);
  }

  __syncthreads();

  if (dim_key3 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key3 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);

  } else {
    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, num_tile_items);
  }

  __syncthreads();

  if (dim_key4 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_key4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key4 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);

  } else {
    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, num_tile_items);
  }

  __syncthreads();

  //BlockLoadInt(temp_storage.load_items).Load(aggr + tile_offset, aggrval, num_tile_items);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr + tile_offset, aggrval, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
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


//THIS WILL NOT WORKING!!! DIM_OFF CAN NO LONGER BE NULL 
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* gpuCache, int* dimkey_idx1, int* dimkey_idx2, int* dimkey_idx3, int* dimkey_idx4, int* aggr_idx,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1,int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4,
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int start_offset, int* segment_group) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_index;
  if (segment_group == NULL)
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
  else {
    int idx = tile_offset / SEGMENT_SIZE;
    segment_index = segment_group[idx];
  }
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Allocate shared memory for BlockLoad
  // __shared__ union TempStorage
  // {
  //   typename BlockLoadInt::TempStorage load_items;
  // } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];
  int aggrval[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  __syncthreads();

  //if (lo_off != NULL) BlockLoadInt(temp_storage.load_items).Load(lo_off + tile_offset, items_lo, num_tile_items);
  if (lo_off != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_lo, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  if (dim_off1 == NULL && dimkey_idx1 != NULL) { //normal operation, here dimkey_idx will be lo_partkey, lo_suppkey, etc (the join key column)

    // Barrier for smem reuse
    __syncthreads();

    if (lo_off == NULL) {
      int dimkey_seg1 = dimkey_idx1[segment_index];
      int* ptr = gpuCache + dimkey_seg1 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx1, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, groupval1, selection_flags, gpuCache, dimkey_idx1, lo_off, ht1, dim_len1, min_key1, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);

  } else if (dim_off1 != NULL && dimkey_idx1 != NULL) { //we take the result from prev join in dim_off but we will also take the groupby column, here dimkey_idx will be the groupby column (d_year, p_brand1, etc.)
    //BlockLoadInt(temp_storage.load_items).Load(dim_off1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, gpuCache, dimkey_idx1, num_tile_items);

  } else if (dim_off1 == NULL && dimkey_idx1 == NULL) { //not doing anything (if dim_off1 == NULL && dimkey_idx1 == NULL)

    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (dim_off2 == NULL && dimkey_idx2 != NULL) {

    if (lo_off == NULL) {
      int dimkey_seg2 = dimkey_idx2[segment_index];
      int* ptr = gpuCache + dimkey_seg2 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx2, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, groupval2, selection_flags, gpuCache, dimkey_idx2, lo_off, ht2, dim_len2, min_key2, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);

  }  else if (dim_off2 != NULL && dimkey_idx2 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, gpuCache, dimkey_idx2, num_tile_items);

  } else if (dim_off2 == NULL && dimkey_idx2 == NULL) {

    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (dim_off3 == NULL && dimkey_idx3 != NULL) {

    if (lo_off == NULL) {
      int dimkey_seg3 = dimkey_idx3[segment_index];
      int* ptr = gpuCache + dimkey_seg3 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx3, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, groupval3, selection_flags, gpuCache, dimkey_idx3, lo_off, ht3, dim_len3, min_key3, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);

  } else if (dim_off3 != NULL && dimkey_idx3 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, gpuCache, dimkey_idx3, num_tile_items);

  } else if (dim_off3 == NULL && dimkey_idx3 == NULL) {

    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (dim_off4 == NULL && dimkey_idx4 != NULL) {

    if (lo_off == NULL) {
      int dimkey_seg4 = dimkey_idx4[segment_index];
      int* ptr = gpuCache + dimkey_seg4 * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, dimkey_idx4, num_tile_items);
    }

    // Barrier for smem reuse
    __syncthreads();

    //BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, items_lo, groupval4, selection_flags, gpuCache, dimkey_idx4, lo_off, ht4, dim_len4, min_key4, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);

  } else if (dim_off4 != NULL && dimkey_idx4 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, items, num_tile_items);

    // Barrier for smem reuse
    __syncthreads();

    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, gpuCache, dimkey_idx4, num_tile_items);

  } else if (dim_off4 == NULL && dimkey_idx4 == NULL) {

    BlockSetGroupValZero<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, num_tile_items);

  } else {
    cudaAssert(0);
  }

  __syncthreads();

  if (lo_off == NULL) {
    int aggr_seg = aggr_idx[segment_index];
    int* ptr = gpuCache + aggr_seg * SEGMENT_SIZE;
    //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, aggrval, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval, num_tile_items);
  } else {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval, gpuCache, aggr_idx, num_tile_items);
  }

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      // if (lo_off != NULL) {
      //   int aggr_seg = aggr_idx[items_lo[ITEM] / SEGMENT_SIZE];
      //   aggrval[ITEM] = gpuCache[aggr_seg * SEGMENT_SIZE + (items_lo[ITEM] % SEGMENT_SIZE)];
      // }

      if (selection_flags[ITEM]) {
        //printf("aggrval = %d\n", aggrval[ITEM]);
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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU(int *dim_key, int *dim_val, int num_tuples, 
  int *hash_table, int num_slots, int val_min, int start_offset, int isoffset) {

  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  //if (filter_col == NULL) {
  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  // } else {
  //   BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
  //   BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare, selection_flags, num_tile_items);   
  // }

  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);

  if (!isoffset) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);

    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
        hash_table, num_slots, val_min, num_tile_items); 
  } else {
    BlockBuildOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, blockIdx.x, start_offset,
      items, selection_flags, hash_table, num_slots, val_min, num_tile_items); 
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU2(int* dim_off, int* gpuCache, int *dimkey_idx, int *dimval_idx, int num_tuples, 
  int *hash_table, int num_slots, int val_min, int start_offset, int isoffset) {

  int items[ITEMS_PER_THREAD];
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off + tile_offset, items, num_tile_items);

  if (!isoffset) {
    BlockBuildValueGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, gpuCache, dimkey_idx, dimval_idx, 
        hash_table, num_slots, val_min, num_tile_items);
  } else {
    BlockBuildOffsetGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, gpuCache, dimkey_idx, 
        hash_table, num_slots, val_min, num_tile_items); 
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_GPU(int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* out_off, int* total, int start_offset, int num_tuples, int mode1, int mode2) {

  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockScanInt::TempStorage scan;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items1[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (filter_col1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col1 + tile_offset, items1, num_tile_items);
    if (mode1 == 0) { //equal to
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, compare1, selection_flags, num_tile_items);
    } else if (mode1 == 1) { //between
      BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, compare1, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, compare2, selection_flags, num_tile_items);
    } else if (mode1 == 2) { //equal or equal
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, compare1, selection_flags, num_tile_items);
      BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, compare2, selection_flags, num_tile_items);
    } else if (mode1 == 3) { //less than
      BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items1, compare1, selection_flags, num_tile_items);
    }

    if (filter_col2 != NULL) {
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col2 + tile_offset, items2, num_tile_items);
      if (mode2 == 0) { //equal to
        BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
      } else if (mode2 == 1) { //between
        BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 2) { //equal or equal
        BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
        BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 3) { //less than
        BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
      }
    }
  } else {
    if (filter_col2 != NULL) {
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col2 + tile_offset, items2, num_tile_items);
      if (mode2 == 0) { //equal to
        BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
      } else if (mode2 == 1) { //between
        BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 2) { //equal or equal
        BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
        BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 3) { //less than
        BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, compare3, selection_flags, num_tile_items);
      }
    }
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if(selection_flags[ITEM]) t_count++;
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

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        out_off[offset] = start_offset + blockIdx.x * tile_size + (threadIdx.x + ITEM * BLOCK_THREADS);
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_GPU2(int* off_col, int* gpuCache, int* filter_idx, int compare1, int compare2,
  int* out_off, int* total, int start_offset, int num_tuples, int mode) {

  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockScanInt::TempStorage scan;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int items_off[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (off_col != NULL)
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(off_col + tile_offset, items_off, num_tile_items);

  if (filter_idx != NULL && off_col != NULL) {
    if (mode == 0) { //equal to
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, filter_idx, num_tile_items);
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    } else if (mode == 1) { //between
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, filter_idx, num_tile_items);
      BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode == 2) { //equal or equal
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, filter_idx, num_tile_items);
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode == 3) { //less than
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, filter_idx, num_tile_items);
      BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    }
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if(selection_flags[ITEM]) {
        t_count++;
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

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        //out_off[offset] = start_offset + blockIdx.x * tile_size + (threadIdx.x + ITEM * BLOCK_THREADS);
        out_off[offset] = items_off[ITEM];
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void groupByGPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int* gpuCache, int* aggr_idx1, int* aggr_idx2, int* group_idx1, int* group_idx2, int* group_idx3, int* group_idx4,
  int min_val1, int min_val2, int min_val3, int min_val4, int unique_val1, int unique_val2, int unique_val3, int unique_val4,
  int total_val, int num_tuples, int* res, int mode) {

  int items_off[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  cudaAssert(lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_off, num_tile_items);

  if (aggr_idx1 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval1, gpuCache, aggr_idx1, num_tile_items);
  else {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) { // CUB use blocked arrangement
        aggrval1[ITEM] = 0;
      }
    }
  }

  if (aggr_idx2 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval2, gpuCache, aggr_idx2, num_tile_items);
  else {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) { // CUB use blocked arrangement
        aggrval2[ITEM] = 0;
      }
    }
  }

  if (group_idx1 != NULL) {
    cudaAssert(dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval1, gpuCache, group_idx1, num_tile_items);    
  } else {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) { // CUB use blocked arrangement
        groupval1[ITEM] = 0;
      }
    }
  }

  if (group_idx2 != NULL) {
    cudaAssert(dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval2, gpuCache, group_idx2, num_tile_items);    
  } else {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) { // CUB use blocked arrangement
        groupval2[ITEM] = 0;
      }
    }
  }

  if (group_idx3 != NULL) {
    cudaAssert(dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval3, gpuCache, group_idx3, num_tile_items);    
  } else {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) { // CUB use blocked arrangement
        groupval3[ITEM] = 0;
      }
    }
  }

  if (group_idx4 != NULL) {
    cudaAssert(dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval4, gpuCache, group_idx4, num_tile_items);   

  } else {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) { // CUB use blocked arrangement
        groupval4[ITEM] = 0;
      }
    }
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3) * unique_val3 + (groupval4[ITEM] - min_val4) * unique_val4) % total_val; //!

      //printf("%d %d %d %d\n", groupval1[ITEM], groupval2[ITEM], groupval3[ITEM], groupval4[ITEM]);
      res[hash * 6] = groupval1[ITEM];
      res[hash * 6 + 1] = groupval2[ITEM];
      res[hash * 6 + 2] = groupval3[ITEM];
      res[hash * 6 + 3] = groupval4[ITEM];

      //cudaAssert(groupval4[ITEM] <= 1998);

      int temp;
      if (mode == 0) temp = aggrval1[ITEM];
      else if (mode == 1) temp = aggrval1[ITEM] - aggrval2[ITEM];
      else if (mode == 2) temp = aggrval1[ITEM] * aggrval2[ITEM];
      atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
    }
  }

}

#endif

// __global__
// void runAggregationQ2GPU(int* gpuCache, int* lo_idx, int* p_idx, int* d_idx, int* lo_off, int* part_off, int* date_off, int num_tuples, int* res, int num_slots) {
//   int offset = blockIdx.x * blockDim.x + threadIdx.x;

//   if (offset < num_tuples) {
//     int revenue_idx = lo_off[offset];
//     int brand_idx = part_off[offset];
//     int year_idx = date_off[offset];

//     int revenue_seg = lo_idx[revenue_idx / SEGMENT_SIZE];
//     int brand_seg = p_idx[brand_idx / SEGMENT_SIZE];
//     int year_seg = d_idx[year_idx / SEGMENT_SIZE];

//     int revenue = gpuCache[revenue_seg * SEGMENT_SIZE + (revenue_idx % SEGMENT_SIZE)];
//     int brand = gpuCache[brand_seg * SEGMENT_SIZE + (brand_idx % SEGMENT_SIZE)];
//     int year = gpuCache[year_seg * SEGMENT_SIZE + (year_idx % SEGMENT_SIZE)];

//     int hash = (brand * 7 + (year - 1992)) % num_slots;

//     res[hash * 6] = 0;
//     res[hash * 6 + 1] = brand;
//     res[hash * 6 + 2] = year;
//     atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(revenue));

//   }
// }