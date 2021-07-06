#ifndef _GPU_PROCESSING_H_
#define _GPU_PROCESSING_H_

#include <cub/cub.cuh>
#include <curand.h>

#include <cuda.h>
// #include <cub/util_allocator.cuh>

#include "crystal/crystal.cuh"
#include "BlockLibrary.cuh"

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
__global__ void filter_probe_group_by_GPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* gpuCache, int* filter_idx1, int* filter_idx2, int compare1, int compare2, int compare3, int compare4,
  int* key_idx1, int* key_idx2, int* key_idx3, int* key_idx4, 
  int* aggr_idx1, int* aggr_idx2, int* group_idx1, int* group_idx2, int* group_idx3, int* group_idx4, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4,
  int total_val, int* res, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int* ptr;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_index;
  if (segment_group == NULL)
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
  else {
    int idx = tile_offset / SEGMENT_SIZE;
    segment_index = segment_group[idx];
  }
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (lo_off != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_lo, num_tile_items);

  if (filter_idx1 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + filter_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, filter_idx1, num_tile_items);
    }
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
  }

  if (filter_idx2 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + filter_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, filter_idx2, num_tile_items);
    }
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
  }



  if (key_idx1 != NULL && ht1 != NULL) { //normal operation, here key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    if (lo_off == NULL) { //there is no result from prev join (this is the first join), segment group can be null or not null
      ptr = gpuCache + key_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
    } else { //there is a result from prev join
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval1, selection_flags, gpuCache, key_idx1, ht1, dim_len1, min_key1, num_tile_items);
    }
  } else if (group_idx1 != NULL) { //we take the result from prev join in dim_off but we will also take the groupby column, here group_idx will be the groupby column (d_year, p_brand1, etc.)
    cudaAssert(dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, gpuCache, group_idx1, num_tile_items);
  } else if (group_idx1 == NULL) { //not doing anything, dim_off can be either NULL or not
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_idx2 != NULL && ht2 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + key_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
    } else {
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval2, selection_flags, gpuCache, key_idx2, ht2, dim_len2, min_key2, num_tile_items);
    }
  } else if (group_idx2 != NULL) {
    cudaAssert(dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, gpuCache, group_idx2, num_tile_items);
  } else if (group_idx2 == NULL) {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_idx3 != NULL && ht3 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + key_idx3[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
    } else {
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval3, selection_flags, gpuCache, key_idx3, ht3, dim_len3, min_key3, num_tile_items);
    }
  } else if (group_idx3 != NULL) {
    cudaAssert(dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, gpuCache, group_idx3, num_tile_items);
  } else if (group_idx3 == NULL) {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_idx4 != NULL && ht4 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + key_idx4[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
    } else {
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval4, selection_flags, gpuCache, key_idx4, ht4, dim_len4, min_key4, num_tile_items);
    }
  } else if (group_idx4 != NULL) {
    cudaAssert(dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, gpuCache, group_idx4, num_tile_items);
  } else if (group_idx4 == NULL) {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (lo_off == NULL) {
    if (aggr_idx1 != NULL) {
      ptr = gpuCache + aggr_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval1, num_tile_items);
    }
    if (aggr_idx2 != NULL) {
      ptr = gpuCache + aggr_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval2, num_tile_items);
    }
  } else {
    if (aggr_idx1 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval1, gpuCache, aggr_idx1, num_tile_items);
    if (aggr_idx2 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval2, gpuCache, aggr_idx2, num_tile_items);
  }


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3) * unique_val3 + (groupval4[ITEM] - min_val4) * unique_val4) % total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp;
        if (mode == 0) temp = aggrval1[ITEM];
        else if (mode == 1) temp = aggrval1[ITEM] - aggrval2[ITEM];
        else if (mode == 2) temp = aggrval1[ITEM] * aggrval2[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* gpuCache, int* key_idx1, int* key_idx2, int* key_idx3, int* key_idx4, 
  int* aggr_idx1, int* aggr_idx2, int* group_idx1, int* group_idx2, int* group_idx3, int* group_idx4, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4,
  int total_val, int* res, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int* ptr;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_index;
  if (segment_group == NULL)
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
  else {
    int idx = tile_offset / SEGMENT_SIZE;
    segment_index = segment_group[idx];
  }
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (lo_off != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_lo, num_tile_items);

  if (key_idx1 != NULL && ht1 != NULL) { //normal operation, here key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    if (lo_off == NULL) { //there is no result from prev join (this is the first join), segment group can be null or not null
      ptr = gpuCache + key_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
    } else { //there is a result from prev join
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval1, selection_flags, gpuCache, key_idx1, ht1, dim_len1, min_key1, num_tile_items);
    }
  } else if (group_idx1 != NULL) { //we take the result from prev join in dim_off but we will also take the groupby column, here group_idx will be the groupby column (d_year, p_brand1, etc.)
    cudaAssert(dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, gpuCache, group_idx1, num_tile_items);
  } else if (group_idx1 == NULL) { //not doing anything, dim_off can be either NULL or not
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_idx2 != NULL && ht2 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + key_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
    } else {
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval2, selection_flags, gpuCache, key_idx2, ht2, dim_len2, min_key2, num_tile_items);
    }
  } else if (group_idx2 != NULL) {
    cudaAssert(dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, gpuCache, group_idx2, num_tile_items);
  } else if (group_idx2 == NULL) {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_idx3 != NULL && ht3 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + key_idx3[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
    } else {
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval3, selection_flags, gpuCache, key_idx3, ht3, dim_len3, min_key3, num_tile_items);
    }
  } else if (group_idx3 != NULL) {
    cudaAssert(dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, gpuCache, group_idx3, num_tile_items);
  } else if (group_idx3 == NULL) {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_idx4 != NULL && ht4 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + key_idx4[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
    } else {
      BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval4, selection_flags, gpuCache, key_idx4, ht4, dim_len4, min_key4, num_tile_items);
    }
  } else if (group_idx4 != NULL) {
    cudaAssert(dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, gpuCache, group_idx4, num_tile_items);
  } else if (group_idx4 == NULL) {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (lo_off == NULL) {
    if (aggr_idx1 != NULL) {
      ptr = gpuCache + aggr_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval1, num_tile_items);
    }
    if (aggr_idx2 != NULL) {
      ptr = gpuCache + aggr_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval2, num_tile_items);
    }
  } else {
    if (aggr_idx1 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval1, gpuCache, aggr_idx1, num_tile_items);
    if (aggr_idx2 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval2, gpuCache, aggr_idx2, num_tile_items);
  }


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items)) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3) * unique_val3 + (groupval4[ITEM] - min_val4) * unique_val4) % total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp;
        if (mode == 0) temp = aggrval1[ITEM];
        else if (mode == 1) temp = aggrval1[ITEM] - aggrval2[ITEM];
        else if (mode == 2) temp = aggrval1[ITEM] * aggrval2[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_GPU2(
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, int* gpuCache, 
  int* filter_idx1, int* filter_idx2, int compare1, int compare2, int compare3, int compare4,
  int* key_idx1, int* key_idx2, int* key_idx3, int* key_idx4,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* out_lo_off, int* out_dim_off1, int* out_dim_off2, int* out_dim_off3, int* out_dim_off4, 
  int *total, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx;    // Current tile index
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size; //how many block per segment
  int* ptr;

  int segment_index;
  if (segment_group == NULL) {
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    tile_idx = blockIdx.x;
  }
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

  //if (lo_off != NULL) BlockLoadInt(temp_storage.load_items).Load(lo_off + tile_offset, items_lo, num_tile_items);
  if (lo_off != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_lo, num_tile_items);

  if (filter_idx1 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + filter_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, filter_idx1, num_tile_items);
    }
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
  }

  if (filter_idx2 != NULL) {
    if (lo_off == NULL) {
      ptr = gpuCache + filter_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, filter_idx2, num_tile_items);
    }
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
  }

  if (key_idx1 != NULL && ht1 != NULL) { //we are doing probing for this column (normal operation)

    if (lo_off == NULL) { //there is no result from prev join (this is the first join), segment_group can be null or not null
      ptr = gpuCache + key_idx1[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else { //there is a result from prev join
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx1, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
  } else if (dim_off1 != NULL && key_idx1 == NULL && ht1 == NULL) { //load result from prev join, we are just passing it through (no probing)
    //BlockLoadInt(temp_storage.load_items).Load(dim_off1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, dim_offset1, num_tile_items);
  } else if (dim_off1 == NULL && key_idx1 == NULL && ht1 == NULL) { //we are not doing join for this column, there is no result from prev join (first join)
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset1, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  if (key_idx2 != NULL && ht2 != NULL) {

    if (lo_off == NULL) {
      ptr = gpuCache + key_idx2[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx2, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
  } else if (dim_off2 != NULL && key_idx2 == NULL && ht2 == NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, dim_offset2, num_tile_items);
  } else if (dim_off2 == NULL && key_idx2 == NULL && ht2 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset2, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  if (key_idx3 != NULL && ht3 != NULL) {

    if (lo_off == NULL) {
      ptr = gpuCache + key_idx3[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx3, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
  } else if (dim_off3 != NULL && key_idx3 == NULL && ht3 == NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, dim_offset3, num_tile_items);
  } else if (dim_off3 == NULL && key_idx3 == NULL && ht3 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset3, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  if (key_idx4 != NULL && ht4 != NULL) {

    if (lo_off == NULL) {
      ptr = gpuCache + key_idx4[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx4, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  } else if (dim_off4 != NULL && key_idx4 == NULL && ht4 == NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, dim_offset4, num_tile_items);
  } else if (dim_off4 == NULL && key_idx4 == NULL && ht4 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset4, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

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
__global__ void probe_GPU2(
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, int* gpuCache, 
  int* key_idx1, int* key_idx2, int* key_idx3, int* key_idx4,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* out_lo_off, int* out_dim_off1, int* out_dim_off2, int* out_dim_off3, int* out_dim_off4, 
  int *total, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx;    // Current tile index
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size; //how many block per segment
  int* ptr;

  int segment_index;
  if (segment_group == NULL) {
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    tile_idx = blockIdx.x;
  }
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

  //if (lo_off != NULL) BlockLoadInt(temp_storage.load_items).Load(lo_off + tile_offset, items_lo, num_tile_items);
  if (lo_off != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_lo, num_tile_items);

  if (key_idx1 != NULL && ht1 != NULL) { //we are doing probing for this column (normal operation)

    if (lo_off == NULL) { //there is no result from prev join (this is the first join), segment_group can be null or not null
      ptr = gpuCache + key_idx1[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else { //there is a result from prev join
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx1, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
  } else if (dim_off1 != NULL && key_idx1 == NULL && ht1 == NULL) { //load result from prev join, we are just passing it through (no probing)
    //BlockLoadInt(temp_storage.load_items).Load(dim_off1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, dim_offset1, num_tile_items);
  } else if (dim_off1 == NULL && key_idx1 == NULL && ht1 == NULL) { //we are not doing join for this column, there is no result from prev join (first join)
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset1, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  if (key_idx2 != NULL && ht2 != NULL) {

    if (lo_off == NULL) {
      ptr = gpuCache + key_idx2[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx2, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
  } else if (dim_off2 != NULL && key_idx2 == NULL && ht2 == NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, dim_offset2, num_tile_items);
  } else if (dim_off2 == NULL && key_idx2 == NULL && ht2 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset2, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  if (key_idx3 != NULL && ht3 != NULL) {

    if (lo_off == NULL) {
      ptr = gpuCache + key_idx3[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx3, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
  } else if (dim_off3 != NULL && key_idx3 == NULL && ht3 == NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, dim_offset3, num_tile_items);
  } else if (dim_off3 == NULL && key_idx3 == NULL && ht3 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset3, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

  if (key_idx4 != NULL && ht4 != NULL) {

    if (lo_off == NULL) {
      ptr = gpuCache + key_idx4[segment_index] * SEGMENT_SIZE;
      //BlockLoadInt(temp_storage.load_items).Load(ptr + segment_tile_offset, items, num_tile_items);
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    } else {
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, key_idx4, num_tile_items);
    }
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  } else if (dim_off4 != NULL && key_idx4 == NULL && ht4 == NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(dim_off4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, dim_offset4, num_tile_items);
  } else if (dim_off4 == NULL && key_idx4 == NULL && ht4 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset4, 1, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }

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
__global__ void build_GPU2(int* dim_off, 
  int* gpuCache, int* filter_idx, int compare1, int compare2, int mode,
  int *key_idx, int *val_idx, int num_tuples, 
  int *hash_table, int num_slots, int val_min, 
  int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  int items[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;

  int segment_index, tile_idx;
  if (segment_group == NULL) {
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    tile_idx = blockIdx.x;
  } else {
    int idx = tile_offset / SEGMENT_SIZE;
    segment_index = segment_group[idx];
    start_offset = segment_index * SEGMENT_SIZE;
    tile_idx = blockIdx.x % tiles_per_segment;
  }
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (dim_off != NULL) {
    cudaAssert(filter_idx == NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off + tile_offset, items, num_tile_items);
    if (val_idx != NULL) {
      BlockBuildValueGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, key_idx, val_idx, 
          hash_table, num_slots, val_min, num_tile_items);
    } else {
      BlockBuildOffsetGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, key_idx, 
          hash_table, num_slots, val_min, num_tile_items); 
    }

  } else {

    if (filter_idx != NULL) {
      int* ptr = gpuCache + filter_idx[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
      if (mode == 0) { //equal to
        BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      } else if (mode == 1) { //between
        BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
      } else if (mode == 2) { //equal or equal
        BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
        BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
      } else if (mode == 3) { //less than
        BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      }
    }

    cudaAssert(key_idx != NULL);
    int* ptr_key = gpuCache + key_idx[segment_index] * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr_key + segment_tile_offset, items, num_tile_items);

    if (val_idx != NULL) {
      int* ptr = gpuCache + val_idx[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, vals, num_tile_items);

      BlockBuildValueGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, tile_idx, start_offset,
        items, vals, selection_flags, hash_table, num_slots, val_min, num_tile_items);
    } else {
      BlockBuildOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, tile_idx, start_offset,
        items, selection_flags, hash_table, num_slots, val_min, num_tile_items);
    }
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_GPU2(int* off_col, 
  int* gpuCache, int* filter_idx1, int* filter_idx2, 
  int compare1, int compare2, int compare3, int compare4, int mode1, int mode2,
  int* out_off, int* total, int num_tuples, int start_offset = 0, short* segment_group = NULL) {

  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

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

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size;

  int segment_index, tile_idx;
  if (segment_group == NULL) {
    segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    tile_idx = blockIdx.x;
  } else {
    int idx = tile_offset / SEGMENT_SIZE;
    segment_index = segment_group[idx];
    start_offset = segment_index * SEGMENT_SIZE;
    tile_idx = blockIdx.x % tiles_per_segment;
  }
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (filter_idx1 != NULL) {
    if (off_col != NULL) {
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(off_col + tile_offset, items_off, num_tile_items);
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, filter_idx1, num_tile_items);
    } else {
      int* ptr = gpuCache + filter_idx1[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    }

    if (mode1 == 0) { //equal to
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    } else if (mode1 == 1) { //between
      BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode1 == 2) { //equal or equal
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode1 == 3) { //less than
      BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    }
  }

  if (filter_idx2 != NULL) {
    if (off_col != NULL) {
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(off_col + tile_offset, items_off, num_tile_items);
      BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, filter_idx2, num_tile_items);
    } else {
      int* ptr = gpuCache + filter_idx2[segment_index] * SEGMENT_SIZE;
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    }

    if (mode2 == 0) { //equal to
      BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
    } else if (mode2 == 1) { //between
      BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
    } else if (mode2 == 2) { //equal or equal
      BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
      BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
    } else if (mode2 == 3) { //less than
      BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
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
        cudaAssert(out_off != NULL);
        if (off_col == NULL) out_off[offset] = start_offset + tile_idx * tile_size + threadIdx.x + ITEM * BLOCK_THREADS;
        else out_off[offset] = items_off[ITEM];
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
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (aggr_idx2 != NULL) BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval2, gpuCache, aggr_idx2, num_tile_items);
  else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  if (group_idx1 != NULL) {
    cudaAssert(dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off1 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval1, gpuCache, group_idx1, num_tile_items);    
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (group_idx2 != NULL) {
    cudaAssert(dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off2 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval2, gpuCache, group_idx2, num_tile_items);    
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (group_idx3 != NULL) {
    cudaAssert(dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off3 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval3, gpuCache, group_idx3, num_tile_items);    
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (group_idx4 != NULL) {
    cudaAssert(dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off4 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval4, gpuCache, group_idx4, num_tile_items);   

  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_group_by_GPU(int* gpuCache,
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int* group_idx1, int* group_idx2, int* group_idx3, int* group_idx4, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4, int total_val, int* res) {
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (filter_col1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col1 + tile_offset, items, num_tile_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
  }

  if (filter_col2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col2 + tile_offset, items, num_tile_items);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
  }



  if (key_col1 != NULL && ht1 != NULL && group_idx1 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col1 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, selection_flags, num_tile_items);
  } else if (key_col1 != NULL && ht1 != NULL && group_idx1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col1 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, groupval1, selection_flags, gpuCache, group_idx1, num_tile_items);
  } else if (key_col1 == NULL && ht1 == NULL && group_idx1 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_col2 != NULL && ht2 != NULL && group_idx2 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col2 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, selection_flags, num_tile_items);
  } else if (key_col2 != NULL && ht2 != NULL && group_idx2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col2 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, groupval2, selection_flags, gpuCache, group_idx2, num_tile_items);
  } else if (key_col2 == NULL && ht2 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_col3 != NULL && ht3 != NULL && group_idx3 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col3 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, selection_flags, num_tile_items);
  } else if (key_col3 != NULL && ht3 != NULL && group_idx3 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col3 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, groupval3, selection_flags, gpuCache, group_idx3, num_tile_items);
  } else if (key_col3 == NULL && ht3 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_col4 != NULL && ht4 != NULL && group_idx4 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col4 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, selection_flags, num_tile_items);
  } else if (key_col4 != NULL && ht4 != NULL && group_idx4 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col4 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, groupval4, selection_flags, gpuCache, group_idx4, num_tile_items);
  } else if (key_col4 == NULL && ht4 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }


  if (aggr_col1 != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr_col1 + tile_offset, aggrval1, num_tile_items);
  if (aggr_col2 != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr_col2 + tile_offset, aggrval2, num_tile_items);

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

        int temp;
        if (mode == 0) temp = aggrval1[ITEM];
        else if (mode == 1) temp = aggrval1[ITEM] - aggrval2[ITEM];
        else if (mode == 2) temp = aggrval1[ITEM] * aggrval2[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU(int* gpuCache,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int* group_idx1, int* group_idx2, int* group_idx3, int* group_idx4, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4, int total_val, int* res) {
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int groupval1[ITEMS_PER_THREAD];
  int groupval2[ITEMS_PER_THREAD];
  int groupval3[ITEMS_PER_THREAD];
  int groupval4[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
    is_last_tile = true;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (key_col1 != NULL && ht1 != NULL && group_idx1 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col1 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, selection_flags, num_tile_items);
  } else if (key_col1 != NULL && ht1 != NULL && group_idx1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col1 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, groupval1, selection_flags, gpuCache, group_idx1, num_tile_items);
  } else if (key_col1 == NULL && ht1 == NULL && group_idx1 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_col2 != NULL && ht2 != NULL && group_idx2 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col2 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, selection_flags, num_tile_items);
  } else if (key_col2 != NULL && ht2 != NULL && group_idx2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col2 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, groupval2, selection_flags, gpuCache, group_idx2, num_tile_items);
  } else if (key_col2 == NULL && ht2 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_col3 != NULL && ht3 != NULL && group_idx3 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col3 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, selection_flags, num_tile_items);
  } else if (key_col3 != NULL && ht3 != NULL && group_idx3 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col3 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, groupval3, selection_flags, gpuCache, group_idx3, num_tile_items);
  } else if (key_col3 == NULL && ht3 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }



  if (key_col4 != NULL && ht4 != NULL && group_idx4 == NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col4 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, selection_flags, num_tile_items);
  } else if (key_col4 != NULL && ht4 != NULL && group_idx4 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col4 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, groupval4, selection_flags, gpuCache, group_idx4, num_tile_items);
  } else if (key_col4 == NULL && ht4 == NULL) {
    BlockSetFilteredValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, selection_flags, num_tile_items);
  } else {
    cudaAssert(0);
  }


  if (aggr_col1 != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr_col1 + tile_offset, aggrval1, num_tile_items);
  if (aggr_col2 != NULL) BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr_col2 + tile_offset, aggrval2, num_tile_items);

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

        int temp;
        if (mode == 0) temp = aggrval1[ITEM];
        else if (mode == 1) temp = aggrval1[ITEM] - aggrval2[ITEM];
        else if (mode == 2) temp = aggrval1[ITEM] * aggrval2[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_GPU(
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int *total, int start_offset = 0) {

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

  if (filter_col1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col1 + tile_offset, items, num_tile_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
  }

  if (filter_col2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col2 + tile_offset, items, num_tile_items);
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
  }

  if (key_col1 != NULL && ht1 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col1 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
  } else if (!(key_col1 == NULL && ht1 == NULL)) {
    cudaAssert(0);
  }

  if (key_col2 != NULL && ht2 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col2 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
  } else if (!(key_col2 == NULL && ht2 == NULL)) {
    cudaAssert(0);
  }

  if (key_col3 != NULL && ht3 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col3 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
  } else if (!(key_col3 == NULL && ht3 == NULL)) {
    cudaAssert(0);
  }

  if (key_col4 != NULL && ht4 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col4 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  } else if (!(key_col4 == NULL && ht4 == NULL)) {
    cudaAssert(0);
  }

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
__global__ void probe_GPU(
  int* key_col1, int* key_col2, int* key_col3, int* key_col4,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int *total, int start_offset = 0) {

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

  if (key_col1 != NULL && ht1 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col1 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col1 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
  } else if (!(key_col1 == NULL && ht1 == NULL)) {
    cudaAssert(0);
  }

  if (key_col2 != NULL && ht2 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col2 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col2 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
  } else if (!(key_col2 == NULL && ht2 == NULL)) {
    cudaAssert(0);
  }

  if (key_col3 != NULL && ht3 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col3 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col3 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
  } else if (!(key_col3 == NULL && ht3 == NULL)) {
    cudaAssert(0);
  }

  if (key_col4 != NULL && ht4 != NULL) {
    //BlockLoadInt(temp_storage.load_items).Load(key_col4 + tile_offset, items, num_tile_items);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col4 + tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  } else if (!(key_col4 == NULL && ht4 == NULL)) {
    cudaAssert(0);
  }

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
__global__ void build_GPU(int* filter_col, int compare1, int compare2, int mode, 
  int *key_col, int *val_col, int num_tuples, 
  int *hash_table, int num_slots, int val_min, int start_offset = 0) {

  int items[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (filter_col != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
    if (mode == 0) { //equal to
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    } else if (mode == 1) { //between
      BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode == 2) { //equal or equal
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode == 3) { //less than
      BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    }  
  }

  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key_col + tile_offset, items, num_tile_items);

  if (val_col != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(val_col + tile_offset, vals, num_tile_items);

    BlockBuildValueGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, blockIdx.x, start_offset,
      items, vals, selection_flags, hash_table, num_slots, val_min, num_tile_items); 
  } else {
    BlockBuildOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, blockIdx.x, start_offset,
      items, selection_flags, hash_table, num_slots, val_min, num_tile_items); 
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_GPU(int* filter_col1, int* filter_col2, 
  int compare1, int compare2, int compare3, int compare4, int mode1, int mode2,
  int* out_off, int* total, int num_tuples, int start_offset = 0) {

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
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col1 + tile_offset, items, num_tile_items);
    if (mode1 == 0) { //equal to
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    } else if (mode1 == 1) { //between
      BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode1 == 2) { //equal or equal
      BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
      BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare2, selection_flags, num_tile_items);
    } else if (mode1 == 3) { //less than
      BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare1, selection_flags, num_tile_items);
    }

    if (filter_col2 != NULL) {
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col2 + tile_offset, items, num_tile_items);
      if (mode2 == 0) { //equal to
        BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
      } else if (mode2 == 1) { //between
        BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 2) { //equal or equal
        BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
        BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 3) { //less than
        BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
      }
    }
  } else {
    if (filter_col2 != NULL) {
      BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col2 + tile_offset, items, num_tile_items);
      if (mode2 == 0) { //equal to
        BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
      } else if (mode2 == 1) { //between
        BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 2) { //equal or equal
        BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
        BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare4, selection_flags, num_tile_items);
      } else if (mode2 == 3) { //less than
        BlockPredLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, compare3, selection_flags, num_tile_items);
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
        cudaAssert(out_off != NULL);
        int offset = block_off + c_t_count++;
        out_off[offset] = start_offset + blockIdx.x * tile_size + (threadIdx.x + ITEM * BLOCK_THREADS);
      }
    }
  }
}

#endif