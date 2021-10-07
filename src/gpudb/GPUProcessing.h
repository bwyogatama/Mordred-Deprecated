#ifndef _GPU_PROCESSING_H_
#define _GPU_PROCESSING_H_

#include <cub/cub.cuh>
#include <curand.h>

#include <cuda.h>

#include "crystal/crystal.cuh"
#include "BlockLibrary.cuh"
#include "KernelArgs.h"
#include "gpu_utils.h"

using namespace cub;


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_group_by_GPU2(
  int* gpuCache, struct filterArgsGPU fargs,
  struct probeArgsGPU pargs, struct groupbyArgsGPU gargs,
  int num_tuples, int* res, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int* ptr;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  __shared__ int key_segment1; 
  __shared__ int key_segment2;
  __shared__ int key_segment3;
  __shared__ int key_segment4;
  __shared__ int aggr_segment1;
  __shared__ int aggr_segment2;
  __shared__ int filter_segment1;
  __shared__ int filter_segment2;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    if (pargs.key_idx1 != NULL) key_segment1 = pargs.key_idx1[segment_index];
    if (pargs.key_idx2 != NULL) key_segment2 = pargs.key_idx2[segment_index];
    if (pargs.key_idx3 != NULL) key_segment3 = pargs.key_idx3[segment_index];
    if (pargs.key_idx4 != NULL) key_segment4 = pargs.key_idx4[segment_index];
    if (gargs.aggr_idx1 != NULL) aggr_segment1 = gargs.aggr_idx1[segment_index];
    if (gargs.aggr_idx2 != NULL) aggr_segment2 = gargs.aggr_idx2[segment_index];
    if (fargs.filter_idx1 != NULL) filter_segment1 = fargs.filter_idx1[segment_index];
    if (fargs.filter_idx2 != NULL) filter_segment2 = fargs.filter_idx2[segment_index];
  }

  __syncthreads();

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);


  if (fargs.filter_idx1 != NULL) {
    ptr = gpuCache + filter_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  if (fargs.filter_idx2 != NULL) {
    ptr = gpuCache + filter_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  }



  if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    ptr = gpuCache + key_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
    ptr = gpuCache + key_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
    ptr = gpuCache + key_segment3 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    ptr = gpuCache + key_segment4 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  }



  if (gargs.aggr_idx1 != NULL) {
    ptr = gpuCache + aggr_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    ptr = gpuCache + aggr_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - gargs.min_val1) * gargs.unique_val1 + (groupval2[ITEM] - gargs.min_val2) * gargs.unique_val2 +  (groupval3[ITEM] - gargs.min_val3) * gargs.unique_val3 + (groupval4[ITEM] - gargs.min_val4) * gargs.unique_val4) % gargs.total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp = (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));

        // if (gargs.mode == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM]));
        // else if (gargs.mode == 1) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] - aggrval2[ITEM]));
        // else if (gargs.mode == 2) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] * aggrval2[ITEM]));
        
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_group_by_GPU3(int* gpuCache, struct offsetGPU offset,
  struct filterArgsGPU fargs, struct probeArgsGPU pargs, struct groupbyArgsGPU gargs, 
  int num_tuples, int* res) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(offset.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.lo_off + tile_offset, items_lo, num_tile_items);

  if (fargs.filter_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, fargs.filter_idx1, num_tile_items);
    // BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  if (fargs.filter_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, fargs.filter_idx2, num_tile_items);
    // BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  }



  if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval1, selection_flags, gpuCache, pargs.key_idx1, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  } else if (gargs.group_idx1 != NULL) { //we take the result from prev join in dim_off but we will also take the groupby column, here gargs.group_idx will be the groupby column (d_year, p_brand1, etc.)
    cudaAssert(offset.dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off1 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, gpuCache, gargs.group_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval2, selection_flags, gpuCache, pargs.key_idx2, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  } else if (gargs.group_idx2 != NULL) {
    cudaAssert(offset.dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off2 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, gpuCache, gargs.group_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval3, selection_flags, gpuCache, pargs.key_idx3, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  } else if (gargs.group_idx3 != NULL) {
    cudaAssert(offset.dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off3 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, gpuCache, gargs.group_idx3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval4, selection_flags, gpuCache, pargs.key_idx4, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else if (gargs.group_idx4 != NULL) {
    cudaAssert(offset.dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off4 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, gpuCache, gargs.group_idx4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  }



  if (gargs.aggr_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval1, gpuCache, gargs.aggr_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval2, gpuCache, gargs.aggr_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - gargs.min_val1) * gargs.unique_val1 + (groupval2[ITEM] - gargs.min_val2) * gargs.unique_val2 +  (groupval3[ITEM] - gargs.min_val3) * gargs.unique_val3 + (groupval4[ITEM] - gargs.min_val4) * gargs.unique_val4) % gargs.total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp = (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));

        // if (gargs.mode == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM]));
        // else if (gargs.mode == 1) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] - aggrval2[ITEM]));
        // else if (gargs.mode == 2) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] * aggrval2[ITEM]));
        
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU2(
  int* gpuCache, struct probeArgsGPU pargs, struct groupbyArgsGPU gargs,
  int num_tuples, int* res, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int* ptr;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  __shared__ int key_segment1; 
  __shared__ int key_segment2;
  __shared__ int key_segment3;
  __shared__ int key_segment4;
  __shared__ int aggr_segment1;
  __shared__ int aggr_segment2;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    if (pargs.key_idx1 != NULL) key_segment1 = pargs.key_idx1[segment_index];
    if (pargs.key_idx2 != NULL) key_segment2 = pargs.key_idx2[segment_index];
    if (pargs.key_idx3 != NULL) key_segment3 = pargs.key_idx3[segment_index];
    if (pargs.key_idx4 != NULL) key_segment4 = pargs.key_idx4[segment_index];
    if (gargs.aggr_idx1 != NULL) aggr_segment1 = gargs.aggr_idx1[segment_index];
    if (gargs.aggr_idx2 != NULL) aggr_segment2 = gargs.aggr_idx2[segment_index];
    // printf("%d\n", key_segment1);
    // printf("%d\n", key_segment2);
    // printf("%d\n", key_segment3);
    // printf("%d\n", key_segment4);
    // printf("%d\n", aggr_segment1);
    // printf("%d\n", aggr_segment2);
  }

  __syncthreads();

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    ptr = gpuCache + key_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
    ptr = gpuCache + key_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
    ptr = gpuCache + key_segment3 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    ptr = gpuCache + key_segment4 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  }



  if (gargs.aggr_idx1 != NULL) {
    ptr = gpuCache + aggr_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    ptr = gpuCache + aggr_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - gargs.min_val1) * gargs.unique_val1 + (groupval2[ITEM] - gargs.min_val2) * gargs.unique_val2 +  (groupval3[ITEM] - gargs.min_val3) * gargs.unique_val3 + (groupval4[ITEM] - gargs.min_val4) * gargs.unique_val4) % gargs.total_val; //!
        // printf("%d %d %d %d\n", groupval1[ITEM], groupval2[ITEM], groupval3[ITEM], groupval4[ITEM]);
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp = (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));

        // if (gargs.mode == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM]));
        // else if (gargs.mode == 1) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] - aggrval2[ITEM]));
        // else if (gargs.mode == 2) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] * aggrval2[ITEM]));
        
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU3 (int* gpuCache, struct offsetGPU offset, struct probeArgsGPU pargs, 
  struct groupbyArgsGPU gargs, int num_tuples, int* res) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(offset.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.lo_off + tile_offset, items_lo, num_tile_items);

  if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval1, selection_flags, gpuCache, pargs.key_idx1, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  } else if (gargs.group_idx1 != NULL) { //we take the result from prev join in dim_off but we will also take the groupby column, here gargs.group_idx will be the groupby column (d_year, p_brand1, etc.)
    cudaAssert(offset.dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off1 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, gpuCache, gargs.group_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval2, selection_flags, gpuCache, pargs.key_idx2, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  } else if (gargs.group_idx2 != NULL) {
    cudaAssert(offset.dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off2 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, gpuCache, gargs.group_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval3, selection_flags, gpuCache, pargs.key_idx3, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  } else if (gargs.group_idx3 != NULL) {
    cudaAssert(offset.dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off3 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, gpuCache, gargs.group_idx3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, groupval4, selection_flags, gpuCache, pargs.key_idx4, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else if (gargs.group_idx4 != NULL) {
    cudaAssert(offset.dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off4 + tile_offset, items, num_tile_items);
    BlockReadFilteredOffset<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, gpuCache, gargs.group_idx4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  }



  if (gargs.aggr_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval1, gpuCache, gargs.aggr_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval2, gpuCache, gargs.aggr_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - gargs.min_val1) * gargs.unique_val1 + (groupval2[ITEM] - gargs.min_val2) * gargs.unique_val2 +  (groupval3[ITEM] - gargs.min_val3) * gargs.unique_val3 + (groupval4[ITEM] - gargs.min_val4) * gargs.unique_val4) % gargs.total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp = (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));

        // if (gargs.mode == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM]));
        // else if (gargs.mode == 1) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] - aggrval2[ITEM]));
        // else if (gargs.mode == 2) atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggrval1[ITEM] * aggrval2[ITEM]));
        
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_GPU2(
  int* gpuCache, struct filterArgsGPU fargs, struct probeArgsGPU pargs, struct offsetGPU out_off, int num_tuples,
  int *total, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size; //how many block per segment
  int* ptr;

  int tile_idx = blockIdx.x % tiles_per_segment;    // Current tile index
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockScanInt::TempStorage scan;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  // int dim_offset1[ITEMS_PER_THREAD];
  // int dim_offset2[ITEMS_PER_THREAD];
  // int dim_offset3[ITEMS_PER_THREAD];
  int dim_offset4[ITEMS_PER_THREAD];
  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  // __shared__ int key_segment1; 
  // __shared__ int key_segment2;
  // __shared__ int key_segment3;
  __shared__ int key_segment4;
  __shared__ int filter_segment1;
  __shared__ int filter_segment2;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    if (segment_group != NULL) segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    else segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    // if (pargs.key_idx1 != NULL) key_segment1 = pargs.key_idx1[segment_index];
    // if (pargs.key_idx2 != NULL) key_segment2 = pargs.key_idx2[segment_index];
    // if (pargs.key_idx3 != NULL) key_segment3 = pargs.key_idx3[segment_index];
    if (pargs.key_idx4 != NULL) key_segment4 = pargs.key_idx4[segment_index];
    if (fargs.filter_idx1 != NULL) filter_segment1 = fargs.filter_idx1[segment_index];
    if (fargs.filter_idx2 != NULL) filter_segment2 = fargs.filter_idx2[segment_index];
  }

  __syncthreads();

  start_offset = segment_index * SEGMENT_SIZE;

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (fargs.filter_idx1 != NULL) {
    ptr = gpuCache + filter_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  if (fargs.filter_idx2 != NULL) {
    ptr = gpuCache + filter_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  }

  // if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //we are doing probing for this column (normal operation)
  //   ptr = gpuCache + key_segment1 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  // } else { //we are not doing join for this column, there is no result from prev join (first join)
  //   BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset1, 1, num_tile_items);
  // }

  // if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
  //   ptr = gpuCache + key_segment2 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  // } else {
  //   BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset2, 1, num_tile_items);
  // }

  // if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
  //   ptr = gpuCache + key_segment3 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  // } else {
  //   BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset3, 1, num_tile_items);
  // }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    ptr = gpuCache + key_segment4 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset4, 1, num_tile_items);
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
        out_off.lo_off[offset] = start_offset + tile_idx * tile_size + threadIdx.x + ITEM * BLOCK_THREADS;
        // if (out_off.dim_off1 != NULL) out_off.dim_off1[offset] = dim_offset1[ITEM];
        // if (out_off.dim_off2 != NULL) out_off.dim_off2[offset] = dim_offset2[ITEM];
        // if (out_off.dim_off3 != NULL) out_off.dim_off3[offset] = dim_offset3[ITEM];
        if (out_off.dim_off4 != NULL) out_off.dim_off4[offset] = dim_offset4[ITEM];
        //printf("%d %d %d %d\n", dim_offset1[ITEM], dim_offset2[ITEM], dim_offset3[ITEM], dim_offset4[ITEM]);
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_GPU3(
  int* gpuCache, struct offsetGPU in_off, struct filterArgsGPU fargs, struct probeArgsGPU pargs, 
  struct offsetGPU out_off, int num_tuples, int *total) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
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
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  // int dim_offset1[ITEMS_PER_THREAD];
  // int dim_offset2[ITEMS_PER_THREAD];
  // int dim_offset3[ITEMS_PER_THREAD];
  int dim_offset4[ITEMS_PER_THREAD];
  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(in_off.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.lo_off + tile_offset, items_lo, num_tile_items);

  if (fargs.filter_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, fargs.filter_idx1, num_tile_items);
    // BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  if (fargs.filter_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, fargs.filter_idx2, num_tile_items);
    // BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  }

  // if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //we are doing probing for this column (normal operation)
  //   BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx1, num_tile_items);
  //   BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  // } else if (in_off.dim_off1 != NULL) { //load result from prev join, we are just passing it through (no probing)
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off1 + tile_offset, dim_offset1, num_tile_items);
  // } else { //we are not doing join for this column, there is no result from prev join (first join)
  //   BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset1, 1, num_tile_items);
  // }

  // if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
  //   BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx2, num_tile_items);
  //   BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  // } else if (in_off.dim_off2 != NULL) {
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off2 + tile_offset, dim_offset2, num_tile_items);
  // } else {
  //   BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset2, 1, num_tile_items);
  // }

  // if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
  //   BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx3, num_tile_items);
  //   BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  // } else if (in_off.dim_off3 != NULL) {
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off3 + tile_offset, dim_offset3, num_tile_items);
  // } else {
  //   BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset3, 1, num_tile_items);
  // }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx4, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else if (in_off.dim_off4 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off4 + tile_offset, dim_offset4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset4, 1, num_tile_items);
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
        out_off.lo_off[offset] = items_lo[ITEM];
        // if (out_off.dim_off1 != NULL) out_off.dim_off1[offset] = dim_offset1[ITEM];
        // if (out_off.dim_off2 != NULL) out_off.dim_off2[offset] = dim_offset2[ITEM];
        // if (out_off.dim_off3 != NULL) out_off.dim_off3[offset] = dim_offset3[ITEM];
        if (out_off.dim_off4 != NULL) out_off.dim_off4[offset] = dim_offset4[ITEM];
        //printf("%d %d %d %d\n", dim_offset1[ITEM], dim_offset2[ITEM], dim_offset3[ITEM], dim_offset4[ITEM]);
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_GPU2(
  int* gpuCache, struct probeArgsGPU pargs, struct offsetGPU out_off, int num_tuples,
  int *total, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScanInt;
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size; //how many block per segment
  int* ptr;

  int tile_idx = blockIdx.x % tiles_per_segment;    // Current tile index
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  __shared__ int key_segment1; 
  __shared__ int key_segment2;
  __shared__ int key_segment3;
  __shared__ int key_segment4;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    if (segment_group != NULL) segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    else segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    if (pargs.key_idx1 != NULL) key_segment1 = pargs.key_idx1[segment_index];
    if (pargs.key_idx2 != NULL) key_segment2 = pargs.key_idx2[segment_index];
    if (pargs.key_idx3 != NULL) key_segment3 = pargs.key_idx3[segment_index];
    if (pargs.key_idx4 != NULL) key_segment4 = pargs.key_idx4[segment_index];
  }

  __syncthreads();

  start_offset = segment_index * SEGMENT_SIZE;

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);


  if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //we are doing probing for this column (normal operation)
    ptr = gpuCache + key_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  } else { //we are not doing join for this column, there is no result from prev join (first join)
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset1, 1, num_tile_items);
  }

  if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
    ptr = gpuCache + key_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset2, 1, num_tile_items);
  }

  if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
    ptr = gpuCache + key_segment3 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset3, 1, num_tile_items);
  }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    ptr = gpuCache + key_segment4 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset4, 1, num_tile_items);
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
        out_off.lo_off[offset] = start_offset + tile_idx * tile_size + threadIdx.x + ITEM * BLOCK_THREADS;
        if (out_off.dim_off1 != NULL) out_off.dim_off1[offset] = dim_offset1[ITEM];
        if (out_off.dim_off2 != NULL) out_off.dim_off2[offset] = dim_offset2[ITEM];
        if (out_off.dim_off3 != NULL) out_off.dim_off3[offset] = dim_offset3[ITEM];
        if (out_off.dim_off4 != NULL) out_off.dim_off4[offset] = dim_offset4[ITEM];
        //printf("%d %d %d %d\n", dim_offset1[ITEM], dim_offset2[ITEM], dim_offset3[ITEM], dim_offset4[ITEM]);
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_GPU3(
  int* gpuCache, struct offsetGPU in_off, 
  struct probeArgsGPU pargs, struct offsetGPU out_off, int num_tuples,
  int *total) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(in_off.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.lo_off + tile_offset, items_lo, num_tile_items);

  if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //we are doing probing for this column (normal operation)
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx1, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset1, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  } else if (in_off.dim_off1 != NULL) { //load result from prev join, we are just passing it through (no probing)
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off1 + tile_offset, dim_offset1, num_tile_items);
  } else { //we are not doing join for this column, there is no result from prev join (first join)
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset1, 1, num_tile_items);
  }

  if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx2, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset2, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  } else if (in_off.dim_off2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off2 + tile_offset, dim_offset2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset2, 1, num_tile_items);
  }

  if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx3, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset3, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  } else if (in_off.dim_off3 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off3 + tile_offset, dim_offset3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset3, 1, num_tile_items);
  }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, items, gpuCache, pargs.key_idx4, num_tile_items);
    BlockProbeGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, dim_offset4, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  } else if (in_off.dim_off4 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(in_off.dim_off4 + tile_offset, dim_offset4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, dim_offset4, 1, num_tile_items);
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
        // printf("%d\n", items_lo[ITEM]);
        out_off.lo_off[offset] = items_lo[ITEM];
        // printf("%d\n", items_lo[ITEM]);
        if (out_off.dim_off1 != NULL) out_off.dim_off1[offset] = dim_offset1[ITEM];
        if (out_off.dim_off2 != NULL) out_off.dim_off2[offset] = dim_offset2[ITEM];
        if (out_off.dim_off3 != NULL) out_off.dim_off3[offset] = dim_offset3[ITEM];
        if (out_off.dim_off4 != NULL) out_off.dim_off4[offset] = dim_offset4[ITEM];
        // printf("%d %d %d %d\n", dim_offset1[ITEM], dim_offset2[ITEM], dim_offset3[ITEM], dim_offset4[ITEM]);
      }
    }
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU2(
  int* gpuCache, struct filterArgsGPU fargs,
  struct buildArgsGPU bargs, int num_tuples, int* hash_table,
  int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  int items[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size;

  int tile_idx = blockIdx.x % tiles_per_segment;
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  __shared__ int segment_index;
  __shared__ int val_segment;
  __shared__ int key_segment;
  __shared__ int filter_segment;

  if (threadIdx.x == 0) {
    if (segment_group != NULL) segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    else segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    if (bargs.val_idx != NULL) val_segment = bargs.val_idx[segment_index];
    if (bargs.key_idx != NULL) key_segment = bargs.key_idx[segment_index];
    if (fargs.filter_idx1 != NULL) filter_segment = fargs.filter_idx1[segment_index];
  }

  __syncthreads();

  start_offset = segment_index * SEGMENT_SIZE;

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (fargs.filter_idx1 != NULL) {
    int* ptr = gpuCache + filter_segment * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // if (fargs.mode1 == 0) { //equal to
    //   BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // } else if (fargs.mode1 == 1) { //between
    //   BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    //   BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    // } else if (fargs.mode1 == 2) { //equal or equal
    //   BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    //   BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    // }
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  cudaAssert(bargs.key_idx != NULL);
  int* ptr_key = gpuCache + key_segment * SEGMENT_SIZE;
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr_key + segment_tile_offset, items, num_tile_items);

  if (bargs.val_idx != NULL) {
    int* ptr = gpuCache + val_segment * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, vals, num_tile_items);
    BlockBuildValueGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, tile_idx, start_offset,
      items, vals, selection_flags, hash_table, bargs.num_slots, bargs.val_min, num_tile_items);
  } else {
    BlockBuildOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, tile_idx, start_offset,
      items, selection_flags, hash_table, bargs.num_slots, bargs.val_min, num_tile_items);
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU3(
  int* gpuCache, int* dim_off, struct filterArgsGPU fargs,
  struct buildArgsGPU bargs, int num_tuples, int* hash_table) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(fargs.filter_idx1 == NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off + tile_offset, items, num_tile_items);
  if (bargs.val_idx != NULL) {
    BlockBuildValueGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, bargs.key_idx, bargs.val_idx, 
        hash_table, bargs.num_slots, bargs.val_min, num_tile_items);
  } else {
    BlockBuildOffsetGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, bargs.key_idx, 
        hash_table, bargs.num_slots, bargs.val_min, num_tile_items); 
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU_minmax(
  int* gpuCache, struct filterArgsGPU fargs,
  struct buildArgsGPU bargs, int num_tuples, int* hash_table, int* min_global, int* max_global,
  int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  int items[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size;

  int tile_idx = blockIdx.x % tiles_per_segment;
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  int min = bargs.val_max, max = bargs.val_min;

  __shared__ int segment_index;
  __shared__ int val_segment;
  __shared__ int key_segment;
  __shared__ int filter_segment;
  __shared__ int min_shared;
  __shared__ int max_shared;

  if (threadIdx.x == 0) {
    if (segment_group != NULL) segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    else segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    if (bargs.val_idx != NULL) val_segment = bargs.val_idx[segment_index];
    if (bargs.key_idx != NULL) key_segment = bargs.key_idx[segment_index];
    if (fargs.filter_idx1 != NULL) filter_segment = fargs.filter_idx1[segment_index];
    min_shared = bargs.val_max;
    max_shared = bargs.val_min;
  }

  __syncthreads();

  start_offset = segment_index * SEGMENT_SIZE;

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (fargs.filter_idx1 != NULL) {
    int* ptr = gpuCache + filter_segment * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  cudaAssert(bargs.key_idx != NULL);
  int* ptr_key = gpuCache + key_segment * SEGMENT_SIZE;
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr_key + segment_tile_offset, items, num_tile_items);
  BlockMinMaxGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, min, max, num_tile_items);

  if (bargs.val_idx != NULL) {
    int* ptr = gpuCache + val_segment * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, vals, num_tile_items);
    BlockBuildValueGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, tile_idx, start_offset,
      items, vals, selection_flags, hash_table, bargs.num_slots, bargs.val_min, num_tile_items);
  } else {
    BlockBuildOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, tile_idx, start_offset,
      items, selection_flags, hash_table, bargs.num_slots, bargs.val_min, num_tile_items);
  }

  __syncthreads();

  atomicMin(&min_shared, min);
  atomicMax(&max_shared, max);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMin(min_global, min_shared);
    atomicMax(max_global, max_shared);    
  }

  __syncthreads();

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU_minmax2(
  int* gpuCache, int* dim_off, struct filterArgsGPU fargs,
  struct buildArgsGPU bargs, int num_tuples, int* hash_table, int* min_global, int* max_global) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)

  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  int min = bargs.val_max, max = bargs.val_min;

  __shared__ int min_shared;
  __shared__ int max_shared;

  if (threadIdx.x == 0) {
    min_shared = bargs.val_max;
    max_shared = bargs.val_min;
  }

  __syncthreads();

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(fargs.filter_idx1 == NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_off + tile_offset, items, num_tile_items);
  BlockMinMaxGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, bargs.key_idx, min, max, num_tile_items);

  if (bargs.val_idx != NULL) {
    BlockBuildValueGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, bargs.key_idx, bargs.val_idx, 
        hash_table, bargs.num_slots, bargs.val_min, num_tile_items);
  } else {
    BlockBuildOffsetGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, gpuCache, bargs.key_idx, 
        hash_table, bargs.num_slots, bargs.val_min, num_tile_items); 
  }

  __syncthreads();

  atomicMin(&min_shared, min);
  atomicMax(&max_shared, max);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMin(min_global, min_shared);
    atomicMax(max_global, max_shared);    
  }

  __syncthreads();

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_GPU2(
  int* gpuCache, struct filterArgsGPU fargs,
  int* out_off, int num_tuples, int* total, int start_offset = 0, short* segment_group = NULL) {

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
  int selection_flags[ITEMS_PER_THREAD];

  int t_count = 0; // Number of items selected per thread
  int c_t_count = 0; //Prefix sum of t_count
  __shared__ int block_off;

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int tiles_per_segment = SEGMENT_SIZE/tile_size;

  int tile_idx = blockIdx.x % tiles_per_segment;    // Current tile index
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  __shared__ int filter_segment1; 
  __shared__ int filter_segment2;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    if (segment_group != NULL) segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    else segment_index = ( start_offset + tile_offset ) / SEGMENT_SIZE;
    if (fargs.filter_idx1 != NULL) filter_segment1 = fargs.filter_idx1[segment_index];
    if (fargs.filter_idx2 != NULL) filter_segment2 = fargs.filter_idx2[segment_index];
  }

  __syncthreads();

  start_offset = segment_index * SEGMENT_SIZE;


  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (fargs.filter_idx1 != NULL) {
    int* ptr = gpuCache + filter_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);

    // if (fargs.mode1 == 0) { //equal to
    //   BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // } else if (fargs.mode1 == 1) { //between
    //   BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    //   BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    // } else if (fargs.mode1 == 2) { //equal or equal
    //   BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    //   BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    // }
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  if (fargs.filter_idx2 != NULL) {
    int* ptr = gpuCache + filter_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);

    // if (fargs.mode2 == 0) { //equal to
    //   BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // } else if (fargs.mode2 == 1) { //between
    //   BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    //   BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    // } else if (fargs.mode2 == 2) { //equal or equal
    //   BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    //   BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    // }
    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
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

  cudaAssert(out_off != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        out_off[offset] = start_offset + tile_idx * tile_size + threadIdx.x + ITEM * BLOCK_THREADS;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_GPU3(
  int* gpuCache, int* off_col, struct filterArgsGPU fargs,
  int* out_off, int num_tuples, int* total) {

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

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  cudaAssert(off_col != NULL);

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  // if (fargs.filter_idx1 != NULL) {
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(off_col + tile_offset, items_off, num_tile_items);
  //   BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, fargs.filter_idx1, num_tile_items);

  //   if (fargs.mode1 == 0) { //equal to
  //     BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
  //   } else if (fargs.mode1 == 1) { //between
  //     BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
  //     BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
  //   } else if (fargs.mode1 == 2) { //equal or equal
  //     BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
  //     BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
  //   }
  //    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  // }

  // if (fargs.filter_idx2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(off_col + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, items, gpuCache, fargs.filter_idx2, num_tile_items);

    // if (fargs.mode2 == 0) { //equal to
    //   BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // } else if (fargs.mode2 == 1) { //between
    //   BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    //   BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    // } else if (fargs.mode2 == 2) { //equal or equal
    //   BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    //   BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    // }

    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  // }

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

  cudaAssert(out_off != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        out_off[offset] = items_off[ITEM];
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void groupByGPU(
  int* gpuCache, struct offsetGPU offset, struct groupbyArgsGPU gargs,
  int num_tuples, int* res) {

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

  cudaAssert(offset.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.lo_off + tile_offset, items_off, num_tile_items);

  if (gargs.aggr_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval1, gpuCache, gargs.aggr_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval2, gpuCache, gargs.aggr_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  if (gargs.group_idx1 != NULL) {
    cudaAssert(offset.dim_off1 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off1 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval1, gpuCache, gargs.group_idx1, num_tile_items);    
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (gargs.group_idx2 != NULL) {
    cudaAssert(offset.dim_off2 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off2 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval2, gpuCache, gargs.group_idx2, num_tile_items);    
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (gargs.group_idx3 != NULL) {
    cudaAssert(offset.dim_off3 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off3 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval3, gpuCache, gargs.group_idx3, num_tile_items);    
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (gargs.group_idx4 != NULL) {
    cudaAssert(offset.dim_off4 != NULL);
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.dim_off4 + tile_offset, items_off, num_tile_items);
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, groupval4, gpuCache, gargs.group_idx4, num_tile_items);   
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  }

  // if (gargs.mode == 0) {
  //   cudaAssert(gargs.aggr_idx1 != NULL);
  // } else if (gargs.mode == 1) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // } else if (gargs.mode == 2) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      int hash = ((groupval1[ITEM] - gargs.min_val1) * gargs.unique_val1 + (groupval2[ITEM] - gargs.min_val2) * gargs.unique_val2 +  (groupval3[ITEM] - gargs.min_val3) * gargs.unique_val3 + (groupval4[ITEM] - gargs.min_val4) * gargs.unique_val4) % gargs.total_val; //!

      //printf("%d %d %d %d\n", groupval1[ITEM], groupval2[ITEM], groupval3[ITEM], groupval4[ITEM]);
      res[hash * 6] = groupval1[ITEM];
      res[hash * 6 + 1] = groupval2[ITEM];
      res[hash * 6 + 2] = groupval3[ITEM];
      res[hash * 6 + 3] = groupval4[ITEM];

      // if (gargs.mode == 0) temp = aggrval1[ITEM];
      // else if (gargs.mode == 1) temp = aggrval1[ITEM] - aggrval2[ITEM];
      // else if (gargs.mode == 2) temp = aggrval1[ITEM] * aggrval2[ITEM];
      int temp = (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
      atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
    }
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void aggregationGPU(
  int* gpuCache, int* lo_off, struct groupbyArgsGPU gargs, int num_tuples, int* res) {

  int items_off[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  int tile_offset = blockIdx.x * tile_size;
  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  cudaAssert(lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_off + tile_offset, items_off, num_tile_items);

  if (gargs.aggr_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval1, gpuCache, gargs.aggr_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_off, aggrval2, gpuCache, gargs.aggr_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }


  // if (gargs.mode == 0) {
  //   cudaAssert(gargs.aggr_idx1 != NULL);
  // } else if (gargs.mode == 1) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // } else if (gargs.mode == 2) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // }

  long long sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      // if (gargs.mode == 0) sum += aggrval1[ITEM];
      // else if (gargs.mode == 1) sum+= (aggrval1[ITEM] - aggrval2[ITEM]);
      // else if (gargs.mode == 2) sum+= (aggrval1[ITEM] * aggrval2[ITEM]);
      sum += (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
    }
  }

  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[4]), aggregate);   
  }

}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_aggr_GPU2(
  int* gpuCache, struct probeArgsGPU pargs, struct groupbyArgsGPU gargs, int num_tuples,
  int* res, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int* ptr;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];
  int temp[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  // __shared__ int key_segment1; 
  // __shared__ int key_segment2;
  // __shared__ int key_segment3;
  __shared__ int key_segment4;
  __shared__ int aggr_segment1;
  __shared__ int aggr_segment2;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    // if (pargs.key_idx1 != NULL) key_segment1 = pargs.key_idx1[segment_index];
    // if (pargs.key_idx2 != NULL) key_segment2 = pargs.key_idx2[segment_index];
    // if (pargs.key_idx3 != NULL) key_segment3 = pargs.key_idx3[segment_index];
    if (pargs.key_idx4 != NULL) key_segment4 = pargs.key_idx4[segment_index];
    if (gargs.aggr_idx1 != NULL) aggr_segment1 = gargs.aggr_idx1[segment_index];
    if (gargs.aggr_idx2 != NULL) aggr_segment2 = gargs.aggr_idx2[segment_index];
  }

  __syncthreads();

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  // if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
  //   ptr = gpuCache + key_segment1 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  // }

  // if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
  //   ptr = gpuCache + key_segment2 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  // }

  // if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
  //   ptr = gpuCache + key_segment3 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  // }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    ptr = gpuCache + key_segment4 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  }

  if (gargs.aggr_idx1 != NULL) {
    ptr = gpuCache + aggr_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    ptr = gpuCache + aggr_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  // if (gargs.mode == 0) {
  //   cudaAssert(gargs.aggr_idx1 != NULL);
  // } else if (gargs.mode == 1) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // } else if (gargs.mode == 2) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // }

  long long sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        // if (gargs.mode == 0) sum += aggrval1[ITEM];
        // else if (gargs.mode == 1) sum+= (aggrval1[ITEM] - aggrval2[ITEM]);
        // else if (gargs.mode == 2) sum+= (aggrval1[ITEM] * aggrval2[ITEM]);
        sum += (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
      }
    }
  }


  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[4]), aggregate);   
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_aggr_GPU3(
  int* gpuCache, struct offsetGPU offset, struct probeArgsGPU pargs, 
  struct groupbyArgsGPU gargs, int num_tuples, int* res) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];
  int temp[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(offset.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.lo_off + tile_offset, items_lo, num_tile_items);

  // if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
  //   BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx1, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  // }

  // if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
  //   BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx2, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  // }

  // if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
  //   BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx3, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  // }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx4, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  }

  if (gargs.aggr_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval1, gpuCache, gargs.aggr_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval2, gpuCache, gargs.aggr_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }


  // if (gargs.mode == 0) {
  //   cudaAssert(gargs.aggr_idx1 != NULL);
  // } else if (gargs.mode == 1) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // } else if (gargs.mode == 2) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // }

  long long sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        // if (gargs.mode == 0) sum += aggrval1[ITEM];
        // else if (gargs.mode == 1) sum+= (aggrval1[ITEM] - aggrval2[ITEM]);
        // else if (gargs.mode == 2) sum+= (aggrval1[ITEM] * aggrval2[ITEM]);
        sum += (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
      }
    }
  }


  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[4]), aggregate);   
  }
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_aggr_GPU2(
  int* gpuCache, struct filterArgsGPU fargs, struct probeArgsGPU pargs, 
  struct groupbyArgsGPU gargs, int num_tuples,
  int* res, int start_offset = 0, short* segment_group = NULL) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;
  int* ptr;

  int tiles_per_segment = SEGMENT_SIZE/tile_size;
  int segment_tile_offset = (blockIdx.x % tiles_per_segment) * tile_size; //tile offset inside a segment

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];
  int temp[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  // __shared__ int key_segment1; 
  // __shared__ int key_segment2;
  // __shared__ int key_segment3;
  __shared__ int key_segment4;
  __shared__ int aggr_segment1;
  __shared__ int aggr_segment2;
  __shared__ int filter_segment1;
  __shared__ int filter_segment2;
  __shared__ int segment_index;

  if (threadIdx.x == 0) {
    segment_index = segment_group[tile_offset / SEGMENT_SIZE];
    // if (pargs.key_idx1 != NULL) key_segment1 = pargs.key_idx1[segment_index];
    // if (pargs.key_idx2 != NULL) key_segment2 = pargs.key_idx2[segment_index];
    // if (pargs.key_idx3 != NULL) key_segment3 = pargs.key_idx3[segment_index];
    if (pargs.key_idx4 != NULL) key_segment4 = pargs.key_idx4[segment_index];
    if (gargs.aggr_idx1 != NULL) aggr_segment1 = gargs.aggr_idx1[segment_index];
    if (gargs.aggr_idx2 != NULL) aggr_segment2 = gargs.aggr_idx2[segment_index];
    if (fargs.filter_idx1 != NULL) filter_segment1 = fargs.filter_idx1[segment_index];
    if (fargs.filter_idx2 != NULL) filter_segment2 = fargs.filter_idx2[segment_index];
  }

  __syncthreads();

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);


  if (fargs.filter_idx1 != NULL) {
    ptr = gpuCache + filter_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare1, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare2, selection_flags, num_tile_items);
    (*(fargs.d_filter_func1))(items, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  }

  if (fargs.filter_idx2 != NULL) {
    ptr = gpuCache + filter_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    // BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare3, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, fargs.compare4, selection_flags, num_tile_items);
    (*(fargs.d_filter_func2))(items, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  }

  // if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
  //   ptr = gpuCache + key_segment1 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  // }

  // if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
  //   ptr = gpuCache + key_segment2 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  // }

  // if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
  //   ptr = gpuCache + key_segment3 * SEGMENT_SIZE;
  //   BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
  //   BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  // }

  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    ptr = gpuCache + key_segment4 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  }

  if (gargs.aggr_idx1 != NULL) {
    ptr = gpuCache + aggr_segment1 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    ptr = gpuCache + aggr_segment2 * SEGMENT_SIZE;
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(ptr + segment_tile_offset, aggrval2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  // if (gargs.mode == 0) {
  //   cudaAssert(gargs.aggr_idx1 != NULL);
  // } else if (gargs.mode == 1) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // } else if (gargs.mode == 2) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // }

  long long sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        // if (gargs.mode == 0) sum += aggrval1[ITEM];
        // else if (gargs.mode == 1) sum+= (aggrval1[ITEM] - aggrval2[ITEM]);
        // else if (gargs.mode == 2) sum+= (aggrval1[ITEM] * aggrval2[ITEM]);
        sum += (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
      }
    }
  }

  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[4]), aggregate);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_aggr_GPU3(
  int* gpuCache, struct offsetGPU offset, struct filterArgsGPU fargs, 
  struct probeArgsGPU pargs, struct groupbyArgsGPU gargs,
  int num_tuples, int* res) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int items_lo[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];
  int temp[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  cudaAssert(offset.lo_off != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(offset.lo_off + tile_offset, items_lo, num_tile_items);

  // if (fargs.filter_idx1 != NULL) {
  //   BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, gpuCache, fargs.filter_idx1, num_tile_items);
  //   BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(temp, fargs.compare1, selection_flags, num_tile_items);
  //   BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(temp, fargs.compare2, selection_flags, num_tile_items);
      // (*(fargs.d_filter_func1))(temp, selection_flags, fargs.compare1, fargs.compare2, num_tile_items);
  // }

  if (fargs.filter_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, gpuCache, fargs.filter_idx2, num_tile_items);
    // BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(temp, fargs.compare3, selection_flags, num_tile_items);
    // BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(temp, fargs.compare4, selection_flags, num_tile_items);
    (*(fargs.d_filter_func2))(temp, selection_flags, fargs.compare3, fargs.compare4, num_tile_items);
  }


  // if (pargs.key_idx1 != NULL && pargs.ht1 != NULL) { //normal operation, here pargs.key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
  //   BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx1, pargs.ht1, pargs.dim_len1, pargs.min_key1, num_tile_items);
  // }


  // if (pargs.key_idx2 != NULL && pargs.ht2 != NULL) {
  //   BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx2, pargs.ht2, pargs.dim_len2, pargs.min_key2, num_tile_items);
  // }


  // if (pargs.key_idx3 != NULL && pargs.ht3 != NULL) {
  //   BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx3, pargs.ht3, pargs.dim_len3, pargs.min_key3, num_tile_items);
  // }


  if (pargs.key_idx4 != NULL && pargs.ht4 != NULL) {
    BlockProbeGroupByGPU2<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, temp, selection_flags, gpuCache, pargs.key_idx4, pargs.ht4, pargs.dim_len4, pargs.min_key4, num_tile_items);
  }

  if (gargs.aggr_idx1 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval1, gpuCache, gargs.aggr_idx1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (gargs.aggr_idx2 != NULL) {
    BlockReadOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items_lo, aggrval2, gpuCache, gargs.aggr_idx2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  // if (gargs.mode == 0) {
  //   cudaAssert(gargs.aggr_idx1 != NULL);
  // } else if (gargs.mode == 1) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // } else if (gargs.mode == 2) {
  //   cudaAssert(gargs.aggr_idx1 != NULL && gargs.aggr_idx2 != NULL);
  // }

  long long sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        // if (gargs.mode == 0) sum += aggrval1[ITEM];
        // else if (gargs.mode == 1) sum+= (aggrval1[ITEM] - aggrval2[ITEM]);
        // else if (gargs.mode == 2) sum+= (aggrval1[ITEM] * aggrval2[ITEM]);
        sum += (*(gargs.d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
      }
    }
  }

  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[4]), aggregate);   
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_GPU(
  int* key, int* val, int* filter, int compare1, int compare2, filter_func_t_dev<int, 128, 4> d_filter_func1,
  int num_tuples, int *hash_table, int num_slots, int val_min, int segment_index) {

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

  int start_offset = segment_index * SEGMENT_SIZE;

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (filter != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter + tile_offset, items, num_tile_items);
    (*(d_filter_func1))(items, selection_flags, compare1, compare2, num_tile_items);
  }

  cudaAssert(key != NULL);
  BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key + tile_offset, items, num_tile_items);

  if (val != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(val + tile_offset, vals, num_tile_items);
    BlockBuildValueGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, blockIdx.x, start_offset,
      items, vals, selection_flags, hash_table, num_slots, val_min, num_tile_items);
  } else {
    BlockBuildOffsetGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, blockIdx.x, start_offset,
      items, selection_flags, hash_table, num_slots, val_min, num_tile_items);
  }

}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void filter_probe_aggr_GPU(
  int* filter1, int* filter2, int compare1, int compare2, int compare3, int compare4,
  filter_func_t_dev<int, 128, 4> d_filter_func1, filter_func_t_dev<int, 128, 4> d_filter_func2,
  int* key4, int* ht4, int dim_len4, int min_key4,
  int* aggr1, int* aggr2, group_func_t<int> d_group_func,
  int num_tuples, int* res) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_offset = blockIdx.x * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int aggrval1[ITEMS_PER_THREAD];
  int aggrval2[ITEMS_PER_THREAD];
  int temp[ITEMS_PER_THREAD];

  int num_tiles = (num_tuples + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);


  if (filter1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter1 + tile_offset, items, num_tile_items);
    (*(d_filter_func1))(items, selection_flags, compare1, compare2, num_tile_items);
  }

  if (filter2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter2 + tile_offset, items, num_tile_items);
    (*(d_filter_func2))(items, selection_flags, compare3, compare4, num_tile_items);
  }

  if (key4 != NULL && ht4 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key4 + tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, temp, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  }

  if (aggr1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr1 + tile_offset, aggrval1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (aggr2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr2 + tile_offset, aggrval2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  long long sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        sum += (*(d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
      }
    }
  }

  __syncthreads();
  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&res[4]), aggregate);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_GPU(
  int* key1, int* key2, int* key3, int* key4, 
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* aggr1, int* aggr2,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4,
  int total_val, group_func_t<int> d_group_func,
  int num_tuples, int* res) {

  //assume start_offset always in the beginning of a segment (ga mungkin start di tengah2 segment)
  //assume tile_size is a factor of SEGMENT_SIZE (SEGMENT SIZE kelipatan tile_size)
  
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
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  if (key1 != NULL && ht1 != NULL) { //normal operation, here key_idx will be lo_partkey, lo_suppkey, etc (the join key column) -> no group by attributes
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key1 + tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval1, selection_flags, ht1, dim_len1, min_key1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval1, 0, num_tile_items);
  }

  if (key2 != NULL && ht2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key2 + tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval2, selection_flags, ht2, dim_len2, min_key2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval2, 0, num_tile_items);
  }

  if (key3 != NULL && ht3 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key3 + tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval3, selection_flags, ht3, dim_len3, min_key3, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval3, 0, num_tile_items);
  }

  if (key4 != NULL && ht4 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key4 + tile_offset, items, num_tile_items);
    BlockProbeGroupByGPU<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, groupval4, selection_flags, ht4, dim_len4, min_key4, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, groupval4, 0, num_tile_items);
  }

  if (aggr1 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr1 + tile_offset, aggrval1, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval1, 0, num_tile_items);
  }

  if (aggr2 != NULL) {
    BlockLoadCrystal<int, BLOCK_THREADS, ITEMS_PER_THREAD>(aggr2 + tile_offset, aggrval2, num_tile_items);
  } else {
    BlockSetValue<BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, aggrval2, 0, num_tile_items);
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = ((groupval1[ITEM] - min_val1) * unique_val1 + (groupval2[ITEM] - min_val2) * unique_val2 +  (groupval3[ITEM] - min_val3) * unique_val3 + (groupval4[ITEM] - min_val4) * unique_val4) % total_val; //!
        res[hash * 6] = groupval1[ITEM];
        res[hash * 6 + 1] = groupval2[ITEM];
        res[hash * 6 + 2] = groupval3[ITEM];
        res[hash * 6 + 3] = groupval4[ITEM];

        int temp = (*(d_group_func))(aggrval1[ITEM], aggrval2[ITEM]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp));
        
      }
    }
  }
}

#endif