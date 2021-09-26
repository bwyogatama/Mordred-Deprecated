#ifndef _BLOCK_LIBRARY_H_
#define _BLOCK_LIBRARY_H_

#pragma once

#define cudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGPUDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (selection_flags[ITEM]) { 
      int hash = HASH(items[ITEM], ht_len, keys_min); 
      int slot = ht[(hash << 1) + 1];
      if (slot != 0) {
        offset[ITEM] = slot - 1;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGPUDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) { 
        int hash = HASH(items[ITEM], ht_len, keys_min); 
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

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, offset, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, offset, selection_flags, ht, ht_len, keys_min, num_items);
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGroupByGPUDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (selection_flags[ITEM]) {
      int hash = HASH(items[ITEM], ht_len, keys_min);
      uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
      if (slot != 0) {
        //res[ITEM] = (slot << 32) >> 32;
        res[ITEM] = slot;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGroupByGPUDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
        if (slot != 0) {
          //res[ITEM] = (slot << 32) >> 32;
          res[ITEM] = slot;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

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

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeGroupByGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, res, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeGroupByGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, res, selection_flags, ht, ht_len, keys_min, num_items);
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (selection_flags[ITEM]) {  
      int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
      int hash = HASH(key, ht_len, keys_min);
      int slot = ht[(hash << 1) + 1];
      if (slot != 0) {
        offset[ITEM] = slot - 1;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {  
        int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
        int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
        int hash = HASH(key, ht_len, keys_min);
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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGPU2(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, offset, selection_flags, gpuCache, key_idx, ht, ht_len, keys_min);
  } else {
    BlockProbeGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, offset, selection_flags, gpuCache, key_idx, ht, ht_len, keys_min, num_items);
  }

}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGroupByGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (selection_flags[ITEM]) {
      int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
      int hash = HASH(key, ht_len, keys_min);
      uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
      if (slot != 0) {
        res[ITEM] = slot;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGroupByGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
        int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
        int hash = HASH(key, ht_len, keys_min);
        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
        if (slot != 0) {
          res[ITEM] = slot;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeGroupByGPU2(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeGroupByGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, res, selection_flags, gpuCache, key_idx, ht, ht_len, keys_min);
  } else {
    BlockProbeGroupByGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, res, selection_flags, gpuCache, key_idx, ht, ht_len, keys_min, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPassThroughOffsetDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD]
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (selection_flags[ITEM]) {
      offset[ITEM] = items[ITEM];
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPassThroughOffsetDirect(
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
__device__ __forceinline__ void BlockPassThroughOffset(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int  (&offset)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPassThroughOffsetDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, offset, selection_flags);
  } else {
    BlockPassThroughOffsetDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, offset, selection_flags, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetFilteredValueDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value,
    int  (&selection_flags)[ITEMS_PER_THREAD]
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (selection_flags[ITEM]) {
      items[ITEM] = value;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetFilteredValueDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value,
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) { // CUB use blocked arrangement
      if (selection_flags[ITEM]) {
        items[ITEM] = value;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetFilteredValue(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value,
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockSetFilteredValueDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, value, selection_flags);
  } else {
    BlockSetFilteredValueDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, value, selection_flags, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetValueDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    items[ITEM] = value;
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetValueDirect(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) { // CUB use blocked arrangement
      items[ITEM] = value;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockSetValue(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockSetValueDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, value);
  } else {
    BlockSetValueDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, value, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadFilteredOffsetDirect(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* col_idx
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (selection_flags[ITEM]) {
      items[ITEM] = gpuCache[col_idx[items_off[ITEM] / SEGMENT_SIZE] * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadFilteredOffsetDirect(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* col_idx,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        items[ITEM] = gpuCache[col_idx[items_off[ITEM] / SEGMENT_SIZE] * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
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

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockReadFilteredOffsetDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, items, selection_flags, gpuCache, col_idx);
  } else {
    BlockReadFilteredOffsetDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, items, selection_flags, gpuCache, col_idx, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadOffsetDirect(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&items)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* col_idx
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    items[ITEM] = gpuCache[col_idx[items_off[ITEM] / SEGMENT_SIZE] * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadOffsetDirect(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&items)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* col_idx,
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = gpuCache[col_idx[items_off[ITEM] / SEGMENT_SIZE] * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockReadOffsetGPU(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD],
    int  (&items)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* col_idx,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockReadOffsetDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, items, gpuCache, col_idx);
  } else {
    BlockReadOffsetDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, items, gpuCache, col_idx, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPUDirect(
    int tid,
    int blockid,
    int start_offset,
    int  (&keys)[ITEMS_PER_THREAD], //equal to items
    int  (&vals)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min
    ) {

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);
      atomicCAS(&ht[hash << 1], 0, vals[ITEM]);
      ht[(hash << 1) + 1] = start_offset + (blockid * tile_size) + (tid + ITEM * BLOCK_THREADS) + 1;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPUDirect(
    int tid,
    int blockid,
    int start_offset,
    int  (&keys)[ITEMS_PER_THREAD], //equal to items
    int  (&vals)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {  // use stripe arrangement since we are using Crystal Blockload instead of CUB
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);
        atomicCAS(&ht[hash << 1], 0, vals[ITEM]);
        ht[(hash << 1) + 1] = start_offset + (blockid * tile_size) + (tid + ITEM * BLOCK_THREADS) + 1;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPU(
    int tid,
    int blockid,
    int start_offset,
    int  (&keys)[ITEMS_PER_THREAD], //equal to items
    int  (&vals)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildValueGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, blockid, start_offset, keys, vals, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildValueGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, blockid, start_offset, keys, vals, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPUDirect(
    int tid,
    int blockid,
    int start_offset,
    int  (&keys)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* ht,
    int ht_len,
    int keys_min
    ) {

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);
      ht[(hash << 1) + 1] = start_offset + (blockid * tile_size) + (tid + ITEM * BLOCK_THREADS) + 1;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPUDirect(
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

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {  // use stripe arrangement since we are using Crystal Blockload instead of CUB
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);
        ht[(hash << 1) + 1] = start_offset + (blockid * tile_size) + (tid + ITEM * BLOCK_THREADS) + 1;
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

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildOffsetGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, blockid, start_offset, keys, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildOffsetGPUDirect<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, blockid, start_offset, keys, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* val_idx,
    int* ht,
    int ht_len,
    int keys_min
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (selection_flags[ITEM]) {
      int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
      int dimval_seg = val_idx[items_off[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
      int val = gpuCache[dimval_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];

      // Out-of-bounds items are selection_flags
      int hash = HASH(key, ht_len, keys_min);
      atomicCAS(&ht[hash << 1], 0, val);
      ht[(hash << 1) + 1] = items_off[ITEM] + 1;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* val_idx,
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
        int dimval_seg = val_idx[items_off[ITEM] / SEGMENT_SIZE];
        int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
        int val = gpuCache[dimval_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];

        // Out-of-bounds items are selection_flags
        int hash = HASH(key, ht_len, keys_min);
        atomicCAS(&ht[hash << 1], 0, val);
        ht[(hash << 1) + 1] = items_off[ITEM] + 1;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildValueGPU2(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* val_idx,
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildValueGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, selection_flags, gpuCache, key_idx, val_idx, ht, ht_len, keys_min);
  } else {
    BlockBuildValueGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, selection_flags, gpuCache, key_idx, val_idx, ht, ht_len, keys_min, num_items);
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (selection_flags[ITEM]) {
      int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
      int hash = HASH(key, ht_len, keys_min);
      ht[(hash << 1) + 1] = items_off[ITEM] + 1;
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPU2Direct(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
        int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];
        int hash = HASH(key, ht_len, keys_min);
        ht[(hash << 1) + 1] = items_off[ITEM] + 1;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildOffsetGPU2(
    int tid,
    int  (&items_off)[ITEMS_PER_THREAD], //equal to items
    int  (&selection_flags)[ITEMS_PER_THREAD],
    int* gpuCache,
    int* key_idx,
    int* ht,
    int ht_len,
    int keys_min, // equal to val_min
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildOffsetGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, selection_flags, gpuCache, key_idx, ht, ht_len, keys_min);
  } else {
    BlockBuildOffsetGPU2Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items_off, selection_flags, gpuCache, key_idx, ht, ht_len, keys_min, num_items);
  }
}

#endif