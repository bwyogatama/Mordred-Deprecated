#pragma once

#define cudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }

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

  cudaAssert(ht != NULL);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];

      int hash = HASH(key, ht_len, keys_min);
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

  cudaAssert(ht != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
      int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];

      int hash = HASH(key, ht_len, keys_min);
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
__device__ __forceinline__ void BlockSetFilteredValue(
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
__device__ __forceinline__ void BlockSetValue(
    int tid,
    int  (&items)[ITEMS_PER_THREAD],
    int value,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = value;
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
  cudaAssert(ht != NULL);
  cudaAssert(key_idx != NULL);
  cudaAssert(val_idx != NULL);

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
        atomicCAS(&ht[hash << 1], 0, key);
        ht[(hash << 1) + 1] = val;
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
  cudaAssert(ht != NULL);
  cudaAssert(key_idx != NULL);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int dimkey_seg = key_idx[items_off[ITEM] / SEGMENT_SIZE];
        int key = gpuCache[dimkey_seg * SEGMENT_SIZE + (items_off[ITEM] % SEGMENT_SIZE)];

        // Out-of-bounds items are selection_flags
        int hash = HASH(key, ht_len, keys_min);
        atomicCAS(&ht[hash << 1], 0, key);
        ht[(hash << 1) + 1] = items_off[ITEM] + 1;
      }
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