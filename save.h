template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_group_by_3_GPU(int* dim_key1, int* dim_key2, int* dim_key3, int* aggr, 
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

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_2_GPU(int* dim_key1, int* dim_key2, 
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
          dim_offset1[ITEM] = slot - 1;
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
          dim_offset2[ITEM] = slot - 1;
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
            dim_offset1[ITEM] = slot - 1;
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
            dim_offset2[ITEM] = slot - 1;
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
        t_table[(offset * 4)] = start_offset + blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
        t_table[(offset * 4) + 1] = dim_offset1[ITEM];
        t_table[(offset * 4) + 2] = dim_offset2[ITEM];
        t_table[(offset * 4) + 3] = 0;
      }
    }
  }
}


void probe_2_CPU(int* h_t_table, int* dimkey_col1, int* ht1, int* h_t_table_res, int h_total, int dim_len1, int start_offset, int min_key1, int* offset) {

  // Probe
  parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    int count = 0;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot;
        int lo_offset = h_t_table[((start_offset + i) * 4)];
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = ht1[hash << 1];
        if (slot != 0) {
          count++;
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot;
      int lo_offset = h_t_table[((start_offset + i) * 4)];
      hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
      slot = ht1[hash << 1];
      if (slot != 0) {
        count++;
      }
    }
    //printf("count = %d\n", count);
    int thread_off = __atomic_fetch_add(offset, count, __ATOMIC_RELAXED);
    int j = 0;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot;
        int lo_offset = h_t_table[((start_offset + i) * 4)];
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = ht1[hash << 1];
        if (slot != 0) {
          int dim_offset1 = ht1[(hash << 1) + 1] - 1;
          int dim_offset2 = h_t_table[((start_offset + i) * 4) + 1];
          int dim_offset3 = h_t_table[((start_offset + i) * 4) + 2];
          int dim_offset4 = h_t_table[((start_offset + i) * 4) + 3];
          h_t_table_res[((thread_off+j) * 6)] = lo_offset;
          h_t_table_res[((thread_off+j) * 6) + 1] = dim_offset1;
          h_t_table_res[((thread_off+j) * 6) + 2] = dim_offset2;
          h_t_table_res[((thread_off+j) * 6) + 3] = dim_offset3;
          h_t_table_res[((thread_off+j) * 6) + 4] = dim_offset4;
          j++;
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      int slot;
      int lo_offset = h_t_table[((start_offset + i) * 4)];
      hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
      slot = ht1[hash << 1];
      if (slot != 0) {
        int dim_offset1 = ht1[(hash << 1) + 1] - 1;
        int dim_offset2 = h_t_table[((start_offset + i) * 4) + 1];
        int dim_offset3 = h_t_table[((start_offset + i) * 4) + 2];
        int dim_offset4 = h_t_table[((start_offset + i) * 4) + 3];
        h_t_table_res[((thread_off+j) * 6)] = lo_offset;
        h_t_table_res[((thread_off+j) * 6) + 1] = dim_offset1;
        h_t_table_res[((thread_off+j) * 6) + 2] = dim_offset2;
        h_t_table_res[((thread_off+j) * 6) + 3] = dim_offset3;
        h_t_table_res[((thread_off+j) * 6) + 4] = dim_offset4;
        j++;
        //printf("%d %d %d %d\n", lo_offset, dim_offset1, dim_offset2, dim_offset3);
      }
    }
  });
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int* lo_orderdate, int* lo_partkey, int* lo_suppkey, int lo_len,
    int* ht_s, int s_len,
    int* ht_p, int p_len,
    int* ht_d, int d_len,
    int* t_table,
    int* total, int start_offset) {

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
      int hash = HASH(items[ITEM], s_len, 0); // hash of lo_suppkey
      int slot = ht_s[hash << 1];
      if (slot != 0) {
        s_offset[ITEM] = ht_s[(hash << 1) + 1];
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
      int hash = HASH(items[ITEM], p_len, 0);
      if (selection_flags[ITEM]) {
        int slot = ht_p[hash << 1];
        if (slot != 0) {
          p_offset[ITEM] = ht_p[(hash << 1) + 1];
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
      int hash = HASH(items[ITEM], d_len, 19920101);
      if (selection_flags[ITEM]) {
        int slot = ht_d[hash << 1];
        if (slot != 0) {
          t_count++; // TODO: check this. count of items that have selection_flag = 1
          d_offset[ITEM] = ht_d[(hash << 1) + 1];
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
      int hash = HASH(items[ITEM], s_len, 0);
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int slot = ht_s[hash << 1];
        if (slot != 0) {
          s_offset[ITEM] = ht_s[(hash << 1) + 1];
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
          int hash = HASH(items[ITEM], p_len, 0);
          int slot = ht_p[hash << 1];
          if (slot != 0) {
            p_offset[ITEM] = ht_p[(hash << 1) + 1];
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
      int hash = HASH(items[ITEM], d_len, 19920101);

      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        if (selection_flags[ITEM]) {
          int slot = ht_d[hash << 1];
          if (slot != 0) {
            t_count++;
            d_offset[ITEM] = ht_d[(hash << 1) + 1];
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

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      if(selection_flags[ITEM]) {
        int offset = block_off + c_t_count++;
        t_table[offset << 2] = s_offset[ITEM];
        t_table[(offset << 2) + 1] = p_offset[ITEM];
        t_table[(offset << 2) + 2] = d_offset[ITEM];
        t_table[(offset << 2) + 3] = start_offset + blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
      }
    }
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_3_GPU(int* gpuCache, int idx_key1, int idx_key2, int idx_key3, 
  int fact_len, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3,
  int min_key1, int min_key2, int min_key3, int* t_table, int *total, int start_offset) {

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
  int* dim_key3 = gpuCache + idx_key3 * SEGMENT_SIZE;

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
          dim_offset1[ITEM] = slot - 1;
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
      int hash = HASH(items[ITEM], dim_len2, min_key2);
      if (selection_flags[ITEM]) {
        int slot = ht2[(hash << 1) + 1];
        if (slot != 0) {
          // printf("brand %d\n", items[ITEM]);
          // cudaDeviceSynchronize();
          dim_offset2[ITEM] = slot - 1;
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
      int hash = HASH(items[ITEM], dim_len3, min_key3); //19920101
      if (selection_flags[ITEM]) {
        int slot = ht3[(hash << 1) + 1];
        if (slot != 0) {
          // printf("item %d\n", items[ITEM]);
          // printf("ID %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM);
          // cudaDeviceSynchronize();
          t_count++;
          dim_offset3[ITEM] = slot - 1;
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
            dim_offset1[ITEM] = slot - 1;
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
        int hash = HASH(items[ITEM], dim_len2, min_key2);
        if (selection_flags[ITEM]) {
          int slot = ht2[(hash << 1) + 1];
          if (slot != 0) {
            // printf("brand %d\n", items[ITEM]);
            // cudaDeviceSynchronize();
            dim_offset2[ITEM] = slot - 1;
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
      if (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items) {
        int hash = HASH(items[ITEM], dim_len3, min_key3); //19920101
        if (selection_flags[ITEM]) {
          int slot = ht3[(hash << 1) + 1];
          if (slot != 0) {
            //printf("item %d\n", items[ITEM]);
            // printf("ID %d\n", blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM);
            // cudaDeviceSynchronize();
            t_count++;
            dim_offset3[ITEM] = slot - 1;
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
        int offset = block_off + c_t_count++;
        t_table[offset << 2] = dim_offset1[ITEM];
        t_table[(offset << 2) + 1] = dim_offset2[ITEM];
        t_table[(offset << 2) + 2] = dim_offset3[ITEM];
        t_table[(offset << 2) + 3] = start_offset + blockIdx.x * tile_size + threadIdx.x * ITEMS_PER_THREAD + ITEM;
      }
    }
  }
}

__global__
void build_offset_GPU(int *gpuCache, int idx_key, int segment_number, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
    int value = segment_number * SEGMENT_SIZE + offset + 1;
    int hash = HASH(key, num_slots, val_min);
    atomicCAS(&hash_table[hash << 1], 0, key);
    hash_table[(hash << 1) + 1] = value;
  }
}

__global__
void build_filter_offset_GPU(int *gpuCache, int idx_filter, int compare, int idx_key, int segment_number, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    if (gpuCache[idx_filter * SEGMENT_SIZE + offset] == compare) {
      int key = gpuCache[idx_key * SEGMENT_SIZE + offset];
      int value = segment_number * SEGMENT_SIZE + offset + 1;
      int hash = HASH(key, num_slots, val_min);
      atomicCAS(&hash_table[hash << 1], 0, key);
      hash_table[(hash << 1) + 1] = value;
    }
  }
}