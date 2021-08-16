#ifndef _CPU_PROCESSING_H_
#define _CPU_PROCESSING_H_

#include <chrono>
#include <atomic>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "tbb/tbb.h"
#include "KernelArgs.h"

using namespace std;
using namespace tbb;

#define BATCH_SIZE 256
#define NUM_THREADS 48

void filter_probe_CPU(
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, int num_tuples,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int* total, int start_offset = 0, short* segment_group = NULL) {


  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);


  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) { //make sure the grainsize is not too big (will result in segfault on temp[5][end-start] -> not enough stack memory)
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int temp[5][end-start];
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned int count = 0;

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        // int slot1 = 1, slot2 = 1, slot3 = 1; 
        int slot4 = 1;
        int lo_offset;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;

        lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        // if (filter_col1 != NULL) {
          if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
        // }

        // if (filter_col2 != NULL) {
          if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
        // }

        // if (ht1 != NULL && key_col1 != NULL) {
        //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        //   slot = reinterpret_cast<long long*>(ht1)[hash];
        //   if (slot == 0) continue;
        //   slot1 = slot >> 32;
        // }


        // if (ht2 != NULL && key_col2 != NULL) {
        //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        //   slot = reinterpret_cast<long long*>(ht2)[hash];
        //   if (slot == 0) continue;
        //   slot2 = slot >> 32;
        // }


        // if (ht3 != NULL && key_col3 != NULL) {
        //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        //   slot = reinterpret_cast<long long*>(ht3)[hash];
        //   if (slot == 0) continue;
        //   slot3 = slot >> 32;
        // }


        // if (ht4 != NULL && key_col4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          slot4 = slot >> 32;
        // }


        temp[0][count] = lo_offset;
        // temp[1][count] = slot1-1;
        // temp[2][count] = slot2-1;
        // temp[3][count] = slot3-1;
        temp[4][count] = slot4-1;
        count++;

      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      // int slot1 = 1, slot2 = 1, slot3 = 1;
      int slot4 = 1;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      // if (filter_col1 != NULL) {
        if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
      // }

      // if (filter_col2 != NULL) {
        if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
      // }

      // if (ht1 != NULL && key_col1 != NULL) {
      //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
      //   slot = reinterpret_cast<long long*>(ht1)[hash];
      //   if (slot == 0) continue;
      //   slot1 = slot >> 32;
      // }


      // if (ht2 != NULL && key_col2 != NULL) {
      //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
      //   slot = reinterpret_cast<long long*>(ht2)[hash];
      //   if (slot == 0) continue;
      //   slot2 = slot >> 32;
      // }


      // if (ht3 != NULL && key_col3 != NULL) {
      //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
      //   slot = reinterpret_cast<long long*>(ht3)[hash];
      //   if (slot == 0) continue;
      //   slot3 = slot >> 32;
      // }


      // if (ht4 != NULL && key_col4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        slot4 = slot >> 32;
      // }


      temp[0][count] = lo_offset;
      // temp[1][count] = slot1-1;
      // temp[2][count] = slot2-1;
      // temp[3][count] = slot3-1;
      temp[4][count] = slot4-1;
      count++;
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      assert(h_lo_off != NULL);
      if (h_lo_off != NULL) h_lo_off[thread_off+i] = temp[0][i];
      // if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      // if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      // if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  }, simple_partitioner());
}

void filter_probe_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, int num_tuples,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int* total, int start_offset = 0) {


  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(h_lo_off != NULL);
  assert(lo_off != NULL);


  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) { //make sure the grainsize is not too big (will result in segfault on temp[5][end-start] -> not enough stack memory)
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int temp[5][end-start];
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned int count = 0;
    
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        // int slot1 = 1, slot2 = 1, slot3 = 1;
        int slot4 = 1;
        int lo_offset;

        lo_offset = lo_off[start_offset + i];

        // if (filter_col1 != NULL) {
        //   if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
        // }

        // if (filter_col2 != NULL) {
          if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
        // }

        // if (ht1 != NULL && key_col1 != NULL) {
        //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        //   slot = reinterpret_cast<long long*>(ht1)[hash];
        //   if (slot == 0) continue;
        //   slot1 = slot >> 32;
        // } else if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;


        // if (ht2 != NULL && key_col2 != NULL) {
        //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        //   slot = reinterpret_cast<long long*>(ht2)[hash];
        //   if (slot == 0) continue;
        //   slot2 = slot >> 32;
        // } else if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;


        // if (ht3 != NULL && key_col3 != NULL) {
        //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        //   slot = reinterpret_cast<long long*>(ht3)[hash];
        //   if (slot == 0) continue;
        //   slot3 = slot >> 32;
        // } else if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;


        // if (ht4 != NULL && key_col4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          slot4 = slot >> 32;
        // } else if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;


        temp[0][count] = lo_offset;
        // temp[1][count] = slot1-1;
        // temp[2][count] = slot2-1;
        // temp[3][count] = slot3-1;
        temp[4][count] = slot4-1;
        count++;

      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      // int slot1 = 1, slot2 = 1, slot3 = 1;
      int slot4 = 1;
      int lo_offset;

      lo_offset = lo_off[start_offset + i];

      // if (filter_col1 != NULL) {
      //   if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
      // }

      // if (filter_col2 != NULL) {
        if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
      // }

      // if (ht1 != NULL && key_col1 != NULL) {
      //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
      //   slot = reinterpret_cast<long long*>(ht1)[hash];
      //   if (slot == 0) continue;
      //   slot1 = slot >> 32;
      // } else if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;


      // if (ht2 != NULL && key_col2 != NULL) {
      //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
      //   slot = reinterpret_cast<long long*>(ht2)[hash];
      //   if (slot == 0) continue;
      //   slot2 = slot >> 32;
      // } else if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;


      // if (ht3 != NULL && key_col3 != NULL) {
      //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
      //   slot = reinterpret_cast<long long*>(ht3)[hash];
      //   if (slot == 0) continue;
      //   slot3 = slot >> 32;
      // } else if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;


      // if (ht4 != NULL && key_col4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        slot4 = slot >> 32;
      // } else if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;  


      temp[0][count] = lo_offset;
      // temp[1][count] = slot1-1;
      // temp[2][count] = slot2-1;
      // temp[3][count] = slot3-1;
      temp[4][count] = slot4-1;
      count++;
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      assert(h_lo_off != NULL);
      if (h_lo_off != NULL) h_lo_off[thread_off+i] = temp[0][i];
      // if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      // if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      // if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  }, simple_partitioner());
}

void probe_CPU(
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, int num_tuples,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int* total, int start_offset = 0, short* segment_group = NULL) {


  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);
  assert(h_lo_off != NULL);


  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) { //make sure the grainsize is not too big (will result in segfault on temp[5][end-start] -> not enough stack memory)
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    //printf("worker_index = %d %d\n", worker_index, end-start);
    unsigned int temp[5][end-start];
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned int count = 0;


    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;
    
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
      int hash;
      long long slot;
      int slot1 = 1, slot2 = 1, slot3 = 1, slot4 = 1;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      if (ht1 != NULL && key_col1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        slot1 = slot >> 32;
      }

      if (ht2 != NULL && key_col2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        slot2 = slot >> 32;
      }

      if (ht3 != NULL && key_col3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        slot3 = slot >> 32;
      }

      if (ht4 != NULL && key_col4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        slot4 = slot >> 32;
      }

      temp[0][count] = lo_offset;
      temp[1][count] = slot1-1;
      temp[2][count] = slot2-1;
      temp[3][count] = slot3-1;
      temp[4][count] = slot4-1;
      count++;

      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      int slot1 = 1, slot2 = 1, slot3 = 1, slot4 = 1;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      if (ht1 != NULL && key_col1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        slot1 = slot >> 32;
      }

      if (ht2 != NULL && key_col2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        slot2 = slot >> 32;
      }

      if (ht3 != NULL && key_col3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        slot3 = slot >> 32;
      }

      if (ht4 != NULL && key_col4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        slot4 = slot >> 32;
      }

      temp[0][count] = lo_offset;
      temp[1][count] = slot1-1;
      temp[2][count] = slot2-1;
      temp[3][count] = slot3-1;
      temp[4][count] = slot4-1;
      count++;
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      assert(h_lo_off != NULL);
      if (h_lo_off != NULL) h_lo_off[thread_off+i] = temp[0][i];
      if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  }, simple_partitioner());
}

void probe_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, int num_tuples,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int* total, int start_offset = 0) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(h_lo_off != NULL);
  assert(lo_off != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) { //make sure the grainsize is not too big (will result in segfault on temp[5][end-start] -> not enough stack memory)
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    //printf("worker_index = %d %d\n", worker_index, end-start);
    unsigned int temp[5][end-start];
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    unsigned int count = 0;
    
    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int slot1 = 1, slot2 = 1, slot3 = 1, slot4 = 1;
        int lo_offset;

        lo_offset = lo_off[start_offset + i];

        if (ht1 != NULL && key_col1 != NULL) {
          hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          if (slot == 0) continue;
          slot1 = slot >> 32;
        } else if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;


        if (ht2 != NULL && key_col2 != NULL) {
          hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          if (slot == 0) continue;
          slot2 = slot >> 32;
        } else if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;


        if (ht3 != NULL && key_col3 != NULL) {
          hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
          slot = reinterpret_cast<long long*>(ht3)[hash];
          if (slot == 0) continue;
          slot3 = slot >> 32;
        } else if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;


        if (ht4 != NULL && key_col4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          slot4 = slot >> 32;
        } else if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;


        temp[0][count] = lo_offset;
        temp[1][count] = slot1-1;
        temp[2][count] = slot2-1;
        temp[3][count] = slot3-1;
        temp[4][count] = slot4-1;
        count++;


      }
    }

    for (int i = end_batch ; i < end; i++) {
      int hash;
      long long slot;
      int slot1 = 1, slot2 = 1, slot3 = 1, slot4 = 1;
      int lo_offset;

      lo_offset = lo_off[start_offset + i];

      if (ht1 != NULL && key_col1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        slot1 = slot >> 32;
      } else if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;


      if (ht2 != NULL && key_col2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        slot2 = slot >> 32;
      } else if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;


      if (ht3 != NULL && key_col3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        slot3 = slot >> 32;
      } else if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;


      if (ht4 != NULL && key_col4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        slot4 = slot >> 32;
      } else if (dim_off4 != NULL) slot4 = dim_off4[start_offset + i] + 1;


      temp[0][count] = lo_offset;
      temp[1][count] = slot1-1;
      temp[2][count] = slot2-1;
      temp[3][count] = slot3-1;
      temp[4][count] = slot4-1;
      count++;
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      assert(h_lo_off != NULL);
      if (h_lo_off != NULL) h_lo_off[thread_off+i] = temp[0][i];
      if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  }, simple_partitioner());
}


void probe_group_by_CPU(
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4, 
  int total_val, int* res, int start_offset = 0, short* segment_group = NULL) {


  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);


  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
        int lo_offset;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;

        lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        if (key_col1 != NULL && ht1 != NULL) {
          hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          if (slot == 0) continue;
          dim_val1 = slot;
        }

        if (key_col2 != NULL && ht2 != NULL) {
          hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          if (slot == 0) continue;
          dim_val2 = slot;
        }

        if (key_col3 != NULL && ht3 != NULL) {
          hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
          slot = reinterpret_cast<long long*>(ht3)[hash];
          if (slot == 0) continue;
          dim_val3 = slot;
        }

        if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          dim_val4 = slot;
        }

        hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
        if (dim_val1 != 0) res[hash * 6] = dim_val1;
        if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
        if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
        if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

        int temp;
        if (mode == 0) {
          assert(aggr_col1 != NULL);
          temp = aggr_col1[lo_offset];
        } else if (mode == 1) {
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
        } else if  (mode == 2){ 
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
        } else assert(0);

        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      if (key_col1 != NULL && ht1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        dim_val1 = slot;
      }

      if (key_col2 != NULL && ht2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        dim_val2 = slot;
      }

      if (key_col3 != NULL && ht3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        dim_val3 = slot;
      }

      if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        dim_val4 = slot;
      }

      hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
      if (dim_val1 != 0) res[hash * 6] = dim_val1;
      if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
      if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
      if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

      int temp;
      if (mode == 0) {
        assert(aggr_col1 != NULL);
        temp = aggr_col1[lo_offset];
      } else if (mode == 1) {
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
      } else if  (mode == 2){ 
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
      } else assert(0);

      __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
    }
  }, simple_partitioner());

}

void probe_group_by_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int* group_col1, int* group_col2, int* group_col3, int* group_col4, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4, 
  int total_val, int* res, int start_offset = 0) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(lo_off != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
        int lo_offset;

        lo_offset = lo_off[start_offset + i];

        if (key_col1 != NULL && ht1 != NULL) {
          hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          if (slot == 0) continue;
          dim_val1 = slot;
        } else if (group_col1 != NULL) {
          assert(dim_off1 != NULL);
          dim_val1 = group_col1[dim_off1[start_offset + i]];
        }

        if (key_col2 != NULL && ht2 != NULL) {
          hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          if (slot == 0) continue;
          dim_val2 = slot;
        } else if (group_col2 != NULL) {
          assert(dim_off2 != NULL);
          dim_val2 = group_col2[dim_off2[start_offset + i]];
        }

        if (key_col3 != NULL && ht3 != NULL) {
          hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
          slot = reinterpret_cast<long long*>(ht3)[hash];
          if (slot == 0) continue;
          dim_val3 = slot;
        } else if (group_col3 != NULL) {
          assert(dim_off3 != NULL);
          dim_val3 = group_col3[dim_off3[start_offset + i]];
        }

        if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          dim_val4 = slot;
        } else if (group_col4 != NULL) {
          assert(dim_off4 != NULL);
          dim_val4 = group_col4[dim_off4[start_offset + i]];
        }

        hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
        if (dim_val1 != 0) res[hash * 6] = dim_val1;
        if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
        if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
        if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

        int temp;
        if (mode == 0) {
          assert(aggr_col1 != NULL);
          temp = aggr_col1[lo_offset];
        } else if (mode == 1) {
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
        } else if  (mode == 2){ 
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
        } else assert(0);

        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
      int lo_offset;

      lo_offset = lo_off[start_offset + i];

      if (key_col1 != NULL && ht1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        dim_val1 = slot;
      } else if (group_col1 != NULL) {
        assert(dim_off1 != NULL);
        dim_val1 = group_col1[dim_off1[start_offset + i]];
      }

      if (key_col2 != NULL && ht2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        dim_val2 = slot;
      } else if (group_col2 != NULL) {
        assert(dim_off2 != NULL);
        dim_val2 = group_col2[dim_off2[start_offset + i]];
      }

      if (key_col3 != NULL && ht3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        dim_val3 = slot;
      } else if (group_col3 != NULL) {
        assert(dim_off3 != NULL);
        dim_val3 = group_col3[dim_off3[start_offset + i]];
      }

      if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        dim_val4 = slot;
      } else if (group_col4 != NULL) {
        assert(dim_off4 != NULL);
        dim_val4 = group_col4[dim_off4[start_offset + i]];
      }

      hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
      if (dim_val1 != 0) res[hash * 6] = dim_val1;
      if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
      if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
      if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

      int temp;
      if (mode == 0) {
        assert(aggr_col1 != NULL);
        temp = aggr_col1[lo_offset];
      } else if (mode == 1) {
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
      } else if  (mode == 2){ 
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
      } else assert(0);

      __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
    }
  }, simple_partitioner());

}

void filter_probe_group_by_CPU(
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4, 
  int total_val, int* res, int start_offset = 0, short* segment_group = NULL) {


  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
        int lo_offset;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;

        lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        if (filter_col1 != NULL) {
          if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
        }

        if (filter_col2 != NULL) {
          if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
        }

        if (key_col1 != NULL && ht1 != NULL) {
          hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          if (slot == 0) continue;
          dim_val1 = slot;
        }

        if (key_col2 != NULL && ht2 != NULL) {
          hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          if (slot == 0) continue;
          dim_val2 = slot;
        }

        if (key_col3 != NULL && ht3 != NULL) {
          hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
          slot = reinterpret_cast<long long*>(ht3)[hash];
          if (slot == 0) continue;
          dim_val3 = slot;
        }

        if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          dim_val4 = slot;
        }

        // if (!(key_col4[lo_offset] > 19930000 && key_col4[lo_offset] < 19940000)) continue;

        hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
        if (dim_val1 != 0) res[hash * 6] = dim_val1;
        if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
        if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
        if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

        int temp;
        if (mode == 0) {
          assert(aggr_col1 != NULL);
          temp = aggr_col1[lo_offset];
        } else if (mode == 1) {
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
        } else if  (mode == 2){ 
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
        } else assert(0);

        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      if (filter_col1 != NULL) {
        if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
      }

      if (filter_col2 != NULL) {
        if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
      }

      if (key_col1 != NULL && ht1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        dim_val1 = slot;
      }

      if (key_col2 != NULL && ht2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        dim_val2 = slot;
      }

      if (key_col3 != NULL && ht3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        dim_val3 = slot;
      }

      if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        dim_val4 = slot;
      }

      // if (!(key_col4[lo_offset] > 19930000 && key_col4[lo_offset] < 19940000)) continue;

      hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
      if (dim_val1 != 0) res[hash * 6] = dim_val1;
      if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
      if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
      if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

      int temp;
      if (mode == 0) {
        assert(aggr_col1 != NULL);
        temp = aggr_col1[lo_offset];
      } else if (mode == 1) {
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
      } else if  (mode == 2){ 
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
      } else assert(0);

      __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
    }
  }, simple_partitioner());

}

void filter_probe_group_by_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int* group_col1, int* group_col2, int* group_col3, int* group_col4, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int min_val1, int min_val2, int min_val3, int min_val4,
  int unique_val1, int unique_val2, int unique_val3, int unique_val4, 
  int total_val, int* res, int start_offset = 0) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(lo_off != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
        int lo_offset;

        lo_offset = lo_off[start_offset + i];

        if (filter_col1 != NULL) {
          if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
        }

        if (filter_col2 != NULL) {
          if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
        }

        if (key_col1 != NULL && ht1 != NULL) {
          hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          if (slot == 0) continue;
          dim_val1 = slot;
        } else if (group_col1 != NULL) {
          assert(dim_off1 != NULL);
          dim_val1 = group_col1[dim_off1[start_offset + i]];
        }

        if (key_col2 != NULL && ht2 != NULL) {
          hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          if (slot == 0) continue;
          dim_val2 = slot;
        } else if (group_col2 != NULL) {
          assert(dim_off2 != NULL);
          dim_val2 = group_col2[dim_off2[start_offset + i]];
        }

        if (key_col3 != NULL && ht3 != NULL) {
          hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
          slot = reinterpret_cast<long long*>(ht3)[hash];
          if (slot == 0) continue;
          dim_val3 = slot;
        } else if (group_col3 != NULL) {
          assert(dim_off3 != NULL);
          dim_val3 = group_col3[dim_off3[start_offset + i]];
        }

        if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
          dim_val4 = slot;
        } else if (group_col4 != NULL) {
          assert(dim_off4 != NULL);
          dim_val4 = group_col4[dim_off4[start_offset + i]];
        }

        hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
        if (dim_val1 != 0) res[hash * 6] = dim_val1;
        if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
        if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
        if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

        int temp;
        if (mode == 0) {
          assert(aggr_col1 != NULL);
          temp = aggr_col1[lo_offset];
        } else if (mode == 1) {
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
        } else if  (mode == 2){ 
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
        } else assert(0);

        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
      int lo_offset;

      lo_offset = lo_off[start_offset + i];

      if (filter_col1 != NULL) {
        if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
      }

      if (filter_col2 != NULL) {
        if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
      }

      if (key_col1 != NULL && ht1 != NULL) {
        hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        if (slot == 0) continue;
        dim_val1 = slot;
      } else if (group_col1 != NULL) {
        assert(dim_off1 != NULL);
        dim_val1 = group_col1[dim_off1[start_offset + i]];
      }

      if (key_col2 != NULL && ht2 != NULL) {
        hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        slot = reinterpret_cast<long long*>(ht2)[hash];
        if (slot == 0) continue;
        dim_val2 = slot;
      } else if (group_col2 != NULL) {
        assert(dim_off2 != NULL);
        dim_val2 = group_col2[dim_off2[start_offset + i]];
      }

      if (key_col3 != NULL && ht3 != NULL) {
        hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        slot = reinterpret_cast<long long*>(ht3)[hash];
        if (slot == 0) continue;
        dim_val3 = slot;
      } else if (group_col3 != NULL) {
        assert(dim_off3 != NULL);
        dim_val3 = group_col3[dim_off3[start_offset + i]];
      }

      if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
        dim_val4 = slot;
      } else if (group_col4 != NULL) {
        assert(dim_off4 != NULL);
        dim_val4 = group_col4[dim_off4[start_offset + i]];
      }

      hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
      if (dim_val1 != 0) res[hash * 6] = dim_val1;
      if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
      if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
      if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

      int temp;
      if (mode == 0) {
        assert(aggr_col1 != NULL);
        temp = aggr_col1[lo_offset];
      } else if (mode == 1) {
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] - aggr_col2[lo_offset];
      } else if  (mode == 2){ 
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggr_col1[lo_offset] * aggr_col2[lo_offset];
      } else assert(0);

      __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
    }
  }, simple_partitioner());

}

void build_CPU(int* filter_col, int compare1, int compare2, int mode,
  int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, 
  int start_offset = 0, short* segment_group = NULL) {

  assert(dim_key != NULL);
  assert(hash_table != NULL);
  assert(segment_group != NULL);

  int grainsize = num_tuples/NUM_THREADS + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {

    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int table_offset;
        int flag = 1;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;

        table_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        if (filter_col != NULL) {
          // if (mode == 0)
          //   flag = (filter_col[table_offset] == compare1);
          // else 
          if (mode == 1)
            flag = (filter_col[table_offset] >= compare1 && filter_col[table_offset] <= compare2);
          else if (mode == 2)
            flag = (filter_col[table_offset] == compare1 || filter_col[table_offset] == compare2);
        }

        if (flag) {
          int key = dim_key[table_offset];
          int hash = HASH(key, num_slots, val_min);
          hash_table[(hash << 1) + 1] = table_offset + 1;
          if (dim_val != NULL) hash_table[hash << 1] = dim_val[table_offset];
        }

      }
    }

    for (int i = end_batch ; i < end; i++) {
      int table_offset;
      int flag = 1;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      table_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      if (filter_col != NULL) {
        // if (mode == 0)
        //   flag = (filter_col[table_offset] == compare1);
        // else 
        if (mode == 1)
          flag = (filter_col[table_offset] >= compare1 && filter_col[table_offset] <= compare2);
        else if (mode == 2)
          flag = (filter_col[table_offset] == compare1 || filter_col[table_offset] == compare2);
      }

      if (flag) {
        int key = dim_key[table_offset];
        int hash = HASH(key, num_slots, val_min);
        hash_table[(hash << 1) + 1] = table_offset + 1;
        if (dim_val != NULL) hash_table[hash << 1] = dim_val[table_offset];
      }
    }

  });
}

void build_CPU2(int *dim_off, int* filter_col, int compare1, int compare2, int mode,
  int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, 
  int start_offset = 0) {

  assert(dim_key != NULL);
  assert(hash_table != NULL);
  assert(dim_off != NULL);

  int grainsize = num_tuples/NUM_THREADS + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {

    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int table_offset;
        // int flag = 1;

        table_offset = dim_off[start_offset + i];

        // if (filter_col != NULL) {
          // if (mode == 0)
          //   flag = (filter_col[table_offset] == compare1);
          // else if (mode == 1)
            // flag = (filter_col[table_offset] >= compare1 && filter_col[table_offset] <= compare2);
          // else if (mode == 2)
          //   flag = (filter_col[table_offset] == compare1 || filter_col[table_offset] == compare2);
        // }

        // if (flag) {
          int key = dim_key[table_offset];
          int hash = HASH(key, num_slots, val_min);
          hash_table[(hash << 1) + 1] = table_offset + 1;
          if (dim_val != NULL) hash_table[hash << 1] = dim_val[table_offset];
        // }

      }
    }

    for (int i = end_batch ; i < end; i++) {
      int table_offset;
      // int flag = 1;

      table_offset = dim_off[start_offset + i];

      // if (filter_col != NULL) {
        // if (mode == 0)
        //   flag = (filter_col[table_offset] == compare1);
        // else if (mode == 1)
          // flag = (filter_col[table_offset] >= compare1 && filter_col[table_offset] <= compare2);
        // else if (mode == 2)
        //   flag = (filter_col[table_offset] == compare1 || filter_col[table_offset] == compare2);
      // }

      // if (flag) {
        int key = dim_key[table_offset];
        int hash = HASH(key, num_slots, val_min);
        hash_table[(hash << 1) + 1] = table_offset + 1;
        if (dim_val != NULL) hash_table[hash << 1] = dim_val[table_offset];
      // }
    }

  }, simple_partitioner());
}


void filter_CPU(int *filter_col1, int* filter_col2, 
  int compare1, int compare2, int compare3, int compare4, int mode1, int mode2,
  int* out_off, int* total, int num_tuples,
  int start_offset = 0, short* segment_group = NULL) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    int count = 0;
    int temp[end-start];

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        bool selection_flag = 1;
        int col_offset;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;   

        col_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        if (filter_col1 != NULL) {
          // if (mode1 == 0)
          //   selection_flag = (filter_col1[col_offset] == compare1);
          // else 
          if (mode1 == 1)
            selection_flag = (filter_col1[col_offset] >= compare1 && filter_col1[col_offset] <= compare2);
          else if (mode1 == 2)
            selection_flag = (filter_col1[col_offset] == compare1 || filter_col1[col_offset] == compare2);
        }

        if (filter_col2 != NULL) {
          // if (mode2 == 0)
          //   selection_flag = selection_flag && (filter_col2[col_offset] == compare3);
          // else 
          if (mode2 == 1)
            selection_flag = selection_flag && (filter_col2[col_offset] >= compare3 && filter_col2[col_offset] <= compare4);
          else if (mode2 == 2)
            selection_flag = selection_flag && (filter_col2[col_offset] == compare3 || filter_col2[col_offset] == compare4);
        }

        if (selection_flag) {
          temp[count] = col_offset;
          count++;        
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      bool selection_flag = 1;
      int col_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;   

      col_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      if (filter_col1 != NULL) {
        // if (mode1 == 0)
        //   selection_flag = (filter_col1[col_offset] == compare1);
        // else 
        if (mode1 == 1)
          selection_flag = (filter_col1[col_offset] >= compare1 && filter_col1[col_offset] <= compare2);
        else if (mode1 == 2)
          selection_flag = (filter_col1[col_offset] == compare1 || filter_col1[col_offset] == compare2);
      }

      if (filter_col2 != NULL) {
        // if (mode2 == 0)
        //   selection_flag = selection_flag && (filter_col2[col_offset] == compare3);
        // else 
        if (mode2 == 1)
          selection_flag = selection_flag && (filter_col2[col_offset] >= compare3 && filter_col2[col_offset] <= compare4);
        else if (mode2 == 2)
          selection_flag = selection_flag && (filter_col2[col_offset] == compare3 || filter_col2[col_offset] == compare4);
      }

      if (selection_flag) {
        temp[count] = col_offset;
        count++;
      }
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    assert(out_off != NULL);
    for (int i = 0; i < count; i++) {
      out_off[thread_off+i] = temp[i];
    }

  }, simple_partitioner());
}

void filter_CPU2(int* off_col, int *filter_col1, int* filter_col2, 
  int compare1, int compare2, int compare3, int compare4, int mode1, int mode2,
  int* out_off, int* total, int num_tuples,
  int start_offset = 0) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(off_col != NULL);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    int count = 0;
    int temp[end-start];

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        bool selection_flag = 1;
        int col_offset;

        col_offset = off_col[start_offset + i];

        // if (filter_col1 != NULL) {
        //   if (mode1 == 0)
        //     selection_flag = (filter_col1[col_offset] == compare1);
        //   else if (mode1 == 1)
        //     selection_flag = (filter_col1[col_offset] >= compare1 && filter_col1[col_offset] <= compare2);
        //   else if (mode1 == 2)
        //     selection_flag = (filter_col1[col_offset] == compare1 || filter_col1[col_offset] == compare2);
        // }

        // if (filter_col2 != NULL) {
          // if (mode2 == 0)
          //   selection_flag = selection_flag && (filter_col2[col_offset] == compare3);
          // else 
          if (mode2 == 1)
            selection_flag = selection_flag && (filter_col2[col_offset] >= compare3 && filter_col2[col_offset] <= compare4);
          else if (mode2 == 2)
            selection_flag = selection_flag && (filter_col2[col_offset] == compare3 || filter_col2[col_offset] == compare4);
        // }

        if (selection_flag) {
          temp[count] = col_offset;
          count++;        
        }
      }
    }

    for (int i = end_batch ; i < end; i++) {
      bool selection_flag = 1;
      int col_offset;

      col_offset = off_col[start_offset + i];

      // if (filter_col1 != NULL) {
      //   if (mode1 == 0)
      //     selection_flag = (filter_col1[col_offset] == compare1);
      //   else if (mode1 == 1)
      //     selection_flag = (filter_col1[col_offset] >= compare1 && filter_col1[col_offset] <= compare2);
      //   else if (mode1 == 2)
      //     selection_flag = (filter_col1[col_offset] == compare1 || filter_col1[col_offset] == compare2);
      // }

      // if (filter_col2 != NULL) {
        // if (mode2 == 0)
        //   selection_flag = selection_flag && (filter_col2[col_offset] == compare3);
        // else 
        if (mode2 == 1)
          selection_flag = selection_flag && (filter_col2[col_offset] >= compare3 && filter_col2[col_offset] <= compare4);
        else if (mode2 == 2)
          selection_flag = selection_flag && (filter_col2[col_offset] == compare3 || filter_col2[col_offset] == compare4);
      // }

      if (selection_flag) {
        temp[count] = col_offset;
        count++;
      }
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    assert(out_off != NULL);
    for (int i = 0; i < count; i++) {
      out_off[thread_off+i] = temp[i];
    }

  }, simple_partitioner());
}

void groupByCPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int* aggr_col1, int* aggr_col2, int* group_col1, int* group_col2, int* group_col3, int* group_col4,
  int min_val1, int min_val2, int min_val3, int min_val4, 
  int unique_val1, int unique_val2, int unique_val3, int unique_val4,
  int total_val, int num_tuples, int* res, int mode) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int groupval1 = 0, groupval2 = 0, groupval3 = 0, groupval4 = 0;
        int aggrval1, aggrval2;

        if (group_col1 != NULL) {
          assert(dim_off1 != NULL);
          groupval1 = group_col1[dim_off1[i]];
        }

        if (group_col2 != NULL) {
          assert(dim_off2 != NULL);
          groupval2 = group_col2[dim_off2[i]];
        }

        if (group_col3 != NULL) {
          assert(dim_off3 != NULL);
          groupval3 = group_col3[dim_off3[i]];
        }

        if (group_col4 != NULL) {
          assert(dim_off4 != NULL);
          groupval4 = group_col4[dim_off4[i]];
        }

        assert(lo_off != NULL);
        if (aggr_col1 != NULL) aggrval1 = aggr_col1[lo_off[i]];
        if (aggr_col2 != NULL) aggrval2 = aggr_col2[lo_off[i]];

        int hash = ((groupval1 - min_val1) * unique_val1 + (groupval2 - min_val2) * unique_val2 +  (groupval3 - min_val3) * unique_val3 + (groupval4 - min_val4) * unique_val4) % total_val;

        if (groupval1 != 0) res[hash * 6] = groupval1;
        if (groupval2 != 0) res[hash * 6 + 1] = groupval2;
        if (groupval3 != 0) res[hash * 6 + 2] = groupval3;
        if (groupval4 != 0) res[hash * 6 + 3] = groupval4;

        int temp;
        if (mode == 0) {
          assert(aggr_col1 != NULL);
          temp = aggrval1;
        } else if (mode == 1) {
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggrval1 - aggrval2;
        } else if  (mode == 2){ 
          assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          temp = aggrval1 * aggrval2;
        } else assert(0);

        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
      }
    }
    for (int i = end_batch ; i < end; i++) {
      int groupval1 = 0, groupval2 = 0, groupval3 = 0, groupval4 = 0;
      int aggrval1, aggrval2;

      if (group_col1 != NULL) {
        assert(dim_off1 != NULL);
        groupval1 = group_col1[dim_off1[i]];
      }

      if (group_col2 != NULL) {
        assert(dim_off2 != NULL);
        groupval2 = group_col2[dim_off2[i]];
      }

      if (group_col3 != NULL) {
        assert(dim_off3 != NULL);
        groupval3 = group_col3[dim_off3[i]];
      }

      if (group_col4 != NULL) {
        assert(dim_off4 != NULL);
        groupval4 = group_col4[dim_off4[i]];
      }

      assert(lo_off != NULL);
      if (aggr_col1 != NULL) aggrval1 = aggr_col1[lo_off[i]];
      if (aggr_col2 != NULL) aggrval2 = aggr_col2[lo_off[i]];

      int hash = ((groupval1 - min_val1) * unique_val1 + (groupval2 - min_val2) * unique_val2 +  (groupval3 - min_val3) * unique_val3 + (groupval4 - min_val4) * unique_val4) % total_val;

      if (groupval1 != 0) res[hash * 6] = groupval1;
      if (groupval2 != 0) res[hash * 6 + 1] = groupval2;
      if (groupval3 != 0) res[hash * 6 + 2] = groupval3;
      if (groupval4 != 0) res[hash * 6 + 3] = groupval4;

      int temp;
      if (mode == 0) {
        assert(aggr_col1 != NULL);
        temp = aggrval1;
      } else if (mode == 1) {
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggrval1 - aggrval2;
      } else if  (mode == 2){ 
        assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        temp = aggrval1 * aggrval2;
      } else assert(0);

      __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
    }
  });
}

void aggregationCPU(int* lo_off, 
  int* aggr_col1, int* aggr_col2, int num_tuples, int* res, int mode) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    long long local_sum = 0;

    assert(lo_off != NULL);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int aggrval1, aggrval2;

        if (aggr_col1 != NULL) aggrval1 = aggr_col1[lo_off[i]];
        if (aggr_col2 != NULL) aggrval2 = aggr_col2[lo_off[i]];

        // if (mode == 0) {
        //   assert(aggr_col1 != NULL);
        //   local_sum += aggrval1;
        // } else if (mode == 1) {
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        //   local_sum += aggrval1 - aggrval2;
        // } else if  (mode == 2){ 
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          local_sum += aggrval1 * aggrval2;
        // } else assert(0);

      }
    }
    for (int i = end_batch ; i < end; i++) {
      int aggrval1, aggrval2;

      assert(lo_off != NULL);
      if (aggr_col1 != NULL) aggrval1 = aggr_col1[lo_off[i]];
      if (aggr_col2 != NULL) aggrval2 = aggr_col2[lo_off[i]];

      // if (mode == 0) {
      //   assert(aggr_col1 != NULL);
      //   local_sum += aggrval1;
      // } else if (mode == 1) {
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
      //   local_sum += (aggrval1 - aggrval2);
      // } else if  (mode == 2){ 
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        local_sum += (aggrval1 * aggrval2);
      // } else assert(0);

    }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

  });
}


void probe_aggr_CPU(
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* res, int start_offset = 0, short* segment_group = NULL) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    long long local_sum = 0;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int lo_offset;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;

        lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        // if (key_col1 != NULL && ht1 != NULL) {
        //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        //   slot = reinterpret_cast<long long*>(ht1)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col2 != NULL && ht2 != NULL) {
        //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        //   slot = reinterpret_cast<long long*>(ht2)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col3 != NULL && ht3 != NULL) {
        //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        //   slot = reinterpret_cast<long long*>(ht3)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
        // }

        // if (mode == 0) {
        //   assert(aggr_col1 != NULL);
        //   local_sum += aggr_col1[lo_offset];
        // } else if (mode == 1) {
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
        // } else if  (mode == 2){ 
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
        // } else assert(0);
      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      // if (key_col1 != NULL && ht1 != NULL) {
      //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
      //   slot = reinterpret_cast<long long*>(ht1)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col2 != NULL && ht2 != NULL) {
      //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
      //   slot = reinterpret_cast<long long*>(ht2)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col3 != NULL && ht3 != NULL) {
      //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
      //   slot = reinterpret_cast<long long*>(ht3)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
      // }

      // if (mode == 0) {
      //   assert(aggr_col1 != NULL);
      //   local_sum += aggr_col1[lo_offset];
      // } else if (mode == 1) {
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
      //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
      // } else if  (mode == 2){ 
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
      // } else assert(0);
    }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

  }, simple_partitioner());

}

void probe_aggr_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* res, int start_offset = 0) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(lo_off != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    long long local_sum = 0;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int lo_offset;

        lo_offset = lo_off[start_offset + i];

        // if (key_col1 != NULL && ht1 != NULL) {
        //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        //   slot = reinterpret_cast<long long*>(ht1)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col2 != NULL && ht2 != NULL) {
        //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        //   slot = reinterpret_cast<long long*>(ht2)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col3 != NULL && ht3 != NULL) {
        //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        //   slot = reinterpret_cast<long long*>(ht3)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
        // }

        // if (mode == 0) {
        //   assert(aggr_col1 != NULL);
        //   local_sum += aggr_col1[lo_offset];
        // } else if (mode == 1) {
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
        // } else if  (mode == 2){ 
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
        // } else assert(0);

      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int lo_offset;

      lo_offset = lo_off[start_offset + i];

      // if (key_col1 != NULL && ht1 != NULL) {
      //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
      //   slot = reinterpret_cast<long long*>(ht1)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col2 != NULL && ht2 != NULL) {
      //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
      //   slot = reinterpret_cast<long long*>(ht2)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col3 != NULL && ht3 != NULL) {
      //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
      //   slot = reinterpret_cast<long long*>(ht3)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
      // }

      // if (mode == 0) {
      //   assert(aggr_col1 != NULL);
      //   local_sum += aggr_col1[lo_offset];
      // } else if (mode == 1) {
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
      //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
      // } else if  (mode == 2){ 
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
      // } else assert(0);
    }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

  }, simple_partitioner());

}

void filter_probe_aggr_CPU(
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4, 
  int* res, int start_offset = 0, short* segment_group = NULL) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(segment_group != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    long long local_sum = 0;

    int start_segment, start_group, segment_idx;
    int end_segment;
    start_segment = segment_group[start / SEGMENT_SIZE];
    end_segment = segment_group[end / SEGMENT_SIZE];
    start_group = start / SEGMENT_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int lo_offset;

        if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
        else segment_idx = end_segment;

        lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

        // lo_offset = start_offset + i;

        // bool selection_flag;
        // selection_flag = (key_col4[i] > 19930000 && key_col4[i] < 19940000);
        // selection_flag = selection_flag && (filter_col1[i] >= compare1 && filter_col1[i] <= compare2);
        // selection_flag = selection_flag && (filter_col2[i] >= compare3 && filter_col2[i] <= compare4);

        // if (selection_flag) local_sum += (aggr_col1[i] * aggr_col2[i]);

        // if (!(key_col4[lo_offset] > 19930000 && key_col4[lo_offset] < 19940000)) continue;

        // if (filter_col1 != NULL) {
          if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
        // }

        // if (filter_col2 != NULL) {
          if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
        // }

        // if (key_col1 != NULL && ht1 != NULL) {
        //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        //   slot = reinterpret_cast<long long*>(ht1)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col2 != NULL && ht2 != NULL) {
        //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        //   slot = reinterpret_cast<long long*>(ht2)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col3 != NULL && ht3 != NULL) {
        //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        //   slot = reinterpret_cast<long long*>(ht3)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
        // }

        // if (!(key_col4[lo_offset] > 19930000 && key_col4[lo_offset] < 19940000)) continue;

        // if (mode == 0) {
        //   assert(aggr_col1 != NULL);
        //   local_sum += aggr_col1[lo_offset];
        // } else if (mode == 1) {
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
        // } else if  (mode == 2){ 
          // assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
        // } else assert(0);

      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int lo_offset;

      if ((i / SEGMENT_SIZE) == start_group) segment_idx = start_segment;
      else segment_idx = end_segment;

      lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);

      // lo_offset = start_offset + i;

      // bool selection_flag;
      // selection_flag = (key_col4[i] > 19930000 && key_col4[i] < 19940000);
      // selection_flag = selection_flag && (filter_col1[i] >= compare1 && filter_col1[i] <= compare2);
      // selection_flag = selection_flag && (filter_col2[i] >= compare3 && filter_col2[i] <= compare4);

      // if (selection_flag) local_sum += (aggr_col1[i] * aggr_col2[i]);

      // if (!(key_col4[lo_offset] > 19930000 && key_col4[lo_offset] < 19940000)) continue;

      // if (filter_col1 != NULL) {
        if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
      // }

      // if (filter_col2 != NULL) {
        if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
      // }

      // if (key_col1 != NULL && ht1 != NULL) {
      //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
      //   slot = reinterpret_cast<long long*>(ht1)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col2 != NULL && ht2 != NULL) {
      //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
      //   slot = reinterpret_cast<long long*>(ht2)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col3 != NULL && ht3 != NULL) {
      //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
      //   slot = reinterpret_cast<long long*>(ht3)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
      // }

      // if (!(key_col4[lo_offset] > 19930000 && key_col4[lo_offset] < 19940000)) continue;

      // if (mode == 0) {
      //   assert(aggr_col1 != NULL);
      //   local_sum += aggr_col1[lo_offset];
      // } else if (mode == 1) {
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
      //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
      // } else if  (mode == 2){ 
        // assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
      // } else assert(0);

    }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

  }, simple_partitioner());

}

void filter_probe_aggr_CPU2(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* filter_col1, int* filter_col2, int compare1, int compare2, int compare3, int compare4,
  int* key_col1, int* key_col2, int* key_col3, int* key_col4, 
  int* aggr_col1, int* aggr_col2, int mode,
  int num_tuples, int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* res, int start_offset = 0) {

  int grainsize = num_tuples/4096 + 4;
  assert(grainsize < 20000);
  assert(grainsize < SEGMENT_SIZE);
  assert(lo_off != NULL);

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    //int worker_index = tbb::task_arena::current_thread_index();
    unsigned int start = range.begin();
    unsigned int end = range.end();
    unsigned int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    //printf("worker index = %d\n", worker_index);

    long long local_sum = 0;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        long long slot;
        int lo_offset;

        lo_offset = lo_off[start_offset + i];

        // if (filter_col1 != NULL) {
        //   if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
        // }

        // if (filter_col2 != NULL) {
          if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
        // }

        // if (key_col1 != NULL && ht1 != NULL) {
        //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
        //   slot = reinterpret_cast<long long*>(ht1)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col2 != NULL && ht2 != NULL) {
        //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
        //   slot = reinterpret_cast<long long*>(ht2)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col3 != NULL && ht3 != NULL) {
        //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
        //   slot = reinterpret_cast<long long*>(ht3)[hash];
        //   if (slot == 0) continue;
        // }

        // if (key_col4 != NULL && ht4 != NULL) {
          hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
          slot = reinterpret_cast<long long*>(ht4)[hash];
          if (slot == 0) continue;
        // }

        // if (mode == 0) {
        //   assert(aggr_col1 != NULL);
        //   local_sum += aggr_col1[lo_offset];
        // } else if (mode == 1) {
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
        // } else if  (mode == 2){ 
        //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
          local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
        // } else assert(0);
      }
    }

    for (int i = end_batch ; i < end; i++) {

      int hash;
      long long slot;
      int lo_offset;

      lo_offset = lo_off[start_offset + i];

      // if (filter_col1 != NULL) {
      //   if (!(filter_col1[lo_offset] >= compare1 && filter_col1[lo_offset] <= compare2)) continue; //only for Q1.x
      // }

      // if (filter_col2 != NULL) {
        if (!(filter_col2[lo_offset] >= compare3 && filter_col2[lo_offset] <= compare4)) continue; //only for Q1.x
      // }

      // if (key_col1 != NULL && ht1 != NULL) {
      //   hash = HASH(key_col1[lo_offset], dim_len1, min_key1);
      //   slot = reinterpret_cast<long long*>(ht1)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col2 != NULL && ht2 != NULL) {
      //   hash = HASH(key_col2[lo_offset], dim_len2, min_key2);
      //   slot = reinterpret_cast<long long*>(ht2)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col3 != NULL && ht3 != NULL) {
      //   hash = HASH(key_col3[lo_offset], dim_len3, min_key3);
      //   slot = reinterpret_cast<long long*>(ht3)[hash];
      //   if (slot == 0) continue;
      // }

      // if (key_col4 != NULL && ht4 != NULL) {
        hash = HASH(key_col4[lo_offset], dim_len4, min_key4);
        slot = reinterpret_cast<long long*>(ht4)[hash];
        if (slot == 0) continue;
      // }

      // if (mode == 0) {
      //   assert(aggr_col1 != NULL);
      //   local_sum += aggr_col1[lo_offset];
      // } else if (mode == 1) {
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
      //   local_sum += (aggr_col1[lo_offset] - aggr_col2[lo_offset]);
      // } else if  (mode == 2){ 
      //   assert(aggr_col1 != NULL); assert(aggr_col2 != NULL);
        local_sum += (aggr_col1[lo_offset] * aggr_col2[lo_offset]);
      // } else assert(0);
    }

     __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

  }, simple_partitioner());

}


void merge(int* resCPU, int* resGPU, int num_tuples) {

  int grainsize = num_tuples/NUM_THREADS + 4;
  // cout << num_tuples << endl;
  // assert(grainsize < 20000);
  // assert(grainsize < SEGMENT_SIZE);

  parallel_for(blocked_range<size_t>(0, num_tuples, grainsize), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        if (resCPU[i * 6] == 0) resCPU[i * 6] = resGPU[i * 6];
        if (resCPU[i * 6 + 1] == 0) resCPU[i * 6 + 1] = resGPU[i * 6 + 1];
        if (resCPU[i * 6 + 2] == 0) resCPU[i * 6 + 2] = resGPU[i * 6 + 2];
        if (resCPU[i * 6 + 3] == 0) resCPU[i * 6 + 3] = resGPU[i * 6 + 3];
        reinterpret_cast<unsigned long long*>(resCPU)[i * 3 + 2] += reinterpret_cast<unsigned long long*>(resGPU)[i * 3 + 2];
      }
    }
    for (int i = end_batch ; i < end; i++) {
      if (resCPU[i * 6] == 0) resCPU[i * 6] = resGPU[i * 6];
      if (resCPU[i * 6 + 1] == 0) resCPU[i * 6 + 1] = resGPU[i * 6 + 1];
      if (resCPU[i * 6 + 2] == 0) resCPU[i * 6 + 2] = resGPU[i * 6 + 2];
      if (resCPU[i * 6 + 3] == 0) resCPU[i * 6 + 3] = resGPU[i * 6 + 3];
      reinterpret_cast<unsigned long long*>(resCPU)[i * 3 + 2] += reinterpret_cast<unsigned long long*>(resGPU)[i * 3 + 2];
    }
  });
}

#endif