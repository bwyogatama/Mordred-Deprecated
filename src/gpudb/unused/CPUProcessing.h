#ifndef _CPU_PROCESSING_H_
#define _CPU_PROCESSING_H_

#include <chrono>
#include <atomic>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "tbb/tbb.h"

using namespace std;
using namespace tbb;

void probe_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int h_total, int start_offset, int* segment_group, int* offset) {

  // Probe
  parallel_for(blocked_range<size_t>(0, h_total, h_total/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;
    int count = 0;
    assert(h_lo_off != NULL);
    int temp[5][end-start];
    //printf("start = %d end = %d\n", start, end);

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int hash;
        int slot1, slot2, slot3, slot4;
        int lo_offset;
        if (lo_off != NULL) lo_offset = lo_off[start_offset + i];
        else {
          if (segment_group == NULL) lo_offset = start_offset + i;
          else {
            int idx = i / SEGMENT_SIZE;
            int segment_idx = segment_group[idx];
            lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);
          }
        }
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
      int lo_offset;
      if (lo_off != NULL) lo_offset = lo_off[start_offset + i];
      else {
        if (segment_group == NULL) lo_offset = start_offset + i;
        else {
          int idx = i / SEGMENT_SIZE;
          int segment_idx = segment_group[idx];
          lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);
        }
      }
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

void probe_group_by_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int fact_len, int start_offset, int* segment_group) {

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
        int lo_offset;
        if (lo_off != NULL) lo_offset = lo_off[start_offset + i];
        else {
          if (segment_group == NULL) lo_offset = start_offset + i;
          else {
            int idx = i / SEGMENT_SIZE;
            int segment_idx = segment_group[idx];
            lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);
          }
        }
        if (dim_off1 == NULL && dimkey_col1 != NULL) {
          hash = HASH_WM(dimkey_col1[lo_offset], dim_len1, min_key1);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          dim_val1 = slot >> 32;
        } else if (dim_off1 != NULL && dimkey_col1 != NULL){
          slot = 1;
          dim_val1 = dimkey_col1[dim_off1[start_offset + i]];
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
            dim_val2 = dimkey_col2[dim_off2[start_offset + i]];
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
              dim_val3 = dimkey_col3[dim_off3[start_offset + i]];
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
                dim_val4 = dimkey_col4[dim_off4[start_offset + i]];
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
      int lo_offset;
      if (lo_off != NULL) lo_offset = lo_off[start_offset + i];
      else {
        if (segment_group == NULL) lo_offset = start_offset + i;
        else {
          int idx = i / SEGMENT_SIZE;
          int segment_idx = segment_group[idx];
          lo_offset = segment_idx * SEGMENT_SIZE + (i % SEGMENT_SIZE);
        }
      }
      if (dim_off1 == NULL && dimkey_col1 != NULL) {
        hash = HASH_WM(dimkey_col1[lo_offset], dim_len1, min_key1);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        dim_val1 = slot >> 32;
      } else if (dim_off1 != NULL && dimkey_col1 != NULL){
        slot = 1;
        dim_val1 = dimkey_col1[dim_off1[start_offset + i]];
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
          dim_val2 = dimkey_col2[dim_off2[start_offset + i]];
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
            dim_val3 = dimkey_col3[dim_off3[start_offset + i]];
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
              dim_val4 = dimkey_col4[dim_off4[start_offset + i]];
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

#endif