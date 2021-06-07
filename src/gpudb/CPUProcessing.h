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
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int num_tuples,
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4,
  int min_key1, int min_key2, int min_key3, int min_key4,
  int* h_lo_off, int* h_dim_off1, int* h_dim_off2, int* h_dim_off3, int* h_dim_off4,
  int start_offset, int* offset, int* segment_group) {

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
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
          assert(ht1 != NULL);
          slot1 = ht1[(hash << 1) + 1];
        } else {
          slot1 = 1;
          if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;
        }
        if (slot1 != 0) {
          if (dimkey_col2 != NULL) {
            hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
            assert(ht2 != NULL);
            slot2 = ht2[(hash << 1) + 1];
          } else {
            slot2 = 1;
            if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;
          }
          if (slot2 != 0) {
            if (dimkey_col3 != NULL) {
              hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
              assert(ht3 != NULL);
              slot3 = ht3[(hash << 1) + 1];
            } else {
              slot3 = 1;
              if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;
            }
            if (slot3 != 0) {
              if (dimkey_col4 != NULL) {
                hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
                assert(ht4 != NULL);
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
                //if (slot4 > 2556) printf("%d\n", slot4-1);
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
        assert(ht1 != NULL);
        slot1 = ht1[(hash << 1) + 1];
      } else {
        slot1 = 1;
        if (dim_off1 != NULL) slot1 = dim_off1[start_offset + i] + 1;
      }
      if (slot1 != 0) {
        if (dimkey_col2 != NULL) {
          hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
          assert(ht2 != NULL);
          slot2 = ht2[(hash << 1) + 1];
        } else {
          slot2 = 1;
          if (dim_off2 != NULL) slot2 = dim_off2[start_offset + i] + 1;
        }
        if (slot2 != 0) {
          if (dimkey_col3 != NULL) {
            hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
            assert(ht3 != NULL);
            slot3 = ht3[(hash << 1) + 1];
          } else {
            slot3 = 1;
            if (dim_off3 != NULL) slot3 = dim_off3[start_offset + i] + 1;
          }
          if (slot3 != 0) {
            if (dimkey_col4 != NULL) {
              hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
              assert(ht4 != NULL);
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
      assert(h_lo_off != NULL);
      h_lo_off[thread_off+i] = temp[0][i];
      if (h_dim_off1 != NULL) h_dim_off1[thread_off+i] = temp[1][i];
      if (h_dim_off2 != NULL) h_dim_off2[thread_off+i] = temp[2][i];
      if (h_dim_off3 != NULL) h_dim_off3[thread_off+i] = temp[3][i];
      if (h_dim_off4 != NULL) h_dim_off4[thread_off+i] = temp[4][i];
    }

  });
}

void probe_group_by_CPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4,
  int* dimkey_col1, int* dimkey_col2, int* dimkey_col3, int* dimkey_col4, int* aggr_col, int num_tuples, 
  int* ht1, int dim_len1, int* ht2, int dim_len2, int* ht3, int dim_len3, int* ht4, int dim_len4, int* res,
  int min_val1, int unique_val1, int min_val2, int unique_val2, int min_val3, int unique_val3, int min_val4, int unique_val4, 
  int total_val, int min_key1, int min_key2, int min_key3, int min_key4, int start_offset, int* segment_group) {

  // Probe
  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
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
          hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
          assert(ht1 != NULL);
          slot = reinterpret_cast<long long*>(ht1)[hash];
          dim_val1 = slot >> 32;
        } else if (dim_off1 != NULL && dimkey_col1 != NULL){
          slot = 1;
          dim_val1 = dimkey_col1[dim_off1[start_offset + i]];
        } else if (dim_off1 == NULL && dimkey_col1 == NULL) {
          slot = 1;
          dim_val1 = 0;
        } else {
          assert(0);
        }
        if (slot != 0) {
          if (dim_off2 == NULL && dimkey_col2 != NULL) {
            hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
            assert(ht2 != NULL);
            slot = reinterpret_cast<long long*>(ht2)[hash];
            dim_val2 = slot >> 32;
          } else if (dim_off2 != NULL && dimkey_col2 != NULL){
            slot = 1;
            dim_val2 = dimkey_col2[dim_off2[start_offset + i]];
          } else if (dim_off2 == NULL && dimkey_col2 == NULL) {
            slot = 1;
            dim_val2 = 0;
          } else {
            assert(0);
          }
          if (slot != 0) {
            if (dim_off3 == NULL && dimkey_col3 != NULL) {
              hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
              assert(ht3 != NULL);
              slot = reinterpret_cast<long long*>(ht3)[hash];
              dim_val3 = slot >> 32;
            } else if (dim_off3 != NULL && dimkey_col3 != NULL){
              slot = 1;
              dim_val3 = dimkey_col3[dim_off3[start_offset + i]];
            } else if (dim_off3 == NULL && dimkey_col3 == NULL) {
              slot = 1;
              dim_val3 = 0;
            } else {
              assert(0);
            }
            if (slot != 0) {
              if (dim_off4 == NULL && dimkey_col4 != NULL) {
                hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
                assert(ht4 != NULL);
                slot = reinterpret_cast<long long*>(ht4)[hash];
                dim_val4 = slot >> 32;
              } else if (dim_off4 != NULL && dimkey_col4 != NULL){
                slot = 1;
                dim_val4 = dimkey_col4[dim_off4[start_offset + i]];
              } else if (dim_off4 == NULL && dimkey_col4 == NULL) {
                slot = 1;
                dim_val4 = 0;
              } else {
                assert(0);
              }
              if (slot != 0) {
                hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
                res[hash * 6] = dim_val1;
                res[hash * 6 + 1] = dim_val2;
                res[hash * 6 + 2] = dim_val3;
                res[hash * 6 + 3] = dim_val4;
                assert(aggr_col != NULL);
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
        hash = HASH(dimkey_col1[lo_offset], dim_len1, min_key1);
        assert(ht1 != NULL);
        slot = reinterpret_cast<long long*>(ht1)[hash];
        dim_val1 = slot >> 32;
      } else if (dim_off1 != NULL && dimkey_col1 != NULL){
        slot = 1;
        dim_val1 = dimkey_col1[dim_off1[start_offset + i]];
      } else if (dim_off1 == NULL && dimkey_col1 == NULL) {
        slot = 1;
        dim_val1 = 0;
      } else {
        assert(0);
      }
      if (slot != 0) {
        if (dim_off2 == NULL && dimkey_col2 != NULL) {
          hash = HASH(dimkey_col2[lo_offset], dim_len2, min_key2);
          assert(ht2 != NULL);
          slot = reinterpret_cast<long long*>(ht2)[hash];
          dim_val2 = slot >> 32;
        } else if (dim_off2 != NULL && dimkey_col2 != NULL){
          slot = 1;
          dim_val2 = dimkey_col2[dim_off2[start_offset + i]];
        } else if (dim_off2 == NULL && dimkey_col2 == NULL) {
          slot = 1;
          dim_val2 = 0;
        } else {
          assert(0);
        }
        if (slot != 0) {
          if (dim_off3 == NULL && dimkey_col3 != NULL) {
            hash = HASH(dimkey_col3[lo_offset], dim_len3, min_key3);
            assert(ht3 != NULL);
            slot = reinterpret_cast<long long*>(ht3)[hash];
            dim_val3 = slot >> 32;
          } else if (dim_off3 != NULL && dimkey_col3 != NULL){
            slot = 1;
            dim_val3 = dimkey_col3[dim_off3[start_offset + i]];
          } else if (dim_off3 == NULL && dimkey_col3 == NULL){
            slot = 1;
            dim_val3 = 0;
          } else {
            assert(0);
          }
          if (slot != 0) {
            if (dim_off4 == NULL && dimkey_col4 != NULL) {
              hash = HASH(dimkey_col4[lo_offset], dim_len4, min_key4);
              assert(ht4 != NULL);
              slot = reinterpret_cast<long long*>(ht4)[hash];
              dim_val4 = slot >> 32;
            } else if (dim_off4 != NULL && dimkey_col4 != NULL){
              slot = 1;
              dim_val4 = dimkey_col4[dim_off4[start_offset + i]];
            } else if (dim_off4 == NULL && dimkey_col4 == NULL) {
              slot = 1;
              dim_val4 = 0;
            } else {
              assert(0);
            }
            if (slot != 0) {
              hash = ((dim_val1 - min_val1) * unique_val1 + (dim_val2 - min_val2) * unique_val2 +  (dim_val3 - min_val3) * unique_val3 + (dim_val4 - min_val4) * unique_val4) % total_val;
              res[hash * 6] = dim_val1;
              res[hash * 6 + 1] = dim_val2;
              res[hash * 6 + 2] = dim_val3;
              res[hash * 6 + 3] = dim_val4;
              assert(aggr_col != NULL);
              __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(aggr_col[lo_offset]), __ATOMIC_RELAXED);
            }
          }
        }
      }
    }
  });

}

void build_CPU(int *dim_off, int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min, int isoffset) {
  assert(dim_key != NULL);
  assert(hash_table != NULL);
  if (dim_off == NULL) {
    parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
      for (int i = range.begin(); i < range.end(); i++) {
          int key = dim_key[i];
          int hash = HASH(key, num_slots, val_min);
          hash_table[hash << 1] = key;
          if (isoffset == 1) hash_table[(hash << 1) + 1] = i + 1;
          else if (isoffset == 0) {
            assert(dim_val != NULL);
            hash_table[(hash << 1) + 1] = dim_val[i];
          } else hash_table[(hash << 1) + 1] = 0;
      }
    });   
  } else {
    parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
      for (int i = range.begin(); i < range.end(); i++) {
          int key = dim_key[dim_off[i]];
          int hash = HASH(key, num_slots, val_min);
          hash_table[hash << 1] = key;
          if (isoffset == 1) hash_table[(hash << 1) + 1] = dim_off[i] + 1;
          else if (isoffset == 0) {
            assert(dim_val != NULL);
            hash_table[(hash << 1) + 1] = dim_val[dim_off[i]];
          } else hash_table[(hash << 1) + 1] = 0;
      }
    });   
  }
}

void filter_CPU(int* off_col, int *filter_col1, int* filter_col2, int compare1, int compare2, 
  int compare3, int compare4, int* out_off, int* total, int start_offset, int num_tuples, int mode1, int mode2) {

  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int count = 0;
    int temp[end-start];

    for (int i = range.begin(); i < range.end(); i++) {
      bool selection_flag = 1;
      int col_offset;

      if (off_col != NULL) col_offset = off_col[start_offset + i];
      else col_offset = start_offset + i;

      if (filter_col1 != NULL) {
        if (mode1 == 0)
          selection_flag = (filter_col1[col_offset] == compare1);
        else if (mode1 == 1)
          selection_flag = (filter_col1[col_offset] >= compare1 && filter_col1[col_offset] <= compare2);
        else if (mode1 == 2)
          selection_flag = (filter_col1[col_offset] == compare1 || filter_col1[col_offset] == compare2);
        else if (mode1 == 3) {
          selection_flag = (filter_col1[col_offset] < compare1);
        }
      }

      if (filter_col2 != NULL) {
        if (mode2 == 0)
          selection_flag = selection_flag && (filter_col2[col_offset] == compare3);
        else if (mode2 == 1) {
          selection_flag = selection_flag && (filter_col2[col_offset] >= compare3 && filter_col2[col_offset] <= compare4);
        }
        else if (mode2 == 2)
          selection_flag = selection_flag && (filter_col2[col_offset] == compare3 || filter_col2[col_offset] == compare4);
        else if (mode2 == 3) {
          selection_flag = selection_flag && (filter_col2[col_offset] < compare3);
        }
      }

      if (selection_flag) {
        temp[count] = start_offset + i;
        count++;        
      }
    }

    int thread_off = __atomic_fetch_add(total, count, __ATOMIC_RELAXED);

    for (int i = 0; i < count; i++) {
      out_off[thread_off+i] = temp[i];
    }
  });
}

void groupByCPU(int* lo_off, int* dim_off1, int* dim_off2, int* dim_off3, int* dim_off4, 
  int* aggr_col1, int* aggr_col2, int* group_col1, int* group_col2, int* group_col3, int* group_col4,
  int min_val1, int min_val2, int min_val3, int min_val4, int unique_val1, int unique_val2, int unique_val3, int unique_val4,
  int total_val, int num_tuples, int* res, int mode) {

  parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
    int start = range.begin();
    int end = range.end();
    int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

    for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
      #pragma simd
      for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
        int groupval1, groupval2, groupval3, groupval4;
        int aggrval1, aggrval2;

        if (group_col1 != NULL) {
          assert(dim_off1 != NULL);
          groupval1 = group_col1[dim_off1[i]];
        } else groupval1 = 0;

        if (group_col2 != NULL) {
          assert(dim_off2 != NULL);
          groupval2 = group_col2[dim_off2[i]];
        } else groupval2 = 0;

        if (group_col3 != NULL) {
          assert(dim_off3 != NULL);
          groupval3 = group_col3[dim_off3[i]];
        } else groupval3 = 0;

        if (group_col4 != NULL) {
          assert(dim_off4 != NULL);
          groupval4 = group_col4[dim_off4[i]];
        } else groupval4= 0;

        assert(lo_off != NULL);
        if (aggr_col1 != NULL) aggrval1 = aggr_col1[lo_off[i]];
        if (aggr_col2 != NULL) aggrval2 = aggr_col2[lo_off[i]];

        int hash = ((groupval1 - min_val1) * unique_val1 + (groupval2 - min_val2) * unique_val2 +  (groupval3 - min_val3) * unique_val3 + (groupval4 - min_val4) * unique_val4) % total_val;

        res[hash * 6] = groupval1;
        res[hash * 6 + 1] = groupval2;
        res[hash * 6 + 2] = groupval3;
        res[hash * 6 + 3] = groupval4;

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
      int groupval1, groupval2, groupval3, groupval4;
      int aggrval1, aggrval2;

      if (group_col1 != NULL) {
        assert(dim_off1 != NULL);
        groupval1 = group_col1[dim_off1[i]];
      } else groupval1 = 0;

      if (group_col2 != NULL) {
        assert(dim_off2 != NULL);
        groupval2 = group_col2[dim_off2[i]];
      } else groupval2 = 0;

      if (group_col3 != NULL) {
        assert(dim_off3 != NULL);
        groupval3 = group_col3[dim_off3[i]];
      } else groupval3 = 0;

      if (group_col4 != NULL) {
        assert(dim_off4 != NULL);
        groupval4 = group_col4[dim_off4[i]];
      } else groupval4= 0;

      assert(lo_off != NULL);
      if (aggr_col1 != NULL) aggrval1 = aggr_col1[lo_off[i]];
      if (aggr_col2 != NULL) aggrval2 = aggr_col2[lo_off[i]];

      int hash = ((groupval1 - min_val1) * unique_val1 + (groupval2 - min_val2) * unique_val2 +  (groupval3 - min_val3) * unique_val3 + (groupval4 - min_val4) * unique_val4) % total_val;

      res[hash * 6] = groupval1;
      res[hash * 6 + 1] = groupval2;
      res[hash * 6 + 2] = groupval3;
      res[hash * 6 + 3] = groupval4;

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

#endif


// void runAggregationQ2CPU(int* lo_revenue, int* p_brand1, int* d_year, int* lo_off, int* part_off, int* date_off, int num_tuples, int* res, int num_slots) {
//   parallel_for(blocked_range<size_t>(0, num_tuples, num_tuples/NUM_THREADS + 4), [&](auto range) {
//     int start = range.begin();
//     int end = range.end();
//     int end_batch = start + ((end - start)/BATCH_SIZE) * BATCH_SIZE;

//     for (int batch_start = start; batch_start < end_batch; batch_start += BATCH_SIZE) {
//       #pragma simd
//       for (int i = batch_start; i < batch_start + BATCH_SIZE; i++) {
//         int brand = p_brand1[part_off[i]];
//         int year = d_year[date_off[i]];
//         int hash = (brand * 7 + (year - 1992)) % num_slots;
//         res[hash * 6 + 1] = brand;
//         res[hash * 6 + 2] = year;
//         __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[lo_off[i]]), __ATOMIC_RELAXED);
//       }
//     }
//     for (int i = end_batch ; i < end; i++) {
//         int brand = p_brand1[part_off[i]];
//         int year = d_year[date_off[i]];
//         int hash = (brand * 7 + (year - 1992)) % num_slots;
//         res[hash * 6 + 1] = brand;
//         res[hash * 6 + 2] = year;
//         __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(lo_revenue[lo_off[i]]), __ATOMIC_RELAXED);
//     }
//   });
// }