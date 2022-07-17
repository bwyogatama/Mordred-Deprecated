#include "CPUProcessingHE.h"

void filter_probe_CPUHE(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset = 0, short* segment_group = NULL) {

          unsigned int count = 0;

          for (int i = 0; i < num_tuples; i++) {
              int hash;
              long long slot;
              int slot4 = 1;
              int lo_offset;

              lo_offset = start_offset + i;

              if (!(fargs.filter_col1[lo_offset] >= fargs.compare1 && fargs.filter_col1[lo_offset] <= fargs.compare2)) continue; //only for Q1.x

              if (!(fargs.filter_col2[lo_offset] >= fargs.compare3 && fargs.filter_col2[lo_offset] <= fargs.compare4)) continue; //only for Q1.x

              hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
              slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
              if (slot == 0) continue;
              slot4 = slot >> 32;

              if (out_off.h_lo_off != NULL) out_off.h_lo_off[count] = lo_offset;
              if (out_off.h_dim_off4 != NULL) out_off.h_dim_off4[count] = slot4-1;

              count++;

          }

          *total = count;
}

void filter_probe_CPU2HE(struct offsetCPU in_off, struct filterArgsCPU fargs, struct probeArgsCPU pargs,
  struct offsetCPU out_off, int num_tuples, int* total, int start_offset = 0) {

    assert(out_off.h_lo_off != NULL);
    assert(in_off.h_lo_off != NULL);

          unsigned int count = 0;
    
          for (int i = 0; i < num_tuples; i++) {
              int hash;
              long long slot;
              int slot4 = 1;
              int lo_offset;

              lo_offset = in_off.h_lo_off[start_offset + i];

              if (!(fargs.filter_col1[lo_offset] >= fargs.compare1 && fargs.filter_col1[lo_offset] <= fargs.compare2)) continue; //only for Q1.x

              if (!(fargs.filter_col2[lo_offset] >= fargs.compare3 && fargs.filter_col2[lo_offset] <= fargs.compare4)) continue; //only for Q1.x

              hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
              slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
              if (slot == 0) continue;
              slot4 = slot >> 32;

              if (out_off.h_lo_off != NULL) out_off.h_lo_off[count] = lo_offset;
              if (out_off.h_dim_off4 != NULL) out_off.h_dim_off4[count] = slot4-1;

              count++;

          }

          *total = count;
}


void probe_CPUHE(
  struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset = 0, short* segment_group = NULL) {

  assert(segment_group != NULL);
  assert(out_off.h_lo_off != NULL);

      unsigned int count = 0;
  
      for (int i = 0; i < num_tuples; i++) {
            int hash;
            long long slot;
            int slot1 = 1, slot2 = 1, slot3 = 1, slot4 = 1;
            int lo_offset;

            lo_offset = start_offset + i;

            if (pargs.ht1 != NULL && pargs.key_col1 != NULL) {
              hash = HASH(pargs.key_col1[lo_offset], pargs.dim_len1, pargs.min_key1);
              slot = reinterpret_cast<long long*>(pargs.ht1)[hash];
              if (slot == 0) continue;
              slot1 = slot >> 32;
            }

            if (pargs.ht2 != NULL && pargs.key_col2 != NULL) {
              hash = HASH(pargs.key_col2[lo_offset], pargs.dim_len2, pargs.min_key2);
              slot = reinterpret_cast<long long*>(pargs.ht2)[hash];
              if (slot == 0) continue;
              slot2 = slot >> 32;
            }

            if (pargs.ht3 != NULL && pargs.key_col3 != NULL) {
              hash = HASH(pargs.key_col3[lo_offset], pargs.dim_len3, pargs.min_key3);
              slot = reinterpret_cast<long long*>(pargs.ht3)[hash];
              if (slot == 0) continue;
              slot3 = slot >> 32;
            }

            if (pargs.ht4 != NULL && pargs.key_col4 != NULL) {
              hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
              slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
              if (slot == 0) continue;
              slot4 = slot >> 32;
            }

            if (out_off.h_lo_off != NULL) out_off.h_lo_off[count] = lo_offset;
            if (out_off.h_dim_off1 != NULL) out_off.h_dim_off1[count] = slot1-1;
            if (out_off.h_dim_off2 != NULL) out_off.h_dim_off2[count] = slot2-1;
            if (out_off.h_dim_off3 != NULL) out_off.h_dim_off3[count] = slot3-1;
            if (out_off.h_dim_off4 != NULL) out_off.h_dim_off4[count] = slot4-1;

            count++;

      }

      *total = count;

}

void probe_CPU2HE(struct offsetCPU in_off, struct probeArgsCPU pargs, struct offsetCPU out_off, int num_tuples,
  int* total, int start_offset = 0) {

  assert(in_off.h_lo_off != NULL);
  assert(out_off.h_lo_off != NULL);

    unsigned int count = 0;

    for (int i = 0; i < num_tuples; i++) {
          int hash;
          long long slot;
          int slot1 = 1, slot2 = 1, slot3 = 1, slot4 = 1;
          int lo_offset;

          lo_offset = in_off.h_lo_off[start_offset + i];

          if (pargs.ht1 != NULL && pargs.key_col1 != NULL) {
            hash = HASH(pargs.key_col1[lo_offset], pargs.dim_len1, pargs.min_key1);
            slot = reinterpret_cast<long long*>(pargs.ht1)[hash];
            if (slot == 0) continue;
            slot1 = slot >> 32;
          } else if (in_off.h_dim_off1 != NULL) slot1 = in_off.h_dim_off1[start_offset + i] + 1;


          if (pargs.ht2 != NULL && pargs.key_col2 != NULL) {
            hash = HASH(pargs.key_col2[lo_offset], pargs.dim_len2, pargs.min_key2);
            slot = reinterpret_cast<long long*>(pargs.ht2)[hash];
            if (slot == 0) continue;
            slot2 = slot >> 32;
          } else if (in_off.h_dim_off2 != NULL) slot2 = in_off.h_dim_off2[start_offset + i] + 1;


          if (pargs.ht3 != NULL && pargs.key_col3 != NULL) {
            hash = HASH(pargs.key_col3[lo_offset], pargs.dim_len3, pargs.min_key3);
            slot = reinterpret_cast<long long*>(pargs.ht3)[hash];
            if (slot == 0) continue;
            slot3 = slot >> 32;
          } else if (in_off.h_dim_off3 != NULL) slot3 = in_off.h_dim_off3[start_offset + i] + 1;


          if (pargs.ht4 != NULL && pargs.key_col4 != NULL) {
            hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
            slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
            if (slot == 0) continue;
            slot4 = slot >> 32;
          } else if (in_off.h_dim_off4 != NULL) slot4 = in_off.h_dim_off4[start_offset + i] + 1;

          if (out_off.h_lo_off != NULL) out_off.h_lo_off[count] = lo_offset;
          if (out_off.h_dim_off1 != NULL) out_off.h_dim_off1[count] = slot1-1;
          if (out_off.h_dim_off2 != NULL) out_off.h_dim_off2[count] = slot2-1;
          if (out_off.h_dim_off3 != NULL) out_off.h_dim_off3[count] = slot3-1;
          if (out_off.h_dim_off4 != NULL) out_off.h_dim_off4[count] = slot4-1;

          count++;

    }

    *total = count;

}

void probe_group_by_CPUHE(
  struct probeArgsCPU pargs,  struct groupbyArgsCPU gargs, int num_tuples, 
  int* res, int start_offset = 0, short* segment_group = NULL) {

    for (int i = 0; i < num_tuples; i++) {

        int hash;
        long long slot;
        int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
        int lo_offset;

        lo_offset = start_offset + i;

        if (pargs.key_col1 != NULL && pargs.ht1 != NULL) {
          hash = HASH(pargs.key_col1[lo_offset], pargs.dim_len1, pargs.min_key1);
          slot = reinterpret_cast<long long*>(pargs.ht1)[hash];
          if (slot == 0) continue;
          dim_val1 = slot;
        }

        if (pargs.key_col2 != NULL && pargs.ht2 != NULL) {
          hash = HASH(pargs.key_col2[lo_offset], pargs.dim_len2, pargs.min_key2);
          slot = reinterpret_cast<long long*>(pargs.ht2)[hash];
          if (slot == 0) continue;
          dim_val2 = slot;
        }

        if (pargs.key_col3 != NULL && pargs.ht3 != NULL) {
          hash = HASH(pargs.key_col3[lo_offset], pargs.dim_len3, pargs.min_key3);
          slot = reinterpret_cast<long long*>(pargs.ht3)[hash];
          if (slot == 0) continue;
          dim_val3 = slot;
        }

        if (pargs.key_col4 != NULL && pargs.ht4 != NULL) {
          hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
          slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
          if (slot == 0) continue;
          dim_val4 = slot;
        }

        hash = ((dim_val1 - gargs.min_val1) * gargs.unique_val1 + (dim_val2 - gargs.min_val2) * gargs.unique_val2 +  (dim_val3 - gargs.min_val3) * gargs.unique_val3 + (dim_val4 - gargs.min_val4) * gargs.unique_val4) % gargs.total_val;
        if (dim_val1 != 0) res[hash * 6] = dim_val1;
        if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
        if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
        if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

        int aggr1 = 0; int aggr2 = 0;
        if (gargs.aggr_col1 != NULL) aggr1 = gargs.aggr_col1[lo_offset];
        if (gargs.aggr_col2 != NULL) aggr2 = gargs.aggr_col2[lo_offset];
        // int temp = (*(gargs.h_group_func))(aggr1, aggr2);
        int temp = aggr1 - aggr2;

        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);

    }

}

void probe_group_by_CPU2HE(struct offsetCPU offset,
  struct probeArgsCPU pargs,  struct groupbyArgsCPU gargs, int num_tuples,
  int* res, int start_offset = 0) {

  assert(offset.h_lo_off != NULL);

    for (int i = 0; i < num_tuples; i++) {

          int hash;
          long long slot;
          int dim_val1 = 0, dim_val2 = 0, dim_val3 = 0, dim_val4 = 0;
          int lo_offset;

          lo_offset = offset.h_lo_off[start_offset + i];

          if (pargs.key_col1 != NULL && pargs.ht1 != NULL) {
            hash = HASH(pargs.key_col1[lo_offset], pargs.dim_len1, pargs.min_key1);
            slot = reinterpret_cast<long long*>(pargs.ht1)[hash];
            if (slot == 0) continue;
            dim_val1 = slot;
          } else if (gargs.group_col1 != NULL) {
            assert(offset.h_dim_off1 != NULL);
            dim_val1 = gargs.group_col1[offset.h_dim_off1[start_offset + i]];
          }

          if (pargs.key_col2 != NULL && pargs.ht2 != NULL) {
            hash = HASH(pargs.key_col2[lo_offset], pargs.dim_len2, pargs.min_key2);
            slot = reinterpret_cast<long long*>(pargs.ht2)[hash];
            if (slot == 0) continue;
            dim_val2 = slot;
          } else if (gargs.group_col2 != NULL) {
            assert(offset.h_dim_off2 != NULL);
            dim_val2 = gargs.group_col2[offset.h_dim_off2[start_offset + i]];
          }

          if (pargs.key_col3 != NULL && pargs.ht3 != NULL) {
            hash = HASH(pargs.key_col3[lo_offset], pargs.dim_len3, pargs.min_key3);
            slot = reinterpret_cast<long long*>(pargs.ht3)[hash];
            if (slot == 0) continue;
            dim_val3 = slot;
          } else if (gargs.group_col3 != NULL) {
            assert(offset.h_dim_off3 != NULL);
            dim_val3 = gargs.group_col3[offset.h_dim_off3[start_offset + i]];
          }

          if (pargs.key_col4 != NULL && pargs.ht4 != NULL) {
            hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
            slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
            if (slot == 0) continue;
            dim_val4 = slot;
          } else if (gargs.group_col4 != NULL) {
            assert(offset.h_dim_off4 != NULL);
            dim_val4 = gargs.group_col4[offset.h_dim_off4[start_offset + i]];
          }

          hash = ((dim_val1 - gargs.min_val1) * gargs.unique_val1 + (dim_val2 - gargs.min_val2) * gargs.unique_val2 +  (dim_val3 - gargs.min_val3) * gargs.unique_val3 + (dim_val4 - gargs.min_val4) * gargs.unique_val4) % gargs.total_val;
          if (dim_val1 != 0) res[hash * 6] = dim_val1;
          if (dim_val2 != 0) res[hash * 6 + 1] = dim_val2;
          if (dim_val3 != 0) res[hash * 6 + 2] = dim_val3;
          if (dim_val4 != 0) res[hash * 6 + 3] = dim_val4;

          int aggr1 = 0; int aggr2 = 0;
          if (gargs.aggr_col1 != NULL) aggr1 = gargs.aggr_col1[lo_offset];
          if (gargs.aggr_col2 != NULL) aggr2 = gargs.aggr_col2[lo_offset];
          // int temp = (*(gargs.h_group_func))(aggr1, aggr2);
          int temp = aggr1 - aggr2;

          __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);

    }

}

void build_CPUHE(struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table,
  int start_offset = 0, short* segment_group = NULL) {

  assert(bargs.key_col != NULL);
  assert(hash_table != NULL);

    for (int i = 0; i < num_tuples; i++) {

          int table_offset;
          int flag = 1;

          table_offset = start_offset+i;

          if (fargs.filter_col1 != NULL) {
            if (fargs.mode1 == 1)
              flag = (fargs.filter_col1[table_offset] >= fargs.compare1 && fargs.filter_col1[table_offset] <= fargs.compare2);
            else if (fargs.mode1 == 2)
              flag = (fargs.filter_col1[table_offset] == fargs.compare1 || fargs.filter_col1[table_offset] == fargs.compare2);
          }

          if (flag) {
            int key = bargs.key_col[table_offset];
            int hash = HASH(key, bargs.num_slots, bargs.val_min);
            hash_table[(hash << 1) + 1] = table_offset + 1;
            if (bargs.val_col != NULL) hash_table[hash << 1] = bargs.val_col[table_offset];
          }

    }
}

void build_CPU2HE(int *dim_off, struct filterArgsCPU fargs,
  struct buildArgsCPU bargs, int num_tuples, int* hash_table,
  int start_offset = 0) {

  assert(bargs.key_col != NULL);
  assert(hash_table != NULL);
  assert(dim_off != NULL);

    for (int i = 0; i < num_tuples; i++) {

        int table_offset;

        table_offset = dim_off[start_offset + i];

        int key = bargs.key_col[table_offset];
        int hash = HASH(key, bargs.num_slots, bargs.val_min);
        hash_table[(hash << 1) + 1] = table_offset + 1;
        if (bargs.val_col != NULL) hash_table[hash << 1] = bargs.val_col[table_offset];

    }

}

void groupByCPUHE(struct offsetCPU offset, 
  struct groupbyArgsCPU gargs, int num_tuples, int* res) {

      for (int i = 0; i < num_tuples; i++) {
              int groupval1 = 0, groupval2 = 0, groupval3 = 0, groupval4 = 0;
              int aggrval1 = 0, aggrval2 = 0;

              if (gargs.group_col1 != NULL) {
                assert(offset.h_dim_off1 != NULL);
                groupval1 = gargs.group_col1[offset.h_dim_off1[i]];
              }

              if (gargs.group_col2 != NULL) {
                assert(offset.h_dim_off2 != NULL);
                groupval2 = gargs.group_col2[offset.h_dim_off2[i]];
              }

              if (gargs.group_col3 != NULL) {
                assert(offset.h_dim_off3 != NULL);
                groupval3 = gargs.group_col3[offset.h_dim_off3[i]];
              }

              if (gargs.group_col4 != NULL) {
                assert(offset.h_dim_off4 != NULL);
                groupval4 = gargs.group_col4[offset.h_dim_off4[i]];
              }

              assert(offset.h_lo_off != NULL);
              if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[offset.h_lo_off[i]];
              if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[offset.h_lo_off[i]];

              int hash = ((groupval1 - gargs.min_val1) * gargs.unique_val1 + (groupval2 - gargs.min_val2) * gargs.unique_val2 +  (groupval3 - gargs.min_val3) * gargs.unique_val3 + (groupval4 - gargs.min_val4) * gargs.unique_val4) % gargs.total_val;

              if (groupval1 != 0) res[hash * 6] = groupval1;
              if (groupval2 != 0) res[hash * 6 + 1] = groupval2;
              if (groupval3 != 0) res[hash * 6 + 2] = groupval3;
              if (groupval4 != 0) res[hash * 6 + 3] = groupval4;

              if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[offset.h_lo_off[i]];
              if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[offset.h_lo_off[i]];
              // int temp = (*(gargs.h_group_func))(aggrval1, aggrval2);
              int temp = aggrval1 - aggrval2;

              __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), (long long)(temp), __ATOMIC_RELAXED);
      }

}

void aggregationCPUHE(int* lo_off, 
  struct groupbyArgsCPU gargs, int num_tuples, int* res) {

  assert(lo_off != NULL);

    long long local_sum = 0;


            for (int i = 0; i < num_tuples; i++) {
              int aggrval1 = 0, aggrval2 = 0;

              if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[lo_off[i]];
              if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[lo_off[i]];

              // local_sum += (*(gargs.h_group_func))(aggrval1, aggrval2);
              local_sum += aggrval1 * aggrval2;

            }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);
}


void probe_aggr_CPUHE(
  struct probeArgsCPU pargs, struct groupbyArgsCPU gargs, int num_tuples,
  int* res, int start_offset = 0, short* segment_group = NULL) {

  assert(segment_group != NULL);

    long long local_sum = 0;

          for (int i = 0; i < num_tuples; i++) {
            int hash;
            long long slot;
            int lo_offset;

            lo_offset = start_offset + i;

              hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
              slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
              if (slot == 0) continue;

            int aggrval1 = 0, aggrval2 = 0;
            if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[lo_offset];
            if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[lo_offset];
            // local_sum += (*(gargs.h_group_func))(aggrval1, aggrval2);
            local_sum += aggrval1 * aggrval2;
          }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);


}

void probe_aggr_CPU2HE(struct offsetCPU offset,
  struct probeArgsCPU pargs, struct groupbyArgsCPU gargs, 
  int num_tuples, int* res, int start_offset = 0) {

  assert(offset.h_lo_off != NULL);

    long long local_sum = 0;

            for (int i = 0; i < num_tuples; i++) {
                int hash;
                long long slot;
                int lo_offset;

                lo_offset = offset.h_lo_off[start_offset + i];

                hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
                slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
                if (slot == 0) continue;

                int aggrval1 = 0, aggrval2 = 0;
                if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[lo_offset];
                if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[lo_offset];
                local_sum += aggrval1 * aggrval2;

            }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);


}

void filter_probe_aggr_CPUHE(
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset = 0, short* segment_group = NULL) {

  // assert(segment_group != NULL);

    long long local_sum = 0;

              for (int i = 0; i < num_tuples; i++) {
                int hash;
                long long slot;
                int lo_offset;

                lo_offset = start_offset + i;

                  if (!(fargs.filter_col1[lo_offset] >= fargs.compare1 && fargs.filter_col1[lo_offset] <= fargs.compare2)) continue; //only for Q1.x

                  if (!(fargs.filter_col2[lo_offset] >= fargs.compare3 && fargs.filter_col2[lo_offset] <= fargs.compare4)) continue; //only for Q1.x

                  hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
                  slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
                  if (slot == 0) continue;

                int aggrval1 = 0, aggrval2 = 0;
                if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[lo_offset];
                if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[lo_offset];
                local_sum += aggrval1 * aggrval2;

              }

    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

}

void filter_probe_aggr_CPU2HE(struct offsetCPU offset,
  struct filterArgsCPU fargs, struct probeArgsCPU pargs, struct groupbyArgsCPU gargs,
  int num_tuples, int* res, int start_offset = 0) {

  assert(offset.h_lo_off != NULL);

    long long local_sum = 0;

            for (int i = 0; i < num_tuples; i++) {
              int hash;
              long long slot;
              int lo_offset;

              lo_offset = offset.h_lo_off[start_offset + i];

                if (!(fargs.filter_col2[lo_offset] >= fargs.compare3 && fargs.filter_col2[lo_offset] <= fargs.compare4)) continue; //only for Q1.x

                hash = HASH(pargs.key_col4[lo_offset], pargs.dim_len4, pargs.min_key4);
                slot = reinterpret_cast<long long*>(pargs.ht4)[hash];
                if (slot == 0) continue;


              int aggrval1 = 0, aggrval2 = 0;
              if (gargs.aggr_col1 != NULL) aggrval1 = gargs.aggr_col1[lo_offset];
              if (gargs.aggr_col2 != NULL) aggrval2 = gargs.aggr_col2[lo_offset];

              local_sum += aggrval1 * aggrval2;
            }

     __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&res[4]), (long long)(local_sum), __ATOMIC_RELAXED);

}


void filter_CPUHE(struct filterArgsCPU fargs,
  int* out_off, int num_tuples, int* total,
  int start_offset = 0, short* segment_group = NULL) {

        int count = 0;

        for (int i = 0; i < num_tuples; i++) {
              bool selection_flag = 1;
              int col_offset;

              col_offset = start_offset + i;

              if (fargs.filter_col1 != NULL) {
                if (fargs.mode1 == 1)
                  selection_flag = (fargs.filter_col1[col_offset] >= fargs.compare1 && fargs.filter_col1[col_offset] <= fargs.compare2);
                else if (fargs.mode1 == 2)
                  selection_flag = (fargs.filter_col1[col_offset] == fargs.compare1 || fargs.filter_col1[col_offset] == fargs.compare2);
              }

              if (fargs.filter_col2 != NULL) {
                if (fargs.mode2 == 1)
                  selection_flag = selection_flag && (fargs.filter_col2[col_offset] >= fargs.compare3 && fargs.filter_col2[col_offset] <= fargs.compare4);
                else if (fargs.mode2 == 2)
                  selection_flag = selection_flag && (fargs.filter_col2[col_offset] == fargs.compare3 || fargs.filter_col2[col_offset] == fargs.compare4);
              }

              if (selection_flag) {
                out_off[count] = col_offset;
                count++;
              }
        }

        *total = count;

}

void filter_CPU2HE(int* off_col, struct filterArgsCPU fargs,
  int* out_off, int num_tuples, int* total,
  int start_offset = 0) {

  assert(off_col != NULL);

        int count = 0;

        for (int i = 0; i < num_tuples; i++) {
              bool selection_flag = 1;
              int col_offset;

              col_offset = off_col[start_offset + i];

              if (fargs.filter_col1 != NULL) {
                if (fargs.mode1 == 1)
                  selection_flag = (fargs.filter_col1[col_offset] >= fargs.compare1 && fargs.filter_col1[col_offset] <= fargs.compare2);
                else if (fargs.mode1 == 2)
                  selection_flag = (fargs.filter_col1[col_offset] == fargs.compare1 || fargs.filter_col1[col_offset] == fargs.compare2);
              }

              if (fargs.filter_col2 != NULL) {
                if (fargs.mode2 == 1)
                  selection_flag = selection_flag && (fargs.filter_col2[col_offset] >= fargs.compare3 && fargs.filter_col2[col_offset] <= fargs.compare4);
                else if (fargs.mode2 == 2)
                  selection_flag = selection_flag && (fargs.filter_col2[col_offset] == fargs.compare3 || fargs.filter_col2[col_offset] == fargs.compare4);
              }

              if (selection_flag) {
                out_off[count] = col_offset;
                count++;        
              }
        }

        *total = count;

}