#ifndef _QUERY_PROCESSING_H_
#define _QUERY_PROCESSING_H_

#include "CPUGPUProcessing.h"

#define NUM_QUERIES 4

class QueryProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;
  CPUGPUProcessing* cgp;

  QueryParams* params;
  cudaStream_t streams[64];

  vector<uint64_t> query_freq;
  int query;

  QueryProcessing(CPUGPUProcessing* _cgp, int _query) {
    cgp = _cgp;
    qo = cgp->qo;
    cm = cgp->cm;
    query_freq.resize(NUM_QUERIES);
    query = _query;
    params = new QueryParams(_query);
  }

  void generate_rand_query() {
    query = rand() % NUM_QUERIES;
  }

  void runQuery();

  void prepareQuery();

  void endQuery();

  void updateStatsQuery();

  void processQuery();

};

void
QueryProcessing::runQuery() {

  for (int i = 0; i < qo->join.size(); i++) {
    int table_id = qo->join[i].second->table_id;

    // for (short j = 0; j < qo->par_segment_count[table_id]; j++) {

    parallel_for(short(0), qo->par_segment_count[table_id], [=](short j){

      int sg = qo->par_segment[table_id][j];

      cudaStreamCreate(&streams[sg]);

      int *h_off_col = NULL, *d_off_col = NULL;

      if (qo->segment_group_count[table_id][sg] > 0) {

        int* d_total = NULL;
        int* h_total = NULL;

        h_total = cm->customCudaHostAlloc(1);
        memset(h_total, 0, sizeof(int));
        d_total = cm->customCudaMalloc(1);

        cout << qo->join[i].second->column_name << endl;

        printf("sg = %d\n", sg);

        if (sg == 0) {

          if (qo->joinCPUcheck[table_id] && qo->joinGPUcheck[table_id]) {
            cgp->call_bfilter_CPU(params, h_off_col, h_total, sg, table_id);
            cgp->switch_device_dim(d_off_col, h_off_col, d_total, h_total, sg, 0, table_id, streams[sg]);
            cgp->call_build_GPU(params, d_off_col, h_total, sg, table_id, streams[sg]);
            cgp->call_build_CPU(params, h_off_col, h_total, sg, table_id);
          } else if (qo->joinCPUcheck[table_id] && !(qo->joinGPUcheck[table_id])) {
            cgp->call_bfilter_build_CPU(params, h_off_col, h_total, sg, table_id);
          } else if (!(qo->joinCPUcheck[table_id]) && qo->joinGPUcheck[table_id]) {
            cgp->call_bfilter_CPU(params, h_off_col, h_total, sg, table_id);
            cgp->switch_device_dim(d_off_col, h_off_col, d_total, h_total, sg, 0, table_id, streams[sg]);
            cgp->call_build_GPU(params, d_off_col, h_total, sg, table_id, streams[sg]);            
          }

        } else {

          if (qo->joinCPUcheck[table_id]) {
            cgp->call_bfilter_build_CPU(params, h_off_col, h_total, sg, table_id);
          }

          if (qo->joinGPUcheck[table_id]) {
            cgp->call_bfilter_build_GPU(params, d_off_col, h_total, sg, table_id, streams[sg]);
          }
          
        }

      }

      cudaStreamSynchronize(streams[sg]);
      cudaStreamDestroy(streams[sg]);

    });

    cudaDeviceSynchronize();
  }

  parallel_for(short(0), qo->par_segment_count[0], [=](short i){

  //for (int i = 0; i < qo->par_segment_count[0]; i++) {
    int sg = qo->par_segment[0][i];

    int** h_off_col = NULL, **off_col = NULL;

    cudaStreamCreate(&streams[sg]);

    if (qo->segment_group_count[0][sg] > 0) {

      int* d_total = NULL;
      int* h_total = NULL;

      h_total = cm->customCudaHostAlloc(1);
      memset(h_total, 0, sizeof(int));
      d_total = cm->customCudaMalloc(1);

      printf("sg = %d\n", sg);

      //printf("%zu %zu %zu %zu\n", qo->selectCPUPipelineCol[sg].size(), qo->selectGPUPipelineCol[sg].size(), qo->joinCPUPipelineCol[sg].size(), qo->joinGPUPipelineCol[sg].size());

      if (qo->selectCPUPipelineCol[sg].size() > 0) {
        if (qo->selectGPUPipelineCol[sg].size() > 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_group_by_GPU(params, off_col, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() > 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            cgp->call_pfilter_probe_group_by_CPU(params, h_off_col, h_total, sg, 0);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }   
        }
      } else {
        if (qo->selectGPUPipelineCol[sg].size() > 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            cgp->call_pfilter_probe_group_by_GPU(params, off_col, h_total, sg, 0, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() > 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }   
        }
      }






      // if (qo->selectGPUPipelineCol[sg].size() > 0) {
      //   if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_pfilter_probe_group_by_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

      //     }
      //   } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_probe_CPU(params, h_off_col, h_total, sg);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_probe_CPU(params, h_off_col, h_total, sg);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_probe_CPU(params, h_off_col, h_total, sg);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

      //     }
      //   } else if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
      //       assert(0);
      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //       assert(0);
      //     }
      //   } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
      //       assert(0);
      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_probe_group_by_GPU(params, off_col, h_total, sg, 0, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //       assert(0);
      //     }
      //   }
      // } else {
      //   if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_probe_group_by_CPU(params, h_off_col, h_total, sg, 0);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

      //     }
      //   } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_probe_CPU(params, h_off_col, h_total, sg);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_probe_CPU(params, h_off_col, h_total, sg);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_probe_CPU(params, h_off_col, h_total, sg);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

      //     }
      //   } else if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
      //       assert(0);
      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //       assert(0);
      //     }
      //   } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
      //     if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
      //       assert(0);
      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

      //       cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //       cgp->call_group_by_CPU(params, h_off_col, h_total);

      //     } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

      //       cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

      //     } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //       assert(0);
      //     }
      //   }
      // }





      // cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);

      // if (qo->selectCPUPipelineCol[sg].size() > 0 && (qo->joinGPUPipelineCol[sg].size() > 0 || qo->selectGPUPipelineCol[sg].size() > 0 || qo->groupbyGPUPipelineCol[sg].size() > 0)) 
      //   cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);

      // cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);

      // cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);

      // if ((qo->selectGPUPipelineCol[sg].size() > 0 || qo->joinGPUPipelineCol[sg].size() > 0) && (qo->joinCPUPipelineCol[sg].size() > 0 || qo->groupbyCPUPipelineCol[sg].size() > 0))
      //   cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);

      // //printf("h_total = %d\n", h_total);

      // cgp->call_probe_CPU(params, h_off_col, h_total, sg);

      // if (qo->groupGPUcheck) {
      //   if (qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //     if (qo->joinCPUPipelineCol[sg].size() > 0)
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
      //     cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);
      //   } else {
      //     cgp->call_group_by_CPU(params, h_off_col, h_total);
      //   }
      // } else {
      //   cgp->call_group_by_CPU(params, h_off_col, h_total);
      // }




      // cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);

      // if (qo->selectGPUPipelineCol[sg].size() > 0 && (qo->joinCPUPipelineCol[sg].size() > 0 || qo->selectCPUPipelineCol[sg].size() > 0 || qo->groupbyCPUPipelineCol[sg].size() > 0))
      // cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);

      // cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());

      // cgp->call_probe_CPU(params, h_off_col, h_total, sg);

      // if ((qo->selectCPUPipelineCol[sg].size() > 0 || qo->joinCPUPipelineCol[sg].size() > 0) && (qo->joinGPUPipelineCol[sg].size() > 0 || qo->groupbyGPUPipelineCol[sg].size() > 0))
      //   cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);

      // cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);

      // if (qo->groupGPUcheck) {
      //   if (qo->groupbyGPUPipelineCol[sg].size() > 0) {
      //     cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);
      //   } else {
      //     if (qo->joinGPUPipelineCol[sg].size() > 0)
      //       cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //     cgp->call_group_by_CPU(params, h_off_col, h_total);
      //   }
      // } else {
      //   if (qo->joinGPUPipelineCol[sg].size() > 0)
      //     cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
      //   cgp->call_group_by_CPU(params, h_off_col, h_total);
      // }

    }

    cudaStreamSynchronize(streams[sg]);
    cudaStreamDestroy(streams[sg]);

  });

  cudaDeviceSynchronize();
  
  int* resGPU = cm->customMalloc(params->total_val * 6);
  CubDebugExit(cudaMemcpy(resGPU, params->d_res, params->total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));
  merge(params->res, resGPU, params->total_val);
}

void 
QueryProcessing::processQuery() {

  // chrono::high_resolution_clock::time_point st, finish;
  // chrono::duration<double> diff;
  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;

  qo->parseQuery(query);
  updateStatsQuery();
  prepareQuery();

  // st = chrono::high_resolution_clock::now();
  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  runQuery();

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  // finish = chrono::high_resolution_clock::now();

  // diff = finish - st;

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i< params->total_val; i++) {
    if (params->res[6*i+4] != 0) {
      cout << params->res[6*i] << " " << params->res[6*i+1] << " " << params->res[6*i+2] << " " << params->res[6*i+3] << " " << reinterpret_cast<unsigned long long*>(&params->res[6*i+4])[0]  << endl;
      res_count++;
    }
  }
  cout << "Res count = " << res_count << endl;
  cout << "Time Taken Total: " << time << endl;

  endQuery();

};

void
QueryProcessing::endQuery() {

  params->min_key.clear();
  params->min_val.clear();
  params->unique_val.clear();
  params->dim_len.clear();

  // unordered_map<ColumnInfo*, int*>::iterator it;
  // for (it = cgp->col_idx.begin(); it != cgp->col_idx.end(); it++) {
  //   it->second = NULL;
  // }

  params->ht_CPU.clear();
  params->ht_GPU.clear();
  //cgp->col_idx.clear();

  params->compare1.clear();
  params->compare2.clear();
  params->mode.clear();

  qo->clearVector();

  cm->resetPointer();

  cgp->resetCGP();

}

void
QueryProcessing::updateStatsQuery() {
  chrono::high_resolution_clock::time_point cur_time = chrono::high_resolution_clock::now();
  chrono::duration<double> timestamp = cur_time - cgp->begin_time;
  query_freq[query]++;

  for (int i = 0; i < qo->querySelectColumn.size(); i++) {
    cm->updateColumnFrequency(qo->querySelectColumn[i]);
    cm->updateColumnTimestamp(qo->querySelectColumn[i], timestamp.count());
    cm->updateQueryFrequency(qo->querySelectColumn[i], query_freq[query]);
  }
  for (int i = 0; i < qo->queryBuildColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryBuildColumn[i]);
    cm->updateColumnTimestamp(qo->queryBuildColumn[i], timestamp.count());
    cm->updateQueryFrequency(qo->queryBuildColumn[i], query_freq[query]);
  }
  for (int i = 0; i < qo->queryProbeColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryProbeColumn[i]);
    cm->updateColumnTimestamp(qo->queryProbeColumn[i], timestamp.count());
    cm->updateQueryFrequency(qo->queryProbeColumn[i], query_freq[query]);
  }
  for (int i = 0; i < qo->queryGroupByColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryGroupByColumn[i]);
    cm->updateColumnTimestamp(qo->queryGroupByColumn[i], timestamp.count());
    cm->updateQueryFrequency(qo->queryGroupByColumn[i], query_freq[query]);
  }
  for (int i = 0; i < qo->queryAggrColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryAggrColumn[i]);
    cm->updateColumnTimestamp(qo->queryAggrColumn[i], timestamp.count());
    cm->updateQueryFrequency(qo->queryAggrColumn[i], query_freq[query]);
  }
}



void 
QueryProcessing::prepareQuery() {

  // chrono::high_resolution_clock::time_point st, finish;
  // chrono::duration<double> diff;

  // cudaEvent_t start, stop;   // variables that holds 2 events 
  // float time;


  if (query == 0) {

    params->selectivity[cm->d_year] = 1;
    params->selectivity[cm->lo_orderdate] = 1;
    params->selectivity[cm->lo_discount] = 3.0/11 * 2;
    params->selectivity[cm->lo_quantity] = 0.5 * 2;

    params->mode[cm->d_year] = 0;
    params->compare1[cm->d_year] = 1993;
    params->mode[cm->lo_discount] = 1;
    params->compare1[cm->lo_discount] = 1;
    params->compare2[cm->lo_discount] = 3;
    params->mode[cm->lo_quantity] = 1;
    params->compare1[cm->lo_quantity] = 0;
    params->compare2[cm->lo_quantity] = 24;
    params->mode_group = 2;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 0;
    params->unique_val[cm->c_custkey] = 0;
    params->unique_val[cm->s_suppkey] = 0;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = 0;
    params->dim_len[cm->c_custkey] = 0;
    params->dim_len[cm->s_suppkey] = 0;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = 1;

    params->ht_p = NULL;
    params->ht_c = NULL;
    params->ht_s = NULL;
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));

    params->d_ht_d = cm->customCudaMalloc(4 * params->dim_len[cm->d_datekey]);
    params->d_ht_p = NULL;
    params->d_ht_c = NULL;
    params->d_ht_s = NULL;

    CubDebugExit(cudaMemset(params->d_ht_d, 0, 4 * params->dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 1) {

    params->selectivity[cm->p_category] = 1.0/25 * 2;
    params->selectivity[cm->s_region] = 0.2 * 2;
    params->selectivity[cm->d_year] = 1;
    params->selectivity[cm->lo_partkey] = 1.0/25 * 2;
    params->selectivity[cm->lo_suppkey] = 0.2 * 2;
    params->selectivity[cm->lo_orderdate] = 1;

    params->mode[cm->s_region] = 0;
    params->compare1[cm->s_region] = 1;
    params->mode[cm->p_category] = 0;
    params->compare1[cm->p_category] = 1;
    params->mode_group = 0;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 7;
    params->unique_val[cm->c_custkey] = 0;
    params->unique_val[cm->s_suppkey] = 0;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = P_LEN;
    params->dim_len[cm->c_custkey] = 0;
    params->dim_len[cm->s_suppkey] = S_LEN;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = ((1998-1992+1) * (5 * 5 * 40));

    params->ht_p = cm->customMalloc(2 * params->dim_len[cm->p_partkey]);
    params->ht_c = NULL;
    params->ht_s = cm->customMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    params->d_ht_p = cm->customCudaMalloc(2 * params->dim_len[cm->p_partkey]);
    params->d_ht_s = cm->customCudaMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->d_ht_d = cm->customCudaMalloc(2 * params->dim_len[cm->d_datekey]);
    params->d_ht_c = NULL;

    // cudaEventCreate(&start);   // creating the event 1
    // cudaEventCreate(&stop);    // creating the event 2
    // cudaEventRecord(start, 0); // start measuring  the time

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
    memset(params->ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int));
    memset(params->ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));

    // cudaEventRecord(stop, 0);                  // Stop time measuring
    // cudaEventSynchronize(stop);               // Wait until the completion of all device 
    //                                           // work preceding the most recent call to cudaEventRecord()
    // cudaEventElapsedTime(&time, start, stop); // Saving the time measured

    CubDebugExit(cudaMemset(params->d_ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 2) {

    params->selectivity[cm->c_region] = 0.2 * 2;
    params->selectivity[cm->s_region] = 0.2 * 2;
    params->selectivity[cm->d_year] =  1;
    params->selectivity[cm->lo_custkey] = 0.2 * 2;
    params->selectivity[cm->lo_suppkey] = 0.2 * 2;
    params->selectivity[cm->lo_orderdate] =  1;

    params->mode[cm->c_region] = 0;
    params->compare1[cm->c_region] = 2;
    params->mode[cm->s_region] = 0;
    params->compare1[cm->s_region] = 2;
    params->mode[cm->d_year] = 1;
    params->compare1[cm->d_year] = 1992;
    params->compare2[cm->d_year] = 1997;
    params->mode_group = 0;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 0;
    params->unique_val[cm->c_custkey] = 7;
    params->unique_val[cm->s_suppkey] = 25 * 7;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = 0;
    params->dim_len[cm->c_custkey] = C_LEN;
    params->dim_len[cm->s_suppkey] = S_LEN;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = ((1998-1992+1) * 25 * 25);

    params->ht_p = NULL;
    params->ht_c = cm->customMalloc(2 * params->dim_len[cm->c_custkey]);
    params->ht_s = cm->customMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
    memset(params->ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int));
    memset(params->ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));

    params->d_ht_c = cm->customCudaMalloc(2 * params->dim_len[cm->c_custkey]);
    params->d_ht_s = cm->customCudaMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->d_ht_d = cm->customCudaMalloc(2 * params->dim_len[cm->d_datekey]);
    params->d_ht_p = NULL;

    CubDebugExit(cudaMemset(params->d_ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 3) {

    params->selectivity[cm->p_mfgr] = 0.4 * 2;
    params->selectivity[cm->c_region] = 0.2 * 2;
    params->selectivity[cm->s_region] = 0.2 * 2;
    params->selectivity[cm->d_year] =  1;
    params->selectivity[cm->lo_partkey] = 0.4 * 2;
    params->selectivity[cm->lo_custkey] = 0.2 * 2;
    params->selectivity[cm->lo_suppkey] = 0.2 * 2;
    params->selectivity[cm->lo_orderdate] =  1;

    params->mode[cm->c_region] = 0;
    params->compare1[cm->c_region] = 1;
    params->mode[cm->s_region] = 0;
    params->compare1[cm->s_region] = 1;
    params->mode[cm->p_mfgr] = 2;
    params->compare1[cm->p_mfgr] = 0;
    params->compare2[cm->p_mfgr] = 1;
    params->mode_group = 1;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 0;
    params->unique_val[cm->c_custkey] = 7;
    params->unique_val[cm->s_suppkey] = 0;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = P_LEN;
    params->dim_len[cm->c_custkey] = C_LEN;
    params->dim_len[cm->s_suppkey] = S_LEN;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = ((1998-1992+1) * 25);

    params->ht_p = cm->customMalloc(4 * params->dim_len[cm->p_partkey]);
    params->ht_c = cm->customMalloc(2 * params->dim_len[cm->c_custkey]);
    params->ht_s = cm->customMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
    memset(params->ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int));
    memset(params->ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));
    memset(params->ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int));

    params->d_ht_p = cm->customCudaMalloc(2 * params->dim_len[cm->p_partkey]);
    params->d_ht_s = cm->customCudaMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->d_ht_d = cm->customCudaMalloc(2 * params->dim_len[cm->d_datekey]);
    params->d_ht_c = cm->customCudaMalloc(2 * params->dim_len[cm->c_custkey]);

    CubDebugExit(cudaMemset(params->d_ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int)));

  }

  // diff = finish - st;
  // cout << "Time Taken Total: " << time << endl;

  params->ht_GPU[cm->p_partkey] = params->d_ht_p;
  params->ht_GPU[cm->c_custkey] = params->d_ht_c;
  params->ht_GPU[cm->s_suppkey] = params->d_ht_s;
  params->ht_GPU[cm->d_datekey] = params->d_ht_d;

  params->ht_CPU[cm->p_partkey] = params->ht_p;
  params->ht_CPU[cm->c_custkey] = params->ht_c;
  params->ht_CPU[cm->s_suppkey] = params->ht_s;
  params->ht_CPU[cm->d_datekey] = params->ht_d;

  int res_array_size = params->total_val * 6;
  params->res = cm->customMalloc(res_array_size);
  memset(params->res, 0, res_array_size * sizeof(int));
     
  params->d_res = cm->customCudaMalloc(res_array_size);
  CubDebugExit(cudaMemset(params->d_res, 0, res_array_size * sizeof(int)));

};

#endif