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
  cudaStream_t streams[128];

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

  void runQuery2();

  void runOnDemand();

  void prepareQuery();

  void endQuery();

  void updateStatsQuery();

  float processQuery();

  float processQuery2();

  float processOnDemand();

};

void
QueryProcessing::runQuery() {

  for (int i = 0; i < qo->join.size(); i++) {
    int table_id = qo->join[i].second->table_id;

    // for (short j = 0; j < qo->par_segment_count[table_id]; j++) {

    parallel_for(short(0), qo->par_segment_count[table_id], [=](short j){

      int sg = qo->par_segment[table_id][j];

      CubDebugExit(cudaStreamCreate(&streams[sg]));

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

      CubDebugExit(cudaStreamSynchronize(streams[sg]));
      CubDebugExit(cudaStreamDestroy(streams[sg]));

    });

    CubDebugExit(cudaDeviceSynchronize());
  }

  parallel_for(short(0), qo->par_segment_count[0], [=](short i){

  //for (int i = 0; i < qo->par_segment_count[0]; i++) {
    int sg = qo->par_segment[0][i];

    int** h_off_col = NULL, **off_col = NULL;

    CubDebugExit(cudaStreamCreate(&streams[sg]));

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
            if (qo->groupby_build.size() == 0) cgp->call_pfilter_probe_aggr_GPU(params, off_col, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]); 
            else cgp->call_pfilter_probe_group_by_GPU(params, off_col, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total); 
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]); 
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);          

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total); 
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

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
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, qo->selectCPUPipelineCol[sg].size(), streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            if (qo->groupby_build.size() == 0) cgp->call_pfilter_probe_aggr_CPU(params, h_off_col, h_total, sg, 0);
            else cgp->call_pfilter_probe_group_by_CPU(params, h_off_col, h_total, sg, 0);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }   
        }
      } else {
        if (qo->selectGPUPipelineCol[sg].size() > 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            if (qo->groupby_build.size() == 0) cgp->call_pfilter_probe_aggr_GPU(params, off_col, h_total, sg, 0, streams[sg]);
            else cgp->call_pfilter_probe_group_by_GPU(params, off_col, h_total, sg, 0, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total); 
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() > 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]); 
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);   

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total); 
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          }
        } else if (qo->selectGPUPipelineCol[sg].size() > 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }
        } else if (qo->selectGPUPipelineCol[sg].size() == 0 && qo->joinGPUPipelineCol[sg].size() == 0) {
          if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]); 
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          } else if (qo->joinCPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg); 
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinCPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          }   
        }
      }


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

    CubDebugExit(cudaStreamSynchronize(streams[sg]));
    CubDebugExit(cudaStreamDestroy(streams[sg]));

  });

  CubDebugExit(cudaDeviceSynchronize());
  
  int* resGPU = cm->customMalloc(params->total_val * 6);
  CubDebugExit(cudaMemcpy(resGPU, params->d_res, params->total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));
  merge(params->res, resGPU, params->total_val);
}



void
QueryProcessing::runQuery2() {

  for (int i = 0; i < qo->join.size(); i++) {
    int table_id = qo->join[i].second->table_id;

    // for (short j = 0; j < qo->par_segment_count[table_id]; j++) {

    parallel_for(short(0), qo->par_segment_count[table_id], [=](short j){

      int sg = qo->par_segment[table_id][j];

      CubDebugExit(cudaStreamCreate(&streams[sg]));

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

      CubDebugExit(cudaStreamSynchronize(streams[sg]));
      CubDebugExit(cudaStreamDestroy(streams[sg]));

    });

    CubDebugExit(cudaDeviceSynchronize());
  }

  parallel_for(short(0), qo->par_segment_count[0], [=](short i){

  //for (int i = 0; i < qo->par_segment_count[0]; i++) {
    int sg = qo->par_segment[0][i];

    int** h_off_col = NULL, **off_col = NULL;

    CubDebugExit(cudaStreamCreate(&streams[sg]));

    if (qo->segment_group_count[0][sg] > 0) {

      int* d_total = NULL;
      int* h_total = NULL;

      h_total = cm->customCudaHostAlloc(1);
      memset(h_total, 0, sizeof(int));
      d_total = cm->customCudaMalloc(1);

      printf("sg = %d\n", sg);

      //printf("%zu %zu %zu %zu\n", qo->selectCPUPipelineCol[sg].size(), qo->selectGPUPipelineCol[sg].size(), qo->joinCPUPipelineCol[sg].size(), qo->joinGPUPipelineCol[sg].size());


      if (qo->selectGPUPipelineCol[sg].size() > 0) {
        if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_pfilter_probe_aggr_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
            else cgp->call_pfilter_probe_group_by_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]);
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          }
        } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg);
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]);
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          }
        } else if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, qo->selectGPUPipelineCol[sg].size());
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          }
        } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_probe_GPU(params, off_col, d_total, h_total, sg, 0, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            if (qo->groupby_build.size() == 0) cgp->call_pfilter_probe_aggr_GPU(params, off_col, h_total, sg, 0, streams[sg]);
            else cgp->call_pfilter_probe_group_by_GPU(params, off_col, h_total, sg, 0, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          }
        }
      } else {
        if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            if (qo->groupby_build.size() == 0) cgp->call_pfilter_probe_aggr_CPU(params, h_off_col, h_total, sg, 0);
            else cgp->call_pfilter_probe_group_by_CPU(params, h_off_col, h_total, sg, 0);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_probe_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]);
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          }
        } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() > 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_CPU(params, h_off_col, h_total, sg);
            else cgp->call_probe_group_by_CPU(params, h_off_col, h_total, sg);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_probe_CPU(params, h_off_col, h_total, sg);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_GPU(params, off_col[0], h_total, streams[sg]);
            else cgp->call_group_by_GPU(params, off_col, h_total, streams[sg]);

          }
        } else if (qo->selectCPUPipelineCol[sg].size() > 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            cgp->call_pfilter_CPU(params, h_off_col, h_total, sg, 0);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 0, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          }
        } else if (qo->selectCPUPipelineCol[sg].size() == 0 && qo->joinCPUPipelineCol[sg].size() == 0) {
          if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {
            assert(0);
          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() == 0) {

            cgp->call_probe_GPU(params, off_col, d_total, h_total, sg, streams[sg]);
            cgp->switch_device_fact(off_col, h_off_col, d_total, h_total, sg, 1, 0, streams[sg]);
            if (qo->groupby_build.size() == 0) cgp->call_aggregation_CPU(params, h_off_col[0], h_total);
            else cgp->call_group_by_CPU(params, h_off_col, h_total);

          } else if (qo->joinGPUPipelineCol[sg].size() > 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {

            if (qo->groupby_build.size() == 0) cgp->call_probe_aggr_GPU(params, off_col, h_total, sg, streams[sg]);
            else cgp->call_probe_group_by_GPU(params, off_col, h_total, sg, streams[sg]);

          } else if (qo->joinGPUPipelineCol[sg].size() == 0 && qo->groupbyGPUPipelineCol[sg].size() > 0) {
            assert(0);
          }
        }
      }


    }

    CubDebugExit(cudaStreamSynchronize(streams[sg]));
    CubDebugExit(cudaStreamDestroy(streams[sg]));

  });

  CubDebugExit(cudaDeviceSynchronize());
  
  int* resGPU = cm->customMalloc(params->total_val * 6);
  CubDebugExit(cudaMemcpy(resGPU, params->d_res, params->total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));
  merge(params->res, resGPU, params->total_val);
}

void
QueryProcessing::runOnDemand() {
  int tile_items = 128 * 4;
  int Batch_Size = 16;
  int sg = 0;

  for (int i = 0; i < qo->join.size(); i++) {
    int table_id = qo->join[i].second->table_id;

    if (qo->segment_group_count[table_id][sg] > 0) {

      int total_segment = qo->segment_group_count[table_id][sg];
      int total_batch = (total_segment + Batch_Size - 1)/Batch_Size;
      int last_batch;
      if (total_segment % Batch_Size == 0) last_batch = Batch_Size;
      else last_batch = total_segment % Batch_Size;

      cout << qo->join[i].second->column_name << endl;

      ColumnInfo* key = qo->join[i].second;
      ColumnInfo *val = NULL, *filter = NULL;

      if (qo->groupby_build.size() > 0 && qo->groupby_build[key].size() > 0) {
        val = qo->groupby_build[key][0];
      }

      if (qo->select_build[key].size() > 0) {
        filter = qo->select_build[key][0];
      }

      short* segment_group_ptr = qo->segment_group[table_id] + (sg * key->total_segment);

      for (int j = 0; j < total_batch; j++) {
        int batch_size;
        if (j == total_batch-1) batch_size = last_batch;
        else batch_size = Batch_Size;

        parallel_for(int(0), batch_size, [=](int batch){
        // for (int batch = 0; batch < batch_size; batch++) {
            CubDebugExit(cudaStreamCreate(&streams[batch]));

            int segment_number = (j * Batch_Size + batch);
            assert(segment_number < qo->segment_group_count[table_id][sg]);
            int segment_idx = segment_group_ptr[segment_number];
            assert(segment_idx < key->total_segment);

            int num_tuples;
            if (key->LEN % SEGMENT_SIZE != 0 && segment_idx == key->total_segment-1) {
              num_tuples = key->LEN % SEGMENT_SIZE;
            } else {
              num_tuples = SEGMENT_SIZE;
            }

            int* key_ptr = NULL, *val_ptr = NULL, *filter_ptr = NULL;

            key_ptr = cm->onDemandTransfer(key->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            if (val != NULL) val_ptr = cm->onDemandTransfer(val->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            if (filter != NULL) filter_ptr = cm->onDemandTransfer(filter->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);

            build_GPU<128,4><<<(num_tuples+ tile_items - 1)/tile_items, 128, 0, streams[batch]>>>(
              key_ptr, val_ptr, filter_ptr, params->compare1[filter], params->compare2[filter], params->mode[filter], num_tuples, 
              params->ht_GPU[key], params->dim_len[key], params->min_key[key], segment_idx);

            CHECK_ERROR_STREAM(streams[batch]);

            CubDebugExit(cudaStreamSynchronize(streams[batch]));
            CubDebugExit(cudaStreamDestroy(streams[batch]));
        });
        // }

        cm->resetOnDemand();
         
        CubDebugExit(cudaDeviceSynchronize());
      }
    }
  }

  if (qo->segment_group_count[0][sg] > 0) {

    int total_segment = qo->segment_group_count[0][sg];
    int total_batch = (total_segment + Batch_Size - 1)/Batch_Size;
    int last_batch;
    if (total_segment % Batch_Size == 0) last_batch = Batch_Size;
    else last_batch = total_segment % Batch_Size;

    if (qo->groupby_build.size() == 0) {

      ColumnInfo *pkey[4] = {}, *fkey[4] = {}, *aggr[2] = {}, *filter[2] = {};

      for (int i = 0; i < qo->select_probe[cm->lo_orderdate].size(); i++) {
        filter[i] = qo->select_probe[cm->lo_orderdate][i];
      }

      for (int i = 0; i < qo->join.size(); i++) {
        int table_id = qo->join[i].second->table_id;
        fkey[table_id - 1] = qo->join[i].first;
        pkey[table_id - 1] = qo->join[i].second;
      }

      for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
        aggr[i] = qo->aggregation[cm->lo_orderdate][i];
      }

      for (int j = 0; j < total_batch; j++) {
        int batch_size;
        if (j == total_batch-1) batch_size = last_batch;
        else batch_size = Batch_Size;

        short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

        parallel_for(int(0), batch_size, [=](int batch){

            CubDebugExit(cudaStreamCreate(&streams[batch]));
            int segment_number = (j * Batch_Size + batch);
            assert(segment_number < qo->segment_group_count[0][sg]);
            int segment_idx = segment_group_ptr[segment_number];
            assert(segment_idx < cm->lo_orderdate->total_segment);

            int* d_key[4] = {}, *d_aggr[2] = {}, *d_filter[2] = {};

            int num_tuples;
            if (cm->lo_orderdate->LEN % SEGMENT_SIZE != 0 && segment_idx == cm->lo_orderdate->total_segment-1) {
              num_tuples = cm->lo_orderdate->LEN % SEGMENT_SIZE;
            } else {
              num_tuples = SEGMENT_SIZE;
            }

            for (int i = 0; i < 2; i++) {
              if (filter[i] != NULL) d_filter[i] = cm->onDemandTransfer(filter[i]->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            }

            for (int i = 0; i < 4; i++) {
              if (fkey[i] != NULL) d_key[i] = cm->onDemandTransfer(fkey[i]->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            }

            for (int i = 0; i < 2; i++) {
              if (aggr[i] != NULL) d_aggr[i] = cm->onDemandTransfer(aggr[i]->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            }

            filter_probe_aggr_GPU<128, 4><<<(num_tuples + tile_items - 1)/tile_items, 128, 0, streams[batch]>>>(
              d_filter[0], d_filter[1], params->compare1[filter[0]], params->compare2[filter[0]], params->compare1[filter[1]], params->compare2[filter[1]], 
              d_key[0], d_key[1], d_key[2], d_key[3], 
              d_aggr[0], d_aggr[1], params->mode_group, num_tuples,
              params->ht_GPU[pkey[0]], params->dim_len[pkey[0]], params->ht_GPU[pkey[1]], params->dim_len[pkey[1]], params->ht_GPU[pkey[2]], params->dim_len[pkey[2]], params->ht_GPU[pkey[3]], params->dim_len[pkey[3]],
              params->min_key[pkey[0]], params->min_key[pkey[1]], params->min_key[pkey[2]], params->min_key[pkey[3]],
              params->d_res);

            CHECK_ERROR_STREAM(streams[batch]);

            CubDebugExit(cudaStreamSynchronize(streams[batch]));
            CubDebugExit(cudaStreamDestroy(streams[batch]));

        });
        cm->resetOnDemand();

      } 
    } else {

      ColumnInfo *pkey[4] = {}, *fkey[4] = {}, *aggr[2] = {};

      for (int i = 0; i < qo->join.size(); i++) {
        int table_id = qo->join[i].second->table_id;
        fkey[table_id - 1] = qo->join[i].first;
        pkey[table_id - 1] = qo->join[i].second;
      }

      for (int i = 0; i < qo->aggregation[cm->lo_orderdate].size(); i++) {
        aggr[i] = qo->aggregation[cm->lo_orderdate][i];
      }

      for (int j = 0; j < total_batch; j++) {

        int batch_size;
        if (j == total_batch-1) batch_size = last_batch;
        else batch_size = Batch_Size;

        short* segment_group_ptr = qo->segment_group[0] + (sg * cm->lo_orderdate->total_segment);

        // for (int batch = 0; batch < batch_size; batch++) {
        parallel_for(int(0), batch_size, [=](int batch){

            CubDebugExit(cudaStreamCreate(&streams[batch]));
            int segment_number = (j * Batch_Size + batch);
            assert(segment_number < qo->segment_group_count[0][sg]);
            int segment_idx = segment_group_ptr[segment_number];
            assert(segment_idx < cm->lo_orderdate->total_segment);

            int* d_key[4] = {}, *d_aggr[2] = {};

            int num_tuples;
            if (cm->lo_orderdate->LEN % SEGMENT_SIZE != 0 && segment_idx == cm->lo_orderdate->total_segment-1) {
              num_tuples = cm->lo_orderdate->LEN % SEGMENT_SIZE;
            } else {
              num_tuples = SEGMENT_SIZE;
            }

            for (int i = 0; i < 4; i++) {
              if (fkey[i] != NULL) d_key[i] = cm->onDemandTransfer(fkey[i]->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            }

            for (int i = 0; i < 2; i++) {
              if (aggr[i] != NULL) d_aggr[i] = cm->onDemandTransfer(aggr[i]->col_ptr + segment_idx * SEGMENT_SIZE, num_tuples, streams[batch]);
            }

            probe_group_by_GPU<128, 4><<<(num_tuples + tile_items - 1)/tile_items, 128, 0, streams[batch]>>>(
              d_key[0], d_key[1], d_key[2], d_key[3], 
              d_aggr[0], d_aggr[1], params->mode_group, num_tuples, 
              params->ht_GPU[pkey[0]], params->dim_len[pkey[0]], params->ht_GPU[pkey[1]], params->dim_len[pkey[1]], params->ht_GPU[pkey[2]], params->dim_len[pkey[2]], params->ht_GPU[pkey[3]], params->dim_len[pkey[3]],
              params->min_key[pkey[0]], params->min_key[pkey[1]], params->min_key[pkey[2]], params->min_key[pkey[3]],
              params->min_val[pkey[0]], params->min_val[pkey[1]], params->min_val[pkey[2]], params->min_val[pkey[3]],
              params->unique_val[pkey[0]], params->unique_val[pkey[1]], params->unique_val[pkey[2]], params->unique_val[pkey[3]],
              params->total_val, params->d_res);

            CHECK_ERROR_STREAM(streams[batch]);

            CubDebugExit(cudaStreamSynchronize(streams[batch]));
            CubDebugExit(cudaStreamDestroy(streams[batch]));

        });
        // }
        cm->resetOnDemand();

        CubDebugExit(cudaDeviceSynchronize());
      }
    }

    CubDebugExit(cudaMemcpy(params->res, params->d_res, params->total_val * 6 * sizeof(int), cudaMemcpyDeviceToHost));

  }
};


float
QueryProcessing::processOnDemand() {
  qo->parseQuery(query);
  updateStatsQuery();
  prepareQuery();

  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;

  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  runOnDemand();

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i< params->total_val; i++) {
    if (params->res[6*i+4] != 0) {
      cout << params->res[6*i] << " " << params->res[6*i+1] << " " << params->res[6*i+2] << " " << params->res[6*i+3] << " " << reinterpret_cast<unsigned long long*>(&params->res[6*i+4])[0]  << endl;
      res_count++;
    }
  }
  cout << "Res count = " << res_count << endl;
  cout << "Query Execution Time: " << time << endl;
  cout << endl;

  endQuery();

  return time;
};


float
QueryProcessing::processQuery() {

  // chrono::high_resolution_clock::time_point st, finish;
  // chrono::duration<double> diff;
  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;

  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  qo->parseQuery(query);

  updateStatsQuery();

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Query Optimization Time: " << time << endl;
  cout << endl;

  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  prepareQuery();

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Query Prepare Time: " << time << endl;
  cout << endl;

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
  cout << "Query Execution Time: " << time << endl;
  cout << endl;

  endQuery();

  return time;

};

float
QueryProcessing::processQuery2() {

  // chrono::high_resolution_clock::time_point st, finish;
  // chrono::duration<double> diff;
  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;

  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  qo->parseQuery(query);

  updateStatsQuery();

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Query Optimization Time: " << time << endl;
  cout << endl;

  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  prepareQuery();

  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);               // Wait until the completion of all device 
                                            // work preceding the most recent call to cudaEventRecord()
  cudaEventElapsedTime(&time, start, stop); // Saving the time measured

  cout << "Query Prepare Time: " << time << endl;
  cout << endl;

  // st = chrono::high_resolution_clock::now();
  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  runQuery2();

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
  cout << "Query Execution Time: " << time << endl;
  cout << endl;

  endQuery();

  return time;

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
  // cout << timestamp.count() * 1000 << endl;
  query_freq[query]++;

  for (int i = 0; i < qo->querySelectColumn.size(); i++) {
    cm->updateColumnFrequency(qo->querySelectColumn[i]);
    cm->updateColumnTimestamp(qo->querySelectColumn[i], timestamp.count());
    cm->updateColumnWeight(qo->querySelectColumn[i], query_freq[query], 1, params->selectivity[qo->querySelectColumn[i]]);
  }
  for (int i = 0; i < qo->queryBuildColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryBuildColumn[i]);
    cm->updateColumnTimestamp(qo->queryBuildColumn[i], timestamp.count());
    cm->updateColumnWeight(qo->queryBuildColumn[i], query_freq[query], 10, 1);
  }
  for (int i = 0; i < qo->queryProbeColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryProbeColumn[i]);
    cm->updateColumnTimestamp(qo->queryProbeColumn[i], timestamp.count());
    cm->updateColumnWeight(qo->queryProbeColumn[i], query_freq[query], 10, params->selectivity[qo->queryProbeColumn[i]]);
  }
  for (int i = 0; i < qo->queryGroupByColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryGroupByColumn[i]);
    cm->updateColumnTimestamp(qo->queryGroupByColumn[i], timestamp.count());
    cm->updateColumnWeight(qo->queryGroupByColumn[i], query_freq[query], 5, 1);
  }
  for (int i = 0; i < qo->queryAggrColumn.size(); i++) {
    cm->updateColumnFrequency(qo->queryAggrColumn[i]);
    cm->updateColumnTimestamp(qo->queryAggrColumn[i], timestamp.count());
    cm->updateColumnWeight(qo->queryAggrColumn[i], query_freq[query], 5, 1);
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

    // params->mode[cm->d_year] = 0;
    // params->compare1[cm->d_year] = 1993;
    params->mode[cm->d_year] = 1;
    params->compare1[cm->d_year] = 1993;
    params->compare2[cm->d_year] = 1993;
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

    // params->mode[cm->s_region] = 0;
    // params->compare1[cm->s_region] = 1;
    params->mode[cm->s_region] = 1;
    params->compare1[cm->s_region] = 1;
    params->compare2[cm->s_region] = 1;
    // params->mode[cm->p_category] = 0;
    // params->compare1[cm->p_category] = 1;
    params->mode[cm->p_category] = 1;
    params->compare1[cm->p_category] = 1;
    params->compare2[cm->p_category] = 1;
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

    // params->mode[cm->c_region] = 0;
    // params->compare1[cm->c_region] = 2;
    params->mode[cm->c_region] = 1;
    params->compare1[cm->c_region] = 2;
    params->compare2[cm->c_region] = 2;
    // params->mode[cm->s_region] = 0;
    // params->compare1[cm->s_region] = 2;
    params->mode[cm->s_region] = 1;
    params->compare1[cm->s_region] = 2;
    params->compare2[cm->s_region] = 2;
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

    // params->mode[cm->c_region] = 0;
    // params->compare1[cm->c_region] = 1;
    params->mode[cm->c_region] = 1;
    params->compare1[cm->c_region] = 1;
    params->compare2[cm->c_region] = 1;
    // params->mode[cm->s_region] = 0;
    // params->compare1[cm->s_region] = 1;
    params->mode[cm->s_region] = 1;
    params->compare1[cm->s_region] = 1;
    params->compare2[cm->s_region] = 1;
    params->mode[cm->p_mfgr] = 1;
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