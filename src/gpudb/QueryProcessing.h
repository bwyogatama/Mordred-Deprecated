#ifndef _QUERY_PROCESSING_H_
#define _QUERY_PROCESSING_H_

#include "CPUGPUProcessing.h"
#include "common.h"

extern int queries[13];

class QueryProcessing {
public:
  CacheManager* cm;
  QueryOptimizer* qo;
  CPUGPUProcessing* cgp;
  QueryParams* params;

  cudaStream_t streams[MAX_GROUPS];

  // map<int, int> query_freq;
  int query;
  bool verbose;
  bool custom;
  bool skipping;

  double logical_time;

  Distribution dist;

  QueryProcessing(CPUGPUProcessing* _cgp, bool _verbose, Distribution _dist = None) {
    cgp = _cgp;
    qo = cgp->qo;
    cm = cgp->cm;
    // query_freq[_query] = 0;
    verbose = _verbose;
    dist = _dist;
    custom = cgp->custom;
    skipping = cgp->skipping;
    logical_time = 0;
  }

  ~QueryProcessing() {
    // query_freq.clear();
  }

  void generate_rand_query() {
    query = queries[rand() % NUM_QUERIES];
    cout << query << endl;
    // query_freq[query] = 0;
  }

  void setQuery(int _query) {
    query = _query;
  }

  void runQuery(CUcontext ctx = NULL);

  void runQuery2(CUcontext ctx = NULL);

  void runQueryNP(CUcontext ctx = NULL);

  void runQueryHE(CUcontext ctx = NULL);

  // void runQueryEMat(CUcontext ctx = NULL);

  void runOnDemand();

  void runHybridOnDemand(int options = 1);

  void endQuery();

  void updateStatsQuery();

  double processQuery(CUcontext ctx = NULL);

  double processQuery2(CUcontext ctx = NULL);

  double processQueryNP(CUcontext ctx = NULL);

  double processQueryEMat(CUcontext ctx = NULL);

  double processQueryHE(CUcontext ctx = NULL);

  double processOnDemand();

  double processHybridOnDemand(int options = 1);

  void profile();

  void percentageData();

  void dumpTrace(string filename);

  void countTouchedSegment(int table_id, int* t_segment, int* t_c_segment);

  void executeTableDim(int table_id, int sg);

  void executeTableDim_HE(int table_id, int sg);

  void executeTableFact_v1(int sg);

  void executeTableFact_v2(int sg);

  void executeTableFact_HE(int sg);

  void executeTableDimOD(int table_id, int sg);

  void executeTableFactOD(int sg);

  void executeTableFactOD2(int sg);

  void executeTableDimNP(int table_id, int sg);

  void executeTableFactNP(int sg);


};

#endif