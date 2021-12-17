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
  bool skew;
  bool custom;
  bool skipping;

  QueryProcessing(CPUGPUProcessing* _cgp, bool _verbose, bool _skew = false) {
    cgp = _cgp;
    qo = cgp->qo;
    cm = cgp->cm;
    // query_freq[_query] = 0;
    verbose = _verbose;
    skew = _skew;
    custom = cgp->custom;
    skipping = cgp->skipping;
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

  void runQuery();

  void runQuery2();

  void runQueryNP();

  void runOnDemand();

  void runHybridOnDemand(int options = 1);

  void endQuery();

  void updateStatsQuery();

  double processQuery();

  double processQuery2();

  double processQueryNP();

  double processOnDemand();

  double processHybridOnDemand(int options = 1);

  void profile();

  void percentageData();

  void dumpTrace(string filename);

  void countTouchedSegment(int table_id, int* t_segment, int* t_c_segment);

  void executeTableDim(int table_id, int sg);

  void executeTableFact_v1(int sg);

  void executeTableFact_v2(int sg);

  void executeTableDimOD(int table_id, int sg);

  void executeTableFactOD(int sg);

  void executeTableFactOD2(int sg);

  void executeTableDimNP(int table_id, int sg);

  void executeTableFactNP(int sg);

};

#endif