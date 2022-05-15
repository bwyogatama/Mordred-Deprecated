#ifndef _COST_MODEL_H
#define _COST_MODEL_H

#define CACHE_LINE 64
#define BW_CPU 42000000
#define BW_PCI 12000000

#include "QueryOptimizer.h"

class CostModel {
public:

	double L;
	double ori_L;
	int sg;
	int table_id;
	int n_group_key;
	int n_aggr_key;
	int total_segment;

	vector<Operator*> opPipeline;

	vector<ColumnInfo*> selectCPU;
	vector<ColumnInfo*> joinCPU;
	vector<ColumnInfo*> groupCPU;
	vector<ColumnInfo*> buildCPU;

	vector<ColumnInfo*> selectGPU;
	vector<ColumnInfo*> joinGPU;
	vector<ColumnInfo*> groupGPU;
	vector<ColumnInfo*> buildGPU;

	QueryOptimizer* qo;

	CostModel(int _L, int _total_segment, int _n_group_key, int _n_aggr_key, int _sg, int _table_id, QueryOptimizer* _qo);
	void clear();
	void permute_cost();
	void permute_costHE();
	double calculate_cost();
	double probe_cost(double selectivity, bool mat_start, bool mat_end);
	double transfer_cost(int M = 2);
	double filter_cost(double selectivity, bool mat_start, bool mat_end);
	double group_cost(bool mat_start);
	double build_cost(bool mat_start);
};

#endif