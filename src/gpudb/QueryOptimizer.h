#ifndef _QUERY_OPTIMIZER_H_
#define _QUERY_OPTIMIZER_H_

#include "CacheManager.h"

class CacheManager;
class ColumnInfo;

class QueryOptimizer {
public:
	CacheManager* cm;

	vector<ColumnInfo*> querySelectColumn;
	vector<ColumnInfo*> queryBuildColumn;
	vector<ColumnInfo*> queryProbeColumn;
	vector<ColumnInfo*> queryGroupByColumn;
	vector<ColumnInfo*> queryAggrColumn;

	vector<pair<ColumnInfo*, ColumnInfo*>> join;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> aggregation;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> groupby_build;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> select_probe;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> select_build;

	unordered_map<ColumnInfo*, ColumnInfo*> fkey_pkey;

	// vector<pair<int, int>> joinGPU;
	// vector<pair<int, int>> joinCPU;
	// pair<int, int> groupbyCPU;
	// pair<int, int> groupbyGPU;
	// vector<pair<int, int>> selectprobeGPU;
	// vector<pair<int, int>> selectprobeCPU;
	// map<ColumnInfo*, vector<pair<int, int>>> selectbuildGPU;
	// map<ColumnInfo*, vector<pair<int, int>>> selectbuildCPU;
	// map<ColumnInfo*, pair<int, int>> transfer;
	// vector<pair<int, int>> joinGPUPipeline;
	// vector<pair<int, int>> joinCPUPipeline;
	// vector<pair<int, int>> selectGPUPipeline;
	// vector<pair<int, int>> selectCPUPipeline;

	vector<vector<ColumnInfo*>> joinGPUPipelineCol;
	vector<vector<ColumnInfo*>> joinCPUPipelineCol;
	vector<vector<ColumnInfo*>> selectGPUPipelineCol;
	vector<vector<ColumnInfo*>> selectCPUPipelineCol;
	vector<vector<ColumnInfo*>> groupbyGPUPipelineCol;
	vector<vector<ColumnInfo*>> groupbyCPUPipelineCol;
	bool groupGPUcheck;
	bool* joinGPUcheck, *joinCPUcheck, **joinGPU, **joinCPU;

	short** segment_group;
	short** segment_group_count;
	short** par_segment;
	short* par_segment_count;
	int* last_segment;

	QueryOptimizer(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize);
	void parseQuery(int query);
	void parseQuery11();
	void parseQuery12();
	void parseQuery13();
	void parseQuery21();
	void parseQuery22();
	void parseQuery23();
	void parseQuery31();
	void parseQuery32();
	void parseQuery33();
	void parseQuery34();
	void parseQuery41();
	void parseQuery42();
	void parseQuery43();

	// void constructPipeline(vector<pair<int, int>>& CPUPipeline, vector<pair<int, int>>& GPUPipeline, 
	// 	vector<vector<ColumnInfo*>>& CPUPipelineCol, vector<vector<ColumnInfo*>>& GPUPipelineCol, multimap<int, ColumnInfo*>& temp, 
	// 	ColumnInfo* column, int N);

	void clearVector();

	void dataDrivenOperatorPlacement();
	void groupBitmap();

	// void latematerialization();
	// void latematerializationflex();
	// void patching();

};

QueryOptimizer::QueryOptimizer(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize) {
	cm = new CacheManager(_cache_size, _ondemand_size, _processing_size, _pinned_memsize);
	fkey_pkey[cm->lo_orderdate] = cm->d_datekey;
	fkey_pkey[cm->lo_partkey] = cm->p_partkey;
	fkey_pkey[cm->lo_custkey] = cm->c_custkey;
	fkey_pkey[cm->lo_suppkey] = cm->s_suppkey;
}

void 
QueryOptimizer::parseQuery(int query) {

	selectGPUPipelineCol.resize(64);
	selectCPUPipelineCol.resize(64);
	joinGPUPipelineCol.resize(64);
	joinCPUPipelineCol.resize(64);
	groupbyGPUPipelineCol.resize(64);
	groupbyCPUPipelineCol.resize(64);

	joinGPUcheck = (bool*) malloc(cm->TOT_TABLE * sizeof(bool));
	joinCPUcheck = (bool*) malloc(cm->TOT_TABLE * sizeof(bool));
	joinGPU = (bool**) malloc(cm->TOT_TABLE * sizeof(bool*));
	joinCPU = (bool**) malloc(cm->TOT_TABLE * sizeof(bool*));

	segment_group = (short**) malloc (cm->TOT_TABLE * sizeof(short*)); //4 tables, 64 possible segment group
	segment_group_count = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	par_segment = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	for (int i = 0; i < cm->TOT_TABLE; i++) {
		CubDebugExit(cudaHostAlloc((void**) &(segment_group[i]), 64 * cm->lo_orderdate->total_segment * sizeof(short), cudaHostAllocDefault));
		segment_group_count[i] = (short*) malloc (64 * sizeof(short));
		par_segment[i] = (short*) malloc (64 * sizeof(short));
		joinGPU[i] = (bool*) malloc(64 * sizeof(bool));
		joinCPU[i] = (bool*) malloc(64 * sizeof(bool));
		memset(joinGPU[i], 0, 64 * sizeof(bool));
		memset(joinCPU[i], 0, 64 * sizeof(bool));
		memset(segment_group_count[i], 0, 64 * sizeof(short));
		memset(par_segment[i], 0, 64 * sizeof(short));
	}

	last_segment = new int[cm->TOT_TABLE];
	par_segment_count = new short[cm->TOT_TABLE];
	memset(par_segment_count, 0, cm->TOT_TABLE * sizeof(short));

	if (query == 11) parseQuery11();
	else if (query == 12) parseQuery12();
	else if (query == 13) parseQuery13();
	else if (query == 21) parseQuery21();
	else if (query == 22) parseQuery22();
	else if (query == 23) parseQuery23();
	else if (query == 31) parseQuery31();
	else if (query == 32) parseQuery32();
	else if (query == 33) parseQuery33();
	else if (query == 34) parseQuery34();
	else if (query == 41) parseQuery41();
	else if (query == 42) parseQuery42();
	else if (query == 43) parseQuery43();
	else assert(0);
}

void
QueryOptimizer::clearVector() {

	for (int i = 0; i < cm->TOT_TABLE; i++) {
		CubDebugExit(cudaFreeHost(segment_group[i]));
		free(segment_group_count[i]);
		free(joinGPU[i]);
		free(joinCPU[i]);
	}

	free(segment_group);
	free(segment_group_count);
	free(joinGPUcheck);
	free(joinCPUcheck);
	free(joinGPU);
	free(joinCPU);

	join.clear();
	aggregation.clear();
	groupby_build.clear();
	select_probe.clear();
	select_build.clear();

	joinCPUPipelineCol.clear(); //vector
	joinGPUPipelineCol.clear();
	selectCPUPipelineCol.clear(); //vector
	selectGPUPipelineCol.clear();
	groupbyGPUPipelineCol.clear();
	groupbyCPUPipelineCol.clear();

	querySelectColumn.clear();
	queryBuildColumn.clear();
	queryProbeColumn.clear();
	queryGroupByColumn.clear();
	queryAggrColumn.clear();
}

void 
QueryOptimizer::parseQuery11() {

	querySelectColumn.push_back(cm->lo_discount);
	querySelectColumn.push_back(cm->lo_quantity);
	querySelectColumn.push_back(cm->d_year);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryAggrColumn.push_back(cm->lo_extendedprice);
	queryAggrColumn.push_back(cm->lo_discount);

	join.resize(1);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	aggregation[cm->lo_orderdate].push_back(cm->lo_extendedprice);
	aggregation[cm->lo_orderdate].push_back(cm->lo_discount);

	select_probe[cm->lo_orderdate].push_back(cm->lo_quantity);
	select_probe[cm->lo_orderdate].push_back(cm->lo_discount);

	select_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();

}

void 
QueryOptimizer::parseQuery12() {

	querySelectColumn.push_back(cm->lo_discount);
	querySelectColumn.push_back(cm->lo_quantity);
	querySelectColumn.push_back(cm->d_yearmonthnum);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryAggrColumn.push_back(cm->lo_extendedprice);
	queryAggrColumn.push_back(cm->lo_discount);

	join.resize(1);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	aggregation[cm->lo_orderdate].push_back(cm->lo_extendedprice);
	aggregation[cm->lo_orderdate].push_back(cm->lo_discount);

	select_probe[cm->lo_orderdate].push_back(cm->lo_quantity);
	select_probe[cm->lo_orderdate].push_back(cm->lo_discount);

	select_build[cm->d_datekey].push_back(cm->d_yearmonthnum);

	dataDrivenOperatorPlacement();

}

void 
QueryOptimizer::parseQuery13() {

	querySelectColumn.push_back(cm->lo_discount);
	querySelectColumn.push_back(cm->lo_quantity);
	querySelectColumn.push_back(cm->d_datekey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryAggrColumn.push_back(cm->lo_extendedprice);
	queryAggrColumn.push_back(cm->lo_discount);

	join.resize(1);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	aggregation[cm->lo_orderdate].push_back(cm->lo_extendedprice);
	aggregation[cm->lo_orderdate].push_back(cm->lo_discount);

	select_probe[cm->lo_orderdate].push_back(cm->lo_quantity);
	select_probe[cm->lo_orderdate].push_back(cm->lo_discount);

	select_build[cm->d_datekey].push_back(cm->d_datekey);

	dataDrivenOperatorPlacement();

}

void 
QueryOptimizer::parseQuery21() {

	querySelectColumn.push_back(cm->p_category);
	querySelectColumn.push_back(cm->s_region);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->p_partkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_partkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->p_brand1);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->s_suppkey].push_back(cm->s_region);
	select_build[cm->p_partkey].push_back(cm->p_category);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);
	groupby_build[cm->p_partkey].push_back(cm->p_brand1);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery22() {

	querySelectColumn.push_back(cm->p_brand1);
	querySelectColumn.push_back(cm->s_region);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->p_partkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_partkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->p_brand1);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->s_suppkey].push_back(cm->s_region);
	select_build[cm->p_partkey].push_back(cm->p_brand1);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);
	groupby_build[cm->p_partkey].push_back(cm->p_brand1);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery23() {

	querySelectColumn.push_back(cm->p_brand1);
	querySelectColumn.push_back(cm->s_region);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->p_partkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_partkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->p_brand1);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->s_suppkey].push_back(cm->s_region);
	select_build[cm->p_partkey].push_back(cm->p_brand1);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);
	groupby_build[cm->p_partkey].push_back(cm->p_brand1);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery31() {
	querySelectColumn.push_back(cm->d_year);
	querySelectColumn.push_back(cm->c_region);
	querySelectColumn.push_back(cm->s_region);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->c_nation);
	queryGroupByColumn.push_back(cm->s_nation);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->c_custkey].push_back(cm->c_region);
	select_build[cm->s_suppkey].push_back(cm->s_region);
	select_build[cm->d_datekey].push_back(cm->d_year);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);

	groupby_build[cm->c_custkey].push_back(cm->c_nation);
	groupby_build[cm->s_suppkey].push_back(cm->s_nation);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery32() {
	querySelectColumn.push_back(cm->d_year);
	querySelectColumn.push_back(cm->c_nation);
	querySelectColumn.push_back(cm->s_nation);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->c_city);
	queryGroupByColumn.push_back(cm->s_city);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->c_custkey].push_back(cm->c_nation);
	select_build[cm->s_suppkey].push_back(cm->s_nation);
	select_build[cm->d_datekey].push_back(cm->d_year);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);

	groupby_build[cm->c_custkey].push_back(cm->c_city);
	groupby_build[cm->s_suppkey].push_back(cm->s_city);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery33() {
	querySelectColumn.push_back(cm->d_year);
	querySelectColumn.push_back(cm->c_city);
	querySelectColumn.push_back(cm->s_city);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->c_city);
	queryGroupByColumn.push_back(cm->s_city);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->c_custkey].push_back(cm->c_city);
	select_build[cm->s_suppkey].push_back(cm->s_city);
	select_build[cm->d_datekey].push_back(cm->d_year);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);

	groupby_build[cm->c_custkey].push_back(cm->c_city);
	groupby_build[cm->s_suppkey].push_back(cm->s_city);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery34() {
	querySelectColumn.push_back(cm->d_yearmonthnum);
	querySelectColumn.push_back(cm->c_city);
	querySelectColumn.push_back(cm->s_city);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->c_city);
	queryGroupByColumn.push_back(cm->s_city);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(3);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->c_custkey].push_back(cm->c_city);
	select_build[cm->s_suppkey].push_back(cm->s_city);
	select_build[cm->d_datekey].push_back(cm->d_yearmonthnum);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);

	groupby_build[cm->c_custkey].push_back(cm->c_city);
	groupby_build[cm->s_suppkey].push_back(cm->s_city);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery41() {
	querySelectColumn.push_back(cm->p_mfgr);
	querySelectColumn.push_back(cm->c_region);
	querySelectColumn.push_back(cm->s_region);
	queryBuildColumn.push_back(cm->p_partkey);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_partkey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->c_nation);
	queryAggrColumn.push_back(cm->lo_supplycost);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(4);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[3] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->p_partkey].push_back(cm->p_mfgr);
	select_build[cm->c_custkey].push_back(cm->c_region);
	select_build[cm->s_suppkey].push_back(cm->s_region);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);
	aggregation[cm->lo_orderdate].push_back(cm->lo_supplycost);

	groupby_build[cm->c_custkey].push_back(cm->c_nation);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery42() {
	querySelectColumn.push_back(cm->p_mfgr);
	querySelectColumn.push_back(cm->c_region);
	querySelectColumn.push_back(cm->s_region);
	querySelectColumn.push_back(cm->d_year);
	queryBuildColumn.push_back(cm->p_partkey);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_partkey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->s_nation);
	queryGroupByColumn.push_back(cm->p_category);
	queryAggrColumn.push_back(cm->lo_supplycost);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(4);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[3] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->p_partkey].push_back(cm->p_mfgr);
	select_build[cm->c_custkey].push_back(cm->c_region);
	select_build[cm->s_suppkey].push_back(cm->s_region);
	select_build[cm->d_datekey].push_back(cm->d_year);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);
	aggregation[cm->lo_orderdate].push_back(cm->lo_supplycost);

	groupby_build[cm->s_suppkey].push_back(cm->s_nation);
	groupby_build[cm->p_partkey].push_back(cm->p_category);
	groupby_build[cm->d_datekey].push_back(cm->d_year);


	dataDrivenOperatorPlacement();
}

void 
QueryOptimizer::parseQuery43() {
	querySelectColumn.push_back(cm->p_category);
	querySelectColumn.push_back(cm->c_region);
	querySelectColumn.push_back(cm->s_nation);
	querySelectColumn.push_back(cm->d_year);
	queryBuildColumn.push_back(cm->p_partkey);
	queryBuildColumn.push_back(cm->s_suppkey);
	queryBuildColumn.push_back(cm->c_custkey);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_partkey);
	queryProbeColumn.push_back(cm->lo_suppkey);
	queryProbeColumn.push_back(cm->lo_custkey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->s_city);
	queryGroupByColumn.push_back(cm->p_brand1);
	queryAggrColumn.push_back(cm->lo_supplycost);
	queryAggrColumn.push_back(cm->lo_revenue);

	join.resize(4);
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[3] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->p_partkey].push_back(cm->p_category);
	select_build[cm->c_custkey].push_back(cm->c_region);
	select_build[cm->s_suppkey].push_back(cm->s_nation);
	select_build[cm->d_datekey].push_back(cm->d_year);

	aggregation[cm->lo_orderdate].push_back(cm->lo_revenue);
	aggregation[cm->lo_orderdate].push_back(cm->lo_supplycost);

	groupby_build[cm->s_suppkey].push_back(cm->s_city);
	groupby_build[cm->p_partkey].push_back(cm->p_brand1);
	groupby_build[cm->d_datekey].push_back(cm->d_year);


	dataDrivenOperatorPlacement();
}

// 

void
QueryOptimizer::dataDrivenOperatorPlacement() {

	groupGPUcheck = true;

	for (int i = 0; i < join.size(); i++) {
		if (groupby_build.size() > 0) {
			for (int j = 0; j < groupby_build[join[i].second].size(); j++) {
				int tot_seg_in_GPU = groupby_build[join[i].second][j]->tot_seg_in_GPU;
				int total_segment = groupby_build[join[i].second][j]->total_segment;
				if (tot_seg_in_GPU < total_segment) {
					groupGPUcheck = false;
					break;
				}
			}
		}
	}

	// printf("%zu\n", groupby_build.size());

	for (int i = 0; i < join.size(); i++) {
		if (join[i].second->tot_seg_in_GPU < join[i].second->total_segment) {
			joinGPUcheck[join[i].second->table_id] = false;
		} else {
			joinGPUcheck[join[i].second->table_id] = true;
		}
	}

	// for (int i = 0; i < cm->lo_orderdate->total_segment; i++) {
	// 	printf("%x\n", cm->segment_bitmap[cm->lo_partkey->column_id][i]);
	// }

	groupBitmap();
}

void
QueryOptimizer::groupBitmap() {

	for (int i = 0; i < cm->lo_orderdate->total_segment; i++) {
		unsigned short temp = 0;

		for (int j = 0; j < select_probe[cm->lo_orderdate].size(); j++) {
			ColumnInfo* column = select_probe[cm->lo_orderdate][j];
			bool isGPU = cm->segment_bitmap[column->column_id][i];
			//temp = temp | (isGPU << (select_probe[cm->lo_orderdate].size() - j - 1));
			temp = temp | (isGPU << j);
		}

		temp = temp << join.size();

		for (int j = 0; j < join.size(); j++) {
			bool isGPU = cm->segment_bitmap[join[j].first->column_id][i];
			//temp = temp | (isGPU << (join.size() - j - 1));
			temp = temp | (isGPU << j);
		}

		temp = temp << aggregation[cm->lo_orderdate].size();

		for (int j = 0; j < aggregation[cm->lo_orderdate].size(); j++) {
			ColumnInfo* column = aggregation[cm->lo_orderdate][j];
			bool isGPU = cm->segment_bitmap[column->column_id][i];
			//temp = temp | (isGPU << (join.size() - j - 1));
			temp = temp | (isGPU << j);
		}

		segment_group[cm->lo_orderdate->table_id][temp * cm->lo_orderdate->total_segment + segment_group_count[cm->lo_orderdate->table_id][temp]] = i;
		segment_group_count[cm->lo_orderdate->table_id][temp]++;
		//printf("temp = %d count = %d\n", temp, segment_group_count[0][temp]);

		if (i == cm->lo_orderdate->total_segment - 1) {
			if (cm->lo_orderdate->LEN % SEGMENT_SIZE != 0) {
				last_segment[0] = temp;
			}
		}
	}

	for (unsigned short i = 0; i < 64; i++) { //64 segment groups
		if (segment_group_count[cm->lo_orderdate->table_id][i] > 0) {

			unsigned short  bit = 1;
			unsigned short sg = i;
			for (int j = 0; j < aggregation[cm->lo_orderdate].size(); j++) {
				bit = (sg & (1 << j)) >> j;
				if (bit == 0) break;
			}

			for (int j = 0; j < aggregation[cm->lo_orderdate].size(); j++) {
				if (bit & groupGPUcheck) groupbyGPUPipelineCol[i].push_back(aggregation[cm->lo_orderdate][j]);
				else groupbyCPUPipelineCol[i].push_back(aggregation[cm->lo_orderdate][j]);
			}

			sg = sg >> aggregation[cm->lo_orderdate].size();

			// cout << sg << endl;

			for (int j = 0; j < join.size(); j++) {
				bit = (sg & (1 << j)) >> j;
				if (bit && joinGPUcheck[join[j].second->table_id]) {
					joinGPU[join[j].second->table_id][i] = 1;
					joinGPUPipelineCol[i].push_back(join[j].first);
				} else {
					joinCPU[join[j].second->table_id][i] = 1;
					joinCPUPipelineCol[i].push_back(join[j].first);
				}
			}

			// for (int k = 0; k < joinGPUPipelineCol[i].size(); k++) {
			// 	cout << joinGPUPipelineCol[i][k]->column_name << endl;
			// }

			sg = sg >> join.size();

			for (int j = 0; j < select_probe[cm->lo_orderdate].size(); j++) {
				bit = (sg & (1 << j)) >> j;
				if (bit) selectGPUPipelineCol[i].push_back(select_probe[cm->lo_orderdate][j]);
				else {
					//cout << select_probe[cm->lo_orderdate][j]->column_name << " " << i << endl;
					selectCPUPipelineCol[i].push_back(select_probe[cm->lo_orderdate][j]);
				}
			}

			sg = sg >> select_probe[cm->lo_orderdate].size();
		}
	}

	for (int i = 0; i < cm->TOT_TABLE; i++) {
		bool checkGPU = false, checkCPU = false;
		for (int j = 0; j < 64; j++) {
			if (joinGPU[i][j] && joinGPUcheck[i]) checkGPU = true;
			if (joinCPU[i][j]) checkCPU = true;
		}
		joinGPUcheck[i] = checkGPU;
		joinCPUcheck[i] = checkCPU;
	}

	for (int i = 0; i < join.size(); i++) {
		
		for (int j = 0; j < join[i].second->total_segment; j++) {
			unsigned short temp = 0;

			for (int k = 0; k < select_build[join[i].second].size(); k++) {

				ColumnInfo* column = select_build[join[i].second][k];
				bool isGPU = cm->segment_bitmap[column->column_id][j];
				//temp = temp | (isGPU << (select_build[join[i].second].size() - k - 1));
				temp = temp | (isGPU << k);

				// cout << select_build[join[i].second][k]->column_name << endl;
				// cout << temp << endl;
			}

			segment_group[join[i].second->table_id][temp * join[i].second->total_segment + segment_group_count[join[i].second->table_id][temp]] = j;
			segment_group_count[join[i].second->table_id][temp]++;

			if (j == join[i].second->total_segment - 1) {
				if (join[i].second->LEN % SEGMENT_SIZE != 0) {
					last_segment[join[i].second->table_id] = temp;
				}
			}

		}
	}


	for (int i = 0; i < cm->TOT_TABLE; i++) {
		short count = 0;
		for (int sg = 0; sg < 64; sg++) {
			if (segment_group_count[i][sg] > 0) {
				par_segment[i][count] = sg;
				count++;
			}
		}
		par_segment_count[i] = count;
	}
}

// void
// QueryOptimizer::patching() {
// 	int max = 0;

// 	for (int i = 0; i < aggregation[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = aggregation[cm->lo_orderdate][i]->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU > max) {
// 			max = tot_seg_in_GPU;
// 		}
// 	}

// 	for (int i = 0; i < join.size(); i++) {
// 		int tot_seg_in_GPU = join[i].first->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU > max) {
// 			max = tot_seg_in_GPU;
// 		}
// 	}

// 	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = select_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU > max) {
// 			max = tot_seg_in_GPU;
// 		}
// 	}

// 	for (int i = 0; i < aggregation[cm->lo_orderdate].size(); i++) {
// 		ColumnInfo* column = aggregation[cm->lo_orderdate][i];
// 		int tot_seg_in_GPU = column->tot_seg_in_GPU;
// 		assert(tot_seg_in_GPU <= max);
// 		if (tot_seg_in_GPU < max) {
// 			transfer[column] = pair<int, int> (tot_seg_in_GPU, max);
// 		}
// 	}

// 	for (int i = 0; i < join.size(); i++) {
// 		ColumnInfo* column = join[i].first;
// 		int tot_seg_in_GPU = column->tot_seg_in_GPU;
// 		assert(tot_seg_in_GPU <= max);
// 		if (tot_seg_in_GPU < max) {
// 			transfer[column] = pair<int, int> (tot_seg_in_GPU, max);
// 		}
// 	}

// 	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
// 		ColumnInfo* column = select_probe[cm->lo_orderdate][i];
// 		int tot_seg_in_GPU = column->tot_seg_in_GPU;
// 		assert(tot_seg_in_GPU <= max);
// 		if (tot_seg_in_GPU < max) {
// 			transfer[column] = pair<int, int> (tot_seg_in_GPU, max);
// 		}
// 	}


// 	for (int i = 0; i < join.size(); i++) {
// 		for (int j = 0; j < groupby_build[join[i].second].size(); j++) {
// 			ColumnInfo* column = groupby_build[join[i].second][j];
// 			int tot_seg_in_GPU = column->tot_seg_in_GPU;
// 			int total_segment = column->total_segment;
// 			assert(tot_seg_in_GPU <= total_segment);
// 			if (tot_seg_in_GPU < total_segment) {
// 				transfer[column] = pair<int, int> (tot_seg_in_GPU, total_segment);
// 			}
// 		}
// 	}


// 	for (int i = 0; i < join.size(); i++) {
// 		ColumnInfo* column = join[i].second;
// 		int tot_seg_in_GPU = column->tot_seg_in_GPU;
// 		int total_segment = column->total_segment;
// 		assert(tot_seg_in_GPU <= total_segment);
// 		if (tot_seg_in_GPU < total_segment) {
// 			transfer[column] = pair<int, int> (tot_seg_in_GPU, total_segment);
// 		}
// 	}
// }
#endif