#ifndef _QUERY_OPTIMIZER_H_
#define _QUERY_OPTIMIZER_H_

#include "CacheManager.h"

#define NUM_QUERIES 13
#define MAX_GROUPS 64

class CacheManager;
class ColumnInfo;

enum OperatorType {
    Filter, Probe, Build, GroupBy, Aggr, CPUtoGPU, GPUtoCPU, Materialize, Merge
};

enum DeviceType {
    CPU, GPU
};

class Operator {
public:
	DeviceType device;
	int table_id;
	OperatorType type;
	unsigned short sg;
	short* segment_group;
	Operator* children;
	Operator* parents;

	vector<ColumnInfo*> columns;
	vector<ColumnInfo*> supporting_columns;

	Operator(DeviceType _device, unsigned short _sg, int _table_id, OperatorType _type) {
		type = _type;
		sg = _sg;
		table_id = _table_id;
		device = _device;
	};
	void addChild(Operator* child) {
		children = child;
		child->parents = this;
	};
	void setDevice(DeviceType _device) {
		device = _device;
	};

};

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
	unordered_map<ColumnInfo*, ColumnInfo*> pkey_fkey;

	vector<vector<vector<vector<Operator*>>>> opGPUPipeline; // for each table, for each segment group, for each pipeline, there is vector of operator
	vector<vector<vector<vector<Operator*>>>> opCPUPipeline; // for each table, for each segment group, for each pipeline, there is vector of operator

	vector<vector<Operator*>> opRoots; // for each table, for each segment group there is operator

	vector<vector<Operator*>> opParsed; // for each table, there is vector of operator

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

	map<int, map<ColumnInfo*, double>> speedup;

	QueryOptimizer(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize);
	~QueryOptimizer();

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

	void clearParsing();
	void clearPlacement();

	void dataDrivenOperatorPlacement();
	void groupBitmap();

};

QueryOptimizer::QueryOptimizer(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize) {
	cm = new CacheManager(_cache_size, _ondemand_size, _processing_size, _pinned_memsize);
	fkey_pkey[cm->lo_orderdate] = cm->d_datekey;
	fkey_pkey[cm->lo_partkey] = cm->p_partkey;
	fkey_pkey[cm->lo_custkey] = cm->c_custkey;
	fkey_pkey[cm->lo_suppkey] = cm->s_suppkey;
	pkey_fkey[cm->d_datekey] = cm->lo_orderdate;
	pkey_fkey[cm->p_partkey] = cm->lo_partkey;
	pkey_fkey[cm->c_custkey] = cm->lo_custkey;
	pkey_fkey[cm->s_suppkey] = cm->lo_suppkey;
}

QueryOptimizer::~QueryOptimizer() {
	fkey_pkey.clear();
	pkey_fkey.clear();
	delete cm;
}

void 
QueryOptimizer::parseQuery(int query) {

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
QueryOptimizer::clearPlacement() {

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

	joinCPUPipelineCol.clear(); //vector
	joinGPUPipelineCol.clear();
	selectCPUPipelineCol.clear(); //vector
	selectGPUPipelineCol.clear();
	groupbyGPUPipelineCol.clear();
	groupbyCPUPipelineCol.clear();

	opCPUPipeline.clear();
	opGPUPipeline.clear();
	opRoots.clear();
}

void
QueryOptimizer::clearParsing() {

	join.clear();
	aggregation.clear();
	groupby_build.clear();
	select_probe.clear();
	select_build.clear();

	querySelectColumn.clear();
	queryBuildColumn.clear();
	queryProbeColumn.clear();
	queryGroupByColumn.clear();
	queryAggrColumn.clear();

	opParsed.clear();
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator (CPU, 0, 0, Filter);
	op->columns.push_back(cm->lo_discount);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Filter);
	op->columns.push_back(cm->lo_quantity);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Aggr);
	op->columns.push_back(cm->lo_extendedprice);
	op->columns.push_back(cm->lo_discount);
	opParsed[0].push_back(op);


	op = new Operator (CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_year);
	opParsed[4].push_back(op);
	op = new Operator (CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);

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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator (CPU, 0, 0, Filter);
	op->columns.push_back(cm->lo_discount);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Filter);
	op->columns.push_back(cm->lo_quantity);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Aggr);
	op->columns.push_back(cm->lo_extendedprice);
	op->columns.push_back(cm->lo_discount);
	opParsed[0].push_back(op);


	op = new Operator (CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_yearmonthnum);
	opParsed[4].push_back(op);
	op = new Operator (CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);


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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator (CPU, 0, 0, Filter);
	op->columns.push_back(cm->lo_discount);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Filter);
	op->columns.push_back(cm->lo_quantity);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Aggr);
	op->columns.push_back(cm->lo_extendedprice);
	op->columns.push_back(cm->lo_discount);
	opParsed[0].push_back(op);


	op = new Operator (CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_datekey);
	opParsed[4].push_back(op);
	op = new Operator (CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);

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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator (CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_partkey);
	op->supporting_columns.push_back(cm->p_partkey);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator (CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->d_year);
	op->supporting_columns.push_back(cm->p_brand1);
	opParsed[0].push_back(op);

	op = new Operator (CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_region);
	opParsed[1].push_back(op);
	op = new Operator (CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator (CPU, 0, 3, Filter);
	op->columns.push_back(cm->p_category);
	opParsed[3].push_back(op);
	op = new Operator (CPU, 0, 3, Build);
	op->columns.push_back(cm->p_partkey);
	op->supporting_columns.push_back(cm->lo_partkey);
	opParsed[3].push_back(op);

	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_partkey);
	op->supporting_columns.push_back(cm->p_partkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->d_year);
	op->supporting_columns.push_back(cm->p_brand1);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_region);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 3, Filter);
	op->columns.push_back(cm->p_brand1);
	opParsed[3].push_back(op);
	op = new Operator(CPU, 0, 3, Build);
	op->columns.push_back(cm->p_partkey);
	op->supporting_columns.push_back(cm->lo_partkey);
	opParsed[3].push_back(op);

	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_partkey);
	op->supporting_columns.push_back(cm->p_partkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->d_year);
	op->supporting_columns.push_back(cm->p_brand1);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_region);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 3, Filter);
	op->columns.push_back(cm->p_brand1);
	opParsed[3].push_back(op);
	op = new Operator(CPU, 0, 3, Build);
	op->columns.push_back(cm->p_partkey);
	op->supporting_columns.push_back(cm->lo_partkey);
	opParsed[3].push_back(op);

	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->c_nation);
	op->supporting_columns.push_back(cm->s_nation);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_region);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_region);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_year);
	opParsed[4].push_back(op);
	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->c_city);
	op->supporting_columns.push_back(cm->s_city);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_nation);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_nation);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_year);
	opParsed[4].push_back(op);
	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->c_city);
	op->supporting_columns.push_back(cm->s_city);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_city);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_city);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_year);
	opParsed[4].push_back(op);
	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->supporting_columns.push_back(cm->c_city);
	op->supporting_columns.push_back(cm->s_city);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_city);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_city);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_yearmonthnum);
	opParsed[4].push_back(op);
	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_partkey);
	op->supporting_columns.push_back(cm->p_partkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->columns.push_back(cm->lo_supplycost);
	op->supporting_columns.push_back(cm->c_nation);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_region);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_region);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 3, Filter);
	op->columns.push_back(cm->p_mfgr);
	opParsed[3].push_back(op);
	op = new Operator(CPU, 0, 3, Build);
	op->columns.push_back(cm->p_partkey);
	op->supporting_columns.push_back(cm->lo_partkey);
	opParsed[3].push_back(op);

	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_partkey);
	op->supporting_columns.push_back(cm->p_partkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->columns.push_back(cm->lo_supplycost);
	op->supporting_columns.push_back(cm->p_category);
	op->supporting_columns.push_back(cm->s_nation);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_region);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_region);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 3, Filter);
	op->columns.push_back(cm->p_mfgr);
	opParsed[3].push_back(op);
	op = new Operator(CPU, 0, 3, Build);
	op->columns.push_back(cm->p_partkey);
	op->supporting_columns.push_back(cm->lo_partkey);
	opParsed[3].push_back(op);

	op = new Operator(CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_year);
	opParsed[4].push_back(op);
	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
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

	// dataDrivenOperatorPlacement();

	opParsed.resize(cm->TOT_TABLE);

	Operator* op;
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_suppkey);
	op->supporting_columns.push_back(cm->s_suppkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_custkey);
	op->supporting_columns.push_back(cm->c_custkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_partkey);
	op->supporting_columns.push_back(cm->p_partkey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, Probe);
	op->columns.push_back(cm->lo_orderdate);
	op->supporting_columns.push_back(cm->d_datekey);
	opParsed[0].push_back(op);
	op = new Operator(CPU, 0, 0, GroupBy);
	op->columns.push_back(cm->lo_revenue);
	op->columns.push_back(cm->lo_supplycost);
	op->supporting_columns.push_back(cm->p_brand1);
	op->supporting_columns.push_back(cm->s_city);
	op->supporting_columns.push_back(cm->d_year);
	opParsed[0].push_back(op);

	op = new Operator(CPU, 0, 1, Filter);
	op->columns.push_back(cm->s_nation);
	opParsed[1].push_back(op);
	op = new Operator(CPU, 0, 1, Build);
	op->columns.push_back(cm->s_suppkey);
	op->supporting_columns.push_back(cm->lo_suppkey);
	opParsed[1].push_back(op);

	op = new Operator(CPU, 0, 2, Filter);
	op->columns.push_back(cm->c_region);
	opParsed[2].push_back(op);
	op = new Operator(CPU, 0, 2, Build);
	op->columns.push_back(cm->c_custkey);
	op->supporting_columns.push_back(cm->lo_custkey);
	opParsed[2].push_back(op);

	op = new Operator(CPU, 0, 3, Filter);
	op->columns.push_back(cm->p_category);
	opParsed[3].push_back(op);
	op = new Operator(CPU, 0, 3, Build);
	op->columns.push_back(cm->p_partkey);
	op->supporting_columns.push_back(cm->lo_partkey);
	opParsed[3].push_back(op);

	op = new Operator(CPU, 0, 4, Filter);
	op->columns.push_back(cm->d_year);
	opParsed[4].push_back(op);
	op = new Operator(CPU, 0, 4, Build);
	op->columns.push_back(cm->d_datekey);
	op->supporting_columns.push_back(cm->lo_orderdate);
	opParsed[4].push_back(op);
}

// 

void
QueryOptimizer::dataDrivenOperatorPlacement() {

	opRoots.resize(cm->TOT_TABLE);
	for (int i = 0; i < cm->TOT_TABLE; i++) opRoots[i].resize(MAX_GROUPS);
	opCPUPipeline.resize(cm->TOT_TABLE);
	for (int i = 0; i < cm->TOT_TABLE; i++) {
		opCPUPipeline[i].resize(MAX_GROUPS);
		for (int j = 0; j < MAX_GROUPS; j++) {
			opCPUPipeline[i][j].resize(1);
		}
	}
	opGPUPipeline.resize(cm->TOT_TABLE);
	for (int i = 0; i < cm->TOT_TABLE; i++) {
		opGPUPipeline[i].resize(MAX_GROUPS);
		for (int j = 0; j < MAX_GROUPS; j++) {
			opGPUPipeline[i][j].resize(1);
		}
	}

	selectGPUPipelineCol.resize(MAX_GROUPS);
	selectCPUPipelineCol.resize(MAX_GROUPS);
	joinGPUPipelineCol.resize(MAX_GROUPS);
	joinCPUPipelineCol.resize(MAX_GROUPS);
	groupbyGPUPipelineCol.resize(MAX_GROUPS);
	groupbyCPUPipelineCol.resize(MAX_GROUPS);

	joinGPUcheck = (bool*) malloc(cm->TOT_TABLE * sizeof(bool));
	joinCPUcheck = (bool*) malloc(cm->TOT_TABLE * sizeof(bool));
	joinGPU = (bool**) malloc(cm->TOT_TABLE * sizeof(bool*));
	joinCPU = (bool**) malloc(cm->TOT_TABLE * sizeof(bool*));

	segment_group = (short**) malloc (cm->TOT_TABLE * sizeof(short*)); //4 tables, 64 possible segment group
	segment_group_count = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	par_segment = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	for (int i = 0; i < cm->TOT_TABLE; i++) {
		CubDebugExit(cudaHostAlloc((void**) &(segment_group[i]), MAX_GROUPS * cm->lo_orderdate->total_segment * sizeof(short), cudaHostAllocDefault));
		segment_group_count[i] = (short*) malloc (MAX_GROUPS * sizeof(short));
		par_segment[i] = (short*) malloc (MAX_GROUPS * sizeof(short));
		joinGPU[i] = (bool*) malloc(MAX_GROUPS * sizeof(bool));
		joinCPU[i] = (bool*) malloc(MAX_GROUPS * sizeof(bool));
		memset(joinGPU[i], 0, MAX_GROUPS * sizeof(bool));
		memset(joinCPU[i], 0, MAX_GROUPS * sizeof(bool));
		memset(segment_group_count[i], 0, MAX_GROUPS * sizeof(short));
		memset(par_segment[i], 0, MAX_GROUPS * sizeof(short));
	}

	last_segment = new int[cm->TOT_TABLE];
	par_segment_count = new short[cm->TOT_TABLE];
	memset(par_segment_count, 0, cm->TOT_TABLE * sizeof(short));

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
		int table_id = cm->lo_orderdate->table_id;

		for (int j = 0; j < opParsed[table_id].size(); j++) {
			Operator* op = opParsed[table_id][j];
			temp = temp << op->columns.size();
			for (int k = 0; k < op->columns.size(); k++) {
				ColumnInfo* column = op->columns[k];
				bool isGPU = cm->segment_bitmap[column->column_id][i];
				temp = temp | (isGPU << k);
			}
		}

		segment_group[table_id][temp * cm->lo_orderdate->total_segment + segment_group_count[table_id][temp]] = i;
		segment_group_count[table_id][temp]++;

		if (i == cm->lo_orderdate->total_segment - 1) {
			if (cm->lo_orderdate->LEN % SEGMENT_SIZE != 0) {
				last_segment[table_id] = temp;
			}
		}
	}

	for (int i = 0; i < join.size(); i++) {

		int table_id = join[i].second->table_id;
		
		for (int j = 0; j < join[i].second->total_segment; j++) {
			unsigned short temp = 0;

			for (int k = 0; k < opParsed[table_id].size(); k++) {
				Operator* op = opParsed[table_id][k];
				temp = temp << op->columns.size();
				for (int l = 0; l < op->columns.size(); l++) {
					ColumnInfo* column = op->columns[l];
					bool isGPU = cm->segment_bitmap[column->column_id][j];
					temp = temp | (isGPU << l);
				}
			}

			segment_group[table_id][temp * join[i].second->total_segment + segment_group_count[table_id][temp]] = j;
			segment_group_count[table_id][temp]++;

			if (j == join[i].second->total_segment - 1) {
				if (join[i].second->LEN % SEGMENT_SIZE != 0) {
					last_segment[table_id] = temp;
				}
			}

		}
	}

	for (unsigned short i = 0; i < MAX_GROUPS; i++) { //64 segment groups
		int table_id = cm->lo_orderdate->table_id;
		if (segment_group_count[table_id][i] > 0) {

			unsigned short sg = i;

			for (int j = opParsed[table_id].size()-1; j >= 0; j--) {

				Operator* op = opParsed[table_id][j];
				unsigned short  bit = 1;
				for (int k = 0; k < op->columns.size(); k++) {
					bit = (sg & (1 << k)) >> k;
					if (!bit) break;
				}

				if (op->type == GroupBy) {
					// cout << bit << " " << groupGPUcheck << endl;
					(bit & groupGPUcheck) ? (op->device = GPU):(op->device = CPU);
				} else if (op->type == Aggr) {
					(bit) ? (op->device = GPU):(op->device = CPU); 		
				} else if (op->type == Probe) {	
					(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (op->device = GPU):(op->device = CPU);
					(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (joinGPU[op->supporting_columns[0]->table_id][i] = 1):(joinCPU[op->supporting_columns[0]->table_id][i] = 1); 			
				} else if (op->type == Filter) {
					(bit) ? (op->device = GPU):(op->device = CPU);
				} else if (op->type == Build) {
					(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (op->device = GPU):(op->device = CPU);
					(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (joinGPU[op->supporting_columns[0]->table_id][i] = 1):(joinCPU[op->supporting_columns[0]->table_id][i] = 1);					
				}

				sg = sg >> op->columns.size();
			}

			for (int j = 0; j < opParsed[table_id].size(); j++) {
				Operator* op = opParsed[table_id][j];
				// cout << op->type << endl;
				if (op->type != Aggr && op->type != GroupBy && op->type != Build) {
					if (op->device == GPU) {
						opGPUPipeline[table_id][i][0].push_back(op);
					} else if (op->device == CPU) {
						opCPUPipeline[table_id][i][0].push_back(op);
					}
				} else if (op->type == GroupBy || op->type == Aggr) {
					if (opCPUPipeline[table_id][i][0].size() > 0) {
						// cout << "ho" << endl;
						opCPUPipeline[table_id][i][0].push_back(op);
					} else {
						// cout << "hi" << endl;
						if (op->device == GPU) opGPUPipeline[table_id][i][0].push_back(op);
						else if (op->device == CPU) opCPUPipeline[table_id][i][0].push_back(op);
					}
				} else if (op->type == Build) { //TODO!! FIX THIS
					if (opCPUPipeline[table_id][i][0].size() > 0) opCPUPipeline[table_id][i][0].push_back(op);
					else {
						if (op->device == GPU) opGPUPipeline[table_id][i][0].push_back(op);
						else if (op->device == CPU) opCPUPipeline[table_id][i][0].push_back(op);
					}					
				}
			}

			Operator* op;

			if (opGPUPipeline[table_id][i][0].size() > 0) {
				opRoots[table_id][i] = opGPUPipeline[table_id][i][0][0];
				op = opRoots[table_id][i];
				for (int j = 1; j < opGPUPipeline[table_id][i][0].size(); j++) {
					op->addChild(opGPUPipeline[table_id][i][0][j]);
					op = opGPUPipeline[table_id][i][0][j];
				}
				if (opCPUPipeline[table_id][i][0].size() > 0) {
					Operator* transferOp = new Operator(GPU, i, table_id, GPUtoCPU);
					op->addChild(transferOp);
					Operator* matOp = new Operator(CPU, i, table_id, Materialize);
					transferOp->addChild(matOp);
					op = matOp;
					for (int j = 1; j < opCPUPipeline[table_id][i][0].size(); j++) {
						op->addChild(opCPUPipeline[table_id][i][0][j]);
						op = opCPUPipeline[table_id][i][0][j];
					}
				}
			} else {
				opRoots[table_id][i] = opCPUPipeline[table_id][i][0][0];
				op = opRoots[table_id][i];
				for (int j = 1; j < opCPUPipeline[table_id][i][0].size(); j++) {
					op->addChild(opCPUPipeline[table_id][i][0][j]);
					op = opCPUPipeline[table_id][i][0][j];
				}
			}
		}
	}

	for (int dim = 0; dim < join.size(); dim++) {
		int table_id = join[dim].second->table_id;
			for (unsigned short i = 0; i < MAX_GROUPS; i++) { //64 segment groups
				if (segment_group_count[table_id][i] > 0) {

					unsigned short sg = i;

					for (int j = opParsed[table_id].size()-1; j >= 0; j--) {

						Operator* op = opParsed[table_id][j];
						unsigned short  bit = 1;
						for (int k = 0; k < op->columns.size(); k++) {
							bit = (sg & (1 << k)) >> k;
							if (!bit) break;
						}

						if (op->type == GroupBy) {
							(bit & groupGPUcheck) ? (op->device = GPU):(op->device = CPU);
						} else if (op->type == Aggr) {
							(bit) ? (op->device = GPU):(op->device = CPU); 		
						} else if (op->type == Probe) {	
							(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (op->device = GPU):(op->device = CPU);
							(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (joinGPU[op->supporting_columns[0]->table_id][i] = 1):(joinCPU[op->supporting_columns[0]->table_id][i] = 1); 			
						} else if (op->type == Filter) {
							(bit) ? (op->device = GPU):(op->device = CPU);
						} else if (op->type == Build) {
							(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (op->device = GPU):(op->device = CPU);
							(bit & joinGPUcheck[op->supporting_columns[0]->table_id]) ? (joinGPU[op->supporting_columns[0]->table_id][i] = 1):(joinCPU[op->supporting_columns[0]->table_id][i] = 1);					
						}

						sg = sg >> op->columns.size();
					}

					for (int j = 0; j < opParsed[table_id].size(); j++) {
						Operator* op = opParsed[table_id][j];
						if (op->type != Aggr && op->type != GroupBy && op->type != Build) {
							if (op->device == GPU) {
								opGPUPipeline[table_id][i][0].push_back(op);
							} else if (op->device == CPU) {
								opCPUPipeline[table_id][i][0].push_back(op);	
							}
						} else if (op->type == GroupBy || op->type == Aggr) {
							if (opCPUPipeline[table_id][i][0].size() > 0) opCPUPipeline[table_id][i][0].push_back(op);
							else {
								if (op->device == GPU) opGPUPipeline[table_id][i][0].push_back(op);
								else if (op->device == CPU) opCPUPipeline[table_id][i][0].push_back(op);
							}
						} else if (op->type == Build) { //TODO!! FIX THIS
							if (opCPUPipeline[table_id][i][0].size() > 0) opCPUPipeline[table_id][i][0].push_back(op);
							else {
								if (op->device == GPU) opGPUPipeline[table_id][i][0].push_back(op);
								else if (op->device == CPU) opCPUPipeline[table_id][i][0].push_back(op);
							}					
						}
					}

					Operator* op;

					if (opGPUPipeline[table_id][i][0].size() > 0) {
						opRoots[table_id][i] = opGPUPipeline[table_id][i][0][0];
						op = opRoots[table_id][i];
						for (int j = 1; j < opGPUPipeline[table_id][i][0].size(); j++) {
							op->addChild(opGPUPipeline[table_id][i][0][j]);
							op = opGPUPipeline[table_id][i][0][j];
						}
						if (opCPUPipeline[table_id][i][0].size() > 0) {
							Operator* transferOp = new Operator(GPU, i, table_id, GPUtoCPU);
							op->addChild(transferOp);
							Operator* matOp = new Operator(CPU, i, table_id, Materialize);
							transferOp->addChild(matOp);
							op = matOp;
							for (int j = 1; j < opCPUPipeline[table_id][i][0].size(); j++) {
								op->addChild(opCPUPipeline[table_id][i][0][j]);
								op = opCPUPipeline[table_id][i][0][j];
							}
						}
					} else {
						opRoots[table_id][i] = opCPUPipeline[table_id][i][0][0];
						op = opRoots[table_id][i];
						for (int j = 1; j < opCPUPipeline[table_id][i][0].size(); j++) {
							op->addChild(opCPUPipeline[table_id][i][0][j]);
							op = opCPUPipeline[table_id][i][0][j];
						}
					}
				}
			}
	}

	for (int i = 0; i < MAX_GROUPS; i++) {
		if (segment_group_count[0][i] > 0) {
			// cout << i << endl;
			for (int j = 0; j < opGPUPipeline[0][i][0].size(); j++) {
				Operator* op = opGPUPipeline[0][i][0][j];
				// for (int col = 0; col < op->columns.size(); col++)
				// 	cout << "GPU " << op->columns[col]->column_name << endl;
				if (op->type == Probe) joinGPUPipelineCol[i].push_back(op->columns[0]);
				else if (op->type == Filter) selectGPUPipelineCol[i].push_back(op->columns[0]);
				else if (op->type == GroupBy || op->type == Aggr) {
					for (int k = 0; k < op->columns.size(); k++)
						groupbyGPUPipelineCol[i].push_back(op->columns[k]);
				}
			}
			for (int j = 0; j < opCPUPipeline[0][i][0].size(); j++) {
				Operator* op = opCPUPipeline[0][i][0][j];
				// for (int col = 0; col < op->columns.size(); col++)
				// 	cout << "CPU " << op->columns[col]->column_name << endl;
				if (op->type == Probe) joinCPUPipelineCol[i].push_back(op->columns[0]);
				else if (op->type == Filter) selectCPUPipelineCol[i].push_back(op->columns[0]);
				else if (op->type == GroupBy || op->type == Aggr) {
					for (int k = 0; k < op->columns.size(); k++)
						groupbyCPUPipelineCol[i].push_back(op->columns[k]);
				}
			}
		}
	}

	for (int i = 0; i < cm->TOT_TABLE; i++) {
		bool checkGPU = false, checkCPU = false;
		for (int j = 0; j < MAX_GROUPS; j++) {
			if (joinGPU[i][j] && joinGPUcheck[i]) checkGPU = true;
			if (joinCPU[i][j]) checkCPU = true;
		}
		joinGPUcheck[i] = checkGPU;
		joinCPUcheck[i] = checkCPU;
		// cout << i << " check " << joinGPUcheck[i] << endl;
		// cout << i << " check " << joinCPUcheck[i] << endl;
	}


	// cout << "joinGPUPipelineCol" << endl;
	// for (int i = 0; i < MAX_GROUPS; i++) {
	// 	for (int j = 0; j < joinGPUPipelineCol[i].size(); j++) {
	// 		cout << joinGPUPipelineCol[i][j]->column_name << endl;
	// 	}
	// }

	// cout << "joinCPUPipelineCol" << endl;
	// for (int i = 0; i < MAX_GROUPS; i++) {
	// 	for (int j = 0; j < joinCPUPipelineCol[i].size(); j++) {
	// 		cout << joinCPUPipelineCol[i][j]->column_name << endl;
	// 	}
	// }

	// cout << "groupbyGPUPipelineCol" << endl;
	// for (int i = 0; i < MAX_GROUPS; i++) {
	// 	for (int j = 0; j < groupbyGPUPipelineCol[i].size(); j++) {
	// 		cout << groupbyGPUPipelineCol[i][j]->column_name << endl;
	// 	}
	// }

	// cout << "groupbyCPUPipelineCol" << endl;
	// for (int i = 0; i < MAX_GROUPS; i++) {
	// 	for (int j = 0; j < groupbyCPUPipelineCol[i].size(); j++) {
	// 		cout << groupbyCPUPipelineCol[i][j]->column_name << endl;
	// 	}
	// }

	// for (int i = 0; i < cm->TOT_TABLE; i++) {
	// 	for (int sg = 0; sg < MAX_GROUPS; sg++) {
	// 		if (segment_group_count[i][sg] > 0) {
	// 			cout << i << " " << sg << " " << segment_group_count[i][sg] << endl;
	// 		}
	// 	}
	// }


	for (int i = 0; i < cm->TOT_TABLE; i++) {
		short count = 0;
		for (int sg = 0; sg < MAX_GROUPS; sg++) {
			if (segment_group_count[i][sg] > 0) {
				par_segment[i][count] = sg;
				count++;
			}
		}
		par_segment_count[i] = count;
	}
}
#endif