#include "QueryOptimizer.h"
#include "CostModel.h"
#include "CacheManager.h"
#include "CPUGPUProcessing.h"

QueryOptimizer::QueryOptimizer(size_t _cache_size, size_t _processing_size, size_t _pinned_memsize, CPUGPUProcessing* _cgp) {
	cm = new CacheManager(_cache_size, _processing_size, _pinned_memsize);
	cgp = _cgp;
	custom = cgp->custom;
	skipping = cgp->skipping;
	fkey_pkey[cm->lo_orderdate] = cm->d_datekey;
	fkey_pkey[cm->lo_partkey] = cm->p_partkey;
	fkey_pkey[cm->lo_custkey] = cm->c_custkey;
	fkey_pkey[cm->lo_suppkey] = cm->s_suppkey;
	pkey_fkey[cm->d_datekey] = cm->lo_orderdate;
	pkey_fkey[cm->p_partkey] = cm->lo_partkey;
	pkey_fkey[cm->c_custkey] = cm->lo_custkey;
	pkey_fkey[cm->s_suppkey] = cm->lo_suppkey;

	speedup_segment = new double*[cm->TOT_COLUMN];
	for (int i = 0; i < cm->TOT_COLUMN; i++) {
		speedup_segment[i] = new double[cm->allColumn[i]->total_segment];
		memset(speedup_segment[i], 0, cm->allColumn[i]->total_segment * sizeof(double));
	}

	double alpha = 0.1;

	zipfian[11] = new Zipfian (7, 0, alpha);
	zipfian[12] = new Zipfian (79, 0, alpha);
	zipfian[13] = new Zipfian (316, 0, alpha);
	zipfian[21] = new Zipfian (7, 0, alpha);
	zipfian[22] = new Zipfian (6, 1, alpha);
	zipfian[23] = new Zipfian (4, 3, alpha);
	zipfian[31] = new Zipfian (7, 0, alpha);
	zipfian[32] = new Zipfian (6, 1, alpha);
	zipfian[33] = new Zipfian (4, 3, alpha);
	zipfian[34] = new Zipfian (79, 0, alpha);
	zipfian[41] = new Zipfian (7, 0, alpha);
	zipfian[42] = new Zipfian (6, 1, alpha);
	zipfian[43] = new Zipfian (4, 3, alpha);

	// cout << "hi" << endl;

	normal[11] = new Normal (3, 0.5, 0, 6);
	normal[12] = new Normal (39, 3, 0, 78);
	normal[13] = new Normal (157, 6, 0, 315);
	normal[21] = new Normal (3, 0.5, 0, 6);
	normal[22] = new Normal (3, 0.5, 0, 6);
	normal[23] = new Normal (3, 0.5, 0, 6);
	normal[31] = new Normal (3, 0.5, 0, 6);
	normal[32] = new Normal (3, 0.5, 0, 6);
	normal[33] = new Normal (3, 0.5, 0, 6);
	normal[34] = new Normal (39, 3, 0, 78);
	normal[41] = new Normal (3, 0.5, 0, 6);
	normal[42] = new Normal (3, 0.5, 0, 6);
	normal[43] = new Normal (3, 0.5, 0, 6);

	// cout << "hi" << endl;
	
}

QueryOptimizer::~QueryOptimizer() {
	fkey_pkey.clear();
	pkey_fkey.clear();
	delete cm;
	delete params;
}

void
QueryOptimizer::setDistributionZipfian(double alpha = 0.1) {
	zipfian[11]->reset(alpha);
	zipfian[12]->reset(alpha);
	zipfian[13]->reset(alpha);
	zipfian[21]->reset(alpha);
	zipfian[22]->reset(alpha);
	zipfian[23]->reset(alpha);
	zipfian[31]->reset(alpha);
	zipfian[32]->reset(alpha);
	zipfian[33]->reset(alpha);
	zipfian[34]->reset(alpha);
	zipfian[41]->reset(alpha);
	zipfian[42]->reset(alpha);
	zipfian[43]->reset(alpha);
}

//mean is from 0 to 6
void
QueryOptimizer::setDistributionNormal(double mean, double stddev) {
	normal[11]->reset(mean, stddev);
	normal[12]->reset(mean * 12, 3);
	normal[13]->reset(mean * 48, 6);
	normal[21]->reset(mean, stddev);
	normal[22]->reset(mean, stddev);
	normal[23]->reset(mean, stddev);
	normal[31]->reset(mean, stddev);
	normal[32]->reset(mean, stddev);
	normal[33]->reset(mean, stddev);
	normal[34]->reset(mean * 12, 3);
	normal[41]->reset(mean, stddev);
	normal[42]->reset(mean, stddev);
	normal[43]->reset(mean, stddev);
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
		free(par_segment[i]);
		free(joinGPU[i]);
		free(joinCPU[i]);
	}

	free(par_segment);
	free(par_segment_count);
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
	if (!index_to_sg.empty()) index_to_sg.clear();
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

	queryColumn.clear();

	opParsed.clear();
}

void 
QueryOptimizer::parseQuery11() {

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_discount);
	queryColumn[0].push_back(cm->lo_quantity);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_extendedprice);
	queryColumn[0].push_back(cm->lo_discount);
	queryColumn[4].push_back(cm->d_year);
	queryColumn[4].push_back(cm->d_datekey);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_discount);
	queryColumn[0].push_back(cm->lo_quantity);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_extendedprice);
	queryColumn[0].push_back(cm->lo_discount);
	queryColumn[4].push_back(cm->d_yearmonthnum);
	queryColumn[4].push_back(cm->d_datekey);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_discount);
	queryColumn[0].push_back(cm->lo_quantity);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_extendedprice);
	queryColumn[0].push_back(cm->lo_discount);
	queryColumn[4].push_back(cm->d_datekey);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_partkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_region);
	queryColumn[3].push_back(cm->p_partkey);
	queryColumn[3].push_back(cm->p_category);
	queryColumn[3].push_back(cm->p_brand1);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_partkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_region);
	queryColumn[3].push_back(cm->p_partkey);
	queryColumn[3].push_back(cm->p_brand1);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_partkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_region);
	queryColumn[3].push_back(cm->p_partkey);
	queryColumn[3].push_back(cm->p_brand1);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_region);
	queryColumn[1].push_back(cm->s_nation);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_region);
	queryColumn[2].push_back(cm->c_nation);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_city);
	queryColumn[1].push_back(cm->s_nation);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_city);
	queryColumn[2].push_back(cm->c_nation);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_city);
	queryColumn[1].push_back(cm->s_nation);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_city);
	queryColumn[2].push_back(cm->c_nation);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_city);
	queryColumn[1].push_back(cm->s_nation);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_city);
	queryColumn[2].push_back(cm->c_nation);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_yearmonthnum);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_partkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_supplycost);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_region);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_region);
	queryColumn[2].push_back(cm->c_nation);
	queryColumn[3].push_back(cm->p_partkey);
	queryColumn[3].push_back(cm->p_mfgr);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_partkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_supplycost);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_region);
	queryColumn[1].push_back(cm->s_nation);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_region);
	queryColumn[3].push_back(cm->p_partkey);
	queryColumn[3].push_back(cm->p_mfgr);
	queryColumn[3].push_back(cm->p_category);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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

	queryColumn.resize(cm->TOT_TABLE);
	queryColumn[0].push_back(cm->lo_suppkey);
	queryColumn[0].push_back(cm->lo_custkey);
	queryColumn[0].push_back(cm->lo_partkey);
	queryColumn[0].push_back(cm->lo_orderdate);
	queryColumn[0].push_back(cm->lo_supplycost);
	queryColumn[0].push_back(cm->lo_revenue);
	queryColumn[1].push_back(cm->s_suppkey);
	queryColumn[1].push_back(cm->s_city);
	queryColumn[1].push_back(cm->s_nation);
	queryColumn[2].push_back(cm->c_custkey);
	queryColumn[2].push_back(cm->c_region);
	queryColumn[3].push_back(cm->p_partkey);
	queryColumn[3].push_back(cm->p_brand1);
	queryColumn[3].push_back(cm->p_category);
	queryColumn[4].push_back(cm->d_datekey);
	queryColumn[4].push_back(cm->d_year);

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
QueryOptimizer::prepareOperatorPlacement() {

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

	memset(joinGPUcheck, 0, cm->TOT_TABLE * sizeof(bool));
	memset(joinCPUcheck, 0, cm->TOT_TABLE * sizeof(bool));

	segment_group = (short**) malloc (cm->TOT_TABLE * sizeof(short*)); //4 tables, 64 possible segment group
	// segment_group_temp = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	segment_group_count = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	// segment_group_temp_count = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
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

	joinGPUall = true;
	for (int i = 0; i < join.size(); i++) {
		if (join[i].second->tot_seg_in_GPU < join[i].second->total_segment) {
			joinCPUcheck[join[i].second->table_id] = true;
			joinGPUcheck[join[i].second->table_id] = false;
			joinGPUall = false;
		} else {
			if (pkey_fkey[join[i].second]->tot_seg_in_GPU == pkey_fkey[join[i].second]->total_segment) {
				joinCPUcheck[join[i].second->table_id] = false;
				joinGPUcheck[join[i].second->table_id] = true;
			} else if (pkey_fkey[join[i].second]->tot_seg_in_GPU == 0) {
				joinCPUcheck[join[i].second->table_id] = true;
				joinGPUcheck[join[i].second->table_id] = false;
			} else {
				joinCPUcheck[join[i].second->table_id] = true;
				joinGPUcheck[join[i].second->table_id] = true;	
			}
		}
	}


}

bool
QueryOptimizer::checkPredicate(int table_id, int segment_idx) {
	assert(table_id <= cm->TOT_TABLE);
	for (int i = 0; i < queryColumn[table_id].size(); i++) {
		int column = queryColumn[table_id][i]->column_id;

		if (params->compare1.find(cm->allColumn[column]) != params->compare1.end()) {
			int compare1 = params->compare1[cm->allColumn[column]];
			int compare2 = params->compare2[cm->allColumn[column]];

			assert(compare1 <= compare2);
			assert(cm->segment_min[column][segment_idx] <= cm->segment_max[column][segment_idx]);

			if (compare2 < cm->segment_min[column][segment_idx] || compare1 > cm->segment_max[column][segment_idx]) {
				return false;
			}
		}
	}
	return true;
}

void
QueryOptimizer::updateSegmentStats(int table_id, int segment_idx, int query) {
 	for (int i = 0; i < queryColumn[table_id].size(); i++) {
	    int column = queryColumn[table_id][i]->column_id;
	    Segment* segment = cm->index_to_segment[column][segment_idx];
	    cm->updateSegmentWeightDirect(cm->allColumn[column], segment, speedup[query][cm->allColumn[column]]);
	}
}

void
QueryOptimizer::groupBitmapSegmentTable(int table_id, int query, bool isprofile) {

	int LEN = cm->allColumn[cm->columns_in_table[table_id][0]]->LEN;
	int total_segment = cm->allColumn[cm->columns_in_table[table_id][0]]->total_segment;

	for (int i = 0; i < total_segment; i++) {
		unsigned short temp = 0;

		for (int j = 0; j < opParsed[table_id].size(); j++) {
			Operator* op = opParsed[table_id][j];
			temp = temp << op->columns.size();
			for (int k = 0; k < op->columns.size(); k++) {
				ColumnInfo* column = op->columns[k];
				bool isGPU = cm->segment_bitmap[column->column_id][i];
				temp = temp | (isGPU << k);
			}
		}

		int count = segment_group_count[table_id][temp];

		if (skipping) {
			if (checkPredicate(table_id, i)) { //DISABLE SEGMENT SKIPPING
				segment_group[table_id][temp * total_segment + count] = i;
				segment_group_count[table_id][temp]++;
				if (!isprofile) updateSegmentStats(table_id, i, query);
				processed_segment += queryColumn[table_id].size();
			} else {
				skipped_segment += queryColumn[table_id].size();
			}
		} else {
			segment_group[table_id][temp * total_segment + count] = i;
			segment_group_count[table_id][temp]++;
			if (!isprofile) updateSegmentStats(table_id, i, query);
		}


		if (i == total_segment - 1) {
			if (LEN % SEGMENT_SIZE != 0) {
				last_segment[table_id] = temp;
			}
		}
	}

	for (unsigned short i = 0; i < MAX_GROUPS/2; i++) { //64 segment groups
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
				} else if (op->type == Filter) {
					(bit) ? (op->device = GPU):(op->device = CPU);
				} else if (op->type == Build) {
					(bit) ? (op->device = GPU):(op->device = CPU);
				}

				sg = sg >> op->columns.size();
			}

			Operator* build_op = NULL;

			for (int j = 0; j < opParsed[table_id].size(); j++) {
				Operator* op = opParsed[table_id][j];
				if (op->type != Aggr && op->type != GroupBy && op->type != Build) {
					if (op->device == GPU) {
						opGPUPipeline[table_id][i][0].push_back(op);
					} else if (op->device == CPU) {
						opCPUPipeline[table_id][i][0].push_back(op);
					}
				} else if (op->type == GroupBy || op->type == Aggr) {
					if ((opCPUPipeline[table_id][i][0].size() > 0) && !isprofile) { //TODO!! FIX THIS
						opCPUPipeline[table_id][i][0].push_back(op);
					} else {
						if (op->device == GPU) opGPUPipeline[table_id][i][0].push_back(op);
						else if (op->device == CPU) opCPUPipeline[table_id][i][0].push_back(op);
					}
				} else if (op->type == Build) { //TODO!! FIX THIS (ONLY WORKS FOR CURRENT KERNEL MATCHING)
					build_op = op;
				}
			}

			Operator* op = NULL;

			//TODO! FIX THIS (ONLY WORKS FOR SSB)
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
					for (int j = 0; j < opCPUPipeline[table_id][i][0].size(); j++) {
						op->addChild(opCPUPipeline[table_id][i][0][j]);
						op = opCPUPipeline[table_id][i][0][j];
					}
				}
				op->addChild(NULL);
			} else if (opCPUPipeline[table_id][i][0].size() > 0) {
				opRoots[table_id][i] = opCPUPipeline[table_id][i][0][0];
				op = opRoots[table_id][i];
				for (int j = 1; j < opCPUPipeline[table_id][i][0].size(); j++) {
					op->addChild(opCPUPipeline[table_id][i][0][j]);
					op = opCPUPipeline[table_id][i][0][j];
				}
				op->addChild(NULL);
			}

			if (build_op != NULL) {
				if (op == NULL) {
					opRoots[table_id][i] = build_op;
				} else if (build_op->device != op->device) {
					if (build_op->device == GPU) {
						Operator* transferOp = new Operator(CPU, i, table_id, CPUtoGPU);
						op->addChild(transferOp);
						Operator* matOp = new Operator(GPU, i, table_id, Materialize);
						transferOp->addChild(matOp);
						op = matOp;
						op->addChild(build_op);
					} else {
						Operator* transferOp = new Operator(GPU, i, table_id, GPUtoCPU);
						op->addChild(transferOp);
						Operator* matOp = new Operator(CPU, i, table_id, Materialize);
						transferOp->addChild(matOp);
						op = matOp;
						op->addChild(build_op);
					}
				} else if (build_op->device == op->device) {
					op->addChild(build_op);
				}
				op = build_op;
				op->addChild(NULL);
			}

			int length = segment_group_count[table_id][i] * SEGMENT_SIZE;
			CostModel* cost = new CostModel(length, total_segment, queryGroupByColumn.size(), queryAggrColumn.size(), i, table_id, this);
			cost->permute_cost();
			delete cost;

		}
	}

	if (table_id == 0) {
		for (int i = 0; i < MAX_GROUPS/2; i++) {
			if (segment_group_count[table_id][i] > 0) {
				for (int j = 0; j < opGPUPipeline[table_id][i][0].size(); j++) {
					Operator* op = opGPUPipeline[table_id][i][0][j];
					if (op->type == Probe) joinGPUPipelineCol[i].push_back(op->columns[0]);
					else if (op->type == Filter) selectGPUPipelineCol[i].push_back(op->columns[0]);
					else if (op->type == GroupBy || op->type == Aggr) {
						for (int k = 0; k < op->columns.size(); k++)
							groupbyGPUPipelineCol[i].push_back(op->columns[k]);
					}
				}
				for (int j = 0; j < opCPUPipeline[table_id][i][0].size(); j++) {
					Operator* op = opCPUPipeline[table_id][i][0][j];
					if (op->type == Probe) joinCPUPipelineCol[i].push_back(op->columns[0]);
					else if (op->type == Filter) selectCPUPipelineCol[i].push_back(op->columns[0]);
					else if (op->type == GroupBy || op->type == Aggr) {
						for (int k = 0; k < op->columns.size(); k++)
							groupbyCPUPipelineCol[i].push_back(op->columns[k]);
					}
				}
			}
		}
	}

	short count = 0;
	for (int sg = 0; sg < MAX_GROUPS; sg++) {
		if (segment_group_count[table_id][sg] > 0) {
			par_segment[table_id][count] = sg;
			count++;
		}
	}
	par_segment_count[table_id] = count;
}

void
QueryOptimizer::prepareQuery(int query, Distribution dist) {

	params = new QueryParams(query);

	if (query == 11 || query == 12 || query == 13) {

		if (query == 11) {
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_orderdate] = 1;
			params->selectivity[cm->lo_discount] = 3.0/11 * 1.5;
			params->selectivity[cm->lo_quantity] = 0.5 * 1.5;

			params->real_selectivity[cm->d_year] = 1.0/8;
			params->real_selectivity[cm->lo_orderdate] = 1;
			params->real_selectivity[cm->lo_discount] = 3.0/11;
			params->real_selectivity[cm->lo_quantity] = 0.5;

			params->compare1[cm->lo_discount] = 1;
			params->compare2[cm->lo_discount] = 3;
			params->compare1[cm->lo_quantity] = 0;
			params->compare2[cm->lo_quantity] = 24;

			params->mode[cm->lo_discount] = 1;
			params->mode[cm->lo_quantity] = 1;
			params->mode[cm->d_year] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->real_selectivity[cm->d_year] = 1.0/8;			
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;			
			} else {
				params->compare1[cm->d_year] = 1993;
				params->compare2[cm->d_year] = 1993;
				params->compare1[cm->lo_orderdate] = 19930101;
				params->compare2[cm->lo_orderdate] = 19931231;
			}

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_year]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->lo_discount]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->lo_quantity]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->d_year] = &host_pred_eq;
			params->map_filter_func_host[cm->lo_discount] = &host_pred_between;
			params->map_filter_func_host[cm->lo_quantity] = &host_pred_between;

		} else if (query == 12) {

			params->selectivity[cm->d_yearmonthnum] = 1;
			params->selectivity[cm->lo_orderdate] = 1;
			params->selectivity[cm->lo_discount] = 3.0/11 * 2;
			params->selectivity[cm->lo_quantity] = 0.2 * 2;

			params->real_selectivity[cm->d_yearmonthnum] = 1.0/84;
			params->real_selectivity[cm->lo_orderdate] = 1;
			params->real_selectivity[cm->lo_discount] = 3.0/11;
			params->real_selectivity[cm->lo_quantity] = 0.2;

			params->compare1[cm->lo_discount] = 4;
			params->compare2[cm->lo_discount] = 6;
			params->compare1[cm->lo_quantity] = 26;
			params->compare2[cm->lo_quantity] = 35;

			params->mode[cm->lo_discount] = 1;
			params->mode[cm->lo_quantity] = 1;
			params->mode[cm->d_yearmonthnum] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->d_yearmonthnum] = zipfian[query]->yearmonth.first;
				params->compare2[cm->d_yearmonthnum] = zipfian[query]->yearmonth.second;
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->real_selectivity[cm->d_yearmonthnum] = 1.0/84;			
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_yearmonthnum] = normal[query]->yearmonth.first;
				params->compare2[cm->d_yearmonthnum] = normal[query]->yearmonth.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->real_selectivity[cm->d_yearmonthnum] = 1.0/84;			
			} else {
				params->compare1[cm->d_yearmonthnum] = 199401;
				params->compare2[cm->d_yearmonthnum] = 199401;
				params->compare1[cm->lo_orderdate] = 19940101;
				params->compare2[cm->lo_orderdate] = 19940131;
			}

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_yearmonthnum]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->lo_discount]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->lo_quantity]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->d_yearmonthnum] = &host_pred_eq;
			params->map_filter_func_host[cm->lo_discount] = &host_pred_between;
			params->map_filter_func_host[cm->lo_quantity] = &host_pred_between;

		} else if (query == 13) {

			params->selectivity[cm->d_datekey] = 1;
			params->selectivity[cm->lo_orderdate] = 1;
			params->selectivity[cm->lo_discount] = 3.0/11 * 1.5;
			params->selectivity[cm->lo_quantity] = 0.2 * 1.5;

			params->real_selectivity[cm->d_datekey] = 1.0/364;
			params->real_selectivity[cm->lo_orderdate] = 1;
			params->real_selectivity[cm->lo_discount] = 3.0/11;
			params->real_selectivity[cm->lo_quantity] = 0.2;

			params->compare1[cm->lo_discount] = 5;
			params->compare2[cm->lo_discount] = 7;
			params->compare1[cm->lo_quantity] = 26;
			params->compare2[cm->lo_quantity] = 35;

			params->mode[cm->lo_discount] = 1;
			params->mode[cm->lo_quantity] = 1;
			params->mode[cm->d_datekey] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->d_datekey] = zipfian[query]->date.first;
				params->compare2[cm->d_datekey] = zipfian[query]->date.second;
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->real_selectivity[cm->d_datekey] = 1.0/364;	
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_datekey] = normal[query]->date.first;
				params->compare2[cm->d_datekey] = normal[query]->date.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->real_selectivity[cm->d_datekey] = 1.0/364;		
			} else {
				params->compare1[cm->d_datekey] = 19940204;
				params->compare2[cm->d_datekey] = 19940210;
				params->compare1[cm->lo_orderdate] = 19940204;
				params->compare2[cm->lo_orderdate] = 19940210;
			}

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_datekey]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->lo_discount]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->lo_quantity]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->d_datekey] = &host_pred_between;
			params->map_filter_func_host[cm->lo_discount] = &host_pred_between;
			params->map_filter_func_host[cm->lo_quantity] = &host_pred_between;
		}

		CubDebugExit(cudaMemcpyFromSymbol(&(params->d_group_func), p_mul_func<int>, sizeof(group_func_t<int>)));
		params->h_group_func = &host_mul_func;

		params->unique_val[cm->p_partkey] = 0;
		params->unique_val[cm->c_custkey] = 0;
		params->unique_val[cm->s_suppkey] = 0;
		params->unique_val[cm->d_datekey] = 1;

		params->dim_len[cm->p_partkey] = 0;
		params->dim_len[cm->c_custkey] = 0;
		params->dim_len[cm->s_suppkey] = 0;
		params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

		params->total_val = 1;

		float time;
		SETUP_TIMING();
		cudaEventRecord(start, 0);

		if (custom) {
			params->ht_CPU[cm->p_partkey] = NULL;
			params->ht_CPU[cm->c_custkey] = NULL;
			params->ht_CPU[cm->s_suppkey] = NULL;
			params->ht_CPU[cm->d_datekey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->d_datekey]);	
		} else {
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int), cudaHostAllocDefault));
		}

		if (custom) {
			params->ht_GPU[cm->p_partkey] = NULL;
			params->ht_GPU[cm->s_suppkey] = NULL;
			params->ht_GPU[cm->d_datekey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->d_datekey]);
			params->ht_GPU[cm->c_custkey] = NULL;
		} else {
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int)));			
		}
		cudaEventRecord(stop, 0);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&time, start, stop);
	  cgp->malloc_time_total += time;		

	  memset(params->ht_CPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));


	} else if (query == 21 || query == 22 || query == 23) {

		if (query == 21) {
			params->selectivity[cm->p_category] = 1.0/25 * 1.5;
			params->selectivity[cm->s_region] = 0.2 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_partkey] = 1.0/25 * 1.5;
			params->selectivity[cm->lo_suppkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->p_category] = 1.0/25;
			params->real_selectivity[cm->s_region] = 0.2;
			params->real_selectivity[cm->d_year] = 1;
			params->real_selectivity[cm->lo_partkey] = 1.0/25;
			params->real_selectivity[cm->lo_suppkey] = 0.2;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->s_region] = 1;
			params->compare2[cm->s_region] = 1;
			params->compare1[cm->p_category] = 1;
			params->compare2[cm->p_category] = 1;

			params->mode[cm->s_region] = 1;
			params->mode[cm->p_category] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->real_selectivity[cm->d_year] = 1.0/8;
				params->mode[cm->d_year] = 1; //TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;
				params->mode[cm->d_year] = 1; //TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN
			} else {
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19981231;
			}

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->p_category]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->s_region] = &host_pred_eq;
			params->map_filter_func_host[cm->p_category] = &host_pred_eq;

		} else if (query == 22) {
			params->selectivity[cm->p_brand1] = 1.0/125 * 1.5;
			params->selectivity[cm->s_region] = 0.2 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_partkey] = 1.0/125 * 1.5;
			params->selectivity[cm->lo_suppkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->p_brand1] = 1.0/125;
			params->real_selectivity[cm->s_region] = 0.2;
			params->real_selectivity[cm->d_year] = 1;
			params->real_selectivity[cm->lo_partkey] = 1.0/125;
			params->real_selectivity[cm->lo_suppkey] = 0.2;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->s_region] = 2;
			params->compare2[cm->s_region] = 2;
			params->compare1[cm->p_brand1] = 260;
			params->compare2[cm->p_brand1] = 267;

			params->mode[cm->s_region] = 1;
			params->mode[cm->p_brand1] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->real_selectivity[cm->d_year] = 2.0/8;		
				params->mode[cm->d_year] = 1;	//TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN	
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;		
				params->mode[cm->d_year] = 1;	//TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN	
			} else {
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19981231;
			}

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->p_brand1]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->s_region] = &host_pred_eq;
			params->map_filter_func_host[cm->p_brand1] = &host_pred_between;

		} else if (query == 23) {
			params->selectivity[cm->p_brand1] = 1.0/1000 * 1.5;
			params->selectivity[cm->s_region] = 0.2 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_partkey] = 1.0/1000 * 1.5;
			params->selectivity[cm->lo_suppkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->p_brand1] = 1.0/1000;
			params->real_selectivity[cm->s_region] = 0.2;
			params->real_selectivity[cm->d_year] = 1;
			params->real_selectivity[cm->lo_partkey] = 1.0/1000;
			params->real_selectivity[cm->lo_suppkey] = 0.2;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->s_region] = 3;
			params->compare2[cm->s_region] = 3;
			params->compare1[cm->p_brand1] = 260;
			params->compare2[cm->p_brand1] = 260;

			params->mode[cm->s_region] = 1;
			params->mode[cm->p_brand1] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->real_selectivity[cm->d_year] = 4.0/8;
				params->mode[cm->d_year] = 1;	//TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;
				params->mode[cm->d_year] = 1;	//TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN
			} else {
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19981231;
			}

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->p_brand1]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->s_region] = &host_pred_eq;
			params->map_filter_func_host[cm->p_brand1] = &host_pred_eq;
		}

		CubDebugExit(cudaMemcpyFromSymbol(&(params->d_group_func), p_sub_func<int>, sizeof(group_func_t<int>)));
		params->h_group_func = &host_sub_func;

		params->unique_val[cm->p_partkey] = 7;
		params->unique_val[cm->c_custkey] = 0;
		params->unique_val[cm->s_suppkey] = 0;
		params->unique_val[cm->d_datekey] = 1;

		params->dim_len[cm->p_partkey] = P_LEN;
		params->dim_len[cm->c_custkey] = 0;
		params->dim_len[cm->s_suppkey] = S_LEN;
		params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

		params->total_val = ((1998-1992+1) * (5 * 5 * 40));

		float time;
		SETUP_TIMING();
		cudaEventRecord(start, 0);

		if (custom) {
			params->ht_CPU[cm->p_partkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->p_partkey]);
			params->ht_CPU[cm->c_custkey] = NULL;
			params->ht_CPU[cm->s_suppkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->s_suppkey]);
			params->ht_CPU[cm->d_datekey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->d_datekey]);			
		} else {
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->p_partkey], 2 * params->dim_len[cm->p_partkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->s_suppkey], 2 * params->dim_len[cm->s_suppkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int), cudaHostAllocDefault));	
		}

		if (custom) {
			params->ht_GPU[cm->p_partkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->p_partkey]);
			params->ht_GPU[cm->s_suppkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->s_suppkey]);
			params->ht_GPU[cm->d_datekey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->d_datekey]);
			params->ht_GPU[cm->c_custkey] = NULL;			
		} else {
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->p_partkey], 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->s_suppkey], 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int)));				
		}

		cudaEventRecord(stop, 0);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&time, start, stop);
	  cgp->malloc_time_total += time;
	  // cout << "malloc time: " << cgp->malloc_time_total << endl;

		memset(params->ht_CPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
		memset(params->ht_CPU[cm->p_partkey], 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int));
		memset(params->ht_CPU[cm->s_suppkey], 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));	

		CubDebugExit(cudaMemset(params->ht_GPU[cm->p_partkey], 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->s_suppkey], 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));

	} else if (query == 31 || query == 32 || query == 33 || query == 34) {

		if (query == 31) {
			params->selectivity[cm->c_region] = 0.2 * 1.5;
			params->selectivity[cm->s_region] = 0.2 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_custkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_suppkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->c_region] = 0.2;
			params->real_selectivity[cm->s_region] = 0.2;
			params->real_selectivity[cm->d_year] = 7.0/8;
			params->real_selectivity[cm->lo_custkey] = 0.2;
			params->real_selectivity[cm->lo_suppkey] = 0.2;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->c_region] = 2;
			params->compare2[cm->c_region] = 2;
			params->compare1[cm->s_region] = 2;
			params->compare2[cm->s_region] = 2;

			params->mode[cm->c_region] = 1;
			params->mode[cm->s_region] = 1;
			params->mode[cm->d_year] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->real_selectivity[cm->d_year] = 1.0/8;			
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;			
			} else {
				params->compare1[cm->d_year] = 1992;
				params->compare2[cm->d_year] = 1997;
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19971231;
			}


			params->unique_val[cm->p_partkey] = 0;
			params->unique_val[cm->c_custkey] = 7;
			params->unique_val[cm->s_suppkey] = 25 * 7;
			params->unique_val[cm->d_datekey] = 1;

			params->total_val = ((1998-1992+1) * 25 * 25);

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_year]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_region] = &host_pred_eq;
			params->map_filter_func_host[cm->s_region] = &host_pred_eq;
			params->map_filter_func_host[cm->d_year] = &host_pred_between;

		} else if (query == 32) {
			params->selectivity[cm->c_nation] = 1.0/25 * 1.5;
			params->selectivity[cm->s_nation] = 1.0/25 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_custkey] = 1.0/25 * 1.5;
			params->selectivity[cm->lo_suppkey] = 1.0/25 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->c_nation] = 1.0/25;
			params->real_selectivity[cm->s_nation] = 1.0/25;
			params->real_selectivity[cm->d_year] = 7.0/8;
			params->real_selectivity[cm->lo_custkey] = 1.0/25;
			params->real_selectivity[cm->lo_suppkey] = 1.0/25;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->c_nation] = 24;
			params->compare2[cm->c_nation] = 24;
			params->compare1[cm->s_nation] = 24;
			params->compare2[cm->s_nation] = 24;

			params->mode[cm->c_nation] = 1;
			params->mode[cm->s_nation] = 1;
			params->mode[cm->d_year] = 1;

			if (dist == Zipf) {
				// do {
					zipfian[query]->generateZipf();
					params->compare1[cm->d_year] = zipfian[query]->year.first;
					params->compare2[cm->d_year] = zipfian[query]->year.second;
					params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
					params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
					params->real_selectivity[cm->d_year] = 2.0/8;	
				// } while (params->compare1[cm->d_year] == 1996 && params->compare2[cm->d_year] == 1996);		
			} else if (dist == Norm) {
				// do {
					normal[query]->generateNorm();
					params->compare1[cm->d_year] = normal[query]->year.first;
					params->compare2[cm->d_year] = normal[query]->year.second;
					params->compare1[cm->lo_orderdate] = normal[query]->date.first;
					params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
					params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;	
				// } while (params->compare1[cm->d_year] == 1996 && params->compare2[cm->d_year] == 1996);
			} else {
				params->compare1[cm->d_year] = 1992;
				params->compare2[cm->d_year] = 1997;
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19971231;
			}

			params->unique_val[cm->p_partkey] = 0;
			params->unique_val[cm->c_custkey] = 7;
			params->unique_val[cm->s_suppkey] = 250 * 7;
			params->unique_val[cm->d_datekey] = 1;

			params->total_val = ((1998-1992+1) * 250 * 250);

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_nation]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_nation]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_year]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_nation] = &host_pred_eq;
			params->map_filter_func_host[cm->s_nation] = &host_pred_eq;
			params->map_filter_func_host[cm->d_year] = &host_pred_between;

		} else if (query == 33) {
			params->selectivity[cm->c_city] = 1.0/125 * 1.5;
			params->selectivity[cm->s_city] = 1.0/125 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_custkey] = 1.0/125 * 1.5;
			params->selectivity[cm->lo_suppkey] = 1.0/125 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->c_city] = 1.0/125;
			params->real_selectivity[cm->s_city] = 1.0/125;
			params->real_selectivity[cm->d_year] = 7.0/8;
			params->real_selectivity[cm->lo_custkey] = 1.0/125;
			params->real_selectivity[cm->lo_suppkey] = 1.0/125;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->c_city] = 231;
			params->compare2[cm->c_city] = 235;
			params->compare1[cm->s_city] = 231;
			params->compare2[cm->s_city] = 235;

			params->mode[cm->c_city] = 2;
			params->mode[cm->s_city] = 2;
			params->mode[cm->d_year] = 1;

			if (dist == Zipf) {
				// do {
					zipfian[query]->generateZipf();
					params->compare1[cm->d_year] = zipfian[query]->year.first;
					params->compare2[cm->d_year] = zipfian[query]->year.second;
					params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
					params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
					params->real_selectivity[cm->d_year] = 4.0/8;	
				// } while (params->compare1[cm->d_year] == 1996 && params->compare2[cm->d_year] == 1996);	
			} else if (dist == Norm) {
				// do {
					normal[query]->generateNorm();
					params->compare1[cm->d_year] = normal[query]->year.first;
					params->compare2[cm->d_year] = normal[query]->year.second;
					params->compare1[cm->lo_orderdate] = normal[query]->date.first;
					params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
					params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;	
				// } while (params->compare1[cm->d_year] == 1996 && params->compare2[cm->d_year] == 1996);
			} else {
				params->compare1[cm->d_year] = 1992;
				params->compare2[cm->d_year] = 1997;
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19971231;
			}

			params->unique_val[cm->p_partkey] = 0;
			params->unique_val[cm->c_custkey] = 7;
			params->unique_val[cm->s_suppkey] = 250 * 7;
			params->unique_val[cm->d_datekey] = 1;

			params->total_val = ((1998-1992+1) * 250 * 250);

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_city]), p_pred_eq_or_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_city]), p_pred_eq_or_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_year]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_city] = &host_pred_eq_or_eq;
			params->map_filter_func_host[cm->s_city] = &host_pred_eq_or_eq;
			params->map_filter_func_host[cm->d_year] = &host_pred_between;

		} else if (query == 34) {
			params->selectivity[cm->c_city] = 1.0/125 * 1.5;
			params->selectivity[cm->s_city] = 1.0/125 * 1.5;
			params->selectivity[cm->d_yearmonthnum] = 1;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_custkey] = 1.0/125 * 1.5;
			params->selectivity[cm->lo_suppkey] = 1.0/125 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->c_city] = 1.0/125;
			params->real_selectivity[cm->s_city] = 1.0/125;
			params->real_selectivity[cm->d_yearmonthnum] = 1.0/84;
			params->real_selectivity[cm->lo_custkey] = 1.0/125;
			params->real_selectivity[cm->lo_suppkey] = 1.0/125;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->c_city] = 231;
			params->compare2[cm->c_city] = 235;
			params->compare1[cm->s_city] = 231;
			params->compare2[cm->s_city] = 235;

			params->mode[cm->c_city] = 2;
			params->mode[cm->s_city] = 2;
			params->mode[cm->d_yearmonthnum] = 1;

			if (dist == Zipf) {
				do {
					zipfian[query]->generateZipf();
					params->compare1[cm->d_yearmonthnum] = zipfian[query]->yearmonth.first;
					params->compare2[cm->d_yearmonthnum] = zipfian[query]->yearmonth.second;
					params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
					params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;
					params->real_selectivity[cm->d_yearmonthnum] = 1.0/84;
				} while (params->compare1[cm->d_yearmonthnum] == 199804 || params->compare1[cm->d_yearmonthnum] == 199802);

				// params->compare1[cm->d_yearmonthnum] = 199804;
				// params->compare2[cm->d_yearmonthnum] = 199804;
				// params->compare1[cm->lo_orderdate] = 19980401;
				// params->compare2[cm->lo_orderdate] = 19980430;	
				 			
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_yearmonthnum] = normal[query]->yearmonth.first;
				params->compare2[cm->d_yearmonthnum] = normal[query]->yearmonth.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;
				params->real_selectivity[cm->d_yearmonthnum] = 1.0/84;
			} else {
				params->compare1[cm->d_yearmonthnum] = 199712;
				params->compare2[cm->d_yearmonthnum] = 199712;
				params->compare1[cm->lo_orderdate] = 19971201;
				params->compare2[cm->lo_orderdate] = 19971231;
			}

			params->unique_val[cm->p_partkey] = 0;
			params->unique_val[cm->c_custkey] = 7;
			params->unique_val[cm->s_suppkey] = 250 * 7;
			params->unique_val[cm->d_datekey] = 1;

			params->total_val = ((1998-1992+1) * 250 * 250);

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_city]), p_pred_eq_or_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_city]), p_pred_eq_or_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_yearmonthnum]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_city] = &host_pred_eq_or_eq;
			params->map_filter_func_host[cm->s_city] = &host_pred_eq_or_eq;
			params->map_filter_func_host[cm->d_yearmonthnum] = &host_pred_eq;
		}

		CubDebugExit(cudaMemcpyFromSymbol(&(params->d_group_func), p_sub_func<int>, sizeof(group_func_t<int>)));
		params->h_group_func = &host_sub_func;

		params->dim_len[cm->p_partkey] = 0;
		params->dim_len[cm->c_custkey] = C_LEN;
		params->dim_len[cm->s_suppkey] = S_LEN;
		params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

		float time;
		SETUP_TIMING();
		cudaEventRecord(start, 0);

		if (custom) {
			params->ht_CPU[cm->p_partkey] = NULL;
			params->ht_CPU[cm->c_custkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->c_custkey]);
			params->ht_CPU[cm->s_suppkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->s_suppkey]);
			params->ht_CPU[cm->d_datekey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->d_datekey]);			
		} else {
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->c_custkey], 2 * params->dim_len[cm->c_custkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->s_suppkey], 2 * params->dim_len[cm->s_suppkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int), cudaHostAllocDefault));			
		}

		if (custom) {
			params->ht_GPU[cm->p_partkey] = NULL;
			params->ht_GPU[cm->s_suppkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->s_suppkey]);
			params->ht_GPU[cm->d_datekey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->d_datekey]);
			params->ht_GPU[cm->c_custkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->c_custkey]);			
		} else {
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->c_custkey], 2 * params->dim_len[cm->c_custkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->s_suppkey], 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int)));					
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cgp->malloc_time_total += time;		

		memset(params->ht_CPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
		memset(params->ht_CPU[cm->s_suppkey], 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));
		memset(params->ht_CPU[cm->c_custkey], 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int));

		CubDebugExit(cudaMemset(params->ht_GPU[cm->s_suppkey], 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->c_custkey], 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int)));

	} else if (query == 41 || query == 42 || query == 43) {

		if (query == 41) {
			params->selectivity[cm->p_mfgr] = 0.4 * 1.5;
			params->selectivity[cm->c_region] = 0.2 * 1.5;
			params->selectivity[cm->s_region] = 0.2 * 1.5;
			params->selectivity[cm->d_year] =  1;
			params->selectivity[cm->lo_partkey] = 0.4 * 1.5;
			params->selectivity[cm->lo_custkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_suppkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_orderdate] =  1;

			params->real_selectivity[cm->p_mfgr] = 0.4;
			params->real_selectivity[cm->c_region] = 0.2;
			params->real_selectivity[cm->s_region] = 0.2;
			params->real_selectivity[cm->d_year] =  1;
			params->real_selectivity[cm->lo_partkey] = 0.4;
			params->real_selectivity[cm->lo_custkey] = 0.2;
			params->real_selectivity[cm->lo_suppkey] = 0.2;
			params->real_selectivity[cm->lo_orderdate] =  1;

			params->compare1[cm->c_region] = 1;
			params->compare2[cm->c_region] = 1;
			params->compare1[cm->s_region] = 1;
			params->compare2[cm->s_region] = 1;
			params->compare1[cm->p_mfgr] = 0;
			params->compare2[cm->p_mfgr] = 1;

			params->mode[cm->c_region] = 1;
			params->mode[cm->s_region] = 1;
			params->mode[cm->p_mfgr] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;	
				params->real_selectivity[cm->d_year] = 1.0/8;	
				params->mode[cm->d_year] = 1;	//TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;	
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;	
				params->mode[cm->d_year] = 1;	//TODO: THERE IS NO FILTER NODE ON THE QUERY PLAN
			} else {
				params->compare1[cm->lo_orderdate] = 19920101;
				params->compare2[cm->lo_orderdate] = 19981231;
			}

			params->unique_val[cm->p_partkey] = 0;
			params->unique_val[cm->c_custkey] = 7;
			params->unique_val[cm->s_suppkey] = 0;
			params->unique_val[cm->d_datekey] = 1;

			params->total_val = ((1998-1992+1) * 25);

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->p_mfgr]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_region] = &host_pred_eq;
			params->map_filter_func_host[cm->s_region] = &host_pred_eq;
			params->map_filter_func_host[cm->p_mfgr] = &host_pred_between;

		} else if (query == 42) {
			params->selectivity[cm->p_mfgr] = 0.4 * 1.5;
			params->selectivity[cm->c_region] = 0.2 * 1.5;
			params->selectivity[cm->s_region] = 0.2 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_partkey] = 0.4 * 1.5;
			params->selectivity[cm->lo_custkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_suppkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->p_mfgr] = 0.4;
			params->real_selectivity[cm->c_region] = 0.2;
			params->real_selectivity[cm->s_region] = 0.2;
			params->real_selectivity[cm->d_year] = 2.0/8;
			params->real_selectivity[cm->lo_partkey] = 0.4;
			params->real_selectivity[cm->lo_custkey] = 0.2;
			params->real_selectivity[cm->lo_suppkey] = 0.2;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->c_region] = 1;
			params->compare2[cm->c_region] = 1;
			params->compare1[cm->s_region] = 1;
			params->compare2[cm->s_region] = 1;
			params->compare1[cm->p_mfgr] = 0;
			params->compare2[cm->p_mfgr] = 1;

			params->mode[cm->c_region] = 1;
			params->mode[cm->s_region] = 1;
			params->mode[cm->p_mfgr] = 1;
			params->mode[cm->d_year] = 1;


			if (dist == Zipf) {
				// do {
					zipfian[query]->generateZipf();
					params->compare1[cm->d_year] = zipfian[query]->year.first;
					params->compare2[cm->d_year] = zipfian[query]->year.second;
					params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
					params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;		
					params->real_selectivity[cm->d_year] = 2.0/8;
				// } while (params->compare1[cm->d_year] == 1996 && params->compare2[cm->d_year] == 1996);

			} else if (dist == Norm) {
				// do {
					normal[query]->generateNorm();
					params->compare1[cm->d_year] = normal[query]->year.first;
					params->compare2[cm->d_year] = normal[query]->year.second;
					params->compare1[cm->lo_orderdate] = normal[query]->date.first;
					params->compare2[cm->lo_orderdate] = normal[query]->date.second;		
					params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;
				// } while (params->compare1[cm->d_year] == 1996 && params->compare2[cm->d_year] == 1996);

			} else {
				params->compare1[cm->d_year] = 1997;
				params->compare2[cm->d_year] = 1998;
				params->compare1[cm->lo_orderdate] = 19970101;
				params->compare2[cm->lo_orderdate] = 19981231;
			}

			params->unique_val[cm->p_partkey] = 1;
			params->unique_val[cm->c_custkey] = 0;
			params->unique_val[cm->s_suppkey] = 25;
			params->unique_val[cm->d_datekey] = 25 * 25;

			params->total_val = (1998-1992+1) * 25 * 25;

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->p_mfgr]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_year]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_region] = &host_pred_eq;
			params->map_filter_func_host[cm->s_region] = &host_pred_eq;
			params->map_filter_func_host[cm->p_mfgr] = &host_pred_between;
			params->map_filter_func_host[cm->d_year] = &host_pred_between;

		} else if (query == 43) {
			params->selectivity[cm->p_category] = 1.0/25 * 1.5;
			params->selectivity[cm->c_region] = 0.2 * 1.5;
			params->selectivity[cm->s_nation] = 1.0/25 * 1.5;
			params->selectivity[cm->d_year] = 1;
			params->selectivity[cm->lo_partkey] = 1.0/25 * 1.5;
			params->selectivity[cm->lo_custkey] = 0.2 * 1.5;
			params->selectivity[cm->lo_suppkey] = 1.0/25 * 1.5;
			params->selectivity[cm->lo_orderdate] = 1;

			params->real_selectivity[cm->p_category] = 1.0/25;
			params->real_selectivity[cm->c_region] = 0.2;
			params->real_selectivity[cm->s_nation] = 1.0/25;
			params->real_selectivity[cm->d_year] = 2.0/8;
			params->real_selectivity[cm->lo_partkey] = 1.0/25;
			params->real_selectivity[cm->lo_custkey] = 0.2;
			params->real_selectivity[cm->lo_suppkey] = 1.0/25;
			params->real_selectivity[cm->lo_orderdate] = 1;

			params->compare1[cm->c_region] = 1;
			params->compare2[cm->c_region] = 1;
			params->compare1[cm->s_nation] = 24;
			params->compare2[cm->s_nation] = 24;
			params->compare1[cm->p_category] = 3;
			params->compare2[cm->p_category] = 3;

			params->mode[cm->c_region] = 1;
			params->mode[cm->s_nation] = 1;
			params->mode[cm->p_category] = 1;
			params->mode[cm->d_year] = 1;

			if (dist == Zipf) {
				zipfian[query]->generateZipf();
				params->compare1[cm->d_year] = zipfian[query]->year.first;
				params->compare2[cm->d_year] = zipfian[query]->year.second;
				params->compare1[cm->lo_orderdate] = zipfian[query]->date.first;
				params->compare2[cm->lo_orderdate] = zipfian[query]->date.second;
				params->real_selectivity[cm->d_year] = 4.0/8;	
			} else if (dist == Norm) {
				normal[query]->generateNorm();
				params->compare1[cm->d_year] = normal[query]->year.first;
				params->compare2[cm->d_year] = normal[query]->year.second;
				params->compare1[cm->lo_orderdate] = normal[query]->date.first;
				params->compare2[cm->lo_orderdate] = normal[query]->date.second;
				params->real_selectivity[cm->d_year] = (normal[query]->year.second - normal[query]->year.first + 1.0)/8;	
			} else {
				params->compare1[cm->d_year] = 1997;
				params->compare2[cm->d_year] = 1998;
				params->compare1[cm->lo_orderdate] = 19970101;
				params->compare2[cm->lo_orderdate] = 19981231;
			}

			params->unique_val[cm->p_partkey] = 1;
			params->unique_val[cm->c_custkey] = 0;
			params->unique_val[cm->s_suppkey] = 1000;
			params->unique_val[cm->d_datekey] = 250 * 1000;

			params->total_val = (1998-1992+1) * 250 * 1000;

			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->c_region]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->s_nation]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->p_category]), p_pred_eq<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));
			CubDebugExit(cudaMemcpyFromSymbol(&(params->map_filter_func_dev[cm->d_year]), p_pred_between<int, 128, 4>, sizeof(filter_func_t_dev<int, 128, 4>)));

			params->map_filter_func_host[cm->c_region] = &host_pred_eq;
			params->map_filter_func_host[cm->s_nation] = &host_pred_eq;
			params->map_filter_func_host[cm->p_category] = &host_pred_eq;
			params->map_filter_func_host[cm->d_year] = &host_pred_between;
		}

		CubDebugExit(cudaMemcpyFromSymbol(&(params->d_group_func), p_sub_func<int>, sizeof(group_func_t<int>)));
		params->h_group_func = &host_sub_func;

		params->dim_len[cm->p_partkey] = P_LEN;
		params->dim_len[cm->c_custkey] = C_LEN;
		params->dim_len[cm->s_suppkey] = S_LEN;
		params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

		float time;
		SETUP_TIMING();
		cudaEventRecord(start, 0);
		if (custom) {
			params->ht_CPU[cm->p_partkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->p_partkey]);
			params->ht_CPU[cm->c_custkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->c_custkey]);
			params->ht_CPU[cm->s_suppkey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->s_suppkey]);
			params->ht_CPU[cm->d_datekey] = (int*) cm->customMalloc<int>(2 * params->dim_len[cm->d_datekey]);			
		} else {
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->p_partkey], 2 * params->dim_len[cm->p_partkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->c_custkey], 2 * params->dim_len[cm->c_custkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->s_suppkey], 2 * params->dim_len[cm->s_suppkey] * sizeof(int), cudaHostAllocDefault));
			CubDebugExit(cudaHostAlloc((void**) &params->ht_CPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int), cudaHostAllocDefault));				
		}

		if (custom) {
			params->ht_GPU[cm->p_partkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->p_partkey]);
			params->ht_GPU[cm->s_suppkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->s_suppkey]);
			params->ht_GPU[cm->d_datekey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->d_datekey]);
			params->ht_GPU[cm->c_custkey] = (int*) cm->customCudaMalloc<int>(2 * params->dim_len[cm->c_custkey]);			
		} else {
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->p_partkey], 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->c_custkey], 2 * params->dim_len[cm->c_custkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->s_suppkey], 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
			CubDebugExit(cudaMalloc((void**) &params->ht_GPU[cm->d_datekey], 2 * params->dim_len[cm->d_datekey] * sizeof(int)));					
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cgp->malloc_time_total += time;	

		memset(params->ht_CPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
		memset(params->ht_CPU[cm->p_partkey], 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int));
		memset(params->ht_CPU[cm->s_suppkey], 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));
		memset(params->ht_CPU[cm->c_custkey], 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int));

		CubDebugExit(cudaMemset(params->ht_GPU[cm->p_partkey], 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->s_suppkey], 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->d_datekey], 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));
		CubDebugExit(cudaMemset(params->ht_GPU[cm->c_custkey], 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int)));


	} else {
		assert(0);
	}

	cout << endl;
	cout << " Query: " << query << " " << params->compare1[cm->lo_orderdate] << " " << params->compare2[cm->lo_orderdate] << endl;

	params->min_key[cm->p_partkey] = 0;
	params->min_key[cm->c_custkey] = 0;
	params->min_key[cm->s_suppkey] = 0;
	params->min_key[cm->d_datekey] = 19920101;

	params->max_key[cm->p_partkey] = P_LEN-1;
	params->max_key[cm->c_custkey] = C_LEN-1;
	params->max_key[cm->s_suppkey] = S_LEN-1;
	params->max_key[cm->d_datekey] = 19981231;

	params->min_val[cm->p_partkey] = 0;
	params->min_val[cm->c_custkey] = 0;
	params->min_val[cm->s_suppkey] = 0;
	params->min_val[cm->d_datekey] = 1992;

	int res_array_size = params->total_val * 6;

	float time;
	SETUP_TIMING();
	cudaEventRecord(start, 0);
	if (custom) params->res = (int*) cm->customCudaHostAlloc<int>(res_array_size);
	else CubDebugExit(cudaHostAlloc((void**) &params->res, res_array_size * sizeof(int), cudaHostAllocDefault));
	if (custom) params->d_res = (int*) cm->customCudaMalloc<int>(res_array_size);
	else CubDebugExit(cudaMalloc((void**) &params->d_res, res_array_size * sizeof(int)));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cgp->malloc_time_total += time;
	// cout << "malloc time: " << cgp->malloc_time_total << endl;

  memset(params->res, 0, res_array_size * sizeof(int));
	CubDebugExit(cudaMemset(params->d_res, 0, res_array_size * sizeof(int)));

};

void
QueryOptimizer::clearPrepare() {

  if (!custom) {
  	cudaFree(params->d_res);
  	cudaFreeHost(params->res);
 		if (params->ht_GPU[cm->p_partkey] != NULL) cudaFree(params->ht_GPU[cm->p_partkey]);
 		if (params->ht_GPU[cm->s_suppkey] != NULL) cudaFree(params->ht_GPU[cm->s_suppkey]);
 		if (params->ht_GPU[cm->c_custkey] != NULL) cudaFree(params->ht_GPU[cm->c_custkey]);
 		if (params->ht_GPU[cm->d_datekey] != NULL) cudaFree(params->ht_GPU[cm->d_datekey]);

 		if (params->ht_CPU[cm->p_partkey] != NULL) cudaFreeHost(params->ht_CPU[cm->p_partkey]);
 		if (params->ht_CPU[cm->s_suppkey] != NULL) cudaFreeHost(params->ht_CPU[cm->s_suppkey]);
 		if (params->ht_CPU[cm->c_custkey] != NULL) cudaFreeHost(params->ht_CPU[cm->c_custkey]);
 		if (params->ht_CPU[cm->d_datekey] != NULL) cudaFreeHost(params->ht_CPU[cm->d_datekey]);
  }

  params->min_key.clear();
  params->min_val.clear();
  params->unique_val.clear();
  params->dim_len.clear();

  params->ht_CPU.clear();
  params->ht_GPU.clear();
  //cgp->col_idx.clear();

  params->compare1.clear();
  params->compare2.clear();
  params->mode.clear();


}