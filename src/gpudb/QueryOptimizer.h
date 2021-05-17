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

	vector<vector<ColumnInfo*>> join;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> groupby_probe;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> groupby_hash;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> select_probe;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> select_hash;

	vector<pair<int, int>> joinGPU;
	vector<pair<int, int>> joinCPU;
	pair<int, int> groupbyCPU;
	pair<int, int> groupbyGPU;
	vector<pair<int, int>> selectprobeGPU;
	vector<pair<int, int>> selectprobeCPU;
	map<ColumnInfo*, vector<pair<int, int>>> selecthashGPU;
	map<ColumnInfo*, vector<pair<int, int>>> selecthashCPU;
	vector<vector<pair<int, int>>> joinGPUPipeline;
	vector<vector<pair<int, int>>> joinCPUPipeline;
	vector<vector<pair<int, int>>> selectGPUPipeline;
	vector<vector<pair<int, int>>> selectCPUPipeline;
	vector<vector<ColumnInfo*>> joinGPUPipelineCol;
	vector<vector<ColumnInfo*>> joinCPUPipelineCol;
	vector<vector<ColumnInfo*>> selectGPUPipelineCol;
	vector<vector<ColumnInfo*>> selectCPUPipelineCol;
	map<ColumnInfo*, pair<int, int>> transfer;

	QueryOptimizer();
	void parseQuery(int query);
	void parseQuery11();
	void parseQuery21();
	void parseQuery31();
	void parseQuery41();

	void constructPipeline(vector<vector<pair<int, int>>>& CPUPipeline, vector<vector<pair<int, int>>>& GPUPipeline, 
	vector<vector<ColumnInfo*>>& CPUPipelineCol, vector<vector<ColumnInfo*>>& GPUPipelineCol, multimap<int, ColumnInfo*>& temp, 
	ColumnInfo* column, int N);
	void clearVector();
	void latematerialization();
	void latematerializationflex();
	void patching();

};

QueryOptimizer::QueryOptimizer() {
	cm = new CacheManager(1000000000, 25);
}

void 
QueryOptimizer::parseQuery(int query) {
	if (query == 0) parseQuery11();
	else if (query == 1) parseQuery21();
	else if (query == 2) parseQuery31();
	else parseQuery41();
}

void
QueryOptimizer::clearVector() {
	querySelectColumn.clear();
	queryBuildColumn.clear();
	queryProbeColumn.clear();
	queryGroupByColumn.clear();
	queryAggrColumn.clear();

	join.clear();
	groupby_probe.clear();
	groupby_hash.clear();
	select_probe.clear();
	select_hash.clear();

	joinCPU.clear(); //vector
	joinGPU.clear();
	selecthashGPU.clear(); //map
	selecthashCPU.clear();
	selectprobeGPU.clear(); //vector
	selectprobeCPU.clear();
	joinCPUPipeline.clear(); //vector
	joinGPUPipeline.clear();
	joinCPUPipelineCol.clear(); //vector
	joinGPUPipelineCol.clear();
	selectCPUPipeline.clear(); //vector
	selectGPUPipeline.clear();
	selectCPUPipelineCol.clear(); //vector
	selectGPUPipelineCol.clear();

}

void 
QueryOptimizer::parseQuery11() {
	querySelectColumn.push_back(cm->lo_discount);
	querySelectColumn.push_back(cm->lo_quantity);
	querySelectColumn.push_back(cm->d_year);
	queryBuildColumn.push_back(cm->d_datekey);
	queryProbeColumn.push_back(cm->lo_orderdate);
	queryGroupByColumn.push_back(cm->d_year);
	queryGroupByColumn.push_back(cm->p_brand1);
	queryAggrColumn.push_back(cm->lo_extendedprice);
	queryAggrColumn.push_back(cm->lo_discount);
}

void 
QueryOptimizer::parseQuery21() {
	// clearVector();

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

	join[0].push_back(cm->lo_suppkey);
	join[0].push_back(cm->s_suppkey);
	join[1].push_back(cm->lo_partkey);
	join[1].push_back(cm->p_partkey);
	join[2].push_back(cm->lo_orderdate);
	join[2].push_back(cm->d_datekey);

	select_hash[cm->s_suppkey].push_back(cm->s_region);
	select_hash[cm->p_partkey].push_back(cm->p_category);

	groupby_probe[cm->lo_orderdate].push_back(cm->lo_revenue);
	groupby_hash[cm->p_partkey].push_back(cm->p_brand1);
	groupby_hash[cm->d_datekey].push_back(cm->d_year);

	// patching();
	// latematerializationflex();
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
}


void 
QueryOptimizer::latematerialization() {
	//(s, 14), (c, 14), (p, 15), (d, 16)

	//group-by

	int min = cm->lo_orderdate->total_segment;
	
	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
		int tot_seg_in_GPU = groupby_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
		if (tot_seg_in_GPU < min) {
			min = tot_seg_in_GPU;
		}
	}

	bool GPU = true;
	for (int i = 0; i < join.size(); i++) {
		for (int j = 0; j < groupby_hash[join[i][1]].size(); j++) {
			int tot_seg_in_GPU = groupby_hash[join[i][1]][j]->tot_seg_in_GPU;
			int total_segment = groupby_hash[join[i][1]][j]->total_segment;
			if (tot_seg_in_GPU < total_segment) {
				GPU = false;
				break;
			}
		}
	}

	if (GPU) {
		assert(min <= cm->lo_orderdate->total_segment);
		groupbyCPU = pair<int, int> (min, cm->lo_orderdate->total_segment);
		groupbyGPU = pair<int, int> (0, min);
	} else {
		groupbyCPU = pair<int, int> (0, cm->lo_orderdate->total_segment);
		groupbyGPU = pair<int, int> (0, 0);
	}

	//join

	multimap<int, ColumnInfo*> temp; //track how many segments can be executed in GPU from the fact table
	// temp is sorted from the column with least segment in GPU to column with most segment in GPU

	min = cm->lo_orderdate->total_segment;

	for (int i = 0; i < join.size(); i++) {
		int tot_seg_in_GPU = join[i][0]->tot_seg_in_GPU;
		if (tot_seg_in_GPU < min && tot_seg_in_GPU > 0) {
			min = tot_seg_in_GPU;
		}
	}

	for (int i = 0; i < join.size(); i++) {
		if (join[i][1]->tot_seg_in_GPU < join[i][1]->total_segment) {
			joinGPU.push_back(pair<int, int> (0, 0));
			joinCPU.push_back(pair<int, int> (0, join[i][0]->total_segment));
			temp.insert(pair <int, ColumnInfo*> (0, join[i][0]));
		} else if (join[i][0]->tot_seg_in_GPU == 0) {
			joinGPU.push_back(pair<int, int> (0, 0));
			joinCPU.push_back(pair<int, int> (0, join[i][0]->total_segment));
			temp.insert(pair <int, ColumnInfo*> (0, join[i][0]));
		} else {
			assert(min <= join[i][0]->total_segment);
			joinGPU.push_back(pair<int, int> (0, min));
			joinCPU.push_back(pair<int, int> (min, join[i][0]->total_segment));
			temp.insert(pair <int, ColumnInfo*> (min, join[i][0]));
		}
	}

	constructPipeline(joinCPUPipeline, joinGPUPipeline, joinCPUPipelineCol, joinGPUPipelineCol, temp, cm->lo_orderdate, join.size());

	//select hash

	for (int i = 0; i < join.size(); i++) {
		int min = join[i][1]->total_segment;
		ColumnInfo* column = join[i][1];

		for (int j = 0; j < select_hash[column].size(); j++) {
			int tot_seg_in_GPU = select_hash[column][j]->tot_seg_in_GPU;
			if (tot_seg_in_GPU < min) {
				min = tot_seg_in_GPU;
			}
		}

		for (int j = 0; j < select_hash[column].size(); j++) {
			int total_segment = select_hash[column][j]->total_segment;
			assert(min <= total_segment);
			selecthashGPU[column].push_back(pair<int, int> (0, min));
			selecthashCPU[column].push_back(pair<int, int> (min, total_segment));
		}
	}

	//select probe

	temp.clear();

	min = cm->lo_orderdate->total_segment;

	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
		int tot_seg_in_GPU = select_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
		if (tot_seg_in_GPU < min) {
			min = tot_seg_in_GPU;
		}
	}

	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
		int total_segment = select_probe[cm->lo_orderdate][i]->total_segment;
		assert(min <= total_segment);
		selectprobeGPU.push_back(pair<int, int> (0, min));
		selectprobeCPU.push_back(pair<int, int> (min, total_segment));
		temp.insert(pair<int, ColumnInfo*> (min, select_probe[cm->lo_orderdate][i]));
	}

	constructPipeline(selectCPUPipeline, selectGPUPipeline, selectCPUPipelineCol, selectGPUPipelineCol, temp, cm->lo_orderdate, select_probe.size());

}



void 
QueryOptimizer::latematerializationflex() {
	//group-by

	int min = cm->lo_orderdate->total_segment;

	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
		int tot_seg_in_GPU = groupby_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
		if (tot_seg_in_GPU < min) {
			min = tot_seg_in_GPU;
		}
	}

	bool GPU = true;
	for (int i = 0; i < join.size(); i++) {
		for (int j = 0; j < groupby_hash[join[i][1]].size(); j++) {
			int tot_seg_in_GPU = groupby_hash[join[i][1]][j]->tot_seg_in_GPU;
			int total_segment = groupby_hash[join[i][1]][j]->total_segment;
			if (tot_seg_in_GPU < total_segment) {
				GPU = false;
				break;
			}
		}
	}

	if (GPU) {
		assert(min <= cm->lo_orderdate->total_segment);
		groupbyCPU = pair<int, int> (min, cm->lo_orderdate->total_segment);
		groupbyGPU = pair<int, int> (0, min);
	} else {
		groupbyCPU = pair<int, int> (0, cm->lo_orderdate->total_segment);
		groupbyGPU = pair<int, int> (0, 0);
	}

	//join

	multimap<int, ColumnInfo*> temp; //track how many segments can be executed in GPU from the fact table
	// temp is sorted from the column with least segment in GPU to column with most segment in GPU

	for (int i = 0; i < join.size(); i++) {
		if (join[i][1]->tot_seg_in_GPU < join[i][1]->total_segment) {
			joinGPU.push_back(pair<int, int> (0, 0));
			joinCPU.push_back(pair<int, int> (0, join[i][0]->total_segment));
			temp.insert(pair <int, ColumnInfo*> (0, join[i][0]));
		} else {
			int tot_seg_in_GPU = join[i][0]->tot_seg_in_GPU;
			int total_segment = join[i][0]->total_segment;
			assert(tot_seg_in_GPU <= total_segment);
			joinGPU.push_back(pair<int, int> (0, tot_seg_in_GPU));
			joinCPU.push_back(pair<int, int> (tot_seg_in_GPU, total_segment));
			temp.insert(pair <int, ColumnInfo*> (tot_seg_in_GPU, join[i][0]));
		}
	}

	constructPipeline(joinCPUPipeline, joinGPUPipeline, joinCPUPipelineCol, joinGPUPipelineCol, temp, cm->lo_orderdate, join.size());

	//select hash

	for (int i = 0; i < join.size(); i++) {
		int min = join[i][1]->total_segment;
		ColumnInfo* column = join[i][1];

		for (int j = 0; j < select_hash[column].size(); j++) {
			int tot_seg_in_GPU = select_hash[column][j]->tot_seg_in_GPU;
			if (tot_seg_in_GPU < min) {
				min = tot_seg_in_GPU;
			}
		}

		for (int j = 0; j < select_hash[column].size(); j++) {
			int total_segment = select_hash[column][j]->total_segment;
			assert(min <= total_segment);
			selecthashGPU[column].push_back(pair<int, int> (0, min));
			selecthashCPU[column].push_back(pair<int, int> (min, total_segment));
		}
	}

	//select probe

	temp.clear();

	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
		int tot_seg_in_GPU = select_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
		int total_segment = select_probe[cm->lo_orderdate][i]->total_segment;
		assert(tot_seg_in_GPU <= total_segment);
		selectprobeGPU.push_back(pair<int, int> (0, tot_seg_in_GPU));
		selectprobeCPU.push_back(pair<int, int> (tot_seg_in_GPU, total_segment));
		temp.insert(pair <int, ColumnInfo*> (tot_seg_in_GPU, select_probe[cm->lo_orderdate][i]));
	}

	constructPipeline(selectCPUPipeline, selectGPUPipeline, selectCPUPipelineCol, selectGPUPipelineCol, temp, cm->lo_orderdate, select_probe.size());
}

void
QueryOptimizer::constructPipeline(vector<vector<pair<int, int>>>& CPUPipeline, vector<vector<pair<int, int>>>& GPUPipeline, 
	vector<vector<ColumnInfo*>>& CPUPipelineCol, vector<vector<ColumnInfo*>>& GPUPipelineCol, multimap<int, ColumnInfo*>& temp, 
	ColumnInfo* column, int N) {

	GPUPipelineCol.resize(N+1);
	CPUPipelineCol.resize(N+1);
	GPUPipeline.resize(N+1);
	CPUPipeline.resize(N+1);
	GPUPipeline[N].push_back(pair<int, int> (0, 0));
	CPUPipeline[0].push_back(pair<int, int> (0, 0));

	multimap<int, ColumnInfo*>::iterator itr;
	multimap<int, ColumnInfo*>::reverse_iterator ritr;

	int previous = 0;
	int i = 0;
	for (itr = temp.begin(); itr != temp.end(); itr++) {
		//ABCD BCD CD D NULL
		for (int j = 0; j < i+1; j++) {
			GPUPipelineCol[j].push_back(itr->second);
		}

		//NULL A AB ABC ABCD
		for (int j = i+1; j <= join.size(); j++) {
			CPUPipelineCol[j].push_back(itr->second);
		}

		//filling select GPU pipeline
		if (previous < itr->first) {
			GPUPipeline[i].push_back(pair<int, int> (previous, itr->first));
		} else if (previous == itr->first) {
			GPUPipeline[i].push_back(pair<int, int> (0, 0));
		} else {
			assert(0);
		}
		assert(i < N);
		previous = itr->first;
		i++;
	}

	int total_segment = column->total_segment;
	previous = total_segment;
	i = N;
	//filling join CPU pipeline
	for (ritr = temp.rbegin(); ritr != temp.rend(); ritr++) {
		if (previous > ritr->first) {
			CPUPipeline[i].push_back(pair<int, int> (ritr->first, previous));
		} else if (previous == ritr->first) {
			CPUPipeline[i].push_back(pair<int, int> (0, 0));
		} else {
			assert(0);
		}
		assert(i > 0);
		previous = ritr->first;
		i--;
	}
}

void
QueryOptimizer::patching() {
	int max = 0;

	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
		int tot_seg_in_GPU = groupby_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
		if (tot_seg_in_GPU > max) {
			max = tot_seg_in_GPU;
		}
	}

	for (int i = 0; i < join.size(); i++) {
		int tot_seg_in_GPU = join[i][0]->tot_seg_in_GPU;
		if (tot_seg_in_GPU > max) {
			max = tot_seg_in_GPU;
		}
	}

	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
		int tot_seg_in_GPU = select_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
		if (tot_seg_in_GPU > max) {
			max = tot_seg_in_GPU;
		}
	}

	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
		ColumnInfo* column = groupby_probe[cm->lo_orderdate][i];
		int tot_seg_in_GPU = column->tot_seg_in_GPU;
		assert(tot_seg_in_GPU <= max);
		if (tot_seg_in_GPU < max) {
			transfer[column] = pair<int, int> (tot_seg_in_GPU, max);
		}
	}

	for (int i = 0; i < join.size(); i++) {
		ColumnInfo* column = join[i][0];
		int tot_seg_in_GPU = column->tot_seg_in_GPU;
		assert(tot_seg_in_GPU <= max);
		if (tot_seg_in_GPU < max) {
			transfer[column] = pair<int, int> (tot_seg_in_GPU, max);
		}
	}

	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
		ColumnInfo* column = select_probe[cm->lo_orderdate][i];
		int tot_seg_in_GPU = column->tot_seg_in_GPU;
		assert(tot_seg_in_GPU <= max);
		if (tot_seg_in_GPU < max) {
			transfer[column] = pair<int, int> (tot_seg_in_GPU, max);
		}
	}


	for (int i = 0; i < join.size(); i++) {
		for (int j = 0; j < groupby_hash[join[i][1]].size(); j++) {
			ColumnInfo* column = groupby_hash[join[i][1]][j];
			int tot_seg_in_GPU = column->tot_seg_in_GPU;
			int total_segment = column->total_segment;
			assert(tot_seg_in_GPU <= total_segment);
			if (tot_seg_in_GPU < total_segment) {
				transfer[column] = pair<int, int> (tot_seg_in_GPU, total_segment);
			}
		}
	}


	for (int i = 0; i < join.size(); i++) {
		ColumnInfo* column = join[i][1];
		int tot_seg_in_GPU = column->tot_seg_in_GPU;
		int total_segment = column->total_segment;
		assert(tot_seg_in_GPU <= total_segment);
		if (tot_seg_in_GPU < total_segment) {
			transfer[column] = pair<int, int> (tot_seg_in_GPU, total_segment);
		}
	}
}
#endif

	/*selectGPUpipelineCol.resize(select_probe.size()+1);
	selectCPUpipelineCol.resize(select_probe.size()+1);
	selectGPUpipeline.resize(select_probe.size()+1);
	selectCPUpipeline.resize(select_probe.size()+1);
	selectGPUpipeline[select_probe.size()].push_back(pair<int, int> (0, 0));
	selectCPUpipeline[0].push_back(pair<int, int> (0, 0));

	int previous = 0;
	int i = 0;
	for (itr = temp.begin(); itr != temp.end(); itr++) {
		//ABCD BCD CD D NULL
		for (int j = 0; j < i+1; j++) {
			selectGPUpipelineCol[j].push_back(itr->second);
		}

		//NULL A AB ABC ABCD
		for (int j = i+1; j <= join.size(); j++) {
			selectCPUpipelineCol[j].push_back(itr->second);
		}

		//filling select GPU pipeline
		selectGPUpipeline[i].push_back(pair<int, int> (previous, itr->first));
		previous = itr->first;
		i++;
	}

	int total_segment = cm->lo_orderdate->total_segment;
	previous = total_segment;
	i = select_probe.size();
	//filling join CPU pipeline
	for (itr = temp.rbegin(); itr != temp.rend(); itr++) {
		selectCPUpipeline[i].push_back(pair<int, int> (itr->first, previous));
		previous = itr->first;
		i--;
	}

	joinGPUpipelineCol.resize(join.size()+1);
	joinCPUpipelineCol.resize(join.size()+1);
	joinGPUpipeline.resize(join.size()+1);
	joinCPUpipeline.resize(join.size()+1);
	joinGPUpipeline[join.size()].push_back(pair<int, int> (0, 0));
	joinCPUpipeline[0].push_back(pair<int, int> (0, 0));

	int previous = 0;
	int i = 0;
	for (itr = temp.begin(); itr != temp.end(); itr++) {
		//ABCD BCD CD D NULL
		for (int j = 0; j < i+1; j++) {
			joinGPUpipelineCol[j].push_back(itr->second);
		}

		//NULL A AB ABC ABCD
		for (int j = i+1; j <= join.size(); j++) {
			joinCPUpipelineCol[j].push_back(itr->second);
		}
		//filling join GPU pipeline
		joinGPUpipeline[i].push_back(pair<int, int> (previous, itr->first));
		transfer[i].push_back(pair<int, int> (previous, itr->first));
		materialize[i].push_back(pair<int, int> (previous, itr->first));
		previous = itr->first;
		i++;
	}

	int total_segment = cm->lo_orderdate->total_segment;
	previous = total_segment;
	i = join.size();
	//filling join CPU pipeline
	for (itr = temp.rbegin(); itr != temp.rend(); itr++) {
		joinCPUpipeline[i].push_back(pair<int, int> (itr->first, previous));
		previous = itr->first;
		i--;
	}*/

