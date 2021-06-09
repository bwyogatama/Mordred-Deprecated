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
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> groupby_probe;
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
	bool* joinGPUcheck;

	short** segment_group;
	short** segment_group_count;

	QueryOptimizer();
	void parseQuery(int query);
	void parseQuery11();
	void parseQuery21();
	void parseQuery31();
	void parseQuery41();

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

QueryOptimizer::QueryOptimizer() {
	cm = new CacheManager(1000000000);
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

	segment_group = (short**) malloc (cm->TOT_TABLE * sizeof(short*)); //4 tables, 64 possible segment group
	segment_group_count = (short**) malloc (cm->TOT_TABLE * sizeof(short*));
	for (int i = 0; i < cm->TOT_TABLE; i++) {
		segment_group[i] = (short*) malloc (64 * cm->lo_orderdate->total_segment * sizeof(short));
		segment_group_count[i] = (short*) malloc (64 * sizeof(short));
		memset(segment_group_count[i], 0, 64 * sizeof(short));
	}

	if (query == 0) parseQuery11();
	else if (query == 1) parseQuery21();
	else if (query == 2) parseQuery31();
	else parseQuery41();
}

void
QueryOptimizer::clearVector() {

	for (int i = 0; i < cm->TOT_TABLE; i++) {
		free(segment_group[i]);
		free(segment_group_count[i]);
	}

	free(segment_group);
	free(segment_group_count);
	free(joinGPUcheck);

	join.clear();
	groupby_probe.clear();
	groupby_build.clear();
	select_probe.clear();
	select_build.clear();

	// joinCPU.clear(); //vector
	// joinGPU.clear();
	// selectbuildGPU.clear(); //map
	// selectbuildCPU.clear();
	// selectprobeGPU.clear(); //vector
	// selectprobeCPU.clear();
	// joinCPUPipeline.clear(); //vector
	// joinGPUPipeline.clear();
	// selectCPUPipeline.clear(); //vector
	// selectGPUPipeline.clear();

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

	groupby_probe[cm->lo_orderdate].push_back(cm->lo_extendedprice);
	groupby_probe[cm->lo_orderdate].push_back(cm->lo_discount);

	select_probe[cm->lo_orderdate].push_back(cm->lo_quantity);
	select_probe[cm->lo_orderdate].push_back(cm->lo_discount);

	select_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();

}

void 
QueryOptimizer::parseQuery21() {
	//clearVector();
	querySelectColumn.push_back(cm->p_category);
	querySelectColumn.push_back(cm->d_year);
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

	groupby_probe[cm->lo_orderdate].push_back(cm->lo_revenue);
	groupby_build[cm->p_partkey].push_back(cm->p_brand1);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	// joinGPUPipeline.resize(join.size());
	// joinCPUPipeline.resize(join.size());
	// selectGPUPipeline.resize(select_probe.size());
	// selectCPUPipeline.resize(select_probe.size());

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
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->c_custkey].push_back(cm->c_region);
	select_build[cm->s_suppkey].push_back(cm->s_region);
	select_build[cm->d_datekey].push_back(cm->d_year);

	groupby_probe[cm->lo_orderdate].push_back(cm->lo_revenue);

	groupby_build[cm->c_custkey].push_back(cm->c_nation);
	groupby_build[cm->s_suppkey].push_back(cm->s_nation);
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
	join[0] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_partkey, cm->p_partkey);
	join[1] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_custkey, cm->c_custkey);
	join[2] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_suppkey, cm->s_suppkey);
	join[3] = pair<ColumnInfo*, ColumnInfo*> (cm->lo_orderdate, cm->d_datekey);

	select_build[cm->p_partkey].push_back(cm->p_mfgr);
	select_build[cm->c_custkey].push_back(cm->c_region);
	select_build[cm->s_suppkey].push_back(cm->s_region);

	groupby_probe[cm->lo_orderdate].push_back(cm->lo_revenue);
	groupby_probe[cm->lo_orderdate].push_back(cm->lo_supplycost);

	groupby_build[cm->c_custkey].push_back(cm->c_nation);
	groupby_build[cm->d_datekey].push_back(cm->d_year);

	dataDrivenOperatorPlacement();
}


// 

void
QueryOptimizer::dataDrivenOperatorPlacement() {

	groupGPUcheck = true;
	joinGPUcheck = (bool*) malloc(join.size() * sizeof(bool));

	for (int i = 0; i < join.size(); i++) {
		for (int j = 0; j < groupby_build[join[i].second].size(); j++) {
			int tot_seg_in_GPU = groupby_build[join[i].second][j]->tot_seg_in_GPU;
			int total_segment = groupby_build[join[i].second][j]->total_segment;
			if (tot_seg_in_GPU < total_segment) {
				groupGPUcheck = false;
				break;
			}
		}
	}

	for (int i = 0; i < join.size(); i++) {
		if (join[i].second->tot_seg_in_GPU < join[i].second->total_segment) {
			joinGPUcheck[i] = false;
		} else {
			joinGPUcheck[i] = true;
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

		temp = temp << groupby_probe[cm->lo_orderdate].size();

		for (int j = 0; j < groupby_probe[cm->lo_orderdate].size(); j++) {
			ColumnInfo* column = groupby_probe[cm->lo_orderdate][j];
			bool isGPU = cm->segment_bitmap[column->column_id][i];
			//temp = temp | (isGPU << (join.size() - j - 1));
			temp = temp | (isGPU << j);
		}

		segment_group[cm->lo_orderdate->table_id][temp * cm->lo_orderdate->total_segment + segment_group_count[cm->lo_orderdate->table_id][temp]] = i;
		segment_group_count[cm->lo_orderdate->table_id][temp]++;
		//printf("temp = %d count = %d\n", temp, segment_group_count[0][temp]);
	}

	for (unsigned short i = 0; i < 64; i++) { //64 segment groups
		if (segment_group_count[cm->lo_orderdate->table_id][i] > 0) {

			unsigned short  bit = 1;
			unsigned short sg = i;
			for (int j = 0; j < groupby_probe[cm->lo_orderdate].size(); j++) {
				bit = (sg & (1 << j)) >> j;
				if (bit == 0) break;
			}

			for (int j = 0; j < groupby_probe[cm->lo_orderdate].size(); j++) {
				if (bit & groupGPUcheck) groupbyGPUPipelineCol[i].push_back(groupby_probe[cm->lo_orderdate][j]);
				else groupbyCPUPipelineCol[i].push_back(groupby_probe[cm->lo_orderdate][j]);
			}

			sg = sg >> groupby_probe[cm->lo_orderdate].size();

			//cout << sg << endl;

			for (int j = 0; j < join.size(); j++) {
				bit = (sg & (1 << j)) >> j;
				//printf("%d %d %d\n", j, bit, joinGPUcheck[j]);
				if (bit && joinGPUcheck[j]) joinGPUPipelineCol[i].push_back(join[j].first);
				else joinCPUPipelineCol[i].push_back(join[j].first);
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

	// for (int j = 0; j < 64; j++) {
	// 	if (segment_group_count[0][j] > 0) {
	// 		printf("%d %d\n", j, segment_group_count[0][j]);
	// 	}
	// 	for (int i = 0; i < joinGPUPipelineCol[j].size(); i++) {
	// 		cout << joinGPUPipelineCol[j][i]->column_name << endl;
	// 	}
	// }

	for (int i = 0; i < join.size(); i++) {
		
		for (int j = 0; j < join[i].second->total_segment; j++) {
			unsigned short temp = 0;

			for (int k = 0; k < select_build[join[i].second].size(); k++) {

				ColumnInfo* column = select_build[join[i].second][k];
				bool isGPU = cm->segment_bitmap[column->column_id][j];
				//temp = temp | (isGPU << (select_build[join[i].second].size() - k - 1));
				temp = temp | (isGPU << k);

			}

			segment_group[join[i].second->table_id][temp * join[i].second->total_segment + segment_group_count[join[i].second->table_id][temp]] = j;
			segment_group_count[join[i].second->table_id][temp]++;

		}
	}
}

// void 
// QueryOptimizer::latematerialization() {
// 	//(s, 14), (c, 14), (p, 15), (d, 16)

// 	//group-by

// 	int min = cm->lo_orderdate->total_segment;
	
// 	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = groupby_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU < min) {
// 			min = tot_seg_in_GPU;
// 		}
// 	}

// 	bool GPU = true;
// 	for (int i = 0; i < join.size(); i++) {
// 		for (int j = 0; j < groupby_build[join[i].second].size(); j++) {
// 			int tot_seg_in_GPU = groupby_build[join[i].second][j]->tot_seg_in_GPU;
// 			int total_segment = groupby_build[join[i].second][j]->total_segment;
// 			if (tot_seg_in_GPU < total_segment) {
// 				GPU = false;
// 				break;
// 			}
// 		}
// 	}

// 	if (GPU) {
// 		assert(min <= cm->lo_orderdate->total_segment);
// 		groupbyCPU = pair<int, int> (min, cm->lo_orderdate->total_segment);
// 		groupbyGPU = pair<int, int> (0, min);
// 	} else {
// 		groupbyCPU = pair<int, int> (0, cm->lo_orderdate->total_segment);
// 		groupbyGPU = pair<int, int> (0, 0);
// 	}

// 	//join

// 	multimap<int, ColumnInfo*> temp; //track how many segments can be executed in GPU from the fact table
// 	// temp is sorted from the column with least segment in GPU to column with most segment in GPU

// 	min = cm->lo_orderdate->total_segment;

// 	for (int i = 0; i < join.size(); i++) {
// 		int tot_seg_in_GPU = join[i].first->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU < min && tot_seg_in_GPU > 0) {
// 			min = tot_seg_in_GPU;
// 		}
// 	}

// 	for (int i = 0; i < join.size(); i++) {
// 		if (join[i].second->tot_seg_in_GPU < join[i].second->total_segment) {
// 			joinGPU.push_back(pair<int, int> (0, 0));
// 			joinCPU.push_back(pair<int, int> (0, join[i].first->total_segment));
// 			temp.insert(pair <int, ColumnInfo*> (0, join[i].first));
// 		} else if (join[i].first->tot_seg_in_GPU == 0) {
// 			joinGPU.push_back(pair<int, int> (0, 0));
// 			joinCPU.push_back(pair<int, int> (0, join[i].first->total_segment));
// 			temp.insert(pair <int, ColumnInfo*> (0, join[i].first));
// 		} else {
// 			assert(min <= join[i].first->total_segment);
// 			joinGPU.push_back(pair<int, int> (0, min));
// 			joinCPU.push_back(pair<int, int> (min, join[i].first->total_segment));
// 			temp.insert(pair <int, ColumnInfo*> (min, join[i].first));
// 		}
// 	}

// 	constructPipeline(joinCPUPipeline, joinGPUPipeline, joinCPUPipelineCol, joinGPUPipelineCol, temp, cm->lo_orderdate, join.size());

// 	//select build

// 	for (int i = 0; i < join.size(); i++) {
// 		int min = join[i].second->total_segment;
// 		ColumnInfo* column = join[i].second;

// 		for (int j = 0; j < select_build[column].size(); j++) {
// 			int tot_seg_in_GPU = select_build[column][j]->tot_seg_in_GPU;
// 			if (tot_seg_in_GPU < min) {
// 				min = tot_seg_in_GPU;
// 			}
// 		}

// 		for (int j = 0; j < select_build[column].size(); j++) {
// 			int total_segment = select_build[column][j]->total_segment;
// 			assert(min <= total_segment);
// 			selectbuildGPU[column].push_back(pair<int, int> (0, min));
// 			selectbuildCPU[column].push_back(pair<int, int> (min, total_segment));
// 		}
// 	}

// 	//select probe

// 	temp.clear();

// 	min = cm->lo_orderdate->total_segment;

// 	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = select_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU < min) {
// 			min = tot_seg_in_GPU;
// 		}
// 	}

// 	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
// 		int total_segment = select_probe[cm->lo_orderdate][i]->total_segment;
// 		assert(min <= total_segment);
// 		selectprobeGPU.push_back(pair<int, int> (0, min));
// 		selectprobeCPU.push_back(pair<int, int> (min, total_segment));
// 		temp.insert(pair<int, ColumnInfo*> (min, select_probe[cm->lo_orderdate][i]));
// 	}

// 	constructPipeline(selectCPUPipeline, selectGPUPipeline, selectCPUPipelineCol, selectGPUPipelineCol, temp, cm->lo_orderdate, select_probe.size());

// }



// void 
// QueryOptimizer::latematerializationflex() {
// 	//group-by

// 	int min = cm->lo_orderdate->total_segment;

// 	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = groupby_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
// 		if (tot_seg_in_GPU < min) {
// 			min = tot_seg_in_GPU;
// 		}
// 	}

// 	bool GPU = true;
// 	for (int i = 0; i < join.size(); i++) {
// 		for (int j = 0; j < groupby_build[join[i].second].size(); j++) {
// 			int tot_seg_in_GPU = groupby_build[join[i].second][j]->tot_seg_in_GPU;
// 			int total_segment = groupby_build[join[i].second][j]->total_segment;
// 			if (tot_seg_in_GPU < total_segment) {
// 				GPU = false;
// 				break;
// 			}
// 		}
// 	}

// 	if (GPU) {
// 		assert(min <= cm->lo_orderdate->total_segment);
// 		groupbyCPU = pair<int, int> (min, cm->lo_orderdate->total_segment);
// 		groupbyGPU = pair<int, int> (0, min);
// 	} else {
// 		groupbyCPU = pair<int, int> (0, cm->lo_orderdate->total_segment);
// 		groupbyGPU = pair<int, int> (0, 0);
// 	}

// 	//join

// 	multimap<int, ColumnInfo*> temp; //track how many segments can be executed in GPU from the fact table
// 	// temp is sorted from the column with least segment in GPU to column with most segment in GPU

// 	for (int i = 0; i < join.size(); i++) {
// 		if (join[i].second->tot_seg_in_GPU < join[i].second->total_segment) {
// 			joinGPU.push_back(pair<int, int> (0, 0));
// 			joinCPU.push_back(pair<int, int> (0, join[i].first->total_segment));
// 			temp.insert(pair <int, ColumnInfo*> (0, join[i].first));
// 		} else {
// 			int tot_seg_in_GPU = join[i].first->tot_seg_in_GPU;
// 			int total_segment = join[i].first->total_segment;
// 			assert(tot_seg_in_GPU <= total_segment);
// 			joinGPU.push_back(pair<int, int> (0, tot_seg_in_GPU));
// 			joinCPU.push_back(pair<int, int> (tot_seg_in_GPU, total_segment));
// 			temp.insert(pair <int, ColumnInfo*> (tot_seg_in_GPU, join[i].first));
// 		}
// 	}

// 	constructPipeline(joinCPUPipeline, joinGPUPipeline, joinCPUPipelineCol, joinGPUPipelineCol, temp, cm->lo_orderdate, join.size());

// 	//select hash

// 	for (int i = 0; i < join.size(); i++) {
// 		int min = join[i].second->total_segment;
// 		ColumnInfo* column = join[i].second;

// 		for (int j = 0; j < select_build[column].size(); j++) {
// 			int tot_seg_in_GPU = select_build[column][j]->tot_seg_in_GPU;
// 			if (tot_seg_in_GPU < min) {
// 				min = tot_seg_in_GPU;
// 			}
// 		}

// 		for (int j = 0; j < select_build[column].size(); j++) {
// 			int total_segment = select_build[column][j]->total_segment;
// 			assert(min <= total_segment);
// 			selectbuildGPU[column].push_back(pair<int, int> (0, min));
// 			selectbuildCPU[column].push_back(pair<int, int> (min, total_segment));
// 		}
// 	}

// 	//select probe

// 	temp.clear();

// 	for (int i = 0; i < select_probe[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = select_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
// 		int total_segment = select_probe[cm->lo_orderdate][i]->total_segment;
// 		assert(tot_seg_in_GPU <= total_segment);
// 		selectprobeGPU.push_back(pair<int, int> (0, tot_seg_in_GPU));
// 		selectprobeCPU.push_back(pair<int, int> (tot_seg_in_GPU, total_segment));
// 		temp.insert(pair <int, ColumnInfo*> (tot_seg_in_GPU, select_probe[cm->lo_orderdate][i]));
// 	}

// 	constructPipeline(selectCPUPipeline, selectGPUPipeline, selectCPUPipelineCol, selectGPUPipelineCol, temp, cm->lo_orderdate, select_probe.size());
// }

// void
// QueryOptimizer::constructPipeline(vector<pair<int, int>>& CPUPipeline, vector<pair<int, int>>& GPUPipeline, 
// 	vector<vector<ColumnInfo*>>& CPUPipelineCol, vector<vector<ColumnInfo*>>& GPUPipelineCol, multimap<int, ColumnInfo*>& temp, 
// 	ColumnInfo* column, int N) {

// 	GPUPipeline[N] = pair<int, int> (0, 0);
// 	CPUPipeline[0] = pair<int, int> (0, 0);
// 	GPUPipelineCol[N].push_back(NULL);
// 	CPUPipelineCol[0].push_back(NULL);

// 	multimap<int, ColumnInfo*>::iterator itr;
// 	multimap<int, ColumnInfo*>::reverse_iterator ritr;

// 	int previous = 0;
// 	int i = 0;
// 	for (itr = temp.begin(); itr != temp.end(); itr++) {
// 		//ABCD BCD CD D NULL
// 		for (int j = 0; j < i+1; j++) {
// 			GPUPipelineCol[j].push_back(itr->second);
// 		}

// 		//NULL A AB ABC ABCD
// 		for (int j = i+1; j <= join.size(); j++) {
// 			CPUPipelineCol[j].push_back(itr->second);
// 		}

// 		//filling select GPU pipeline
// 		if (previous < itr->first) {
// 			GPUPipeline[i] = pair<int, int> (previous, itr->first);
// 		} else if (previous == itr->first) {
// 			GPUPipeline[i] = pair<int, int> (0, 0);
// 		} else {
// 			assert(0);
// 		}
// 		assert(i < N);
// 		previous = itr->first;
// 		i++;
// 	}

// 	int total_segment = column->total_segment;
// 	previous = total_segment;
// 	i = N;
// 	//filling join CPU pipeline
// 	for (ritr = temp.rbegin(); ritr != temp.rend(); ritr++) {
// 		if (previous > ritr->first) {
// 			CPUPipeline[i] = pair<int, int> (ritr->first, previous);
// 		} else if (previous == ritr->first) {
// 			CPUPipeline[i] = pair<int, int> (0, 0);
// 		} else {
// 			assert(0);
// 		}
// 		assert(i > 0);
// 		previous = ritr->first;
// 		i--;
// 	}
// }

// void
// QueryOptimizer::patching() {
// 	int max = 0;

// 	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
// 		int tot_seg_in_GPU = groupby_probe[cm->lo_orderdate][i]->tot_seg_in_GPU;
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

// 	for (int i = 0; i < groupby_probe[cm->lo_orderdate].size(); i++) {
// 		ColumnInfo* column = groupby_probe[cm->lo_orderdate][i];
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