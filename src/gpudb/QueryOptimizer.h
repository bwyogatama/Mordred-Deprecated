#ifndef _QUERY_OPTIMIZER_H_
#define _QUERY_OPTIMIZER_H_

#include "CacheManager.cuh"
#include "KernelArgs.h"

#define NUM_QUERIES 13
#define MAX_GROUPS 128

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

	vector<vector<ColumnInfo*>> queryColumn;

	bool groupGPUcheck;
	bool joinGPUall;
	bool* joinGPUcheck, *joinCPUcheck, **joinGPU, **joinCPU;

	short** segment_group, **segment_group_temp;
	short** segment_group_count, **segment_group_temp_count;
	short** par_segment;
	short* par_segment_count;
	int* last_segment;

	map<int, map<ColumnInfo*, double>> speedup;

	QueryParams* params;

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

	void prepareQuery(int query);

	void clearParsing();
	void clearPlacement();
	void clearPrepare();

	void dataDrivenOperatorPlacement(int query, bool isprofile = 0);
	void prepareOperatorPlacement();
	void groupBitmap(bool isprofile = 0);
	void groupBitmapSegment(int query, bool isprofile = 0);
	void groupBitmapSegmentTable(int table_id, int query, bool isprofile = 0);
	void groupBitmapSegmentTableOD(int table_id, int query, bool isprofile = 0);

	bool checkPredicate(int table_id, int segment_idx);
	void updateSegmentStats(int table_id, int segment_idx, int query);

};

#endif