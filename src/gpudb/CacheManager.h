#ifndef _CACHE_MANAGER_H_
#define _CACHE_MANAGER_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <assert.h>

#include <curand.h>
#include <cuda.h>
#include <cub/util_allocator.cuh>
//#include <cub/test/test_util.h>
#include "crystal/crystal.cuh"
#include "ssb_utils.h"

//#include "QueryOptimizer.h"

using namespace std;

#define CUB_STDERR

//class QueryOptimizer;
class Statistics;
class CacheManager;
class Segment;
class ColumnInfo;
class priority_stack;
class custom_priority_queue;

enum ReplacementPolicy {
    LRU, LFU, LRU_v2, LFU_v2, New, New_v2, Segmented
};

class Statistics{
public:
	Statistics() {
		col_freq = 0;
		timestamp = 0;
		speedup = 0;
	};
	double col_freq;
	double timestamp;
	double speedup;
};

class Segment {
public:
	Segment(ColumnInfo* _column, int* _seg_ptr, int _priority);
	Segment(ColumnInfo* _column, int* _seg_ptr);
	ColumnInfo* column;
	int segment_id;
	int* col_ptr; //ptr to the beginning of the column
	int* seg_ptr; //ptr to the beginning of the segment
	int priority;
	int seg_size;
	double weight;

	Statistics* stats;
};

class ColumnInfo{
public:
	ColumnInfo(string _column_name, string _table_name, int _LEN, int _column_id, int _table_id, int* _col_ptr);
	Statistics* stats;
	string column_name;
	string table_name;
	int table_id;
	int LEN;
	int column_id;
	int* col_ptr; //ptr to the beginning of the column
	int* seg_ptr; //ptr to the last segment in the column
	int tot_seg_in_GPU; //total segments in GPU (based on current weight)
	double weight;
	int total_segment;

	Segment* getSegment(int index);
};

class priority_stack {
public:
	vector<Segment*> stack;
    bool empty() { return stack.size()==0; } 
    void push(Segment* x) { //const so that it can't be modified, passed by reference so that large object not get copied 
        stack.push_back(x);
        percolateDown();
    } 
    void pop() {
        if (!empty())
            stack.resize(stack.size()-1);
    }
    Segment* top() { 
        if (!empty()) 
        	return stack[stack.size()-1]; 
        else
        	return NULL;
    }
    void percolateDown() {
        for (int i=stack.size()-1; i>0; i--)
            if (stack[i]->priority > stack[i-1]->priority)
                swap(stack[i-1], stack[i]);
    }
    void percolateUp() {
    	for (int i=0; i<stack.size()-1; i++)
    		if (stack[i]->priority < stack[i+1]->priority)
    			swap(stack[i], stack[i+1]);
    }
    vector<Segment*> return_stack() {
    	return stack;
    }
};

class custom_priority_queue {
public:
	vector<Segment*> queue;
    bool empty() { return queue.size()==0; } 
    void push(Segment* x) {
        queue.push_back(x);
        percolateDown();
    } 
    void pop() {
        if (!empty())
            queue.erase(queue.begin());
    }
    Segment* front() { 
        if (!empty()) 
        	return queue[0]; 
        else
        	return NULL;
    }
    void percolateDown() {
        for (int i=queue.size()-1; i>0; i--)
            if (queue[i]->priority > queue[i-1]->priority)
                swap(queue[i-1], queue[i]);
    }
    void percolateUp() {
    	for (int i=0; i<queue.size()-1; i++)
    		if (queue[i]->priority < queue[i+1]->priority)
    			swap(queue[i], queue[i+1]);
    }
    vector<Segment*> return_queue() {
    	return queue;
    }
};

class CacheManager {
public:
	int* gpuCache;
	uint64_t* gpuProcessing, *cpuProcessing, *pinnedMemory;
	int gpuPointer, cpuPointer, pinnedPointer, onDemandPointer;
	int cache_total_seg, cache_size, processing_size, pinned_memsize, ondemand_size, ondemand_segment;
	int TOT_COLUMN;
	int TOT_TABLE;
	vector<ColumnInfo*> allColumn;

	queue<int> empty_gpu_segment; //free list
	vector<priority_stack> cached_seg_in_GPU; //track segments that are already cached in GPU
	int** segment_list; //segment list in GPU for each column
	unordered_map<Segment*, int> cache_mapper; //map segment to index in GPU
	// vector<custom_priority_queue> next_seg_to_cache; //a priority queue to store the special segment to be cached to GPU
	vector<vector<Segment*>> index_to_segment; //track which segment has been created from a particular segment id
	// vector<unordered_map<int, Segment*>> special_segment; //special segment id (segment with priority) to segment itself
	char** segment_bitmap; //bitmap to store information which segment is in GPU

	vector<vector<int>> columns_in_table;
	int** segment_min;
	int** segment_max;

	int *h_lo_orderkey, *h_lo_orderdate, *h_lo_custkey, *h_lo_suppkey, *h_lo_partkey, *h_lo_revenue, *h_lo_discount, *h_lo_quantity, *h_lo_extendedprice, *h_lo_supplycost;
	int *h_c_custkey, *h_c_nation, *h_c_region, *h_c_city;
	int *h_s_suppkey, *h_s_nation, *h_s_region, *h_s_city;
	int *h_p_partkey, *h_p_brand1, *h_p_category, *h_p_mfgr;
	int *h_d_datekey, *h_d_year, *h_d_yearmonthnum;

	ColumnInfo *lo_orderkey, *lo_orderdate, *lo_custkey, *lo_suppkey, *lo_partkey, *lo_revenue, *lo_discount, *lo_quantity, *lo_extendedprice, *lo_supplycost;
	ColumnInfo *c_custkey, *c_nation, *c_region, *c_city;
	ColumnInfo *s_suppkey, *s_nation, *s_region, *s_city;
	ColumnInfo *p_partkey, *p_brand1, *p_category, *p_mfgr;
	ColumnInfo *d_datekey, *d_year, *d_yearmonthnum;

	CacheManager(size_t cache_size, size_t ondemand_size, size_t _processing_size, size_t _pinned_memsize);

	void resetCache(size_t cache_size, size_t ondemand_size, size_t _processing_size, size_t _pinned_memsize);

	~CacheManager();

	void cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment);

	void deleteColumnSegmentInGPU(ColumnInfo* column, int total_segment);

	void cacheSegmentInGPU(Segment* seg);

	void deleteSegmentInGPU(Segment* seg);

	// void cacheSegmentFromQueue(ColumnInfo* column);

	void cacheListSegmentInGPU(vector<Segment*> v_seg);

	void deleteListSegmentInGPU(vector<Segment*> v_seg);

	//void constructListSegmentInGPU(ColumnInfo* column);

	void SegmentTablePriority(int table_id, int segment_idx, int priority);

	void updateSegmentPriority(Segment* seg, int priority);

	void updateSegmentInColumn(ColumnInfo* column);

	void updateColumnInGPU();

	void updateColumnFrequency(ColumnInfo* column);

	void updateColumnWeight(ColumnInfo* column, int freq, double speedup, double selectivity);

	void updateColumnWeightDirect(ColumnInfo* column, double speedup);

	void updateColumnTimestamp(ColumnInfo* column, double timestamp);

	void updateSegmentWeightDirect(ColumnInfo* column, Segment* segment, double speedup);

	void weightAdjustment();

	void runReplacement(ReplacementPolicy strategy);

	void LFUReplacement();

	void LFU2Replacement();

	void LRUReplacement();

	void LRU2Replacement();

	void NewReplacement();

	void New2Replacement();

	void SegmentReplacement();

	void loadColumnToCPU();

	template <typename T>
	T* customMalloc(int size);

	template <typename T>
	T* customCudaMalloc(int size);

	template <typename T>
	T* customCudaHostAlloc(int size);

	// int* customMalloc(int size);

	// int* customCudaMalloc(int size);

	// int* customCudaHostAlloc(int size);

	int* onDemandTransfer(int* data_ptr, int size, cudaStream_t stream);

	void indexTransfer(int** col_idx, ColumnInfo* column, cudaStream_t stream);

	void resetPointer();

	void resetOnDemand();

	void readSegmentMinMax();
};

#endif