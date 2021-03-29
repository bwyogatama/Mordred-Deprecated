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
#include "cub/test/test_util.h"
#include "crystal/crystal.cuh"
#include "ssb_utils.h"

//#include "QueryOptimizer.h"

using namespace std;

#define SEGMENT_SIZE 1000
#define CUB_STDERR

//class QueryOptimizer;
class Statistics;
class CacheManager;
class Segment;
class ColumnInfo;
class priority_stack;
class custom_priority_queue;

enum GPUBufferManagementStrategy {
    LEAST_RECENTLY_USED, LEAST_FREQUENTLY_USED, ON_DEMAND, DISABLED_GPU_BUFFER
};

class Statistics{
public:
	Statistics() {
		col_freq = 0;
		query_freq = 0;
		timestamp = 0;
	};
	int col_freq;
	int query_freq;
	uint64_t timestamp;
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
};

class ColumnInfo{
public:
	ColumnInfo(string _column_name, string _table_name, int _LEN, int _column_id, int* _col_ptr);
	Statistics* stats;
	string column_name;
	string table_name;
	int LEN;
	int column_id;
	int* col_ptr; //ptr to the beginning of the column
	int* seg_ptr; //ptr to the last segment in the column
	int tot_seg_in_GPU; //total segments in GPU (based on current weight)
	float weight;
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
    void push(Segment* x) { //const so that it can't be modified, passed by reference so that large object not get copied 
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
	int cache_total_seg;
	int TOT_COLUMN;
	vector<ColumnInfo*> allColumn;

	queue<int> empty_gpu_segment; //free list
	vector<priority_stack> cached_seg_in_GPU; //track segments that are already cached in GPU
	int** segment_list; //segment list in GPU for each column
	unordered_map<Segment*, int> cache_mapper; //map segment to index in GPU
	vector<custom_priority_queue> next_seg_to_cache; //a priority queue to store the special segment to be cached to GPU
	vector<unordered_map<int, Segment*>> index_to_segment; //track which segment has been created from a particular segment id
	vector<unordered_map<int, Segment*>> special_segment; //special segment id (segment with priority) to segment itself

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

	CacheManager(size_t cache_size, int _TOT_COLUMN);

	~CacheManager();

	void cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment);

	void cacheSegmentInGPU(Segment* seg);

	void cacheSegmentFromQueue(ColumnInfo* column);

	void cacheListSegmentInGPU(vector<Segment*> v_seg);

	void deleteColumnSegmentInGPU(ColumnInfo* column, int total_segment);

	void constructListSegmentInGPU(ColumnInfo* column);

	void updateSegmentTablePriority(string table_name, int segment_idx, int priority);

	void updateSegmentPriority(Segment* seg, int priority);

	void updateSegmentInColumn(ColumnInfo* column);

	void updateColumnWeight();

	void updateColumnInGPU();

	void weightAdjustment();

	void loadColumnToCPU();
};

Segment::Segment(ColumnInfo* _column, int* _seg_ptr, int _priority)
: column(_column), seg_ptr(_seg_ptr), priority(_priority), seg_size(SEGMENT_SIZE) {
	col_ptr = column->col_ptr;
	segment_id = (seg_ptr - col_ptr)/seg_size;
}

Segment::Segment(ColumnInfo* _column, int* _seg_ptr)
: column(_column), seg_ptr(_seg_ptr), priority(0), seg_size(SEGMENT_SIZE) {
	col_ptr = column->col_ptr;
	segment_id = (seg_ptr - col_ptr)/seg_size;
}

ColumnInfo::ColumnInfo(string _column_name, string _table_name, int _LEN, int _column_id, int* _col_ptr)
: column_name(_column_name), table_name(_table_name), LEN(_LEN), column_id(_column_id), col_ptr(_col_ptr) {
	stats = new Statistics();
	tot_seg_in_GPU = 0;
	weight = 0;
	seg_ptr = col_ptr;
	total_segment = (LEN+SEGMENT_SIZE-1)/SEGMENT_SIZE;
}

Segment*
ColumnInfo::getSegment(int index) {
	Segment* seg = new Segment(this, col_ptr+SEGMENT_SIZE*index, 0);
	return seg;
}

CacheManager::CacheManager(size_t cache_size, int _TOT_COLUMN) {
	cache_total_seg = cache_size/SEGMENT_SIZE;
	TOT_COLUMN = _TOT_COLUMN;

	CubDebugExit(cudaMalloc((void**) &gpuCache, cache_size * sizeof(int)));

	cached_seg_in_GPU.resize(TOT_COLUMN);
	allColumn.resize(TOT_COLUMN);

	segment_list = (int**) malloc (TOT_COLUMN * sizeof(int*));
	for (int i = 0; i < TOT_COLUMN; i++) {
		segment_list[i] = (int*) malloc(cache_total_seg * sizeof(int));
	}

	next_seg_to_cache.resize(TOT_COLUMN);
	index_to_segment.resize(TOT_COLUMN);
	special_segment.resize(TOT_COLUMN);

	loadColumnToCPU();
}

CacheManager::~CacheManager() {
	CubDebugExit(cudaFree((void**) &gpuCache));
}

void
CacheManager::cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
	assert(column->tot_seg_in_GPU + total_segment <= column->total_segment);
	for (int i = 0; i < total_segment; i++) {
		if (!next_seg_to_cache[column->column_id].empty()) {
			cacheSegmentFromQueue(column);
		} else {
			int segment_idx = (column->seg_ptr - column->col_ptr)/SEGMENT_SIZE;
			if (segment_idx >= column->total_segment) {
				segment_idx = 0;
				column->seg_ptr = column->col_ptr;
			}
			while (special_segment[column->column_id].find(segment_idx) != special_segment[column->column_id].end()) { //selama next segment pointer masih termasuk special segment
				assert(cache_mapper.find(index_to_segment[column->column_id][segment_idx]) != cache_mapper.end());
				assert(segment_idx < column->total_segment); //will have to delete this later
				segment_idx++;
				column->seg_ptr += SEGMENT_SIZE;
				if (segment_idx >= column->total_segment) {
					segment_idx = 0;
					column->seg_ptr = column->col_ptr;
				}
			}
			Segment* seg = new Segment(column, column->seg_ptr);
			assert(segment_idx == seg->segment_id);
			assert(seg->priority == 0);
			index_to_segment[column->column_id][seg->segment_id] = seg;
			column->seg_ptr += SEGMENT_SIZE;
			cacheSegmentInGPU(seg);
		}
	}
}

void 
CacheManager::cacheSegmentInGPU(Segment* seg) {
	int idx = empty_gpu_segment.front();
	empty_gpu_segment.pop();
	assert(cache_mapper.find(seg) == cache_mapper.end());
	cache_mapper[seg] = idx;
	cached_seg_in_GPU[seg->column->column_id].push(seg);
	CubDebugExit(cudaMemcpy(&gpuCache[idx * SEGMENT_SIZE], seg->seg_ptr, SEGMENT_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	allColumn[seg->column->column_id]->tot_seg_in_GPU++;
	assert(allColumn[seg->column->column_id]->tot_seg_in_GPU <= allColumn[seg->column->column_id]->total_segment);
}

void
CacheManager::cacheSegmentFromQueue(ColumnInfo* column) {
	Segment* seg = next_seg_to_cache[column->column_id].front();
	next_seg_to_cache[column->column_id].pop();
	assert(special_segment[column->column_id].find(seg->segment_id) != special_segment[column->column_id].end());
	cacheSegmentInGPU(seg);
}

void
CacheManager::cacheListSegmentInGPU(vector<Segment*> v_seg) {
	for (int i = 0; i < v_seg.size(); i++) {
		cacheSegmentInGPU(v_seg[i]);
	}
}

void
CacheManager::deleteColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
	assert(column->tot_seg_in_GPU - total_segment >= 0);
	for (int i = 0; i < total_segment; i++) {
		Segment* seg = cached_seg_in_GPU[column->column_id].top();
		cached_seg_in_GPU[column->column_id].pop();
		assert(cache_mapper.find(seg) != cache_mapper.end());
		int idx = cache_mapper[seg];
		int ret = cache_mapper.erase(seg);
		assert(ret == 1);
		empty_gpu_segment.push(idx);
		column->tot_seg_in_GPU--;
		if (special_segment[column->column_id].find(seg->segment_id) != special_segment[column->column_id].end()) {
			assert(seg->priority > 0);
			next_seg_to_cache[column->column_id].push(seg);
		} else {
			assert(seg->priority == 0);
			delete seg;
			column->seg_ptr -= SEGMENT_SIZE;
			int segment_idx = (column->seg_ptr - column->col_ptr)/SEGMENT_SIZE;
			while (special_segment[column->column_id].find(segment_idx) != special_segment[column->column_id].end()) { //selama next segment pointer masih termasuk special segment
				segment_idx--;
				column->seg_ptr -= SEGMENT_SIZE;
				assert(segment_idx > 0);
			}
			assert(cache_mapper.find(index_to_segment[column->column_id][segment_idx]) != cache_mapper.end());
		}
	}
}

void
CacheManager::constructListSegmentInGPU(ColumnInfo* column) {
	vector<Segment*> temp = cached_seg_in_GPU[column->column_id].return_stack();
	for (int i = 0; i < temp.size(); i++) {
		segment_list[column->column_id][i] = cache_mapper[temp[i]];
	}
}

void
CacheManager::updateSegmentTablePriority(string table_name, int segment_idx, int priority) {
	for (int i = 0; i < TOT_COLUMN; i++) {
		int column_id = allColumn[i]->column_id;
		if (allColumn[i]->table_name == table_name) {
			if (index_to_segment[column_id].find(segment_idx) != index_to_segment[column_id].end()) { // the segment is created already
				Segment* seg = index_to_segment[column_id][segment_idx]; //get the segment
				if (special_segment[column_id].find(segment_idx) == special_segment[column_id].end()) { // kalau segment sudah dibuat tapi bukan special segment (segment biasa yg udah dicache)
					assert(cache_mapper.find(seg) != cache_mapper.end());
					assert(priority > 0);
					special_segment[column_id][segment_idx] = seg; //is a special segment now
				}
				updateSegmentPriority(seg, priority); //update priority of that segment (either in the queue or the stack)
				updateSegmentInColumn(allColumn[i]); //check the stack and update it accordingly
			} else { //the segment is not created already
				Segment* seg = allColumn[i]->getSegment(segment_idx);
				index_to_segment[column_id][segment_idx] = seg; //mark the segment is created
				assert(priority > 0);
				next_seg_to_cache[column_id].push(seg); //push it to next seg to cache
				special_segment[column_id][segment_idx] = seg; //is a special segment now
				updateSegmentInColumn(allColumn[i]); //check the stack and update it accordingly
			}
		}
	}
}

void
CacheManager::updateSegmentPriority(Segment* seg, int priority) {
	assert(special_segment[seg->column->column_id].find(seg->segment_id) != special_segment[seg->column->column_id].end());
	if (cache_mapper.find(seg) != cache_mapper.end()) { // the segment is created and mapped to GPU
		if (priority > seg->priority) { //if the priority increase, do percolateDown
			seg->priority = priority;
			cached_seg_in_GPU[seg->column->column_id].percolateDown();
		} else if (priority < seg->priority) { //if the priority decrease, do percolateUp
			seg->priority = priority;
			cached_seg_in_GPU[seg->column->column_id].percolateUp();
		}
		if (priority == 0) { //priority diturunkan jadi segment biasa
			special_segment[seg->column->column_id].erase(seg->segment_id);
		}
	} else { //the segment is created but not mapped to GPU (sit on the next_seg_to_cache queue), in this case just update the priority
		if (priority == 0) { //if the priority becomes 0, we need to get rid of this segment from next seg to cache queue
			priority = 100000; // assign a very big number
		}

		if (priority > seg->priority) { //if the priority increase, do percolateDown
			seg->priority = priority;
			next_seg_to_cache[seg->column->column_id].percolateDown();
		} else if (priority < seg->priority) { //if the priority decrease, do percolateUp
			seg->priority = priority;
			next_seg_to_cache[seg->column->column_id].percolateUp();
		}
		if (priority == 100000) { //priority diturunkan jadi segment biasa
			special_segment[seg->column->column_id].erase(seg->segment_id);
			next_seg_to_cache[seg->column->column_id].pop();
		}
	}

}

void
CacheManager::updateSegmentInColumn(ColumnInfo* column) {
	if (next_seg_to_cache[column->column_id].empty()) return; //if no segment in the queue
	Segment* seg1 = next_seg_to_cache[column->column_id].front();
	Segment* seg2 = cached_seg_in_GPU[column->column_id].top();
	if (seg1->priority > seg2->priority) { //if the priority of segment in the queue is higher than the stack
		deleteColumnSegmentInGPU(column, 1); //delete segment in the stack, push it to the queue
		cacheSegmentFromQueue(column); //pop segment in the queue, push it to the stack
	}
}

void
CacheManager::updateColumnWeight() {
	for (int i = 0; i < TOT_COLUMN; i++) {
		allColumn[i]->weight = allColumn[i]->stats->col_freq;
	}
}

void
CacheManager::updateColumnInGPU() {
	int sum = 0;
	for (int i = 0; i < TOT_COLUMN; i++) {
		sum += allColumn[i]->weight;
	}

	weightAdjustment();

	for (int i = 0; i < TOT_COLUMN; i++) {
		ColumnInfo* column = allColumn[i];
		int temp = column->weight * cache_total_seg / sum;
		if (temp > column->tot_seg_in_GPU) {
			cacheColumnSegmentInGPU(column, temp - column->tot_seg_in_GPU);
		} else {
			deleteColumnSegmentInGPU(column, column->tot_seg_in_GPU - temp);
		}
	}
	//sendMetadata();
}

void
CacheManager::weightAdjustment() {
	float temp[TOT_COLUMN];
	bool check[TOT_COLUMN], stop;
	float remainder, sum_temp;
	float sum = 0;

	for (int i = 0; i < TOT_COLUMN; i++) {
		sum += allColumn[i]->weight;
		temp[i] = 0;
		check[i] = 0;
	}

	do {
		remainder = 0;
		for (int i = 0; i < TOT_COLUMN; i++) {
			if (allColumn[i]->weight > 0) {
				temp[i] = allColumn[i]->LEN * 1.0 /(cache_total_seg*SEGMENT_SIZE) * sum; //ngitung max weight kalo sesuai ukuran kolomnya
				if (allColumn[i]->weight > temp[i]) {
					remainder += (allColumn[i]->weight - temp[i]); //lebihnya kasih ke remainder
					allColumn[i]->weight = temp[i];
					check[i] = 1;
				}
			}
		}

		sum_temp = 0;
		stop = 1;
		for (int i = 0; i < TOT_COLUMN; i++) {
			if (allColumn[i]->weight > 0) {
				if (check[i] == 0) {
					sum_temp += allColumn[i]->weight;
					stop = 0;
				}
			}
		}

		for (int i = 0; i < TOT_COLUMN; i++) {
			if (allColumn[i]->weight > 0) {
				if (check[i] == 0) {
					allColumn[i]->weight += (allColumn[i]->weight * 1.0/sum_temp * remainder); //tiap kolom dapet porsi dari remainder seuai sama weightnya, makin tinggi weight, makin banyak dapet remainder
				}
			}
		}
	} while (remainder != 0 && !stop);
}

void
CacheManager::loadColumnToCPU() {
	h_lo_orderkey = loadColumn<int>("lo_orderkey", LO_LEN);
	h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
	h_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
	h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
	h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
	h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
	h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
	h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
	h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
	h_lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);

	h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
	h_c_nation = loadColumn<int>("c_nation", C_LEN);
	h_c_region = loadColumn<int>("c_region", C_LEN);
	h_c_city = loadColumn<int>("c_city", C_LEN);

	h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
	h_s_nation = loadColumn<int>("s_nation", S_LEN);
	h_s_region = loadColumn<int>("s_region", S_LEN);
	h_s_city = loadColumn<int>("s_city", S_LEN);

	h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
	h_p_brand1 = loadColumn<int>("p_brand1", P_LEN);
	h_p_category = loadColumn<int>("p_category", P_LEN);
	h_p_mfgr = loadColumn<int>("p_mfgr", P_LEN);

	h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
	h_d_year = loadColumn<int>("d_year", D_LEN);
	h_d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

	lo_orderkey = new ColumnInfo("lo_orderkey", "lo", LO_LEN, 0, h_lo_orderkey);
	lo_orderdate = new ColumnInfo("lo_orderdate", "lo", LO_LEN, 1, h_lo_orderdate);
	lo_custkey = new ColumnInfo("lo_custkey", "lo", LO_LEN, 2, h_lo_custkey);
	lo_suppkey = new ColumnInfo("lo_suppkey", "lo", LO_LEN, 3, h_lo_suppkey);
	lo_partkey = new ColumnInfo("lo_partkey", "lo", LO_LEN, 4, h_lo_partkey);
	lo_revenue = new ColumnInfo("lo_revenue", "lo", LO_LEN, 5, h_lo_revenue);
	lo_discount = new ColumnInfo("lo_discount", "lo", LO_LEN, 6, h_lo_discount);
	lo_quantity = new ColumnInfo("lo_quantity", "lo", LO_LEN, 7, h_lo_quantity);
	lo_extendedprice = new ColumnInfo("lo_extendedprice", "lo", LO_LEN, 8, h_lo_extendedprice);
	lo_supplycost = new ColumnInfo("lo_supplycost", "lo", LO_LEN, 9, h_lo_supplycost);

	c_custkey = new ColumnInfo("c_custkey", "c", C_LEN, 10, h_c_custkey);
	c_nation = new ColumnInfo("c_nation", "c", C_LEN, 11, h_c_nation);
	c_region = new ColumnInfo("c_region", "c", C_LEN, 12, h_c_region);
	c_city = new ColumnInfo("c_city", "c", C_LEN, 13, h_c_city);

	s_suppkey = new ColumnInfo("s_suppkey", "s", S_LEN, 14, h_s_suppkey);	
	s_nation = new ColumnInfo("s_nation", "s", S_LEN, 15, h_s_nation);
	s_region = new ColumnInfo("s_region", "s", S_LEN, 16, h_s_region);
	s_city = new ColumnInfo("s_city", "s", S_LEN, 17, h_s_city);

	p_partkey = new ColumnInfo("p_partkey", "p", P_LEN, 18, h_p_partkey);
	p_brand1 = new ColumnInfo("p_brand1", "p", P_LEN, 19, h_p_brand1);
	p_category = new ColumnInfo("p_category", "p", P_LEN, 20, h_p_category);
	p_mfgr = new ColumnInfo("p_mfgr", "p", P_LEN, 21, h_p_mfgr);

	d_datekey = new ColumnInfo("d_datekey", "d", D_LEN, 22, h_d_datekey);
	d_year = new ColumnInfo("d_year", "d", D_LEN, 23, h_d_year);
	d_yearmonthnum = new ColumnInfo("d_yearmonthnum", "d", D_LEN, 24, h_d_yearmonthnum);

	allColumn[0] = lo_orderkey;
	allColumn[1] = lo_orderdate;
	allColumn[2] = lo_custkey;
	allColumn[3] = lo_suppkey;
	allColumn[4] = lo_partkey;
	allColumn[5] = lo_revenue;
	allColumn[6] = lo_discount;
	allColumn[7] = lo_quantity;
	allColumn[8] = lo_extendedprice;
	allColumn[9] = lo_supplycost;

	allColumn[10] = c_custkey;
	allColumn[11] = c_nation;
	allColumn[12] = c_region;
	allColumn[13] = c_city;

	allColumn[14] = s_suppkey;
	allColumn[15] = s_nation;
	allColumn[16] = s_region;
	allColumn[17] = s_city;

	allColumn[18] = p_partkey;
	allColumn[19] = p_brand1;
	allColumn[20] = p_category;
	allColumn[21] = p_mfgr;

	allColumn[22] = d_datekey;
	allColumn[23] = d_year;
	allColumn[24] = d_yearmonthnum;
}

#endif