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

using namespace std;

#define SEGMENT_SIZE 1000
#define CUB_STDERR

class Statistics;
class CacheManager;
class Segment;
class ColumnInfo;
template<class T> 
class priority_stack;

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

	Segment* getSegment(int index);
};

template<class T>
class priority_stack {
public:
	vector<T> stack;
    bool empty() { return stack.size()==0; } 
    void push(T x) { //const so that it can't be modified, passed by reference so that large object not get copied 
        stack.push_back(x);
        percolateDown();
    } 
    void pop() {
        if (!empty())
            stack.resize(stack.size()-1);
    }
    T top() { 
        if (!empty()) 
        	return stack[stack.size()-1]; 
        else
        	return NULL;
    }
    void percolateDown() {
        for (int i=stack.size()-1; i!=0; i--)
            if (stack[i]->priority > stack[i-1]->priority)
                swap(stack[i-1], stack[i]);
    }
    void percolateUp() {
    	for (int i=0; i<stack.size(); i++)
    		if (stack[i]->priority < stack[i+1]->priority)
    			swap(stack[i], stack[i+1]);
    }
    vector<T> return_stack() {
    	return stack;
    }
};

class OrderByPriority
{
public:
    bool operator() (Segment* a, Segment* b)  { 
		return a->priority < b->priority; 
	}
};

class CacheManager {
public:
	int* gpuCache;
	int cache_total_seg;
	int TOT_COLUMN;

	vector<ColumnInfo*> allColumn;
	queue<int> empty_gpu_segment; //free list
	vector<priority_stack<Segment*>> cached_seg_in_GPU; //track segments that are already cached in GPU
	int** segment_idx; //segment index in GPU for each column
	unordered_map<Segment*, int> cache_mapper; //map segment to index in GPU
	vector<priority_queue<Segment*, vector<Segment*>, OrderByPriority>> next_seg_to_cache; //a priority queue to store next segment to be cache to GPU
	vector<unordered_map<int, Segment*>> index_to_segment; //track which segment has been created from a particular segment id

	//int* metaCache; //metadata in GPU
	//int** meta_st_end; //no longer used, start and end index of metadata in GPU

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

	/*void sendMetadata() {
		int idx = 0;
		for (int i = 0; i < allColumn.size(); i++) {
			ColumnInfo* column = allColumn[i];
			CubDebugExit(cudaMemcpy(&metaCache[idx], &segment_idx[column->column_id], column->tot_seg_in_GPU * sizeof(int), cudaMemcpyHostToDevice));
			meta_st_end[column->column_id][0] = idx;
			meta_st_end[column->column_id][1] = idx + column->tot_seg_in_GPU - 1;
			idx += column->tot_seg_in_GPU;
		}
	}*/
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

	segment_idx = (int**) malloc (TOT_COLUMN * sizeof(int*));
	for (int i = 0; i < TOT_COLUMN; i++) {
		segment_idx[i] = (int*) malloc(cache_total_seg * sizeof(int));
	}

	next_seg_to_cache.resize(TOT_COLUMN);
	index_to_segment.resize(TOT_COLUMN);
	//CubDebugExit(cudaMalloc((void**) &metaCache, cache_total_seg * sizeof(int)));
	//meta_st_end = new int[TOT_COLUMN][2];
}

CacheManager::~CacheManager() {
	CubDebugExit(cudaFree((void**) &gpuCache));
	//CubDebugExit(cudaFree((void**) &metaCache));
}

void
CacheManager::cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
	for (int i = 0; i < total_segment; i++) {
		if (next_seg_to_cache[column->column_id].size() > 0) {
			cacheSegmentFromQueue(column);
		} else {
			int segment_idx = (column->seg_ptr - column->col_ptr)/SEGMENT_SIZE;
			while (index_to_segment[column->column_id].find(segment_idx) != index_to_segment[column->column_id].end()) {
				assert(cache_mapper.find(index_to_segment[column->column_id][segment_idx]) != cache_mapper.end());
				column->seg_ptr += SEGMENT_SIZE;
				segment_idx = (column->seg_ptr - column->col_ptr)/SEGMENT_SIZE;
			}
			Segment* seg = new Segment(column, column->seg_ptr);
			assert(segment_idx == seg->segment_id);
			index_to_segment[column->column_id][seg->segment_id] = seg;
			assert(cache_mapper.find(seg) == cache_mapper.end());
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
}

void
CacheManager::cacheSegmentFromQueue(ColumnInfo* column) {
	Segment* seg = next_seg_to_cache[column->column_id].top();
	next_seg_to_cache[column->column_id].pop();
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
	for (int i = 0; i < total_segment; i++) {
		Segment* seg = cached_seg_in_GPU[column->column_id].top();
		cached_seg_in_GPU[column->column_id].pop();
		assert(cache_mapper.find(seg) != cache_mapper.end());
		int idx = cache_mapper[seg];
		int ret = cache_mapper.erase(seg);
		assert(ret == 1);
		empty_gpu_segment.push(idx);
		column->tot_seg_in_GPU--;
		next_seg_to_cache[column->column_id].push(seg);
	}
}

void
CacheManager::constructListSegmentInGPU(ColumnInfo* column) {
	vector<Segment*> temp = cached_seg_in_GPU[column->column_id].return_stack();
	for (int i = 0; i < temp.size(); i++) {
		segment_idx[column->column_id][i] = cache_mapper[temp[i]];
	}
}

void
CacheManager::updateSegmentTablePriority(string table_name, int segment_idx, int priority) {
	for (int i = 0; i < TOT_COLUMN; i++) {
		int column_id = allColumn[i]->column_id;
		if (allColumn[i]->table_name == table_name) {
			if (index_to_segment[column_id].find(segment_idx) != index_to_segment[column_id].end()) { // the segment is created already
				Segment* seg = index_to_segment[column_id][segment_idx]; //get the segment
				updateSegmentPriority(seg, priority); //update priority of that segment (either in the queue or the stack)
				updateSegmentInColumn(allColumn[i]); //check the stack and update it accordingly
			} else { //the segment is not created already
				Segment* seg = allColumn[i]->getSegment(segment_idx);
				index_to_segment[column_id][segment_idx] = seg; //mark the segment is created
				next_seg_to_cache[column_id].push(seg); //push it to next seg to cache
				updateSegmentPriority(seg, priority); //update priority of that segment in the queue
				updateSegmentInColumn(allColumn[i]); //check the stack and update it accordingly
			}
		}
	}
}

void
CacheManager::updateSegmentPriority(Segment* seg, int priority) {
	if (cache_mapper.find(seg) != cache_mapper.end()) { // the segment is created and mapped to GPU
		if (priority > seg->priority) { //if the priority increase, do percolateDown
			seg->priority = priority;
			cached_seg_in_GPU[seg->column->column_id].percolateDown();
		} else if (priority < seg->priority) { //if the priority decrease, do percolateUp
			seg->priority = priority;
			cached_seg_in_GPU[seg->column->column_id].percolateUp();
		}
	} else { //the segment is created but not mapped to GPU (sit on the next_seg_to_cache queue), in this case just update the priority
		seg->priority = priority;
	}
}

void
CacheManager::updateSegmentInColumn(ColumnInfo* column) {
	if (next_seg_to_cache[column->column_id].empty()) return; //if no segment in the queue
	Segment* seg1 = next_seg_to_cache[column->column_id].top();
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

#endif