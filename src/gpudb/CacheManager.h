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
	uint64_t col_freq;
	uint64_t query_freq;
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
	int* gpuProcessing, *cpuProcessing;
	int gpuPointer, cpuPointer;
	int cache_total_seg, processing_size;
	int TOT_COLUMN;
	int TOT_TABLE;
	vector<ColumnInfo*> allColumn;

	queue<int> empty_gpu_segment; //free list
	vector<priority_stack> cached_seg_in_GPU; //track segments that are already cached in GPU
	int** segment_list; //segment list in GPU for each column
	unordered_map<Segment*, int> cache_mapper; //map segment to index in GPU
	vector<custom_priority_queue> next_seg_to_cache; //a priority queue to store the special segment to be cached to GPU
	vector<unordered_map<int, Segment*>> index_to_segment; //track which segment has been created from a particular segment id
	vector<unordered_map<int, Segment*>> special_segment; //special segment id (segment with priority) to segment itself
	char** segment_bitmap; //bitmap to store information which segment is in GPU

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

	CacheManager(size_t cache_size, size_t _processing_size);

	~CacheManager();

	void cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment);

	void cacheSegmentInGPU(Segment* seg);

	void cacheSegmentFromQueue(ColumnInfo* column);

	void cacheListSegmentInGPU(vector<Segment*> v_seg);

	void deleteColumnSegmentInGPU(ColumnInfo* column, int total_segment);

	//void constructListSegmentInGPU(ColumnInfo* column);

	void updateSegmentTablePriority(int table_id, int segment_idx, int priority);

	void updateSegmentPriority(Segment* seg, int priority);

	void updateSegmentInColumn(ColumnInfo* column);

	void updateColumnWeight();

	void updateColumnInGPU();

	void updateColumnFrequency(ColumnInfo* column);

	void updateQueryFrequency(ColumnInfo* column, int freq);

	void updateColumnTimestamp(ColumnInfo* column, double timestamp);

	void weightAdjustment();

	void runReplacement(int strategy);

	void loadColumnToCPU();

	int* customMalloc(int size);

	int* customCudaMalloc(int size);
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

ColumnInfo::ColumnInfo(string _column_name, string _table_name, int _LEN, int _column_id, int _table_id, int* _col_ptr)
: column_name(_column_name), table_name(_table_name), LEN(_LEN), column_id(_column_id), table_id(_table_id), col_ptr(_col_ptr) {
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

CacheManager::CacheManager(size_t cache_size, size_t _processing_size) {
	cache_total_seg = cache_size/SEGMENT_SIZE;
	processing_size = _processing_size;
	TOT_COLUMN = 25;
	TOT_TABLE = 5;

	CubDebugExit(cudaMalloc((void**) &gpuCache, cache_size * sizeof(int)));
	CubDebugExit(cudaMemset(gpuCache, 0, cache_size * sizeof(int)));
	CubDebugExit(cudaMalloc((void**) &gpuProcessing, _processing_size * sizeof(int)));
	cpuProcessing = (int*) malloc(_processing_size * sizeof(int));
	gpuPointer = 0;
	cpuPointer = 0;

	cached_seg_in_GPU.resize(TOT_COLUMN);
	allColumn.resize(TOT_COLUMN);

	next_seg_to_cache.resize(TOT_COLUMN);
	index_to_segment.resize(TOT_COLUMN);
	special_segment.resize(TOT_COLUMN);

	for(int i = 0; i < cache_total_seg; i++) {
		empty_gpu_segment.push(i);
	}

	loadColumnToCPU();

	segment_bitmap = (char**) malloc (TOT_COLUMN * sizeof(char*));
	segment_list = (int**) malloc (TOT_COLUMN * sizeof(int*));
	for (int i = 0; i < TOT_COLUMN; i++) {
		int n = allColumn[i]->total_segment;
		segment_bitmap[i] = (char*) malloc(n * sizeof(char));
		segment_list[i] = (int*) malloc(n * sizeof(int));
		memset(segment_bitmap[i], 0, n * sizeof(char));
		memset(segment_list[i], -1, n * sizeof(int));
	}
}

int* 
CacheManager::customMalloc(int size) {
  // printf("%d\n", size);
  int start = __atomic_fetch_add(&cpuPointer, size, __ATOMIC_RELAXED);
  // printf("%d\n", start + size);
  assert(start + size < processing_size);
  return cpuProcessing + start;
};

int*
CacheManager::customCudaMalloc(int size) {
  int start = __atomic_fetch_add(&gpuPointer, size, __ATOMIC_RELAXED);
  assert(start + size < processing_size);
  return gpuProcessing + start;
};

void
CacheManager::cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
	assert(column->tot_seg_in_GPU + total_segment <= column->total_segment);
	for (int i = 0; i < total_segment; i++) {
		if (!next_seg_to_cache[column->column_id].empty()) {
			cacheSegmentFromQueue(column);
		} else {
			int segment_idx = (column->seg_ptr - column->col_ptr)/SEGMENT_SIZE;
			//cout << segment_idx << endl;
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
			//printf("%d\n", i);
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
	assert(segment_bitmap[seg->column->column_id][seg->segment_id] == 0x00);
	segment_bitmap[seg->column->column_id][seg->segment_id] = 0x01;
	assert(segment_list[seg->column->column_id][seg->segment_id] == -1);
	segment_list[seg->column->column_id][seg->segment_id] = idx;
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
		assert(segment_bitmap[column->column_id][seg->segment_id] == 0x01);
		segment_bitmap[column->column_id][seg->segment_id] = 0x00;
		assert(segment_list[seg->column->column_id][seg->segment_id] != -1);
		segment_list[column->column_id][seg->segment_id] = -1;
		empty_gpu_segment.push(idx);
		column->tot_seg_in_GPU--;
		if (special_segment[column->column_id].find(seg->segment_id) != special_segment[column->column_id].end()) {
			assert(seg->priority > 0);
			next_seg_to_cache[column->column_id].push(seg);
		} else {
			assert(seg->priority == 0);
			delete seg;
			column->seg_ptr -= SEGMENT_SIZE;
			int segment_idx = (column->seg_ptr - SEGMENT_SIZE - column->col_ptr)/SEGMENT_SIZE;
			while (special_segment[column->column_id].find(segment_idx) != special_segment[column->column_id].end()) { //selama next segment pointer masih termasuk special segment
				segment_idx--;
				column->seg_ptr -= SEGMENT_SIZE;
				assert(segment_idx >= 0);
			}
			if (segment_idx < 0) {
				column->seg_ptr = column->col_ptr;
				segment_idx = 0;
			} else {
				assert(cache_mapper.find(index_to_segment[column->column_id][segment_idx]) != cache_mapper.end());				
			}
			//cout << segment_idx << endl;

		}
	}
}

// void
// CacheManager::constructListSegmentInGPU(ColumnInfo* column) {
// 	vector<Segment*> temp = cached_seg_in_GPU[column->column_id].return_stack();
// 	delete segment_list[column->column_id];
// 	//segment_list[column->column_id] = (int*) malloc(temp.size() * sizeof(int));
// 	segment_list[column->column_id] = (int*) malloc(cache_total_seg * sizeof(int));
// 	for (int i = 0; i < temp.size(); i++) {
// 		segment_list[column->column_id][i] = cache_mapper[temp[i]];
// 	}
// }

void
CacheManager::updateSegmentTablePriority(int table_id, int segment_idx, int priority) {
	for (int i = 0; i < TOT_COLUMN; i++) {
		int column_id = allColumn[i]->column_id;
		if (allColumn[i]->table_id == table_id) {
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
CacheManager::updateColumnFrequency(ColumnInfo* column) {
	column->stats->col_freq++;
}

void
CacheManager::updateQueryFrequency(ColumnInfo* column, int freq) {
	if (freq > column->stats->query_freq) {
		column->stats->query_freq = freq;
	}
}

void
CacheManager::updateColumnTimestamp(ColumnInfo* column, double timestamp) {
	column->stats->timestamp = timestamp;
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
CacheManager::runReplacement(int strategy) {
	std::multimap<uint64_t, ColumnInfo*> access_frequency_map;
	std::map<ColumnInfo*, uint64_t>::const_iterator it;

	if (strategy == 0) { //LEAST FREQUENTLY USED
		for (int i = 0; i < TOT_COLUMN; i++) {
			access_frequency_map.insert({allColumn[i]->stats->col_freq, allColumn[i]});
		}
	} else if (strategy == 1) { //LEAST RECENTLY USED
		for (int i = 0; i < TOT_COLUMN; i++) {
			access_frequency_map.insert({allColumn[i]->stats->timestamp, allColumn[i]});
		}
	}

	int temp_buffer_size = 0; // in segment
	std::set<ColumnInfo*> columns_to_place;
	std::multimap<uint64_t, ColumnInfo*>::reverse_iterator cit;

    for(cit=access_frequency_map.rbegin(); cit!=access_frequency_map.rend(); ++cit){
        if(temp_buffer_size + cit->second->total_segment < cache_total_seg && cit->first>0){
            temp_buffer_size+=cit->second->total_segment;
            columns_to_place.insert(cit->second);
        }
    }

    cout << "Cached segment: " << temp_buffer_size << " Cache total: " << cache_total_seg << endl;
    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->tot_seg_in_GPU > 0 && columns_to_place.find(allColumn[i]) == columns_to_place.end()) {
			cout << "Deleting column ";
			cout << allColumn[i]->column_name << endl;
			deleteColumnSegmentInGPU(allColumn[i], allColumn[i]->tot_seg_in_GPU);
		}
    }

    std::set<ColumnInfo*>::const_iterator cit2;
    for(cit2 = columns_to_place.cbegin();cit2 != columns_to_place.cend(); ++cit2){
    	if ((*cit2)->tot_seg_in_GPU == 0) {
			cout << "Caching column ";
			cout << (*cit2)->column_name << endl;
    		cacheColumnSegmentInGPU(*cit2, (*cit2)->total_segment);
    	}
    }

};

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


	lo_orderkey = new ColumnInfo("lo_orderkey", "lo", LO_LEN, 0, 0, h_lo_orderkey);
	lo_orderdate = new ColumnInfo("lo_orderdate", "lo", LO_LEN, 1, 0, h_lo_orderdate);
	lo_custkey = new ColumnInfo("lo_custkey", "lo", LO_LEN, 2, 0, h_lo_custkey);
	lo_suppkey = new ColumnInfo("lo_suppkey", "lo", LO_LEN, 3, 0, h_lo_suppkey);
	lo_partkey = new ColumnInfo("lo_partkey", "lo", LO_LEN, 4, 0, h_lo_partkey);
	lo_revenue = new ColumnInfo("lo_revenue", "lo", LO_LEN, 5, 0, h_lo_revenue);
	lo_discount = new ColumnInfo("lo_discount", "lo", LO_LEN, 6, 0, h_lo_discount);
	lo_quantity = new ColumnInfo("lo_quantity", "lo", LO_LEN, 7, 0, h_lo_quantity);
	lo_extendedprice = new ColumnInfo("lo_extendedprice", "lo", LO_LEN, 8, 0, h_lo_extendedprice);
	lo_supplycost = new ColumnInfo("lo_supplycost", "lo", LO_LEN, 9, 0, h_lo_supplycost);

	c_custkey = new ColumnInfo("c_custkey", "c", C_LEN, 10, 2, h_c_custkey);
	c_nation = new ColumnInfo("c_nation", "c", C_LEN, 11, 2, h_c_nation);
	c_region = new ColumnInfo("c_region", "c", C_LEN, 12, 2, h_c_region);
	c_city = new ColumnInfo("c_city", "c", C_LEN, 13, 2, h_c_city);

	s_suppkey = new ColumnInfo("s_suppkey", "s", S_LEN, 14, 1, h_s_suppkey);	
	s_nation = new ColumnInfo("s_nation", "s", S_LEN, 15, 1, h_s_nation);
	s_region = new ColumnInfo("s_region", "s", S_LEN, 16, 1, h_s_region);
	s_city = new ColumnInfo("s_city", "s", S_LEN, 17, 1, h_s_city);

	p_partkey = new ColumnInfo("p_partkey", "p", P_LEN, 18, 3, h_p_partkey);
	p_brand1 = new ColumnInfo("p_brand1", "p", P_LEN, 19, 3, h_p_brand1);
	p_category = new ColumnInfo("p_category", "p", P_LEN, 20, 3, h_p_category);
	p_mfgr = new ColumnInfo("p_mfgr", "p", P_LEN, 21, 3, h_p_mfgr);

	d_datekey = new ColumnInfo("d_datekey", "d", D_LEN, 22, 4, h_d_datekey);
	d_year = new ColumnInfo("d_year", "d", D_LEN, 23, 4, h_d_year);
	d_yearmonthnum = new ColumnInfo("d_yearmonthnum", "d", D_LEN, 24, 4, h_d_yearmonthnum);

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

CacheManager::~CacheManager() {
	CubDebugExit(cudaFree((void**) &gpuCache));
	CubDebugExit(cudaFree((void**) &gpuProcessing));
	delete[] cpuProcessing;

	delete[] h_lo_orderkey;
	delete[] h_lo_orderdate;
	delete[] h_lo_custkey;
	delete[] h_lo_suppkey;
	delete[] h_lo_partkey;
	delete[] h_lo_revenue;
	delete[] h_lo_discount; 
	delete[] h_lo_quantity;
	delete[] h_lo_extendedprice;
	delete[] h_lo_supplycost;

	delete[] h_c_custkey;
	delete[] h_c_nation;
	delete[] h_c_region;
	delete[] h_c_city;

	delete[] h_s_suppkey;
	delete[] h_s_nation;
	delete[] h_s_region;
	delete[] h_s_city;

	delete[] h_p_partkey;
	delete[] h_p_brand1;
	delete[] h_p_category;
	delete[] h_p_mfgr;

	delete[] h_d_datekey;
	delete[] h_d_year;
	delete[] h_d_yearmonthnum;

	delete lo_orderkey;
	delete lo_orderdate;
	delete lo_custkey;
	delete lo_suppkey;
	delete lo_partkey;
	delete lo_revenue;
	delete lo_discount;
	delete lo_quantity;
	delete lo_extendedprice;
	delete lo_supplycost;

	delete c_custkey;
	delete c_nation;
	delete c_region;
	delete c_city;

	delete s_suppkey;	
	delete s_nation;
	delete s_region;
	delete s_city;

	delete p_partkey;
	delete p_brand1;
	delete p_category;
	delete p_mfgr;

	delete d_datekey;
	delete d_year;
	delete d_yearmonthnum;

	for (int i = 0; i < TOT_COLUMN; i++) {
		free(segment_list[i]);
	}
	free(segment_list);
}

#endif