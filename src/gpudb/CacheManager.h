#ifndef _CACHE_MANAGER_H_
#define _CACHE_MANAGER_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <utility>
#include <functional>
#include <algorithm>
#include <cmath>
#include <assert.h>

using namespace std;

#define SEGMENT_SIZE 1000
#define BASE_PATH "/CS764-GPUCompression/test/ssb/data/"
#define DATA_DIR BASE_PATH "s1_columnar/"

class Statistics;
class CacheManager;
class Segment;
class ColumnInfo;
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
}

class ColumnInfo{
public:
	ColumnInfo(string _column_name, string _table_name, int _LEN, int _column_id, int* _col_ptr)
		: column_name(_column_name), table_name(_table_name), LEN(_LEN), column_id(_column_id), col_ptr(_col_ptr) {
			stats = new Statistics();
			tot_seg_in_GPU = 0;
			preg_seg_in_GPU = 0;
			weight = 0;
		}
	Statistics* stats;
	string column_name;
	string table_name;
	int LEN;
	int column_id;
	int* col_ptr; //ptr to the beginning of the column
	int tot_seg_in_GPU; //total segments in GPU (based on current weight)
	float weight;

	Segment* getSegment(int index) {
		return seg = new Segment(this, col_ptr+SEGMENT_SIZE*index);
	}
};

class Segment{
public:
	Segment(ColumnInfo* _column, int* _seg_ptr, int _priority)
		: column(_column), seg_ptr(_seg_ptr), priority(_priority), seg_size(SEGMENT_SIZE) {
			col_ptr = column->col_ptr;
			segment_id = (seg_ptr - col_ptr)/seg_size;
		};
	Segment(ColumnInfo* _column, int* _seg_ptr)
		: column(_column), seg_ptr(_seg_ptr), priority(0), seg_size(SEGMENT_SIZE) {
			col_ptr = column->col_ptr;
			segment_id = (seg_ptr - col_ptr)/seg_size;
		};
	ColumnInfo* column;
	int segment_id;
	int* col_ptr; //ptr to the beginning of the column
	int* seg_ptr; //ptr to the beginning of the segment
	int priority;
	int seg_size;
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
}

class CacheManager{
public:
	int* gpuCache;
	int cache_total_seg;
	int TOT_COLUMN;

	vector<ColumnInfo*> allColumn;
	queue<int> empty_gpu_segment; //free list
	vector<priority_stack<Segment*>> cached_seg_in_GPU; //track segments cached in GPU
	int** segment_idx; //segment index in GPU for each column
	unordered_map<Segment*, int> cache_mapper; //map segment to index in GPU

	//int* metaCache; //metadata in GPU
	//int** meta_st_end; //no longer used, start and end index of metadata in GPU

	CacheManager(size_t cache_size, int _TOT_COLUMN) {
		cache_total_seg = cache_size/SEGMENT_SIZE;
		TOT_COLUMN = _TOT_COLUMN;

		CubDebugExit(cudaMalloc((void**) &gpuCache, cache_size * sizeof(int)));

		cached_seg_in_GPU.resize(TOT_COLUMN);
		allColumn.resize(TOT_COLUMN);
		segment_idx = new int [TOT_COLUMN][cache_total_seg];

		//CubDebugExit(cudaMalloc((void**) &metaCache, cache_total_seg * sizeof(int)));
		//meta_st_end = new int[TOT_COLUMN][2];
	}

	~CacheManager() {
		CubDebugExit(cudaFree((void**) &gpuCache));
		//CubDebugExit(cudaFree((void**) &metaCache));
	}


	int* loadColumnToCPU(string col_name, int num_entries) {
	  int* h_col = new int[num_entries];
	  string filename = DATA_DIR + lookup(col_name);
	  ifstream colData (filename.c_str(), ios::in | ios::binary);
	  if (!colData) {
	    return NULL;
	  }

	  colData.read((char*)h_col, num_entries * sizeof(int));
	  return h_col;
	}

	void cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
		int* seg_ptr = column->col_ptr + column->tot_seg_in_GPU*SEGMENT_SIZE;
		for (int i = 0; i < total_segment; i++) {
			Segment* seg = Segment(column->column_id, column->col_ptr, seg_ptr, 0);
			int idx = empty_gpu_segment.front();
			empty_gpu_segment.pop();
			assert(cache_mapper.find(seg) == cache_mapper.end());
			cache_mapper[seg] = idx;
			cached_seg_in_GPU[column->column_id].push(seg);
			CubDebugExit(cudaMemcpy(&gpuCache[idx * SEGMENT_SIZE], seg->seg_ptr, SEGMENT_SIZE * sizeof(int), cudaMemcpyHostToDevice));
			column->tot_seg_in_GPU++;
			seg_ptr += SEGMENT_SIZE;
		}
	}

	void cacheSegmentInGPU(Segment* seg) {
		int idx = empty_gpu_segment.front();
		empty_gpu_segment.pop();
		assert(cache_mapper.find(seg) == cache_mapper.end());
		cache_mapper[seg] = idx;
		cached_seg_in_GPU[seg->column->column_id].push(seg);
		CubDebugExit(cudaMemcpy(&gpuCache[idx * SEGMENT_SIZE], seg->seg_ptr, SEGMENT_SIZE * sizeof(int), cudaMemcpyHostToDevice));
		allColumn[seg->column->column_id]->tot_seg_in_GPU++;
	}

	void cacheListSegmentInGPU(vector<Segment*> v_seg) {
		for (int i = 0; i < v_seg.size(); i++) {
			int idx = empty_gpu_segment.front();
			empty_gpu_segment.pop();
			assert(cache_mapper.find(v_seg[i]) == cache_mapper.end());
			cache_mapper[v_seg[i]] = idx;
			cached_seg_in_GPU[v_seg[i]->column->column_id].push(v_seg[i]);
			CubDebugExit(cudaMemcpy(&gpuCache[idx * SEGMENT_SIZE], v_seg[i]->seg_ptr, SEGMENT_SIZE * sizeof(int), cudaMemcpyHostToDevice));
			allColumn[v_seg[i]->column->column_id]->tot_seg_in_GPU++;
		}
	}

	void deleteColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
		for (int i = 0; i < total_segment; i++) {
			Segment* seg = cached_seg_in_GPU[column->column_id].top();
			cached_seg_in_GPU[column->column_id].pop();
			assert(cache_mapper.find(seg) != cache_mapper.end());
			int idx = cache_mapper[seg];
			int ret = cache_mapper.erase(seg);
			assert(ret == 1);
			empty_gpu_segment.push(idx);
			column->tot_seg_in_GPU--;
		}
	}

	void constructListSegmentInGPU(ColumnInfo* column) {
		vector<Segment*> temp = cached_seg_in_GPU[column->column_id].return_stack();
		for (int i = 0; i < temp.size(); i++) {
			segment_idx[column->column_id][i] = temp[i];
		}
	}

	void updateSegmentPriority(Segment* seg, int priority) {
		if (priority > seg->priority) {
			seg->priority = priority;
			cached_seg_in_GPU[seg->column->column_id].percolateDown();
		} else if (priority < seg->priority) {
			seg->priority = priority;
			cached_seg_in_GPU[seg->column->column_id].percolateUp();
		}
	}

	void updateColumnWeight() {
		for (int i = 0; i < TOT_COLUMN; i++) {
			allColumn[i]->weight = allColumn[i]->stats->col_freq;
		}
	}

	void updateColumnInGPU() {
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

	void weightAdjustment() {
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