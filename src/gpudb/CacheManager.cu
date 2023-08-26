#include "CacheManager.h"

Segment::Segment(ColumnInfo* _column, int* _seg_ptr, int _priority)
: column(_column), seg_ptr(_seg_ptr), priority(_priority), seg_size(SEGMENT_SIZE) {
	stats = new Statistics();
	col_ptr = column->col_ptr;
	segment_id = (seg_ptr - col_ptr)/seg_size;
	weight = 0;
}

Segment::Segment(ColumnInfo* _column, int* _seg_ptr)
: column(_column), seg_ptr(_seg_ptr), priority(0), seg_size(SEGMENT_SIZE) {
	stats = new Statistics();
	col_ptr = column->col_ptr;
	segment_id = (seg_ptr - col_ptr)/seg_size;
	weight = 0;
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
	Segment* seg = new Segment(this, col_ptr+SEGMENT_SIZE*index);
	return seg;
}

CacheManager::CacheManager(size_t _cache_size, size_t _processing_size, size_t _pinned_memsize) {
	cache_size = _cache_size;
	cache_total_seg = _cache_size/SEGMENT_SIZE;
	processing_size = _processing_size;
	pinned_memsize = _pinned_memsize;
	TOT_COLUMN = 25;
	TOT_TABLE = 5;

	CubDebugExit(cudaMalloc((void**) &gpuCache, (cache_size) * sizeof(int)));
	CubDebugExit(cudaMemset(gpuCache, 0, (cache_size) * sizeof(int)));
	CubDebugExit(cudaMalloc((void**) &gpuProcessing, _processing_size * sizeof(uint64_t)));

	cpuProcessing = (uint64_t*) malloc(_processing_size * sizeof(uint64_t));
	CubDebugExit(cudaHostAlloc((void**) &pinnedMemory, _pinned_memsize * sizeof(uint64_t), cudaHostAllocDefault));
	gpuPointer = 0;
	cpuPointer = 0;
	pinnedPointer = 0;

	cached_seg_in_GPU.resize(TOT_COLUMN);
	allColumn.resize(TOT_COLUMN);

	index_to_segment.resize(TOT_COLUMN);

	for(int i = 0; i < cache_total_seg; i++) {
		empty_gpu_segment.push(i);
	}

	loadColumnToCPU();

	segment_bitmap = (char**) malloc (TOT_COLUMN * sizeof(char*));
	segment_list = (int**) malloc (TOT_COLUMN * sizeof(int*));
	segment_min = (int**) malloc (TOT_COLUMN * sizeof(int*));
	segment_max = (int**) malloc (TOT_COLUMN * sizeof(int*));

	for (int i = 0; i < TOT_COLUMN; i++) {
		int n = allColumn[i]->total_segment;
		segment_bitmap[i] = (char*) malloc(n * sizeof(char));
		CubDebugExit(cudaHostAlloc((void**) &(segment_list[i]), n * sizeof(int), cudaHostAllocDefault));

		segment_min[i] = (int*) malloc(n * sizeof(int));
		segment_max[i] = (int*) malloc(n * sizeof(int));

		memset(segment_bitmap[i], 0, n * sizeof(char));
		memset(segment_list[i], -1, n * sizeof(int));
	}

	readSegmentMinMax();

	for (int i = 0; i < TOT_COLUMN; i++) {
		index_to_segment[i].resize(allColumn[i]->total_segment);
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			index_to_segment[i][j] = allColumn[i]->getSegment(j);
		}
	}
	
}

void
CacheManager::resetCache(size_t _cache_size, size_t _processing_size, size_t _pinned_memsize) {

	CubDebugExit(cudaFree(gpuCache));
	CubDebugExit(cudaFree(gpuProcessing));
	delete[] cpuProcessing;
	CubDebugExit(cudaFreeHost(pinnedMemory));

	for (int i = 0; i < TOT_COLUMN; i++) {
		CubDebugExit(cudaFreeHost(segment_list[i]));
		free(segment_bitmap[i]);
	}
	free(segment_list);
	free(segment_bitmap);

	cache_size = _cache_size;
	cache_total_seg = _cache_size/SEGMENT_SIZE;
	processing_size = _processing_size;
	pinned_memsize = _pinned_memsize;

	CubDebugExit(cudaMalloc((void**) &gpuCache, (cache_size) * sizeof(int)));
	CubDebugExit(cudaMemset(gpuCache, 0, (cache_size) * sizeof(int)));
	CubDebugExit(cudaMalloc((void**) &gpuProcessing, _processing_size * sizeof(uint64_t)));

	cpuProcessing = (uint64_t*) malloc(_processing_size * sizeof(uint64_t));
	CubDebugExit(cudaHostAlloc((void**) &pinnedMemory, _pinned_memsize * sizeof(uint64_t), cudaHostAllocDefault));
	gpuPointer = 0;
	cpuPointer = 0;
	pinnedPointer = 0;

	while (!empty_gpu_segment.empty()) {
		empty_gpu_segment.pop();
	}

	for(int i = 0; i < cache_total_seg; i++) {
		empty_gpu_segment.push(i);
	}

	segment_bitmap = (char**) malloc (TOT_COLUMN * sizeof(char*));
	segment_list = (int**) malloc (TOT_COLUMN * sizeof(int*));
	for (int i = 0; i < TOT_COLUMN; i++) {
		int n = allColumn[i]->total_segment;
		segment_bitmap[i] = (char*) malloc(n * sizeof(char));
		CubDebugExit(cudaHostAlloc((void**) &(segment_list[i]), n * sizeof(int), cudaHostAllocDefault));
		memset(segment_bitmap[i], 0, n * sizeof(char));
		memset(segment_list[i], -1, n * sizeof(int));
	}
}

void 
CacheManager::readSegmentMinMax() {

	for (int i = 0; i < TOT_COLUMN; i++) {
		string line;
		ifstream myfile (DATA_DIR + allColumn[i]->column_name + "minmax");
		if (myfile.is_open()) {
			int segment_idx = 0;
			string del = " ";
			while ( getline (myfile,line) )
			{
				int start = 0;
				int end = line.find(del);
				if (end != -1) {
				    string minstring = line.substr(start, end - start);
				    segment_min[i][segment_idx] = stoi(minstring);
				    start = end + del.size();
				}
				string maxstring = line.substr(start, end - start);
				segment_max[i][segment_idx] = stoi(maxstring);
				segment_idx++;
			}
			assert(segment_idx == allColumn[i]->total_segment);
			myfile.close();
		} else {
			cout << "Unable to open file" << endl; 
			assert(0);
		}

	}
}

template <typename T>
T*
CacheManager::customMalloc(int size) {
	int alloc = ((size * sizeof(T)) + sizeof(uint64_t) - 1)/ sizeof(uint64_t);
	int start = __atomic_fetch_add(&cpuPointer, alloc, __ATOMIC_RELAXED);
	assert((start + alloc) < processing_size);
	return reinterpret_cast<T*>(cpuProcessing + start);
};

template <typename T>
T*
CacheManager::customCudaMalloc(int size) {
	int alloc = ((size * sizeof(T)) + sizeof(uint64_t) - 1)/ sizeof(uint64_t);
	int start = __atomic_fetch_add(&gpuPointer, alloc, __ATOMIC_RELAXED);
	assert((start + alloc) < processing_size);
	return reinterpret_cast<T*>(gpuProcessing + start);
};

template <typename T>
T*
CacheManager::customCudaHostAlloc(int size) {
	int alloc = ((size * sizeof(T)) + sizeof(uint64_t) - 1)/ sizeof(uint64_t);
	int start = __atomic_fetch_add(&pinnedPointer, alloc, __ATOMIC_RELAXED);
	assert((start + alloc) < processing_size);
	return reinterpret_cast<T*>(pinnedMemory + start);
};


void
CacheManager::indexTransfer(int** col_idx, ColumnInfo* column, cudaStream_t stream, bool custom) {
    if (col_idx[column->column_id] == NULL) {
      int* desired;
      if (custom) desired = (int*) customCudaMalloc<int>(column->total_segment); 
      else CubDebugExit(cudaMalloc((void**) &desired, column->total_segment * sizeof(int)));
      int* expected = NULL;
      CubDebugExit(cudaMemcpyAsync(desired, segment_list[column->column_id], column->total_segment * sizeof(int), cudaMemcpyHostToDevice, stream));
      CubDebugExit(cudaStreamSynchronize(stream));
      __atomic_compare_exchange_n(&(col_idx[column->column_id]), &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
};


void
CacheManager::resetPointer() {
	gpuPointer = 0;
	cpuPointer = 0;
	pinnedPointer = 0;
};

void
CacheManager::cacheColumnSegmentInGPU(ColumnInfo* column, int total_segment) {
	assert(column->tot_seg_in_GPU + total_segment <= column->total_segment);
	for (int i = 0; i < total_segment; i++) {
			int segment_idx = (column->seg_ptr - column->col_ptr)/SEGMENT_SIZE;
			if (segment_idx >= column->total_segment) {
				segment_idx = 0;
				column->seg_ptr = column->col_ptr;
			}
			assert(index_to_segment[column->column_id][segment_idx] != NULL);
			Segment* seg = index_to_segment[column->column_id][segment_idx];
			assert(segment_idx == seg->segment_id);
			assert(seg->priority == 0);
			column->seg_ptr += SEGMENT_SIZE;
			cacheSegmentInGPU(seg);
			cached_seg_in_GPU[column->column_id].push(seg);
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
	CubDebugExit(cudaMemcpy(&gpuCache[idx * SEGMENT_SIZE], seg->seg_ptr, SEGMENT_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	allColumn[seg->column->column_id]->tot_seg_in_GPU++;
	assert(allColumn[seg->column->column_id]->tot_seg_in_GPU <= allColumn[seg->column->column_id]->total_segment);
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
		deleteSegmentInGPU(seg);
			assert(seg->priority == 0);
			column->seg_ptr -= SEGMENT_SIZE;
			int segment_idx = (column->seg_ptr - SEGMENT_SIZE - column->col_ptr)/SEGMENT_SIZE;
			if (segment_idx < 0) {
				column->seg_ptr = column->col_ptr;
				segment_idx = 0;
			} else {
				assert(cache_mapper.find(index_to_segment[column->column_id][segment_idx]) != cache_mapper.end());
			}

	}
}

void 
CacheManager::deleteSegmentInGPU(Segment* seg) {
	assert(cache_mapper.find(seg) != cache_mapper.end());
	int idx = cache_mapper[seg];
	int ret = cache_mapper.erase(seg);
	assert(ret == 1);
	assert(segment_bitmap[seg->column->column_id][seg->segment_id] == 0x01);
	segment_bitmap[seg->column->column_id][seg->segment_id] = 0x00;
	assert(segment_list[seg->column->column_id][seg->segment_id] != -1);
	segment_list[seg->column->column_id][seg->segment_id] = -1;
	empty_gpu_segment.push(idx);
	seg->column->tot_seg_in_GPU--;
	assert(seg->column->tot_seg_in_GPU >= 0);
}

void
CacheManager::deleteListSegmentInGPU(vector<Segment*> v_seg) {
	for (int i = 0; i < v_seg.size(); i++) {
		deleteSegmentInGPU(v_seg[i]);
	}
}

void
CacheManager::updateColumnFrequency(ColumnInfo* column) {
	column->stats->col_freq+=(1.0 / column->total_segment);
}

void
CacheManager::updateColumnWeightDirect(ColumnInfo* column, double speedup) {
	if (column->table_id == 0) {
		column->stats->speedup += speedup/column->total_segment;
		column->weight += speedup/column->total_segment;		
	} else {
		column->stats->speedup += speedup*3/column->total_segment;
		column->weight += speedup*3/column->total_segment;			
	}

}

void
CacheManager::updateSegmentWeightDirect(ColumnInfo* column, Segment* segment, double speedup) {
	if (speedup > 0) {
		if (column->table_id == 0) {
			segment->stats->speedup += speedup/column->total_segment;
			segment->weight += speedup/column->total_segment;
		} else {
			segment->stats->speedup += speedup*3/column->total_segment;
			segment->weight += speedup*3/column->total_segment;
		}
	}
}

void
CacheManager::updateSegmentWeightCostDirect(ColumnInfo* column, Segment* segment, double speedup) {
	if (speedup > 0) {
		if (column->table_id == 0) {
			segment->stats->speedup += (speedup/column->total_segment);
			segment->weight += (speedup/column->total_segment);
		} else {
			segment->stats->speedup += (speedup/column->total_segment);
			segment->weight += (speedup/column->total_segment);
		}
	}
}

void
CacheManager::updateSegmentFreqDirect(ColumnInfo* column, Segment* segment) {
	segment->stats->col_freq += (1.0 / column->total_segment);
}

void
CacheManager::updateSegmentTimeDirect(ColumnInfo* column, Segment* segment, double timestamp) {
	segment->stats->backward_t = timestamp - (segment->stats->timestamp * column->total_segment);
	segment->stats->timestamp = (timestamp/ column->total_segment);
}

void
CacheManager::updateColumnTimestamp(ColumnInfo* column, double timestamp) {
	column->stats->backward_t = timestamp - (column->stats->timestamp * column->total_segment);
	column->stats->timestamp = (timestamp/ column->total_segment);
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
	double temp[TOT_COLUMN];
	bool check[TOT_COLUMN], stop;
	double remainder, sum_temp;
	double sum = 0;

	for (int i = 0; i < TOT_COLUMN; i++) {
		sum += allColumn[i]->weight;
		temp[i] = 0;
		check[i] = 0;
	}

	do {
		remainder = 0;
		for (int i = 0; i < TOT_COLUMN; i++) {
			if (allColumn[i]->weight > 0) {
				temp[i] = (allColumn[i]->total_segment * sum /cache_total_seg); //ngitung max weight kalo sesuai ukuran kolomnya
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
					allColumn[i]->weight += (allColumn[i]->weight /sum_temp * remainder); //tiap kolom dapet porsi dari remainder seuai sama weightnya, makin tinggi weight, makin banyak dapet remainder
				}
			}
		}
	} while (remainder != 0 && !stop);
}

float
CacheManager::runReplacement(ReplacementPolicy strategy, unsigned long long* traffic) {

  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  float time;
  cudaEventRecord(start, 0);

  unsigned long long traf = 0;

  if (traffic != NULL) traf = (*traffic);

	if (strategy == LFU) { //LEAST FREQUENTLY USED
		traf += LFUReplacement();
	} else if (strategy == LRU) { //LEAST RECENTLY USED
		traf += LRUReplacement();
	} else if (strategy == LRU2) { //LEAST RECENTLY USED
		traf += LRU_2Replacement();
	} else if (strategy == LRU2Segmented) { //LEAST RECENTLY USED
		traf += LRU_2SegmentedReplacement();
	} else if (strategy == LFUSegmented) { //LEAST RECENTLY USED
		traf += LFUSegmentedReplacement();
	} else if (strategy == LRUSegmented) { //LEAST RECENTLY USED
		traf += LRUSegmentedReplacement();
	} else if (strategy == Segmented) {
		traf += SegmentReplacement();
	}

  if (traffic != NULL) (*traffic) = traf;

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  return time;
};

unsigned long long
CacheManager::SegmentReplacement() {
	multimap<double, Segment*> access_weight_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[i][j];
			access_weight_map.insert({segment->weight, segment});
			cout << allColumn[i]->column_name << " " << j << " " << segment->weight << endl;
		}
	}

	int temp_buffer_size = 0; // in segment
	set<Segment*> segments_to_place;
	multimap<double, Segment*>::reverse_iterator cit;

    for(cit = access_weight_map.rbegin();cit != access_weight_map.rend(); ++cit){
        if(temp_buffer_size + 1 < cache_total_seg && cit->first > 0){
            temp_buffer_size+=1;
            segments_to_place.insert(cit->second);
            cout << "Should place ";
            cout << cit->second->column->column_name << " segment " << cit->second->segment_id << endl;
        }
    }

    cout << "Cached segment: " << temp_buffer_size << " Cache total: " << cache_total_seg << endl;
    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
    	for (int j = 0; j < allColumn[i]->total_segment; j++) {
    		Segment* segment = index_to_segment[i][j];
				if (segments_to_place.find(segment) == segments_to_place.end()) {
					if (segment_bitmap[i][j]) {
						cout << "Deleting segment ";
						cout << segment->column->column_name << " segment " << segment->segment_id << endl;
						deleteSegmentInGPU(segment);
					}
				}
			}
    }

    set<Segment*>::const_iterator cit2;
    for(cit2 = segments_to_place.cbegin();cit2 != segments_to_place.cend(); ++cit2){
    	if (segment_bitmap[(*cit2)->column->column_id][(*cit2)->segment_id] == 0) {
				cout << "Caching segment ";
				cout << (*cit2)->column->column_name << " " << (*cit2)->segment_id << endl;
				cacheSegmentInGPU(*cit2);
				traffic += SEGMENT_SIZE * sizeof(int);
    	}
    }
    cout << "Successfully cached" << endl;

    return traffic;
}

unsigned long long
CacheManager::LRUReplacement() {
	multimap<double, ColumnInfo*> access_timestamp_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		access_timestamp_map.insert({allColumn[i]->stats->timestamp, allColumn[i]});
	}

	int temp_buffer_size = 0; // in segment
	set<ColumnInfo*> columns_to_place;
	multimap<double, ColumnInfo*>::reverse_iterator cit;

    for(cit=access_timestamp_map.rbegin(); cit!=access_timestamp_map.rend(); ++cit){
        if(temp_buffer_size + cit->second->total_segment < cache_total_seg && cit->first>0){
            temp_buffer_size+=cit->second->total_segment;
            columns_to_place.insert(cit->second);
        }
    }

    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->tot_seg_in_GPU > 0 && columns_to_place.find(allColumn[i]) == columns_to_place.end()) {
			deleteColumnSegmentInGPU(allColumn[i], allColumn[i]->tot_seg_in_GPU);
		}
    }

    set<ColumnInfo*>::const_iterator cit2;
    for(cit2 = columns_to_place.cbegin();cit2 != columns_to_place.cend(); ++cit2){
    	if ((*cit2)->tot_seg_in_GPU == 0) {
    		cacheColumnSegmentInGPU(*cit2, (*cit2)->total_segment);
    		traffic += (*cit2)->total_segment * SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::LRUSegmentedReplacement() {
	multimap<double, Segment*> access_timestamp_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[i][j];
			access_timestamp_map.insert({segment->stats->timestamp, segment});
		}
	}

	int temp_buffer_size = 0; // in segment
	set<Segment*> segments_to_place;
	multimap<double, Segment*>::reverse_iterator cit;

    for(cit = access_timestamp_map.rbegin();cit != access_timestamp_map.rend(); ++cit){
        if(temp_buffer_size + 1 < cache_total_seg && cit->first > 0){
            temp_buffer_size+=1;
            segments_to_place.insert(cit->second);
        }
    }

    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
    	for (int j = 0; j < allColumn[i]->total_segment; j++) {
    		Segment* segment = index_to_segment[i][j];
				if (segments_to_place.find(segment) == segments_to_place.end()) {
					if (segment_bitmap[i][j]) {
						deleteSegmentInGPU(segment);
					}
				}
			}
    }

    set<Segment*>::const_iterator cit2;
    for(cit2 = segments_to_place.cbegin();cit2 != segments_to_place.cend(); ++cit2){
    	if (segment_bitmap[(*cit2)->column->column_id][(*cit2)->segment_id] == 0) {
				cacheSegmentInGPU(*cit2);
				traffic += SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::LRU_2Replacement() {
	multimap<double, ColumnInfo*> access_backward_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		access_backward_map.insert({allColumn[i]->stats->backward_t, allColumn[i]});
	}

	int temp_buffer_size = 0; // in segment
	set<ColumnInfo*> columns_to_place;
	multimap<double, ColumnInfo*>::iterator cit;

    for(cit=access_backward_map.begin(); cit!=access_backward_map.end(); ++cit){
        if(temp_buffer_size + cit->second->total_segment < cache_total_seg && cit->first>0){
            temp_buffer_size+=cit->second->total_segment;
            columns_to_place.insert(cit->second);
        }
    }

    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->tot_seg_in_GPU > 0 && columns_to_place.find(allColumn[i]) == columns_to_place.end()) {
			deleteColumnSegmentInGPU(allColumn[i], allColumn[i]->tot_seg_in_GPU);
		}
    }

    set<ColumnInfo*>::const_iterator cit2;
    for(cit2 = columns_to_place.cbegin();cit2 != columns_to_place.cend(); ++cit2){
    	if ((*cit2)->tot_seg_in_GPU == 0) {
    		cacheColumnSegmentInGPU(*cit2, (*cit2)->total_segment);
    		traffic += (*cit2)->total_segment * SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::LRU_2SegmentedReplacement() {
	multimap<double, Segment*> access_backward_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[i][j];
			access_backward_map.insert({segment->stats->backward_t, segment});
		}
	}

	int temp_buffer_size = 0; // in segment
	set<Segment*> segments_to_place;
	multimap<double, Segment*>::iterator cit;

    for(cit = access_backward_map.begin();cit != access_backward_map.end(); ++cit){
        if(temp_buffer_size + 1 < cache_total_seg && cit->first > 0){
            temp_buffer_size+=1;
            segments_to_place.insert(cit->second);
        }
    }

    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
    	for (int j = 0; j < allColumn[i]->total_segment; j++) {
    		Segment* segment = index_to_segment[i][j];
				if (segments_to_place.find(segment) == segments_to_place.end()) {
					if (segment_bitmap[i][j]) {
						deleteSegmentInGPU(segment);
					}
				}
			}
    }

    set<Segment*>::const_iterator cit2;
    for(cit2 = segments_to_place.cbegin();cit2 != segments_to_place.cend(); ++cit2){
    	if (segment_bitmap[(*cit2)->column->column_id][(*cit2)->segment_id] == 0) {
				cacheSegmentInGPU(*cit2);
				traffic += SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::LFUSegmentedReplacement() {
	multimap<double, Segment*> access_frequency_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[i][j];
			access_frequency_map.insert({segment->stats->col_freq, segment});
		}
	}

	int temp_buffer_size = 0; // in segment
	set<Segment*> segments_to_place;
	multimap<double, Segment*>::reverse_iterator cit;

    for(cit = access_frequency_map.rbegin();cit != access_frequency_map.rend(); ++cit){
        if(temp_buffer_size + 1 < cache_total_seg && cit->first > 0){
            temp_buffer_size+=1;
            segments_to_place.insert(cit->second);
        }
    }

    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
    	for (int j = 0; j < allColumn[i]->total_segment; j++) {
    		Segment* segment = index_to_segment[i][j];
				if (segments_to_place.find(segment) == segments_to_place.end()) {
					if (segment_bitmap[i][j]) {
						deleteSegmentInGPU(segment);
					}
				}
			}
    }

    set<Segment*>::const_iterator cit2;
    for(cit2 = segments_to_place.cbegin();cit2 != segments_to_place.cend(); ++cit2){
    	if (segment_bitmap[(*cit2)->column->column_id][(*cit2)->segment_id] == 0) {
				cacheSegmentInGPU(*cit2);
				traffic += SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::LFUReplacement() {
	multimap<double, ColumnInfo*> access_frequency_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		access_frequency_map.insert({allColumn[i]->stats->col_freq, allColumn[i]});
	}

	int temp_buffer_size = 0; // in segment
	set<ColumnInfo*> columns_to_place;
	multimap<double, ColumnInfo*>::reverse_iterator cit;
	set<double> access;
	set<double>::reverse_iterator acc_it;


  for(cit=access_frequency_map.rbegin(); cit!=access_frequency_map.rend(); ++cit){
  	// cout << cit->second->column_name << endl;
  	access.insert(cit->first);
  }

  for(acc_it = access.rbegin();acc_it != access.rend(); ++acc_it){
  	size_t count = access_frequency_map.count(*acc_it);

  	multimap<double, ColumnInfo*> access_timestamp_map;
		multimap<double, ColumnInfo*>::iterator it;
 		for (it=access_frequency_map.equal_range(*acc_it).first; it!=access_frequency_map.equal_range(*acc_it).second; ++it) {
 			if ((*acc_it) > 0) access_timestamp_map.insert({it->second->stats->timestamp, it->second});
 		}	

 		multimap<double, ColumnInfo*>::iterator cit_time;

 		//IF FREQ THE SAME, USE MRU
  	for(cit_time=access_timestamp_map.begin(); cit_time!=access_timestamp_map.end(); ++cit_time){
      if(temp_buffer_size + cit_time->second->total_segment < cache_total_seg){
          temp_buffer_size+=cit_time->second->total_segment;
          columns_to_place.insert(cit_time->second);
      }
 		}
  }

  assert(temp_buffer_size <= cache_total_seg);

  for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->tot_seg_in_GPU > 0 && columns_to_place.find(allColumn[i]) == columns_to_place.end()) {
			deleteColumnSegmentInGPU(allColumn[i], allColumn[i]->tot_seg_in_GPU);
		}
  }

  set<ColumnInfo*>::const_iterator cit2;
  for(cit2 = columns_to_place.cbegin();cit2 != columns_to_place.cend(); ++cit2){
  	if ((*cit2)->tot_seg_in_GPU == 0) {
  		cacheColumnSegmentInGPU(*cit2, (*cit2)->total_segment);
  		traffic += (*cit2)->total_segment * SEGMENT_SIZE * sizeof(int);
  	}
  }

  return traffic;
}

unsigned long long
CacheManager::LRU2Replacement() {
	multimap<double, ColumnInfo*> access_timestamp_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		access_timestamp_map.insert({allColumn[i]->stats->timestamp, allColumn[i]});
	}

	int temp_buffer_size = 0; // in segment
	set<ColumnInfo*> columns_to_place;
	multimap<double, ColumnInfo*>::reverse_iterator cit;

	set<double> access;
	set<double>::reverse_iterator acc_it;


    for(cit=access_timestamp_map.rbegin(); cit!=access_timestamp_map.rend(); ++cit){
    	// cout << cit->second->column_name << endl;
    	access.insert(cit->first);
    }

    for(acc_it = access.rbegin();acc_it != access.rend(); ++acc_it){
    	size_t count = access_timestamp_map.count(*acc_it);
    	multimap<double, ColumnInfo*> weight_map;
			multimap<double, ColumnInfo*>::iterator it;
	 		for (it=access_timestamp_map.equal_range(*acc_it).first; it!=access_timestamp_map.equal_range(*acc_it).second; ++it) {
	 			if ((*acc_it) > 0) weight_map.insert({it->second->weight, it->second});
	 		}

		 	multimap<double, ColumnInfo*>::reverse_iterator cit_time;

		 	for(cit_time=weight_map.rbegin(); cit_time!=weight_map.rend(); ++cit_time){
		        if(temp_buffer_size + cit_time->second->total_segment < cache_total_seg){
		            temp_buffer_size+=cit_time->second->total_segment;
		            columns_to_place.insert(cit_time->second);
		            cout << "Should place ";
		            cout << cit_time->second->column_name << endl;
		        }
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

    set<ColumnInfo*>::const_iterator cit2;
    for(cit2 = columns_to_place.cbegin();cit2 != columns_to_place.cend(); ++cit2){
    	if ((*cit2)->tot_seg_in_GPU == 0) {
				cout << "Caching column ";
				cout << (*cit2)->column_name << endl;
    		cacheColumnSegmentInGPU(*cit2, (*cit2)->total_segment);
    		cout << "Successfully cached" << endl;
    		traffic += (*cit2)->total_segment * SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::LFU2Replacement() {
	multimap<double, ColumnInfo*> access_frequency_map;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		access_frequency_map.insert({allColumn[i]->stats->col_freq, allColumn[i]});
	}

	int temp_buffer_size = 0; // in segment
	set<ColumnInfo*> columns_to_place;
	multimap<double, ColumnInfo*>::reverse_iterator cit;

	set<double> access;
	set<double>::reverse_iterator acc_it;

    for(cit=access_frequency_map.rbegin(); cit!=access_frequency_map.rend(); ++cit){
    	// cout << cit->second->column_name << endl;
    	access.insert(cit->first);
    }

    for(acc_it = access.rbegin();acc_it != access.rend(); ++acc_it){
    	size_t count = access_frequency_map.count(*acc_it);

	    multimap<double, ColumnInfo*> weight_map;
			multimap<double, ColumnInfo*>::iterator it;
		 	for (it=access_frequency_map.equal_range(*acc_it).first; it!=access_frequency_map.equal_range(*acc_it).second; ++it) {
		 		if ((*acc_it) > 0) weight_map.insert({it->second->weight, it->second});
		 	}

		 	multimap<double, ColumnInfo*>::reverse_iterator cit_time;

		 	for(cit_time=weight_map.rbegin(); cit_time!=weight_map.rend(); ++cit_time){
        if(temp_buffer_size + cit_time->second->total_segment < cache_total_seg){
            temp_buffer_size+=cit_time->second->total_segment;
            columns_to_place.insert(cit_time->second);
            cout << "Should place ";
            cout << cit_time->second->column_name << endl;
        }
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

    set<ColumnInfo*>::const_iterator cit2;
    for(cit2 = columns_to_place.cbegin();cit2 != columns_to_place.cend(); ++cit2){
    	if ((*cit2)->tot_seg_in_GPU == 0) {
				cout << "Caching column ";
				cout << (*cit2)->column_name << endl;
    		cacheColumnSegmentInGPU(*cit2, (*cit2)->total_segment);
    		cout << "Successfully cached" << endl;
    		traffic += (*cit2)->total_segment * SEGMENT_SIZE * sizeof(int);
    	}
    }

    return traffic;
}

unsigned long long
CacheManager::NewReplacement() {
	std::map<ColumnInfo*, int> column_portion;
	int moved_segment = 0;
	int erased_segment = 0;
	int filled_cache = 0;
	unsigned long long traffic = 0;

	double sum = 0;
	for (int i = 0; i < TOT_COLUMN; i++) {
		cout << "Column: " << allColumn[i]->column_name << " weight: " << allColumn[i]->weight << endl;
		sum += allColumn[i]->weight;
	}

	weightAdjustment();


	cout << endl;

	sum = 0;
	for (int i = 0; i < TOT_COLUMN; i++) {
		cout << "Column: " << allColumn[i]->column_name << " weight: " << allColumn[i]->weight << endl;
		sum += allColumn[i]->weight;
	}

	//cout << sum << endl;

	cout << endl;

	for (int i = 0; i < TOT_COLUMN; i++) {
		double temp = (allColumn[i]->weight / sum) * cache_total_seg;
		column_portion[allColumn[i]] = (int) (temp + 0.00001);
		if (column_portion[allColumn[i]] > allColumn[i]->total_segment) {
			column_portion[allColumn[i]] = allColumn[i]->total_segment;
		}
		filled_cache += column_portion[allColumn[i]];
	}

	for (int i = 0; i < TOT_COLUMN; i++) {
		if (column_portion[allColumn[i]] < allColumn[i]->tot_seg_in_GPU) {
			assert(column_portion[allColumn[i]] <= allColumn[i]->total_segment);
			erased_segment = (allColumn[i]->tot_seg_in_GPU - column_portion[allColumn[i]]);
			cout << "Deleting " << erased_segment << " segments for column " << allColumn[i]->column_name << endl;
			deleteColumnSegmentInGPU(allColumn[i], erased_segment);
		}
	}

	for (int i = 0; i < TOT_COLUMN; i++) {
		if (column_portion[allColumn[i]] > allColumn[i]->tot_seg_in_GPU) {
			assert(column_portion[allColumn[i]] <= allColumn[i]->total_segment);
			moved_segment = (column_portion[allColumn[i]] - allColumn[i]->tot_seg_in_GPU);
			cout << "Caching " << moved_segment << " segments for column " << allColumn[i]->column_name << endl;
			cacheColumnSegmentInGPU(allColumn[i], moved_segment);
			traffic += moved_segment * SEGMENT_SIZE * sizeof(int);
		}
	}

    cout << "Cached segment: " << filled_cache << " Cache total: " << cache_total_seg << endl;
    assert(filled_cache <= cache_total_seg);

   	for (int i = 0; i < TOT_COLUMN; i++) {
			allColumn[i]->weight = allColumn[i]->stats->speedup;
   	}

   	return traffic;

};

unsigned long long
CacheManager::New2Replacement() {
	std::multimap<double, ColumnInfo*> weight_map;
	std::map<ColumnInfo*, int> should_cached;
	int moved_segment = 0;
	int erased_segment = 0;
	unsigned long long traffic = 0;

	for (int i = TOT_COLUMN-1; i >= 0; i--) {
		weight_map.insert({allColumn[i]->weight, allColumn[i]});
		cout << "Column: " << allColumn[i]->column_name << " weight: " << allColumn[i]->weight << endl;
	}

	int temp_buffer_size = 0; // in segment
	std::multimap<double, ColumnInfo*>::reverse_iterator cit;

    for(cit=weight_map.rbegin(); cit!=weight_map.rend(); ++cit){
    	should_cached[cit->second] = 0;
    	if (temp_buffer_size < cache_total_seg) {
	        if(temp_buffer_size + cit->second->total_segment <= cache_total_seg && cit->first>0){
	            should_cached[cit->second] = cit->second->total_segment;
	            temp_buffer_size+=cit->second->total_segment;
	        } else if (temp_buffer_size + cit->second->total_segment > cache_total_seg && cit->first > 0) {
	        	should_cached[cit->second] = cache_total_seg - temp_buffer_size;
	        	temp_buffer_size=cache_total_seg;
	        }
    	}
    }

    cout << "Cached segment: " << temp_buffer_size << " Cache total: " << cache_total_seg << endl;
    assert(temp_buffer_size <= cache_total_seg);

    for (int i = 0; i < TOT_COLUMN; i++) {
			if (should_cached[allColumn[i]] < allColumn[i]->tot_seg_in_GPU) {
				erased_segment = (allColumn[i]->tot_seg_in_GPU - should_cached[allColumn[i]]);
				cout << "Deleting " << erased_segment << " segments for column " << allColumn[i]->column_name << endl;
				deleteColumnSegmentInGPU(allColumn[i], erased_segment);
			}
    }

	for (int i = 0; i < TOT_COLUMN; i++) {
		if (should_cached[allColumn[i]] > allColumn[i]->tot_seg_in_GPU) {
			moved_segment = (should_cached[allColumn[i]] - allColumn[i]->tot_seg_in_GPU);
			cout << "Caching " << moved_segment << " segments for column " << allColumn[i]->column_name << endl;
			cacheColumnSegmentInGPU(allColumn[i], moved_segment);
			traffic += moved_segment * SEGMENT_SIZE * sizeof(int);
		}
	}

	return traffic;

};

void
CacheManager::newEpoch(double param) {

	for (int i = 0; i < TOT_COLUMN; i++) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[allColumn[i]->column_id][j];
			segment->weight = param * segment->weight;
		}
	}

	for (int i = 0; i < TOT_COLUMN; i++) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[allColumn[i]->column_id][j];
			segment->stats->col_freq = param * segment->stats->col_freq;
		}
	}

	for (int i = 0; i < TOT_COLUMN; i++) {
		for (int j = 0; j < allColumn[i]->total_segment; j++) {
			Segment* segment = index_to_segment[allColumn[i]->column_id][j];
			segment->stats->backward_t = param * segment->stats->backward_t;
		}
	}

};

int
CacheManager::cacheSpecificColumn(string column_name) {
	ColumnInfo* column;
	bool found = false;
	for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->column_name.compare(column_name) == 0) {
			column = allColumn[i];
			found = true;
			break;
		}
	}

	if (!found) return -1;

	assert(column->tot_seg_in_GPU == 0);
	cacheColumnSegmentInGPU(column, column->total_segment);
	return 0;
}

void
CacheManager::deleteColumnsFromGPU() {
	for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->tot_seg_in_GPU == allColumn[i]->total_segment) {
			deleteColumnSegmentInGPU(allColumn[i], allColumn[i]->tot_seg_in_GPU);
		}
	}
}

int
CacheManager::deleteSpecificColumnFromGPU(string column_name) {
	ColumnInfo* column;
	bool found = false;
	for (int i = 0; i < TOT_COLUMN; i++) {
		if (allColumn[i]->column_name.compare(column_name) == 0) {
			column = allColumn[i];
			found = true;
			break;
		}
	}

	if (!found) return -1;

	assert(column->tot_seg_in_GPU == column->total_segment);
	deleteColumnSegmentInGPU(column, column->total_segment);
	return 0;
}

void
CacheManager::deleteAll() {
	for (int i = 0; i < TOT_COLUMN; i++) {
		ColumnInfo* column = allColumn[i];
		for (int j = 0; j < column->total_segment; j++) {
			if (segment_bitmap[column->column_id][j] == 1) {
				Segment* seg = index_to_segment[column->column_id][j];
				assert(cache_mapper.find(seg) != cache_mapper.end());
				deleteSegmentInGPU(seg);
			}
		}
	}
}

void
CacheManager::loadColumnToCPU() {

	h_lo_orderkey = loadColumnPinnedSort<int>("lo_orderkey", LO_LEN);
	h_lo_suppkey = loadColumnPinnedSort<int>("lo_suppkey", LO_LEN);
	h_lo_custkey = loadColumnPinnedSort<int>("lo_custkey", LO_LEN);
	h_lo_partkey = loadColumnPinnedSort<int>("lo_partkey", LO_LEN);
	h_lo_orderdate = loadColumnPinnedSort<int>("lo_orderdate", LO_LEN);
	h_lo_revenue = loadColumnPinnedSort<int>("lo_revenue", LO_LEN);
	h_lo_discount = loadColumnPinnedSort<int>("lo_discount", LO_LEN);
	h_lo_quantity = loadColumnPinnedSort<int>("lo_quantity", LO_LEN);
	h_lo_extendedprice = loadColumnPinnedSort<int>("lo_extendedprice", LO_LEN);
	h_lo_supplycost = loadColumnPinnedSort<int>("lo_supplycost", LO_LEN);

	h_c_custkey = loadColumnPinned<int>("c_custkey", C_LEN);
	h_c_nation = loadColumnPinned<int>("c_nation", C_LEN);
	h_c_region = loadColumnPinned<int>("c_region", C_LEN);
	h_c_city = loadColumnPinned<int>("c_city", C_LEN);

	h_s_suppkey = loadColumnPinned<int>("s_suppkey", S_LEN);
	h_s_nation = loadColumnPinned<int>("s_nation", S_LEN);
	h_s_region = loadColumnPinned<int>("s_region", S_LEN);
	h_s_city = loadColumnPinned<int>("s_city", S_LEN);

	h_p_partkey = loadColumnPinned<int>("p_partkey", P_LEN);
	h_p_brand1 = loadColumnPinned<int>("p_brand1", P_LEN);
	h_p_category = loadColumnPinned<int>("p_category", P_LEN);
	h_p_mfgr = loadColumnPinned<int>("p_mfgr", P_LEN);

	h_d_datekey = loadColumnPinned<int>("d_datekey", D_LEN);
	h_d_year = loadColumnPinned<int>("d_year", D_LEN);
	h_d_yearmonthnum = loadColumnPinned<int>("d_yearmonthnum", D_LEN);

	lo_orderkey = new ColumnInfo("lo_orderkey", "lo", LO_LEN, 0, 0, h_lo_orderkey);
	lo_suppkey = new ColumnInfo("lo_suppkey", "lo", LO_LEN, 1, 0, h_lo_suppkey);
	lo_custkey = new ColumnInfo("lo_custkey", "lo", LO_LEN, 2, 0, h_lo_custkey);
	lo_partkey = new ColumnInfo("lo_partkey", "lo", LO_LEN, 3, 0, h_lo_partkey);
	lo_orderdate = new ColumnInfo("lo_orderdate", "lo", LO_LEN, 4, 0, h_lo_orderdate);
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
	allColumn[1] = lo_suppkey;
	allColumn[2] = lo_custkey;
	allColumn[3] = lo_partkey;
	allColumn[4] = lo_orderdate;
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

	columns_in_table.resize(TOT_TABLE);
	for (int i = 0; i < TOT_COLUMN; i++) {
		columns_in_table[allColumn[i]->table_id].push_back(allColumn[i]->column_id);
	}
}

CacheManager::~CacheManager() {
	CubDebugExit(cudaFree(gpuCache));
	CubDebugExit(cudaFree(gpuProcessing));
	delete[] cpuProcessing;
	CubDebugExit(cudaFreeHost(pinnedMemory));

	CubDebugExit(cudaFreeHost(h_lo_orderkey));
	CubDebugExit(cudaFreeHost(h_lo_suppkey));
	CubDebugExit(cudaFreeHost(h_lo_custkey));
	CubDebugExit(cudaFreeHost(h_lo_partkey));
	CubDebugExit(cudaFreeHost(h_lo_orderdate));
	CubDebugExit(cudaFreeHost(h_lo_revenue));
	CubDebugExit(cudaFreeHost(h_lo_discount)); 
	CubDebugExit(cudaFreeHost(h_lo_quantity));
	CubDebugExit(cudaFreeHost(h_lo_extendedprice));
	CubDebugExit(cudaFreeHost(h_lo_supplycost));

	CubDebugExit(cudaFreeHost(h_c_custkey));
	CubDebugExit(cudaFreeHost(h_c_nation));
	CubDebugExit(cudaFreeHost(h_c_region));
	CubDebugExit(cudaFreeHost(h_c_city));

	CubDebugExit(cudaFreeHost(h_s_suppkey));
	CubDebugExit(cudaFreeHost(h_s_nation));
	CubDebugExit(cudaFreeHost(h_s_region));
	CubDebugExit(cudaFreeHost(h_s_city));

	CubDebugExit(cudaFreeHost(h_p_partkey));
	CubDebugExit(cudaFreeHost(h_p_brand1));
	CubDebugExit(cudaFreeHost(h_p_category));
	CubDebugExit(cudaFreeHost(h_p_mfgr));

	CubDebugExit(cudaFreeHost(h_d_datekey));
	CubDebugExit(cudaFreeHost(h_d_year));
	CubDebugExit(cudaFreeHost(h_d_yearmonthnum));

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
		CubDebugExit(cudaFreeHost(segment_list[i]));
		//free(segment_list[i]);
		free(segment_bitmap[i]);
	}
	free(segment_list);
	free(segment_bitmap);
}


template int*
CacheManager::customMalloc<int>(int size);

template int*
CacheManager::customCudaMalloc<int>(int size);

template int*
CacheManager::customCudaHostAlloc<int>(int size);

template short*
CacheManager::customMalloc<short>(int size);

template short*
CacheManager::customCudaMalloc<short>(int size);

template short*
CacheManager::customCudaHostAlloc<short>(int size);