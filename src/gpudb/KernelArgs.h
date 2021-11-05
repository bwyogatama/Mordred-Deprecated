#ifndef _KERNEL_ARGS_H_
#define _KERNEL_ARGS_H_

#include "common.h"

// #define BLOCK_T 128
// #define ITEMS_PER_T 4

class ColumnInfo;

template<typename T>
using group_func_t = T (*) (T, T);

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
using filter_func_t_dev = void (*) (T(&) [ITEMS_PER_THREADS], int(&) [ITEMS_PER_THREADS], T, T, int);

template<typename T>
using filter_func_t_host = bool (*) (T, T, T);

template <typename T> 
__device__ T sub_func (T x, T y)
{
    return x - y;
}

template <typename T> 
__device__ T mul_func (T x, T y)
{
    return x * y;
}

template <typename T> 
T host_sub_func (T x, T y)
{
    return x - y;
}

template <typename T> 
T host_mul_func (T x, T y)
{
    return x * y;
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
__device__ void pred_eq (
	T  (&items)[ITEMS_PER_THREADS],
	int (&selection_flags)[ITEMS_PER_THREADS], 
	T compare1, T compare2, int num_tile_items)
{
    BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREADS>(items, compare1, selection_flags, num_tile_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
__device__ void pred_eq_or_eq (
	T  (&items)[ITEMS_PER_THREADS],
	int (&selection_flags)[ITEMS_PER_THREADS], 
	T compare1, T compare2, int num_tile_items)
{
	BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREADS>(items, compare1, selection_flags, num_tile_items);
	BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREADS>(items, compare2, selection_flags, num_tile_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
__device__ void pred_between (
	T  (&items)[ITEMS_PER_THREADS],
	int (&selection_flags)[ITEMS_PER_THREADS], 
	T compare1, T compare2, int num_tile_items)
{
    BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREADS>(items, compare1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREADS>(items, compare2, selection_flags, num_tile_items);
}

template<typename T>
bool host_pred_eq (T x, T compare1, T compare2) {
	return (x == compare1);
}

template<typename T>
bool host_pred_eq_or_eq (T x, T compare1, T compare2) {
	return ((x == compare1) || (x == compare2));
}

template<typename T>
bool host_pred_between (T x, T compare1, T compare2) {
	return ((x >= compare1) && (x <= compare2));
}

template <typename T> 
__device__ group_func_t<T> p_sub_func = sub_func<T>;

template <typename T> 
__device__ group_func_t<T> p_mul_func = mul_func<T>;

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
__device__ filter_func_t_dev<T, BLOCK_THREADS, ITEMS_PER_THREADS> p_pred_eq = pred_eq<T, BLOCK_THREADS, ITEMS_PER_THREADS>;

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
__device__ filter_func_t_dev<T, BLOCK_THREADS, ITEMS_PER_THREADS> p_pred_eq_or_eq = pred_eq_or_eq<T, BLOCK_THREADS, ITEMS_PER_THREADS>;

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREADS>
__device__ filter_func_t_dev<T, BLOCK_THREADS, ITEMS_PER_THREADS> p_pred_between = pred_between<T, BLOCK_THREADS, ITEMS_PER_THREADS>;

class QueryParams{
public:

  int query;
  
  map<ColumnInfo*, int> min_key;
  map<ColumnInfo*, int> max_key;
  map<ColumnInfo*, int> min_val;
  map<ColumnInfo*, int> unique_val;
  map<ColumnInfo*, int> dim_len;

  map<ColumnInfo*, int*> ht_CPU;
  map<ColumnInfo*, int*> ht_GPU;

  map<ColumnInfo*, int> compare1;
  map<ColumnInfo*, int> compare2;
  map<ColumnInfo*, int> mode;

  map<ColumnInfo*, float> selectivity;
  map<ColumnInfo*, float> sel;

  int total_val, mode_group;

  // int *ht_p, *ht_c, *ht_s, *ht_d;
  // int *d_ht_p, *d_ht_c, *d_ht_s, *d_ht_d;

  int* res;
  int* d_res;

  group_func_t<int> d_group_func;
  group_func_t<int> h_group_func;

  map<ColumnInfo*, filter_func_t_dev<int, 128, 4>> map_filter_func_dev;
  map<ColumnInfo*, filter_func_t_host<int>> map_filter_func_host;

  QueryParams(int _query): query(_query) {
    assert(_query == 11 || _query == 12 || _query == 13 ||
          _query == 21 || _query == 22 || _query == 23 ||
          _query == 31 || _query == 32 || _query == 33 || _query == 34 ||
          _query == 41 || _query == 42 || _query == 43); 
  };
};

typedef struct probeArgsGPU {
	int* key_idx1;
	int* key_idx2;
	int* key_idx3;
	int* key_idx4;
	int* ht1;
	int* ht2;
	int* ht3;
	int* ht4;
	int dim_len1;
	int dim_len2;
	int dim_len3; 
	int dim_len4;
	int min_key1;
	int min_key2;
	int min_key3;
	int min_key4;

	// probeArgsGPU()
	// : key_idx1(NULL), key_idx2(NULL), key_idx3(NULL), key_idx4(NULL),
	// 	ht1(NULL), ht2(NULL), ht3(NULL), ht4(NULL),
	// 	dim_len1(0), dim_len2(0), dim_len3(0), dim_len4(0),
	// 	min_key1(0), min_key2(0), min_key3(0), min_key4(0) {}
} probeArgsGPU;

typedef struct probeArgsCPU {
	int* key_col1;
	int* key_col2;
	int* key_col3;
	int* key_col4;
	int* ht1;
	int* ht2;
	int* ht3;
	int* ht4;
	int dim_len1;
	int dim_len2;
	int dim_len3; 
	int dim_len4;
	int min_key1;
	int min_key2;
	int min_key3;
	int min_key4;

	// probeArgsCPU()
	// : key_col1(NULL), key_col2(NULL), key_col3(NULL), key_col4(NULL),
	// 	ht1(NULL), ht2(NULL), ht3(NULL), ht4(NULL),
	// 	dim_len1(0), dim_len2(0), dim_len3(0), dim_len4(0),
	// 	min_key1(0), min_key2(0), min_key3(0), min_key4(0) {}
} probeArgsCPU;

typedef struct buildArgsGPU {
	int *key_idx;
	int *val_idx;
	int num_slots;
	int val_min;
	int val_max;

	// buildArgsGPU()
	// : key_idx(NULL), val_idx(NULL), num_slots(0), val_min(0) {}
} buildArgsGPU;

typedef struct buildArgsCPU {
	int *key_col;
	int *val_col;
	int num_slots;
	int val_min;
	int val_max;

	// buildArgsCPU()
	// : key_col(NULL), val_col(NULL), num_slots(0), val_min(0) {}
} buildArgsCPU;

typedef struct groupbyArgsGPU {
	int* aggr_idx1;
	int* aggr_idx2;
	int* group_idx1;
	int* group_idx2;
	int* group_idx3;
	int* group_idx4;
	int min_val1;
	int min_val2;
	int min_val3;
	int min_val4;
	int unique_val1;
	int unique_val2;
	int unique_val3;
	int unique_val4;
	int total_val;
	int mode;

	group_func_t<int> d_group_func;

	// groupbyArgsGPU()
	// : aggr_idx1(NULL), aggr_idx2(NULL), group_idx1(NULL), group_idx2(NULL), group_idx3(NULL), group_idx4(NULL),
	//   min_val1(0), min_val2(0), min_val3(0), min_val4(0), unique_val1(0), unique_val2(0), unique_val3(0), unique_val4(0),
	//   total_val(0), mode(0) {}
} groupbyArgsGPU;

typedef struct groupbyArgsCPU {
	int* aggr_col1;
	int* aggr_col2;
	int* group_col1;
	int* group_col2;
	int* group_col3;
	int* group_col4;
	int min_val1;
	int min_val2;
	int min_val3;
	int min_val4;
	int unique_val1;
	int unique_val2;
	int unique_val3;
	int unique_val4;
	int total_val;
	int mode;

	group_func_t<int> h_group_func;

	// groupbyArgsCPU()
	// : aggr_col1(NULL), aggr_col2(NULL), group_col1(NULL), group_col2(NULL), group_col3(NULL), group_col4(NULL),
	//   min_val1(0), min_val2(0), min_val3(0), min_val4(0), unique_val1(0), unique_val2(0), unique_val3(0), unique_val4(0),
	//   total_val(0), mode(0) {}
} groupbyArgsCPU;

typedef struct filterArgsGPU {
	int* filter_idx1;
	int* filter_idx2;
	int compare1;
	int compare2;
	int compare3;
	int compare4;
	int mode1;
	int mode2;

	filter_func_t_dev<int, 128, 4> d_filter_func1;
	filter_func_t_dev<int, 128, 4> d_filter_func2;

	// filterArgsGPU()
	// : filter_idx1(NULL), filter_idx2(NULL), compare1(0), compare2(0), compare3(0), compare4(0),
	// mode1(0), mode2(0) {}
} filterArgsGPU;

typedef struct filterArgsCPU {
	int* filter_col1;
	int* filter_col2;
	int compare1;
	int compare2;
	int compare3;
	int compare4;
	int mode1;
	int mode2;

	filter_func_t_host<int> h_filter_func1;
	filter_func_t_host<int> h_filter_func2;

	// filterArgsCPU()
	// : filter_col1(NULL), filter_col2(NULL), compare1(0), compare2(0), compare3(0), compare4(0),
	// mode1(0), mode2(0) {}
} filterArgsCPU;

typedef struct offsetGPU {
	int* lo_off;
	int* dim_off1;
	int* dim_off2;
	int* dim_off3;
	int* dim_off4;

	// offsetGPU()
	// : lo_off(NULL), dim_off1(NULL), dim_off2(NULL), dim_off3(NULL), dim_off4(NULL) {}
} offsetGPU;

typedef struct offsetCPU {
	int* h_lo_off;
	int* h_dim_off1;
	int* h_dim_off2;
	int* h_dim_off3;
	int* h_dim_off4;

	// offsetCPU()
	// : h_lo_off(NULL), h_dim_off1(NULL), h_dim_off2(NULL), h_dim_off3(NULL), h_dim_off4(NULL) {}
} offsetCPU;

#endif