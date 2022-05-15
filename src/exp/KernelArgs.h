#ifndef _KERNEL_ARGS_H_
#define _KERNEL_ARGS_H_

#include "crystal/crystal.cuh"
#include "BlockLibrary.cuh"

// #define BLOCK_T 128
// #define ITEMS_PER_T 4

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

  int *ht_p, *ht_c, *ht_s, *ht_d;
  int *d_ht_p, *d_ht_c, *d_ht_s, *d_ht_d;

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
	int num_join;
	int num_join_mw;
	int** key_idx;
	int** ht;
	int* S;
	int* R;
	int* dim_len;
	int* min_key;

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

} probeArgsCPU;

typedef struct buildArgsGPU {
	int *key_idx;
	int *val_idx;
	int num_slots;
	int val_min;

} buildArgsGPU;

typedef struct buildArgsCPU {
	int *key_col;
	int *val_col;
	int num_slots;
	int val_min;

} buildArgsCPU;

typedef struct groupbyArgsGPU {
	int num_group;
	int num_group_mw;
	int** group_idx;
	int* min_val;
	int* unique_val;
	int* G;
	int total_val;

} groupbyArgsGPU;

typedef struct aggrArgsGPU {
	int num_aggr;
	int num_aggr_mw;
	int** aggr_idx;
	int num_func;

	aggr_func_t<int>* d_aggr_func;
} aggrArgsGPU;

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
} groupbyArgsCPU;

typedef struct filterArgsGPU {
	int** filter_idx;
	int* compare1;
	int* compare2;

	filter_func_t_dev<int, 128, 4>* d_filter_func;
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
} filterArgsCPU;

typedef struct offsetGPU {
	int* lo_off;
	int* dim_off1;
	int* dim_off2;
	int* dim_off3;
	int* dim_off4;

} offsetGPU;

typedef struct offsetCPU {
	int* h_lo_off;
	int* h_dim_off1;
	int* h_dim_off2;
	int* h_dim_off3;
	int* h_dim_off4;

} offsetCPU;

#endif