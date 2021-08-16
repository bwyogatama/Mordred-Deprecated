#ifndef _KERNEL_ARGS_H_
#define _KERNEL_ARGS_H_

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

	// buildArgsGPU()
	// : key_idx(NULL), val_idx(NULL), num_slots(0), val_min(0) {}
} buildArgsGPU;

typedef struct buildArgsCPU {
	int *key_col;
	int *val_col;
	int num_slots;
	int val_min;

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