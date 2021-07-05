class QueryParams{
public:

  int query;
  
  unordered_map<ColumnInfo*, int> min_key;
  unordered_map<ColumnInfo*, int> min_val;
  unordered_map<ColumnInfo*, int> unique_val;
  unordered_map<ColumnInfo*, int> dim_len;

  unordered_map<ColumnInfo*, int*> ht_CPU;
  unordered_map<ColumnInfo*, int*> ht_GPU;

  unordered_map<ColumnInfo*, int> compare1;
  unordered_map<ColumnInfo*, int> compare2;
  unordered_map<ColumnInfo*, int> mode;

  int total_val, mode_group;

  int *ht_p, *ht_c, *ht_s, *ht_d;
  int *d_ht_p, *d_ht_c, *d_ht_s, *d_ht_d;

  int* res;
  int* d_res;

  QueryParams(int _query);
};

void 
QueryParams::QueryParams(int _query): query(_query) {

  // chrono::high_resolution_clock::time_point st, finish;
  // chrono::duration<double> diff;

  // cudaEvent_t start, stop;   // variables that holds 2 events 
  // float time;


  if (query == 0) {

    params->mode[cm->d_year] = 0;
    params->compare1[cm->d_year] = 1993;
    params->mode[cm->lo_discount] = 1;
    params->compare1[cm->lo_discount] = 1;
    params->compare2[cm->lo_discount] = 3;
    params->mode[cm->lo_quantity] = 1;
    params->compare1[cm->lo_quantity] = 0;
    params->compare2[cm->lo_quantity] = 24;
    params->mode_group = 2;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 0;
    params->unique_val[cm->c_custkey] = 0;
    params->unique_val[cm->s_suppkey] = 0;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = 0;
    params->dim_len[cm->c_custkey] = 0;
    params->dim_len[cm->s_suppkey] = 0;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = 1;

    params->ht_p = NULL;
    params->ht_c = NULL;
    params->ht_s = NULL;
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));

    params->d_ht_d = cm->customCudaMalloc(4 * params->dim_len[cm->d_datekey]);
    params->d_ht_p = NULL;
    params->d_ht_c = NULL;
    params->d_ht_s = NULL;

    CubDebugExit(cudaMemset(params->d_ht_d, 0, 4 * params->dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 1) {

    params->mode[cm->s_region] = 0;
    params->compare1[cm->s_region] = 1;
    params->mode[cm->p_category] = 0;
    params->compare1[cm->p_category] = 1;
    params->mode_group = 0;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 7;
    params->unique_val[cm->c_custkey] = 0;
    params->unique_val[cm->s_suppkey] = 0;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = P_LEN;
    params->dim_len[cm->c_custkey] = 0;
    params->dim_len[cm->s_suppkey] = S_LEN;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = ((1998-1992+1) * (5 * 5 * 40));

    params->ht_p = cm->customMalloc(2 * params->dim_len[cm->p_partkey]);
    params->ht_c = NULL;
    params->ht_s = cm->customMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    params->d_ht_p = cm->customCudaMalloc(2 * params->dim_len[cm->p_partkey]);
    params->d_ht_s = cm->customCudaMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->d_ht_d = cm->customCudaMalloc(2 * params->dim_len[cm->d_datekey]);
    params->d_ht_c = NULL;

    // cudaEventCreate(&start);   // creating the event 1
    // cudaEventCreate(&stop);    // creating the event 2
    // cudaEventRecord(start, 0); // start measuring  the time

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
    memset(params->ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int));
    memset(params->ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));

    // cudaEventRecord(stop, 0);                  // Stop time measuring
    // cudaEventSynchronize(stop);               // Wait until the completion of all device 
    //                                           // work preceding the most recent call to cudaEventRecord()
    // cudaEventElapsedTime(&time, start, stop); // Saving the time measured

    CubDebugExit(cudaMemset(params->d_ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 2) {

    params->mode[cm->c_region] = 0;
    params->compare1[cm->c_region] = 2;
    params->mode[cm->s_region] = 0;
    params->compare1[cm->s_region] = 2;
    params->mode[cm->d_year] = 1;
    params->compare1[cm->d_year] = 1992;
    params->compare2[cm->d_year] = 1997;
    params->mode_group = 0;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 0;
    params->unique_val[cm->c_custkey] = 7;
    params->unique_val[cm->s_suppkey] = 25 * 7;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = 0;
    params->dim_len[cm->c_custkey] = C_LEN;
    params->dim_len[cm->s_suppkey] = S_LEN;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = ((1998-1992+1) * 25 * 25);

    params->ht_p = NULL;
    params->ht_c = cm->customMalloc(2 * params->dim_len[cm->c_custkey]);
    params->ht_s = cm->customMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
    memset(params->ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int));
    memset(params->ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));

    params->d_ht_c = cm->customCudaMalloc(2 * params->dim_len[cm->c_custkey]);
    params->d_ht_s = cm->customCudaMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->d_ht_d = cm->customCudaMalloc(2 * params->dim_len[cm->d_datekey]);
    params->d_ht_p = NULL;

    CubDebugExit(cudaMemset(params->d_ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));

  } else if (query == 3) {

    params->mode[cm->c_region] = 0;
    params->compare1[cm->c_region] = 1;
    params->mode[cm->s_region] = 0;
    params->compare1[cm->s_region] = 1;
    params->mode[cm->p_mfgr] = 2;
    params->compare1[cm->p_mfgr] = 0;
    params->compare2[cm->p_mfgr] = 1;
    params->mode_group = 1;

    params->min_key[cm->p_partkey] = 0;
    params->min_key[cm->c_custkey] = 0;
    params->min_key[cm->s_suppkey] = 0;
    params->min_key[cm->d_datekey] = 19920101;

    params->min_val[cm->p_partkey] = 0;
    params->min_val[cm->c_custkey] = 0;
    params->min_val[cm->s_suppkey] = 0;
    params->min_val[cm->d_datekey] = 1992;

    params->unique_val[cm->p_partkey] = 0;
    params->unique_val[cm->c_custkey] = 7;
    params->unique_val[cm->s_suppkey] = 0;
    params->unique_val[cm->d_datekey] = 1;

    params->dim_len[cm->p_partkey] = P_LEN;
    params->dim_len[cm->c_custkey] = C_LEN;
    params->dim_len[cm->s_suppkey] = S_LEN;
    params->dim_len[cm->d_datekey] = 19981230 - 19920101 + 1;

    params->total_val = ((1998-1992+1) * 25);

    params->ht_p = cm->customMalloc(4 * params->dim_len[cm->p_partkey]);
    params->ht_c = cm->customMalloc(2 * params->dim_len[cm->c_custkey]);
    params->ht_s = cm->customMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->ht_d = cm->customMalloc(2 * params->dim_len[cm->d_datekey]);

    memset(params->ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int));
    memset(params->ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int));
    memset(params->ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int));
    memset(params->ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int));

    params->d_ht_p = cm->customCudaMalloc(2 * params->dim_len[cm->p_partkey]);
    params->d_ht_s = cm->customCudaMalloc(2 * params->dim_len[cm->s_suppkey]);
    params->d_ht_d = cm->customCudaMalloc(2 * params->dim_len[cm->d_datekey]);
    params->d_ht_c = cm->customCudaMalloc(2 * params->dim_len[cm->c_custkey]);

    CubDebugExit(cudaMemset(params->d_ht_p, 0, 2 * params->dim_len[cm->p_partkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_s, 0, 2 * params->dim_len[cm->s_suppkey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_d, 0, 2 * params->dim_len[cm->d_datekey] * sizeof(int)));
    CubDebugExit(cudaMemset(params->d_ht_c, 0, 2 * params->dim_len[cm->c_custkey] * sizeof(int)));

  }

  // diff = finish - st;
  // cout << "Time Taken Total: " << time << endl;

  params->ht_GPU[cm->p_partkey] = params->d_ht_p;
  params->ht_GPU[cm->c_custkey] = params->d_ht_c;
  params->ht_GPU[cm->s_suppkey] = params->d_ht_s;
  params->ht_GPU[cm->d_datekey] = params->d_ht_d;

  params->ht_CPU[cm->p_partkey] = params->ht_p;
  params->ht_CPU[cm->c_custkey] = params->ht_c;
  params->ht_CPU[cm->s_suppkey] = params->ht_s;
  params->ht_CPU[cm->d_datekey] = params->ht_d;

  int res_array_size = params->total_val * 6;
  params->res = cm->customMalloc(res_array_size);
  memset(params->res, 0, res_array_size * sizeof(int));
     
  params->d_res = cm->customCudaMalloc(res_array_size);
  CubDebugExit(cudaMemset(params->d_res, 0, res_array_size * sizeof(int)));

};