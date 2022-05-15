#include "QueryProcessing.h"
#include "QueryOptimizer.h"
#include "CPUGPUProcessing.h"
#include "CacheManager.h"
#include "CPUProcessing.h"
#include "CostModel.h"
// #include "pcm-cache.cpp"

// #define PROFILE 1

// #define CHECK_CU_ERROR(err, cufunc)                                     \
//     if (err != CUDA_SUCCESS) { printf ("Error %d for CUDA Driver API function '%s'\n", err, cufunc); return -1; }

// tbb::task_scheduler_init init(24);

int main() {

#ifdef PROFILE
	int pid = getpid();
	cout << pid << endl;
	string command = "sudo taskset -p --cpu-list 0-23 " + to_string(pid);
	int ret = system(command.c_str());
	assert(ret == 0);
#endif

	cudaSetDevice(0);
	CUdevice device;
	cuDeviceGet(&device, 0);

	bool verbose = 0;

	// srand(249); //for phase change
	srand(123);

	unsigned int size = 52428800 * 24; //400 MB
	double alpha = 1.0;
	bool custom = true;
	bool skipping = true;
	bool emat = false;
	bool nopipe = false;
	bool HE = false;

	//2, 4, 8, 12, 16, 20, 24, 32, 40
	//4, 10, 20, 30, 40
	
	//TODO: make it support cache size > 8 GB (there are lots of integer overflow resulting in negative offset to the gpuCache) should have used unsigned int everywhere
	CPUGPUProcessing* cgp = new CPUGPUProcessing(size, 0, 52428800 * 15, 52428800 * 20, verbose, custom, skipping);
	QueryProcessing* qp;

	// cout << "Profiling" << endl;
	// qp = new QueryProcessing(cgp, 0);
	// qp->profile();
	// delete qp;

	// cgp->cm->resetCache(52428800 / 2, 209715200 / 2, 536870912, 536870912);
	// cgp->cm->resetCache(314572800, 209715200 / 2, 536870912, 536870912);
	// cgp->cm->resetCache(629145600, 209715200 / 2, 536870912, 536870912);

	cout << endl;

	CacheManager* cm = cgp->cm;

	// cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 100);
	// cm->cacheColumnSegmentInGPU(cm->lo_suppkey, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_custkey, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_partkey, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_revenue, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_supplycost, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_discount, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_quantity, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_extendedprice, 0);

	// cm->cacheColumnSegmentInGPU(cm->d_datekey, cm->d_datekey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->d_year, 0);
	// cm->cacheColumnSegmentInGPU(cm->d_yearmonthnum, 0);
	// cm->cacheColumnSegmentInGPU(cm->p_partkey, cm->p_partkey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->p_category, 0);
	// cm->cacheColumnSegmentInGPU(cm->p_brand1, 0);
	// cm->cacheColumnSegmentInGPU(cm->p_mfgr, 0);
	// cm->cacheColumnSegmentInGPU(cm->c_custkey, cm->c_custkey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->c_region, 0);
	// cm->cacheColumnSegmentInGPU(cm->c_nation, 0);
	// cm->cacheColumnSegmentInGPU(cm->c_city, 0);
	// cm->cacheColumnSegmentInGPU(cm->s_suppkey, cm->s_suppkey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->s_region, 0);
	// cm->cacheColumnSegmentInGPU(cm->s_nation, 0);
	// cm->cacheColumnSegmentInGPU(cm->s_city, 0);

	// cm->cacheColumnSegmentInGPU(cm->d_datekey, cm->d_datekey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->d_year, cm->d_year->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->d_yearmonthnum, cm->d_yearmonthnum->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->p_partkey, cm->p_partkey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->p_category, cm->p_category->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->p_brand1, cm->p_brand1->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->p_mfgr, cm->p_mfgr->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->c_custkey, cm->c_custkey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->c_region, cm->c_region->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->c_nation, cm->c_nation->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->c_city, cm->c_city->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->s_suppkey, cm->s_suppkey->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->s_region, cm->s_region->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->s_nation, cm->s_nation->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->s_city, cm->s_city->total_segment);

	bool exit = 0;
	string input, query, many, policy;
	int many_query;
	ReplacementPolicy repl_policy;
	double time = 0;
	double malloc_time_total = 0, execution_time = 0, optimization_time = 0, merging_time = 0;
	double time1 = 0, time2 = 0;
	unsigned long long cpu_to_gpu = 0, gpu_to_cpu = 0;
	unsigned long long cpu_to_gpu1 = 0, gpu_to_cpu1 = 0;
	unsigned long long cpu_to_gpu2 = 0, gpu_to_cpu2 = 0;
	unsigned long long gpu_traffic = 0, cpu_traffic = 0, repl_traffic = 0;
	double malloc_time_total1 = 0, execution_time1 = 0, optimization_time1 = 0, merging_time1 = 0;
	double malloc_time_total2 = 0, execution_time2 = 0, optimization_time2 = 0, merging_time2 = 0;
	double repl_time = 0;
	Distribution dist = Zipf;
	string dist_string;
	FILE *fptr, *fptr2;
	if (dist == Norm) dist_string = "Norm";
	else if (dist == None) dist_string = "None";
	else if (dist == Zipf) dist_string = "Zipf";
	int processed_segment = 0;
	int skipped_segment = 0;
	double mean = 1;

	qp = new QueryProcessing(cgp, verbose, dist);

	if (dist == Zipf) {
		qp->qo->setDistributionZipfian(alpha);
	} else if (dist == Norm) {
		qp->qo->setDistributionNormal(mean, 0.5);
	}

#ifdef PROFILE
	PAPI_library_init( PAPI_VER_CURRENT );
	InitMonitor();

	int retval;
	char const *EventName[] = { "cuda:::dram__bytes_read.sum:device=0", "cuda:::dram__bytes_write.sum:device=0"}; // CUPTI_11 event.
	int* events = new int [NUM_EVENTS];
	int EventSet = PAPI_NULL;

	for(int i = 0; i < NUM_EVENTS; i++ ){
	    retval = PAPI_event_name_to_code( (char *)EventName[i], &events[i] );
	    assert(retval == PAPI_OK);
	}

	retval = PAPI_create_eventset( &EventSet ); assert(retval == PAPI_OK);
	retval = PAPI_add_events( EventSet, events, NUM_EVENTS ); assert(retval == PAPI_OK);

	long long init_metric[NUM_EVENTS];
	for(int i = 0; i < NUM_EVENTS; i++ ){
		init_metric[i] = 0;
	}
#endif

	CUcontext sessionCtx = NULL;
	CUcontext poppedCtx, curCtx;
	CHECK_CU_ERROR( cuCtxCreate(&sessionCtx, 0, device), "cuCtxCreate");
	cuCtxGetCurrent(&curCtx); cout << curCtx << endl;
	CHECK_CU_ERROR( cuCtxPopCurrent(&poppedCtx), "cuCtxPopCurrent" );
	cuCtxGetCurrent(&curCtx); cout << curCtx << endl;

#ifdef PROFILE
	//Eliminating initial values (not sure why but there is 500MB initial write and after trial and error this is the only way to eliminate it)
	CHECK_CU_ERROR(cuCtxPushCurrent(sessionCtx), "cuCtxPushCurrent");
	retval = PAPI_start( EventSet ); assert(retval == PAPI_OK);
	CHECK_CU_ERROR( cuCtxPopCurrent(&sessionCtx), "cuCtxPopCurrent" );

	parallel_for(int(0), 1, [=](int j){

		CUcontext poppedCtx;
		cuCtxPushCurrent(sessionCtx);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		cudaStreamDestroy(stream);
		cuCtxPopCurrent(&poppedCtx);

	});
    CHECK_CU_ERROR(cuCtxPushCurrent(sessionCtx), "cuCtxPushCurrent");
    CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );
    CHECK_CU_ERROR( cuCtxPopCurrent(&sessionCtx), "cuCtxPopCurrent" );
	retval = PAPI_stop( EventSet, init_metric ); assert(retval == PAPI_OK);
	for(int i = 0; i < NUM_EVENTS; i++ ) {
		printf( "stop: %12llu \t=0X%016llX \t\t --> %s \n", init_metric[i], init_metric[i], EventName[i] );
	}
#endif

	while (!exit) {
		cout << "Select Options:" << endl;
		cout << "1. Run Specific Query" << endl;
		cout << "2. Run Random Queries" << endl;
		cout << "3. Run Experiment" << endl;
		cout << "4. Replacement" << endl;
		cout << "5. Dump Trace" << endl;
		cout << "6. Exit" << endl;
		cout << "cache. Cache Specific Column" << endl;
		cout << "clear. Delete Columns from GPU" << endl;
		cout << "custom. Toggle custom malloc" << endl;
		cout << "skipping. Toggle segment skipping" << endl;
		cout << "nopipe. Toggle operator pipelining" << endl;
		cout << "emat. Toggle late materialization" << endl;
		cout << "HE. Toggle segment-level query execution" << endl;
		cout << "Your Input: ";
		cin >> input;

#ifdef PROFILE
		cout << endl;
		cout << "We are profiling" << endl;

		long long p_values[NUM_EVENTS];
		long long read = 0, write = 0;
		
		for(int i = 0; i < NUM_EVENTS; i++ ){
			p_values[i] = 0;
		}
#endif

		if (input.compare("1") == 0) {
			time = 0; cpu_traffic = 0; gpu_traffic = 0; malloc_time_total = 0; cpu_to_gpu = 0; gpu_to_cpu = 0; execution_time = 0; optimization_time = 0; merging_time = 0;
			cgp->resetTime();
			cout << "Input Query: ";
			cin >> query;
			qp->setQuery(stoi(query));

#ifdef PROFILE
			// CHECK_CU_ERROR(cuCtxGetCurrent(&curCtx), "cuCtxGetCurrent"); cout << curCtx << endl;
			// CHECK_CU_ERROR(cuCtxSetCurrent(sessionCtx), "cuCtxSetCurrent");
			CHECK_CU_ERROR(cuCtxPushCurrent(sessionCtx), "cuCtxPushCurrent");
			retval = PAPI_start( EventSet ); assert(retval == PAPI_OK);
			CHECK_CU_ERROR( cuCtxPopCurrent(&sessionCtx), "cuCtxPopCurrent" );
			// CHECK_CU_ERROR(cuCtxSetCurrent(curCtx), "cuCtxSetCurrent");
			StartMonitor();
#endif

			if (emat) {
				time1 = qp->processQueryEMat(sessionCtx);
				malloc_time_total1 = cgp->malloc_time_total;
				cpu_to_gpu1 = cgp->cpu_to_gpu_total;
				gpu_to_cpu1 = cgp->gpu_to_cpu_total;
				execution_time1 = cgp->execution_total;
				optimization_time1 = cgp->optimization_total;
				merging_time1 = cgp->merging_total;
				cgp->resetTime();	

				time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
				execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;

			} else if (nopipe) {	
				time1 = qp->processQueryNP(sessionCtx);
				malloc_time_total1 = cgp->malloc_time_total;
				cpu_to_gpu1 = cgp->cpu_to_gpu_total;
				gpu_to_cpu1 = cgp->gpu_to_cpu_total;
				execution_time1 = cgp->execution_total;
				optimization_time1 = cgp->optimization_total;
				merging_time1 = cgp->merging_total;
				cgp->resetTime();

				time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
				execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;

			} else if (HE) {
				time1 = qp->processQueryHE(sessionCtx);
				malloc_time_total1 = cgp->malloc_time_total;
				cpu_to_gpu1 = cgp->cpu_to_gpu_total;
				gpu_to_cpu1 = cgp->gpu_to_cpu_total;
				execution_time1 = cgp->execution_total;
				optimization_time1 = cgp->optimization_total;
				merging_time1 = cgp->merging_total;
				cgp->resetTime();

				time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
				execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;

			} else {
				time1 = qp->processQuery(sessionCtx);
				malloc_time_total1 = cgp->malloc_time_total;
				cpu_to_gpu1 = cgp->cpu_to_gpu_total;
				gpu_to_cpu1 = cgp->gpu_to_cpu_total;
				execution_time1 = cgp->execution_total;
				optimization_time1 = cgp->optimization_total;
				merging_time1 = cgp->merging_total;
				cgp->resetTime();

				time2 = qp->processQuery2(sessionCtx);
				malloc_time_total2 = cgp->malloc_time_total;
				cpu_to_gpu2 = cgp->cpu_to_gpu_total;
				gpu_to_cpu2 = cgp->gpu_to_cpu_total;
				execution_time2 = cgp->execution_total;
				optimization_time2 = cgp->optimization_total;
				merging_time2 = cgp->merging_total;
				cgp->resetTime();

				if (time1 <= time2) {
					time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
					execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;
				} else {
					time += time2; cpu_to_gpu += cpu_to_gpu2; gpu_to_cpu += gpu_to_cpu2; malloc_time_total += malloc_time_total2;
					execution_time += execution_time2; optimization_time += optimization_time2; merging_time += merging_time2;
				}
			}

#ifdef PROFILE
				EndMonitor(read, write);
				cout << endl;

				CHECK_CU_ERROR(cuCtxPushCurrent(sessionCtx), "cuCtxPushCurrent");
		        CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );
		        CHECK_CU_ERROR( cuCtxPopCurrent(&sessionCtx), "cuCtxPopCurrent" );
				retval = PAPI_stop( EventSet, p_values ); assert(retval == PAPI_OK);
				for(int i = 0; i < NUM_EVENTS; i++ ) {
					printf( "stop: %12llu \t=0X%016llX \t\t --> %s \n", p_values[i] - init_metric[i], p_values[i] - init_metric[i], EventName[i] );
				}
				retval = PAPI_reset(EventSet); assert(retval == PAPI_OK);

			   	cout << "CPU Read: " << read << endl;
			   	cout << "CPU Write: " << write << endl;

			   	cout << "GPU Read: " << p_values[0] - init_metric[0] << endl;
			   	cout << "GPU Write: " << p_values[1] - init_metric[1] << endl;

				cout << "CPU to GPU traffic: " << cpu_to_gpu  << endl;
				cout << "GPU to CPU traffic: " << gpu_to_cpu  << endl;

				gpu_traffic = 0;
				for(int i = 0; i < NUM_EVENTS; i++ ) {
					// printf( "stop: %12llu \t=0X%016llX \t\t --> %s \n", p_values[i] - init_metric[i], p_values[i] - init_metric[i], EventName[i] );
					gpu_traffic += (p_values[i] - init_metric[i]);
					init_metric[i] = p_values[i];
				}
	 			cpu_traffic = read + write;
#endif	

		} else if (input.compare("2") == 0) {
			time = 0; cpu_traffic = 0; gpu_traffic = 0; malloc_time_total = 0; cpu_to_gpu = 0; gpu_to_cpu = 0; execution_time = 0; optimization_time = 0; merging_time = 0;
			cout << "How many queries: ";
			cin >> many;
			many_query = stoi(many);
			cgp->resetTime();
			cgp->qo->processed_segment = 0;
			cgp->qo->skipped_segment = 0;
			cout << "Executing Random Query" << endl;
			for (int i = 0; i < many_query; i++) {
				qp->generate_rand_query();

				time1 = qp->processQuery(sessionCtx);
				malloc_time_total1 = cgp->malloc_time_total;
				cpu_to_gpu1 = cgp->cpu_to_gpu_total;
				gpu_to_cpu1 = cgp->gpu_to_cpu_total;
				execution_time1 = cgp->execution_total;
				optimization_time1 = cgp->optimization_total;
				merging_time1 = cgp->merging_total;
				cgp->resetTime();

				time2 = qp->processQuery2(sessionCtx);
				malloc_time_total2 = cgp->malloc_time_total;
				cpu_to_gpu2 = cgp->cpu_to_gpu_total;
				gpu_to_cpu2 = cgp->gpu_to_cpu_total;
				execution_time2 = cgp->execution_total;
				optimization_time2 = cgp->optimization_total;
				merging_time2 = cgp->merging_total;
				cgp->resetTime();

				if (time1 <= time2) {
					time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
					execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;
				} else {
					time += time2; cpu_to_gpu += cpu_to_gpu2; gpu_to_cpu += gpu_to_cpu2; malloc_time_total += malloc_time_total2;
					execution_time += execution_time2; optimization_time += optimization_time2; merging_time += merging_time2;
				}
				
			}
			processed_segment = cgp->qo->processed_segment;
			skipped_segment = cgp->qo->skipped_segment;
			srand(123);
		} else if (input.compare("3") == 0) {
			time = 0; cpu_traffic = 0; gpu_traffic = 0; malloc_time_total = 0; cpu_to_gpu = 0; gpu_to_cpu = 0; execution_time = 0; optimization_time = 0; merging_time = 0;
			repl_traffic = 0; repl_time = 0;
			cout << "How many queries: ";
			cin >> many;
			many_query = stoi(many);

			cout << "Replacement Policy: ";
			cin >> policy;

			if (policy == "LRU") {
				repl_policy = LRU;
			} else if (policy == "LFU") {
				repl_policy = LFU;
			} else if (policy == "LRUSegmented") {
				repl_policy = LRUSegmented;
			} else if (policy == "LFUSegmented") {
				repl_policy = LFUSegmented;
			} else if (policy == "LRU2") {
				repl_policy = LRU2;
			} else if (policy == "LRU2Segmented") {
				repl_policy = LRU2Segmented;
			} else if (policy == "SemanticAware") {
				repl_policy = Segmented;
			} else {
				repl_policy = Segmented;
			}

			cuCtxGetCurrent(&curCtx); cout << curCtx << endl; cout <<sessionCtx << endl;

			cgp->resetTime();
			cgp->qo->processed_segment = 0;
			cgp->qo->skipped_segment = 0;

			if (dist != Norm) {
				cout << "Warmup" << endl;
				for (int i = 0; i < 100; i++) {
					qp->generate_rand_query();
					time1 = qp->processQuery(sessionCtx);
					cgp->resetTime();
				}
				cgp->cm->runReplacement(repl_policy);				
			}


			cout << "Run Experiment" << endl;

#ifndef PROFILE
			if (dist == Norm) {
				string runs = "logs/runs/" + dist_string + "/" + policy + to_string(size / (1048576) * 4);
				fptr = fopen(runs.c_str(), "w");
				if (fptr == NULL) assert(0);
			}
#endif


			for (int iter = 0; iter < 20; iter++) {

#ifdef PROFILE
				CHECK_CU_ERROR(cuCtxPushCurrent(sessionCtx), "cuCtxPushCurrent");
				retval = PAPI_start( EventSet ); assert(retval == PAPI_OK);
				CHECK_CU_ERROR( cuCtxPopCurrent(&sessionCtx), "cuCtxPopCurrent" );

				StartMonitor();
#endif

				for (int i = 0; i < many_query; i++) {
					qp->generate_rand_query();

					if (emat) {
						time1 = qp->processQueryEMat(sessionCtx);
						malloc_time_total1 = cgp->malloc_time_total;
						cpu_to_gpu1 = cgp->cpu_to_gpu_total;
						gpu_to_cpu1 = cgp->gpu_to_cpu_total;
						execution_time1 = cgp->execution_total;
						optimization_time1 = cgp->optimization_total;
						merging_time1 = cgp->merging_total;
						cgp->resetTime();	

						time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
						execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;

					} else if (nopipe) {	
						time1 = qp->processQueryNP(sessionCtx);
						malloc_time_total1 = cgp->malloc_time_total;
						cpu_to_gpu1 = cgp->cpu_to_gpu_total;
						gpu_to_cpu1 = cgp->gpu_to_cpu_total;
						execution_time1 = cgp->execution_total;
						optimization_time1 = cgp->optimization_total;
						merging_time1 = cgp->merging_total;
						cgp->resetTime();

						time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
						execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;

					} else if (HE) {
						time1 = qp->processQueryHE(sessionCtx);
						malloc_time_total1 = cgp->malloc_time_total;
						cpu_to_gpu1 = cgp->cpu_to_gpu_total;
						gpu_to_cpu1 = cgp->gpu_to_cpu_total;
						execution_time1 = cgp->execution_total;
						optimization_time1 = cgp->optimization_total;
						merging_time1 = cgp->merging_total;
						cgp->resetTime();

						time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
						execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;
					} else {
						time1 = qp->processQuery(sessionCtx);
						malloc_time_total1 = cgp->malloc_time_total;
						cpu_to_gpu1 = cgp->cpu_to_gpu_total;
						gpu_to_cpu1 = cgp->gpu_to_cpu_total;
						execution_time1 = cgp->execution_total;
						optimization_time1 = cgp->optimization_total;
						merging_time1 = cgp->merging_total;
						cgp->resetTime();

						time2 = qp->processQuery2(sessionCtx);
						malloc_time_total2 = cgp->malloc_time_total;
						cpu_to_gpu2 = cgp->cpu_to_gpu_total;
						gpu_to_cpu2 = cgp->gpu_to_cpu_total;
						execution_time2 = cgp->execution_total;
						optimization_time2 = cgp->optimization_total;
						merging_time2 = cgp->merging_total;
						cgp->resetTime();

						if (time1 <= time2) {
							time += time1; cpu_to_gpu += cpu_to_gpu1; gpu_to_cpu += gpu_to_cpu1; malloc_time_total += malloc_time_total1;
							execution_time += execution_time1; optimization_time += optimization_time1; merging_time += merging_time1;
						} else {
							time += time2; cpu_to_gpu += cpu_to_gpu2; gpu_to_cpu += gpu_to_cpu2; malloc_time_total += malloc_time_total2;
							execution_time += execution_time2; optimization_time += optimization_time2; merging_time += merging_time2;
						}
					}


				}

#ifdef PROFILE
				EndMonitor(read, write);
				cout << endl;

				CHECK_CU_ERROR(cuCtxPushCurrent(sessionCtx), "cuCtxPushCurrent");
		        CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );
		        CHECK_CU_ERROR( cuCtxPopCurrent(&sessionCtx), "cuCtxPopCurrent" );
				retval = PAPI_stop( EventSet, p_values ); assert(retval == PAPI_OK);
				for(int i = 0; i < NUM_EVENTS; i++ ) {
					printf( "stop: %12llu \t=0X%016llX \t\t --> %s \n", p_values[i] - init_metric[i], p_values[i] - init_metric[i], EventName[i] );
				}
				retval = PAPI_reset(EventSet); assert(retval == PAPI_OK);
#endif				

				cgp->cm->runReplacement(repl_policy, &repl_traffic);
				qp->percentageData();
				if (repl_policy == Segmented || repl_policy == LFUSegmented) cgp->cm->newEpoch(0.5);
				if (repl_policy == LRU2Segmented) cgp->cm->newEpoch(2.0);

#ifndef PROFILE
				if (dist == Norm) {
					fprintf(fptr, "{\"iter\":%d,\"cache_size\":%u,\"cumulated_time\":%.2f,\"execution_time\":%.2f,\"merging_time\":%.2f,\"optimization_time\":%.2f}\n", \
						iter, size, time, execution_time, merging_time, optimization_time);
					time = 0; execution_time = 0; optimization_time = 0; merging_time = 0;
					if ((iter + 1) % 5 == 0) {
						// mean = ((int) mean + 3) % 7;
						if (iter == 4) mean = 4;
						else if (iter == 9) mean = 2;
						else if (iter == 14) mean = 5;
						qp->qo->setDistributionNormal(mean, 0.5);
						// cgp->cm->deleteAll();
					}
					
				}
#endif

			}

			cout << "Replacement traffic: " << repl_traffic << endl;

			if (dist == Norm) {
				fclose(fptr);
			}

#ifdef PROFILE

			if (custom && skipping && !nopipe && !emat) {
				string traf = "logs/traffic/" + dist_string + "/" + policy + to_string(size / (1048576) * 4);
			    fptr2 = fopen(traf.c_str(), "w");
			    if (fptr2 == NULL)
			    {
			        printf("Could not open file\n");
			        assert(0);
			    }
			   	fprintf(fptr2, "{\"cache_size\":%u,\"gpu_read\":%llu,\"gpu_write\":%llu,\"cpu_read\":%llu,\"cpu_write\":%llu,\"cpu_to_gpu\":%llu,\"gpu_to_cpu\":%llu}\n", \
			   		size, p_values[0] - init_metric[0], p_values[1] - init_metric[1], read, write, cpu_to_gpu, gpu_to_cpu);
			   	fclose(fptr2);
			}

		   	if (dist == None && repl_policy == Segmented) {
		   		string opt;
		   		if (!custom && !skipping && nopipe && emat && !HE)
		   			opt = "logs/traffic/opt/NoOpt" + to_string(size / (1048576) * 4);
		   		else if (custom && !skipping && nopipe && emat && !HE)
					opt = "logs/traffic/opt/LiteMalloc" + to_string(size / (1048576) * 4);
				else if (custom && !skipping && nopipe && !emat && !HE)
					opt = "logs/traffic/opt/LateMat" + to_string(size / (1048576) * 4);
				else if (custom && !skipping && !nopipe && !emat && !HE)
					opt = "logs/traffic/opt/OpPipe" + to_string(size / (1048576) * 4);
				else if (custom && skipping && !nopipe && !emat && !HE)
					opt = "logs/traffic/opt/Skipping" + to_string(size / (1048576) * 4);

			    fptr2 = fopen(opt.c_str(), "w");
			    if (fptr2 == NULL)
			    {
			        printf("Could not open file\n");
			        assert(0);
			    }
			   	fprintf(fptr2, "{\"cache_size\":%u,\"gpu_read\":%llu,\"gpu_write\":%llu,\"cpu_read\":%llu,\"cpu_write\":%llu,\"cpu_to_gpu\":%llu,\"gpu_to_cpu\":%llu}\n", \
			   		size, p_values[0] - init_metric[0], p_values[1] - init_metric[1], read, write, cpu_to_gpu, gpu_to_cpu);
			   	fclose(fptr2);
		   	}

		   	// cout << "Read: " << read << " Write: " << write  << endl;

		   	cout << "CPU Read: " << read << endl;
		   	cout << "CPU Write: " << write << endl;

		   	cout << "GPU Read: " << p_values[0] - init_metric[0] << endl;
		   	cout << "GPU Write: " << p_values[1] - init_metric[1] << endl;

			cout << "CPU to GPU traffic: " << cpu_to_gpu  << endl;
			cout << "GPU to CPU traffic: " << gpu_to_cpu  << endl;

			gpu_traffic = 0;
			for(int i = 0; i < NUM_EVENTS; i++ ) {
				// printf( "stop: %12llu \t=0X%016llX \t\t --> %s \n", p_values[i] - init_metric[i], p_values[i] - init_metric[i], EventName[i] );
				gpu_traffic += (p_values[i] - init_metric[i]);
				init_metric[i] = p_values[i];
			}
 			cpu_traffic = read + write;

#endif

			processed_segment = cgp->qo->processed_segment;
			skipped_segment = cgp->qo->skipped_segment;
			// cgp->cm->deleteAll();
			srand(123);

#ifndef PROFILE
			if (dist != Norm) {

				if (custom && skipping && !nopipe && !emat && !HE) {
					string runs;
					if (dist == None) runs = "logs/runs/" + dist_string + "/" + policy + to_string(size / (1048576) * 4);
					else if (dist == Zipf) runs = "logs/runs/" + dist_string + "/" + policy + to_string(size / (1048576) * 4) + "alpha" + to_string( (int) (alpha * 10) );

				    fptr = fopen(runs.c_str(), "w");
				    if (fptr == NULL)
				    {
				        printf("Could not open file\n");
				        assert(0);
				    }
				   	fprintf(fptr, "{\"cache_size\":%u,\"cumulated_time\":%.2f,\"execution_time\":%.2f,\"merging_time\":%.2f,\"optimization_time\":%.2f}\n", \
				   		size, time, execution_time, merging_time, optimization_time);
				   	fclose(fptr);
				}	

			   	if (dist == None && repl_policy == Segmented) {
			   		string opt;
			   		if (!custom && !skipping && nopipe && emat && !HE)
			   			opt = "logs/runs/opt/NoOpt" + to_string(size / (1048576) * 4);
			   		else if (custom && !skipping && nopipe && emat && !HE)
						opt = "logs/runs/opt/LiteMalloc" + to_string(size / (1048576) * 4);
					else if (custom && !skipping && nopipe && !emat && !HE)
						opt = "logs/runs/opt/LateMat" + to_string(size / (1048576) * 4);
					else if (custom && !skipping && !nopipe && !emat && !HE)
						opt = "logs/runs/opt/OpPipe" + to_string(size / (1048576) * 4);
					else if (custom && skipping && !nopipe && !emat && !HE)
						opt = "logs/runs/opt/Skipping" + to_string(size / (1048576) * 4);

					if (custom && skipping && !nopipe && !emat && HE) {
						opt = "logs/runs/opt/HetExchange" + to_string(size / (1048576) * 4);
					}

				    fptr = fopen(opt.c_str(), "w");
				    if (fptr == NULL)
				    {
				        printf("Could not open file\n");
				        assert(0);
				    }
				   	fprintf(fptr, "{\"cache_size\":%u,\"cumulated_time\":%.2f,\"execution_time\":%.2f,\"merging_time\":%.2f,\"optimization_time\":%.2f}\n", \
				   		size, time, execution_time, merging_time, optimization_time);
				   	fclose(fptr);
			   	}

			}
#endif

		} else if (input.compare("4") == 0) {
			cout << "Replacement Policy: ";
			cin >> policy;

			if (policy == "LRU") {
				repl_policy = LRU;
			} else if (policy == "LFU") {
				repl_policy = LFU;
			} else if (policy == "LRUSegmented") {
				repl_policy = LRUSegmented;
			} else if (policy == "LFUSegmented") {
				repl_policy = LFUSegmented;
			} else if (policy == "LRU2") {
				repl_policy = LRU2;
			} else if (policy == "LRU2Segmented") {
				repl_policy = LRU2Segmented;
			} else if (policy == "SemanticAware") {
				repl_policy = Segmented;
			}

			cgp->cm->runReplacement(repl_policy);
			qp->percentageData();
			srand(123);
		} else if (input.compare("5") == 0) {
			string filename;
			cout << "File name: ";
			cin >> filename;
			qp->dumpTrace("logs/"+filename);
			cout << "Dumped Trace" << endl;
		} else if (input.compare("cache") == 0) {
			string column_name;
			int ret;
			do {
				cout << "	Column to cache: ";
				cin >> column_name;
				ret = cgp->cm->cacheSpecificColumn(column_name);
			} while (ret != 0);
		} else if (input.compare("clear") == 0) {
			cgp->cm->deleteAll();
		} else if (input.compare("skipping") == 0) {
			skipping = !skipping;
			cgp->skipping = skipping;
			cgp->qo->skipping = skipping;
			qp->skipping = skipping;
			if (skipping) cout << "Segment skipping is enabled" << endl;
			else cout << "Segment skipping is disabled" << endl;
		} else if (input.compare("custom") == 0) {
			custom = !custom;
			cgp->custom = custom;
			cgp->qo->custom = custom;
			qp->custom = custom;
			if (custom) cout << "Custom malloc is enabled" << endl;
			else cout << "Custom malloc is disabled" << endl;		
		} else if (input.compare("emat") == 0) {
			emat = !emat;
			if (emat) cout << "Early Materialization" << endl;
			else cout << "Late Materialization" << endl;		
		} else if (input.compare("nopipe") == 0) {
			nopipe = !nopipe;
			if (nopipe) cout << "Pipelining is disabled" << endl;
			else cout << "Pipelining is enabled" << endl;		
		} else if (input.compare("HE") == 0) {
			HE = !HE;
			if (HE) cout << "HetExchange execution" << endl;
			else cout << "Segment level query execution" << endl;		
		} else {
			exit = true;
		}

		cout << endl;
		cout << "Cumulated Time: " << time << endl;
		cout << "CPU traffic: " << cpu_traffic << endl;
		cout << "GPU traffic: " << gpu_traffic << endl;
		cout << "CPU to GPU traffic: " << cpu_to_gpu  << endl;
		cout << "GPU to CPU traffic: " << gpu_to_cpu  << endl;
		cout << "Malloc time: " << malloc_time_total << endl;
		cout << "Execution time: " << execution_time << endl;
		cout << "Optimization time: " << optimization_time << endl;
		cout << "Merging time: " << merging_time << endl;
		cout << "Fraction Skipped Segment: " << skipped_segment * 1.0 /(processed_segment + skipped_segment) << endl;
		cout << endl;

		// cout << endl;
	 //    cout<< "{"
	 //        << "\"cache_size\":" << size 
	 //        << ",\"cumulated_time\":" << time_query
	 //        << "}" << endl;


	}

    if (sessionCtx != NULL) {
        cuCtxDestroy(sessionCtx);
    }


#ifdef PROFILE
		retval = PAPI_cleanup_eventset(EventSet); assert(retval == PAPI_OK);
		retval = PAPI_destroy_eventset(&EventSet); assert(retval == PAPI_OK);
		PAPI_shutdown();
#endif

}