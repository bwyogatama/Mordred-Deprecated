#include "QueryProcessing.h"
#include "QueryOptimizer.h"
#include "CPUGPUProcessing.h"
#include "CacheManager.h"
#include "CPUProcessing.h"
#include "CostModel.h"

int main() {

	cudaSetDevice(0);
	CUdevice device;
	cuDeviceGet(&device, 0);

	bool verbose = 0;

	srand(123);

	unsigned int size = 52428800 * 24; //400 MB
	double alpha = 1.0;
	bool custom = true;
	bool skipping = true;
	bool emat = false;
	bool nopipe = false;
	bool HE = false;
	
	CPUGPUProcessing* cgp = new CPUGPUProcessing(size, 0, 52428800 * 15, 52428800 * 20, verbose, custom, skipping);
	QueryProcessing* qp;

	cout << endl;

	CacheManager* cm = cgp->cm;

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

	CUcontext sessionCtx = NULL;
	CUcontext poppedCtx, curCtx;
	CHECK_CU_ERROR( cuCtxCreate(&sessionCtx, 0, device), "cuCtxCreate");
	cuCtxGetCurrent(&curCtx); cout << curCtx << endl;
	CHECK_CU_ERROR( cuCtxPopCurrent(&poppedCtx), "cuCtxPopCurrent" );
	cuCtxGetCurrent(&curCtx); cout << curCtx << endl;

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

		if (input.compare("1") == 0) {
			time = 0; cpu_traffic = 0; gpu_traffic = 0; malloc_time_total = 0; cpu_to_gpu = 0; gpu_to_cpu = 0; execution_time = 0; optimization_time = 0; merging_time = 0;
			cgp->resetTime();
			cout << "Input Query: ";
			cin >> query;
			qp->setQuery(stoi(query));

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


			for (int iter = 0; iter < 20; iter++) {

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

				cgp->cm->runReplacement(repl_policy, &repl_traffic);
				qp->percentageData();
				if (repl_policy == Segmented || repl_policy == LFUSegmented) cgp->cm->newEpoch(0.5);
				if (repl_policy == LRU2Segmented) cgp->cm->newEpoch(2.0);

				if (dist == Norm) {
					fprintf(fptr, "{\"iter\":%d,\"cache_size\":%u,\"cumulated_time\":%.2f,\"execution_time\":%.2f,\"merging_time\":%.2f,\"optimization_time\":%.2f}\n", \
						iter, size, time, execution_time, merging_time, optimization_time);
					time = 0; execution_time = 0; optimization_time = 0; merging_time = 0;
					if ((iter + 1) % 5 == 0) {
						if (iter == 4) mean = 4;
						else if (iter == 9) mean = 2;
						else if (iter == 14) mean = 5;
						qp->qo->setDistributionNormal(mean, 0.5);
					}
					
				}

			}

			cout << "Replacement traffic: " << repl_traffic << endl;

			if (dist == Norm) {
				fclose(fptr);
			}

			processed_segment = cgp->qo->processed_segment;
			skipped_segment = cgp->qo->skipped_segment;
			srand(123);

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
						opt = "logs/runs/opt/NonSegmentGrouping" + to_string(size / (1048576) * 4);
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
			if (HE) cout << "Non Segment-grouping execution" << endl;
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


	}

    if (sessionCtx != NULL) {
        cuCtxDestroy(sessionCtx);
    }

}