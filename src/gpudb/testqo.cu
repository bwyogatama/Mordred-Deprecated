#include "QueryProcessing.h"
#include "QueryOptimizer.h"
#include "CPUGPUProcessing.h"
#include "CacheManager.h"
#include "CPUProcessing.h"
#include "CostModel.h"

// tbb::task_scheduler_init init(1);

int main() {

	bool verbose = 0;

	srand(123);

	unsigned int size = 52428800;
	double alpha = 0.5;
	bool custom = true;
	bool skipping = true;

	//2, 4, 8, 12, 16, 24, 32, 40
	//4, 10, 20, 30, 40
	
	//TODO: make it support cache size > 8 GB (there are lots of integer overflow resulting in negative offset to the gpuCache) should have used unsigned int everywhere
	CPUGPUProcessing* cgp = new CPUGPUProcessing(size * 24, 0, 52428800 * 15, 52428800 * 20, verbose, custom, skipping, alpha);
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

	// cm->cacheColumnSegmentInGPU(cm->lo_orderdate, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_suppkey, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_custkey, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_partkey, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_revenue, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_supplycost, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_discount, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_quantity, cm->lo_orderdate->total_segment);
	// cm->cacheColumnSegmentInGPU(cm->lo_extendedprice, cm->lo_orderdate->total_segment);

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
	string input;
	string query;
	string many;
	int many_query;
	double time = 0;
	double cpu_time_total = 0, gpu_time_total = 0, transfer_time_total = 0, malloc_time_total = 0;
	double time1 = 0, time2 = 0;
	double cpu_time_total1 = 0, cpu_time_total2 = 0;
	double gpu_time_total1 = 0, gpu_time_total2 = 0;
	double transfer_time_total1 = 0, transfer_time_total2 = 0;
	double malloc_time_total1 = 0, malloc_time_total2 = 0;
	bool skew = true;
	int processed_segment = 0;
	int skipped_segment = 0;

	qp = new QueryProcessing(cgp, verbose, skew);

	while (!exit) {
		cout << "Select Options:" << endl;
		cout << "1. Run Specific Query" << endl;
		cout << "2. Run Hybrid On Demand" << endl;
		cout << "3. Run Random Queries" << endl;
		cout << "4. Run Random Queries (HOD)" << endl;
		cout << "5. Update Cache (LFU)" << endl;
		cout << "6. Update Cache (LRU)" << endl;
		cout << "7. Update Cache (LFUSegmented)" << endl;
		cout << "8. Update Cache (LRUSegmented)" << endl;
		cout << "9. Update Cache (Segmented)" << endl;
		cout << "10. Dump Trace" << endl;
		cout << "11. Exit" << endl;
		cout << "cache. Cache Specific Column" << endl;
		cout << "clear. Delete Columns from GPU" << endl;
		cout << "custom. Toggle custom malloc" << endl;
		cout << "skipping. Toggle segment skipping" << endl;
		cout << "Your Input: ";
		cin >> input;

		if (input.compare("1") == 0) {
			time = 0; cpu_time_total = 0; gpu_time_total = 0; transfer_time_total = 0; malloc_time_total = 0;
			cgp->resetTime();
			cout << "Input Query: ";
			cin >> query;
			qp->setQuery(stoi(query));

			time2 = qp->processQuery2();
			cpu_time_total2 = cgp->cpu_time_total;
			gpu_time_total2 = cgp->gpu_time_total;
			transfer_time_total2 = cgp->transfer_time_total;
			malloc_time_total2 = cgp->malloc_time_total;
			cgp->resetTime();
			cout << endl;
			cout << endl;

			time1 = qp->processQuery();
			cpu_time_total1 = cgp->cpu_time_total;
			gpu_time_total1 = cgp->gpu_time_total;
			transfer_time_total1 = cgp->transfer_time_total;
			malloc_time_total1 = cgp->malloc_time_total;
			cgp->resetTime();
			cout << endl;
			cout << endl;

			if (time1 <= time2) {
				time += time1; cpu_time_total += cpu_time_total1; gpu_time_total += gpu_time_total1; transfer_time_total += transfer_time_total1; malloc_time_total += malloc_time_total1;
			} else {
				time += time2; cpu_time_total += cpu_time_total2; gpu_time_total += gpu_time_total2; transfer_time_total += transfer_time_total2; malloc_time_total += malloc_time_total2;
			}

			// time+=time1;
		} else if (input.compare("2") == 0) {
			time = 0; cpu_time_total = 0; gpu_time_total = 0; transfer_time_total = 0; malloc_time_total = 0;
			cgp->resetTime();
			cout << "Input Query: ";
			cin >> query;
			qp->setQuery(stoi(query));

			// time2 = qp->processHybridOnDemand(2);
			// cout << endl;
			// cout << endl;
			// time1 = qp->processHybridOnDemand(1);
			// cout << endl;
			// cout << endl;
			// if (time1 <= time2) time += time1;
			// else time += time2;

			time1 = qp->processQueryNP();
			time += time1;
			cpu_time_total += cgp->cpu_time_total;
			gpu_time_total += cgp->gpu_time_total;
			transfer_time_total += cgp->transfer_time_total;
			malloc_time_total += cgp->malloc_time_total;
			cgp->resetTime();
		} else if (input.compare("3") == 0) {
			time = 0; cpu_time_total = 0; gpu_time_total = 0; transfer_time_total = 0; malloc_time_total = 0;
			cout << "How many queries: ";
			cin >> many;
			many_query = stoi(many);
			cgp->resetTime();
			cgp->qo->processed_segment = 0;
			cgp->qo->skipped_segment = 0;
			cout << "Executing Random Query" << endl;
			for (int i = 0; i < many_query; i++) {
				qp->generate_rand_query();
				time1 = qp->processQuery();
				cpu_time_total1 = cgp->cpu_time_total;
				gpu_time_total1 = cgp->gpu_time_total;
				transfer_time_total1 = cgp->transfer_time_total;
				malloc_time_total1 = cgp->malloc_time_total;
				cgp->resetTime();

				time2 = qp->processQuery2();
				cpu_time_total2 = cgp->cpu_time_total;
				gpu_time_total2 = cgp->gpu_time_total;
				transfer_time_total2 = cgp->transfer_time_total;
				malloc_time_total2 = cgp->malloc_time_total;
				cgp->resetTime();

				if (time1 <= time2) {
					time += time1; cpu_time_total += cpu_time_total1; gpu_time_total += gpu_time_total1; transfer_time_total += transfer_time_total1; malloc_time_total += malloc_time_total1;
				} else {
					time += time2; cpu_time_total += cpu_time_total2; gpu_time_total += gpu_time_total2; transfer_time_total += transfer_time_total2; malloc_time_total += malloc_time_total2;
				}
				
			}
			processed_segment = cgp->qo->processed_segment;
			skipped_segment = cgp->qo->skipped_segment;
			srand(123);
		} else if (input.compare("4") == 0) {
			// time = 0; cpu_time_total = 0; gpu_time_total = 0; transfer_time_total = 0;
			// cout << "Executing Random Query" << endl;
			// for (int i = 0; i < 100; i++) {
			// 	qp->generate_rand_query();
			// 	time1 = qp->processHybridOnDemand(1);
			// 	time2 = qp->processHybridOnDemand(2);
			// 	if (time1 <= time2) time += time1;
			// 	else time += time2;
			// }
			// srand(123);
			time = 0; cpu_time_total = 0; gpu_time_total = 0; transfer_time_total = 0; malloc_time_total = 0;
			cout << "How many queries: ";
			cin >> many;
			many_query = stoi(many);
			cgp->resetTime();
			cgp->qo->processed_segment = 0;
			cgp->qo->skipped_segment = 0;
			cout << "Executing Random Query" << endl;
			for (int i = 0; i < many_query; i++) {
				qp->generate_rand_query();
				time1 = qp->processQueryNP();
				time += time1;
				cpu_time_total += cgp->cpu_time_total;
				gpu_time_total += cgp->gpu_time_total;
				transfer_time_total += cgp->transfer_time_total;
				malloc_time_total += cgp->malloc_time_total;
				cgp->resetTime();
			}
			processed_segment = cgp->qo->processed_segment;
			skipped_segment = cgp->qo->skipped_segment;
			srand(123);
		} else if (input.compare("5") == 0) {
			cout << "LFU Replacement" << endl;
			cgp->cm->runReplacement(LFU);
			qp->percentageData();
			srand(123);
		} else if (input.compare("6") == 0) {
			cout << "LRU Replacement" << endl;
			cgp->cm->runReplacement(LRU);
			qp->percentageData();
			srand(123);
		} else if (input.compare("7") == 0) {
			cout << "LFU Segmented Replacement" << endl;
			cgp->cm->runReplacement(LFUSegmented);
			qp->percentageData();
			srand(123);
		} else if (input.compare("8") == 0) {
			cout << "LRU Segmented Replacement" << endl;
			cgp->cm->runReplacement(LRUSegmented);
			qp->percentageData();
			srand(123);
		} else if (input.compare("9") == 0) {
			cout << "Segmented Replacement" << endl;
			cgp->cm->runReplacement(Segmented);
			qp->percentageData();
			srand(123);
		} else if (input.compare("10") == 0) {
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
		} else {
			exit = true;
		}

		cout << endl;
		cout << "Cumulated Time: " << time << endl;
		cout << "CPU time: " << cpu_time_total << endl;
		cout << "GPU time: " << gpu_time_total << endl;
		cout << "Transfer time: " << transfer_time_total << endl;
		cout << "Malloc time: " << malloc_time_total << endl;
		cout << "Fraction Skipped Segment: " << skipped_segment * 1.0 /(processed_segment + skipped_segment) << endl;
		cout << endl;

	}

}