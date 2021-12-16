#include "QueryProcessing.h"
#include "QueryOptimizer.h"
#include "CPUGPUProcessing.h"
#include "CacheManager.h"
#include "CPUProcessing.h"
#include "CostModel.h"

int main() {

	bool verbose = 1;

	srand(123);
	
	CPUGPUProcessing* cgp = new CPUGPUProcessing(52428800 * 10, 52428800 * 30, 52428800 * 18, 52428800 * 20, verbose);
	QueryProcessing* qp;

	// cout << "Profiling" << endl;
	// qp = new QueryProcessing(cgp, verbose);
	// qp->profile();
	// delete qp;

	// cgp->cm->resetCache(52428800 * 4, 209715200 / 2, 536870912, 536870912);
	// // cgp->cm->resetCache(314572800, 209715200 / 2, 536870912, 536870912);
	// // cgp->cm->resetCache(629145600, 209715200 / 2, 536870912, 536870912);

	cout << endl;

	// CacheManager* cm = cgp->cm;

	// cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 58);
	// cm->cacheColumnSegmentInGPU(cm->lo_suppkey, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_custkey, 58);
	// cm->cacheColumnSegmentInGPU(cm->lo_partkey, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_revenue, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_supplycost, 0);
	// cm->cacheColumnSegmentInGPU(cm->lo_discount, 58);
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
	float time = 0;
	string input;
	string query;
	bool skew = false;

	qp = new QueryProcessing(cgp, verbose, skew);

	while (!exit) {
		cout << "Select Options:" << endl;
		cout << "1. Run Specific Query" << endl;
		cout << "2. Run Specific Query HOD" << endl;
		cout << "3. Run Random Queries" << endl;
		cout << "4. Run Random Queries HOD" << endl;
		cout << "5. Exit" << endl;
		cout << "Your Input: ";
		cin >> input;

		if (input.compare("1") == 0) {
			// cout << "Input Query: ";
			// cin >> query;
			// qp->setQuery(stoi(query));
			// time += qp->processOnDemand();
			cout << "Input Query: ";
			cin >> query;
			qp->setQuery(stoi(query));
			// qp->processQuery();
			time = qp->processOnDemand();
		} else if (input.compare("2") == 0) {
			cout << "Input Query: ";
			cin >> query;
			qp->setQuery(stoi(query));
			qp->processHybridOnDemand();
			time = qp->processHybridOnDemand();
		} else if (input.compare("3") == 0) {
			time = 0;
			cout << "Executing Random Query" << endl;
			for (int i = 0; i < 100; i++) {
				qp->generate_rand_query();
				time += qp->processOnDemand();
			}
			// time = 0;
			// cout << "Executing Random Query" << endl;
			// for (int i = 0; i < 100; i++) {
			// 	qp->generate_rand_query();
			// 	qp->processQuery();
			// 	time += qp->processQuery();
			// }
			srand(123);
		} else if (input.compare("4") == 0) {
			time = 0;
			cout << "Executing Random Query" << endl;
			for (int i = 0; i < 100; i++) {
				qp->generate_rand_query();
				time += qp->processHybridOnDemand();
			}
			srand(123);
		} else {
			exit = true;
		}

		cout << endl;
		cout << "Cumulated Time: " << time << endl;
		cout << endl;

	}

}