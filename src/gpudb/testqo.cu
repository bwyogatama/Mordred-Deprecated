#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

//tbb::task_scheduler_init init(1);

int main() {

	CPUGPUProcessing* cgp = new CPUGPUProcessing(1000000000, 1000000000, 500000000);
	CacheManager* cm = cgp->cm;

	cm->cacheColumnSegmentInGPU(cm->lo_orderdate, 10);
	cm->cacheColumnSegmentInGPU(cm->lo_partkey, 20);
	cm->cacheColumnSegmentInGPU(cm->lo_custkey, 30);
	cm->cacheColumnSegmentInGPU(cm->lo_suppkey, 40);
	cm->cacheColumnSegmentInGPU(cm->lo_revenue, cm->lo_revenue->total_segment);
	cm->cacheColumnSegmentInGPU(cm->lo_supplycost, cm->lo_supplycost->total_segment);
	cm->cacheColumnSegmentInGPU(cm->lo_discount, 20);
	cm->cacheColumnSegmentInGPU(cm->lo_quantity, 40);
	cm->cacheColumnSegmentInGPU(cm->lo_extendedprice, cm->lo_extendedprice->total_segment);
	cm->cacheColumnSegmentInGPU(cm->d_datekey, cm->d_datekey->total_segment);
	cm->cacheColumnSegmentInGPU(cm->d_year, cm->d_year->total_segment);
	cm->cacheColumnSegmentInGPU(cm->p_partkey, cm->p_partkey->total_segment);
	cm->cacheColumnSegmentInGPU(cm->p_category, cm->p_category->total_segment);
	cm->cacheColumnSegmentInGPU(cm->p_brand1, cm->p_brand1->total_segment);
	cm->cacheColumnSegmentInGPU(cm->p_mfgr, cm->p_mfgr->total_segment);
	cm->cacheColumnSegmentInGPU(cm->c_custkey, cm->c_custkey->total_segment);
	cm->cacheColumnSegmentInGPU(cm->c_region, cm->c_region->total_segment);
	cm->cacheColumnSegmentInGPU(cm->c_nation, cm->c_nation->total_segment);
	cm->cacheColumnSegmentInGPU(cm->s_suppkey, cm->s_suppkey->total_segment);
	cm->cacheColumnSegmentInGPU(cm->s_region, cm->s_region->total_segment);
	cm->cacheColumnSegmentInGPU(cm->s_nation, cm->s_nation->total_segment);

	QueryProcessing* qp;
	qp = new QueryProcessing(cgp, 0);
	for (int i = 0; i < 10; i++) {
		cout << i << endl;
		qp->processQuery();
	}
	qp = new QueryProcessing(cgp, 1);
	for (int i = 0; i < 10; i++) {
		cout << i << endl;
		qp->processQuery();
	}
	qp = new QueryProcessing(cgp, 2);
	for (int i = 0; i < 10; i++) {
		cout << i << endl;
		qp->processQuery();
	}
	qp = new QueryProcessing(cgp, 3);
	for (int i = 0; i < 10; i++) {
		cout << i << endl;
		qp->processQuery();
	}

	// qp = new QueryProcessing(cgp, 1);
	// qp->processQuery();
	// qp->processQuery();
	// qp = new QueryProcessing(cgp, 2);
	// qp->processQuery();
	// qp->processQuery();
	// qp = new QueryProcessing(cgp, 3);
	// qp->processQuery();
	// qp->processQuery();

	// bool exit = 0;
	// string input;

	// while (!exit) {
	// 	cout << "Select Options:" << endl;
	// 	cout << "1. Run Query 1.1" << endl;
	// 	cout << "2. Run Query 2.1" << endl;
	// 	cout << "3. Run Query 3.1" << endl;
	// 	cout << "4. Run Query 4.1" << endl;
	// 	cout << "5. Update Cache (LFU)" << endl;
	// 	cout << "6. Update Cache (LRU)" << endl;
	// 	cout << "7. Exit" << endl;
	// 	cout << "Your Input: ";
	// 	cin >> input;

	// 	if (input.compare("1") == 0) {
	// 		cout << "Executing Query 1.1" << endl;
	// 		QueryProcessing* qp = new QueryProcessing(cgp, 0);
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 	} else if (input.compare("2") == 0) {
	// 		cout << "Executing Query 2.1" << endl;
	// 		QueryProcessing* qp = new QueryProcessing(cgp, 1);
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 	} else if (input.compare("3") == 0) {
	// 		cout << "Executing Query 3.1" << endl;
	// 		QueryProcessing* qp = new QueryProcessing(cgp, 2);
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 	} else if (input.compare("4") == 0) {
	// 		cout << "Executing Query 4.1" << endl;
	// 		QueryProcessing* qp = new QueryProcessing(cgp, 3);
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 		qp->processQuery();
	// 	} else if (input.compare("5") == 0) {
	// 		cout << "LFU Replacement" << endl;
	// 		cgp->cm->runReplacement(0);
	// 	} else if (input.compare("6") == 0) {
	// 		cout << "LRU Replacement" << endl;
	// 		cgp->cm->runReplacement(1);
	// 	} else if (input.compare("7") == 0) {
	// 		exit = true;
	// 	} else {
	// 		exit = true;
	// 	}

	// 	cout << endl;

	// }

}