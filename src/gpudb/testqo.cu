#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

//tbb::task_scheduler_init init(1);

int main() {

	CPUGPUProcessing* cgp = new CPUGPUProcessing(1000000000, 1000000000);

	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_orderdate, 6);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_partkey, qp->cm->lo_partkey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_custkey, qp->cm->lo_custkey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_suppkey, qp->cm->lo_suppkey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_revenue, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_supplycost, 6);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_discount, 2);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_quantity, 2);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_extendedprice, 4);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->d_datekey, qp->cm->d_datekey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->d_year, qp->cm->d_year->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_partkey, qp->cm->p_partkey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_category, qp->cm->p_category->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_brand1, qp->cm->p_brand1->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_mfgr, qp->cm->p_mfgr->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->c_custkey, qp->cm->c_custkey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->c_region, qp->cm->c_region->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->c_nation, qp->cm->c_nation->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->s_suppkey, qp->cm->s_suppkey->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->s_region, qp->cm->s_region->total_segment);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->s_nation, qp->cm->s_nation->total_segment);

	// qp->processQuery(0);
	// qp->processQuery(1);
	// qp->processQuery(2);
	// qp->processQuery(3);

	bool exit = 0;
	string input;

	while (!exit) {
		cout << "Select Options:" << endl;
		cout << "1. Run Query 1.1" << endl;
		cout << "2. Run Query 2.1" << endl;
		cout << "3. Run Query 3.1" << endl;
		cout << "4. Run Query 4.1" << endl;
		cout << "5. Update Cache (LFU)" << endl;
		cout << "6. Update Cache (LRU)" << endl;
		cout << "7. Exit" << endl;
		cout << "Your Input: ";
		cin >> input;

		if (input.compare("1") == 0) {
			cout << "Executing Query 1.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 0);
			qp->processQuery();
			qp->processQuery();
			qp->processQuery();
		} else if (input.compare("2") == 0) {
			cout << "Executing Query 2.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 1);
			qp->processQuery();
			qp->processQuery();
			qp->processQuery();
		} else if (input.compare("3") == 0) {
			cout << "Executing Query 3.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 2);
			qp->processQuery();
			qp->processQuery();
			qp->processQuery();
		} else if (input.compare("4") == 0) {
			cout << "Executing Query 4.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 3);
			qp->processQuery();
			qp->processQuery();
			qp->processQuery();
		} else if (input.compare("5") == 0) {
			cout << "LFU Replacement" << endl;
			cgp->cm->runReplacement(0);
		} else if (input.compare("6") == 0) {
			cout << "LRU Replacement" << endl;
			cgp->cm->runReplacement(1);
		} else if (input.compare("7") == 0) {
			exit = true;
		} else {
			exit = true;
		}

		cout << endl;

	}

}