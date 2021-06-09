#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

int main() {

	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_orderdate, 7);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_partkey, 6);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_custkey, 4);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_suppkey, 3);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_revenue, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_supplycost, 7);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_discount, 2);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_quantity, 2);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_extendedprice, 4);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->d_datekey, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->d_year, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_partkey, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_category, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_brand1, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->p_mfgr, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->c_custkey, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->c_region, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->c_nation, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->s_suppkey, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->s_region, 1);
	// qp->cm->cacheColumnSegmentInGPU(qp->cm->s_nation, 1);

	// qp->qo->parseQuery(1);

	// printf("hi\n");

	// for (int i = 0; i < qo->joinCPUPipelineCol.size(); i++) {
	// 	for (int j = 0; j < qo->joinCPUPipelineCol[i].size(); j++) {
	// 		if (qo->joinCPUPipelineCol[i][j] != NULL) {
	// 			cout << qo->joinCPUPipelineCol[i][j]->column_name << endl;
	// 		}
	// 	}
	// }

	// qp->processQuery(0);
	// qp->processQuery(1);
	// qp->processQuery(2);
	// qp->processQuery(3);

	QueryProcessing* qp = new QueryProcessing();

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
			qp->processQuery(0);
		} else if (input.compare("2") == 0) {
			cout << "Executing Query 2.1" << endl;
			qp->processQuery(1);
		} else if (input.compare("3") == 0) {
			cout << "Executing Query 3.1" << endl;
			qp->processQuery(2);
		} else if (input.compare("4") == 0) {
			cout << "Executing Query 4.1" << endl;
			qp->processQuery(3);
		} else if (input.compare("5") == 0) {
			cout << "LFU Replacement" << endl;
			qp->cm->runReplacement(0);
		} else if (input.compare("6") == 0) {
			cout << "LRU Replacement" << endl;
			qp->cm->runReplacement(1);
		} else if (input.compare("7") == 0) {
			exit = true;
		}

		cout << endl;

	}

}