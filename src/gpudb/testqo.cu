#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

int main() {

	QueryProcessing* qp = new QueryProcessing();

	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_orderdate, 6002);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_partkey, 6002);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_custkey, 6002);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_suppkey, 6002);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_revenue, 6002);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_supplycost, 6002);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_discount, 4000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_quantity, 1000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_extendedprice, 5000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->d_datekey, 3);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->d_year, 3);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_partkey, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_category, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_brand1, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_mfgr, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->c_custkey, 30);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->c_region, 30);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->c_nation, 30);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->s_suppkey, 2);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->s_region, 2);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->s_nation, 2);

	printf("LOADED\n");


	// qp->qo->parseQuery(1);

	// printf("hi\n");

	// for (int i = 0; i < qo->joinCPUPipelineCol.size(); i++) {
	// 	for (int j = 0; j < qo->joinCPUPipelineCol[i].size(); j++) {
	// 		if (qo->joinCPUPipelineCol[i][j] != NULL) {
	// 			cout << qo->joinCPUPipelineCol[i][j]->column_name << endl;
	// 		}
	// 	}
	// }

	qp->processQuery();

	// for (int j = 0; j < 64; j++) {
	// 	if (qo->segment_group_count[0][j] > 0) {
	// 		printf("%d %d\n", j, qo->segment_group_count[0][j]);
	// 	}
	// }

}