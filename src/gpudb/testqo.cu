#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

int main() {

	QueryProcessing* qp = new QueryProcessing();

	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_orderdate, 6000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_partkey, 4000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_suppkey, 2000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->lo_revenue, 6000);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->d_datekey, 3);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->d_year, 3);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_partkey, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_category, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->p_brand1, 200);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->s_suppkey, 2);
	qp->cm->cacheColumnSegmentInGPU(qp->cm->s_region, 2);

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