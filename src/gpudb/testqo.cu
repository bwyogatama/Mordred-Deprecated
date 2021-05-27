#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

int main() {

	QueryOptimizer* qo = new QueryOptimizer();

	qo->cm->cacheColumnSegmentInGPU(qo->cm->lo_orderdate, 6000);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->lo_partkey, 4000);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->lo_suppkey, 2000);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->lo_revenue, 0);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->d_datekey, 3);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->d_year, 3);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->p_partkey, 200);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->p_category, 200);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->p_brand1, 200);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->s_suppkey, 2);
	qo->cm->cacheColumnSegmentInGPU(qo->cm->s_region, 2);

	qo->parseQuery(1);

	printf("hi\n");

	// for (int i = 0; i < qo->joinCPUPipelineCol.size(); i++) {
	// 	for (int j = 0; j < qo->joinCPUPipelineCol[i].size(); j++) {
	// 		if (qo->joinCPUPipelineCol[i][j] != NULL) {
	// 			cout << qo->joinCPUPipelineCol[i][j]->column_name << endl;
	// 		}
	// 	}
	// }

	for (int j = 0; j < 64; j++) {
		if (qo->segment_group_count[0][j] > 0) {
			printf("%d %d\n", j, qo->segment_group_count[0][j]);
		}
	}

}