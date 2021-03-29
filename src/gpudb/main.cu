#include "QueryOptimizer.h"

int main () {

	//CacheManager* cm = new CacheManager(1000000000, 25);

	QueryOptimizer* qo = new QueryOptimizer();

	qo->parseQuery(2);

	//cm->cacheColumnSegmentInGPU(cm->lo_orderkey, 5);

	//cm->deleteColumnSegmentInGPU(cm->lo_orderkey, 4);

	//delete cm;

	return 0;
}