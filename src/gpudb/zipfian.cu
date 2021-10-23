#include "QueryProcessing.cuh"


int main() {
	CPUGPUProcessing* cgp = new CPUGPUProcessing(0, 0, 536870912, 536870912, 0);
	QueryProcessing* qp = new QueryProcessing(cgp, 0);
	QueryOptimizer* qo = qp->qo;
	CacheManager* cm = qo->cm;


	qo->zipfian[11] = new Zipfian (7, 0);
	qo->zipfian[12] = new Zipfian (79, 0);
	qo->zipfian[13] = new Zipfian (316, 0);
	qo->zipfian[21] = new Zipfian (7, 0);
	qo->zipfian[22] = new Zipfian (7, 0);
	qo->zipfian[23] = new Zipfian (7, 0);
	qo->zipfian[31] = new Zipfian (7, 0);
	qo->zipfian[32] = new Zipfian (7, 0);
	qo->zipfian[33] = new Zipfian (7, 0);
	qo->zipfian[34] = new Zipfian (79, 0);
	qo->zipfian[41] = new Zipfian (7, 0);
	qo->zipfian[42] = new Zipfian (6, 1);
	qo->zipfian[43] = new Zipfian (6, 1);

	
	// qo->speedup[query][cm->lo_orderdate] = cm->lo_orderdate->total_segment;

	srand(123);

	for (int i = 0; i < 100; i++) {
		// qo->zipfian[query]->generateZipf();
		// cout << qo->zipfian[query]->date.first << " " << qo->zipfian[query]->date.second << endl;
		qp->generate_rand_query();
		int query = qp->query;

		qo->speedup[query][cm->lo_orderdate] = cm->lo_orderdate->total_segment;

		qo->parseQuery(query);
		qo->prepareQuery(query, 1);
		qo->prepareOperatorPlacement();
		qo->groupBitmapSegmentTable(0, query);

		qo->clearPlacement();
		qp->endQuery();
		qo->clearParsing();
	}

	for (int j = 0; j < cm->lo_orderdate->total_segment; j++) {
		Segment* segment = cm->index_to_segment[cm->lo_orderdate->column_id][j];
		cout << cm->lo_orderdate->column_name << " " << segment->weight << " " << cm->segment_min[cm->lo_orderdate->column_id][j] << " " << cm->segment_max[cm->lo_orderdate->column_id][j] << endl;
	}

	return 0;
}