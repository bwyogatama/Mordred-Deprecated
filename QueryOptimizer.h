#ifndef _QUERY_OPTIMIZER_H_
#define _QUERY_OPTIMIZER_H_

#include "CacheManager.h"

class QueryOptimizer {
	vector<ColumnInfo*> join;
	vector<ColumnInfo*> groupby;
	vector<ColumnInfo*> select;
	QueryOptimizer();
	int generate_rand_query();
	void parseQuery11();
	void parseQuery21();
	void parseQuery31();
	void parseQuery41();
};

QueryOptimizer::QueryOptimizer() {
	
}

int 
QueryOptimizer::generate_rand_query() {
	return rand() % 4;
}

void 
QueryOptimizer::parseQuery(int query) {
	if (query == 0) parseQuery11();
	else if (query == 1) parseQuery21();
	else if (query == 2) parseQuery31();
	else parseQuery41();
}

void 
QueryOptimizer::parseQuery11() {
	//parse selection

}

void 
QueryOptimizer::parseQuery21() {

}

#endif