#include <CacheManager.h>

#define LO_LEN 6000000

int main () {

	CacheManager* cache_manager = new CacheManager(2000000000, 3);

	int *h_lo_orderdate = cache_manager->loadColumnToCPU("lo_orderdate", LO_LEN);
	int *h_lo_suppkey = cache_manager->loadColumnToCPU("lo_suppkey", LO_LEN);
	int *h_lo_partkey = cache_manager->loadColumnToCPU("lo_partkey", LO_LEN);

	ColumnInfo* orderdate = new ColumnInfo("lo_orderdate", "lo", LO_LEN, 0, h_lo_orderdate);
	ColumnInfo* suppkey = new ColumnInfo("lo_suppkey", "lo", LO_LEN, 1, h_lo_suppkey);
	ColumnInfo* partkey = new ColumnInfo("lo_partkey", "lo", LO_LEN, 2, h_lo_partkey);
	
	cache_manager->allColumn[0] = orderdate;
	cache_manager->allColumn[1] = suppkey;
	cache_manager->allColumn[2] = partkey;

	cache_manager->cacheColumnSegmentInGPU(orderdate, 5);

	cache_manager->deleteColumnSegmentInGPU(orderdate, 4);

	Segment* seg = suppkey->getSegment(4);

	cache_manager->cacheSegmentInGPU(seg);

	delete cache_manager;

	return 0;
}