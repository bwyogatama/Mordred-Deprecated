#include "QueryProcessing.h"
#include "QueryOptimizer.h"
#include "CPUGPUProcessing.h"
#include "CacheManager.h"
#include "CPUProcessing.h"
#include "CostModel.h"

// using namespace std;

int main() {
	// CPUGPUProcessing* cgp = new CPUGPUProcessing(0, 0, 0, 0, 0, 0, 1, 1, 0);
	// QueryProcessing* qp = new QueryProcessing(cgp, 0);
	// QueryOptimizer* qo = qp->qo;
	// CacheManager* cm = qo->cm;

	Normal* norm = new Normal(35, 3, 0, 315);

	for (int i = 0; i < 50; i++) {
		norm->generateNorm();
		cout << norm->date.first << " " << norm->date.second << endl;
	}

	norm->reset(300, 3);

	for (int i = 0; i < 50; i++) {
		norm->generateNorm();
		cout << norm->date.first << " " << norm->date.second << endl;
	}

	return 0;
}