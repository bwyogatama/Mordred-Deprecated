#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

int main() {

	CPUGPUProcessing* cgp = new CPUGPUProcessing(209715200, 209715200, 536870912, 536870912);

	bool exit = 0;
	string input;

	while (!exit) {
		cout << "Select Options:" << endl;
		cout << "1. Run Query 1.1" << endl;
		cout << "2. Run Query 2.1" << endl;
		cout << "3. Run Query 3.1" << endl;
		cout << "4. Run Query 4.1" << endl; 
		cout << "Your Input: ";
		cin >> input;

		if (input.compare("1") == 0) {
			cout << "Executing Query 1.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 0);
			qp->processOnDemand();
		} else if (input.compare("2") == 0) {
			cout << "Executing Query 2.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 1);
			qp->processOnDemand();
		} else if (input.compare("3") == 0) {
			cout << "Executing Query 3.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 2);
			qp->processOnDemand();
		} else if (input.compare("4") == 0) {
			cout << "Executing Query 4.1" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 3);
			qp->processOnDemand();
		} else {
			exit = true;
		}

		cout << endl;

	}

}