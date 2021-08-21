#include "QueryProcessing.h"

#include <chrono>
#include <atomic>

int main() {

	bool verbose = 0;

	srand(123);

	CPUGPUProcessing* cgp = new CPUGPUProcessing(209715200, 209715200, 536870912, 536870912, verbose);

	bool exit = 0;
	float time = 0;
	string input;
	string query;

	while (!exit) {
		cout << "Select Options:" << endl;
		cout << "1. Run Specific Query" << endl;
		cout << "2. Run Random Queries" << endl;
		cout << "3. Exit" << endl;
		cout << "Your Input: ";
		cin >> input;

		if (input.compare("1") == 0) {
			cout << "Input Query: ";
			cin >> query;
			QueryProcessing* qp = new QueryProcessing(cgp, stoi(query), verbose);
			time += qp->processOnDemand();
		} else if (input.compare("2") == 0) {
			time = 0;
			cout << "Executing Random Query" << endl;
			QueryProcessing* qp = new QueryProcessing(cgp, 11, verbose);
			for (int i = 0; i < 100; i++) {
				qp->generate_rand_query();
				time += qp->processOnDemand();
			}
			delete qp;
			srand(123);
		} else {
			exit = true;
		}

		cout << endl;
		cout << "Cumulated Time: " << time << endl;
		cout << endl;

	}

}