#include <chrono>
#include <atomic>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "tbb/tbb.h"

#include <cub/cub.cuh>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <thread>

#include "crystal/crystal.cuh"

using namespace cub;

using namespace std;
using namespace tbb;

void run(int* values, int size, int offset) {

    tbb::parallel_for( tbb::blocked_range<int>(offset, offset+size),
                       [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            values[i] = values[i] * values[i];
            printf("index = %d\n", i);
        }
    });


}

int main() {

	int* values = new int[3000];

	for (int i = 0; i < 3000; i++) {
		values[i] = i;
	}

	vector<thread> vec_th(3);

	for (int i = 0; i < 3; i++) {
		vec_th[i] = thread{run, values, 1000, i*1000};
	}

	for (int i = 0; i < 3; i++) {
		vec_th[i].join();
	}

	for (int i = 0; i < 3000; i++) {
		if (values[i] != i * i) printf("Error\n");
	}


	return 0;

}