#include <assert.h>
#include <stdio.h>
#include <chrono>

#include "tbb/tbb.h"

#include <cub/cub.cuh>
#include <curand.h>
#include <cuda.h>
#include <thread>

#include "crystal/crystal.cuh"

using namespace cub;
using namespace std;
using namespace tbb;

void runCPU(int* values, int size, int offset) {

    parallel_for( blocked_range<int>(offset, offset+size),
                       [&](blocked_range<int> r)
    {
    	// int worker_index = tbb::task_arena::current_thread_index();
    	//printf("worker_index = %d\n", worker_index);
        for (int i=r.begin(); i<r.end(); ++i)
        {
            values[i] = values[i] * values[i];
            //printf("index = %d\n", i);
        }
    });
}

__global__ void kernel(int* d_values, int size, int offset) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < size) {
		//printf("%d %d\n", tid + offset, d_values[tid + offset]);
		d_values[tid + offset] = d_values[tid + offset] * d_values[tid + offset];
		//if ((tid + offset) == 160) printf("%d\n", d_values[tid + offset]);
	}
}

void runGPU(int* d_values, int size, int offset) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //cout << offset << endl;

    kernel<<<(size + 128 - 1)/128, 128, 0, stream>>>(d_values, size, offset); 

    cudaStreamDestroy(stream);
}

int main() {

	int* values = new int[64000];
	int* h_values = new int[64000];

	int* d_values;

	for (int i = 0; i < 64000; i++) {
		values[i] = i;
	}

    cudaMalloc(&(d_values), 64000 * sizeof(int));
    cudaMemcpy(d_values, values, 64000 * sizeof(int), cudaMemcpyHostToDevice);

	// vector<thread> vec_th(3);

	// for (int i = 0; i < 3; i++) {
	// 	vec_th[i] = thread{run, values, 1000, i*1000};
	// }

	// for (int i = 0; i < 3; i++) {
	// 	vec_th[i].join();
	// }

	// for (int i = 0; i < 3000; i++) {
	// 	if (values[i] != i * i) printf("Error\n");
	// }

	cudaEvent_t start, stop; 
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0);

	parallel_for(int(0), 64, [=](int i){
		//runCPU(values, 1000, i*1000);
		runGPU(d_values, 1000, i*1000);
	});

	// for (int i = 0; i < 64; i++) {
	// 	//runCPU(values, 1000, i*1000);
	// 	runGPU(d_values, 1000, i*1000);
	// }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cout << "Time Taken Total: " << time << endl;

	cudaMemcpy(h_values, d_values, 64000 * sizeof(int), cudaMemcpyDeviceToHost);

	//printf("test\n");

	// for (int i = 0; i < 64000; i++) {
		//assert(h_values[i] == values[i]);
	// }

	return 0;

}