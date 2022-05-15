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

__global__ void kernel2D(int** d_A, int row, int cols) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < cols) {
		d_A[row][tid] = d_A[row][tid] * d_A[row][tid];
	}
}

void runGPU2D(int** d_A, int row, int cols) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    kernel2D<<<(cols + 128 - 1)/128, 128, 0, stream>>>(d_A, row, cols); 

    cudaStreamDestroy(stream);
}

__global__ void kernel1D(int* d_A, int row, int cols) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < cols) {
		d_A[row * cols + tid] = d_A[row * cols + tid] * d_A[row * cols + tid];
	}
}

void runGPU1D(int* d_A, int row, int cols) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    kernel1D<<<(cols + 128 - 1)/128, 128, 0, stream>>>(d_A, row, cols); 

    cudaStreamDestroy(stream);
}

int main() {

	int rows = 64;
	int cols = 1024 * 1024;

	int** A = new int*[rows];
	A[0] = new int[rows * cols];
	for (int i = 1; i < rows; ++i) A[i] = A[i-1] + cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			A[i][j] = i*cols+j;
		}
	}

	int** temp = new int*[rows];
	cudaMalloc((void**) &(temp[0]), rows * cols * sizeof(int));
	for (int i = 1; i < rows; ++i) temp[i] = temp[i-1] + cols;
	cudaMemcpy(temp[0], A[0], rows * cols * sizeof(int), cudaMemcpyHostToDevice);

	int** d_A;
	cudaMalloc((void**) &(d_A), rows * sizeof(int*));
	cudaMemcpy(d_A, temp, rows * sizeof(int*), cudaMemcpyHostToDevice);


	cudaEvent_t start, stop; 
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0);

	parallel_for(int(0), rows, [=](int i){
		runGPU2D(d_A, i, cols);
	});

	// parallel_for(int(0), rows, [=](int i){
	// 	runGPU1D(temp[0], i, cols);
	// });

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cout << "Time Taken Total: " << time << endl;


	int** B = new int*[rows];
	B[0] = new int[rows * cols];	
	for (int i = 1; i < rows; ++i) B[i] = B[i-1] + cols;
	cudaMemcpy(B[0], temp[0], rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			assert(B[i][j] == A[i][j] * A[i][j]);
		}
	}	

	return 0;

}