// How to "wrap" a CUDA kernel with a C++ class; the kernel must be defined outside of
// the class and launched from within a class instance's method.

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define LEN 10

__global__ void kernel(int *a, int *b, unsigned int N);

class MyClass {
public:
  MyClass() {
    cudaMalloc((void **)&data, sizeof(int)*LEN);
    cudaMemset((void *)data, 0, sizeof(int)*LEN);
  };
  
  ~MyClass() {
    cudaFree((void *)data);
  };
  
  void run(int *b) {
    dim3 grid(1);
    dim3 block(LEN);
    kernel<<<grid, block>>>(data, b, LEN);
  };
  
  int *get(void) {
    return data;
  };
private:
  int *data;
};

__global__ void kernel(int *a, int *b, unsigned int N) {
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    a[i] += b[i];
  }
}

void show(int *data, unsigned int N) {
  for (int i=0; i<N; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}
  
int main(void) {
  MyClass c;
  int *b_gpu, b_host[LEN];
  
  for (int i=0; i<LEN; i++) {
    b_host[i] = i;
  }

  cudaMalloc((void **)&b_gpu, sizeof(int)*LEN);
  cudaMemcpy(b_gpu, b_host, sizeof(int)*LEN, cudaMemcpyHostToDevice);
  c.run(b_gpu);
  cudaMemcpy(b_host, c.get(), sizeof(int)*LEN, cudaMemcpyDeviceToHost);
  cudaFree(b_gpu);

  show(b_host, LEN);
}