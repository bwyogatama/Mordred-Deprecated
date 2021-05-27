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
    ptr_data = (int**) malloc(5 * sizeof(int*));
    cudaMalloc((void **)&data, sizeof(int)*LEN*5);
    cudaMemset((void *)data, 0, sizeof(int)*LEN*5);
    for (int i = 0; i < 5; i++) {
      ptr_data[i] = data + LEN * i;
    }
  };
  
  ~MyClass() {
    free(ptr_data);
    cudaFree((void *)data);
  };
  
  void run(int *b, int i) {
    dim3 grid(1);
    dim3 block(LEN);
    kernel<<<grid, block>>>(ptr_data[i], b, LEN);
  };
  
  int *get(int i) {
    return ptr_data[i];
  };
private:
  int *data;
  int **ptr_data;
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
  int *b_gpu, b_host[LEN * 5];
  int **ptr;
  
  for (int i=0; i<LEN * 5; i++) {
    b_host[i] = i;
  }

  cudaMalloc((void **)&b_gpu, sizeof(int)*LEN*5);
  cudaMemcpy(b_gpu, b_host, sizeof(int)*LEN*5, cudaMemcpyHostToDevice);
  ptr = (int**) malloc(5 * sizeof(int*));
  for (int i = 0; i < 5 ; i++) {
    ptr[i] = b_gpu + i * LEN;
  }

  for (int i = 0; i < 5; i++) {
    c.run(ptr[i], i);
    cudaMemcpy(b_host + i * LEN, c.get(i), sizeof(int)*LEN, cudaMemcpyDeviceToHost);
  }
  
  cudaFree(b_gpu);
  free(ptr);

  show(b_host, LEN * 5);
}