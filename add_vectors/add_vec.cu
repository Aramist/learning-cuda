#include <stdlib.h>
#include <stdio.h>

#define ERRCHK(err) \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(EXIT_FAILURE); \
	}

__global__ void kernel_vecAdd(float *A, float *B, float *C, size_t n);


int main(void) {
  
  float *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  cudaError_t err;

  int numThreadsPerBlock = 64;

  int vecLength = 4;
  size_t size = vecLength*sizeof(float);

  A_h = (float *) malloc(size);
  B_h = (float *) malloc(size);
  C_h = (float *) malloc(size);

  err = cudaMalloc( (void**)&A_d, size);
  ERRCHK(err);
  err = cudaMalloc( (void**)&B_d, size);
  ERRCHK(err);
  err = cudaMalloc( (void**)&C_d, size);
  ERRCHK(err);


  for(int i = 0; i < vecLength; i++) {
    printf("A[%d]: ", i);
    fscanf(stdin, "%f", A_h + i);
  }
  for(int i = 0; i < vecLength; i++) {
    printf("B[%d]: ", i);
    fscanf(stdin, "%f", B_h + i);
  }

  
  err = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  ERRCHK(err);
  err = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
  ERRCHK(err);
  free(A_h);
  free(B_h);


  kernel_vecAdd<<<(int)ceil(vecLength / (float)numThreadsPerBlock), numThreadsPerBlock>>>(A_d, B_d, C_d, vecLength);

  err = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
  ERRCHK(err);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  printf("C = ");
  for(int i = 0; i < vecLength; i++) {
    printf("%.2f, ", C_h[i]);
  }
  printf("\b\b\n");

  free(C_h);
}

__global__ void kernel_vecAdd(float *A, float *B, float *C, size_t n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Grids may contain more threads than indices in the vector
  // they are spawned to process
  if (i < n)  {
    C[i] = A[i] + B[i];
  }
}
