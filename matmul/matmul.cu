#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

extern "C"{
#include "rand_lib.h"
#define NPY_LIB_IMPL
#include "npy_lib.h"
}

#define CUDA_CHK_ERR(err) if (err != cudaSuccess) { \
    fprintf(stderr, "%s (%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    return; \
  }

#define RETURN_FAIL 0
#define RETURN_SUCCESS 1
#define BLOCK_SIZE 32

struct matrix {
  size_t m, n;
  float *data;
};


double difftimespec(struct timespec *a, struct timespec *b);
int alloc_matrix(struct matrix **mat, size_t m, size_t n, int init);
void free_matrix(struct matrix *mat);
dim3 computeGridDim(size_t m, size_t k, size_t n);
void matmul(struct matrix *left, struct matrix *right, struct matrix *result);
__global__ void matmul_k(float *data_l, float *data_r, float *out, size_t m, size_t k, size_t n, size_t tile_len);

int main(void) {
  size_t m, k, n;
  size_t l_shape[2], r_shape[2], result_shape[2];
  struct matrix *left, *right, *result;

  fprintf(stdout, "Matrix 1 height: ");
  fscanf(stdin, "%lu", &m);
  fprintf(stdout, "Matrix 1 width: ");
  fscanf(stdin, "%lu", &k);
  fprintf(stdout, "Matrix 2 width: ");
  fscanf(stdin, "%lu", &n);
  
  if (alloc_matrix(&left, m, k, true) == RETURN_FAIL) {
    fprintf(stderr, "Failed to allocate matrix of shape (%lu, %lu)\n", m, k);
    return EXIT_FAILURE;
  }
  if (alloc_matrix(&right, k, n, true) == RETURN_FAIL) {
    fprintf(stderr, "Failed to allocate matrix of shape (%lu, %lu)\n", k, n);
    return EXIT_FAILURE;
  }
  if (alloc_matrix(&result, m, n, false) == RETURN_FAIL) {
    fprintf(stderr, "Failed to allocate matrix of shape (%lu, %lu)\n", m, n);
    return EXIT_FAILURE;
  }

  fprintf(stdout, "Saving initial matrices to L.npy and R.npy\n");
  l_shape[0] = m; l_shape[1] = k;
  r_shape[0] = k; r_shape[1] = n;
  result_shape[0] = m; result_shape[1] = n;
  save_float_arr("L.npy", left->data, l_shape, 2);
  save_float_arr("R.npy", right->data, r_shape, 2);

  matmul(left, right, result);

  fprintf(stdout, "Saving multiplication result to result.npy\n");
  save_float_arr("result.npy", result->data, result_shape, 2);

  free_matrix(left);
  free_matrix(right);
}

double difftimespec(struct timespec *a, struct timespec *b) {
  return (a->tv_sec - b->tv_sec) + 1e-9 * (a->tv_nsec - b->tv_nsec);
}


int alloc_matrix(struct matrix **mat, size_t m, size_t n, int init) {
  *mat = (struct matrix *) malloc(sizeof(struct matrix));
  if (*mat == NULL) {
    return RETURN_FAIL;
  }

  (*mat)->data = (float *) malloc (sizeof(float) * m * n);
  (*mat)->m = m;
  (*mat)->n = n;

  if(init)
    rand_matrix((*mat)->data, m, n);
  return RETURN_SUCCESS;
}

void free_matrix(struct matrix *mat) {
  free(mat->data);
  free(mat);
}

dim3 computeGridDim(size_t m, size_t k, size_t n) {
  // Need enough blocks to cover the full output size
  size_t yd, xd;

  yd = (size_t) ceil(m / (float)BLOCK_SIZE);
  xd = (size_t) ceil(n / (float)BLOCK_SIZE);
  return dim3(xd, yd, 1);
}

void matmul(struct matrix *left, struct matrix *right, struct matrix *result){
  cudaError_t err;
  size_t num_shared;
  dim3 blockDim(32, 32);
  dim3 gridDim;
  float *left_d, *right_d, *result_d;
  struct timespec start_time, end_time, start_copy, end_copy;
  double copy_time=0.0;

  clock_gettime(CLOCK_MONOTONIC, &start_copy);
  err = cudaMalloc(&left_d, sizeof(float) * left->m * left->n);
  CUDA_CHK_ERR(err);
  err = cudaMalloc(&right_d, sizeof(float) * right->m * right->n);
  CUDA_CHK_ERR(err);
  err = cudaMalloc(&result_d, sizeof(float) * left->m * right->n);
  CUDA_CHK_ERR(err);

  // copy mem
  err = cudaMemcpy(left_d, left->data, sizeof(float) * left->m * left->n, cudaMemcpyHostToDevice);
  CUDA_CHK_ERR(err);
  err = cudaMemcpy(right_d, right->data, sizeof(float) * right->m * right->n, cudaMemcpyHostToDevice);
  CUDA_CHK_ERR(err);
  clock_gettime(CLOCK_MONOTONIC, &end_copy);
  copy_time += difftimespec(&end_copy, &start_copy);

  // Need space for tiles of both lhs and rhs matrices
  num_shared = BLOCK_SIZE * BLOCK_SIZE * sizeof(float) * 2;
  // Compute grid dimension
  gridDim = computeGridDim(left->m, left->n, right->n);

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  matmul_k<<<gridDim, blockDim, num_shared>>>(
        left_d, right_d, result_d,
        left->m, left->n, right->n,
        BLOCK_SIZE
  );

  // Move data back to host
  err = cudaMemcpy(result->data, result_d, sizeof(float) * result->m * result->n, cudaMemcpyDeviceToHost);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  fprintf(stdout, "Elapsed time: %.8fs\n", difftimespec(&end_time, &start_time));

  fprintf(stdout, "Elapsed time (copies): %.1fs\n", copy_time);

  CUDA_CHK_ERR(err);
  CUDA_CHK_ERR(cudaFree(left_d));
  cudaFree(right_d);
  cudaFree(result_d);
}

__global__ void matmul_k(float *data_l, float *data_r, float *out, size_t m, size_t k, size_t n, size_t tile_len){
  /* Performs a tiled matrix multiplication
   * goal: compute M_ij = Sum_k L_ik * N_kj
   * i and j given by block and threod idx
   * Each block uses shared memory for M and N
   */
   
  size_t mtile_size;
  size_t ls_i, ls_j, rs_i, rs_j, os_i, os_j; // strides
  extern __shared__ float mn_tiles[];
  size_t result_idx;
  size_t result_i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t result_j = blockIdx.x * blockDim.x + threadIdx.x;
  // allow some threads to quit early. Shouldn't affect sync threads?

  mtile_size = tile_len * tile_len; // Offset to get to the start of the N tile
  ls_i = k;  // L is (m, k)
  ls_j = 1;
  rs_i = n;  // R is (k, n)
  rs_j = 1;
  os_i = n;  // Out is (m, n)
  os_j = 1;
  result_idx = result_i * os_i + result_j * os_j;

  // Blockidx determines the 'i' of the tile in lhs matrix

  // Number of tiles to grab: k / tile_len rounded up
  for(int tileIdx = 0; tileIdx < ceil(k / (float)tile_len); tileIdx++){
    size_t i_start_m = blockIdx.y * blockDim.y;
    size_t j_start_m = tileIdx * tile_len;
    size_t i_start_n = j_start_m;
    size_t j_start_n = blockIdx.x * blockDim.x;

    // Copy the data to shared
    for(size_t mi=0; mi < tile_len; mi++){
      // ensure we haven't run over
      if (i_start_m + mi >= m)
        break;
      for(size_t mj=0; mj < tile_len; mj++){
        if (j_start_m + mj >= k)
          break;
        mn_tiles[mi * tile_len + mj] = data_l[(mi + i_start_m) * ls_i + (mj + j_start_m) * ls_j];
      }
    }

    for(size_t ni=0; ni < tile_len; ni++){
      if (i_start_n + ni >= k)
        break;
      for(size_t nj=0; nj < tile_len; nj++){
        if (j_start_n + nj >= n)
          break;
        mn_tiles[mtile_size + ni * tile_len + nj] = data_r[(ni + i_start_n) * rs_i + (nj + j_start_n) * rs_j];
      }
    }

    __syncthreads();  // Need the whole block to finish copy before running dot product

    // Compute inner product terms on this tile
    // Should only run on threads with a valid result pointer
    if ( (result_i < m) && (result_j < n) )
      for(int inner = 0; inner < tile_len; inner++) {
        // Ensure the current value of k is valid
        if (tileIdx * tile_len + inner >= k)
          continue;
        float M_ik, N_kj;
        M_ik = mn_tiles[threadIdx.y * tile_len + inner];
        N_kj = mn_tiles[mtile_size + inner * tile_len + threadIdx.x];
        out[result_idx] += M_ik * N_kj;
      }
    
    // Sync again to ensure no threads start overwriting the data while others are still working
    __syncthreads();
  }
}
