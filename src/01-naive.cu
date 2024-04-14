#include <iostream>

#include "common.hh"
#include "common_gpu.cuh"

__global__ void kernel_gpu_naive(size_t N, size_t K, size_t M, GemmBench::T* A,
                                 GemmBench::T* B, GemmBench::T* C) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t x = idx % M;  // col idx of C
  size_t y = idx / M;  // row idx of C

  if (idx < N * M) {
    GemmBench::T sum = 0;
    for (size_t i = 0; i < K; i++) {
      sum += A[x * K + i] * B[i * M + y];
    }
    C[x * M + y] = sum;
  }
}

class GemmBenchNaive : public GemmBench {
 public:
  void run(int num_iter, size_t N, size_t K, size_t M, T* A, T* B, T* C) {
    size_t block_size = 256;
    size_t grid_size = (N * M + block_size - 1) / block_size;

    for (int i = 0; i < num_iter; i++) {
      kernel_gpu_naive<<<grid_size, block_size>>>(N, K, M, A, B, C);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};

int main(int argc, char** argv) {
  Options opts(argc, argv);
  GemmBenchNaive bench;

  run_benchmark(bench, opts.num_iter, opts.N, opts.K, opts.M);
}