#include <iostream>

#include "common.hh"
#include "common_gpu.cuh"

__global__ void kernel_gpu_naive(size_t N, size_t K, size_t M, GemmBench::T* A,
                                 GemmBench::T* B, GemmBench::T* C) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t x = idx / M;  // row idx of C
  size_t y = idx % M;  // col idx of C

  if (x < M && y < N) {
    GemmBench::T sum = 0;
    for (size_t k = 0; k < K; k++) {
      sum += A[x * K + k] * B[k * M + y];
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

  if (opts.test) {
    test_benchmark(bench, opts.N, opts.K, opts.M);
    return 0;
  }

  run_benchmark(bench, opts.num_iter, opts.N, opts.K, opts.M);
}