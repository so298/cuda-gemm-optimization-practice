#include <cublas_v2.h>

#include <iostream>

#include "common.hh"
#include "common_gpu.cuh"

#define CUBLAS_CHECK(call)                                  \
  {                                                         \
    cublasStatus_t status = call;                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                  \
      std::cerr << "CUBLAS Error: " << status << std::endl; \
      exit(1);                                              \
    }                                                       \
  }

class GemmBenchCublasImpl : public GemmBench {
 public:
  void run(int num_iter, size_t N, size_t K, size_t M, T* A, T* B, T* C) {
    cublasHandle_t handle;
    float alpha = 1.0;
    float beta = 0.0;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (int i = 0; i < num_iter; i++) {
      // cuBLAS uses column-major order
      // row-major C = AB => column-major C = B^T A^T
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                               &alpha, B, M, A, K, &beta, C, M));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUBLAS_CHECK(cublasDestroy(handle));
  }
};

int main(int argc, char** argv) {
  Options opts(argc, argv);
  GemmBenchCublasImpl bench;

  if (opts.test) {
    test_benchmark(bench, opts.N, opts.K, opts.M);
    return 0;
  }

  run_benchmark(bench, opts.num_iter, opts.N, opts.K, opts.M);
}