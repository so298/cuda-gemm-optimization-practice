#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "common.hh"

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", __FILE__, \
              __LINE__, (int)err, cudaGetErrorString(err), #call);          \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

void run_benchmark(GemmBench& bench, int num_iter, size_t N, size_t K,
                   size_t M) {
  GemmBench::T *A, *B, *C;

  // Allocate memory
  CUDA_CHECK(cudaMalloc(&A, N * K * sizeof(GemmBench::T)));
  CUDA_CHECK(cudaMalloc(&B, K * M * sizeof(GemmBench::T)));
  CUDA_CHECK(cudaMalloc(&C, N * M * sizeof(GemmBench::T)));

  // Host data
  std::vector<GemmBench::T> h_A(N * K);
  std::vector<GemmBench::T> h_B(K * M);
  std::vector<GemmBench::T> h_C(N * M);

  // Initialize randomly
  int seed = 42;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<GemmBench::T> dis(0.0, 1.0);
  for (size_t i = 0; i < N * K; i++) h_A[i] = dis(gen);
  for (size_t i = 0; i < K * M; i++) h_B[i] = dis(gen);

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(A, h_A.data(), N * K * sizeof(GemmBench::T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, h_B.data(), K * M * sizeof(GemmBench::T),
                        cudaMemcpyHostToDevice));

  // Warm up
  bench.run(10, N, K, M, A, B, C);

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  bench.run(num_iter, N, K, M, A, B, C);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  // Results
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
  std::cout << "Elapsed time per iteration: "
            << static_cast<double>(elapsed) / num_iter << " ms" << std::endl;
  std::cout << "Throughput: "
            << 2.0 * static_cast<double>(N) * K * M * num_iter / (elapsed * 1e6)
            << " GFLOPs" << std::endl;

  // Clean up
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
}