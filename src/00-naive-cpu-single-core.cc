#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "common.hh"

void run_benchmark(GemmBench& bench, int num_iter, size_t N, size_t K,
                   size_t M) {
  // Host data
  GemmBench::T* A = new GemmBench::T[N * K];
  GemmBench::T* B = new GemmBench::T[K * M];
  GemmBench::T* C = new GemmBench::T[N * M];

  // Initialize randomly
  int seed = 42;
  std::mt19937 gen(seed);
  std::normal_distribution<GemmBench::T> dis(0.0, 1.0);
  for (size_t i = 0; i < N * K; i++) A[i] = dis(gen);
  for (size_t i = 0; i < K * M; i++) B[i] = dis(gen);

  // Warm up
  bench.run(100, N, K, M, A, B, C);

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  bench.run(num_iter, N, K, M, A, B, C);
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
  delete[] A;
  delete[] B;
  delete[] C;
}

void kernel_cpu_single_core_naive(size_t N, size_t K, size_t M, GemmBench::T* A,
                                  GemmBench::T* B, GemmBench::T* C) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      GemmBench::T sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * M + j];
      }
      C[i * M + j] = sum;
    }
  }
}

class BenchCpuSingleCoreNaive : public GemmBench {
 public:
  void run(int num_iter, size_t N, size_t K, size_t M, T* A, T* B, T* C) {
    for (int i = 0; i < num_iter; i++) {
      kernel_cpu_single_core_naive(N, K, M, A, B, C);
    }
  }
};

int main(int argc, char** argv) {
  Options opts(argc, argv);
  BenchCpuSingleCoreNaive bench;
  run_benchmark(bench, opts.num_iter, opts.N, opts.K, opts.M);

  return 0;
}