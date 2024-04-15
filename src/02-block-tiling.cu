#include <cassert>
#include <iostream>

#include "common.hh"
#include "common_gpu.cuh"

constexpr size_t TILE_SIZE_N = 32;
constexpr size_t TILE_SIZE_M = TILE_SIZE_N;
constexpr size_t TILE_SIZE_K = TILE_SIZE_N;

__device__ void copy_tile(size_t N, size_t K, size_t M, const GemmBench::T* A,
                          const GemmBench::T* B, GemmBench::T* tile_A,
                          GemmBench::T* tile_B, size_t r_grid, size_t c_grid,
                          size_t k_grid) {
  // Copy tile A
  {
    size_t tile_r = threadIdx.y;
    size_t tile_c = threadIdx.x;
    size_t tile_idx = tile_r * TILE_SIZE_K + tile_c;

    size_t global_r = r_grid * TILE_SIZE_N + tile_r;
    size_t global_c = k_grid * TILE_SIZE_K + tile_c;
    size_t idx_A = K * global_r + global_c;
    tile_A[tile_idx] = (global_r < N && global_c < K) ? A[idx_A] : 0;
  }
  __syncthreads();

  // Copy tile B
  {
    size_t tile_r = threadIdx.y;
    size_t tile_c = threadIdx.x;
    size_t tile_idx = tile_r * TILE_SIZE_M + tile_c;

    size_t global_r = k_grid * TILE_SIZE_K + tile_r;
    size_t global_c = c_grid * TILE_SIZE_M + tile_c;
    size_t idx_B = M * global_r + global_c;
    tile_B[tile_idx] = (global_r < K && global_c < M) ? B[idx_B] : 0;
  }
}

__global__ void kernel_block_tiling(size_t N, size_t K, size_t M,
                                    GemmBench::T* A, GemmBench::T* B,
                                    GemmBench::T* C) {
  __shared__ GemmBench::T tile_A[TILE_SIZE_N * TILE_SIZE_K];
  __shared__ GemmBench::T tile_B[TILE_SIZE_K * TILE_SIZE_M];

  size_t r_grid = blockIdx.y;
  size_t c_grid = blockIdx.x;
  size_t r_local = threadIdx.y;
  size_t c_local = threadIdx.x;

  GemmBench::T sum = 0;

  size_t k_grid_max = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;
  for (size_t k_grid = 0; k_grid < k_grid_max; k_grid++) {
    copy_tile(N, K, M, A, B, tile_A, tile_B, r_grid, c_grid, k_grid);
    __syncthreads();

    for (size_t k_local = 0; k_local < TILE_SIZE_K; k_local++) {
      if (k_grid * TILE_SIZE_K + k_local >= K) break;
      sum += tile_A[r_local * TILE_SIZE_K + k_local] *
             tile_B[k_local * TILE_SIZE_M + c_local];
    }
  }
  __syncthreads();

  // Write back to C
  size_t global_r = r_grid * TILE_SIZE_N + r_local;
  size_t global_c = c_grid * TILE_SIZE_M + c_local;
  if (global_r < N && global_c < M) {
    // assert(global_r * M + global_c < N * M);
    C[global_r * M + global_c] = sum;
  }
}

class GemmBenchImpl : public GemmBench {
 public:
  void run(int num_iter, size_t N, size_t K, size_t M, T* A, T* B, T* C) {
    dim3 block_dim(TILE_SIZE_M, TILE_SIZE_N);
    dim3 grid_dim((M + TILE_SIZE_M - 1) / TILE_SIZE_M,
                  (N + TILE_SIZE_N - 1) / TILE_SIZE_N);

    for (int i = 0; i < num_iter; i++) {
      kernel_block_tiling<<<grid_dim, block_dim>>>(N, K, M, A, B, C);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};

int main(int argc, char** argv) {
  Options opts(argc, argv);
  GemmBenchImpl bench;

  if (opts.test) {
    test_benchmark(bench, opts.N, opts.K, opts.M);
    return 0;
  }

  run_benchmark(bench, opts.num_iter, opts.N, opts.K, opts.M);
}