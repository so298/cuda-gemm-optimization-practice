#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

class GemmBench {
 public:
  using T = float;

  /**
   * @brief Run the benchmark once.
   *
   * @param n The number of rows in matrix A.
   * @param l The number of columns in matrix A and rows in matrix B.
   * @param m The number of columns in matrix B.
   * @param A The input matrix A.
   * @param B The input matrix B.
   * @param C The output matrix C.
   */
  virtual void run(int num_iter, size_t N, size_t K, size_t M, T* A, T* B,
                   T* C) = 0;
};

class Options {
 public:
  size_t N;
  size_t K;
  size_t M;
  size_t num_iter;

  Options(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
      std::string arg(argv[i]);
      if (arg == "-n") {
        N = std::stoul(argv[++i]);
      } else if (arg == "-k") {
        K = std::stoul(argv[++i]);
      } else if (arg == "-m") {
        M = std::stoul(argv[++i]);
      } else if (arg == "-i") {
        num_iter = std::stoul(argv[++i]);
      }
    }

    if (N == 0 || K == 0 || M == 0 || num_iter == 0) {
      std::cerr << "Usage: " << argv[0] << " -n <N> -k <K> -m <M> -i <num_iter>"
                << std::endl;
      std::exit(1);
    }

    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "num_iter: " << num_iter << std::endl;
  }
};
