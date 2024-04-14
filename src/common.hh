#pragma once

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
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
  bool test = false;

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
      } else if (arg == "--test") {
        test = true;
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

std::string matrix_to_str(size_t row, size_t col, GemmBench::T* matrix) {
  std::ostringstream oss;
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      size_t index = i * col + j;
      oss << std::setw(9) << std::setfill(' ') << std::left << matrix[index];
    }
    oss << "\n";
  }
  return oss.str();
}