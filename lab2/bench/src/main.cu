#include <benchmark/benchmark.h>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include "matrix.cuh"
#include "matrixOperationsKernel.cuh"


static void BM_Eigen_Matrix(benchmark::State& state){
    int N = state.range(0);

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
    Eigen::MatrixXf C(N, N);

    for (auto _ : state)
    {
      C = A * B;
      benchmark::DoNotOptimize(C.data());
      // benchmark::ClobberMemory();
    }
}

static void BM_Cuda_Matrix(benchmark::State& state){
  int N = state.range(0);

  Matrix<float> A(N, N);
  Matrix<float> B(N, N);
  Matrix<float> C(N, N);

  A.fill(25.0);
  B.fill(25.0);

  constexpr std::size_t BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  
  for (auto _ : state)
    {
      matrixMultiplyKernel<<<gridDim, blockDim>>>(A.view(), B.view(), C.view());
      benchmark::DoNotOptimize(C.data());
      // benchmark::ClobberMemory();
    }

}

BENCHMARK(BM_Eigen_Matrix)
    ->Name("Eigen Matrix Mupliplication (CPU)")
    ->RangeMultiplier(2)
    ->Range(1<<4, 1<<10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Cuda_Matrix)
    ->Name("CUDA Matrix Mupliplication (GPU)")
    ->RangeMultiplier(2)
    ->Range(1<<4, 1<<10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();