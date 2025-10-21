#include <benchmark/benchmark.h>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include "matrix.cuh"
#include "matrixOperationsKernel.cuh"
#include "matrixOperations.cuh"
#include "cuda_timer.hpp"


static void BM_Eigen_Matrix(benchmark::State& state){
    int n_ = state.range(0);

    Eigen::MatrixXf matrixA = Eigen::MatrixXf::Random(n_, n_);
    Eigen::MatrixXf matrixB = Eigen::MatrixXf::Random(n_, n_);
    Eigen::MatrixXf matrixC(n_, n_);

    for (auto _ : state)
    {
      matrixC = matrixA * matrixB;
      benchmark::DoNotOptimize(matrixC.data());
    }
}

static void BM_Cuda_Matrix(benchmark::State& state){
  int n_ = state.range(0);

  Matrix<float> matrixA(n_, n_);
  Matrix<float> matrixB(n_, n_);
  Matrix<float> matrixC(n_, n_);

  matrixA.fill(25.0);
  matrixB.fill(25.0);

  constexpr std::size_t BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n_ + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n_ + BLOCK_SIZE - 1) / BLOCK_SIZE);
  
  for (auto _ : state)
    {
      float elapsed_time = 0;

      {
        CUDATimer timer(elapsed_time);
        matrixMultiplyKernel<float><<<gridDim, blockDim>>>(matrixA.view(), matrixB.view(), matrixC.view());
      }
      benchmark::DoNotOptimize(elapsed_time);
      benchmark::ClobberMemory();

      state.SetIterationTime(elapsed_time);
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