#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "matrix.cuh"
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

static void BM_Shared_Matrix(benchmark::State& state){
    float n_ = state.range(0);
    constexpr std::size_t BLOCK_SIZE = 16;

    Matrix<float, MultiplyAlgorithm::shared> matrixA(n_, n_);
    matrixA.fill(25.0);
    
    Matrix<float, MultiplyAlgorithm::shared> matrixB(n_, n_);
    matrixB.fill(25.0);

    Matrix<float, MultiplyAlgorithm::shared> matrixC(n_, n_);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n_ + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n_ + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    for (auto _ : state){

        float elapsed_time_ = 0;
        {
            CUDATimer timer(elapsed_time_);
            matrixMultiplyKernelShared<float><<<gridDim, blockDim>>>(matrixA.view(), matrixB.view(), matrixC.view());
        }
        benchmark::DoNotOptimize(elapsed_time_);
        benchmark::ClobberMemory();

        state.SetIterationTime(elapsed_time_);
    }
}

static void BM_WMMA_Matrix(benchmark::State& state){
    float n_ = state.range(0);
    constexpr std::size_t BLOCK_SIZE = 16;

    Matrix<__half, MultiplyAlgorithm::wmma> matrixA(n_, n_);
    matrixA.fill(25.0);
    
    Matrix<__half, MultiplyAlgorithm::wmma> matrixB(n_, n_);
    matrixB.fill(25.0);

    Matrix<__half, MultiplyAlgorithm::wmma> matrixC(n_, n_);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n_ + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n_ + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    for (auto _ : state){

        float elapsed_time_ = 0;
        {
            CUDATimer timer(elapsed_time_);
            matrixMultiplyKernelShared<__half><<<gridDim, blockDim>>>(matrixA.view(), matrixB.view(), matrixC.view());
        }
        benchmark::DoNotOptimize(elapsed_time_);
        benchmark::ClobberMemory();

        state.SetIterationTime(elapsed_time_);
    }
}

BENCHMARK(BM_Eigen_Matrix)
    ->Name("Eigen Matrix Mupliplication (CPU)")
    ->RangeMultiplier(2)
    ->Range(1<<4, 1<<10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Shared_Matrix)
    ->Name("CUDA Shared Matrix Mupliplication (GPU)")
    ->RangeMultiplier(2)
    ->Range(1<<4, 1<<10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_WMMA_Matrix)
    ->Name("CUDA WMMA Matrix Mupliplication (GPU)")
    ->RangeMultiplier(2)
    ->Range(1<<4, 1<<10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();