#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cstdint>

#include "vector.cuh"
#include "vectorOperations.cuh"
#include "cuda_timer.hpp"

using T = float;

static void addVectorRanges(benchmark::internal::Benchmark* b) {
    int64_t start = 8LL;
    int64_t end   = (int64_t(8) << 28);
    b->RangeMultiplier(2)->Range(start, end)->Unit(benchmark::kMillisecond)->UseManualTime();
}


template<AddAlgorithm Alg>
void warmup_kernel(Vector<T, Alg>& v) {
    std::size_t n = v.size();
    if (n == 0) return;

    Data<T> blockSum(1);
    constexpr std::size_t block_size = 256;

    std::size_t nBlocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;

    if constexpr (Alg == AddAlgorithm::nobr) {
        vectorAddNobrKernel<T><<<nBlocks, block_size, block_size * sizeof(T)>>>(v.view(), blockSum.data(), n);
    } else {
        vectorAddBrKernel<T><<<nBlocks, block_size, ((block_size + 31) / 32) * sizeof(T)>>>(v.view(), blockSum.data(), n);
    }
    cudaDeviceSynchronize();
}


static void BM_VectorSum_Nobr(benchmark::State& state) {
    int64_t n64 = state.range(0);
    std::size_t n = static_cast<std::size_t>(n64);

    Vector<T, AddAlgorithm::nobr> v(n);
    v.random();

    warmup_kernel<AddAlgorithm::nobr>(v);

    constexpr std::size_t block_size = 256;
    std::size_t nBlocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;

    Data<T> blockSum(nBlocks);
    Data<T> result(1);

    for (auto _ : state) {
        float elapsed_time_ = 0;
        {
            CUDATimer timer(elapsed_time_);

            vectorAddNobrKernel<T><<<static_cast<int>(nBlocks), static_cast<int>(block_size), block_size * sizeof(T)>>>(
                v.view(), blockSum.data(), n
            );

            finalReductionNobrKernel<T><<<1, static_cast<int>(block_size), block_size * sizeof(T)>>>(
                blockSum.data(), result.data(), nBlocks
            );
        }
        
        benchmark::DoNotOptimize(elapsed_time_);
        benchmark::ClobberMemory();

        state.SetIterationTime(elapsed_time_);
    }
}
BENCHMARK(BM_VectorSum_Nobr)->Apply(addVectorRanges)->Name("VectorSum_nobr")->UseManualTime()->Unit(benchmark::kMillisecond);

static void BM_VectorSum_Br(benchmark::State& state) {
    int64_t n64 = state.range(0);
    std::size_t n = static_cast<std::size_t>(n64);

    Vector<T, AddAlgorithm::br> v(n);
    v.random();

    warmup_kernel<AddAlgorithm::br>(v);

    constexpr std::size_t block_size = 256;
    std::size_t nBlocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;

    Data<T> blockSum(nBlocks);
    Data<T> result(1);

    for (auto _ : state) {
        float elapsed_time_ = 0;
        {
            CUDATimer timer(elapsed_time_);

            vectorAddBrKernel<T><<<static_cast<int>(nBlocks), static_cast<int>(block_size), ((block_size + 31) / 32) * sizeof(T)>>>(
                v.view(), blockSum.data(), n
            );

            finalReductionBrKernel<T><<<1, static_cast<int>(block_size), ((block_size + 31) / 32) * sizeof(T)>>>(
                blockSum.data(), result.data(), nBlocks
            );
        }

        benchmark::DoNotOptimize(elapsed_time_);
        benchmark::ClobberMemory();

        state.SetIterationTime(elapsed_time_);
    }
}
BENCHMARK(BM_VectorSum_Br)->Apply(addVectorRanges)->Name("VectorSum_br")->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
