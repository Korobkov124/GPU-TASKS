#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "gpu_addVect.cuh"
#include "cuda_timer.hpp"

static void Bench_CPU(benchmark::State& state) {
  int N = state.range(0);
  std::vector<float> a(N), b(N), cpu_res(N);


  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (int i = 0; i < N; ++i) {
      a[i] = dist(rng);
      b[i] = dist(rng);
  }

  for (auto _ : state) {
        AddVect::CpuAddVect(a.data(), b.data(), cpu_res.data(), N);
        benchmark::DoNotOptimize(cpu_res.data());
    }
}

static void Bench_GPU(benchmark::State& state) {
  int N = state.range(0);
  std::vector<float> a(N), b(N), gpu_res(N);


  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (int i = 0; i < N; ++i) {
      a[i] = dist(rng);
      b[i] = dist(rng);
  }

  for (auto _ : state) {
        AddVect::RunGpu(a.data(), b.data(), gpu_res.data(), N);
        benchmark::DoNotOptimize(gpu_res.data());
        benchmark::ClobberMemory();
    }
}

static void Bench_GPU_Esh(benchmark::State& state) {
  std::size_t N = state.range(0);
  std::vector<float> a(N), b(N), gpu_res(N);

  for (int i = 0; i < N; ++i) {
      a[i] = 0.1f * i;
      b[i] = 0.2f * i;
  }

  float *devVect1, *devVect2, *devResult;

  cudaMalloc((void**)&devVect1, sizeof(float) * N);
  cudaMalloc((void**)&devVect2, sizeof(float) * N);
  cudaMalloc((void**)&devResult, sizeof(float) * N);

  cudaMemcpy(devVect1, a.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(devVect2, b.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(devResult, gpu_res.data(), sizeof(float) * N, cudaMemcpyHostToDevice);


  for (auto _ : state) {
    float elapsed_time = 0;
    
    {
      CUDATimer timer(elapsed_time);
      AddVect::FullGpuAddVect(devVect1, devVect2, devResult, N);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }

  cudaFree(devVect1);
  cudaFree(devVect2);
  cudaFree(devResult);
}

BENCHMARK(Bench_CPU)
  ->RangeMultiplier(2)
  ->Range(1<<10, 1<<22);

BENCHMARK(Bench_GPU)
  ->RangeMultiplier(2)
  ->Range(1<<10, 1<<22);

BENCHMARK(Bench_GPU_Esh)
  ->RangeMultiplier(2)
  ->Range(1<<10, 1<<22)
  ->UseManualTime();

BENCHMARK_MAIN();