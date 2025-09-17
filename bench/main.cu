#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "gpu_addVect.cuh"


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
        AddVect::AddingVectors::CpuAddVect(a.data(), b.data(), cpu_res.data(), N);
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
        AddVect::AddingVectors::RunGpu(a.data(), b.data(), gpu_res.data(), N);
        benchmark::DoNotOptimize(gpu_res.data());
    }
}

BENCHMARK(Bench_CPU)
  ->RangeMultiplier(2)
  ->Range(1<<10, 1<<22);

BENCHMARK(Bench_GPU)
  ->RangeMultiplier(2)
  ->Range(1<<10, 1<<22);


BENCHMARK_MAIN();