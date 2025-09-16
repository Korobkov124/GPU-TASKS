#include "googletest/googletest/include/gtest/gtest.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "gpu_addVect.cuh"

TEST(AddVectTest, CpuVsGpuAccuracy) {
    const int N = 256;
    std::vector<float> a(N), b(N), cpu_res(N), gpu_res(N);

    
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    AddVect::AddingVectors::CpuAddVect(a.data(), b.data(), cpu_res.data(), N);

    AddVect::AddingVectors::RunGpu(a.data(), b.data(), gpu_res.data(), N);

    for (int i = 0; i < N; ++i){
      std::cout << gpu_res[i];
    }
    const float abs_error = 1e-6f;

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = std::fabs(cpu_res[i] - gpu_res[i]);
        
        ASSERT_NEAR(cpu_res[i], gpu_res[i], abs_error);

        if (d > max_diff) max_diff = d;
    }

    std::cout << "Max: " << max_diff;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

