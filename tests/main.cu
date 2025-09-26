#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "gpu_addVect.cuh"


std::vector<std::size_t> g_vector_sizes = {256};

class AddVectParamTest : public ::testing::TestWithParam<std::size_t> {};

std::vector<std::size_t> ParseVectorSizes(const std::string& arg) {
    std::vector<std::size_t> result;
    size_t pos = arg.find("=");
    
    if (pos == std::string::npos) return result;
    std::string values = arg.substr(pos + 1);
    std::stringstream ss(values);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        try {
            result.push_back(std::stoi(item));
        } catch (...) {
            std::cerr << "Invalid integer in --vector_sizes: " << item << "\n";
        }
    }
    
    return result;
}


TEST_P(AddVectParamTest, CpuVsGpuAccuracy) {

    std::size_t N = GetParam();
    std::vector<float> a(N), b(N), cpu_res(N), gpu_res(N);

    for (std::size_t i = 0; i < N; i++) {
        a[i] = 0.1f * i;
        b[i] = 0.2f * i;
    }

    AddVect::CpuAddVect(a.data(), b.data(), cpu_res.data(), N);

    AddVect::RunGpu(a.data(), b.data(), gpu_res.data(), N);

    cudaDeviceSynchronize();

    const float abs_error = 1e-6f;

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        float d = std::fabs(cpu_res[i] - gpu_res[i]);
        if (d > max_diff) max_diff = d;
    }

    ASSERT_LE(max_diff, abs_error) << "Max absolute difference (" << max_diff << ") exceeds abs_error " << abs_error;

    std::cout << "Metrics test complete successfull!\n";
}

INSTANTIATE_TEST_SUITE_P(
    VectorSizes,
    AddVectParamTest,
    ::testing::ValuesIn(g_vector_sizes)
);


int main(int argc, char **argv)
{
  for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--vector_sizes=", 0) == 0) {
            g_vector_sizes = ParseVectorSizes(arg);
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            break;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

