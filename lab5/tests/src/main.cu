#define EIGEN_NO_CUDA
#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "vector.cuh"
#include "vectorOperations.cuh"

#include <random>
#include <vector>
#include <iostream>


static std::vector<std::size_t> vector_sizes = {1, 2, 3, 127, 129, 512, 541, 1037};


template<typename T, AddAlgorithm Alg>
Eigen::VectorXf copyVectorToEigen(const Vector<T, Alg>& v) {
    std::size_t n = v.size();
    std::vector<T> hostBuf(n);

    cudaDeviceSynchronize();
    v.getData().copyToHost(hostBuf.data());

    Eigen::VectorXf eigen = Eigen::Map<Eigen::VectorXf>(hostBuf.data(), static_cast<int>(n));
    return eigen;
}


class VecSumNobrTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecSumNobrTest, sum_matches_eigen) {
    std::size_t n = GetParam();

    Vector<float, AddAlgorithm::nobr> v(n);
    v.random();

    Eigen::VectorXf eigenVec = copyVectorToEigen(v);
    float expected = eigenVec.sum();

    float got = v.sum();

    EXPECT_NEAR(got, expected, 1e-4f)
        << "nobr sum mismatch for n=" << n
        << "\nexpected: " << expected << ", got: " << got;
}


class VecSumBrTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecSumBrTest, sum_matches_eigen) {
    std::size_t n = GetParam();

    Vector<float, AddAlgorithm::br> v(n);
    v.random();

    Eigen::VectorXf eigenVec = copyVectorToEigen(v);
    float expected = eigenVec.sum();

    float got = v.sum();

    EXPECT_NEAR(got, expected, 1e-4f)
        << "br sum mismatch for n=" << n
        << "\nexpected: " << expected << ", got: " << got;
}

INSTANTIATE_TEST_SUITE_P(NobrSizes, VecSumNobrTest, ::testing::ValuesIn(vector_sizes));
INSTANTIATE_TEST_SUITE_P(BrSizes,    VecSumBrTest,  ::testing::ValuesIn(vector_sizes));

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
