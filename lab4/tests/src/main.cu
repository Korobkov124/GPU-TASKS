#define EIGEN_NO_CUDA
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "matrix.cuh"
#include "matrixOperations.cuh"
#include "data.cuh"

std::vector<std::size_t> matrix_sizes = {16, 32, 64, 128, 256, 512};

Eigen::MatrixXf createAndFillEigenMatrix(std::size_t rows, std::size_t cols) {
    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(rows, cols);
    return matrix;
}

Eigen::MatrixXf convertToEigenMatrix(const Matrix<__half>& matrix) {
    std::size_t rows = matrix.rows();
    std::size_t cols = matrix.cols();

    Eigen::MatrixXf eigen_matrix(rows, cols);
    
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            eigen_matrix(i, j) = matrix(i, j);
        }
    }
    
    return eigen_matrix;
}

class KernelParamTest: public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(KernelParamTest, wmma_matrix_multiply_test){
    auto [m, n, k] = GetParam();
    
    Eigen::MatrixXf eigenA = createAndFillEigenMatrix(m, k);
    Eigen::MatrixXf eigenB = createAndFillEigenMatrix(k, n);

    Eigen::MatrixXf eigenC = eigenA * eigenB;

    Matrix<__half, MultiplyAlgorithm::wmma> matrixA(m, k); 
    Matrix<__half, MultiplyAlgorithm::wmma> matrixB(k, n);

    std::vector<half> halfEigenA(m * k);
    std::vector<half> halfEigenB(k * n);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            halfEigenA[i*k + j] = __float2half(eigenA(i,j));

    for (int i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j)
            halfEigenB[i*n + j] = __float2half(eigenB(i,j));

    matrixA.getData().copyFromHost(halfEigenA.data());
    matrixB.getData().copyFromHost(halfEigenB.data());

    Matrix<__half, MultiplyAlgorithm::wmma> matrixC = matrixA * matrixB;

    cudaDeviceSynchronize();

    std::vector<__half> hostC(matrixC.size());
    matrixC.getData().copyToHost(hostC.data());

    Eigen::MatrixXf toСompareС = convertToEigenMatrix(matrixC);
    
    std::cout << "Running test with M=" << m << ", N=" << n << ", K=" << k << std::endl;

    EXPECT_TRUE(toСompareС.isApprox(eigenC, 1e-2f))
        << "Matrix multiplication failed for sizes: " 
        << m << " × " << k << " and " << k << " × " << n
        << "\nEigen result:\n" << eigenC
        << "\nResult:\n" << toСompareС;
}

INSTANTIATE_TEST_SUITE_P(
    wmma_matrix_multiply_test,
    KernelParamTest,
    ::testing::Combine(
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(matrix_sizes)
    )
);

int main(int argc, char **argv){

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
