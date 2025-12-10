#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "matrix.cuh"
#include "matrixOperations.cuh"

std::vector<std::size_t> matrix_sizes = {1, 2, 3, 127, 128, 129, 512};

Matrix<float, MultiplyAlgorithm::shared> createAndFillSharedMatrix(std::size_t rows, std::size_t cols, float value){
    Matrix<float, MultiplyAlgorithm::shared> matrix(rows, cols);
    matrix.fill(value);
    return matrix;
}

Eigen::MatrixXf createAndFillEigenMatrix(std::size_t rows, std::size_t cols, float value) {
    Eigen::MatrixXf matrix(rows, cols);
    matrix.setConstant(value);
    return matrix;
}

Eigen::MatrixXf convertToEigenMatrix(const Matrix<float>& matrix) {
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

TEST_P(KernelParamTest, shared_matrix_multiply_test){
    auto [m, n, k] = GetParam();

    Matrix<float, MultiplyAlgorithm::shared> matrixA = createAndFillSharedMatrix(m, k, 2.0f);
    Matrix<float, MultiplyAlgorithm::shared> matrixB = createAndFillSharedMatrix(k, n, 4.0f);

    Matrix<float, MultiplyAlgorithm::shared> matrixC = matrixA * matrixB;
    
    Eigen::MatrixXf eigenA = createAndFillEigenMatrix(m, k, 2.0f);
    Eigen::MatrixXf eigenB = createAndFillEigenMatrix(k, n, 4.0f);

    Eigen::MatrixXf eigenC = eigenA * eigenB;

    Eigen::MatrixXf to_compare_C = convertToEigenMatrix(matrixC);
    
    std::cout << "Running test with M=" << m << ", N=" << n << ", K=" << k << std::endl;

    EXPECT_TRUE(to_compare_C.isApprox(eigenC, 1e-5f))
        << "Matrix multiplication failed for sizes: " 
        << m << " × " << k << " and " << k << " × " << n
        << "\nEigen result:\n" << eigenC
        << "\nResult:\n" << to_compare_C;
}

INSTANTIATE_TEST_SUITE_P(
    shared_matrix_multiply_test,
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
