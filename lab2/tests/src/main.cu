#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "matrix.cuh"
#include "matrixOperations.cuh"


std::vector<std::size_t> matrix_sizes = {1, 2, 3, 127, 128, 129, 512};

Matrix<float> createAndFillMatrix(std::size_t rows, std::size_t cols, float value) {
    Matrix<float> matrix(rows, cols);
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

TEST_P(KernelParamTest, SquareMatrixTest) {
  auto [m, k, n] = GetParam();

  Matrix<float> A = createAndFillMatrix(m, k, 2.0f);
  Matrix<float> B = createAndFillMatrix(k, n, 3.0f);

  Matrix<float> C = A * B;
  
  Eigen::MatrixXf eigen_A = createAndFillEigenMatrix(m, k, 2.0f);
  Eigen::MatrixXf eigen_B = createAndFillEigenMatrix(k, n, 3.0f);
  Eigen::MatrixXf eigen_C = eigen_A * eigen_B;

  Eigen::MatrixXf to_compare_C = convertToEigenMatrix(C);
  
  EXPECT_TRUE(to_compare_C.isApprox(eigen_C, 1e-5f))
        << "Matrix multiplication failed for sizes: " 
        << m << " × " << k << " and " << k << " × " << n
        << "\nEigen result:\n" << eigen_C
        << "\nResult:\n" << to_compare_C;
}

INSTANTIATE_TEST_SUITE_P(
    SquareMatrix,
    KernelParamTest,
    ::testing::Combine(
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(matrix_sizes)
    )
);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}