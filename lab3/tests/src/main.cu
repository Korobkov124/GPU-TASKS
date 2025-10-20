#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "matrix.cuh"
#include "matrixOperations.cuh"

std::vector<std::size_t> matrix_sizes = {1, 2, 3, 127, 128, 129, 512};

int main(int argc, char **argv){

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
