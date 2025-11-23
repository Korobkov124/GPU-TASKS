#pragma once
#include "matrix.cuh"
#include "matrixOperationsKernel.cuh"
#include <stdexcept>

template<typename T, MultiplyAlgorithm Algorithm>
template<MultiplyAlgorithm OtherAlgorithm>
Matrix<T, Algorithm> Matrix<T, Algorithm>::operator*(const Matrix<T, OtherAlgorithm>& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Invalid matrix for multiply");
    }

    Matrix<T, Algorithm> result(rows(), other.cols());
    
    constexpr std::size_t BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((other.cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    if constexpr (Algorithm == MultiplyAlgorithm::shared) {
        matrixMultiplyKernelShared<T><<<gridDim, blockDim>>>(view(), other.view(), result.view());
    } else if constexpr (Algorithm == MultiplyAlgorithm::naive) {
        matrixMultiplyKernel<T><<<gridDim, blockDim>>>(view(), other.view(), result.view());
    } else if constexpr (Algorithm == MultiplyAlgorithm::wmma) {
        dim3 wmmaBlock(128);
        dim3 wmmaGrid((other.cols() + 15) / 16, (rows() + 15) / 16);
        matrixMultiplyKernelWMMA<T><<<wmmaGrid, wmmaBlock>>>(view(), other.view(), result.view());
    }

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return result;
}