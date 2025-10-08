#pragma once
#include "matrixView.cuh"

template<typename T>
__global__ void matrixMultiplyKernel(MatrixView<T> A, MatrixView<T> B, MatrixView<T> C) {
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.rows() && col < B.cols()) {
        T sum = static_cast<T>(0);
        for (std::size_t i = 0; i < A.cols(); ++i) {
            sum += A(row, i) * B(i, col);
        }
        C(row, col) = sum;
    }
}

template<typename T>
__global__ void matrixAddKernel(MatrixView<T> A, MatrixView<T> B, MatrixView<T> C) {
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.rows() && col < B.cols()) {
        C(row, col) = A(row, col) + B(row, col);
    }
}

template<typename T>
__global__ void matrixSubKernel(MatrixView<T> A, MatrixView<T> B, MatrixView<T> C) {
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.rows() && col < B.cols()) {
        C(row, col) = A(row, col) - B(row, col);
    }
}

template<typename T>
__global__ void matrixMultiplyScalarKernel(MatrixView<T> A, T scalar, MatrixView<T> C) {
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        C(row, col) = A(row, col) * scalar;

}

template<typename T>
__global__ void matrixTransposeKernel(MatrixView<T> A, MatrixView<T> B) {
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.rows() && col < B.cols()) {
        B(col, row) = A(row, col);
    }
}
