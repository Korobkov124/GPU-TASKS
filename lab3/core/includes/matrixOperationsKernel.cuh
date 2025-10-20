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
__global__ void matrixMultiplyKernelShared(MatrixView<T> A, MatrixView<T> B, MatrixView<T> C) {
    constexpr std::size_t tile_size = 16;

    __shared__ T tileA[tile_size][tile_size];
    __shared__ T tileB[tile_size][tile_size];
    std::size_t row = blockIdx.y * tile_size + threadIdx.y;
    std::size_t col = blockIdx.x * tile_size + threadIdx.x;
    std::size_t x = threadIdx.x;
    std::size_t y = threadIdx.y;

    T sum = static_cast<T>(0);

    for (std::size_t i = 0; i < (A.cols() + tile_size - 1) / tile_size; ++i) {
        std::size_t aCol = i * tile_size + x;
        if(row < A.rows() && aCol < A.cols()) {
            tileA[y][x] = A(row, aCol);
        } else {
            tileA[y][x] = static_cast<T>(0);
        }
        
        std::size_t bRow = i * tile_size + y;
        if (bRow < B.rows() && col < B.cols()) {
            tileB[y][x] = B(bRow, col);
        }else {
            tileB[y][x] = static_cast<T>(0);
        }

        __syncthreads();

        for(std::size_t j = 0; j < tile_size; ++j) {
            sum += tileA[y][j] * tileB[j][x];
        }

        __syncthreads();
    }

    if (row < C.rows() && col < C.cols()) {
        C(row, col) = sum;
    }

}