#pragma once
#include "matrixView.cuh"
#include <mma.h>
#include <cuda_fp16.h>

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

template<typename T>
constexpr size_t type_id() {
    if constexpr (std::is_same_v<T, float>) return 8;
    else return 16;
}

template<typename T>
__global__ void matrixMultiplyKernelWMMA(MatrixView<T> A, MatrixView<T> B, MatrixView<T> C) {
    using namespace nvcuda;

    
    constexpr size_t wmma_m = 16;
    constexpr size_t wmma_n = 16;
    constexpr size_t wmma_k = type_id<T>();
    constexpr size_t warpSize = 32;

    const size_t warp_id = threadIdx.x / warpSize;
    
    const size_t warpM = blockIdx.y * (blockDim.x / warpSize) + warp_id;
    const size_t warpN = blockIdx.x;

    if (warpM >= (C.rows() + wmma_m - 1) / wmma_m || 
        warpN >= (C.cols() + wmma_n - 1) / wmma_n) {
        return;
    }


    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, T, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, T> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);


    if (warpM >= C.rows() / wmma_m || warpN >= C.cols() / wmma_n){
        return;
    }

    for (size_t i = 0; i < A.cols(); i += wmma_k) {
        if (i + wmma_k <= A.cols()) {
            wmma::load_matrix_sync(a_frag, &A(warpM * wmma_m, i), A.cols());
            wmma::load_matrix_sync(b_frag, &B(i, warpN * wmma_n), B.cols());

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    wmma::store_matrix_sync(&C(warpM * wmma_m, warpN * wmma_n), c_frag, C.cols(), wmma::mem_row_major);


}