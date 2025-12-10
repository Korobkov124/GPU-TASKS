#pragma once
#include "vectorView.cuh"
#include <cuda_runtime.h>

template<typename T>
__global__ void vectorAddNobrKernel(VectorView<T> a, T* blockSum, std::size_t n) {
    extern __shared__ T sharedMem[];

    std::size_t tid = threadIdx.x;
    std::size_t blockSize = blockDim.x;

    T threadSum = static_cast<T>(0);

    for (std::size_t i = tid; i < n; i += blockSize) {
        threadSum += a[i];
    }

    sharedMem[tid] = threadSum;
    __syncthreads();

    for (int i = blockSize / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sharedMem[tid] += sharedMem[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSum[tid] = sharedMem[0];
    }
}

template<typename T>
__global__ void finalReductionNobrKernel(T* blockSum, T* sum, std::size_t n) {
        extern __shared__ T sharedMem[];
        std::size_t tid = threadIdx.x;

        if (tid < n) {
            sharedMem[tid] = blockSum[tid];
        } else {
            sharedMem[tid] = static_cast<T>(0);
        }
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i >>= 1) {
            if (tid < i) {
                sharedMem[tid] += sharedMem[tid + i];
            }
            __syncthreads();
        }

        if (tid == 0) {
            *sum = sharedMem[0];
        }
}

template<typename T>
__global__ void vectorAddBrKernel(VectorView<T> a, T* blockSum, std::size_t n) {
    extern __shared__ T warpSum[];

    std::size_t tid = threadIdx.x;
    std::size_t blockSize = blockDim.x;
    std::size_t lane = tid % 32;
    std::size_t warp = tid / 32;
    std::size_t warpPerBlock = (blockDim.x + 31) / 32;

    T threadSum = static_cast<T>(0);
    for (std::size_t i = tid; i < n; i += blockSize) {
        threadSum += a[i];
    }
    
    T warpS = threadSum;
    unsigned mask = 0xFFFFFFFF;

    for (int j = 16; j > 0; j /=2) {
        warpS += __shfl_down_sync(mask, warpS, j);
    }

    if (lane == 0) {
        warpSum[warp] = warpS;
    }
    __syncthreads();

    if (warp == 0) {
        T val = (lane < warpPerBlock) ? warpSum[lane] : static_cast<T>(0);

        for (int i = 16; i > 0; i /= 2) {
            val += __shfl_down_sync(mask, val, i);
        }

        if (lane == 0) {
            blockSum[tid] = val;
        }
    }
}

template<typename T>
__global__ void finalReductionBrKernel(T* blockSum, T* sum, std::size_t n) {
    extern __shared__ T warpSum[];

    std::size_t tid = threadIdx.x;
    std::size_t blockSize = blockDim.x;
    std::size_t lane = tid % 32;
    std::size_t warp = tid / 32;
    std::size_t warpPerBlock = (blockDim.x + 31) / 32;

    T threadVal = static_cast<T>(0);
    if (tid < n) {
        threadVal = blockSum[tid];
    }

    T warpS = threadVal;
    unsigned mask = 0xFFFFFFFF;

    for (int j = 16; j > 0; j /= 2) {
        warpS += __shfl_down_sync(mask, warpS, j);
    }

    if (lane == 0) {
        warpSum[warp] = warpS;
    }
    __syncthreads();

    if (warp == 0) {
        T val = (lane < warpPerBlock) ? warpSum[lane] : static_cast<T>(0);

        for (int i = 16; i > 0; i /= 2) {
            val += __shfl_down_sync(mask, val, i);
        }

        if (lane == 0) {
            *sum = val;
        }
    }
}