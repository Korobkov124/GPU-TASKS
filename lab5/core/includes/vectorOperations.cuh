#pragma once
#include "vector.cuh"
#include "vectorOperationsKernel.cuh"

template<typename T, AddAlgorithm Algorithm>
T Vector<T, Algorithm>::sum() const {
    std::size_t n = view().size();

    if (n == 0) return static_cast<T>(0);

    constexpr std::size_t block_size = 256;
    //

    std::size_t nBlocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;
    
    Data<T> blockSum(nBlocks);
    Data<T> result(1);

    T* blockSumData = blockSum.data();
    T* resultData = result.data();

    if constexpr (Algorithm == AddAlgorithm::nobr) {
        vectorAddNobrKernel<T><<<nBlocks, block_size, block_size * sizeof(T)>>> (view(), blockSumData, n);
    } else if constexpr (Algorithm == AddAlgorithm::br) {
        vectorAddBrKernel<T><<<nBlocks, block_size, ((block_size + 31) / 32) * sizeof(T)>>>(view(), blockSumData, n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error 1: " + std::string(cudaGetErrorString(err)));
    }


    if constexpr (Algorithm == AddAlgorithm::nobr) {
        finalReductionNobrKernel<T><<<1, block_size, block_size * sizeof(T)>>>(blockSumData, resultData, n);
    } else if constexpr (Algorithm == AddAlgorithm::br) {
        finalReductionBrKernel<T><<<1, block_size, ((block_size + 31) / 32) * sizeof(T)>>>(blockSumData, resultData, n);
    }

    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        throw std::runtime_error("CUDA error 2: " + std::string(cudaGetErrorString(err2)));
    }


    T final;
    result.copyToHost(&final);

    cudaDeviceSynchronize();
    return final;
}


