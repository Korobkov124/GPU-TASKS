#pragma once
#include "matrix.cuh"
#include "matrixOperationsKernel.cuh"
#include <stdexcept>

template<typename T>
__host__ Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Invalid matrix for multiply");
    }

    Matrix<T> result(rows(), other.cols());
    
    constexpr std::size_t BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((other.cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixMultiplyKernel<T><<<gridDim, blockDim>>>(view(), other.view(), result.view());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return result;
}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> result(rows(), cols());
    
    constexpr std::size_t BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixMultiplyScalarKernel<T><<<gridDim, blockDim>>>(view(), scalar, result.view());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return result;
}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (!isSameSize(other)) {
        throw std::invalid_argument("Invalid matrix for addition");
    }

    Matrix<T> result(rows(), cols());
    
    constexpr std::size_t BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixAddKernel<T><<<gridDim, blockDim>>>(view(), other.view(), result.view());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return result;
}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (!isSameSize(other)) {
        throw std::invalid_argument("Invalid matrix for subtraction");
    }

    Matrix<T> result(rows(), cols());
    
    constexpr std::size_t BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixSubKernel<T><<<gridDim, blockDim>>>(view(), other.view(), result.view());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return result;
}

template<typename T>
__host__ Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols(), rows());

    constexpr std::size_t BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((result.cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (result.rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixTransposeKernel<T><<<gridDim, blockDim>>>(view(), result.view());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return result;
}