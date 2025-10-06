#pragma once
#include "matrix.cuh"
#include "matrixOperationsKernel.cuh"
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace MatrixOperations {

    using ::gpuMultiply;
    using ::gpuAdd;
    using ::gpuSub;
    using ::gpuTranspose;

    template<typename T>
    void cpuMultiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        if (A.cols() != B.rows()) throw std::invalid_argument("Invalid matrix for multiply");
        for (std::size_t i = 0; i < A.rows(); i++) {
            for (std::size_t j = 0; j < B.cols(); j++) {
                T sum = static_cast<T>(0);
                for(std::size_t k = 0; k < A.cols(); k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
    }

    template<typename T>
    void cpuAdd(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for addition");
        for(std::size_t i = 0; i < A.size(); i++) {
            C.data()[i] = A.data()[i] + B.data()[i];
        }
    }

    template<typename T>
    void cpuSub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for subtraction");
        for(std::size_t i = 0; i < A.size(); i++) {
            C.data()[i] = A.data()[i] - B.data()[i];
        }
    }

    template<typename T>
    Matrix<T> cpuTranspose(const Matrix<T>& A) {
        Matrix<T> result(A.cols(), A.rows());
        for (std::size_t i = 0; i < A.rows(); i++) {
            for (std::size_t j = 0; j < A.cols(); j++) {
                result(j, i) = A(i, j);
            }
        }
        return result;
    }

} // namespace MatrixOperations
// matrixOperations.cuh
// #pragma once
// #include <cuda_runtime.h>
// #include <iostream>
// #include <stdexcept>
// #include <cmath>
// #include "matrix.cuh"

// namespace MatrixOperations {

//     // Объявления ядер CUDA
//     template<typename T>
//     __global__ void matrixMultiplyKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
//         #ifdef __CUDA_ARCH__  // Проверяем, что компилируется для GPU
//         std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//         std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//         if (row < A.rows() && col < B.cols()) {
//             T sum = static_cast<T>(0);
//             for(std::size_t i = 0; i < A.cols(); i++) {
//                 sum += A(row, i) * B(i, col);
//             }
//             C(row, col) = sum;
//         }
//         #endif
//     }

//     template<typename T>
//     __global__ void matrixAddKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
//         #ifdef __CUDA_ARCH__
//         std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//         std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//         if (row < A.rows() && col < B.cols()) {
//             C(row, col) = A(row, col) + B(row, col);
//         }
//         #endif
//     }

//     template<typename T>
//     __global__ void matrixSubKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
//         #ifdef __CUDA_ARCH__
//         std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//         std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//         if (row < A.rows() && col < B.cols()) {
//             C(row, col) = A(row, col) - B(row, col);
//         }
//         #endif
//     }

//     template<typename T>
//     __global__ void matrixTransposeKernel(const Matrix<T> A, Matrix<T> result) {
//         #ifdef __CUDA_ARCH__
//         std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//         std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//         if (row < A.rows() && col < A.cols()) {
//             result(col, row) = A(row, col);
//         }
//         #endif
//     }

//     // Реализации функций, которые используют ядра
//     template<typename T>
//     void gpuMultiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//         if (A.cols() != B.rows()) throw std::invalid_argument("Invalid matrix for multiply");

//         dim3 blockSize(16,16);
//         dim3 gridSize((B.cols() + 15) / 16, (A.rows() + 15) / 16);
//         matrixMultiplyKernel<T><<<gridSize, blockSize>>>(A, B, C);

//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
//         }

//         cudaDeviceSynchronize();
//     }

//     template<typename T>
//     void gpuAdd(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//         if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for addition");
//         dim3 blockSize(16,16);
//         dim3 gridSize((A.cols() + 15) / 16, (A.rows() + 15) / 16);
//         matrixAddKernel<T><<<gridSize, blockSize>>>(A, B, C);

//         cudaDeviceSynchronize();
//     }

//     template<typename T>
//     void gpuSub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//         if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for subtraction");
//         dim3 blockSize(16,16);
//         dim3 gridSize((A.cols() + 15) / 16, (A.rows() + 15) / 16);
//         matrixSubKernel<T><<<gridSize, blockSize>>>(A, B, C);

//         cudaDeviceSynchronize();
//     }
    
//     template<typename T>
//     Matrix<T> gpuTranspose(const Matrix<T>& A) {
//         Matrix<T> result(A.cols(), A.rows());
//         dim3 blockSize(16,16);
//         dim3 gridSize((A.cols() + 15) / 16, (A.rows() + 15) / 16);
//         matrixTransposeKernel<T><<<gridSize, blockSize>>>(A, result);

//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
//         }

//         cudaDeviceSynchronize();
//         return result;
//     }

//     // CPU реализации
//     template<typename T>
//     void cpuMultiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//         if (A.cols() != B.rows()) throw std::invalid_argument("Invalid matrix for multiply");
//         for (std::size_t i = 0; i < A.rows(); i++) {
//             for (std::size_t j = 0; j < B.cols(); j++) {
//                 T sum = static_cast<T>(0);
//                 for(std::size_t k = 0; k < A.cols(); k++) {
//                     sum += A(i, k) * B(k, j);
//                 }
//                 C(i, j) = sum;
//             }
//         }
//     }

//     template<typename T>
//     void cpuAdd(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//         if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for addition");
//         for(std::size_t i = 0; i < A.size(); i++) {
//             C.data()[i] = A.data()[i] + B.data()[i];
//         }
//     }

//     template<typename T>
//     void cpuSub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//         if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for subtraction");
//         for(std::size_t i = 0; i < A.size(); i++) {
//             C.data()[i] = A.data()[i] - B.data()[i];
//         }
//     }

//     template<typename T>
//     Matrix<T> cpuTranspose(const Matrix<T>& A) {
//         Matrix<T> result(A.cols(), A.rows());
//         for (std::size_t i = 0; i < A.rows(); i++) {
//             for (std::size_t j = 0; j < A.cols(); j++) {
//                 result(j, i) = A(i, j);
//             }
//         }
//         return result;
//     }

// } // namespace MatrixOperations