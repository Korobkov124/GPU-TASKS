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



// #pragma once
// #include "matrix.cuh"
// #include <cuda_runtime.h>
// #include <stdexcept>

// template<typename T>
// __global__ void matrixMultiplyKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
//     std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//     std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        

//     if (row < A.rows() && col < B.cols()) {
//         if (row < 2 && col < 2) {
//             printf("Thread (%lu,%lu): ", row, col);
//         }
//         T sum = static_cast<T>(0);
//         for(std::size_t i = 0; i < A.cols(); i++) {
//             sum += A(row, i) * B(i, col);
//         }
//         C(row, col) = sum;
//     }
// }

// template<typename T>
// __global__ void matrixAddKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
//     std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//     std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < A.rows() && col < B.cols()) {
//         C(row, col) = A(row, col) + B(row, col);
//     }
// }

// template<typename T>
// __global__ void matrixSubKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
//     std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//     std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < A.rows() && col < B.cols()) {
//         C(row, col) = A(row, col) - B(row, col);
//     }
// }

// template<typename T>
// __global__ void matrixTransposeKernel(const Matrix<T> A, Matrix<T> result) {
//     std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//     std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < A.rows() && col < A.cols()) {
//         result(col, row) = A(row, col);
//     }
// }

// template<typename T>
// inline void gpuMultiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//     if (A.cols() != B.rows()) throw std::invalid_argument("Invalid matrix for multiply");

//     dim3 blockSize(16,16);
//     dim3 gridSize((B.cols() + 15) / 16, (A.rows() + 15) / 16);


//     matrixMultiplyKernel<T><<<gridSize, blockSize>>>(A, B, C);

    

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
//     }
//     cudaDeviceSynchronize();
// }

// template<typename T>
// inline void gpuAdd(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//     if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for addition");
//     dim3 blockSize(16,16);
//     dim3 gridSize((A.cols() + 15) / 16, (A.rows() + 15) / 16);
//     matrixAddKernel<T><<<gridSize, blockSize>>>(A, B, C);
//     cudaDeviceSynchronize();
// }

// template<typename T>
// inline void gpuSub(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
//     if (!A.isSameSize(B) || !A.isSameSize(C)) throw std::invalid_argument("Invalid matrix for subtraction");
//     dim3 blockSize(16,16);
//     dim3 gridSize((A.cols() + 15) / 16, (A.rows() + 15) / 16);
//     matrixSubKernel<T><<<gridSize, blockSize>>>(A, B, C);
//     cudaDeviceSynchronize();
// }

// template<typename T>
// inline Matrix<T> gpuTranspose(const Matrix<T>& A) {
//     Matrix<T> result(A.cols(), A.rows());
//     dim3 blockSize(16,16);
//     dim3 gridSize((A.cols() + 15) / 16, (A.rows() + 15) / 16);
//     matrixTransposeKernel<T><<<gridSize, blockSize>>>(A, result);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
//     }
//     cudaDeviceSynchronize();
//     return result;
// }