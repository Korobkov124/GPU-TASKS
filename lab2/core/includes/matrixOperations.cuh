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

}