#pragma once
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

template<typename T>
class Matrix {
private:
    T* data_;
    std::size_t rows_;
    std::size_t cols_;
    bool owns_data_;

public:
    __host__ __device__ Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), owns_data_(true) {
        allocateMemory();
    }
    
    __host__ __device__ Matrix(T* data, std::size_t rows, std::size_t cols) : data_(data), rows_(rows), cols_(cols), owns_data_(false) {}

    __host__ __device__ Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), owns_data_(true) {
        allocateMemory();
        copyData(other.data_);
    }

    __host__ __device__ ~Matrix() {
        if (owns_data_ && data_) {
            deAllocateMemory();
        }
    }

    __host__ __device__ Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                if (owns_data_ && data_) {
                    deAllocateMemory();
                }
                rows_ = other.rows_;
                cols_ = other.cols_;
                allocateMemory();
            }
            copyData(other.data_);
        }
        return *this;
    }

    __host__ __device__ std::size_t rows() const { return rows_; }
    __host__ __device__ std::size_t cols() const { return cols_; }
    __host__ __device__ std::size_t size() const { return rows_ * cols_; }
    __host__ __device__ T* data() { return data_; }
    __host__ __device__ const T* data() const { return data_; }
    __host__ __device__ bool isSameSize(const Matrix<T>& other) const { 
        return rows_ == other.rows_ && cols_ == other.cols_; 
    }

    __host__ __device__ T& operator()(std::size_t row, std::size_t col) {
        return data_[row * cols_ + col];
    }

    __host__ __device__ const T& operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols_ + col];
    }

    Matrix operator+(const Matrix<T>& other) const {
        if (!isSameSize(other)) throw std::invalid_argument("Invalid matrix for addition");
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < other.size(); i++) {
            result.data()[i] = data_[i] + other.data()[i];
        }
        return result;
    }

    Matrix operator-(const Matrix<T>& other) const {
        if (!isSameSize(other)) throw std::invalid_argument("Invalid matrix for subtraction");
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < other.size(); i++) {
            result.data()[i] = data_[i] - other.data()[i];
        }
        return result;
    }

    Matrix operator*(const Matrix<T>& other) const {
        if (cols_ != other.rows()) throw std::invalid_argument("Invalid matrix for multiply");
        Matrix<T> result(rows_, other.cols());
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < other.cols(); j++) {
                T sum = static_cast<T>(0);
                for(std::size_t k = 0; k < other.cols(); k++) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    Matrix& operator+=(const Matrix<T>& other) {
        if (!isSameSize(other)) throw std::invalid_argument("Invalid matrix for addition");
        for (std::size_t i = 0; i < size(); i++) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix<T>& other) {
        if (!isSameSize(other)) throw std::invalid_argument("Invalid matrix for subtraction");
        for (std::size_t i = 0; i < size(); i++) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Matrix& operator*=(const T scalar) {
        for (std::size_t i = 0; i < size(); i++) {
            data_[i] *= scalar;
        }
        return *this;
    }

    Matrix transpose() const {
        Matrix<T> result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    void print(const char* name = "") const {
        std::cout << name << " (" << rows_ << "x" << cols_ << "):\n";
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void fill(const T& value) {
        for (std::size_t i = 0; i < size(); i++) {
            data_[i] = value;
        }
    }

private:
    void allocateMemory() {
        #ifdef __CUDA_ARCH__
            data_ = new T[rows_ * cols_];
        #else 
            cudaMallocManaged(&data_, rows_ * cols_ * sizeof(T));
        #endif    
    }

    void deAllocateMemory() {
        #ifdef __CUDA_ARCH__
            delete[] data_;
        #else
            cudaFree(data_);
        #endif
        data_ = nullptr;
    }

    void copyData(const T* src) {
        for(std::size_t i = 0; i < size(); i++){
            data_[i] = src[i];
        }
    }
};