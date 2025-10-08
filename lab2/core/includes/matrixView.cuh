#pragma once
#include <cstddef>
#include <cuda_runtime.h>

template<typename T>
class MatrixView {
private:
    T* data_;
    std::size_t rows_;
    std::size_t cols_;
public:
    __host__ __device__ MatrixView() : data_(nullptr), rows_(0), cols_(0) {}
    __host__ __device__ MatrixView(T* data, std::size_t rows, std::size_t cols) : data_(data), rows_(rows), cols_(cols) {}
    __host__ __device__ MatrixView(const T* data, std::size_t rows, std::size_t cols) : data_(const_cast<T*>(data)), rows_(rows), cols_(cols) {}
    __host__ __device__ ~MatrixView() = default;
    __host__ __device__ MatrixView(const MatrixView&) = default;
    __host__ __device__ MatrixView& operator=(const MatrixView&) = default;

    __host__ __device__ T& operator()(std::size_t row, std::size_t col) {return data_[row * cols_ + col];}
    __host__ __device__ const T& operator()(std::size_t row, std::size_t col) const {return data_[row * cols_ + col];}

    __host__ __device__ T* data() {return data_;}
    __host__ __device__ const T* data() const {return data_;}

    __host__ __device__ std::size_t rows() const {return rows_;}
    __host__ __device__ std::size_t cols() const {return cols_;}
    __host__ __device__ std::size_t size() const {return rows_ * cols_;}

    __host__ __device__ bool isSameSize(const MatrixView<T>& other) const {return rows_ == other.rows_ && cols_ == other.cols_;}
};