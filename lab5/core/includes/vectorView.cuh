#pragma once
#include <cstddef>
#include <cuda_runtime.h>

template<typename T>
class VectorView {
private:
    T* data_;
    std::size_t size_;
public:
    __host__ __device__ VectorView() : data_(nullptr), size_(0) {}
    __host__ __device__ VectorView(T* data, std::size_t size) : data_(data), size_(size) {}
    __host__ __device__ VectorView(const T* data, std::size_t size) : data_(const_cast<T*>(data)), size_(size) {}
    __host__ __device__ ~VectorView() = default;
    __host__ __device__ VectorView(const VectorView&) = default;
    __host__ __device__ VectorView& operator=(const VectorView&) = default;

    __host__ __device__ T& operator[](std::size_t index) {return data_[index];}
    __host__ __device__ const T& operator[](std::size_t index) const {return data_[index];}

    __host__ __device__ T* data() {return data_;}
    __host__ __device__ const T* data() const {return data_;}
    
    __host__ __device__ std::size_t size() const {return size_;}
};