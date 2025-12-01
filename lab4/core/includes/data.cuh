#pragma once
#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

template<typename T>
class Data {
private:
    T* data_;
    std::size_t size_;
public:
    Data(std::size_t size) : size_(size) {cudaMallocManaged(&data_, size * sizeof(T));}

    ~Data() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;

    T* data() {return data_;}
    const T* data() const {return data_;}
    std::size_t size() const {return size_;}

    void copyToHost(T* hostPtr) const {
        cudaMemcpy(hostPtr, &data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
    }

    void copyFromHost(const T*hostPtr) const {
        cudaMemcpy(&data_, hostPtr, size_ * sizeof(T), cudaMemcpyHostToDevice);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
    }
};