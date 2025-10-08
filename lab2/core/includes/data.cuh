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
};