#pragma once
#include "data.cuh"
#include "vectorView.cuh"
#include <memory>
#include <stdexcept>
#include <iostream>
#include <random>
#include <chrono>

enum class AddAlgorithm {
    br,
    nobr
};

template<typename T, AddAlgorithm Algorithm = AddAlgorithm::nobr>
class Vector {
private:
    std::shared_ptr<Data<T>> data_;
    VectorView<T> view_;

    void randomFill(std::mt19937& gen) {
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(0.0, 1.0);
            for (std::size_t i = 0; i < size(); ++i) {
                data()[i] = dis(gen);
            }
        }else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(0, 100);
            for (std::size_t i = 0; i < size(); ++i) {
                data()[i] = dis(gen);
            }
        }else {
            for (std::size_t i = 0; i < size(); ++i) {
                data()[i] = T{};
            }
        }
    }
public:
    Vector(std::size_t size) : data_(std::make_shared<Data<T>>(size)), view_(data_->data(), size) {}
    Vector(T* externalData, std::size_t size) : data_(nullptr), view_(externalData, size) {}
    Vector(const Vector& other) = default;

    Vector& operator=(const Vector& other) {
        if(this != &other) {
            data_ = other.data_;
            view_ = VectorView<T>(other.view_.data(), other.size());
        }
        return *this;
    }

    ~Vector() = default;

    VectorView<T> view() const {return view_;}
    T* data() {return view_.data();}
    const T* data() const {return view_.data();}
    Data<T>& getData() {return *data_;}
    const Data<T>& getData() const {return *data_;}
    std::size_t size() const {return view_.size();}

    T& operator[](std::size_t index) {return view_[index];}
    const T& operator[](std::size_t index) const {return view_[index];}

    void fill(const T& value) {
        for (std::size_t i = 0; i < size(); ++i) {
            data()[i] = value;
        }
    }

    void random() {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        randomFill(gen);
    }

    void random(unsigned int seed) {
        std::mt19937 gen(seed);
        randomFill(gen);
    }

    void print(const char* name = "") const {
        std::cout << name << " [" << size() << "]:\n";
        for (std::size_t i = 0; i < size(); ++i) {
            std::cout << (*this)[i] << " ";
        }
        std::cout << std::endl;
    }

    T sum() const;
};