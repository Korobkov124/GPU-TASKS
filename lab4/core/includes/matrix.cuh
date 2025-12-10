#pragma once
#include "data.cuh"
#include "matrixView.cuh"
#include "matrixOperationsKernel.cuh"
#include <memory>
#include <stdexcept>
#include <iostream>

enum class MultiplyAlgorithm {
    naive,
    shared,
    wmma
};

template<typename T, MultiplyAlgorithm Algorithm = MultiplyAlgorithm::shared>
class Matrix {
private:
    std::shared_ptr<Data<T>> data_;
    MatrixView<T> view_;
public:
    Matrix(std::size_t rows, std::size_t cols) : data_(std::make_shared<Data<T>>(rows * cols)), view_(data_->data(), rows, cols) {}
    Matrix(T* externalData, std::size_t rows, std::size_t cols) : data_(nullptr), view_(externalData, rows, cols) {}
    Matrix(const Matrix& other) = default;

    template<typename U, MultiplyAlgorithm OtherAlgorithm>
    Matrix(const Matrix<U, OtherAlgorithm>& other) : data_(std::make_shared<Data<T>>(other.rows() * other.cols())), view_(data_->data(), other.rows(), other.cols()) {
        for(size_t i = 0; i < other.size(); ++i) {
            if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, float>) {
                data()[i] = __float2half(other.data()[i]);
            } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half>) {
                data()[i] = __half2float(other.data()[i]);
            } else data()[i] = static_cast<T>(other.data()[i]); 
            
        }
    }

    template<typename U, MultiplyAlgorithm OtherAlgorithm>
    Matrix& operator=(const Matrix<U, OtherAlgorithm>& other) {
        if(rows() != other.rows() || cols() != other.cols()) {
            data_ = std::make_shared<Data<T>>(other.rows() * other.cols());
            view_ = MatrixView<T>(data_->data(), other.rows(), other.cols());
        }

        for(size_t i = 0; i < other.size(); ++i) {
            if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, float>) {
                data()[i] = __float2half(other.data()[i]);
            } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half>) {
                data()[i] = __half2float(other.data()[i]);
            } else data()[i] = static_cast<T>(other.data()[i]); 
            
        }
        return *this;
    }

    ~Matrix() = default;

    MatrixView<T> view() const {return view_;}
    const Data<T> getData() const {return data_;}
    T* data() {return view_.data();}
    const T* data() const {return view_.data();}
    std::size_t rows() const {return view_.rows();}
    std::size_t cols() const {return view_.cols();}
    std::size_t size() const {return view_.size();}

    T& operator()(std::size_t row, std::size_t col) {return view_(row, col);}
    const T& operator()(std::size_t row, std::size_t col) const {return view_(row, col);}

    bool isSameSize(const Matrix& other) const {return view_.isSameSize(other.view_);}

    void fill(const T& value) {
        for (std::size_t i = 0; i < size(); i++) {
            data()[i] = i;
        }
    }

    void print(const char* name = "") const {
        std::cout << name << " (" << rows() << "x" << cols() << "):\n";
        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t j = 0; j < cols(); ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    template<MultiplyAlgorithm OtherAlgorithm>
    Matrix operator*(const Matrix<T, OtherAlgorithm>& other) const;

    Matrix<__half, Algorithm> toHalf(){
        Matrix<__half, Algorithm> result(rows(), cols());
        for (std::size_t i = 0; i < size(); i++) {
            result.data()[i] = __float2half(static_cast<float>(data()[i]));
        }
        return result;
    }

    Matrix<float, Algorithm> toFloat(){
        Matrix<float, Algorithm> result(rows(), cols());
        for(std::size_t i = 0; i < size(); i++) {
            result.data()[i] = __half2float(data()[i]);
        }
        return result;
    }

    Data<T>& getData() { return *data_; }
    
    const Data<T>& getData() const { return *data_; }
};

std::ostream& operator<<(std::ostream& os, const __half& h) {
    os << __half2float(h);
    return os;
}