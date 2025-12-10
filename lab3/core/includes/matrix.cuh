#pragma once
#include "data.cuh"
#include "matrixView.cuh"
#include "matrixOperationsKernel.cuh"
#include <memory>
#include <stdexcept>
#include <iostream>

enum class MultiplyAlgorithm {
    naive,
    shared
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

    template<MultiplyAlgorithm OtherAlgorithm>
    Matrix(const Matrix<T, OtherAlgorithm>& other) : data_(std::make_shared<Data<T>>(other.rows() * other.cols())), view_(data_->data(), other.rows(), other.cols()) {
        for (std::size_t i = 0; i < size(); ++i) {
            data()[i] = other.data()[i];
        }
    }

    Matrix& operator=(const Matrix& other) {
        if(this != &other) {
            data_ = other.data_;
            view_ = MatrixView<T>(other.view_.data(), other.view_.rows(), other.view_.cols());
        }
        return *this;
    }

    ~Matrix() = default;

    MatrixView<T> view() const {return view_;}
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
            data()[i] = value;
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
};
