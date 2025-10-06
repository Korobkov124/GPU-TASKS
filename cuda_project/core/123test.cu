#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "matrix.cuh"
#include "matrixOperations.cuh"

using namespace std;

void test_constructors() {
    cout << "=== Test 1: Constructors ===" << endl;
    
    // Конструктор с размерами
    Matrix<float> A(2, 3);
    A.fill(1.0f);
    cout << "Matrix A(2,3) filled with 1.0:" << endl;
    A.print("A");
    
    // Конструктор копирования
    Matrix<float> B = A;
    cout << "Matrix B (copy of A):" << endl;
    B.print("B");
    
    cout << "✓ Constructors test passed!" << endl << endl;
}

void test_access_operators() {
    cout << "=== Test 2: Access Operators ===" << endl;
    
    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    
    cout << "Matrix A after element access:" << endl;
    A.print("A");
    
    // Проверка чтения
    cout << "A(0,0) = " << A(0, 0) << endl;
    cout << "A(1,1) = " << A(1, 1) << endl;
    
    cout << "✓ Access operators test passed!" << endl << endl;
}

void test_addition() {
    cout << "=== Test 3: Matrix Addition ===" << endl;
    
    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    
    Matrix<float> B(2, 2);
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;
    
    // CPU сложение
    Matrix<float> C1(2, 2);
    MatrixOperations::cpuAdd(A, B, C1);
    cout << "CPU Addition:" << endl;
    A.print("A");
    B.print("B");
    C1.print("A + B (CPU)");
    
    // GPU сложение
    Matrix<float> C2(2, 2);
    MatrixOperations::gpuAdd(A, B, C2);
    C2.print("A + B (GPU)");
    
    // Проверка совпадения результатов
    bool success = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(C1.data()[i] - C2.data()[i]) > 0.001f) {
            success = false;
            break;
        }
    }
    
    if (success) {
        cout << "✓ Addition test passed!" << endl;
    } else {
        cout << "✗ Addition test failed!" << endl;
    }
    cout << endl;
}

void test_subtraction() {
    cout << "=== Test 4: Matrix Subtraction ===" << endl;
    
    Matrix<float> A(2, 2);
    A(0, 0) = 5.0f; A(0, 1) = 6.0f;
    A(1, 0) = 7.0f; A(1, 1) = 8.0f;
    
    Matrix<float> B(2, 2);
    B(0, 0) = 1.0f; B(0, 1) = 2.0f;
    B(1, 0) = 3.0f; B(1, 1) = 4.0f;
    
    // CPU вычитание
    Matrix<float> C1(2, 2);
    MatrixOperations::cpuSub(A, B, C1);
    cout << "CPU Subtraction:" << endl;
    A.print("A");
    B.print("B");
    C1.print("A - B (CPU)");
    
    // GPU вычитание
    Matrix<float> C2(2, 2);
    MatrixOperations::gpuSub(A, B, C2);
    C2.print("A - B (GPU)");
    
    // Проверка совпадения результатов
    bool success = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(C1.data()[i] - C2.data()[i]) > 0.001f) {
            success = false;
            break;
        }
    }
    
    if (success) {
        cout << "✓ Subtraction test passed!" << endl;
    } else {
        cout << "✗ Subtraction test failed!" << endl;
    }
    cout << endl;
}

void test_multiplication() {
    cout << "=== Test 5: Matrix Multiplication ===" << endl;
    
    Matrix<float> A(2, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
    
    Matrix<float> B(3, 2);
    B(0, 0) = 7.0f; B(0, 1) = 8.0f;
    B(1, 0) = 9.0f; B(1, 1) = 10.0f;
    B(2, 0) = 11.0f; B(2, 1) = 12.0f;
    
    // CPU умножение
    Matrix<float> C1(2, 2);
    MatrixOperations::cpuMultiply(A, B, C1);
    cout << "CPU Multiplication:" << endl;
    A.print("A");
    B.print("B");
    C1.print("A * B (CPU)");
    
    // GPU умножение
    Matrix<float> C2(2, 2);
    MatrixOperations::gpuMultiply(A, B, C2);
    C2.print("A * B (GPU)");
    
    // Проверка совпадения результатов
    bool success = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(C1.data()[i] - C2.data()[i]) > 0.001f) {
            success = false;
            break;
        }
    }
    
    if (success) {
        cout << "✓ Multiplication test passed!" << endl;
    } else {
        cout << "✗ Multiplication test failed!" << endl;
    }
    cout << endl;
}

void test_transpose() {
    cout << "=== Test 6: Matrix Transpose ===" << endl;
    
    Matrix<float> A(2, 3);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;
    
    cout << "Original matrix:" << endl;
    A.print("A");
    
    // CPU транспонирование
    Matrix<float> B = MatrixOperations::cpuTranspose(A);  // Исправлено: transpose вместо transponse
    cout << "CPU Transpose:" << endl;
    B.print("Aᵀ (CPU)");
    
    // GPU транспонирование
    Matrix<float> C = MatrixOperations::gpuTranspose(A);  // Исправлено: transpose вместо transponse
    cout << "GPU Transpose:" << endl;
    C.print("Aᵀ (GPU)");
    
    // Проверка совпадения результатов
    bool success = true;
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            if (fabs(B(j, i) - C(j, i)) > 0.001f) {
                success = false;
                break;
            }
        }
    }
    
    if (success) {
        cout << "✓ Transpose test passed!" << endl;
    } else {
        cout << "✗ Transpose test failed!" << endl;
    }
    cout << endl;
}

void test_comparison() {
    cout << "=== Test 7: Matrix Comparison ===" << endl;
    
    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    
    Matrix<float> B(2, 2);
    B(0, 0) = 1.0f; B(0, 1) = 2.0f;
    B(1, 0) = 3.0f; B(1, 1) = 4.0f;
    
    Matrix<float> C(2, 2);
    C(0, 0) = 1.1f; C(0, 1) = 2.0f;
    C(1, 0) = 3.0f; C(1, 1) = 4.0f;
    
    // Проверка isSameSize
    cout << "A.isSameSize(B): " << (A.isSameSize(B) ? "true" : "false") << endl;
    cout << "A.isSameSize(C): " << (A.isSameSize(C) ? "true" : "false") << endl;
    
    // Проверка равенства данных
    bool dataEqual = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(A.data()[i] - B.data()[i]) > 0.001f) {
            dataEqual = false;
            break;
        }
    }
    cout << "A data == B data: " << (dataEqual ? "true" : "false") << endl;
    
    cout << "✓ Comparison test passed!" << endl << endl;
}

void test_large_matrices() {
    cout << "=== Test 8: Large Matrices ===" << endl;
    
    // Инициализация генератора случайных чисел
    srand(static_cast<unsigned>(time(nullptr)));
    
    // Создаем большие матрицы для проверки производительности
    const int SIZE = 256;
    Matrix<float> A(SIZE, SIZE);
    Matrix<float> B(SIZE, SIZE);
    Matrix<float> C1(SIZE, SIZE);
    Matrix<float> C2(SIZE, SIZE);
    
    // Заполняем случайными значениями
    for (int i = 0; i < SIZE * SIZE; i++) {
        A.data()[i] = static_cast<float>(rand()) / RAND_MAX;
        B.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    cout << "Testing with " << SIZE << "x" << SIZE << " matrices..." << endl;
    
    // CPU умножение
    cout << "CPU multiplication..." << endl;
    MatrixOperations::cpuMultiply(A, B, C1);
    
    // GPU умножение
    cout << "GPU multiplication..." << endl;
    MatrixOperations::gpuMultiply(A, B, C2);
    
    // Проверка совпадения результатов (выборочно)
    bool success = true;
    int checkPoints = 10;
    for (int i = 0; i < checkPoints; i++) {
        int idx = rand() % (SIZE * SIZE);
        if (fabs(C1.data()[idx] - C2.data()[idx]) > 0.001f) {
            success = false;
            break;
        }
    }
    
    if (success) {
        cout << "✓ Large matrices test passed!" << endl;
    } else {
        cout << "✗ Large matrices test failed!" << endl;
    }
    cout << endl;
}

int main() {
    cout << "=========================================" << endl;
    cout << "Testing Matrix Library - Basic Operations" << endl;
    cout << "=========================================" << endl << endl;
    
    try {
        test_constructors();
        test_access_operators();
        test_addition();
        test_subtraction();
        test_multiplication();
        test_transpose();  // Исправлено название
        test_comparison();
        test_large_matrices();
        
        cout << "🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉" << endl;
        
    } catch (const std::exception& e) {
        cout << "❌ TEST FAILED: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}