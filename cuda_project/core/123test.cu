#include "matrix.cuh"
#include "matrixOperations.cuh"

int main() {
    try {

        Matrix<float> A(2, 3);
        Matrix<float> B(3, 2);
        Matrix<float> D(2, 3);
        Matrix<float> E(2, 3);

        D.fill(3.0f);
        E.fill(2.0f);

        A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
        A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

        B(0, 0) = 7.0f; B(0, 1) = 8.0f;
        B(1, 0) = 9.0f; B(1, 1) = 10.0f;
        B(2, 0) = 11.0f; B(2, 1) = 12.0f;


        Matrix<float> C = A * B;


        A.print("Matrix A");
        B.print("Matrix B");
        C.print("Result A * B");

        D = D + E;
        D.print("A + B");
        C = D - E;
        C.print("A - B");
        

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}