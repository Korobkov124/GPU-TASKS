#include <iostream>
#include <gpu_addVect.cuh>

namespace AddVect{

__global__ void AddingVectors::GpuAddVect(float* vect1, float* vect2, float* resultVect, int N){
    int i = threadIdx.x;
    resultVect[i] = vect1[i] + vect2[i];
};

void AddingVectors::CpuAddVect(float* vect1, float* vect2, float* resultVect, int N){
    for(int i = 0; i < N; i++){
        resultVect[i] = vect1[i] + vect2[i];
    }
}

};