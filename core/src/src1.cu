#include <iostream>
#include <gpu_addVect.cuh>

namespace AddVect{

    __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect, int N){
        int i = threadIdx.x;
        resultVect[i] = vect1[i] + vect2[i];
    };

    float* AddingVectors::CpuAddVect(float* vect1, float* vect2, float* resultVect, int N){
        for(int i = 0; i < N; i++){
            resultVect[i] = vect1[i] + vect2[i];
        }
        
        return resultVect;
    };

    float* AddingVectors::RunGpu(float* vect1, float* vect2, float* resultVect, int N){
        float *devVect1, *devVect2, *devResult;

        cudaMalloc((void**)&devVect1, sizeof(float) * N);
        cudaMalloc((void**)&devVect2, sizeof(float) * N);
        cudaMalloc((void**)&devResult, sizeof(float) * N);

        cudaMemcpy(devVect1, vect1, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(devVect2, vect2, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(devResult, resultVect, sizeof(float) * N, cudaMemcpyHostToDevice);

        GpuAddVect <<< 1, N >>>(devVect2, devVect2, resultVect, N);

        cudaMemcpy(resultVect, devResult, sizeof(float), cudaMemcpyDeviceToHost);

        
        cudaFree(devVect2);
        cudaFree(devVect2);
        cudaFree(devResult);
        return resultVect;
    };

};