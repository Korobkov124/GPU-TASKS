#include <iostream>
#include "gpu_addVect.cuh"
#include "cuda_timer.hpp"

namespace AddVect{

    std::size_t GetGridSize(std::size_t vectSize) {
        return (vectSize + blockSize - 1) / blockSize;
    }

    __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < vectSize) resultVect[i] = vect1[i] + vect2[i];
    };

    void FullGpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize){
        std::size_t gridSize = GetGridSize(vectSize);
        GpuAddVect <<< blockSize, gridSize >>>(vect1, vect2, resultVect, vectSize);

    };

    float* CpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize){
        for(int i = 0; i < vectSize; i++){
            resultVect[i] = vect1[i] + vect2[i];
        }
        
        return resultVect;
    };

    float* RunGpu(float* vect1, float* vect2, float* resultVect, std::size_t vectSize){
        std::size_t gridSize = GetGridSize(vectSize);
        float *devVect1, *devVect2, *devResult;
        
        cudaMalloc(&devVect1, sizeof(float) * vectSize);
        cudaMalloc(&devVect2, sizeof(float) * vectSize);
        cudaMalloc(&devResult, sizeof(float) * vectSize);

        cudaMemcpy(devVect1, vect1, sizeof(float) * vectSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devVect2, vect2, sizeof(float) * vectSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devResult, resultVect, sizeof(float) * vectSize, cudaMemcpyHostToDevice);

        GpuAddVect <<< blockSize, gridSize >>>(devVect1, devVect2, devResult, vectSize);

        cudaMemcpy(resultVect, devResult, sizeof(float) * vectSize, cudaMemcpyDeviceToHost);
        
        cudaFree(devVect1);
        cudaFree(devVect2);
        cudaFree(devResult);
        return resultVect;
    };

};