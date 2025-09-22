#include <iostream>
#include "gpu_addVect.cuh"
#include <cuda_runtime.h>

namespace AddVect{

    __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect){
        int i = threadIdx.x;
        resultVect[i] = vect1[i] + vect2[i];
    };

    void FullGpuAddVect(float* vect1, float* vect2, float* resultVect, int gridSize, float* kernel_ms){
        float *devVect1, *devVect2, *devResult;
        cudaEvent_t start, stop;

        cudaMalloc((void**)&devVect1, sizeof(float) * gridSize);
        cudaMalloc((void**)&devVect2, sizeof(float) * gridSize);
        cudaMalloc((void**)&devResult, sizeof(float) * gridSize);

        cudaMemcpy(devVect1, vect1, sizeof(float) * gridSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devVect2, vect2, sizeof(float) * gridSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devResult, resultVect, sizeof(float) * gridSize, cudaMemcpyHostToDevice);
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        GpuAddVect <<< 1, gridSize >>>(devVect1, devVect2, devResult);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(kernel_ms, start, stop);

        cudaMemcpy(resultVect, devResult, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);
        cudaFree(devVect1);
        cudaFree(devVect2);
        cudaFree(devResult);
    };

    float* AddingVectors::CpuAddVect(float* vect1, float* vect2, float* resultVect, int gridSize){
        for(int i = 0; i < gridSize; i++){
            resultVect[i] = vect1[i] + vect2[i];
        }
        
        return resultVect;
    };

    float* AddingVectors::RunGpu(float* vect1, float* vect2, float* resultVect, int gridSize){
        float *devVect1, *devVect2, *devResult;

        cudaMalloc((void**)&devVect1, sizeof(float) * gridSize);
        cudaMalloc((void**)&devVect2, sizeof(float) * gridSize);
        cudaMalloc((void**)&devResult, sizeof(float) * gridSize);

        cudaMemcpy(devVect1, vect1, sizeof(float) * gridSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devVect2, vect2, sizeof(float) * gridSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devResult, resultVect, sizeof(float) * gridSize, cudaMemcpyHostToDevice);

        GpuAddVect <<< 1, gridSize >>>(devVect1, devVect2, devResult);

        cudaMemcpy(resultVect, devResult, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);
        
        cudaFree(devVect1);
        cudaFree(devVect2);
        cudaFree(devResult);
        return resultVect;
    };

};