#pragma once


namespace AddVect{

    std::size_t blockSize = 128;

    std::size_t GetGridSize(std::size_t vectSize);

    __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect);

    void FullGpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize, float* kernel_ms);

    static float* CpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize);

    static float* RunGpu(float* vect1, float* vect2, float* resultVect, std::size_t vectSize);
}