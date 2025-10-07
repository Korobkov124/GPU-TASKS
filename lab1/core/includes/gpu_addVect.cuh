#pragma once


namespace AddVect{

    std::size_t blockSize = 128;

    std::size_t GetGridSize(std::size_t vectSize);

    __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize);

    void FullGpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize);

    float* CpuAddVect(float* vect1, float* vect2, float* resultVect, std::size_t vectSize);

    float* RunGpu(float* vect1, float* vect2, float* resultVect, std::size_t vectSize);
}