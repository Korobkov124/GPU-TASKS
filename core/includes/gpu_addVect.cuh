#pragma once


namespace AddVect{

    __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect);

    void FullGpuAddVect(float* vect1, float* vect2, float* resultVect, int gridSize, float* kernel_ms);

    class AddingVectors{
        public:
            static float* CpuAddVect(float* vect1, float* vect2, float* resultVect, int N);

            static float* RunGpu(float* vect1, float* vect2, float* resultVect, int N);
    };
}