#pragma once


namespace AddVect{

    static __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect, int N);

    class AddingVectors{
        public:
            static float* CpuAddVect(float* vect1, float* vect2, float* resultVect, int N);

            static float* RunGpu(float* vect1, float* vect2, float* resultVect, int N);
    };
}