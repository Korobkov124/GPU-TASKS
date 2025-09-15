#pragma once

namespace AddVect{
    class AddingVectors{
        static __global__ void GpuAddVect(float* vect1, float* vect2, float* resultVect, int N);

        public:
            static void CpuAddVect(float* vect1, float* vect2, float* resultVect, int N);
    };
}