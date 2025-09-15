cmake_minimum_required(VERSION 3.18)
project(GPU_TASKS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(src1 src/src1.cu)
