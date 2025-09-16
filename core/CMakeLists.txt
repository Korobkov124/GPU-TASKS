cmake_minimum_required(VERSION 3.18)
project(gpu_tasks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(gpu_tasks SHARED
    src/src1.cu
)

target_include_directories(gpu_tasks PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/includes"
)