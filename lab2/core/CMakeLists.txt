cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF) 
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(matrix_core INTERFACE)
add_library(matrix::core ALIAS matrix_core)

set_target_properties(matrix_core PROPERTIES CUDA_ARCHITECTURES "75")

target_compile_options(matrix_core INTERFACE 
    --expt-relaxed-constexpr
)

target_include_directories(matrix_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/includes")
