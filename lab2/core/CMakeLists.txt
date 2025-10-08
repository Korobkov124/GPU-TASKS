cmake_minimum_required(VERSION 3.18)
project(matrix_tests LANGUAGES CUDA)


add_executable(123test 123test.cu)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF) 
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(matrix_core INTERFACE)
add_library(matrix::core ALIAS matrix_core)

set_target_properties(123test PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(123test PRIVATE includes)

target_link_libraries(123test PRIVATE 
    cudart
    cudadevrt
)

set(CMAKE_CUDA_STANDARD 17)