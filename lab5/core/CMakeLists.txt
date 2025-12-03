cmake_minimum_required(VERSION 3.18)

project(vector_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF) 
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(vector_core INTERFACE)
add_library(vector::core ALIAS vector_core)

set_target_properties(vector_core PROPERTIES CUDA_ARCHITECTURES "86")

target_compile_options(vector_core INTERFACE 
    --expt-relaxed-constexpr
    --expt-extended-lambda                        
)

target_include_directories(vector_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/includes")

add_executable(123test 123test.cu)

# Устанавливаем архитектуру для исполняемого файла
set_target_properties(123test PROPERTIES CUDA_ARCHITECTURES "86")

# Настраиваем опции компиляции для исполняемого файла
target_compile_options(123test PRIVATE
    --expt-relaxed-constexpr
    --expt-extended-lambda
)

# Если исполняемый файл использует вашу библиотеку
target_link_libraries(123test PRIVATE vector::core)