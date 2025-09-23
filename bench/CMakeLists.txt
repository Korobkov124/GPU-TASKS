cmake_minimum_required(VERSION 3.18)
project(bench LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.9.2
)
FetchContent_MakeAvailable(benchmark)

add_executable(bench main.cu)

target_include_directories(bench PRIVATE
    ${PROJECT_SOURCE_DIR}/сore/includes
)

target_link_libraries(bench PRIVATE gpu_tasks benchmark)
