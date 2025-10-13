cmake_minimum_required(VERSION 3.18)

include(FetchContent)

FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.9.2
)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4
    )
FetchContent_MakeAvailable(Eigen)


add_executable(bench src/main.cu)

set_target_properties(bench PROPERTIES
    CUDA_ARCHITECTURES "75"
)

target_link_libraries(bench PRIVATE matrix_core benchmark cuda_timer Eigen3::Eigen)