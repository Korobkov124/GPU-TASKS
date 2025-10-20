cmake_minimum_required(VERSION 3.18)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4
    )
FetchContent_MakeAvailable(Eigen)

add_executable(test_matrix src/main.cu)
set_target_properties(test_matrix PROPERTIES
    CUDA_ARCHITECTURES "75"
)

enable_testing()

target_include_directories(test_matrix PRIVATE
    ${eigen_SOURCE_DIR}
)

target_link_libraries(test_matrix PRIVATE matrix_core gtest_main ${EIGEN3_INCLUDE_DIR})