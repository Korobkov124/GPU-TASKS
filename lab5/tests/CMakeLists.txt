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

add_executable(test_br_nobr src/main.cu)
set_target_properties(test_br_nobr PROPERTIES
    CUDA_ARCHITECTURES "75"
)

enable_testing()

target_include_directories(test_br_nobr PRIVATE
    ${eigen_SOURCE_DIR}
)

target_link_libraries(test_br_nobr PRIVATE vector_core gtest_main)