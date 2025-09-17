cmake_minimum_required(VERSION 3.18)

project(AddVect LANGUAGES CUDA CXX)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_addvect main.cu)

target_compile_features(test_addvect PRIVATE cxx_std_20)
target_compile_options(test_addvect PRIVATE -Wall -O3)


target_include_directories(test_addvect PRIVATE
    ${PROJECT_SOURCE_DIR}/сore/includes
)

target_link_libraries(test_addvect PRIVATE gpu_tasks gtest_main)