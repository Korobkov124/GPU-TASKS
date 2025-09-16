cmake_minimum_required(VERSION 3.18)

project(AddVect LANGUAGES CUDA CXX)

enable_testing()

add_subdirectory(googletest)

add_executable(test_addvect main.cpp)

target_link_libraries(test_addvect PUBLIC gpu_tasks gtest_main)