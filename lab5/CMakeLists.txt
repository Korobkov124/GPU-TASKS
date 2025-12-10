cmake_minimum_required(VERSION 3.18)
project(lab4 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(core)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(utils)