cmake_minimum_required(VERSION 3.18)

project(AddVect LANGUAGES CUDA CXX)

enable_testing()

add_subdirectory(googletest)

add_executable(test_addvect main.cu)

target_include_directories(test_addvect PRIVATE
    ${PROJECT_SOURCE_DIR}/сore/includes
)

target_link_libraries(test_addvect PRIVATE gpu_tasks gtest_main)