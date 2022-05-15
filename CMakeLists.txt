cmake_minimum_required(VERSION 3.21)
project(playing_with_cuda_api CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 99)

add_executable(vector_addition vector_addition.cu helper/memory_tracker.h)
add_executable(parallelize_prefix_sums parallelize_prefix_sums.cu helper/memory_tracker.h)
add_subdirectory(algorithms)
set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(parallelize_prefix_sums PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
