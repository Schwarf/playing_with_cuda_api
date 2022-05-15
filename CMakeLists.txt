cmake_minimum_required(VERSION 3.21)
project(playing_with_cuda_api CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 99)

add_executable(cuda_api memory_tracking_and_vector_addition.cu helper/memory_tracking.h)

add_subdirectory(algorithms)
set_target_properties(cuda_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
