cmake_minimum_required(VERSION 3.21)
project(playing_with_cuda_api CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 99)

add_executable(vector_addition vector_addition.cu)
add_executable(shared_memory_basics shared_memory.cu)
add_executable(stencils stencils.cu )
add_executable(hello_thread hello_thread.cu)

add_subdirectory(algorithms)
add_subdirectory(basics)

set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shared_memory_basics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stencils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(hello_thread PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
