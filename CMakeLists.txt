cmake_minimum_required(VERSION 3.21)
project(playing_with_cuda_api CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 99)

add_executable(vector_addition vector_addition.cu)
add_executable(shared_memory_basics shared_memory.cu)
add_executable(stencils stencils.cu )
add_executable(hello_thread hello_thread.cu random_using_device_api.cu)
add_executable(random_host random_using_host_api.cu)
add_executable(random_device random_using_device_api.cu)

add_subdirectory(algorithms)
add_subdirectory(basics)


set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shared_memory_basics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stencils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(hello_thread PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(random_host PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(random_device PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(random_host -lcublas -lcurand)
target_link_libraries(random_device -lcublas -lcurand)