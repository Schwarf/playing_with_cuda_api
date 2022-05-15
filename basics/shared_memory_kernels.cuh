//
// Created by andreas on 15.05.22.
//

#ifndef SHARED_MEMORY_KERNELS_CUH
#define SHARED_MEMORY_KERNELS_CUH

template<typename T, size_t memory_size>
__global__ void reverse_array_static_shared_memory(T * input_array, int size)
{
	__shared__ T shared[memory_size];
	auto thread_id = threadIdx.x;
	auto reversed_thread_id = size - thread_id -1;
	shared[thread_id] = input_array[thread_id];
	__syncthreads();
	input_array[thread_id] = shared[reversed_thread_id];
}


template<typename T>
__global__ void reverse_array_dynamic_shared_memory(T * input_array, int size)
{
	extern __shared__ T shared[];
	auto thread_id = threadIdx.x;
	auto reversed_thread_id = size - thread_id -1;
	shared[thread_id] = input_array[thread_id];
	__syncthreads();
	input_array[thread_id] = shared[reversed_thread_id];
}

#endif //SHARED_MEMORY_KERNELS_CUH
