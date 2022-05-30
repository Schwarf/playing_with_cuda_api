//
// Created by andreas on 30.05.22.
//

#ifndef K_1D_STENCIL_CUH
#define K_1D_STENCIL_CUH
#include <stdio.h>
#include <iostream>


template <typename T>
__global__ void
kth_stencil(T * input, T * output, int input_size, int k)
{

	extern __shared__ T shared_memory[];
	// id of thread in block
	int local_thread_id = threadIdx.x;

	// first index of output-element by the block
	int start_of_thread_block = blockIdx.x * blockDim.x;
	// The id of the thread in the scope of the grid
	int global_id = local_thread_id + start_of_thread_block;

	if(global_id >= input_size)
		return;

	// Fetching into shared memory
	shared_memory[local_thread_id] = input[global_id];
	if (local_thread_id < 2*k && blockDim.x + global_id < input_size) {
		shared_memory[blockDim.x + local_thread_id] = input[blockDim.x + global_id];
	}
	__syncthreads();

	T denominator = static_cast<T>(2*k + 1);
	if(global_id < input_size - 2*k)
	{
		output[global_id] = shared_memory[local_thread_id];
		for(int i = 1 ; i < (2*k+1); ++i)
		{
			output[global_id] += shared_memory[local_thread_id + i];
		}
		output[global_id] /= denominator;
	}
}

template <typename T, int k>
__global__ void
kth_stencil_warped(T * input, T * output, int input_size)
{
	T register_cache[2];
	constexpr int WARP_SIZE = 32;
	int local_thread_id = threadIdx.x % WARP_SIZE;
	int start_of_WARP = blockIdx.x * blockDim.x + WARP_SIZE*(threadIdx.x /WARP_SIZE);
	int global_id = local_thread_id + start_of_WARP;

	if(global_id >= input_size)
		return;
	register_cache[0] = input[global_id];
	if(global_id < 2*k && WARP_SIZE + global_id < input_size)
	{
		register_cache[1] = input[WARP_SIZE + global_id];
	}
	T denominator = static_cast<T>(2*k+1);
	T accumulated_sum_per_thread{};
	T shared = register_cache[0];
	for(int i=0; i < 2*k+1; ++i )
	{
		// Threads decide what value will be published in the following access.
		if (local_thread_id < i)
			shared = register_cache[1];
		unsigned mask = __activemask();
		accumulated_sum_per_thread += __shfl_sync(mask, shared, (local_thread_id + i) % WARP_SIZE);
	}
	if (global_id < input_size - 2*k)
		output[global_id] = accumulated_sum_per_thread/denominator;
}



template<typename T>
T *kth_stencil(T *input, int input_size, int k)
{
	if (input_size < 2 * k)
		return nullptr;
	auto output_size = input_size - 2 * k;
	T denominator = static_cast<T>(2 * k + 1);
	T *stencil = new T[output_size];
	int max_index = (2 * k + 1);
	for (int output_index = 0; output_index < output_size; output_index++) {
		stencil[output_index] = T{};

		int index{};
		while (index < max_index) {
			stencil[output_index] += input[output_index + index++];
		}
		stencil[output_index] /= denominator;
	}

	return stencil;
}




#endif //K_1D_STENCIL_CUH
