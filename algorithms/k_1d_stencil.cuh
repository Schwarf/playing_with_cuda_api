//
// Created by andreas on 30.05.22.
//

#ifndef K_1D_STENCIL_CUH
#define K_1D_STENCIL_CUH
#include <cstdio>
#include <iostream>


template <typename T, int k>
__global__ void
kth_stencil_warped(T * input, T * output, int input_size)
{
	T register_cache[2];
	const int local_thread_id = threadIdx.x % warpSize;
	const int start_of_WARP = blockIdx.x * blockDim.x + warpSize*(threadIdx.x /warpSize);
	const int global_id = local_thread_id + start_of_WARP;

	if(global_id >= input_size)
		return;
	register_cache[0] = input[global_id];
	if(local_thread_id < 2*k && warpSize + global_id < input_size)
	{
		register_cache[1] = input[warpSize + global_id];
	}
	const T denominator = static_cast<T>(2*k+1);
	T accumulated_sum_per_thread{};
	T shared = register_cache[0];
	for(int i=0; i < 2*k+1; ++i )
	{
		// Threads decide what value will be published in the following access.
		if (local_thread_id < i)
			shared = register_cache[1];
		unsigned mask = __activemask();
		accumulated_sum_per_thread += __shfl_sync(mask, shared, (local_thread_id + i) % warpSize);
	}
	if (global_id < input_size - 2*k)
		output[global_id] = accumulated_sum_per_thread/denominator;
}

template <typename T>
__global__ void
kth_stencil(T * input, T * output, int input_size, int k)
{

	extern __shared__ T shared_memory[];
	// index of thread in block
	const int local_thread_index = threadIdx.x;
	// The index of the thread in the scope of the grid (respecting different blocks)
	const int global_index = local_thread_index + blockIdx.x * blockDim.x;

	if(global_index >= input_size)
		return;

	// load parts of the input into local shared memory
	shared_memory[local_thread_index] = input[global_index];
	//
	if (local_thread_index < 2*k && blockDim.x + global_index < input_size) {
		shared_memory[blockDim.x + local_thread_index] = input[blockDim.x + global_index];
	}
	__syncthreads();

	T denominator = static_cast<T>(2*k + 1);
	if(global_index < input_size)
	{
		output[global_index] = shared_memory[local_thread_index];
		for(int i = 1 ; i < (2*k+1); ++i)
		{
			output[global_index] += shared_memory[local_thread_index + i];
		}
		output[global_index] /= denominator;
	}
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
