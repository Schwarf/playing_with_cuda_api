//
// Created by andreas on 30.05.22.
//

#ifndef K_1D_STENCIL_CUH
#define K_1D_STENCIL_CUH
#include <iostream>

template <typename T>
__global__ void
kth_stencil(T * input, T * output, int input_size, int k)
{
	extern __shared__ T shared_memory[];
	// id of thread in block
	int local_thread_id = threadIdx.x;

	// first index of output-element by the block
	int start_of_thread_block = blockIdx.x * blockDim.y;

	// The id of the thread in the scope of the grid
	int globalId = local_thread_id + start_of_thread_block;

	if(globalId > input_size)
		return;

	// Fetching into shared memory
	shared_memory[local_thread_id] = input[globalId];
	if(local_thread_id < 2 && blockDim.x + globalId < input_size)
		shared_memory[blockDim.x + local_thread_id] = input[blockDim.x + globalId];

	__syncthreads();

	T denom= static_cast<T>(2*k+1);
	if(globalId < input_size - 2*k)
	{
		output[globalId] = shared_memory[local_thread_id];
		for(int i = 1 ; i < (2*k+1); ++i)
		{
			output[globalId] = shared_memory[local_thread_id + i];
		}
		output[globalId] /= denom;
	}

}

template<typename T>
T *kth_stencil(T *input, int input_size, int k)
{
	if (input_size < 2 * k)
		return nullptr;
	auto output_size = input_size - 2 * k;
	T denominator = static_cast<T>(2 * k + 1);
	T *stencil = new T[output_size + 1];
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
