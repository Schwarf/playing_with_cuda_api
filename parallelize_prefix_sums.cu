//
// Created by andreas on 15.05.22.
//


#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "helper/memory_tracker.h"
#include "algorithms/vector_addition.cuh"

template <typename T>
void sequential_prefix_sum(T * input, T * output, int length)
{
	output[0] = input[0];
	for(int index = 1; index < length; ++index )
	{
		output[index] = input[index] + output[index-1];
	}
}

template <typename T>
void sequential_pre_scan(T * input, T * output, int length)
{
	output[0] = 0;
	for(int index = 1; index < length; ++index )
	{
		output[index] = input[index -1] + output[index-1];
	}
}


template <typename T>
__global__
void parallel_pre_scan(T * input, T * output, int length)
{
	__shared__ float *temp;
	int thread_id = threadIdx.x;
	int out{};
	int in{1};
	temp[out*length + thread_id] = (thread_id > 0) ? input[thread_id - 1] : 0;
	__syncthreads();

	for(int offset = 1; offset < length; offset*=2)
	{
		out = 1 - out;
		in = 1 - out;
		if(thread_id >= offset)
			temp[out*length + thread_id] += temp[in*length+thread_id - offset];
		else
			temp[out*length + thread_id] += temp[in*length+thread_id];
		__syncthreads();
	}
	output[thread_id] = temp[out*length+thread_id];
}


int main()
{
	// Error code to check return values for CUDA calls
	cudaError_t error = cudaSuccess;
	int number_of_elements = 10000;
	auto tracker = MemoryTracker<float>();
	auto host_input_array = tracker.allocate_host_memory(number_of_elements);
	auto host_output_array = tracker.allocate_host_memory(number_of_elements);
	sequential_prefix_sum(host_input_array, host_output_array, number_of_elements);
	for (int i = 0; i < number_of_elements; ++i)
	{
		host_input_array[i] = rand()/(float)RAND_MAX;
	}

	auto device_input_array = tracker.allocate_device_memory(number_of_elements);
	auto device_output_array = tracker.allocate_device_memory(number_of_elements);
	tracker.copy_host_array_to_device_array(host_input_array, device_input_array, number_of_elements);

	parallel_pre_scan<float><<<>>>()

}