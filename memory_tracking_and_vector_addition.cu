
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "helper/memory_tracking.h"
#include "algorithms/vector_addition.cuh"


int main()
{
	// Error code to check return values for CUDA calls
	cudaError_t error = cudaSuccess;

	// Print the vector length to be used, and compute its size
	size_t number_of_elements = 50000;
	std::cout << "Vector addition of "<< number_of_elements << " elements \n";

	// Allocate the host vectors
	auto tracker = MemoryTracking<int>();
	auto array_A = tracker.allocate_host_memory(number_of_elements);
	auto array_B = tracker.allocate_host_memory(number_of_elements);
	auto array_C = tracker.allocate_host_memory(number_of_elements);

	// Initialize the host input vectors
	for (int i = 0; i < number_of_elements; ++i)
	{
		array_A[i] = rand()/(int)RAND_MAX;
		array_B[i] = rand()/(int)RAND_MAX;
	}

	auto device_array_A = tracker.allocate_device_memory(number_of_elements);
	auto device_array_B = tracker.allocate_device_memory(number_of_elements);
	auto device_array_C = tracker.allocate_device_memory(number_of_elements);


	printf("Copy input data from the host memory to the CUDA device\n");
	tracker.copy_host_array_to_device_array(array_A, device_array_A, number_of_elements);
	tracker.copy_host_array_to_device_array(array_B, device_array_B, number_of_elements);
	tracker.copy_host_array_to_device_array(array_C, device_array_C, number_of_elements);


	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(number_of_elements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd<int><<<blocksPerGrid, threadsPerBlock>>>(device_array_A, device_array_B, device_array_C, number_of_elements);
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	tracker.copy_device_array_to_host_array(array_C, device_array_C, number_of_elements);

	// Verify that the result vector is correct
	for (int i = 0; i < number_of_elements; ++i)
	{
		if (fabs(array_A[i] + array_B[i] - array_C[i]) > 1e-5 || array_A[i] == 0)
		{
			std::cout << "ERROR in result." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	std::cout << "Test PASSED " <<std::endl;


	tracker.free_device_memory(device_array_A);
	tracker.free_device_memory(device_array_B);
	tracker.free_device_memory(device_array_C);

	tracker.free_host_memory(array_A);
	tracker.free_host_memory(array_B);
	tracker.free_host_memory(array_C);

	// Free host memory
	return 0;
}
