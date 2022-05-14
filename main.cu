
#include <cstdio>
#include <iostream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "helper/memory_tracking.h"

template <typename T>
__global__ void
vectorAdd(const T *A, const T *B, T *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

template <typename T>
void copy_host_array_to_device_array(T * host_memory, T* device_memory, size_t number_of_elements)
{
	cudaError_t error = cudaSuccess;
	auto size = number_of_elements*sizeof(T);
	error = cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void copy_device_array_to_host_array(T * host_memory, T* device_memory, size_t number_of_elements)
{
	cudaError_t error = cudaSuccess;
	auto size = number_of_elements*sizeof(T);
	error = cudaMemcpy(host_memory, device_memory, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

}




int main()
{
	// Error code to check return values for CUDA calls
	cudaError_t error = cudaSuccess;

	// Print the vector length to be used, and compute its size
	size_t number_of_elements = 50000;
	std::cout << "Vector addition of "<< number_of_elements << " elements \n";

	// Allocate the host vectors
	auto tracker = MemoryTracking<float>();
	auto array_A = tracker.allocate_host_memory(number_of_elements);
	auto array_B = tracker.allocate_host_memory(number_of_elements);
	auto array_C = tracker.allocate_host_memory(number_of_elements);

	// Initialize the host input vectors
	for (int i = 0; i < number_of_elements; ++i)
	{
		array_A[i] = rand()/(float)RAND_MAX;
		array_B[i] = rand()/(float)RAND_MAX;
	}

	auto device_array_A = tracker.allocate_device_memory(number_of_elements);
	auto device_array_B = tracker.allocate_device_memory(number_of_elements);
	auto device_array_C = tracker.allocate_device_memory(number_of_elements);


	printf("Copy input data from the host memory to the CUDA device\n");
	copy_host_array_to_device_array<float>(array_A, device_array_A, number_of_elements);
	copy_host_array_to_device_array<float>(array_B, device_array_B, number_of_elements);
	copy_host_array_to_device_array<float>(array_C, device_array_C, number_of_elements);


	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(number_of_elements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd<float><<<blocksPerGrid, threadsPerBlock>>>(device_array_A, device_array_B, device_array_C, number_of_elements);
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	copy_device_array_to_host_array<float>(array_C, device_array_C, number_of_elements);

	// Verify that the result vector is correct
	for (int i = 0; i < number_of_elements; ++i)
	{
		if (fabs(array_A[i] + array_B[i] - array_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
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
