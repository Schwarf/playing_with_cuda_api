/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

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

template<typename T>
T * allocate_host_memory(size_t number_of_elements)
{
	T * memory = new T[number_of_elements];
	return memory;
}


template<typename T>
T * allocate_device_memory(size_t number_of_elements)
{
	cudaError_t error = cudaSuccess;
	auto size = number_of_elements*sizeof(T);
	T * device_memory = nullptr;
	error = cudaMalloc((void **)&device_memory, size);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	return device_memory;
}

template <typename T>
void copy_host_array_to_device_array(T * host_memory, T* device_memory, size_t number_of_elements)
{
	cudaError_t error = cudaSuccess;
	auto size = number_of_elements*sizeof(T);
	error = cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice);

}


/**
 * Host main routine
 */
int main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	size_t numElements = 50000;
	auto size = numElements*sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host vectors
	auto array_A = allocate_host_memory<float>(numElements);
	auto array_B = allocate_host_memory<float>(numElements);
	auto array_C = allocate_host_memory<float>(numElements);

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		array_A[i] = rand()/(float)RAND_MAX;
		array_B[i] = rand()/(float)RAND_MAX;
	}

	auto device_array_A = allocate_device_memory<float>(numElements);
	auto device_array_B = allocate_device_memory<float>(numElements);
	auto device_array_C = allocate_device_memory<float>(numElements);


	printf("Copy input data from the host memory to the CUDA device\n");
	copy_host_array_to_device_array<float>(array_A, device_array_A, numElements);
	copy_host_array_to_device_array<float>(array_B, device_array_B, numElements);
	copy_host_array_to_device_array<float>(array_C, device_array_C, numElements);


	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd<float><<<blocksPerGrid, threadsPerBlock>>>(device_array_A, device_array_B, device_array_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(array_C, device_array_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(array_A[i] + array_B[i] - array_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");

	// Free device global memory
	err = cudaFree(device_array_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_array_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_array_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	delete[] array_A;
	delete[] array_B;
	delete[] array_C;


	printf("Done\n");
	return 0;
}
