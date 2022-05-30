//
// Created by andreas on 30.05.22.
//
#include "algorithms/k_1d_stencil.cuh"
#include "helper/memory_tracker.cuh"
#include <iostream>
#include <type_traits>
#include <random>
#include <cuda_runtime.h>

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
static inline T get_random_integral(const T &lower_bound, const T &upper_bound, T *output, int output_size)
{
	static std::mt19937 generator;
	auto int_distribution = std::uniform_int_distribution<T>(lower_bound, upper_bound);
	for (int i = 0; i < output_size; i++) {
		output[i] = int_distribution(generator);
	}
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
static inline T get_random_float(const T &lower_bound, const T &upper_bound, T *output, int output_size)
{
	static std::mt19937 generator;
	auto real_distribution = std::uniform_real_distribution<T>(lower_bound, upper_bound);
	for (int i = 0; i < output_size; i++) {
		output[i] = real_distribution(generator);
	}
}

int main()
{
	constexpr size_t size = 30;
	constexpr size_t k = 1;
	auto tracker = MemoryTracker<float>();
	auto host_input = tracker.allocate_host_memory(size, "host_input");
	auto host_output = tracker.allocate_host_memory(size-2*k, "host_output");
	get_random_float<float>(0.f, 1.f, host_input, size);
	auto host_expected_output = kth_stencil(host_input, size, k);
//	for (int i = 0; i < size - 2 * k; ++i) {
//		std::cout
//			<< (host_input[i] + host_input[i + 1] + host_input[i + 2] + host_input[i + 3] + host_input[i + 4]) / 5.f
//			<< "  " << host_expected_output[i] << std::endl;
		//std::cout << (A[i] + A[i+1] + A[i+2] )/3 / B[i] << std::endl;
//	}

	auto device_input = tracker.allocate_device_memory(size, "device_input");
	auto device_output = tracker.allocate_device_memory(size-2*k, "device_output");
	tracker.copy_host_array_to_device_array(host_input, device_input, size);

	int threadsPerBlock = 1024;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	int needed_shared_memory = (threadsPerBlock + 2) *sizeof(float);
	kth_stencil_warped<float, k><<<blocksPerGrid, threadsPerBlock, needed_shared_memory>>>(device_input, device_output, size);
	auto error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kth_stencil kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	tracker.copy_device_array_to_host_array(host_output, device_output, size-2*k);

	for (int i = 0; i < size - 2 * k; ++i) {
		if(std::fabs(1.f-host_expected_output[i]/host_output[i]) > 0.001f)
		{
			std::cout
				<< i << "  " << host_expected_output[i]/host_output[i] << "  " << host_output[i] << "  " << host_expected_output[i]  << std::endl;

		}
	}
	tracker.free_device_memory(device_output);
	tracker.free_device_memory(device_input);
	tracker.free_host_memory(host_output);
	tracker.free_host_memory(host_input);
}