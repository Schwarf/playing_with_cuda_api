//
// Created by andreas on 31.05.22.
//

#include "algorithms/parallel_reduction_samples.cuh"
#include "helper/memory_tracker.cuh"
#include <iostream>
#include <type_traits>
#include <random>
#include <cuda_runtime.h>
constexpr int MAX_THREADS = 1024;

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
static inline void get_random_integral(const T &lower_bound, const T &upper_bound, T *output, int output_size)
{
	static std::mt19937 generator;
	auto int_distribution = std::uniform_int_distribution<T>(lower_bound, upper_bound);
	for (int i = 0; i < output_size; i++) {
		output[i] = int_distribution(generator);
	}
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
static inline void get_random_float(const T &lower_bound, const T &upper_bound, T *output, int output_size)
{
	static std::mt19937 generator;
	auto real_distribution = std::uniform_real_distribution<T>(lower_bound, upper_bound);
	for (int i = 0; i < output_size; i++) {
		output[i] = real_distribution(generator);
	}
}
template<typename T>
T linear_sum(T *input, int size)
{
	T result{};
	for (int i = 0; i < size; ++i) {
		result += input[i];
	}
	return result;
}

int next_power_of_2(int n)
{
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}

int2 compute_number_of_threads_and_blocks(int input_size)
{
	int threads = (input_size < MAX_THREADS) ? next_power_of_2(input_size) : MAX_THREADS;
	int blocks = (input_size + threads - 1)/threads;
	return make_int2(threads, blocks);
}

int main()
{
	constexpr size_t size = 32;
	auto threads_and_blocks = compute_number_of_threads_and_blocks(size);
	int shared_memory_size = (threads_and_blocks.x <= 32) ? 2 * threads_and_blocks.x * sizeof(float) : threads_and_blocks.x * sizeof(float);
	int output_size = std::min<int>(size/MAX_THREADS, 65000);
	auto tracker = MemoryTracker<float>();
	auto host_input = tracker.allocate_host_memory(size, "host_input");
	auto host_output = tracker.allocate_host_memory(output_size, "host_output");
	get_random_float<float>(0.f, 3.f, host_input, size);
	auto host_expected_output = linear_sum(host_input, size);

	auto device_input = tracker.allocate_device_memory(size, "device_input");
	auto device_output = tracker.allocate_device_memory(output_size, "device_output");
	tracker.copy_host_array_to_device_array(host_input, device_input, size);

	parallel_array_sum_v0<float><<<threads_and_blocks.x, threads_and_blocks.y, shared_memory_size>>>(device_input, device_output, size);
	auto error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kth_stencil kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	tracker.copy_device_array_to_host_array(host_output, device_output, 1);

	std::cout << host_expected_output << "  " << host_output[0] << std::endl;
	tracker.free_device_memory(device_output);
	tracker.free_device_memory(device_input);
	tracker.free_host_memory(host_output);
	tracker.free_host_memory(host_input);
	return 0;





}