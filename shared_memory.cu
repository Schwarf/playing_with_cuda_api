//
// Created by andreas on 15.05.22.
//
#include <iostream>
#include <cuda_runtime.h>
#include "helper/memory_tracker.cuh"
#include "basics/shared_memory_kernels.cuh"


int main()
{
	constexpr int n = 1024;
	const int threads_per_block = 1024;
	int input[n], expected_output[n], output_static[n], output_dynamic[n];
	for (int i = 0; i < n; ++i) {
		input[i] = 1 + i;
		expected_output[n - i - 1] = 1 + i;
	}
	auto tracker = MemoryTracker<int>();
	auto device_input = tracker.allocate_device_memory(n, "d_input");
	tracker.copy_host_array_to_device_array(input, device_input, n);
	const int blocks = (n + threads_per_block - 1)/threads_per_block;
	reverse_array_static_shared_memory<int, n><<<1, threads_per_block>>>(device_input, n);
	tracker.copy_device_array_to_host_array(output_static, device_input, n);
	for (int i = 0; i < n; ++i) {
		if (output_static[i] == 0 || output_static[i] != expected_output[i])
			std::cerr << "ERROR: " << output_static[i] << " vs " << expected_output[i] << "\n";
	}

	// dynamic case
	auto device_input2 = tracker.allocate_device_memory(n, "d_input2");
	tracker.copy_host_array_to_device_array(input, device_input2, n);
	reverse_array_dynamic_shared_memory<int><<<1, threads_per_block, n * sizeof(int)>>>(device_input2, n);
	tracker.copy_device_array_to_host_array(output_dynamic, device_input2, n);
	for (int i = 0; i < n; ++i) {
		if (output_dynamic[i] == 0 || output_dynamic[i] != expected_output[i])
			std::cerr << "ERROR: " << output_dynamic[i] << " vs " << expected_output[i] << "\n";
	}

	return 0;
}