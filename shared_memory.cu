//
// Created by andreas on 15.05.22.
//
#include <iostream>
#include <cuda_runtime.h>
#include "helper/memory_tracker.cuh"
#include "basics/shared_memory_kernels.cuh"


int main()
{
	constexpr int n = 200;
	int input[n], expected_output[n], output_static[n], output_dynamic[n];
	for (int i = 0; i < n; ++i) {
		input[i] = 1 + i;
		expected_output[n - i - 1] = 1 + i;
	}
	auto tracker = MemoryTracker<int>();
	auto device_input = tracker.allocate_device_memory(n, "d_input");
	tracker.copy_host_array_to_device_array(input, device_input, n);
	reverse_array_static_shared_memory<int, n><<<1, n>>>(device_input, n);
	tracker.copy_device_array_to_host_array(output_static, device_input, n);
	for (int i = 0; i < n; ++i) {
		if (output_static[i] == 0 || output_static[i] != expected_output[i])
			std::cerr << "ERROR: " << output_static[i] << " vs " << expected_output[i] << "\n";
	}

	// dynamic cass
	auto device_input2 = tracker.allocate_device_memory(n, "d_input2");
	tracker.copy_host_array_to_device_array(input, device_input2, n);
	reverse_array_dynamic_shared_memory<int><<<1, n, n * sizeof(int)>>>(device_input2, n);
	tracker.copy_device_array_to_host_array(output_dynamic, device_input2, n);
	for (int i = 0; i < n; ++i) {
		if (output_dynamic[i] == 0 || output_dynamic[i] != expected_output[i])
			std::cerr << "ERROR: " << output_dynamic[i] << " vs " << expected_output[i] << "\n";
	}

	return 0;
}