//
// Created by andreas on 31.07.22.
//

#include <curand.h>
#include "./../helper/memory_tracker.cuh"
int main()
{
	curandGenerator_t generator;
	unsigned long seed;
	curandStatus_t curand_result;
	cudaError_t cuda_result;
	MemoryTracker<float> tracker;
	auto host_sample = tracker.allocate_host_memory(100, "host_sample");
	auto device_sample = tracker.allocate_device_memory(100, "device_sample");

	return 0;
}
