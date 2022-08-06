//
// Created by andreas on 31.07.22.
//

#include "curand.h"
#include <stdexcept>
#include "helper/memory_tracker.cuh"
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      return EXIT_FAILURE;}} while(0)

int main()
{
	curandGenerator_t generator;
	unsigned long seed;
	curandStatus_t curand_result;
	cudaError_t cuda_result;
	MemoryTracker<unsigned long long> tracker;
	auto host_sample = tracker.allocate_host_memory(100, "host_sample");
	auto device_sample = tracker.allocate_device_memory(100, "device_sample");
	auto error = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
	if (error != CURAND_STATUS_SUCCESS)
	{
		std::string msg("Could not create pseudo-random number generator: ");
		msg += error;
		throw std::runtime_error(msg);
	}
	error = curandGenerateLongLong(generator, device_sample, 100);
	if (error != CURAND_STATUS_SUCCESS)
	{
		std::string msg("Could not run random number generator: ");
		msg += error;
		throw std::runtime_error(msg);
	}

	return 0;
}
