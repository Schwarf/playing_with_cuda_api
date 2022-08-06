//
// Created by andreas on 31.07.22.
//

#include "curand.h"
#include <stdexcept>
#include "helper/memory_tracker.cuh"
#include "helper/macros.h"


int main()
{
	int n = 100;
	curandGenerator_t generator;
	unsigned long seed;
	curandStatus_t curand_result;
	cudaError_t cuda_result;
	MemoryTracker<float> tracker;
	auto host_sample = tracker.allocate_host_memory(n, "host_sample");
	auto device_sample = tracker.allocate_device_memory(n, "device_sample");
	CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));
/*	auto error = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
	if (error != CURAND_STATUS_SUCCESS)
	{
		std::string msg("Could not create pseudo-random number generator: ");
		msg += error;
		throw std::runtime_error(msg);
	}
 */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, 17891989ULL));
	CURAND_CALL(curandGenerateUniform(generator, device_sample, n));
	tracker.copy_device_array_to_host_array(host_sample, device_sample,n);
	int i{};
	while(i < n)
	{
		std::cout << host_sample[i] << std::endl;
		i++;
	}
	return 0;
}
