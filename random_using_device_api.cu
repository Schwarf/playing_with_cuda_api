//
// Created by andreas on 06.08.22.
//

#include <cstdlib>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>
#include "helper/memory_tracker.cuh"
#include "helper/macros.h"


__global__ void initialize_curand_on_device(curandState * state, unsigned long seed)
{

}

__device__ float generate(curandState * all_states, int index)
{
	//copy state to local mem
	curandState localState = all_states[index];
	//apply uniform distribution with calculated random
	float result = curand_( &localState );
	//update state
	all_states[index] = localState;
	return result;
}


__global__ void get_random_floats(curandState * all_states, float * output, const unsigned int points)
{

}


int main()
{
	srand(time(nullptr));
	auto init_seed = rand();


	return 0;
}