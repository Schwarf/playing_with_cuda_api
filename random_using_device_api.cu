#include "cuda_runtime.h"

#include <cstdio>
#include <ctime>
#include <curand_kernel.h>
#include <vector>
#include <iostream>


#define N 10  // number of trials
#define DICE 20      // number of the maximum desired value

class Histogram
{
public:
	Histogram(size_t number_of_bins)
		:
		number_of_bins_(number_of_bins)
	{
		result_ = std::vector<long long unsigned>(number_of_bins, 0);
	}
	void add_values(unsigned int *input, size_t size)
	{
		for(size_t i =0; i < size; ++i)
		{
			result_[input[i]]++;
		}
	}
	void print_result(){
		for (size_t i = 0; i < DICE + 1; i++)
			printf("%2d : %10d\n", i, result_[i]);
	}
private:
	size_t number_of_bins_;
	std::vector<long long unsigned> result_;
};

__global__ void init(unsigned int seed, curandState_t *states)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, /* the seed can be the same for each thread, here we pass the time from CPU */
				id,   /* the sequence number should be different for each core */
				0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&states[id]);
}

__global__ void random_generation_with_ceiling(curandState_t *states, unsigned int *numbers)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	numbers[id] = ceilf(curand_uniform(&states[id]) * DICE);
}


int main()
{
	auto histogram = Histogram(DICE);
	curandState_t *states;
	cudaMalloc((void **)&states, N * sizeof(curandState_t));

	// initialize the random states
	dim3 blkDim = 1000;
	dim3 grdDim = (N + blkDim.x - 1) / blkDim.x;
	auto seed = time(0);
	std::cout << "Seed: " << seed <<std::endl;
	init<<<grdDim, blkDim >>>(seed, states);

	// allocate an array of unsigned ints on the CPU and GPU
	unsigned int host_random_nums[N];
	unsigned int *device_random_nums;
	cudaMalloc((void **)&device_random_nums, N * sizeof(unsigned int));


	// get random number with ceiling
	random_generation_with_ceiling<<<grdDim, blkDim >>>(states, device_random_nums);
	cudaMemcpy(host_random_nums, device_random_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	printf("Histogram for random numbers generated with ceiling\n");
	histogram.add_values(host_random_nums, N);
	histogram.print_result();
	cudaFree(states);
	cudaFree(device_random_nums);

	return 0;
}
