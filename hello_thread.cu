//
// Created by andreas on 31.05.22.
//
#include <cstdio>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 2

__global__ void hello()
{
	printf("Hello Thread! I am a thread %d in block %d. Block dim is %d\n", threadIdx.x, blockIdx.x, blockDim.x);
}

int main()
{
	hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
	cudaDeviceSynchronize();
	return 0;
}