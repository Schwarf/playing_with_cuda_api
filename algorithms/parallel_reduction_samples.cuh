//
// Created by andreas on 31.05.22.
//

#ifndef PARALLEL_REDUCTION_SAMPLES_CUH
#define PARALLEL_REDUCTION_SAMPLES_CUH

// WITH DIVERGENT WARPS
template <typename T>
__global__ void parallel_array_sum_v0(T * input, T* output, int input_size)
{
	extern __shared__ T shared_data[];
	const int thread_id = threadIdx.x;
	const int data_index = blockIdx.x * blockDim.x + threadIdx.x;
	shared_data[thread_id] = (data_index < input_size) ? input[data_index] : 0;
	__syncthreads();

	for(int i =1; i < blockDim.x ; i *=2)
	{
		if(thread_id % (2*i) ==0)
		{
			shared_data[thread_id] += shared_data[thread_id + i];
		}
		__syncthreads();
	}

	if(thread_id == 0)
		output[blockIdx.x] = shared_data[0];

}


#endif //PARALLEL_REDUCTION_SAMPLES_CUH
