//
// Created by andreas on 15.05.22.
//

#ifndef VECTOR_ADDITION_CUH
#define VECTOR_ADDITION_CUH
template <typename T>
__global__ void
vectorAdd(const T *A, const T *B, T *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

#endif //VECTOR_ADDITION_CUH
