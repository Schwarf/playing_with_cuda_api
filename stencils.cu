//
// Created by andreas on 30.05.22.
//
#include "algorithms/k_1d_stencil.cuh"
#include <iostream>
int main()
{
	constexpr int size = 8;
	constexpr int k = 1;
	int A[size] = {0,1,2,3,4,5,6,7};
	auto B = kth_stencil(A, size, k);
	for(int i =0; i < size-2*k; ++i)
	{
		std::cout << B[i] << std::endl;
	}

}