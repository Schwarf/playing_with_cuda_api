//
// Created by andreas on 30.05.22.
//
#include "algorithms/k_1d_stencil.cuh"
#include "helper/memory_tracker.cuh"
#include <iostream>
#include <type_traits>
#include <random>

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
static inline T get_random_integral(const T & lower_bound, const T & upper_bound, T * output, int output_size)
{
	static std::mt19937 generator;
	auto int_distribution = std::uniform_int_distribution<T>(lower_bound, upper_bound);
	for(int i = 0; i < output_size; i++)
	{
		output[i] = int_distribution(generator);
	}
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
static inline T get_random_float(const T & lower_bound, const T & upper_bound, T * output, int output_size)
{
	static std::mt19937 generator;
	auto real_distribution = std::uniform_real_distribution<T>(lower_bound, upper_bound);
	for(int i = 0; i < output_size; i++)
	{
		output[i] = real_distribution(generator);
	}
}



int main()
{
	constexpr int size = 800;
	constexpr int k = 2;
	float A[size];
	get_random_float<float>(0.f, 100.f, A, size);
	auto B = kth_stencil(A, size, k);
	for(int i =0; i < size-2*k; ++i)
	{
		std::cout << (A[i] + A[i+1] + A[i+2] + A[i+3] + A[i+4])/5.f << "  " << B[i] << std::endl;
		//std::cout << (A[i] + A[i+1] + A[i+2] )/3 / B[i] << std::endl;
	}
	//auto tracker = MemoryTracker<int>();
	//tracker.allocate_device_memory()


}