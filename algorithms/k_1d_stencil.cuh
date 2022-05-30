//
// Created by andreas on 30.05.22.
//

#ifndef K_1D_STENCIL_CUH
#define K_1D_STENCIL_CUH
#include <iostream>

template<typename T>
T *kth_stencil(T *input, int input_size, int k)
{
	if (input_size < 2 * k)
		return nullptr;
	auto output_size = input_size - 2 * k;
	T denominator = static_cast<T>(2 * k + 1);
	T *stencil = new T[output_size + 1];
	int max_index = (2 * k + 1);
	for (int output_index = 0; output_index < output_size; output_index++) {
		stencil[output_index] = T{};

		int index{};
		while (index < max_index) {
			stencil[output_index] += input[output_index + index++];
		}
		stencil[output_index] /= denominator;
	}

	return stencil;
}


#endif //K_1D_STENCIL_CUH
