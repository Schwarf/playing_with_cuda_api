//
// Created by andreas on 14.05.22.
//

#ifndef MEMORY_TRACKING_H
#define MEMORY_TRACKING_H

#include <unordered_set>
template <class T> class MemoryTracking
{
public:
	MemoryTracking() =default;

	T * allocate_host_memory(size_t number_of_elements)
	{
		T * host_memory = new T[number_of_elements];

		host_memory_hashes_.insert(PointerHash<T>()(host_memory));
		return host_memory;
	}

	T * allocate_device_memory(size_t number_of_elements)
	{
		cudaError_t error = cudaSuccess;
		auto size = number_of_elements*sizeof(T);
		T * device_memory = nullptr;
		error = cudaMalloc((void **)&device_memory, size);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		device_memory_hashes_.insert(PointerHash<T>()(device_memory));
		return device_memory;
	}

	void free_device_memory(T * device_memory)
	{

		if(device_memory_hashes_.find(PointerHash<T>()(device_memory)) == device_memory_hashes_.end())
		{
			std::cout << "Device memory is not registered." << std::endl;
		}
		cudaError_t error = cudaSuccess;
		error = cudaFree(device_memory);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to free device memory (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		device_memory_hashes_.erase(PointerHash<T>()(device_memory));
	}

	void free_host_memory(T * host_memory)
	{
		if(host_memory_hashes_.find(PointerHash<T>()(host_memory)) == host_memory_hashes_.end())
		{
			std::cout << "Host memory is not registered." << std::endl;
		}

		delete [] host_memory;
		host_memory_hashes_.erase(PointerHash<T>()(host_memory));

	}

	void copy_host_array_to_device_array(T * host_memory, T* device_memory, size_t number_of_elements)
	{
		cudaError_t error = cudaSuccess;
		auto size = number_of_elements*sizeof(T);
		error = cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}

	void copy_device_array_to_host_array(T * host_memory, T* device_memory, size_t number_of_elements)
	{
		cudaError_t error = cudaSuccess;
		auto size = number_of_elements*sizeof(T);
		error = cudaMemcpy(host_memory, device_memory, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

	}

	bool is_device_memory_freed()
	{
		return device_memory_hashes_.empty();
	}

	bool is_host_memory_freed()
	{
		return host_memory_hashes_.empty();
	}

private:
	std::unordered_set<size_t> device_memory_hashes_;
	std::unordered_set<size_t> host_memory_hashes_;
	template <typename U>
	struct PointerHash
	{
		size_t operator()(const U * value) const
		{
			static const auto shift = (size_t)log2(1+ sizeof(U));
			return (size_t) (value) >> shift;
		}
	};

};

#endif //MEMORY_TRACKING_H
