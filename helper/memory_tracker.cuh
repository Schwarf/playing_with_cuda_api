//
// Created by andreas on 14.05.22.
//

#ifndef MEMORY_TRACKER_CUH
#define MEMORY_TRACKER_CUH
#include <string>
#include <unordered_map>
template <class T> class MemoryTracker
{
public:
	MemoryTracker() =default;

	T * allocate_host_memory(size_t number_of_elements, std::string && name)
	{
		T * host_memory = new T[number_of_elements];

		host_memory_hashes_.insert(std::make_pair(PointerHash<T>()(host_memory), name));
		return host_memory;
	}

	T * allocate_device_memory(size_t number_of_elements, std::string && name)
	{
		cudaError_t error = cudaSuccess;
		auto size = number_of_elements*sizeof(T);
		T * device_memory = nullptr;
		error = cudaMalloc((void **)&device_memory, size);
		if (error != cudaSuccess)
		{
			std::cerr << "Failed to allocate device memory of '" << name << "'. Error code " << cudaGetErrorString(error) << "\n";
			exit(EXIT_FAILURE);
		}
		device_memory_hashes_.insert(std::make_pair(PointerHash<T>()(device_memory), name));
		return device_memory;
	}

	void free_device_memory(T * device_memory)
	{

		auto hash = PointerHash<T>()(device_memory);
		if(device_memory_hashes_.find(hash) == device_memory_hashes_.end())
		{
			std::cout << "Device memory is not registered." << std::endl;
		}
		cudaError_t error = cudaSuccess;
		error = cudaFree(device_memory);
		if (error != cudaSuccess)
		{
			std::cerr << "Failed to free device memory of '" << device_memory_hashes_[hash] << "'. Error code " << cudaGetErrorString(error) << "\n";
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
			std::cerr<< "Failed to copy from HOST to DEVICE (error code "  << cudaGetErrorString(error) << ")!\n";
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
			std::cerr<< "Failed to copy from DEVICE to HOST (error code "  << cudaGetErrorString(error) << ")!\n";
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
	std::unordered_map<size_t, std::string> device_memory_hashes_;
	std::unordered_map<size_t, std::string> host_memory_hashes_;
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

#endif //MEMORY_TRACKER_CUH
