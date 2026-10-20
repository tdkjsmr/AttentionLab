// 只查询已编译 kernel 的资源与理论驻留上限，不读取硬件性能计数器。
#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <string>

using LaunchResources = std::map<std::string, int64_t>;

template <typename Kernel>
LaunchResources query_launch_resources(Kernel kernel, int threads, int dynamic_shared_bytes) {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, threads, dynamic_shared_bytes));
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return {
        {"threads_per_block", threads},
        {"registers_per_thread", attributes.numRegs},
        // local 包括编译器使用的本地存储，不能直接把它全部称为寄存器溢出。
        {"local_bytes_per_thread", static_cast<int64_t>(attributes.localSizeBytes)},
        {"static_shared_bytes", static_cast<int64_t>(attributes.sharedSizeBytes)},
        {"dynamic_shared_bytes", dynamic_shared_bytes},
        {"max_dynamic_shared_bytes", attributes.maxDynamicSharedSizeBytes},
        {"max_active_blocks_per_sm", blocks},
        {"max_active_threads_per_sm", blocks * threads},
        {"device_max_threads_per_sm", properties.maxThreadsPerMultiProcessor},
        {"device_shared_bytes_per_sm", static_cast<int64_t>(properties.sharedMemPerMultiprocessor)},
        {"sm_count", properties.multiProcessorCount},
        {"binary_version", attributes.binaryVersion},
        {"ptx_version", attributes.ptxVersion},
    };
}
