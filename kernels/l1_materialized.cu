// L1：逐元素计算 QK 和 PV，显式保存完整 FP32 分数与概率矩阵。
// 输入约定由 attention.py 检查；S/P/O 也由同一入口创建。
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;

__global__ void qk_kernel(
    const __half* q, const __half* k, float* scores,
    int n, int d, bool causal) {
    // 一个 block 负责一个 head 内的 query 行 i；线程分别负责 key 列 j。
    const int i = blockIdx.x;
    const int bh = blockIdx.y;
    const int64_t head_base = static_cast<int64_t>(bh) * n * d;
    const int64_t score_base = (static_cast<int64_t>(bh) * n + i) * n;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (causal && j > i) {
            scores[score_base + j] = -CUDART_INF_F;
            continue;
        }
        float dot = 0.0f;
        // 相同的分量 t 相乘，在 FP32 中累积完整点积。
        for (int t = 0; t < d; ++t) {
            const float q_value = __half2float(q[head_base + i * d + t]);
            const float k_value = __half2float(k[head_base + j * d + t]);
            dot = fmaf(q_value, k_value, dot);
        }
        scores[score_base + j] = dot * rsqrtf(static_cast<float>(d));
    }
}

__global__ void softmax_kernel(const float* scores, float* probabilities, int n) {
    // 一个 block 负责一行；所有线程都参加归约，包括尾部没有列的线程。
    const int i = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t row = (static_cast<int64_t>(bh) * n + i) * n;
    __shared__ float scratch[kThreads];

    float local_max = -CUDART_INF_F;
    for (int j = tid; j < n; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[row + j]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = scratch[0];
    // 等所有线程取走最大值，再复用同一块 shared memory 存放局部和。
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        const float weight = expf(scores[row + j] - row_max);
        probabilities[row + j] = weight;
        local_sum += weight;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / scratch[0];
    for (int j = tid; j < n; j += blockDim.x) {
        probabilities[row + j] *= inverse_sum;
    }
}

__global__ void pv_kernel(
    const float* probabilities, const __half* v, __half* output, int n, int d) {
    // 一个 block 负责一个输出行；每个有效线程负责一个特征分量 t。
    const int i = blockIdx.x;
    const int bh = blockIdx.y;
    const int t = threadIdx.x;
    if (t >= d) {
        return;
    }
    const int64_t head_base = static_cast<int64_t>(bh) * n * d;
    const int64_t row = (static_cast<int64_t>(bh) * n + i) * n;
    float value = 0.0f;
    for (int j = 0; j < n; ++j) {
        value = fmaf(probabilities[row + j], __half2float(v[head_base + j * d + t]), value);
    }
    output[head_base + i * d + t] = __float2half_rn(value);
}

}  // 匿名命名空间

void l1_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& scores, const at::Tensor& probabilities,
    const at::Tensor& output, bool causal) {
    // 跟随输入设备和 PyTorch 当前 stream，三个 kernel 按依赖顺序执行。
    const c10::cuda::CUDAGuard guard(q.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int n = static_cast<int>(q.size(2));
    const int d = static_cast<int>(q.size(3));
    const dim3 grid(n, static_cast<unsigned int>(q.size(0) * q.size(1)));
    const auto* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const auto* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    qk_kernel<<<grid, kThreads, 0, stream>>>(q_ptr, k_ptr, scores.data_ptr<float>(), n, d, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    softmax_kernel<<<grid, kThreads, 0, stream>>>(scores.data_ptr<float>(), probabilities.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pv_kernel<<<grid, kThreads, 0, stream>>>(probabilities.data_ptr<float>(), v_ptr, output_ptr, n, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
