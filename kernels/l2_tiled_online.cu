// L2：单 CTA/head，串行遍历 Q 小块；每个 warp 独占一行 query。
// Q 外循环、未归一化输出与按 Q 行分工已借鉴 FA2，不称为原样 FA1 复现。
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {

constexpr int kQueries = 4;
constexpr int kKeys = 64;
constexpr int kThreads = kQueries * 32;

__device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int D>
__global__ void tiled_online_kernel(
    const __half* q, const __half* k, const __half* v,
    __half* output, int n, bool causal) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int64_t head_base = static_cast<int64_t>(blockIdx.x) * n * D;
    __shared__ __half q_tile[kQueries][D];
    // K 转置存放，方便各 lane 读取同一特征下的不同 key；多留一列错开 bank。
    __shared__ __half k_tile[D][kKeys + 1];
    __shared__ __half v_tile[kKeys][D];
    __shared__ float probabilities[kQueries][kKeys];

    for (int q_start = 0; q_start < n; q_start += kQueries) {
        const int query = q_start + warp;
        for (int x = tid; x < kQueries * D; x += kThreads) {
            const int row = q_start + x / D;
            q_tile[x / D][x % D] = row < n ? q[head_base + row * D + x % D] : __float2half(0.0f);
        }
        float m = -CUDART_INF_F;
        float l = 0.0f;
        // 每个 lane 持有输出向量的 D/32 个分量，跨所有 K/V 块保留在寄存器中。
        float accum[D / 32] = {};
        const int key_end = causal ? min(n, q_start + kQueries) : n;

        for (int k_start = 0; k_start < key_end; k_start += kKeys) {
            for (int x = tid; x < kKeys * D; x += kThreads) {
                const int row = k_start + x / D;
                const int feature = x % D;
                k_tile[feature][x / D] = row < n ? k[head_base + row * D + feature] : __float2half(0.0f);
                v_tile[x / D][feature] = row < n ? v[head_base + row * D + feature] : __float2half(0.0f);
            }
            // 首轮也在这里等待 Q 加载；无效 query 的 warp 仍然参加 CTA 同步。
            __syncthreads();

            if (query < n) {
                // 一个 lane 计算当前 64 列中的两列分数，点积使用 FP32。
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (int t = 0; t < D; ++t) {
                    const float q_value = __half2float(q_tile[warp][t]);
                    s0 = fmaf(q_value, __half2float(k_tile[t][lane]), s0);
                    s1 = fmaf(q_value, __half2float(k_tile[t][lane + 32]), s1);
                }
                const int key0 = k_start + lane;
                const int key1 = key0 + 32;
                const float scale = rsqrtf(static_cast<float>(D));
                s0 = key0 < n && (!causal || key0 <= query) ? s0 * scale : -CUDART_INF_F;
                s1 = key1 < n && (!causal || key1 <= query) ? s1 * scale : -CUDART_INF_F;

                const float next_m = fmaxf(m, warp_max(fmaxf(s0, s1)));
                const float alpha = expf(m - next_m);
                const float p0 = expf(s0 - next_m);
                const float p1 = expf(s1 - next_m);
                l = alpha * l + warp_sum(p0 + p1);
                m = next_m;
                probabilities[warp][lane] = p0;
                probabilities[warp][lane + 32] = p1;
                // shuffle 本身不提供 shared memory 的读写排序，这里明确同步本 warp。
                __syncwarp();

#pragma unroll
                for (int c = 0; c < D / 32; ++c) {
                    float value = alpha * accum[c];
                    for (int j = 0; j < kKeys; ++j) {
                        value = fmaf(probabilities[warp][j],
                                     __half2float(v_tile[j][lane + c * 32]), value);
                    }
                    accum[c] = value;
                }
            }
            // 全 CTA 读完本块 K/V 后，才允许下一轮覆盖 shared memory。
            __syncthreads();
        }

        if (query < n) {
            const float inverse_l = 1.0f / l;
#pragma unroll
            for (int c = 0; c < D / 32; ++c) {
                output[head_base + query * D + lane + c * 32] = __float2half_rn(accum[c] * inverse_l);
            }
        }
    }
}

}  // 匿名命名空间

void l2_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& output, bool causal) {
    const c10::cuda::CUDAGuard guard(q.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int n = static_cast<int>(q.size(2));
    const int heads = static_cast<int>(q.size(0) * q.size(1));
    const auto* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const auto* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    if (q.size(3) == 64) {
        tiled_online_kernel<64><<<heads, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr, output_ptr, n, causal);
    } else {
        tiled_online_kernel<128><<<heads, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr, output_ptr, n, causal);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
