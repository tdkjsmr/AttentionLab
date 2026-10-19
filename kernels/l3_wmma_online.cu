// L3：QK 与 PV 使用同步 WMMA；一个 CTA 仍串行遍历一个 head 的 Q 块。
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <cstdint>

namespace {

namespace wmma = nvcuda::wmma;
constexpr int kQueries = 16;
constexpr int kKeys = 64;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

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
__global__ void wmma_online_kernel(
    const __half* q, const __half* k, const __half* v,
    __half* output, int n, bool causal) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int64_t head_base = static_cast<int64_t>(blockIdx.x) * n * D;
    // WMMA 入口使用 32 字节对齐；半精度步长为 8 的倍数，单精度为 4 的倍数。
    __shared__ __align__(32) __half q_tile[kQueries][D];
    // QK 结束后 K 不再存活，同一缓冲改装当前 V 块。
    __shared__ __align__(32) __half kv_tile[kKeys][D];
    __shared__ __align__(32) float scores[kQueries][kKeys];
    __shared__ __align__(32) __half probabilities[kQueries][kKeys];
    __shared__ __align__(32) float accum[kQueries][D];
    __shared__ float row_max[kQueries];
    __shared__ float row_sum[kQueries];

    for (int q_start = 0; q_start < n; q_start += kQueries) {
        for (int x = tid; x < kQueries * D; x += kThreads) {
            const int row = q_start + x / D;
            q_tile[x / D][x % D] = row < n ? q[head_base + row * D + x % D] : __float2half(0.0f);
            accum[x / D][x % D] = 0.0f;
        }
        if (tid < kQueries) {
            row_max[tid] = -CUDART_INF_F;
            row_sum[tid] = 0.0f;
        }
        __syncthreads();
        const int key_end = causal ? min(n, q_start + kQueries) : n;

        for (int k_start = 0; k_start < key_end; k_start += kKeys) {
            for (int x = tid; x < kKeys * D; x += kThreads) {
                const int key = k_start + x / D;
                kv_tile[x / D][x % D] = key < n ? k[head_base + key * D + x % D] : __float2half(0.0f);
            }
            __syncthreads();

            // 每个 warp 负责分数块的 16 列；K 的行优先存储就是 K 转置的列优先视图。
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
#pragma unroll
            for (int t = 0; t < D; t += 16) {
                wmma::load_matrix_sync(a, &q_tile[0][t], D);
                wmma::load_matrix_sync(b, &kv_tile[warp * 16][t], D);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(&scores[0][warp * 16], c, kKeys, wmma::mem_row_major);
            // 所有 warp 都结束读取 K 后，才允许将 kv_tile 覆盖为 V。
            __syncthreads();

            // softmax 按行分工；每个 warp 串行处理 4 行，每个 lane 处理两列。
            for (int row = warp; row < kQueries; row += kWarps) {
                const int query = q_start + row;
                float p0 = 0.0f;
                float p1 = 0.0f;
                if (query < n) {
                    const int key0 = k_start + lane;
                    const int key1 = key0 + 32;
                    const float scale = rsqrtf(static_cast<float>(D));
                    const float s0 = key0 < n && (!causal || key0 <= query)
                                         ? scores[row][lane] * scale : -CUDART_INF_F;
                    const float s1 = key1 < n && (!causal || key1 <= query)
                                         ? scores[row][lane + 32] * scale : -CUDART_INF_F;
                    const float next_m = fmaxf(row_max[row], warp_max(fmaxf(s0, s1)));
                    const float alpha = expf(row_max[row] - next_m);
                    p0 = expf(s0 - next_m);
                    p1 = expf(s1 - next_m);
                    const float next_l = alpha * row_sum[row] + warp_sum(p0 + p1);
                    // 通过显式行布局重缩放旧输出，不假定 WMMA fragment 的元素映射。
#pragma unroll
                    for (int t = lane; t < D; t += 32) {
                        accum[row][t] *= alpha;
                    }
                    // 先让本 warp 的所有 lane 读完旧 m/l，再发布新状态。
                    __syncwarp();
                    if (lane == 0) {
                        row_max[row] = next_m;
                        row_sum[row] = next_l;
                    }
                }
                // 无效 query 写零概率；所有 lane 仍参与后续 WMMA 和 CTA 同步。
                probabilities[row][lane] = __float2half_rn(p0);
                probabilities[row][lane + 32] = __float2half_rn(p1);
            }
            for (int x = tid; x < kKeys * D; x += kThreads) {
                const int key = k_start + x / D;
                kv_tile[x / D][x % D] = key < n ? v[head_base + key * D + x % D] : __float2half(0.0f);
            }
            __syncthreads();

            // PV 按输出列分工，FP16 概率乘 FP16 V，累加到已重缩放的 FP32 输出。
            for (int feature = warp * 16; feature < D; feature += kWarps * 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> values;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> result;
                wmma::load_matrix_sync(result, &accum[0][feature], D, wmma::mem_row_major);
#pragma unroll
                for (int t = 0; t < kKeys; t += 16) {
                    wmma::load_matrix_sync(p, &probabilities[0][t], kKeys);
                    wmma::load_matrix_sync(values, &kv_tile[t][feature], D);
                    wmma::mma_sync(result, p, values, result);
                }
                wmma::store_matrix_sync(&accum[0][feature], result, D, wmma::mem_row_major);
            }
            // 所有输出片段完成后，才开始下一轮重缩放或重用 K/V 缓冲。
            __syncthreads();
        }

        for (int x = tid; x < kQueries * D; x += kThreads) {
            const int row = x / D;
            if (q_start + row < n) {
                output[head_base + (q_start + row) * D + x % D] =
                    __float2half_rn(accum[row][x % D] / row_sum[row]);
            }
        }
        // 所有线程读完当前行的分母后，才初始化下一 Q 块的在线状态。
        __syncthreads();
    }
}

}  // 匿名命名空间

void l3_forward(
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
        wmma_online_kernel<64><<<heads, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr, output_ptr, n, causal);
    } else {
        wmma_online_kernel<128><<<heads, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr, output_ptr, n, causal);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
