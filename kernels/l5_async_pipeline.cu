// L5：K/V 分离缓冲与 cp.async 预取；CTA 分工和数值计算沿用 L4。
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <cstdint>
#include "launch_resources.cuh"

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
constexpr int shared_bytes() {
    return 352 * D + 6272;
}

// 每个线程拷贝若干个 16 字节片段；尾行补零，不形成越界源地址。
template <int D>
__device__ void copy_tile_async(__half* destination, const __half* source,
                                int64_t head_base, int key_start, int n) {
    for (int chunk = threadIdx.x; chunk < kKeys * D / 8; chunk += kThreads) {
        const int x = chunk * 8;
        const int key = key_start + x / D;
        const int64_t offset = key < n ? static_cast<int64_t>(key) * D + x % D : 0;
        const auto address = static_cast<unsigned>(__cvta_generic_to_shared(destination + x));
        const __half* src = source + head_base + offset;
        if (key < n) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(address), "l"(src) : "memory");
        } else {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;"
                         :: "r"(address), "l"(src) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int Pending>
__device__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

template <int D>
__global__ void wmma_online_kernel(
    const __half* q, const __half* k, const __half* v,
    __half* output, int n, bool causal) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int q_tiles = (n + kQueries - 1) / kQueries;
    const int64_t head_base = static_cast<int64_t>(blockIdx.x / q_tiles) * n * D;
    // 分配顺序与 shared_bytes 保持一致；各段首地址均为 32 字节对齐。
    extern __shared__ __align__(32) unsigned char storage[];
    auto* q_tile = reinterpret_cast<__half (*)[D]>(storage);
    auto* k_tile = reinterpret_cast<__half (*)[D]>(q_tile + kQueries);
    auto* v_tile = reinterpret_cast<__half (*)[D]>(k_tile + kKeys);
    auto* scores = reinterpret_cast<float (*)[kKeys]>(v_tile + kKeys);
    auto* probabilities = reinterpret_cast<__half (*)[kKeys]>(scores + kQueries);
    auto* accum = reinterpret_cast<float (*)[D]>(probabilities + kQueries);
    auto* row_max = reinterpret_cast<float*>(accum + kQueries);
    auto* row_sum = row_max + kQueries;

    // 首个 K 块的拷贝与 Q/累积状态初始化重叠。
    copy_tile_async<D>(&k_tile[0][0], k, head_base, 0, n);

    const int q_start = (blockIdx.x % q_tiles) * kQueries;
    for (int x = tid; x < kQueries * D; x += kThreads) {
        const int row = q_start + x / D;
        q_tile[x / D][x % D] = row < n ? q[head_base + row * D + x % D] : __float2half(0.0f);
        accum[x / D][x % D] = 0.0f;
    }
    if (tid < kQueries) {
        row_max[tid] = -CUDART_INF_F;
        row_sum[tid] = 0.0f;
    }
    const int key_end = causal ? min(n, q_start + kQueries) : n;

    for (int k_start = 0; k_start < key_end; k_start += kKeys) {
        // 等待每个线程的 K 拷贝，再让整个 CTA 看到完整 K 和初始化状态。
        wait_copies<0>();
        __syncthreads();
        // 搬运当前 V 时继续执行 QK；两个缓冲互不覆盖。
        copy_tile_async<D>(&v_tile[0][0], v, head_base, k_start, n);

        // 每个 warp 负责分数块的 16 列；K 的行优先存储就是 K 转置的列优先视图。
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.0f);
#pragma unroll
        for (int t = 0; t < D; t += 16) {
            wmma::load_matrix_sync(a, &q_tile[0][t], D);
            wmma::load_matrix_sync(b, &k_tile[warp * 16][t], D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(&scores[0][warp * 16], c, kKeys, wmma::mem_row_major);
        // 所有 warp 都读完当前 K 后，才允许预取下一 K 块覆盖它。
        __syncthreads();
        const bool has_next = k_start + kKeys < key_end;
        if (has_next) {
            copy_tile_async<D>(&k_tile[0][0], k, head_base, k_start + kKeys, n);
        }

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
        // 已提交 V、下一 K 两组时只等 V；保留 K 预取与 PV 计算重叠。
        if (has_next) {
            wait_copies<1>();
        } else {
            wait_copies<0>();
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
                wmma::load_matrix_sync(values, &v_tile[t][feature], D);
                wmma::mma_sync(result, p, values, result);
            }
            wmma::store_matrix_sync(&accum[0][feature], result, D, wmma::mem_row_major);
        }
        // 读完当前 V 后才允许下一轮覆盖它；下一 K 可在另一缓冲继续搬运。
        __syncthreads();
    }

    for (int x = tid; x < kQueries * D; x += kThreads) {
        const int row = x / D;
        if (q_start + row < n) {
            output[head_base + (q_start + row) * D + x % D] =
                __float2half_rn(accum[row][x % D] / row_sum[row]);
        }
    }
}

}  // 匿名命名空间

// D128 超过默认的 48 KiB，在准备入口配置；不放进重复计时调用。
void l5_configure(const at::Tensor& q) {
    const c10::cuda::CUDAGuard guard(q.device());
    if (q.size(3) == 128) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(wmma_online_kernel<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes<128>()));
    }
}

// 调用前由 prepare_l5 配置动态 shared 上限，查询时使用实际启动的字节数。
LaunchResources l5_resources(const at::Tensor& q) {
    const c10::cuda::CUDAGuard guard(q.device());
    if (q.size(3) == 64) {
        return query_launch_resources(wmma_online_kernel<64>, kThreads, shared_bytes<64>());
    }
    return query_launch_resources(wmma_online_kernel<128>, kThreads, shared_bytes<128>());
}

void l5_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& output, bool causal) {
    const c10::cuda::CUDAGuard guard(q.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int n = static_cast<int>(q.size(2));
    const int heads = static_cast<int>(q.size(0) * q.size(1));
    const int blocks = heads * ((n + kQueries - 1) / kQueries);
    const auto* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const auto* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    if (q.size(3) == 64) {
        wmma_online_kernel<64><<<blocks, kThreads, shared_bytes<64>(), stream>>>(q_ptr, k_ptr, v_ptr, output_ptr, n, causal);
    } else {
        wmma_online_kernel<128><<<blocks, kThreads, shared_bytes<128>(), stream>>>(q_ptr, k_ptr, v_ptr, output_ptr, n, causal);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
