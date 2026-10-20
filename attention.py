"""注意力公式、独立参考、SDPA 后端与分阶段 CUDA 实现。"""

from functools import cache
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


BACKENDS = ("explicit", "math", "flash", "l1", "l2", "l3", "l4")
L1_MAX_N = 512


@cache
def _load_cuda(backend):
    """首次使用才编译；结果留在 PyTorch 扩展缓存，不写入仓库。"""
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    source = {"l1": "l1_materialized.cu", "l2": "l2_tiled_online.cu",
              "l3": "l3_wmma_online.cu", "l4": "l4_q_parallel.cu"}[backend]
    return load(
        name=f"attention_lab_{backend}",
        sources=[str(root / "bindings.cu"), str(root / "kernels" / source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", f"-DATTENTION_LAB_{backend.upper()}"],
        verbose=False,
    )


def validate_inputs(q, k, v):
    """当前只接收形状相同、连续存储的 FP16 自注意力输入。"""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V 必须具有相同的 [B,H,N,D] 形状")
    if min(q.shape) <= 0 or q.shape[-1] not in (64, 128):
        raise ValueError("各维必须为正数，当前 D 仅支持 64 或 128")
    for x in (q, k, v):
        if x.dtype != torch.float16 or x.device != q.device or not x.is_contiguous():
            raise ValueError("Q/K/V 必须同设备、FP16 且连续")


def explicit_attention(q, k, v, causal=False):
    """显式保存 S/P；两个矩阵乘法和 softmax 使用 FP32 中间结果。"""
    # 每个 batch、head 分别计算；最后两维为 [N,D] @ [D,N]。
    scores = q.float() @ k.float().transpose(-2, -1)
    scores.mul_(1.0 / math.sqrt(q.shape[-1]))
    if causal:
        n = q.shape[-2]
        future = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(future, float("-inf"))
    # torch.softmax 内部使用稳定算法；沿 key 维归一化，而非沿 query 维。
    probabilities = torch.softmax(scores, dim=-1)
    return (probabilities @ v.float()).to(q.dtype)


def prepare_l1(q, k, v, causal=False):
    """一次检查并创建 S/P/O；返回固定缓冲的调用入口，供计时复用。"""
    validate_inputs(q, k, v)
    if q.device.type != "cuda" or q.shape[-2] > L1_MAX_N:
        raise ValueError(f"L1 仅支持 CUDA、小尺寸 N <= {L1_MAX_N}")
    module = _load_cuda("l1")
    b, h, n, _ = q.shape
    scores = torch.empty((b, h, n, n), device=q.device, dtype=torch.float32)
    probabilities = torch.empty_like(scores)
    output = torch.empty_like(q)

    def run():
        module.forward(q, k, v, scores, probabilities, output, causal)
        return output

    return run


def prepare_fused(q, k, v, backend, causal=False):
    """融合后端只创建输出；分块数据与在线状态全部留在 kernel 内。"""
    validate_inputs(q, k, v)
    if q.device.type != "cuda":
        raise ValueError(f"{backend.upper()} 仅支持 CUDA")
    module = _load_cuda(backend)
    output = torch.empty_like(q)

    def run():
        module.forward(q, k, v, output, causal)
        return output

    return run


def prepare_l2(q, k, v, causal=False):
    """创建 L2 固定输出，供计时复用。"""
    return prepare_fused(q, k, v, "l2", causal)


def prepare_l3(q, k, v, causal=False):
    """创建 L3 固定输出，供计时复用。"""
    return prepare_fused(q, k, v, "l3", causal)


def prepare_l4(q, k, v, causal=False):
    """创建 L4 固定输出，供计时复用。"""
    return prepare_fused(q, k, v, "l4", causal)


def attention(q, k, v, backend="explicit", causal=False):
    """公开入口；每次调用都明确限定一个后端，禁止自动回退。"""
    if backend == "l1":
        return prepare_l1(q, k, v, causal)()
    if backend == "l2":
        return prepare_l2(q, k, v, causal)()
    if backend == "l3":
        return prepare_l3(q, k, v, causal)()
    if backend == "l4":
        return prepare_l4(q, k, v, causal)()
    validate_inputs(q, k, v)
    if backend == "explicit":
        return explicit_attention(q, k, v, causal)
    if backend not in ("math", "flash"):
        raise ValueError(f"未知后端：{backend}")
    if backend == "flash" and q.device.type != "cuda":
        raise RuntimeError("本实验只在 CUDA 上验证 SDPA flash 后端")
    selected = SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(selected):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def reference_fp64(q, k, v, causal=False):
    """CPU/FP64 逐 query 参考；不调用显式实现或 SDPA，仅用于小尺寸。"""
    validate_inputs(q, k, v)
    if q.shape[-2] > 129:
        raise ValueError("FP64 参考仅用于 N <= 129 的小尺寸")
    q64, k64, v64 = (x.to(device="cpu", dtype=torch.float64) for x in (q, k, v))
    result = torch.empty_like(q64)
    # 单独计算每行，以避免直接复用待测路径的整张分数矩阵与 mask。
    for i in range(q.shape[-2]):
        end = i + 1 if causal else q.shape[-2]
        row = torch.einsum("bhd,bhjd->bhj", q64[:, :, i], k64[:, :, :end])
        row /= math.sqrt(q.shape[-1])
        weights = torch.exp(row - row.amax(dim=-1, keepdim=True))
        weights /= weights.sum(dim=-1, keepdim=True)
        result[:, :, i] = torch.einsum("bhj,bhjd->bhd", weights, v64[:, :, :end])
    return result
