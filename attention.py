"""L0：显式注意力公式、独立小尺寸参考与强制 SDPA 后端。"""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


BACKENDS = ("explicit", "math", "flash")


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


def attention(q, k, v, backend="explicit", causal=False):
    """公开入口；每次调用都明确限定一个后端，禁止自动回退。"""
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
