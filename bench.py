"""统一执行已实现后端的正确性、CUDA events 计时及计时外显存检查。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import uuid

import torch

from attention import BACKENDS, L1_MAX_N, attention, prepare_l1, prepare_l2, prepare_l3, prepare_l4, prepare_l5, prepare_l6, reference_fp64


ROOT = Path(__file__).resolve().parent
# 固定 FP16 验收预算；失败时定位原因，不自动放宽阈值。
ATOL, RTOL = 2e-3, 5e-3
L2_CONFIG = {"q_tile": 4, "kv_tile": 64, "threads_per_block": 128, "warps": 4,
             "cta_mapping": "one_cta_per_head_serial_q_tiles"}
L3_CONFIG = {"q_tile": 16, "kv_tile": 64, "threads_per_block": 128, "warps": 4,
             "cta_mapping": "one_cta_per_head_serial_q_tiles", "mma": "wmma_m16n16k16",
             "probability_dtype": "float16", "accumulator_dtype": "float32",
             "output_accumulator_storage": "shared"}
L4_CONFIG = {**L3_CONFIG, "cta_mapping": "one_cta_per_q_tile"}
L5_CONFIG = {**L4_CONFIG, "async_copy": "cp.async.cg.shared.global", "copy_bytes": 16,
             "kv_buffers": 2, "max_pending_copy_groups": 2, "shared_allocation": "dynamic",
             "pipeline": "v_with_qk_next_k_with_pv", "min_compute_capability": [8, 0]}
L6_CONFIG = {**L5_CONFIG, "score_probability_storage": "inplace_per_row",
             "probability_row_stride_elements": 128}


def command(argv):
    """只记录所需版本字段，不导出环境变量或用户配置。"""
    try:
        p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def configure_precision():
    # 小尺寸逐行 CPU 参考固定为单线程，避免小矩阵的线程调度开销。
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    setting = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    if setting is not None:
        setting(False)
    return {"tf32": False, "fp16_reduced_precision_reduction": False,
            "math_sdp_reduced_precision_reduction": False if setting else "api_unavailable"}


def environment(device, precision):
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    status = command(git + ["status", "--porcelain", "--untracked-files=normal"])
    data = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "git_sha": command(git + ["rev-parse", "HEAD"]),
            "git_dirty": bool(status) if status is not None else None,
            "device": str(device), "precision": precision, "cpu_num_threads": torch.get_num_threads(),
            "nvcc": command(["nvcc", "--version"]),
            "driver_inventory": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                          "--format=csv,noheader"])}
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        data.update(gpu=prop.name, compute_capability=[prop.major, prop.minor],
                    total_memory_bytes=prop.total_memory, cudnn=torch.backends.cudnn.version())
    return data


def inputs(shape, device, seed, kind="random"):
    # 先在 CPU 生成再转换，所有后端使用同一组已经量化为 FP16 的输入。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = [torch.randn(shape, generator=generator) for _ in range(3)]
    if kind == "zero_q":
        q.zero_()
    elif kind == "large_scores":
        q.mul_(4)
        k.mul_(4)
    elif kind == "late_max":
        # 将最大分数明确放在后续 K 块，检查旧累积是否按新最大值重新缩放。
        q.zero_()
        q[..., 0] = 8
        k.zero_()
        k[..., 0] = -8
        k[:, :, 64:128, 0] = 0
        last_start = (shape[-2] - 1) // 64 * 64
        k[:, :, last_start:, 0] = 8
    return tuple(x.to(device=device, dtype=torch.float16) for x in (q, k, v))


def errors(actual, expected):
    actual = actual.to(device="cpu", dtype=torch.float64)
    expected = expected.to(device="cpu", dtype=torch.float64)
    diff = (actual - expected).abs()
    bad = (~torch.isfinite(actual)) | (diff > ATOL + RTOL * expected.abs())
    finite = bool(torch.isfinite(diff).all())
    return {"max_abs": diff.max().item() if finite else None,
            "normalized_l2": (diff.norm() / expected.norm().clamp_min(1e-30)).item() if finite else None,
            "failed_elements": bad.sum().item(), "elements": actual.numel(),
            "atol": ATOL, "rtol": RTOL}


def verify(args, device, emit):
    failures = 0
    cases = [("random", n) for n in (1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                                     127, 128, 129, 191, 192, 193, 255, 256, 257)]
    cases += [("zero_q", 33), ("zero_q", 129), ("large_scores", 65), ("late_max", 129), ("late_max", 257)]
    for index, (kind, n) in enumerate(cases):
        q, k, v = inputs((args.batch, args.heads, n, args.head_dim), device, args.seed + index, kind)
        expected = reference_fp64(q, k, v, args.causal)
        # 零 Q 还用独立解析关系检查；causal 对应各前缀的均值。
        if kind == "zero_q":
            values = v.to(device="cpu", dtype=torch.float64)
            if args.causal:
                count = torch.arange(1, n + 1, dtype=torch.float64).view(1, 1, n, 1)
                analytic = values.cumsum(dim=-2) / count
            else:
                analytic = values.mean(dim=-2, keepdim=True).expand_as(values)
            torch.testing.assert_close(expected, analytic, rtol=1e-12, atol=1e-12)
        for backend in args.backends:
            row = {"kind": "verify", "backend": backend, "case": kind,
                   "shape": list(q.shape), "case_seed": args.seed + index,
                   "reference": "cpu_fp64_rowwise"}
            if backend == "l2":
                row.update(L2_CONFIG)
            elif backend in ("l3", "l4", "l5", "l6"):
                tiles = 1 if backend == "l3" else (n + 15) // 16
                config = {"l3": L3_CONFIG, "l4": L4_CONFIG, "l5": L5_CONFIG, "l6": L6_CONFIG}[backend]
                row.update(config,
                           cta_count=args.batch * args.heads * tiles)
            if device.type == "cpu" and backend in ("flash", "l1", "l2", "l3", "l4", "l5", "l6"):
                emit(**row, status="skipped", reason="CPU 模式没有 CUDA 后端覆盖")
                continue
            try:
                metrics = errors(attention(q, k, v, backend, args.causal), expected)
                passed = metrics["failed_elements"] == 0
                emit(**row, status="pass" if passed else "fail", **metrics)
                failures += not passed
            except (RuntimeError, NotImplementedError) as exc:
                # 后端不可用也是本次验证未通过；不以另一个后端替代。
                emit(**row, status="error", reason=str(exc))
                if backend in ("l1", "l2", "l3", "l4", "l5", "l6"):
                    # 编译失败只报告一次，停止本轮，避免每个形状重复构建。
                    emit(kind="verify_summary", status="fail", failed_cases=failures + 1,
                         cuda_flash_verified=False, cuda_l1_verified=False, cuda_l2_verified=False,
                         cuda_l3_verified=False, cuda_l4_verified=False, cuda_l5_verified=False, cuda_l6_verified=False,
                         head_dim=args.head_dim, tested_cases=index + 1)
                    return failures + 1
                failures += 1
    emit(kind="verify_summary", status="pass" if failures == 0 else "fail",
         failed_cases=failures,
         cuda_flash_verified=device.type == "cuda" and "flash" in args.backends and failures == 0,
         cuda_l1_verified=device.type == "cuda" and "l1" in args.backends and failures == 0,
         cuda_l2_verified=device.type == "cuda" and "l2" in args.backends and failures == 0,
         cuda_l3_verified=device.type == "cuda" and "l3" in args.backends and failures == 0,
         cuda_l4_verified=device.type == "cuda" and "l4" in args.backends and failures == 0,
         cuda_l5_verified=device.type == "cuda" and "l5" in args.backends and failures == 0,
         cuda_l6_verified=device.type == "cuda" and "l6" in args.backends and failures == 0,
         head_dim=args.head_dim, tested_cases=len(cases))
    return failures


def elapsed(fn, args):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.groups):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.repeats)
    return {"gpu_elapsed_ms_median": statistics.median(samples), "group_ms_per_call": samples}


def peak_memory(fn):
    # 预热之后先等待；公共输入等存活分配计入 baseline，不计入增量峰值。
    fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    result = {"baseline_allocated_bytes": baseline,
              "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
              "output_bytes": output.numel() * output.element_size(),
              "workspace_bytes": None, "workspace_note": "PyTorch 内部分配未通过公共接口暴露"}
    return result


def measure(args, device, emit):
    failures = 0
    for n in args.n:
        shape = (args.batch, args.heads, n, args.head_dim)
        # 两张 FP32 N*N 矩阵本身已需这些字节；这不是完整峰值估算。
        score_probability_bytes = 2 * args.batch * args.heads * n * n * 4
        if score_probability_bytes > args.max_score_mib * 1024**2:
            emit(kind=args.mode, shape=list(shape), status="skipped",
                 reason="显式 S/P 超出预先设置的内存预算", score_probability_bytes=score_probability_bytes)
            continue
        q, k, v = inputs(shape, device, args.seed)
        try:
            # 大尺寸仅用显式 FP32 路径配对检查；不冒称 FP64 独立真值。
            expected = attention(q, k, v, "explicit", args.causal).cpu()
        except torch.cuda.OutOfMemoryError as exc:
            emit(kind=args.mode, shape=list(shape), status="skipped", reason=f"参考路径 OOM：{exc}")
            continue
        for backend in args.backends:
            row = {"kind": args.mode, "backend": backend, "shape": list(shape),
                   "reference": "explicit_fp32_with_fp16_output", "case_seed": args.seed,
                   "score_probability_bytes_theoretical": score_probability_bytes}
            if backend == "l1" and n > L1_MAX_N:
                emit(**row, status="skipped", reason=f"L1 只测小尺寸 N <= {L1_MAX_N}")
                continue
            try:
                custom = backend in ("l1", "l2", "l3", "l4", "l5", "l6")
                row.update(allocation="preallocated" if custom and args.mode == "bench" else "per_call")
                if backend == "l1":
                    row.update(threads_per_block=128, warps=4)
                elif backend == "l2":
                    row.update(L2_CONFIG, shared_bytes_declared=266 * args.head_dim + 1024)
                elif backend in ("l3", "l4", "l5", "l6"):
                    tiles = 1 if backend == "l3" else (n + 15) // 16
                    config = {"l3": L3_CONFIG, "l4": L4_CONFIG, "l5": L5_CONFIG, "l6": L6_CONFIG}[backend]
                    shared_bytes = ((352 * args.head_dim + 4224) if backend == "l6" else
                                    (352 if backend == "l5" else 224) * args.head_dim + 6272)
                    row.update(config, shared_bytes_declared=shared_bytes,
                               cta_count=args.batch * args.heads * tiles)
                if custom and args.mode == "bench":
                    prepare = {"l1": prepare_l1, "l2": prepare_l2, "l3": prepare_l3,
                               "l4": prepare_l4, "l5": prepare_l5, "l6": prepare_l6}[backend]
                    fn = prepare(q, k, v, args.causal)
                else:
                    fn = lambda: attention(q, k, v, backend, args.causal)
                metrics = errors(fn(), expected)
                if metrics["failed_elements"]:
                    emit(**row, status="fail", **metrics)
                    failures += 1
                    continue
                result = elapsed(fn, args) if args.mode == "bench" else peak_memory(fn)
                if backend == "l1":
                    result.update(workspace_bytes=score_probability_bytes, workspace_note="已知的两张 FP32 S/P 矩阵")
                elif backend == "l2":
                    result.update(workspace_bytes=0, workspace_note="无额外全局张量；分块位于 shared，在线累积位于寄存器")
                elif backend in ("l3", "l4", "l5", "l6"):
                    result.update(workspace_bytes=0, workspace_note="无额外全局张量；分块和在线输出累积位于 shared")
                emit(**row, status="pass", **metrics, **result)
            except torch.cuda.OutOfMemoryError as exc:
                emit(**row, status="skipped", reason=f"OOM：{exc}")
            except (RuntimeError, NotImplementedError) as exc:
                emit(**row, status="error", reason=str(exc))
                failures += 1
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("env", "verify", "bench", "memory"), default="verify")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--n", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--max-score-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, help="追加到一个 JSONL 文件，省略时只打印")
    parser.add_argument("--summary-only", action="store_true", help="精简终端输出，JSONL 文件仍保存完整记录")
    args = parser.parse_args()
    if min(args.batch, args.heads, *args.n, args.repeats, args.groups, args.max_score_mib) <= 0 or args.warmup < 0:
        parser.error("形状、次数和预算必须为正数，warmup 可为零")
    if args.device == "cpu" and args.mode in ("bench", "memory"):
        parser.error("计时和显存检查仅支持 CUDA；CPU 只做公式验证")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 Python 的 CUDA 不可用；不要把 CPU 结果当作 GPU 验证")
    device = torch.device(args.device)
    precision = configure_precision()
    run_id = uuid.uuid4().hex[:12]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open("a", encoding="utf-8") if args.output else None

    def emit(**row):
        record = {"run_id": run_id, "time_utc": datetime.now(timezone.utc).isoformat(),
                  "dtype": "float16", "causal": args.causal, **row}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if (not args.summary_only or args.mode == "env"
                or record["kind"] in ("verify_summary", "bench", "memory")
                or record.get("status") in ("fail", "error", "skipped")):
            print(line, flush=True)
        if stream:
            stream.write(line + "\n")
            stream.flush()

    try:
        emit(kind="environment", **environment(device, precision), config={
            "mode": args.mode, "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
            "n": args.n, "backends": args.backends, "seed": args.seed, "warmup": args.warmup,
            "repeats": args.repeats, "groups": args.groups, "max_score_mib": args.max_score_mib,
            "tile": None, "warps": None, "timing": "eager_cuda_events_grouped_same_stream",
            "allocation": "PyTorch 路径逐次分配；L1/L2/L3/L4/L5/L6 计时预分配，memory 模式逐次分配",
            "l1": {"threads_per_block": 128, "warps": 4, "max_n": L1_MAX_N,
                   "cuda_flags": ["-O3", "-lineinfo", "-DATTENTION_LAB_L1"]},
            "l2": {**L2_CONFIG, "shared_bytes_declared": 266 * args.head_dim + 1024,
                   "cuda_flags": ["-O3", "-lineinfo", "-DATTENTION_LAB_L2"]},
            "l3": {**L3_CONFIG, "shared_bytes_declared": 224 * args.head_dim + 6272,
                   "cuda_flags": ["-O3", "-lineinfo", "-DATTENTION_LAB_L3"]},
            "l4": {**L4_CONFIG, "shared_bytes_declared": 224 * args.head_dim + 6272,
                   "cuda_flags": ["-O3", "-lineinfo", "-DATTENTION_LAB_L4"]},
            "l5": {**L5_CONFIG, "shared_bytes_declared": 352 * args.head_dim + 6272,
                   "configure": "dynamic_shared_opt_in_before_timing",
                   "cuda_flags": ["-O3", "-lineinfo", "-DATTENTION_LAB_L5"]},
            "l6": {**L6_CONFIG, "shared_bytes_declared": 352 * args.head_dim + 4224,
                   "configure": "dynamic_shared_opt_in_before_timing",
                   "cuda_flags": ["-O3", "-lineinfo", "-DATTENTION_LAB_L6"]}})
        if args.mode == "env":
            return 0
        with torch.inference_mode():
            failures = verify(args, device, emit)
            if failures == 0 and args.mode in ("bench", "memory"):
                failures += measure(args, device, emit)
        return 1 if failures else 0
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    sys.exit(main())
