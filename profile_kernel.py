"""验证并预热后，只采集选定的 L4/L5/L6 kernel；另记录无需计数器的资源信息。"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import torch
import torch.cuda.nvtx
import torch.cuda.profiler

from attention import _load_cuda, attention, prepare_l4, prepare_l5, prepare_l6
from bench import configure_precision, environment, errors, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("l4", "l5", "l6"), default="l6")
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--launches", type=int, default=1)
    parser.add_argument("--capture", action="store_true", help="启用 CUDA profiler 起止标记与 NVTX 范围")
    parser.add_argument("--output", type=Path, help="追加环境、资源和采集入口执行状态到 JSONL")
    args = parser.parse_args()
    if min(args.n, args.batch, args.heads, args.launches) <= 0 or args.warmup < 0:
        parser.error("形状和启动次数必须为正数，warmup 可为零")
    if not torch.cuda.is_available():
        parser.error("此入口仅支持 CUDA")
    run_id = uuid.uuid4().hex[:12]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open("a", encoding="utf-8") if args.output else None

    def emit(**row):
        record = {"run_id": run_id, "time_utc": datetime.now(timezone.utc).isoformat(),
                  "backend": args.backend, "dtype": "float16", "causal": args.causal, **row}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        if stream:
            stream.write(line + "\n")
            stream.flush()

    try:
        device = torch.device("cuda")
        precision = configure_precision()
        shape = (args.batch, args.heads, args.n, args.head_dim)
        config = {"shape": list(shape), "seed": args.seed, "warmup": args.warmup,
                  "launches": args.launches, "capture": args.capture,
                  "nvtx_range": "attention_profile" if args.capture else None,
                  "allocation": "preallocated", "reference": "explicit_fp32_with_fp16_output"}
        emit(kind="environment", **environment(device, precision), config=config)
        with torch.inference_mode():
            q, k, v = inputs(shape, device, args.seed)
            prepare = {"l4": prepare_l4, "l5": prepare_l5, "l6": prepare_l6}[args.backend]
            run = prepare(q, k, v, args.causal)
            expected = attention(q, k, v, "explicit", args.causal)
            metrics = errors(run(), expected)
            if metrics["failed_elements"]:
                emit(kind="profile_summary", status="fail", reason="采集前数值检查失败", **metrics)
                return 1
            del expected
            resources = _load_cuda(args.backend).resources(q)
            # 这是资源约束下的理论上限，不是硬件计数器测得的实际占用率。
            resources["theoretical_occupancy_fraction"] = (
                resources["max_active_threads_per_sm"] / resources["device_max_threads_per_sm"])
            emit(kind="kernel_resources", status="pass", shape=list(shape),
                 cta_count=args.batch * args.heads * ((args.n + 15) // 16),
                 resources=resources, **metrics)
            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()

            if args.capture:
                torch.cuda.profiler.start()
            try:
                scope = torch.cuda.nvtx.range("attention_profile") if args.capture else nullcontext()
                with scope:
                    for _ in range(args.launches):
                        run()
                    torch.cuda.synchronize()
            finally:
                if args.capture:
                    torch.cuda.profiler.stop()
            # 此状态仅表示入口执行成功，外部报告是否含有效采集数据需单独核对。
            emit(kind="profile_summary", status="pass", shape=list(shape),
                 target_launches=args.launches, capture_requested=args.capture,
                 note="此入口不报告性能数字；正式耗时使用 bench.py")
        return 0
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
