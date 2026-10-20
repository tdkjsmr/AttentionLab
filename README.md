# AttentionLab

## 如何运行

支持连续存储的 FP16 自注意力输入 `[B,H,N,D]`，D 为 64 或 128；提供 noncausal / causal 前向计算，不含反向与 dropout。
可选后端：`explicit`、PyTorch SDPA `math` / `flash`，以及自定义 CUDA `l1` / `l2` / `l3` / `l4` / `l5`。`flash` 指 PyTorch 后端。

下面以 Linux、RTX 3090 为例。已使用的环境为 Python 3.12.3、PyTorch 2.5.1+cu124、CUDA toolkit 12.4.131。
自定义 CUDA 扩展需要 nvcc、C++ 编译器和 Ninja；首次调用自动编译到 PyTorch 缓存。
L5 要求计算能力至少 8.0；D128 所需的动态 shared 上限由准备入口配置，该配置位于重复计时之外。

基础 Python 已有可用 CUDA PyTorch 时，可以复用其包并创建独立环境：

```bash
git clone https://github.com/tdkjsmr/AttentionLab.git
cd AttentionLab
python -c 'import sys, torch; print(torch.__version__, torch.version.cuda); assert sys.prefix == sys.base_prefix; assert torch.cuda.is_available()'
python -m venv --system-site-packages .venv
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=1
python bench.py --mode env --output results/environment.jsonl
```

若 Ninja 缺失，在当前环境内运行 `python -m pip install ninja`。其他 GPU 需按实际计算能力调整 `TORCH_CUDA_ARCH_LIST`。
复用基础包的环境不锁定依赖；每次运行的 JSONL 会记录实际版本、GPU、提交 SHA 和工作区是否有修改。

先验证 L5 的两个 D 和两种 mask：

```bash
python bench.py --mode verify --backends l5 --summary-only --output results/l5.jsonl
python bench.py --mode verify --backends l5 --causal --summary-only --output results/l5.jsonl
python bench.py --mode verify --backends l5 --head-dim 128 --summary-only --output results/l5.jsonl
python bench.py --mode verify --backends l5 --head-dim 128 --causal --summary-only --output results/l5.jsonl
python bench.py --mode verify --backends l4 l5 --batch 2 --heads 3 --head-dim 128 --causal --summary-only --output results/l5.jsonl
```

每条命令应成功退出，汇总应为 `status=pass`、`failed_cases=0`、`cuda_l5_verified=true`。
前四条覆盖两个 D 和两种 mask，第五条检查多 batch/head 的配对结果。
`head_dim` 和 `causal` 区分配置。未选择的其他后端，其验证标记为 false。
小尺寸 N<=257 与 CPU FP64 参考比较，固定条件为 `abs(error) <= 0.002 + 0.005*abs(reference)`。
`--summary-only` 精简终端显示，JSONL 仍保存环境与逐项结果。

验证通过后进行同输入配对的计时与显存测量：

```bash
python bench.py --mode bench --backends flash l4 l5 --n 128 512 2048 --summary-only --output results/l5.jsonl
python bench.py --mode bench --backends flash l4 l5 --head-dim 128 --n 128 512 2048 --summary-only --output results/l5.jsonl
python bench.py --mode memory --backends flash l4 l5 --n 128 512 2048 --summary-only --output results/l5.jsonl
python bench.py --mode memory --backends flash l4 l5 --head-dim 128 --n 128 512 2048 --summary-only --output results/l5.jsonl
```

- 默认 B=1、H=8、D=64、noncausal；添加 `--head-dim 128` 或 `--causal` 切换配置。
- `--backends` 指定待测后端；省略时选择全部。L1 限制 N<=512，超出记为 skipped。
- `--device cpu --mode verify` 可运行 CPU 公式验证，CUDA 后端明确跳过。
- `bench` / `memory` 都先执行小尺寸验证，失败则不进入测量；大尺寸另与显式 FP32 路径比较。
- `--warmup 10 --repeats 100 --groups 5` 为计时默认值；同 stream 的 CUDA events 测组内每次调用耗时，再取各组中位数。
- PyTorch 后端在每次调用中分配输出及内部缓冲，自定义 CUDA 后端在计时前预分配；`memory` 模式均在调用内分配。耗时包含可能的主机提交间隙，不等同于纯 kernel 时间。
- 显存报告增量峰值 allocated，包含输出、排除公共输入；不使用 reserved 作为峰值。
- `--max-score-mib 512` 限制配对参考所需的理论 S/P 字节数；超预算或 OOM 记 skipped。
- JSONL 按 `run_id` 关联环境和结果。结果追加保存；复现配对测量时应固定完整 SHA，确认 `git_dirty=false`。

采集 L4/L5 的编译资源与理论驻留上限，默认 N=2048。该步骤不需要安装 NSYS/NCU：

```bash
mkdir -p results/profile
for backend in l4 l5; do
  for d in 64 128; do
    python profile_kernel.py --backend "$backend" --head-dim "$d" --output results/profile/resources.jsonl
  done
done
```

资源 JSONL 记录寄存器、local/shared 字节数和理论驻留 CTA 数；理论占用率由 CUDA occupancy API 计算，不等于计数器测得的实际占用率。local 字节数也不能全部解释为寄存器溢出。
入口会先与显式 FP32 路径检查数值，再预热；`--capture` 仅包围目标启动，编译、输出分配和检查均在区间外。
`profile_summary=pass` 只表示入口执行成功，报告中是否有目标 kernel 与有效指标需要单独核对。

使用 Nsight Systems 采集时间线，每个配置记录 10 次目标启动：

```bash
nsys --version
for backend in l4 l5; do
  for d in 64 128; do
    prefix="results/profile/${backend}_d${d}"
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop --output "$prefix" \
      python profile_kernel.py --backend "$backend" --head-dim "$d" --capture --launches 10 \
      --output results/profile/resources.jsonl
    nsys stats --report cuda_gpu_kern_sum --format csv "$prefix.nsys-rep" > "$prefix.nsys.csv"
  done
done
```

需要硬件计数器且环境允许时，再使用 Nsight Compute；每个配置只选取一个目标启动。首次失败就停止后续 NCU 采集：

```bash
ncu --version
for backend in l4 l5; do
  for d in 64 128; do
    prefix="results/profile/${backend}_d${d}"
    if ! ncu --profile-from-start off --set full --launch-count 1 --export "$prefix" \
      python profile_kernel.py --backend "$backend" --head-dim "$d" --capture \
      --output results/profile/resources.jsonl; then
      break 2
    fi
    ncu --import "$prefix.ncu-rep" --page raw --csv > "$prefix.ncu.csv"
  done
done
```

若 NCU 报告 `ERR_NVGPUCTRPERM`，保留错误并停止计数器采集；资源 JSONL 与 NSYS 时间线可单独使用。profiler 会增加执行开销，正式耗时继续使用 `bench.py`；已有报告不自动覆盖，重新采集时使用新的输出前缀。

## 性能测试结果

以下为 2026-10-20 的同输入配对测量，对应提交 `d49795714ac53914ded207f46444e1036eb1795f`，工作区无修改。

环境：RTX 3090 24 GiB（sm86）、驱动 595.71.05、Python 3.12.3、PyTorch 2.5.1+cu124、CUDA toolkit 12.4.131。
输入为 B=1、H=8、FP16、noncausal；eager、无梯度，seed=20261019。关闭 TF32 和可用的降精度归约开关。
计时预热 10 次，每组 100 次，取 5 组中位数。PyTorch flash 逐次分配，L4/L5 在计时前预分配输出。

单次 GPU elapsed（微秒），加速比为同轮 L4 耗时除以 L5 耗时：

| D | N | SDPA flash | L4 | L5 | L4/L5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 30.945 | 16.681 | 10.045 | 1.66 |
| 64 | 512 | 36.731 | 105.615 | 81.244 | 1.30 |
| 64 | 2048 | 155.259 | 1211.413 | 794.040 | 1.53 |
| 128 | 128 | 30.720 | 28.396 | 14.981 | 1.90 |
| 128 | 512 | 41.021 | 246.886 | 176.026 | 1.40 |
| 128 | 2048 | 306.954 | 3183.596 | 1850.470 | 1.72 |

增量峰值 allocated（MiB，包含输出、排除公共输入）：

| D | N | SDPA flash | L4 | L5 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 0.129 | 0.125 | 0.125 |
| 64 | 512 | 2.547 | 0.500 | 0.500 |
| 64 | 2048 | 2.062 | 2.000 | 2.000 |
| 128 | 128 | 0.254 | 0.250 | 0.250 |
| 128 | 512 | 5.047 | 1.000 | 1.000 |
| 128 | 2048 | 4.062 | 4.000 | 4.000 |

D64/D128 计时 run_id 分别为 `4e98279b2704` / `a59c7d8706a2`，显存 run_id 分别为 `0f90743257a4` / `6749fbf4fba6`。
L4/L5 的增量峰值等于输出大小；shared memory 和寄存器不计入这里的 allocated。
L4/L5 使用相同分配口径；与 PyTorch flash 的耗时比较包含分配方式和主机提交开销的差异，不代表纯 kernel 性能比。
这些性能结果覆盖 D64/D128 noncausal；尚无本提交的 profiler 报告。
