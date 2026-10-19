# AttentionLab

## 如何运行

支持连续存储的 FP16 自注意力输入 `[B,H,N,D]`，D 为 64 或 128；提供 noncausal / causal 前向计算，不含反向与 dropout。
可选后端：`explicit`、PyTorch SDPA `math` / `flash`，以及自定义 CUDA `l1` / `l2` / `l3`。`flash` 指 PyTorch 后端。

下面以 Linux、RTX 3090 为例。已使用的环境为 Python 3.12.3、PyTorch 2.5.1+cu124、CUDA toolkit 12.4.131。
自定义 CUDA 扩展需要 nvcc、C++ 编译器和 Ninja；首次调用自动编译到 PyTorch 缓存。

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

先验证 L3 的两个 D 和两种 mask：

```bash
python bench.py --mode verify --backends l3 --summary-only --output results/l3.jsonl
python bench.py --mode verify --backends l3 --causal --summary-only --output results/l3.jsonl
python bench.py --mode verify --backends l3 --head-dim 128 --summary-only --output results/l3.jsonl
python bench.py --mode verify --backends l3 --head-dim 128 --causal --summary-only --output results/l3.jsonl
```

每条命令应成功退出，汇总应为 `status=pass`、`failed_cases=0`、`cuda_l3_verified=true`。
`head_dim` 和 `causal` 区分四种配置。未选择的其他后端，其验证标记为 false。
小尺寸与 CPU FP64 参考比较，固定条件为 `abs(error) <= 0.002 + 0.005*abs(reference)`。
`--summary-only` 精简终端显示，JSONL 仍保存环境与逐项结果。

验证通过后进行同输入配对的计时与显存测量：

```bash
python bench.py --mode bench --backends explicit math flash l2 l3 --n 128 512 2048 --summary-only --output results/l3.jsonl
python bench.py --mode memory --backends explicit math flash l2 l3 --n 128 512 2048 --summary-only --output results/l3.jsonl
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

## 性能测试结果

以下为 2026-10-19 的历史 L0 基线，对应提交 `c274c5e74fba965e67cf1d618a01bff57cd85e38`，不代表 L1/L2/L3 的性能。

环境：RTX 3090 24 GiB（sm86）、驱动 595.71.05、Python 3.12.3、PyTorch 2.5.1+cu124、CUDA toolkit 12.4.131。
输入为 B=1、H=8、D=64、FP16、noncausal；eager、无梯度。关闭 TF32 和可用的降精度归约开关。
计时预热 10 次，每组 100 次，取 5 组中位数；三条路径都包含各自调用中的分配。

单次 GPU elapsed（微秒）：

| N | explicit | SDPA math | SDPA flash |
| ---: | ---: | ---: | ---: |
| 128 | 82.412 | 143.135 | 30.464 |
| 512 | 151.460 | 189.184 | 36.116 |
| 2048 | 1549.629 | 2077.962 | 159.242 |

增量峰值 allocated（MiB，包含输出、排除公共输入）：

| N | explicit | SDPA math | SDPA flash | 输出大小 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.500 | 2.627 | 0.129 | 0.125 |
| 512 | 18.000 | 30.005 | 2.547 | 0.500 |
| 2048 | 264.000 | 432.017 | 2.062 | 2.000 |

显式实现采用 FP32 中间计算，各 SDPA 后端的中间精度和归约次序可能不同；这些数字比较完整调用路径，不能将耗时比直接解释为某项 kernel 优化的收益。
L1/L2/L3 的性能数据尚未发布，后续实测后补充。
