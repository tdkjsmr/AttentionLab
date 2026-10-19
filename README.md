# AttentionLab

Dense 自注意力 Prefill forward 的分阶段实验。当前只有 L0，尚未包含自定义 CUDA kernel；阶段编号不对应官方 FlashAttention 的版本号。

## 计算约定

`O = softmax(Q @ K.T / sqrt(D) + mask) @ V`。

- Q/K/V/O：连续的 `[B,H,N,D]`，FP16；Q/K/V 形状相同，D 为 64 或 128。
- dropout 为 0；默认 noncausal，`--causal` 启用包含对角线的下三角 mask。
- 当前仅 forward；不含 backward、GQA、KV 分页、变长 batch 或完整模型。
- 显式路径的矩阵乘法、分数和概率使用 FP32，最后转换为 FP16。测量入口关闭 TF32 和可用的 FP16 降精度归约开关，并记录设置。
- SDPA math / flash 只允许指定的单个后端。后端不可用时报告错误；CPU 模式明确跳过 CUDA flash。
- 不同后端的中间精度、归约次序不同；使用同一 attention 公式不代表位级相等。

## 文件与阶段

| 文件 | 作用 |
| --- | --- |
| `attention.py` | 显式公式、CPU FP64 逐行参考、强制 SDPA math/flash |
| `bench.py` | 环境记录、正确性、计时和计时外显存检查 |
| `.gitignore` | 排除本地环境、缓存和机器相关结果 |

L0 建立公式与统一测量入口。下一阶段 L1 才增加 QK、稳定 softmax、PV 三个可读 CUDA kernel，并显式保存 N*N 中间矩阵；当前没有预建后续实现。

## L0：先理解一行 attention

忽略 B、H，Q/K/V 是 `[N,D]`，S/P 是 `[N,N]`，O 是 `[N,D]`。

```text
Q 的第 i 行 × K 的各行 → S 的第 i 行
S 的第 i 行经 softmax → 对所有 key 的权重
这些权重 × V 的各行 → O 的第 i 行
```

softmax 沿 key 维计算，每行权重和为 1。若 Q 全为零，S 每项为零，但 `exp(0)=1`。
例如 N=3 时，`softmax([0,0,0])=[1/3,1/3,1/3]`。
若 V 的三行为 `[1,2]`、`[3,4]`、`[5,6]`，noncausal 输出每行都是 `[3,4]`。
causal 情况下则分别对允许访问的 V 前缀求均值。

建议先阅读 `explicit_attention` 的三个计算步骤，再看 `reference_fp64`；其余测量代码按需阅读。

## 环境

需要 Python、PyTorch，以及提供 `torch.nn.attention.sdpa_kernel` 的版本。CUDA 验证需要可用的 CUDA PyTorch 与 GPU。
L0 不编译扩展，不要求安装官方 flash-attn；`nvcc` 不存在时版本字段记录为 null。
尚未固定或验证云端依赖版本，实际版本由每次运行的 environment 记录给出。不要据此假定最新 PyTorch 与现有驱动兼容。

优先复用已确认可用的云镜像所提供的 PyTorch，但为本仓库建立独立环境，不对已有项目环境执行安装。
以下方式让独立环境读取当前基础解释器的包，前提是当前解释器不属于其他项目的 venv，且其 torch/CUDA 已检查可用：

```bash
python -c 'import sys, torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert sys.prefix == sys.base_prefix, "先退出已有项目的 venv，再检查基础解释器"; assert torch.cuda.is_available(), "先确认 CUDA 环境"'
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -c 'import torch; from torch.nn.attention import sdpa_kernel; print(torch.__version__)'
```

基础环境没有可用 torch 时，先记录 Python、GPU、驱动与 CUDA 信息，再选择兼容安装版本。这里不预先给出未经环境验证的安装命令。
该方式继承基础包，不是锁定依赖的独立副本；基础包改变也会影响它。首轮运行成功后再固定实际依赖记录。

## 运行

远端提交到位后，先 clone 或在已有检出中 fetch，再切换到双方确认的完整 SHA。
每次结果中的 `git_sha` 应一致，正式配对结果要求 `git_dirty=false`。

```bash
git clone https://github.com/tdkjsmr/AttentionLab.git
cd AttentionLab
git rev-parse HEAD
nvidia-smi
python bench.py --mode env --output results/l0.jsonl
python bench.py --mode verify --output results/l0.jsonl
```

先通过 noncausal，再检查 causal。随后运行代表形状的计时与显存检查；每次测量也会先执行小尺寸验证，失败时不进入测量。

```bash
python bench.py --mode verify --causal --output results/l0.jsonl
python bench.py --mode bench --n 128 512 2048 --output results/l0.jsonl
python bench.py --mode memory --n 128 512 2048 --output results/l0.jsonl
```

先分析 D64，再按需用 `--head-dim 128` 检查 D128。无需第一轮运行所有参数组合。
CPU 环境可以运行 `python bench.py --device cpu --mode verify`，该结果不覆盖 CUDA 和 flash，也不输出 GPU 性能。
默认比较 explicit/math/flash；可用 `--backends explicit math` 定点排查，但这不算完成 flash 验证。

## 正确性口径

- 所有后端使用同一组 CPU 生成、转换到 FP16 后的输入，记录 seed 与每个 case 的实际 seed。
- 小尺寸 CPU FP64 参考逐 query 计算，手动减最大值、指数、归一化和加权求和，不调用 SDPA 或待测显式实现。入口固定 CPU 计算为单线程并记录，避免小矩阵线程调度开销。
- N 覆盖 1、31/32/33、63/64/65、127/128/129；这些是边界探针，不代表 SDPA 内部 tile 已知。
- 另测零 Q 的均匀权重和 Q/K 各放大 4 倍的分数。零 Q 参考还与普通均值或 causal 前缀均值做解析交叉检查。
- 固定逐元素条件 `abs(actual-reference) <= 0.002 + 0.005*abs(reference)`，报告 max_abs、归一化 L2 和失败元素数；非有限输出必失败。
- FP16 在 1 附近的单位舍入误差约为 4.88e-4；上述绝对预算约为其 4 倍，另留相对预算覆盖归约和融合差异。这是本实验的验收预算，不是任意输入的误差定理；失败后定位原因，不调宽阈值。
- 大尺寸测量前再与显式 FP32、FP16 输出路径检查一致性，明确区别于小尺寸独立 FP64 参考。explicit 的自比较不构成额外正确性证据。
- CUDA 上所选后端不可用或数值失败都会返回非零退出码；CPU flash 明确记 skipped。OOM 或预算限制记 skipped，不产生性能数字。

## 测量口径

- 默认 B1/H8/D64，N128/512/2048；eager、无梯度、同一个 CUDA stream。
- 预热 10 次，CUDA events 包围每组 100 次调用，组末等待，取 5 组每次耗时的中位数，同时保存各组值。
- 输入生成、FP64 参考、打印、预热不计时；输出分配、显式 FP32 转换和中间张量分配计入各路径调用。
- 短调用可能包含 Python 提交空洞，字段称 GPU elapsed，不声称纯内核时间。所有路径均为 eager，不混入 Graph 结果。
- 显式路径的 FP32 运算与融合 flash 的中间精度不同，结果用于解释路径成本，不把耗时比直接归因为某一种优化。
- memory 模式在计时之外测增量峰值 allocated，公共输入分配计入 baseline；不混 reserved，不在循环中 empty_cache。
- PyTorch 没有通过此处公共接口暴露实际内部 workspace，`workspace_bytes=null`。S/P 的 `2*B*H*N*N*4` 只表示两张 FP32 矩阵的理论字节数，不是总峰值或实测 DRAM 流量。
- 默认 S/P 理论预算 512 MiB，超出则跳过代表形状；实际 OOM 同样记录。此预算也限制配对参考路径，因此可能连同本来可运行的 flash 一起跳过。
- 每次运行只追加一种 JSONL 结果，通过 run_id 关联 environment 与数据行；记录 SHA、dirty、GPU、torch/CUDA/nvcc、精度、mask、形状、预热与次数。当前 tile/warps 未由实验控制，记 null。

## 来源

公式、SDPA 后端控制与数值边界参考：

- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch sdpa_kernel](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)

当前实现依据标准公式独立编写，尚未引入第三方 kernel 源码。后续若复制或改编代码，保留其许可证、版权与具体来源。
算法与论文中的已有思想不作为本仓库原创贡献声明。
