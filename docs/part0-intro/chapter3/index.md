---
title: "第3章 第一个程序 + Roofline 心智模型"
description: "Hello GPU 第3章 · vector add 跑通、建立性能上限直觉、benchmark 习惯"
---

# 第3章 第一个程序 + Roofline 心智模型

## 本章导读

> 前面两章我们铺开了环境验证和 GPU 体系结构两张地图。现在，地图已经在你手上了——终于到了**写一点代码、量一组数字**的时候。本章会做三件事：跑通第一个手写的 HIP kernel（vector add）、建立一个可复用的 baseline benchmark、再用 Roofline 心智模型把"这个算子离硬件极限有多远"的直觉建起来。vector add 会贯穿整个 Part 1 profiling 篇，所以这章是后面所有实验的起点。

本章对应代码在：

```text
code/part0-intro/
├── pyproject.toml
├── uv.lock
├── activate-rocm.sh
└── chapter3/
    ├── vector_add.hip
    └── benchmark_vector_add.py
```

## 3.1 从已经验证的环境开始

在写第一行 GPU 代码之前，有一件事必须确认：环境是通的。好在这件事[第 1 章](../chapter1/index.md)已经帮你做完了——三道环境验证门（`rocminfo` 能看到 GPU、PyTorch ROCm 能跑 GPU tensor、最小 HIP 程序能编译运行）都已经通过。如果你还没做，请先回去跑完那三道门。

进入本篇环境：

```bash
cd code/part0-intro
uv sync
source ./activate-rocm.sh
```

如果这里无法激活环境，先回到[第 1 章环境准备](../chapter1/index.md)排查 `uv` 环境、ROCm wheel 和 `_rocm_sdk_devel` 初始化问题。

## 3.2 跑通 vector add

环境就绪，开始写代码。一个最小但完整的 HIP 程序通常包含六步：Host 准备数据、Device 分配显存、Host 到 Device 拷贝、启动 kernel、Device 到 Host 拷回、检查结果。这六步构成了所有 GPU 程序的骨架，后面的 Reduction、Softmax、Matmul 无论多复杂，骨架都是这六步。

::: figure fig-vec-add-data-path
```mermaid
flowchart LR
    A[Host 输入] --> B[hipMalloc]
    B --> C[hipMemcpy H2D]
    C --> D[Kernel Launch]
    D --> E[hipMemcpy D2H]
    E --> F[校验结果]
```

最小 HIP Vector Add 程序的数据路径
:::

完整的 `vector_add.hip` 文件在[第 1 章 1.6 节](../chapter1/index.md#_1-6-验证最小-hip-程序)已经作为环境验证出现过，这里不重复贴。**真正值得重点看的是中间那段 kernel**，它才是跑在 GPU 上的代码：

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
```

这段 kernel 的映射关系很直接：一个 GPU 线程负责一个元素。`blockIdx.x * blockDim.x + threadIdx.x` 计算出当前线程负责的全局下标，`if (idx < n)` 用来处理最后一个 block 可能越界的情况。

新语法速查：

- `__global__`：告诉编译器"这是一个 GPU 函数，由 CPU 调用、在 GPU 上运行"；
- `<<<blocks, threads>>>`：HIP / CUDA 特有的 kernel 启动语法，`blocks` 是要启动多少组，`threads` 是每组多少个线程；
- `blockIdx.x` / `threadIdx.x`：每个 GPU 线程拿到的"工号"，用它来算自己负责数组里的哪个位置。

换成大白话：`vector_add<<<blocks, threads>>>(...)` 就是在 GPU 上同时叫起 `blocks × threads` 个工人，每个工人执行一次 `vector_add` 函数（如 @fig-block-thread-hierarchy 所示）。

::: figure fig-block-thread-hierarchy
![Block 与 Thread 的层级关系](./images/block-thread-hierarchy.png)

Grid → Block → Thread 的层级，以及 blockIdx/threadIdx 如何算出全局下标
:::

编译并运行：

```bash
cd chapter3
hipcc vector_add.hip -O2 -o vector_add && echo "compile_status: PASS"
./vector_add
```

<details>
<summary>🚧 待实测：vector add 运行结果</summary>

```text
（机器就绪后在此粘贴实际输出。预期 device_name 显示 9070XT，max_error: 0，status: PASS）
```

</details>

看到 `status: PASS` 后，你已经跑通了第一段真正由自己编译的 GPU kernel——欢迎正式进入 GPU 编程的世界。

## 3.3 建立 baseline benchmark

这一节加入一个很小的 benchmark。它不是为了证明 Vector Add 有多快，而是为了提前建立后续章节会反复使用的习惯：固定输入规模、先 warmup、重复运行多次、记录 mean / median / min。

完整代码在下面的折叠块里。如果你只想先看懂大意，记住三件事（如 @fig-benchmark-warmup-repeat-sync 所示）：

1. 正式计时前先 warmup 5 次，让 GPU 进入比较稳定的状态；
2. 正式跑 30 次，每次用 `torch.cuda.Event` 量 GPU 真正执行完成的时间；
3. 最后用最小值估算带宽，因为最小值更像"没被外部干扰"的那次。

::: figure fig-benchmark-warmup-repeat-sync
![Benchmark 的赛前准备](./images/benchmark-warmup-repeat-sync.png)

准确 benchmark 前先 warmup、多次 repeat，并在同步后计时
:::

<details>
<summary>代码：benchmark_vector_add.py</summary>

```python
import argparse
import statistics
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark torch vector add on CPU and ROCm GPU.")
    parser.add_argument("--size", type=int, default=1 << 24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    return parser.parse_args()


def benchmark_cpu(size, warmup, repeat):
    a = torch.ones(size, dtype=torch.float32)
    b = torch.full((size,), 2.0, dtype=torch.float32)

    for _ in range(warmup):
        c = a + b
    _ = c.sum().item()

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        c = a + b
        _ = c.sum().item()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return times


def benchmark_gpu(size, warmup, repeat):
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch ROCm backend is not available")

    a = torch.ones(size, device="cuda", dtype=torch.float32)
    b = torch.full((size,), 2.0, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        c = a + b
    torch.cuda.synchronize()
    _ = c.sum().item()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c = a + b
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    _ = c.sum().item()
    return times


def summarize(name, times, size):
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    min_ms = min(times)
    bytes_moved = size * 3 * 4
    bandwidth_gb_s = bytes_moved / (min_ms / 1000) / 1e9

    print(f"{name}_mean_ms: {mean_ms:.6f}")
    print(f"{name}_median_ms: {median_ms:.6f}")
    print(f"{name}_min_ms: {min_ms:.6f}")
    print(f"{name}_bandwidth_gb_s_by_min: {bandwidth_gb_s:.6f}")


def main():
    args = parse_args()
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_name: {torch.cuda.get_device_name(0)}")
    print(f"vector_size: {args.size}")
    print(f"warmup: {args.warmup}")
    print(f"repeat: {args.repeat}")

    cpu_times = benchmark_cpu(args.size, args.warmup, args.repeat)
    gpu_times = benchmark_gpu(args.size, args.warmup, args.repeat)

    summarize("cpu", cpu_times, args.size)
    summarize("gpu", gpu_times, args.size)
    print("status: PASS")


if __name__ == "__main__":
    main()
```

</details>

其中 GPU 计时最关键的是使用 event 并在每轮后同步：

```python
start.record()
c = a + b
end.record()
torch.cuda.synchronize()
times.append(start.elapsed_time(end))
```

否则 CPU 端可能只是把任务提交出去，计到的不是 GPU 真正执行完成的时间。

运行：

```bash
python benchmark_vector_add.py
```

<details>
<summary>🚧 待实测：Vector Add baseline benchmark @ 9070XT + ROCm 6.4.x</summary>

```text
（机器就绪后在此粘贴实际输出。预期会看到 torch、cuda_available、device_name、
vector_size、warmup、repeat，以及 cpu/gpu 各自的 mean/median/min 延迟和带宽估算）
```

</details>

这里的带宽估算使用的是 Vector Add 的最简单数据量模型。一次 Vector Add 要读 `a`、读 `b`、写 `c`，一共经过 3 个数组；每个元素是 `float32`，也就是 4 字节。所以一次完整 Vector Add 搬动的数据量是：

```text
bytes_moved = vector_size × 3 × 4
```

> 🚧 跑出数字后，把 GPU 的带宽估算填到这里，并和下一节的 Roofline 上限对比。

## 3.4 Roofline 心智模型

有了 baseline 数字，现在把它和理论上限对比一下，建立"这个算子离硬件极限有多远"的直觉。

[第 2 章 2.7 节](../chapter2/index.md#_2-7-roofline-的硬件来源) 我们建立了 Roofline 模型：任何 kernel 的实际性能都被**算力上限**和**带宽上限**两条线卡住。现在用 vector add 实测一下它落在哪。

先算 vector add 的算术强度：

```text
对每个 float32 元素：
  读 a[i]：4 Byte
  读 b[i]：4 Byte
  写 c[i]：4 Byte
  做 1 次加法：1 FLOP

算术强度 = FLOP / Byte = 1 / 12 ≈ 0.083 FLOP/Byte
```

算术强度极低（约 0.083 FLOP/Byte），这意味着 vector add **必然落在 Roofline 的斜线那一侧——它是 memory-bound 的**。换句话说，vector add 慢不慢，几乎完全取决于显存带宽，和算力无关。

现在把实测带宽和理论上限对比：

| 项目 | 数值 |
| ---- | ---- |
| vector add 算术强度 | ~0.083 FLOP/Byte（memory-bound）|
| 🚧 实测有效带宽 | 待 job 填充（9070XT）|
| 🚧 理论峰值带宽 | ~760 GB/s（标称，以 micro-benchmark 实测为准）|
| 🚧 带宽利用率 | 实测 / 理论 |

> 🚧 上表的数字需在 9070XT 实验机就绪后实测回填。填完后你能直观看到：vector add 这个最简单的算子，离硬件带宽上限有多远。通常 PyTorch elementwise 能跑到理论带宽的相当比例——因为它的访存模式非常友好（完全合并）。

这个对比建立了一个重要的直觉：**判断一个算子优化得好不好，不是看绝对延迟，而是看它离 Roofline 上限有多远**。这个直觉会在整个 Part 1 profiling 篇反复用到——[第 6 章](../../part1-profiling/chapter6/index.md) 会把算子点正式画到 Roofline 曲线上。

## 3.5 留下实验底稿

跑完前面三段命令，你大概觉得事情已经做完了——其实还没有。**性能工作真正麻烦的一刻，往往不是第一次没跑快，而是过几天回头看时，你自己也说不清当时跑了哪个版本、用了什么输入、那个数字到底是怎么量出来的。** 没留记录的实验，三天后基本等于白做。

所以从这第一个 GPU 程序开始，建议你养成一个小习惯：**每跑完一组实验，顺手把目标、命令、结果记到同一个地方**。在哪里记并不重要——一个 markdown 文件、一份 notebook 都行；重要的是这份记录能在几天后让你（或者别人）一眼看回当初做了什么。

够用的结构其实只有三段：**目标 → 流程 → 结论**。下面这个模板可以直接拿去用：

````markdown
# 实验记录：第一个 GPU 程序

## 实验目标

验证 PyTorch ROCm、最小 HIP kernel 和 Vector Add baseline benchmark 是否能跑通。

## 实测流程

（贴出可以从章节目录直接复制运行的命令）

```bash
cd code/part0-intro
source ./activate-rocm.sh
python chapter1/check_torch_rocm.py
cd chapter3
hipcc vector_add.hip -O2 -o vector_add
./vector_add
python benchmark_vector_add.py
```

## 实测结论

| 项目 | 数值 |
| ---- | ---- |
| 硬件 | 🚧 9070XT + ROCm 6.4.x（待填）|
| 输入规模 | 🚧 待填 |
| GPU min 延迟 | 🚧 待填 |
| GPU 估算带宽 | 🚧 待填 |
| status | PASS |
````

看起来朴素，但半年后你回头翻这些记录，会非常感谢现在的自己。后面这本教程会一路写到 Reduction、Softmax、Matmul、Attention，外加 rocprof / Omniperf 一堆 profiling 实验——等到 kernel 版本越积越多、benchmark 配置越改越乱时，**能不能一眼看回当初跑过什么**，往往就是"顺利继续"和"回头返工"的分界线。

试一试：把 `--size` 从默认的 `1 << 24` 改成 `1 << 20` 和 `1 << 26`，分别再跑一次 benchmark，把每次的硬件、输入规模、GPU min 延迟和估算带宽随手记到你的实验记录里。先猜一下——GPU 带宽会一直变大、一直变小，还是先升后降？这道题没有标准答案，目的是让你亲手建立"输入规模 vs 性能"的第一感觉，后面 Part 1 会反复用到。

## 本章小结

- 本章在 `part0-intro` 环境里跑通了第一个手写 HIP Vector Add kernel。
- 第一个 HIP kernel 使用"一线程处理一个元素"的最简单映射方式，方便理解 block、thread 和全局下标。
- baseline benchmark 使用 warmup + repeat，并用 GPU event 计时，避免只量到 CPU 提交开销。
- vector add 的算术强度极低（~0.083 FLOP/Byte），必然是 memory-bound——它的性能几乎完全取决于显存带宽。
- 用实测带宽和 Roofline 理论上限对比，建立了"算子离极限有多远"的直觉，这个直觉会在整个 Part 1 反复用到。
- 下一章进入 Part 1 profiling 篇，先系统学怎么量准数字（benchmark 与可信计时）。

## 延伸阅读

- [ROCm HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [ROCm System Management Interface](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/)
