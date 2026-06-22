---
title: "第2章 GPU 体系结构速通"
description: "Hello GPU 第2章 · CU/Wavefront/LDS/寄存器/显存层次，RDNA4 视角，不讲 MFMA/CDNA/HBM"
---

# 第2章 GPU 体系结构速通

## 本章导读

> 本章建立后续优化会反复用到的 GPU 硬件最小模型，但只讲 9070XT（RDNA4）用得到的部分。读完后，你应该能用自己的话讲清楚一个 kernel 从 launch 到执行经过哪些硬件单元，并知道 LDS、寄存器、wavefront 为什么是优化的核心资源。

[第 1 章](../chapter1/index.md) 我们跑通了环境，确认了 ROCm 能看到 GPU、PyTorch 能跑、最小 HIP 程序能编译。这一章把镜头继续往下推，对准链路最底下、也是后面最常被点名的那一层：**GPU 这块芯片本身**。

接下来你会反复在 profiler 报告、HIP 编译警告、AMD 工程博客里看到这些词：CU、SIMD、Wavefront、VGPR、SGPR、LDS、WMMA。第一次读它们像在背单词，但一旦把它们组装成一张完整的硬件图，后面看 rocprof 输出、Triton autotune log 时，你会突然发现指标和指标之间的因果关系自己浮出来了。

> **阅读方式提示**：本章是硬件概念章节，不要求你执行命令，也不要求你现在能写 HIP kernel。第一次阅读只需要建立一条主线——一个 kernel launch 后，会被拆成 workgroup / wavefront，最终落到 SIMD lane 上执行；VGPR、SGPR、LDS 会限制同时能跑多少 wave。其它术语第一次看不进去都没关系，下面这张表给你"现在掌握到什么程度"的尺子。

| 类型 | 术语 | 现在掌握到什么程度 |
| ---- | ---- | ---- |
| 本章必须懂 | CU、SIMD、Wavefront、VGPR、SGPR、LDS | 能说出它们在 GPU 执行路径中的位置和作用 |
| 先知道名字 | WMMA、Roofline、occupancy | 知道它们和性能分析有关，后面会实测 |
| 后续再深入 | EXEC mask、bank conflict、intrinsic、BLOCK_M/N/K | 现在不用掌握细节，实验章节会展开 |

## 2.1 GPU 为什么能并行

如果你刚从 CPU 思维切到 GPU 思维，可以先用一张图建立直觉：CPU 像少数几个擅长复杂判断的工人，GPU 像一大群动作统一的工人，适合把同一种计算同时铺到大量数据上。

::: figure fig-cpu-vs-gpu-workers
![CPU vs GPU 工人对比](./images/cpu-vs-gpu-workers.png)

CPU 更适合复杂控制，GPU 更适合大规模同构并行计算
:::

CPU 的设计目标是**低延迟**——少量核心，每个核心都有大量缓存、分支预测、乱序执行，擅长处理复杂的控制逻辑。GPU 的设计目标是**高吞吐**——大量核心（在 AMD 上叫 CU），每个核心能同时处理成百上千个线程，擅长把同一种计算铺到大量数据上。

这就是为什么 GPU 适合做矩阵乘、卷积、attention 这类「同样的计算在大量数据上重复」的任务。本书后面写的每一个算子，本质上都是在回答一个问题：**怎么把计算高效地铺到这成千上万个线程上**。

## 2.2 Compute Unit（CU）的内部结构

这一节剖开"GPU 的最小车间"——Compute Unit（计算单元，简称 CU），看清楚 kernel 真正落到哪些执行单元上。

把整块 GPU 想象成一座大型工厂。从上往下看：芯片 → Shader Engine（着色引擎，分区）→ 一组 Workgroup Processor（WGP，工作组处理器）/ 一组 CU → 每个 CU 里若干 SIMD（单指令多数据执行管线）→ 每个 SIMD 里若干 lane（向量通道）。一个 kernel launch 出去，最终落到这些 lane 上完成乘加和访存。

写 HIP kernel 时看到的 thread / block / shared memory 这些概念，并不是凭空的抽象——它们都对应着上面这条硬件路径上的某一格。先把对照关系记下来，后面读代码就不会"硬件词"和"编程词"两套打架：

| 写代码时看到 | 硬件上大致对应 | 现在怎么理解 |
| ---- | ---- | ---- |
| thread / work-item | 一个 lane 上的一份执行 | 一个数据位置上的工作 |
| warp / wavefront | 一组同步执行的 work-item | AMD 上叫 wavefront；RDNA 默认 32 |
| block / workgroup | 一组 wave（同一 WGP 或同一 CU 上） | 块内 wave 之间可以共享 LDS |
| shared memory | LDS | block 内共享的高速 scratchpad |
| register | VGPR / SGPR | 每个线程或每个 wave 的临时变量空间 |

RDNA 体系下 CU 不是孤立的，而是两两组成一个 WGP；workgroup 内部的 wave（也叫 wavefront，波前）会被发射到同一个 WGP 的四条 SIMD32 上，并且可以共享 LDS。这是 RDNA 1 引入、RDNA 2/3/4 一路保留的核心组织方式。

如 @fig-wgp-simd-layout 所示，可以把 RDNA 一个 WGP 内部画成下面的层级：

::: figure fig-wgp-simd-layout
```mermaid
flowchart TB
    WGP[WGP · Workgroup Processor]
    WGP --> CU0[CU 0]
    WGP --> CU1[CU 1]
    CU0 --> S0[SIMD32 #0]
    CU0 --> S1[SIMD32 #1]
    CU1 --> S2[SIMD32 #2]
    CU1 --> S3[SIMD32 #3]
    S0 --> VR0[VGPR File]
    S0 --> VALU0[VALU · 32 lanes]
    S0 --> SALU0[SALU + Scalar regs]
    S0 --> BR0[Branch / Message]
    WGP --> LDS[LDS · 共享内存 / 最多 64 KB]
    WGP --> L0[L0 / Vector L1 / Texture]
```

RDNA 体系下 WGP / CU / SIMD 的层级
:::

如果觉得 Mermaid 层级图还是太抽象，可以对照 @fig-gpu-factory 的工厂类比看：GPU 不是"一大堆散乱线程"，而是按 CU / Workgroup / Wavefront / Lane 一层层组织起来的并行工厂。

::: figure fig-gpu-factory
![GPU 工厂内部结构](./images/gpu-factory-cu-wavefront.png)

GPU 工厂里 CU、Workgroup、Wavefront、Lane 之间的层级关系
:::

要看懂这张图，记住几个关键单元就够了：

- **VALU（Vector ALU，向量算术逻辑单元）**：每条 SIMD 一组，宽度 32 lane（RDNA）。kernel 里"每个线程做一次乘加"最终就落在它身上。
- **SALU（Scalar ALU，标量算术单元）**：每个 wavefront 共享一组标量执行单元和 SGPR（标量寄存器堆）。控制流（if / loop / 分支）、地址常量、wave 级广播信息都走这里。
- **Branch / Message Unit**：处理跳转、栈、s_waitcnt（等待访存完成）、跨 wave 通信等控制信号。
- **VGPR File（Vector 寄存器堆）**：每个 lane 私有一组 32 位寄存器，是 GPU 上"一线"的存储——SIMD 真正吃饭用的本子。
- **LDS（Local Data Share，本地共享内存）**：CU 或 WGP 内部一块软件可控的高速 scratchpad，多数 RDNA 上一个 workgroup 最多可见 64 KB，wavefront 之间通过它做 block 内归约和 tile 复用。
- **L0 / Vector L1 / Texture cache**：和 SIMD 配套的近端缓存。

把这些单元摆在一起，CU 的"分工"就很清楚了：SALU 决定**走哪条路**，VALU 决定**算什么**，LDS / 缓存决定**数据从哪里来**，VGPR/SGPR 决定**多少东西能同时在飞**。后面三节会逐一展开后两件事。

## 2.3 Wavefront 与 SIMT 执行

这一节讲 GPU 是怎么"一群线程一起走"的——wavefront 的概念，以及它对分支、访存的影响。

先记一个数字：**RDNA 上一个 wave 通常是 32 个 work-item（也叫 wave32）**，但也支持 wave64 模式。为什么是 32？因为它对应硬件 SIMD 的宽度——RDNA 的 SIMD 是 SIMD32，单周期发射就能覆盖一个 wave32。

::: figure fig-wave-issue-timing
```mermaid
sequenceDiagram
    participant CU as RDNA SIMD32
    Note over CU: 一条 v_add_f32 指令
    CU->>CU: cycle t · 32 lanes 同时算
    Note over CU: wave32 与 SIMD32 同宽，单周期完成
```

同一条向量指令在 RDNA SIMD32（wave32）上的执行节奏
:::

无论 wave 多宽，**同一个 wave 内部的所有 lane 永远共享同一个程序计数器（PC）**——这就是 SIMT（Single Instruction Multiple Threads，单指令多线程）执行模型。它带来一个直接后果：**分支发散（branch divergence）会让某些 lane 被强制空转**。

考虑这样一段代码：

```cpp
// 简化的 HIP kernel 片段
if (threadIdx.x < 16) {
    // 走 A 路
    compute_a(data);
} else {
    // 走 B 路
    compute_b(data);
}
```

在 wave32 模式下，前 16 个 lane 走 A、后 16 个 lane 走 B。硬件做不到"两路同时跑"——它会先用 EXEC mask（execution mask，执行掩码）把后 16 个 lane 屏蔽掉跑完 A，再把前 16 个屏蔽掉跑完 B。实际花费的时间是两条路径之和，而不是更长那条。如果一个 wave 内部有 N 个互不相同的分支去向，最坏情况就是 N 倍代价——所以"让同一个 wave 里的 lane 走相同分支"是 GPU 算子的基本功。

> **如果你还没写过 HIP kernel，可以这样理解**：同一组一起出发的 32 个线程最好走同一条路；如果一半走 A、一半走 B，硬件会先把走 B 那一半"暂停"跑完 A，再反过来跑完 B。被暂停的 lane 不是"另一个 CPU 核去做别的事"，而是真的空着等。EXEC mask 是硬件自己维护的开关，**不需要你在代码里手动控制**，你能做的只是别写出让一个 wave 切两半的 if。

::: figure fig-wavefront-divergence
![Wavefront 分支发散](./images/wavefront-branch-divergence.png)

同一个 Wavefront 内部分支越整齐，lane 利用率越高
:::

> 经验法则：写 kernel 时，**让分支沿着 wave 边界（32 的整数倍）切**，不要让一个 if 把同一个 wave 切两半。

另一个相关概念是 **occupancy（占用率）**：一个 SIMD 上同时驻留多少个 wave。RDNA 上一个 SIMD 最多可以驻留多个 wave，让它们轮流"挡延迟"——访存的 wave 先去等数据，计算的 wave 抢上 VALU 干活。但驻留 wave 数受 VGPR、SGPR、LDS 的约束（详见下一节）。这就是为什么我们说 occupancy、VGPR 用量、LDS 用量是绑在一起的三件事。

> **现在不需要手算 occupancy**。你只需要知道：occupancy 反映 GPU 同时塞进多少 wave 来隐藏访存等待；高不一定就好，关键是有没有被 VGPR / LDS 这些资源卡住。后面 [Profiling 篇](../../part1-profiling/chapter5/index.md) 会教你从 rocprof / Omniperf 的输出里看它是被哪个资源限制的。

## 2.4 VGPR、SGPR 与 LDS 资源

这一节讲清楚为什么 VGPR / SGPR / LDS 是 kernel 优化绕不开的"片上资源池"。

GPU 不像 CPU 那样靠"几级 cache + 重命名 + 大量乱序"来藏延迟。它靠的是**同一时刻让足够多的 wave 在飞**：当一个 wave 卡在访存上时，调度器立即切到另一个 wave 上跑 VALU。能"在飞"的 wave 数量越多，访存延迟越容易被掩盖。但 wave 不是免费的——每个 wave 都要占一份寄存器和共享内存。

把这块片上资源想象成一个**有上限的资源池**。每个 wave 在池子里"租"几样东西：

| 资源 | 谁在用 | 单位 | 多了的副作用 |
| ---- | ---- | ---- | ---- |
| VGPR（向量寄存器） | 每个 lane 私有 | 32-bit 寄存器 | 单 wave 越大 → 同 SIMD 上能驻留的 wave 越少 |
| SGPR（标量寄存器） | 整个 wave 共享 | 32-bit 寄存器 | 控制流、地址、常量太多会撑爆 SGPR 上限 |
| LDS（共享内存 / 工作台） | 整个 workgroup 共享 | KB | 单 workgroup 越大 → 同 WGP 上能驻留的 workgroup 越少 |

如 @fig-vgpr-sgpr-lds-pool 所示，可以把"占用率约束"画成一组并列的水位线：

::: figure fig-vgpr-sgpr-lds-pool
```mermaid
flowchart LR
    subgraph SIMD["一条 SIMD 的资源水位"]
        direction TB
        A[VGPR 池] -->|"一个 wave 用 N 个"| W1[Wave A]
        A --> W2[Wave B]
        A --> W3[Wave C]
        S[SGPR 池] --> W1
        S --> W2
        S --> W3
        L[LDS 池 / WGP 共享] --> WG[整个 workgroup]
    end
    W1 -->|当 A 等访存| W2
    W2 -->|当 B 等访存| W3
    W3 -->|当 C 等访存| W1
```

VGPR/SGPR/LDS 任何一项打满，都会限制可同时驻留的 wave 数量
:::

三个资源池中**任何一个**先打满，就决定了 occupancy 的上限。这也是 rocprof / Omniperf 常给出 "VGPR-limited" / "LDS-limited" 这类标签的原因——它在告诉你瓶颈是哪个池子先没水了。

LDS 还有第二个性质：它不是一块纯线性内存，而是**分 bank** 的（多数 AMD GPU 上 32 个 32-bit bank）。同一周期里，如果同一个 wave 内不同 lane 访问到同一个 bank 的不同地址，就会发生 **bank conflict（bank 冲突）**——硬件会串行化访问。LDS 用得好不好的标准之一就是有没有让 bank conflict 控制住。

对 kernel 写手来说，这一节的实操含义是：

- 看到编译器报 VGPR > 某阈值就别忽略——同 SIMD 上能驻留的 wave 少了；
- block size（每个 workgroup 多少线程）和 LDS 用量是两个互相约束的旋钮，不能只看一个；
- 用 LDS 复用 tile 是手段，不是目的——如果一个 kernel 已经是 compute-bound，再多塞一份 LDS 反而会挤占其他 workgroup 的位置。

> **一个最小例子**：假设你写的 kernel 里每个线程有很多临时变量（中间结果、循环展开后的临时值、不必要的寄存器变量），编译器为了把它们都放下，每个线程要占的 VGPR 数就上去了。VGPR 总量是固定的，每个 wave 占的越多，同一个 SIMD 上能同时塞下的 wave 就越少。wave 少了，原来"A 在等访存时 B 顶上"的腾挪空间也就小了——结果就是访存延迟没人挡，VALU 闲在那里等数据。所以"减少不必要的临时变量、控制循环展开次数"在 GPU 上不仅是代码风格问题，而是直接影响 occupancy。

## 2.5 显存层次：寄存器 → LDS → L1/L2 → GDDR6

这一节把"数据从哪里来"完整拆开——从寄存器一直走到显存，让你看清一次访存到底走了几跳、哪一跳代价最高。

9070XT（RDNA4）的显存层次和数据中心 GPU（用 HBM）有很大不同。记住一个核心事实：**9070XT 用的是 16GB GDDR6，不是 HBM**。这意味着它的带宽上限远低于数据中心卡——后面算 Roofline 时，这条"带宽斜线"的位置会明显不同。

| 层级 | 位置 | 容量量级 | 带宽 | 延迟 |
| ---- | ---- | ---- | ---- | ---- |
| 寄存器（VGPR/SGPR）| 每个 lane/wave 私有 | 每个 wave 几 KB | 最高 | 最低 |
| LDS | CU/WGP 内部 | 每 workgroup 最多 64 KB | 很高 | 很低 |
| L0 / Vector L1 | CU 私有缓存 | 几十 KB | 高 | 低 |
| L2 cache | 全局共享 | 几 MB | 中 | 中 |
| GDDR6 显存 | 板载 | 16 GB | 🚧 待实测（标称 ~760 GB/s） | 高 |

> 🚧 上表中的带宽和延迟数字需在 9070XT 实验机就绪后用 micro-benchmark 实测回填。标称值仅作参考，实测可能不同。

几个对优化最关键的点：

- **离 lane 越近越快**：寄存器 > LDS > L1 > L2 > GDDR6。优化的核心思路之一就是**让数据尽量待在离 lane 近的地方**——用寄存器复用、用 LDS 缓存 tile、用合并访存提升 cache 命中率。
- **GDDR6 不是 HBM**：9070XT 的显存带宽（标称 ~760 GB/s）远低于 HBM 设备（动辄几 TB/s）。这意味着对 9070XT 来说，**memory-bound 算子的优化空间更大也更关键**——很多算子会卡在带宽上。
- **合并访存（Coalescing）**：连续的线程访问连续的地址时，硬件可以把多次访问合并成一次大事务，充分利用带宽。反之，strided 访问（线程访问间隔地址）会让带宽利用率大跌。这是第 5 章 profiling 会用 strided 反例演示的重点。

## 2.6 WMMA：RDNA4 的矩阵加速单元

这一节把"AI 算子怎么落到硬件矩阵单元上"讲清楚——它是 GEMM、Attention 这类算子能跑得多快的核心来源。

**WMMA（Wave Matrix Multiply Accumulate，波矩阵乘累加）** 是 RDNA 3 引入、RDNA 4 进一步增强的矩阵指令族。它的核心想法是：**用一条 wave 协作完成一个小 matmul tile** `D = A·B + C`，而不是让 VALU 一条一条加。

WMMA 是 wave 级指令——一条指令需要整个 wave 的所有 lane 配合完成。它把 A、B、C 三个 tile 的元素**分散到 wave 内每个 lane 的 VGPR 上**，硬件再用矩阵单元做一次乘累加。

GPUOpen 官方博客 "How to accelerate AI applications on RDNA 3 using WMMA" 给出的关键事实：

- **支持 wave32 与 wave64 两种模式**，编译器选哪种由 build target 决定；
- RDNA 3 / 3.5 上**只支持 16×16×16 一种 tile 形状**，RDNA 4 有所增强；
- 输入 A/B 类型可以是 FP16、BF16、INT8、INT4；累加器 C/D 类型可以是 FP16、FP32、BF16、INT32；
- 编译器内置函数形如（**只是指令入口形态，本章不要求复制运行**）：

```cpp
// RDNA：16x16x16 fp16 → fp32，wave32
float8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
// RDNA：16x16x16 int8 → int32，wave32
int8   d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(/*unsigned*/ false, a8,
                                                      false, b8, c32, false);
```

> **听说过 NVIDIA Tensor Core 的话**：WMMA 在角色上和 Tensor Core 一致——都是"一条指令完成一个小 matmul tile"的专用矩阵单元，不是同一套指令集，但解决的是同一类问题。

回到 AI 算子层。**为什么 GEMM、Attention、Conv 都喜欢 WMMA？** 因为它把"乘加密集"这件事直接交给硬件矩阵单元，不再走 VALU 一条一条加；外加它是 wave 级指令，VGPR 之间自然形成数据复用，把"重复加载同一个 tile"的代价摊薄掉。

::: figure fig-matrix-decision-tree
```mermaid
flowchart TD
    A[AI 算子: matmul / attention / conv] --> B{WMMA 可用?}
    B -- "是（RDNA3+）" --> C[WMMA · 16x16x16]
    B -- "否" --> E[纯 VALU · 手写 GEMM]
    C --> F[rocWMMA / Triton / 自写 HIP intrinsic]
```

在 9070XT 上选择矩阵单元路径的简化决策
:::

实操含义：

- 在 9070XT（RDNA4）上写矩阵相关 kernel，**优先确认 WMMA 路径**：要么用 rocWMMA 帮你封好，要么直接调 `__builtin_amdgcn_wmma_*` 内置函数。第 2 篇的 [GEMM](../../part2-kernels/chapter9/index.md) 会做对比实验。
- WMMA 的 tile 形状决定了 BLOCK_M / BLOCK_N / BLOCK_K 的最佳取值。第 2 篇的 Triton 章节会把这点反复用到。

## 2.7 Roofline 的硬件来源

最后一节把硬件参数翻译成 **Roofline 模型** 上的两条线，这样后面我们打开 profiler 时，就能很快判断"还有多少空间"。

Roofline 的核心想法非常朴素：**任何 kernel 的实际性能都被两件事卡住——算力上限和带宽上限**。把它们画在同一张坐标系里：

- 横轴：**arithmetic intensity（算术强度）**，单位 FLOP/Byte，意思是"每搬运 1 字节数据，能做多少次浮点运算"；
- 纵轴：**实际性能**，单位 FLOP/s（或 TFLOPS）；
- 两条线：
  - **水平线**：硬件的峰值算力 P_peak（给定精度下的 TFLOPS）；
  - **斜线**：硬件的峰值带宽 B_peak 乘以算术强度。

::: figure fig-roofline
```mermaid
flowchart LR
    A["低算术强度<br/>memory-bound 区"] --> B["拐点 P_peak / B_peak"]
    B --> C["高算术强度<br/>compute-bound 区"]
    A -.斜线.-> R["实际性能<br/>= min P_peak, B_peak * AI"]
    C -.水平线.-> R
```

Roofline 的两条线：左半边由带宽决定，右半边由算力决定
:::

两条线的硬件来源很具体：

- **峰值算力 P_peak** 来自"每 CU / 每周期能做多少 FLOP × CU 数 × 时钟"。对 9070XT，这条线在"使能 WMMA"的前提下较高，不走 WMMA 时会大幅下降。
- **峰值带宽 B_peak** 来自 GDDR6 显存。9070XT 标称 ~760 GB/s（**以实测为准**）。注意这个数字远低于 HBM 设备的几 TB/s——所以 9070XT 的 Roofline 拐点位置会和数据中心卡很不一样。

> 🚧 P_peak 和 B_peak 的具体数字需在 9070XT 实验机就绪后用 micro-benchmark 实测。标称值不作结论。

把这两条线画到同一张图上，每个 kernel 都会落在某一个点上。先用最简单的 Vector Add `c[i] = a[i] + b[i]` 走一遍"算术强度怎么从代码里算出来"：

```text
对每个 float32 元素：
  读 a[i]：4 Byte
  读 b[i]：4 Byte
  写 c[i]：4 Byte
  做 1 次加法：1 FLOP

算术强度 = FLOP / Byte = 1 / 12 ≈ 0.083 FLOP/Byte
```

关键不是死记这个数字，而是理解：**Vector Add 的算术强度天然很低**——搬一堆数据只做一次加法，所以它必然落在 Roofline 的斜线那一侧，属于 memory-bound。

| 算子典型 | 算术强度量级 | 落在哪一边 |
| ---- | ---- | ---- |
| Vector add（按元素加） | 很低（约 0.083–0.25 FLOP/Byte，取决于读写口径） | memory-bound |
| Reduction / Softmax 单趟 | < 1 FLOP/Byte | memory-bound |
| Conv / GEMM（小 batch） | 数 FLOP/Byte | 取决于 tile 是否有效复用 |
| GEMM（大 M/N/K，有 tile + 矩阵单元） | 数十至数百 FLOP/Byte | compute-bound |
| Attention（Flash 风格融合） | 取决于 head_dim 与序列长度 | 中间区，常被融合策略左右 |

读到这里，三件事应该串起来了：

1. **"算子是 memory-bound 还是 compute-bound" = "落在斜线上还是水平线上"**；
2. **优化 memory-bound 算子，就是想办法把工作点往右挪**——更高的复用、更大的 tile、合并访存、LDS 缓存；
3. **优化 compute-bound 算子，就是想办法把水平线往上抬**——用矩阵单元（WMMA）、更合适的精度、更高 occupancy 来逼近 P_peak。

后面 [第 3 章](../chapter3/index.md) 会用 vector add 跑通第一个程序，[第 6 章](../../part1-profiling/chapter6/index.md) 会把算子点画到真实 Roofline 上。

## 本章小结

- **CU 内部不是黑盒**：SIMD（VALU + SALU）、寄存器堆、LDS、L0 缓存各司其职；RDNA 还在 CU 之上多了一层 WGP，WGP 内 4 条 SIMD32 共享 LDS。
- **wave 是 GPU 的最小调度单位**：RDNA 默认 wave32；同一个 wave 内分支发散会让 lane 串行化执行。
- **VGPR / SGPR / LDS 是同一个 occupancy 池子的三个水位**：任何一个先满，都会限制可同时驻留的 wave 数，从而限制延迟隐藏能力。
- **9070XT 用 16GB GDDR6（非 HBM）**，带宽标称 ~760 GB/s；离 lane 越近的存储越快，优化的核心是让数据待在离 lane 近的地方。
- **WMMA 是 RDNA3+ 的矩阵加速单元**，一条 wave 协作完成一个 16×16×16 的 matmul tile；GEMM/Attention 优化的核心是利用好它。
- **Roofline 上的两条线直接来自硬件参数**：水平线是峰值算力（受是否使能 WMMA 影响），斜线是峰值带宽。

下一章 [第 3 章](../chapter3/index.md) 会用 vector add 跑通第一个真正的 GPU 程序，并把这一章建立的 Roofline 心智模型用起来。

## 自检问题

读完本章后，你应该能回答：

1. CU、SIMD、wavefront 三者是什么关系？一个 wave 是怎么落到 SIMD 上的？
2. 为什么同一个 wave 内分支走向不同会变慢？EXEC mask 是硬件做的还是你写代码控制的？
3. VGPR / SGPR / LDS 为什么会限制 occupancy？哪一个先打满，会出现什么样的标签？
4. 9070XT 的显存层次有哪几层？为什么说它和 HBM 设备的 Roofline 拐点不一样？
5. WMMA 是什么？它和"手写 GEMM 一条一条加"有什么区别？
6. Vector Add 为什么通常是 memory-bound？大 GEMM 配合 WMMA 为什么能跑到 compute-bound？

## 延伸阅读

**初学者优先看**

- [ROCm Conceptual: Device hardware glossary](https://rocm.docs.amd.com/en/latest/reference/glossary/device-hardware.html) — 本章所有硬件术语的对照字典
- [AMD Radeon RX 9070 XT 官方页面](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070-xt.html) — 本书基线硬件

**写 kernel 时再看：架构文档**

- [AMD RDNA Architecture White Paper（PDF）](https://www.amd.com/system/files/documents/rdna-whitepaper.pdf)
- [AMD GPU Architecture Programming Documentation Hub](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)
- [LLVM AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html)

**写矩阵相关 kernel 时再看：WMMA**

- [GPUOpen：How to accelerate AI applications on RDNA 3 using WMMA](https://gpuopen.com/learn/wmma_on_rdna3/)
- [GPUOpen：Using the Matrix Cores of AMD RDNA 4 architecture GPUs](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
- [GPUOpen Lab Notes：AMD matrix cores](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
- [ROCm/amd_matrix_instruction_calculator（GitHub 工具）](https://github.com/ROCm/amd_matrix_instruction_calculator)
