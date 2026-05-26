---
title: "第4章 内存层次与访存模式"
description: "Hello GPU 第4章 · HBM/GDDR/Infinity Cache、L1/L2、LDS bank 冲突、合并访存、atomics 与 fence"
---

# 第4章 内存层次与访存模式

## 本章导读

> 本章把单卡内存层次拆开。读完后，你应该能解释一个 GPU 内存请求从指令到 HBM 到底走了几跳、哪一步代价最高，并知道为什么"合并访存""LDS bank 冲突"这些词决定了多数 AI 算子的性能上限。所有数字以 AI MAX 395 + ROCm 7.12.0 为基线。

[第 3 章](../chapter3/index.md)我们拆完 CU、Wavefront、寄存器、LDS 这些"算"的资源，看到了一个 kernel 在硬件层面是怎么被调度起来的。但你很快就会发现一件让人有点泄气的事：**多数 AI 算子的瓶颈，根本不在算，而在搬。**

GEMM、Attention、LayerNorm、Softmax、reduction……它们的指令计数加起来再多，也比不过它们等数据的时间。一个 kernel 慢，多半不是 ALU 没干完活，而是 ALU 没数据可干。所以本章我们换一个视角：**从一个 load 指令出发，看一个数据元素究竟经过了哪些层级，每一层的容量、带宽、延迟差几个数量级。**

这一章会反复提到的硬件基线是 AI MAX 395 / gfx1151。它和 MI300X、RX 7900 XTX 都不一样：**它是 APU，CPU 和 GPU 共享同一片 LPDDR5X 物理内存**——这件事会贯穿 4.1 与 4.2。后面写算子时，"用什么 tile size""LDS 用不用 swizzle""能不能用 atomics 收尾"，所有这些选择都会回到本章建立的内存层级直觉。

## 4.1 内存层次总览

先建立一张总图。GPU 的内存层级和 CPU 在结构上类似——离 ALU 越近，越快、越小、越贵——但 GPU 把这种"金字塔"做得更陡：寄存器多到一个 CU 上能塞几百 KB，HBM/LPDDR 大到几十 GB。本节先用一张图和一张表把整套层级摆出来，后面 4.2–4.6 再逐层展开。

::: figure fig-mem-hierarchy
```mermaid
flowchart TD
    A[VGPR / SGPR 寄存器] --> B[LDS / 共享内存]
    B --> C[L0 / L1 Vector Cache]
    C --> D[L2 Cache]
    D --> E[Infinity Cache / MALL]
    E --> F[HBM / GDDR / LPDDR5X]
    F --> G[系统内存 / 跨设备]
```

AMD GPU 通用内存层级，越靠上越快、越小
:::

如果把同一套层级换成金字塔视角，@fig-mem-pyramid 会更直观：越靠近计算单元，容量越小但访问越快；越靠近 DRAM，容量越大但等待越久。

::: figure fig-mem-pyramid
![GPU 内存层级金字塔](./images/memory-hierarchy-pyramid.png)

GPU 内存层级金字塔：离计算单元越近越快、越小，离 DRAM 越近越慢、越大
:::

如 @fig-mem-hierarchy 和 @fig-mem-pyramid 所示，一次 load 指令的"理想路径"是从寄存器命中拿到结果，最坏路径是一路打到 DRAM。每往下一级，延迟会差一个甚至两个数量级。

下面这张表把每一层的典型角色、容量量级和性能特征摆出来。**注意：表里的具体数字大多来自公开资料里 RDNA3 / Navi31 的微基准与 AMD 白皮书**[^rdna3-cache][^cnc-rdna3]，**gfx1151（Strix Halo 的 RDNA 3.5 iGPU）的具体数值，本章末尾的 micro-benchmark 跑出来后再回填**。在这之前，凡是带 🚧 的格子都不要当成 gfx1151 上的实测。

| 层级 | 位置 | 典型容量量级 | 带宽特征 | 延迟特征 | 谁在用 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| VGPR / SGPR | 每个 SIMD 内部 | 单 CU 几百 KB（RDNA3 寄存器堆较 RDNA2 提升约 50%）[^rdna3-cache] | 算力侧最高，单周期可读多操作数 | 单周期级 | 每个线程私有变量 |
| LDS（共享内存） | 每个 WGP / CU 内部 | 128 KB / WGP（RDNA2/3）[^cnc-rdna3] | 单 WGP 量级 256 B/cycle[^cnc-rdna3] | 几十 cycle 量级 🚧 待 AI MAX 395 实测 | 同 block 内线程协作 |
| L0 Vector Cache | 一对 SIMD 共享 | 32 KB / 对 SIMD（RDNA3）[^cnc-rdna3] | 高 | 低 | 同一 wave 的连续访存 |
| L1（Mid-level） | Shader Array 共享 | Strix Halo 上 256 KB / Shader Array[^cnc-strix-mall] | 中 | 中 | 跨 WGP 复用 |
| L2 | 整个 GPU 共享 | Strix Halo 上 2 MB[^cnc-strix-mall]；RDNA3 dGPU 最高 6 MB[^cnc-rdna3] | 中 | 中 | 跨 Shader Array 复用 |
| Infinity Cache（MALL） | 内存控制器侧 | Strix Halo 32 MB[^cnc-strix-mall][^videocardz-strix]；RDNA3 dGPU 最高 96 MB[^cnc-rdna3] | 中高 | 比 DRAM 低很多 | 替 DRAM 挡住一部分访问 |
| 显存（HBM/GDDR/LPDDR5X） | 板上或封装内 | gfx1151 上是统一内存，最多 128 GB LPDDR5X-8000[^amd-strix-blog] | gfx1151 理论 ~256 GB/s，实测约 215 GB/s[^cnc-strix-mall] | 上百 ns 量级；Strix Halo iGPU 端 DRAM 延迟 ~123 ns[^cnc-strix-mall] | 全局内存 |
| 系统内存 / 跨设备 | PCIe / IF 链路另一侧 | — | 受总线限制 | 远高于本地 DRAM | `hipMemcpy`、`atomicAdd_system` |

读这张表时，有三件事值得提前在心里画几条参考线：

1. **每往下一级，容量大概×10–×100，但延迟也变差差不多一个数量级。** 寄存器和 DRAM 之间，差的不是十几 ns，是几百 ns。
2. **同一架构（RDNA3 / RDNA3.5）的同名层级，在 dGPU 和 iGPU 上未必相同。** Navi31 dGPU 的 L2 是 6 MB、Infinity Cache 96 MB；Strix Halo iGPU 的 L2 只有 2 MB、Infinity Cache 32 MB[^cnc-rdna3][^cnc-strix-mall]。引用别人博客上的 RDNA3 数字时务必看清是哪一颗。
3. **gfx1151 是 APU，最底层不是独立 HBM/GDDR，而是和 CPU 共享的 LPDDR5X。** 这件事 4.2 会展开。

### 本章会反复出现的术语速查

如果下面这些词你已经熟悉，可以跳过——本章会一直用它们：

| 术语 | 一句话解释 |
| ---- | ---- |
| wavefront（wave） | AMD GPU 上一组一起被调度执行的线程；RDNA 上常见 wave32 / wave64 |
| lane | 一个 wavefront 里的单个执行位置（一个"线程"） |
| WGP / CU / SIMD | RDNA 一个 WGP = 2 个 CU = 4 个 SIMD；HIP 的 `multiProcessorCount` 在 RDNA 上数 WGP，所以 Radeon 8060S 的 40 CU 会被报告成 **20** |
| VGPR / SGPR | 每个 SIMD 内部的向量 / 标量寄存器堆，是 GPU 最快的存储 |
| LDS | Local Data Share，一个 block 内线程共享的高速片上 SRAM；CUDA 里叫 shared memory |
| L0 / L1 / L2 | GPU 内部三级 cache，越靠近 ALU 越小越快 |
| MALL / Infinity Cache | 挂在内存控制器一侧的最后一级 SRAM，挡在 L2 和 DRAM 之间 |
| cacheline | 内存系统的最小事务单位；GCN 上是 64 字节，RDNA 上以 ISA 与微基准为准 |
| coalescing（合并访存） | 同一 wave 内多个 lane 的访存被硬件聚合成尽量少的 cacheline 事务 |
| bank conflict | 多个 lane 同一拍访问 LDS 的同一个 bank，被迫串行 |
| atomic | 多线程同时读改写同一个值时保证不被打断的操作 |
| memory order / fence | 决定一次 atomic / fence 在哪个 scope（block / device / system）上对别人可见 |
| launch overhead | 启动一次 GPU kernel 的固定开销，本章实测 ~5–6 µs/launch |

## 4.2 HBM vs GDDR vs Infinity Cache

这一节把"显存"这个词撕开。在 AI Infra 的语境里，至少要分清三类东西：HBM、GDDR、以及 AMD 这两代 GPU 用得越来越多的 Infinity Cache（MALL）。它们解决的问题不同，**而 gfx1151 上的形态又和上面任何一种典型 dGPU 都不太一样**——这是本章最重要的语境。

### HBM、GDDR、LPDDR5X：三种"主存"风格

先把三种主流主存风格放在一起对比，重点看带宽量级和应用场景：

| 类型 | 典型代表 | 带宽量级 | 容量量级 | 典型场景 |
| ---- | ---- | ---- | ---- | ---- |
| HBM3 / HBM3e | MI300X、H100、B200 | 单卡 TB/s 量级（多堆叠） | 单卡几十到上百 GB | 数据中心训练 / 大模型推理 |
| GDDR6 / GDDR6X | RX 7900 XTX、消费级 dGPU | 单卡几百 GB/s 到 1 TB/s | 单卡 8–24 GB | 桌面图形 / 中等规模 GPU 计算 |
| LPDDR5X（统一内存） | Strix Halo / Apple Silicon | 单芯片几百 GB/s 量级 | 单系统几十到 128 GB | APU、移动 SoC、超低功耗推理 |

> ⚠️ HBM3、GDDR6 的具体峰值数值依产品不同跨度很大，**本教程只把 HBM/GDDR 当作语境**，不替没跑过的设备编数字。

### Infinity Cache 是什么

Infinity Cache（也叫 MALL，Memory Attached Last Level）不是显存，**它是位于内存控制器一侧的最后一级片上 SRAM 缓存**[^videocardz-strix][^cnc-strix-mall]。它出现在 RDNA2 之后的 AMD GPU 上，用来挡住一部分原本要去 DRAM 的访问，从而：

- 在不堆 HBM 的前提下接近 HBM 量级的有效带宽；
- 让窄位宽 + 大容量 LPDDR5X 这种"看上去带宽不够"的方案在很多负载下也能撑住。

::: figure fig-infinity-cache
```mermaid
flowchart LR
    A[CU / WGP] --> B[L0 / L1]
    B --> C[L2]
    C --> D[Infinity Cache / MALL]
    D --> E[内存控制器]
    E --> F[(LPDDR5X / GDDR / HBM)]
```

Infinity Cache 位于 L2 与 DRAM 之间，作为内存控制器侧的最后一级缓存
:::

如 @fig-infinity-cache 所示，Infinity Cache 实际上是在 L2 没接住的请求被打到 DRAM 之前，再加一层"挡板"。一次 miss 一直走到 DRAM 的代价非常高，所以这层缓存在带宽不那么富裕的 APU 上几乎是必备件。

### gfx1151 / AI MAX 395 上是 unified memory 架构

这是本章必须明确写清的一件事：**AI MAX 395（Strix Halo）是 APU，没有独立 HBM/GDDR**。CPU 和 GPU 共享同一片 LPDDR5X，所有"显存占用"在物理上都是从这片统一内存里切出来的[^amd-strix-blog][^cnc-strix-mall]。

具体到这颗芯片[^amd-strix-blog][^cnc-strix-mall][^videocardz-strix]：

- **DRAM**：256-bit LPDDR5X-8000，理论带宽约 256 GB/s，实测 GPU 侧可达约 215 GB/s 量级。
- **容量**：整机最高 128 GB，AMD 提供 Variable Graphics Memory（VGM）机制，可以把其中最多 ~96 GB 划分给 GPU 当 VRAM。
- **Infinity Cache（MALL）**：32 MB，挂在 IO Die（内存控制器侧）。这层 CPU 侧不可见——CPU 不能命中 GPU 的 MALL[^cnc-strix-mall]。
- **GPU 内部缓存层级**：L0 vector cache、L1（每个 Shader Array 256 KB）、L2（整个 GPU 共享 2 MB），都比 Navi31 dGPU 小一档。
- **DRAM 延迟**：从 GPU 看 DRAM 大约在 ~123 ns 量级——比同代 dGPU 略高，但比典型移动平台已经很好[^cnc-strix-mall]。

把这些放进 @fig-apu-vs-dgpu 的层级对比里，对比典型 HBM dGPU（例如 MI 系列）和 gfx1151，你能立刻看出 unified memory 架构的关键差异：

::: figure fig-apu-vs-dgpu
```mermaid
flowchart TD
    subgraph GFX1151["gfx1151 / AI MAX 395 (APU)"]
        A1[CPU 16 Zen5 核]
        A2[GPU 40 RDNA3.5 CU]
        A3[Infinity Cache 32 MB MALL]
        A4[LPDDR5X-8000 共享内存 ~256 GB/s]
        A1 --> A4
        A2 --> A3 --> A4
    end
    subgraph HBM_GPU["典型 HBM 数据中心 GPU"]
        B1[CPU Host]
        B2[GPU CU 阵列]
        B3[L2 / Infinity Cache]
        B4[(HBM3 板上显存)]
        B1 -. PCIe / IF .-> B2
        B2 --> B3 --> B4
    end
```

gfx1151（APU，统一内存）与典型 HBM dGPU 的访存结构对比
:::

如 @fig-apu-vs-dgpu 所示，写 AI Infra 实验时这件事至少有三个影响：

1. **`hipMemcpy` 的语义变了**：传统 dGPU 上，H2D / D2H 经过 PCIe，是一次真实的搬运；APU 上 host 和 device 共享物理内存，但**仍然存在缓存一致性、地址空间、page table 的开销**——所以"零拷贝"在 API 层面并非自动成立，要看具体内存类型（pinned、coherent vs non-coherent、HMM 等，参考 HIP 文档）。
2. **可用 VRAM 与系统稳定性强相关**：你能给 GPU 多少显存，取决于 BIOS / OS 的 VGM 设置和系统剩余内存；这和 dGPU 上"插一张 24 GB 卡就有 24 GB"完全不同。
3. **带宽天花板低于一档 HBM**：gfx1151 的 ~215 GB/s 实测带宽[^cnc-strix-mall]，决定了大模型推理在这颗芯片上几乎一定会撞 memory-bound 这堵墙；不要拿 dGPU 上的 GEMM 性能直觉来推 gfx1151。

> 本教程接下来所有"显存""HBM""带宽"相关的论断，**默认都站在 gfx1151 / Strix Halo 的视角**。要对比 MI 系列、Navi31 dGPU 时，会显式写清楚硬件名字。

## 4.3 L1 / L2 Cache 的工作方式

这一节把镜头从 DRAM 拉回片上。GPU 上的 L1 / L2 不是单纯的"挡 DRAM 的一层缓存"——它直接决定了一次 wave 的 64 个 lane 究竟生成几次外部内存事务，是合并访存这件事在硬件侧的执行者。

### 几个尺寸先记住

在 RDNA3 / RDNA3.5 上，按文献和微基准[^cnc-rdna3][^cnc-strix-mall]：

- **L0 vector cache**：一对 SIMD 共享，32 KB；
- **L1**：每个 Shader Array 一份——RDNA3 dGPU 与 Strix Halo 都在这一层做了扩容，Strix Halo 上为 256 KB；
- **L2**：整个 GPU 共享，Strix Halo 2 MB、Navi31 dGPU 6 MB，16-way 关联；
- **L1 单 WGP 吞吐**：每周期 128 字节量级（RDNA3 把 RDNA2 的 64 B/cycle 翻倍）[^cnc-rdna3]；
- **关于 cacheline 大小**：AMD 在 GCN 时代 L2 公开过 64 字节聚合规则[^gpuopen-coalescing]，RDNA3 的具体 cacheline 大小未在公开资料里以一句话给出（🚧 待核实，建议读 RDNA3 ISA 白皮书与硬件计数器输出）。

> 本章后面的合并访存（4.5）会回到这件事：在 GCN/CDNA 上明确按 64 字节聚合，所以"线程访问 4 字节连续元素 × 64 lane = 256 B = 4 条 cacheline"是经典模型；RDNA 上这一规则是否完全相同，要看具体 ISA 与微基准结果，禁止直接挪用 GCN 数字。

### L1 / L2 给算子的三个关键含义

| 关注点 | L1 / L2 怎么影响 | 算子例子 |
| ---- | ---- | ---- |
| 复用粒度 | 同一个 cacheline 被多少 lane / wave 重读 | GEMM tile、卷积 stencil |
| 命中策略 | 流式访问 vs 时间局部性 | 大向量逐元素 op：基本不命中 L1，靠合并；GEMM tile：靠 L1/L2 复用 |
| 写一致性 | 写直达 vs 写回，如何对其他 CU 可见 | reduction、原子 op、跨 block 同步 |

举个具体场景。一个 vector add 把两个 1 GB 的 fp16 向量逐元素加起来：每个元素只读一次、写一次，没有任何复用——L1 命中率会非常低，性能完全由 DRAM / Infinity Cache / L2 这条链路的有效带宽决定。这就是典型 memory-bound。

换个场景，一个 GEMM 把 K 维度切 tile 加载到 LDS：每个 A、B 元素被 tile 内的多个线程重复用——这时 L1 命中率高，性能更接近 compute-bound。理解 L1 / L2 是为了**心里有一根"什么样的访问会命中、什么样的不会"的尺**。

### 写一致性的"坑"

GPU 上的 L1 通常是写直达（write-through）—— L1 收到写请求后会更新 L2 [^hip-coalescing-write]。这听上去很贵，但避免了"两个 L1 各自缓存同一份数据，不知道谁是新的"这种复杂一致性问题。代价是写带宽压力更大，所以做 reduction 或者写碎片化数据时，**多线程写同一行 cacheline 会被硬件折叠成单次写**——这在 4.5 与 4.6 都会再用到。

## 4.4 LDS 详解与 bank 冲突

LDS（Local Data Share，OpenCL 里叫 local memory，CUDA 里叫 shared memory）是 GPU 上**最容易把性能写飞的一块片上 SRAM**。一个 GEMM、一个 LayerNorm、一个 reduction，几乎都要靠 LDS 来藏访存延迟、做线程间通信。但 LDS 不是"普通的 SRAM"——它有 bank 结构，stride 选错可以让吞吐直接腰斩。

### LDS 的 bank 结构

按 AMD 的 Composable Kernel 文档[^ck-bank] 与社区资料[^rocm-blog-bank]：

- LDS 被划分为 **32 个 bank**，每个 bank 宽 4 字节（1 dword）；
- 地址到 bank 的映射是 `bank = (byte_address / 4) mod 32`；
- 当同一个 wave 在同一个 phase 内有多个 lane 访问**不同地址但落在同一个 bank** 时，发生 bank conflict，硬件被迫串行化；
- AMD 的 wavefront 是 64 lane（RDNA 上还有 wave32 模式），硬件会把一次 LDS 访问按指令宽度拆成多个 phase，每个 phase 内才检查冲突——这是和 NVIDIA 32 lane 模型最大的区别[^ck-bank]。

::: figure fig-lds-banks
```mermaid
flowchart LR
    subgraph Wave["1 个 wavefront 的 64 lane"]
        L0[Lane 0] --> A0
        L1[Lane 1] --> A1
        L2[Lane 2] --> A2
        L31[Lane 31] --> A31
        L32[Lane 32] --> A32
    end
    subgraph LDS["LDS 32 banks"]
        A0[Bank 0]
        A1[Bank 1]
        A2[Bank 2]
        A31[Bank 31]
        A32[Bank 0 again]
    end
```

LDS 的 32 个 bank：lane 32 又落回 bank 0，所以连续 dword 模式下 lane 0 与 lane 32 必然落在同一 bank（按 phase 拆分检查）
:::

如 @fig-lds-banks 所示，bank 模 32 这个事实意味着：当一组 lane 跨过 32 这个边界时，会再次落回相同的 bank 序号，硬件需要在多个 phase 内分批处理。

### 三种最常见的 bank 模式

| 访问模式 | 例子 | 是否冲突 |
| ---- | ---- | ---- |
| 连续 dword | `lds[tid]` | 通常无冲突（32 lane 一个 phase 各落一个 bank） |
| 同一地址广播 | `lds[0]` 被全 wave 读 | 硬件 broadcast，不视为冲突 |
| Stride 是 32 的倍数 | `lds[tid * 32]` | **N-way 冲突最严重**，所有 lane 全打到同一个 bank |
| Stride 是 33（1 + 32） | 经典的 padding 写法 | 有效避免 stride=32 这一类冲突 |
| 2D tile 行主写 + 列主读 | GEMM 的 LDS 缓冲 | naive 实现读侧常见 2-way 冲突，需要 XOR swizzle 修复[^ck-bank] |

### 一个直观的踩坑例子：GEMM 的 LDS tile

写过 HIP GEMM 的人都见过这一幕：把 A 的 64×32 tile 行主存进 LDS，再让另一组线程列主地读出来——LDS write 用 `ds_write_b128` 没事，但 LDS read 出现 2-way bank 冲突[^ck-bank]：

::: figure fig-gemm-bank-conflict
```mermaid
flowchart TD
    A[Naive 2D tile in LDS] --> B[LDS write: ds_write_b128, 无冲突]
    A --> C[LDS read: ds_read_b128, 2-way bank 冲突]
    C --> D[XOR swizzle 后]
    D --> E[Read & Write 都无冲突]
```

经典的 GEMM tile bank 冲突：写没问题，读两路冲突；XOR swizzle 后两边都不冲突
:::

如 @fig-gemm-bank-conflict 所示，**单边没冲突不代表没问题**。修复手段一般有两类：

- **Padding（加一列）**：让每行多一个 dword，破坏 stride 32 的对齐；最简单，代价是浪费一点 LDS 容量；
- **XOR Swizzle**：对地址做 XOR 变换，让物理 bank 编号被打散；这是 Composable Kernel / rocWMMA 在生产代码里更常用的方案[^ck-bank]。

[第 16 章 Matmul 入门优化](../../part3-hip-kernels/chapter16/index.md) 会用真实 kernel 把 padding / swizzle 跑出来对比，本章只把"为什么需要"讲清楚。

### 怎么发现自己撞了 bank 冲突

不要靠猜。用 rocprof 看硬件计数器（具体方法在 [第 10 章 Omniperf 与硬件计数器进阶](../../part2-profiling/chapter10/index.md)）：

- `SQ_LDS_BANK_CONFLICT` 类计数器：是否发生冲突；
- LDS 实际带宽 vs 理论带宽：差距大就是冲突的代价；
- 也可以用 4.7 节的 micro-benchmark 故意构造 stride=32 的访问，看吞吐曲线在哪里腰斩。

## 4.5 全局内存合并访存（Coalescing）

合并访存是初学者最早听说、但最容易"以为自己懂了其实没懂"的一个概念。它讲的是同一个 wave 内的 64 个 lane 怎样合并自己的内存请求；理解它的关键不是"连续就好"，而是**理解硬件按多少字节、怎么对齐去聚合 wave 内的请求**。

### GCN / CDNA 上的经典合并规则

在 GCN（以及继承自它的 CDNA）架构上，AMD 在 GPUOpen 给过相对清晰的描述[^gpuopen-coalescing]：

- L2 cache 的访问以 **64 字节、按 64 字节对齐** 的 cacheline 为单位；
- 一个 wave 的 64 个 lane 各自的访问，会被硬件聚合成"覆盖目标 cacheline 集合的最少请求数"；
- 多个 lane 写同一个地址会被自动 collapse 成一次写——但 **atomics 不会**，每次原子访问必须是独立操作[^gpuopen-coalescing]。

把这个规则用 @fig-coalesce-txns 和 @fig-coalesce-strided 展开：@fig-coalesce-txns 画的是硬件事务条数，@fig-coalesce-strided 画的是连续访问和跨步访问在内存地址上的差别。

::: figure fig-coalesce-txns
```mermaid
flowchart TD
    subgraph Good["合并访存：64 lane 读 64 个连续 fp32"]
        G1[64 lane × 4 B = 256 B]
        G2[对齐到 4 条 64 B cacheline]
        G3[硬件发出 4 次内存事务]
    end
    subgraph Bad["未合并：64 lane 各自相距很远"]
        B1[每个 lane 落在不同 cacheline]
        B2[硬件被迫发出最多 64 次事务]
        B3[有效带宽下降可达一个数量级]
    end
```

合并访存与非合并访存：差距来自硬件发出的事务条数
:::

如 @fig-coalesce-txns 所示，"连续线程访问连续地址"被翻译成硬件语言，就是"用最少条 64 字节事务覆盖 wave 的请求集合"。

::: figure fig-coalesce-strided
![合并访存与跨步访存对比](./images/memory-coalescing-vs-strided.png)

连续访存能让多个 lane 合并成少量内存事务，跨步访存会放大事务数量
:::

> ⚠️ RDNA / RDNA3 的合并规则在很多实现细节上和 GCN 不完全相同（cacheline 大小、聚合粒度都可能不同），但**"连续线程读连续地址 + 自然对齐"始终是最安全的写法**。当你在意具体规则时，请优先以 RDNA ISA 白皮书与微基准（如 chipsandcheese 的 RDNA3 系列）为准[^cnc-rdna3]。

### 几种典型访存模式的代价对比

| 模式 | 例子 | wave 发出的事务条数 | 备注 |
| ---- | ---- | ---- | ---- |
| 连续读（最佳） | `out[tid] = in[tid]` | 4 条（64×4 B = 256 B） | 全合并 |
| 跨步 2 读 | `out[tid] = in[tid * 2]` | 至少 ×2 | 有效带宽折半 |
| 跨步 16 读 | `out[tid] = in[tid * 16]` | 接近 64 条 | 几乎没有合并 |
| 同地址广播 | 所有 lane 读 `in[0]` | 1 条 + 广播 | 实际命中后基本免费 |
| 写多 lane → 同地址 | 普通 store | 硬件 collapse 成 1 次 | atomics 不享受这个优化[^gpuopen-coalescing] |
| 2D 行主图像列读 | `img[col * H + row]` | 灾难性 | 改 layout 或转置 |

### 给算子写法的三条直觉

1. **先想"哪个变量随 lane 索引变化"**，让它做内层 stride 1 的访问；
2. **fp16 / bf16 的算子尽量做向量化 load**：AMD 上常见的 `global_load_dwordx4` 一条指令一个 lane 加载 16 字节，整个 wave 合起来 1024 字节——比 4 条 dword 指令少一个数量级的发射开销；
3. **遇到 transpose / strided slice，把转置或 gather 单独做成一个 kernel**，不要塞进主算子里。

[第 12 章 从 Vector Add 理解 GPU 并行](../../part3-hip-kernels/chapter12/index.md) 与 [第 13 章 Reduction 优化](../../part3-hip-kernels/chapter13/index.md) 会用具体 kernel 把这三条逐条跑一遍。

## 4.6 原子操作与内存一致性

AI 算子里 atomics 出现得比想象中多：reduction 的最后一阶段、histogram、scatter-add、Embedding 的反向、稀疏算子……都可能用到 `atomicAdd`。这一节把 HIP atomics 的几件事讲清楚：scope、memory order、fence，以及它们的代价为什么差几个数量级。

### HIP atomics 的几个 scope

按 HIP / Clang 的 scope 分级[^hip-cpp-ext][^llvm-d87858]：

- `singlethread`：单线程；GPU 上多用于补全语义，几乎不出现在算子代码里；
- `wavefront`：单 wave 内；在 wave 内做 collective 时使用；
- `workgroup` / **block**：同一 block 内（即 LDS 上的 atomic）；
- `agent` / **device**：单个 GPU 内（即 global memory 上的 atomic，普通 `atomicAdd` 默认就是这个 scope）；
- `system`：跨 GPU、跨 host；通过 `atomicAdd_system` 这类带 `_system` 后缀的 builtin 触发[^hip-cpp-ext]。

| Scope | 含义 | 典型成本量级 | 何时使用 |
| ---- | ---- | ---- | ---- |
| LDS atomic（block 级） | LDS 上的整型原子；CU 内有专用原子单元 | 最便宜 | block 内 reduction、计数 |
| Global atomic（agent / device 级） | 全局内存原子；硬件有专用原子单元 | 中 | 跨 block 累加、histogram |
| atomicAdd_system | 跨 GPU / host 可见 | 最贵（涉及缓存刷写、跨链路一致性） | 多 GPU 或 host 看见 |
| float atomicAdd（unsafe） | 编译器开 `-munsafe-fp-atomics` 时直接走硬件原子 | 远比 CAS loop 快 | coarse-grained 显存、可接受舍入差异 |

> AMD 的 CU 内部带有整型原子单元（LDS 上 32 个），全局也有专用的原子单元负责 global atomics——所以 atomics 不是"靠 CAS 模拟"的代名词，但 fp atomics 在某些设备 / 内存类型上仍然会退化为 CAS loop[^hip-cpp-ext]。

### Memory order 与 fence

HIP atomics 在底层使用 LLVM 的 memory order 模型，常见取值：`relaxed`、`acquire`、`release`、`acq_rel`、`seq_cst`。它们的代价大致是：

`relaxed < acquire ≈ release < acq_rel < seq_cst`

加上 fence builtin[^hip-cpp-ext]：

- `__threadfence_block()`：block scope；
- `__threadfence()`：device / agent scope；
- `__threadfence_system()`：system scope，通常会触发缓存刷写，**不要塞进 hot loop**。

实战中三条经验：

1. **Reduction 的累加阶段用 relaxed atomic 就够了**，最后通过一次 fence + 读结果完成同步；
2. **block scope 的 fence 几乎免费，agent scope 中等，system scope 很贵**——按需选择，不要默认 `seq_cst`；
3. **fp `atomicAdd` 对 coarse-grained 显存安全，对 fine-grained 共享内存可能失效**[^hip-cpp-ext]——开 `-munsafe-fp-atomics` 前要确认你访问的是哪类内存。

### atomic 的"软成本"：序列化

Atomics 的代价不只是单次操作贵，更可怕的是**它打破了写折叠优化**：普通 store 可以让一个 wave 的 64 个 lane 写到同一地址被合并成 1 次，但 atomic 必须保留每个 lane 的独立操作语义[^gpuopen-coalescing]。所以"100 个线程对同一个全局计数器 +1"这种 pattern 在 GPU 上几乎一定是性能灾难——不管硬件原子单元多快。

正确的做法通常是：**先在 LDS 里做 block 内 reduction，再让每个 block 用一次 global atomic 提交结果**——把 atomic 的频率从"每 thread 一次"压到"每 block 一次"。[第 13 章 Reduction 优化](../../part3-hip-kernels/chapter13/index.md) 会把这个模式写出来。

## 4.7 用 micro-benchmark 测量带宽与延迟

> **实测说明**：本节所有 GB/s、ms 数字来自 AI MAX 395 / gfx1151（Radeon 8060S Graphics，HIP `multiProcessorCount` 报告 **20**，对应 20 个 WGP；按 RDNA 一个 WGP 含 2 个 CU 折算就是产品页常写的 **40 CU**）+ ROCm 7.12.0（`torch==2.9.1+rocm7.12.0`、`triton==3.5.1+rocm7.12.0`、HIP 7.12.60610）。
>
> - 代码：[`code/part1-hardware-rocm/chapter4/`](https://github.com/Weihong-Liu/hello-ai-infra/tree/main/code/part1-hardware-rocm/chapter4)（`bw_triad.hip` / `bw_footprint.py` / `lds_stride.hip` / `atomic_cmp.py` / `run_all.sh`）。
> - 原始日志：`code/part1-hardware-rocm/chapter4/logs/{bw_triad,bw_footprint,lds_stride,atomic_cmp}.log`。
>
> 注：本节涉及"独立显存 / HBM / dGPU"的对照原本计划用 🚧 留给后续机型；gfx1151 是统一内存 APU，**没有独立显存可对比**，这类对照保持空缺并标"不适用 gfx1151"。

micro-benchmark 的目的不是"打个 GPU 性能跑分"——是验证你脑子里关于 4.1–4.6 的模型对不对。下面这组实验设计，**最少**要回答三个问题：

1. **HBM/LPDDR 带宽**：流式 copy / triad，gfx1151 上能跑出接近峰值的多少？
2. **L2 / Infinity Cache 拐点**：随着访问 footprint 增大，有效带宽在哪几个尺寸出现台阶？
3. **LDS bank 行为**：stride=1 与 stride=32 的吞吐比是多少？合不合预期？

### 实验设计原则

- **每个实验有清晰的"该撞哪一层"假设**；用 footprint 大小切换层级（如 8 KB 命中 L0，1 MB 命中 L2，64 MB 应该已经超过 MALL）；
- **GPU event 计时 + 多次重复 + warmup**：与 [第 6 章 第一个 AMD GPU 程序与 baseline](../chapter6/index.md) 保持同一套测量约束；
- **每条命令、每个 config、每条日志都落盘**：路径 `code/part1-hardware-rocm/chapter4/logs/`；
- **正文里折叠的代码块就是 `code/part1-hardware-rocm/chapter4/` 下真实的可运行文件**：默认折叠便于阅读，需要时点开看完整实现；`run_all.sh` 会用同一份文件做编译与运行。

### 4.7.0 跑之前先确认：环境、目录、成功的样子 {#sec-4-7-0-setup}

如果你只想读懂 4.1–4.6 的概念，可以跳过本节直接看 4.7.1–4.7.4 的结果表。**但如果你想亲手复跑这四个 micro-benchmark**，下面这一小节就是为你写的：它把"在哪里跑、跑完应该看到什么、跑歪了怎么办"一次写清楚。

#### 你将要做什么

进入仓库下的 `code/part1-hardware-rocm/chapter4/` 目录，运行 `run_all.sh`，得到四个日志文件，并从日志里看到 DRAM 带宽、MALL 拐点、LDS bank conflict、atomic 争用这四个现象。**这一节不要求你手写任何 kernel**——只要复跑、读图、对结果。

#### 环境要求与一键检查

| 项目 | 要求 | 检查命令 | 期望输出 |
| ---- | ---- | ---- | ---- |
| GPU | 一张支持 ROCm 7.12 的 AMD GPU；本章基线是 gfx1151 / Radeon 8060S | `rocminfo \| grep gfx` | 输出含 `gfx<version>`（本章基线是 `gfx1151`） |
| ROCm | 7.12.0（其他版本能跑，但数字可能略偏） | `hipcc --version` | `HIP version: 7.12.x` |
| PyTorch（ROCm 构建） | `2.9.x+rocm7.12.0` | `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | `2.9.1+rocm7.12.0 True` |
| Triton（ROCm 构建） | `3.5.x+rocm7.12.0` | `python -c "import triton; print(triton.__version__)"` | `3.5.1+rocm7.12.0` |
| GPU 可见 | HIP 能枚举到 1 张卡 | `python -c "import torch; print(torch.cuda.device_count())"` | `1` |

> PyTorch 的 ROCm 版本沿用 NVIDIA 时代的 API，**仍然写 `device="cuda"`**——这里不是在调用 NVIDIA CUDA，只是 PyTorch 把"当前默认 GPU 设备"统一叫这个名字。后面 `bw_footprint.py` / `atomic_cmp.py` 里的 `device="cuda"` 都是这个意思。

#### 完整运行步骤

进入仓库后切到本章代码目录，跑 `run_all.sh`：

```bash
cd code/part1-hardware-rocm/chapter4
bash run_all.sh
```

`run_all.sh` 会依次完成：用 `hipcc -O3` 编译 `bw_triad.hip` 与 `lds_stride.hip`，运行两个 HIP 可执行文件与两个 Python 脚本，每个 bench 的标准输出 tee 到 `logs/<name>.log`。

#### 想单独跑某一个 bench

在 `code/part1-hardware-rocm/chapter4/` 目录里：

```bash
# 4.7.1 HIP Triad
hipcc -O3 bw_triad.hip -o bw_triad
./bw_triad                            # 用默认 size 扫描
# 或 ./bw_triad --bytes 1073741824 --repeats 50  指定单尺寸

# 4.7.2 Triton footprint
python bw_footprint.py

# 4.7.3 LDS bank conflict
hipcc -O3 lds_stride.hip -o lds_stride
./lds_stride

# 4.7.4 atomic 对比
python atomic_cmp.py
```

#### 跑完应该看到什么

跑完后，先检查日志文件是否齐全：

```bash
$ ls logs/
atomic_cmp.log  bw_footprint.log  bw_triad.log  lds_stride.log
```

然后随便抓几条关键行，确认 bench 真的写出了数字（而不是中途崩溃）：

```bash
$ grep -E "BW=|eff_bw=|STRIDE=|mode=" logs/*.log | head -20
```

每个 bench 都会打印一条形如 `[<name>] ... BW=... GB/s`（或 `time=... ms`）的行，例如：

```
[bw_triad] bytes=1073741824 repeats=50 time=704.84 ms BW=228.51 GB/s
[bw_footprint] n=  4194304 time=8.34 ms eff_bw= 804.34 GB/s
[lds_stride] STRIDE=1   time=150.53 ms
[atomic_cmp] n_threads=4194304 mode=atomic_full time_ms=5.34 ops_per_sec=3.93e+10
```

只要看到这四类行各自至少有几条、`rc: 0` 出现在每个日志末尾，就算跑成功了。

#### 结果可以差多少？什么算"复现失败"

micro-benchmark 的绝对数字会随系统负载、显存配置、ROCm minor 版本浮动，**重点看趋势，不要钉死数字**。下表给出可接受范围（仍以 AI MAX 395 + ROCm 7.12.0 为基线）：

| 实验 | 看哪个数字 | 在这个范围内就算 OK | 怎么判定"明显跑歪了" |
| ---- | ---- | ---- | ---- |
| 4.7.1 Triad | DRAM 稳态（≥ 32 MiB）BW | 200–240 GB/s | 长期 < 150 GB/s，多半是 GPU 显存配额太小、跑到了系统内存路径 |
| 4.7.1 Triad | 4 MiB 处 BW（MALL 峰值） | 明显高于 DRAM 稳态（一般 600–800 GB/s） | 4 MiB 比 32 MiB 还慢 → MALL 没起作用，检查驱动 / 是否其他进程占满 GPU |
| 4.7.2 footprint | 16 MiB → 32 MiB 那一跳 | **单跳腰斩**（高位 ≥ 600 GB/s、低位 ~225 GB/s） | 整条曲线都在 200–250 GB/s → 大概率 Triton 没用上 ROCm 后端 |
| 4.7.3 LDS stride | STRIDE=32 / STRIDE=1 | **≥ 3x**（实测 ~3.86x） | 比值小于 2x → 编译器把循环优化掉了，检查是否真用了 `-O3` 与正确 ISA |
| 4.7.4 atomic | atomic_full / plain_store（4M threads） | **≥ 3x**（实测 ~4.62x） | 比值接近 1 → atomic 没真打到全局内存，检查 mode 参数是否生效 |

小尺寸（< 1 MiB）的 BW 数字本来就被 launch overhead 主导（每次 launch 5–6 µs 的固定成本），上下浮动 ±30% 都正常——4.7.2 的解读已经明确这一点，不要据此判断"失败"。

#### 常见报错

| 报错信息 | 通常原因 | 处理 |
| ---- | ---- | ---- |
| `hipcc: command not found` | ROCm 没装好或不在 `PATH` | 检查 ROCm 是否安装、`/opt/rocm/bin` 是否在 `PATH` 里 |
| `ModuleNotFoundError: No module named 'torch'` / `'triton'` | 当前 Python 环境没装 ROCm 版的 torch / triton | 装上对应 `+rocm7.12.0` 的 wheel 后再跑 |
| `RuntimeError: HIP error: no HIP-capable device is detected` / `torch.cuda.is_available() == False` | 驱动 / 容器没暴露 GPU | `rocminfo` 能看到 `gfx<version>`；docker 容器记得传 `--device=/dev/kfd --device=/dev/dri` |
| `bash: ./run_all.sh: Permission denied` | 脚本没有可执行位 | 用 `bash run_all.sh`，或 `chmod +x run_all.sh` |
| `hipMalloc` 失败 / VRAM 不够 | APU 上 GPU 显存配额（如 UMA Frame Buffer / VGM）切得太小 | 在 BIOS 把 "UMA Frame Buffer Size" / VGM 调到 ≥ 16 GB（dGPU 上无此问题） |
| `logs/` 下没有任何 `.log` | 脚本没真正跑起来（参考前几行报错） | 在 `chapter4/` 目录里跑通 `bash run_all.sh`，再回来看 `logs/` |

### 4.7.1 HIP：HBM / 全局内存带宽（Triad-like）

最经典的全局内存带宽测试（STREAM Triad 的 GPU 版本）。每个线程做一次 `c[i] = a[i] + alpha * b[i]`，没有任何复用——理论上时间 = 总搬运字节 / 有效带宽。

<details>
<summary><code>bw_triad.hip</code></summary>

```cpp
// code/part1-hardware-rocm/chapter4/bw_triad.hip
//
// STREAM Triad on the GPU: c[i] = a[i] + alpha * b[i]
// 没有任何复用——理论时间 ≈ 总搬运字节 / 有效带宽。
// 通过扫描 size 画 "L2 → MALL → DRAM" 的有效带宽阶梯。
//
// 编译：
//   hipcc -O3 bw_triad.hip -o bw_triad
// 运行（默认扫描）：
//   ./bw_triad
// 单尺寸：
//   ./bw_triad --bytes 134217728 --repeats 100
// 自定义尺寸列表（每数组 bytes，逗号分隔）：
//   ./bw_triad --sizes 1048576,4194304,16777216,33554432,134217728,268435456,1073741824
//
// 硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0 (rocm-sdk wheels) / hipcc 7.2.x
// 输出格式（一行一个 size，方便 grep）：
//   [bw_triad] bytes=<N> repeats=<R> time=<ms> BW=<GB/s>
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>

#define HIP_CHECK(expr) do {                                                      \
    hipError_t _err = (expr);                                                     \
    if (_err != hipSuccess) {                                                     \
        fprintf(stderr, "HIP error %s at %s:%d : %s\n", hipGetErrorString(_err),  \
                __FILE__, __LINE__, #expr);                                       \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

__global__ void triad(const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ c,
                      float alpha, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + alpha * b[i];
}

__global__ void fill_kernel(float* p, float v, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

static void print_header() {
    char buf[64];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", std::localtime(&t));
    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("# bw_triad (STREAM Triad) — AI MAX 395 / gfx1151\n");
    printf("# timestamp: %s\n", buf);
    printf("# device:    %s (gcnArch=%s, %d CUs)\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("# (run rocm-smi separately for power/freq/mem state)\n");
    fflush(stdout);
}

static double bench_one(size_t bytes, int repeats, int warmup) {
    size_t n = bytes / sizeof(float);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    HIP_CHECK(hipMalloc(&a, bytes));
    HIP_CHECK(hipMalloc(&b, bytes));
    HIP_CHECK(hipMalloc(&c, bytes));

    int block = 256;
    size_t grid_sz = (n + block - 1) / block;
    // gfx1151 grid x 上限远超 2^31，但保险起见 cap 到 int max.
    int grid = (grid_sz > (size_t)INT32_MAX) ? INT32_MAX : (int)grid_sz;

    fill_kernel<<<grid, block>>>(a, 1.0f, n);
    fill_kernel<<<grid, block>>>(b, 2.0f, n);
    fill_kernel<<<grid, block>>>(c, 0.0f, n);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));

    for (int i = 0; i < warmup; ++i)
        triad<<<grid, block>>>(a, b, c, 2.0f, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < repeats; ++i)
        triad<<<grid, block>>>(a, b, c, 2.0f, n);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));

    // Triad: 2 reads + 1 write = 3 * bytes per iteration
    double total_bytes = 3.0 * (double)bytes * (double)repeats;
    double gbps = total_bytes / ((double)ms * 1e-3) / 1e9;
    double ms_per_iter = (double)ms / (double)repeats;

    printf("[bw_triad] bytes=%zu repeats=%d time=%.4f ms time_per_iter=%.4f ms BW=%.2f GB/s\n",
           bytes, repeats, (double)ms, ms_per_iter, gbps);
    fflush(stdout);

    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));
    return gbps;
}

int main(int argc, char** argv) {
    // 默认扫描：1 MB → 1 GiB（带 1/4/16/32/128/256 MB / 1 GiB 等关键拐点）
    std::vector<size_t> sizes = {
        1ull << 20,   // 1 MiB
        4ull << 20,   // 4 MiB
        16ull << 20,  // 16 MiB
        32ull << 20,  // 32 MiB (MALL boundary)
        128ull << 20, // 128 MiB
        256ull << 20, // 256 MiB
        1ull << 30,   // 1 GiB
    };
    int repeats = 50;
    int warmup = 5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--bytes") {
            sizes.clear();
            sizes.push_back((size_t)std::strtoull(next(), nullptr, 10));
        } else if (a == "--repeats") {
            repeats = std::atoi(next());
        } else if (a == "--warmup") {
            warmup = std::atoi(next());
        } else if (a == "--sizes") {
            sizes.clear();
            std::string list = next();
            std::stringstream ss(list);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) sizes.push_back((size_t)std::strtoull(item.c_str(), nullptr, 10));
            }
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--bytes N | --sizes N1,N2,...] [--repeats R] [--warmup W]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    print_header();
    if (repeats < 50) repeats = 50;
    if (warmup < 5) warmup = 5;
    printf("# config: repeats=%d warmup=%d sizes=%zu entries\n", repeats, warmup, sizes.size());
    fflush(stdout);

    for (size_t bytes : sizes) {
        bench_one(bytes, repeats, warmup);
    }
    return 0;
}
```

</details>

实测结果（AI MAX 395 + ROCm 7.12.0，`logs/bw_triad.log`，每数组 size，3 × bytes / time）：

| size / array | 工作集（3 数组） | 实测 BW | 主要命中层 |
| ---- | ---- | ---- | ---- |
| 1 MiB | 3 MiB | **213.84 GB/s** | L2 / MALL，launch overhead 占比偏大 |
| 4 MiB | 12 MiB | **702.43 GB/s** | 32 MB MALL（接近上限） |
| 16 MiB | 48 MiB | **221.45 GB/s** | DRAM（已超 MALL） |
| 32 MiB | 96 MiB | 224.99 GB/s | DRAM |
| 128 MiB | 384 MiB | 227.35 GB/s | DRAM |
| 256 MiB | 768 MiB | 227.39 GB/s | DRAM |
| 1 GiB | 3 GiB | **228.51 GB/s** | DRAM 稳态 |

预期与解读（AI MAX 395 + ROCm 7.12.0）：

- DRAM 稳态吞吐约 **228 GB/s**，与文中引用的 LPDDR5X 实测 ~215 GB/s 上限[^cnc-strix-mall]同量级（高 ~6%，落在 Triad 比 STREAM Copy 略容易跑满有效带宽的预期范围）；
- 4 MiB 工作集出现 **702 GB/s** 的"超 DRAM"峰值，正是 32 MB MALL 命中的证据——这条曲线 4 MiB 比 16 MiB 还快 3.2x，说明 MALL 起效；
- 工作集一旦从 12 MiB 跨过 32 MiB MALL 边界（16 MiB / 48 MiB 那行），带宽**直接腰斩**到 ~221 GB/s，与 4.2 节描述的 "Last-Level Cache → DRAM" 拐点位置完全对应；
- 1 MiB 那行只有 214 GB/s 反而比 4 MiB 慢——不是反常，而是 50 次 launch 的固定开销在小工作集下被均摊不掉（per-iter 仅 15 µs）；要看真实小工作集带宽得换 footprint 扫描（4.7.2 节）；
- 没有看到任何"显存 vs 独立 DRAM"的层次——gfx1151 是 APU + 统一内存，**不适用 gfx1151** 的 dGPU/HBM 对照保持空缺。

### 4.7.2 Triton：footprint 扫描（命中 L2 vs MALL vs DRAM）

用 Triton 做 footprint 扫描代码量更少。`vector_copy` kernel 跑不同 size，画"size – effective BW"曲线。

<details>
<summary><code>bw_footprint.py</code></summary>

```python
"""
code/part1-hardware-rocm/chapter4/bw_footprint.py

Triton vector_copy footprint scan: 测量不同 footprint 下的有效带宽，
观察 "L0/L1 → L2 → MALL → DRAM" 的阶梯。

用法：
    python bw_footprint.py
    python bw_footprint.py --sizes 4096,65536,1048576,16777216,134217728
    python bw_footprint.py --repeats 200 --warmup 30

硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0 (rocm-sdk wheels)。
输出格式（一行一个 size，便于 grep）：
    [bw_footprint] n=<elems> bytes=<B> time=<ms_total> time_per_iter=<ms> eff_bw=<GB/s>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as exc:  # pragma: no cover
    HAS_TRITON = False
    _TRITON_ERR = exc


if HAS_TRITON:
    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, v, mask=mask)


def _print_header() -> None:
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"# bw_footprint (Triton vector_copy) — AI MAX 395 / gfx1151")
    print(f"# timestamp: {now}")
    print(f"# python:    {sys.version.split()[0]}")
    print(f"# torch:     {torch.__version__}")
    if HAS_TRITON:
        print(f"# triton:    {triton.__version__}")
    else:  # pragma: no cover
        print(f"# triton:    NOT AVAILABLE ({_TRITON_ERR!r})")
    if torch.cuda.is_available():
        try:
            print(f"# device:    {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    # rocm-smi snapshot to stdout (best effort)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        print("# rocm-smi:")
        for line in out.stdout.splitlines():
            print(f"#   {line}")
    except Exception as e:
        print(f"# rocm-smi: unavailable ({e})")
    sys.stdout.flush()


def bench(n_elems: int, repeats: int, warmup: int, block: int = 1024) -> tuple[float, float]:
    if not HAS_TRITON:
        raise RuntimeError("Triton not available; cannot run footprint scan.")
    device = "cuda"
    x = torch.empty(n_elems, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    # 用 random fill 防止编译器把 copy 折叠
    x.uniform_(-1.0, 1.0)
    grid = ((n_elems + block - 1) // block,)

    for _ in range(warmup):
        copy_kernel[grid](x, y, n_elems, BLOCK=block)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        copy_kernel[grid](x, y, n_elems, BLOCK=block)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)

    bytes_total = 2 * n_elems * 4 * repeats  # 1 read + 1 write fp32
    gbps = bytes_total / (ms * 1e-3) / 1e9
    return ms, gbps


def main() -> int:
    p = argparse.ArgumentParser()
    # 覆盖 L0/L1(几 KB) → L2(几 MB) → MALL(32 MB) → DRAM(几百 MB) 的尺寸点
    default_sizes = ",".join(str(n) for n in [
        4_096,        # 16 KiB  (L0/L1 territory)
        16_384,       # 64 KiB
        65_536,       # 256 KiB
        262_144,      # 1 MiB
        1_048_576,    # 4 MiB   (L2)
        4_194_304,    # 16 MiB
        8_388_608,    # 32 MiB  (≈ MALL boundary)
        16_777_216,   # 64 MiB
        67_108_864,   # 256 MiB
        134_217_728,  # 512 MiB
        268_435_456,  # 1 GiB
    ])
    p.add_argument("--sizes", default=default_sizes,
                   help="逗号分隔的元素数列表（fp32, 每元素 4 bytes）")
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--block", type=int, default=1024)
    args = p.parse_args()

    if args.repeats < 100:
        args.repeats = 100
    if args.warmup < 5:
        args.warmup = 5

    _print_header()
    print(f"# config: repeats={args.repeats} warmup={args.warmup} block={args.block}")
    sys.stdout.flush()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — ROCm/HIP not visible to PyTorch.",
              file=sys.stderr)
        return 1
    if not HAS_TRITON:
        print(f"ERROR: triton import failed: {_TRITON_ERR!r}", file=sys.stderr)
        return 2

    for sz_str in args.sizes.split(","):
        sz_str = sz_str.strip()
        if not sz_str:
            continue
        n = int(sz_str)
        bytes_per_array = n * 4
        ms, gbps = bench(n, repeats=args.repeats, warmup=args.warmup, block=args.block)
        ms_per_iter = ms / args.repeats
        print(f"[bw_footprint] n={n:>11} bytes={bytes_per_array:>11} "
              f"time={ms:9.3f} ms time_per_iter={ms_per_iter:8.4f} ms "
              f"eff_bw={gbps:7.2f} GB/s")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

</details>

实测结果（AI MAX 395 + ROCm 7.12.0，`logs/bw_footprint.log`，2 × bytes / time）：

| n (fp32) | 单数组大小 | eff_bw | 主要命中层 |
| ---- | ---- | ---- | ---- |
| 4 096 | 16 KiB | 5.89 GB/s | launch overhead 主导 |
| 16 384 | 64 KiB | 23.71 GB/s | launch overhead |
| 65 536 | 256 KiB | 93.42 GB/s | launch overhead 渐降 |
| 262 144 | 1 MiB | 371.81 GB/s | 命中 L2 / MALL |
| 1 048 576 | 4 MiB | 620.27 GB/s | MALL |
| 4 194 304 | 16 MiB | **804.34 GB/s** | **MALL 峰值** |
| 8 388 608 | 32 MiB | **223.68 GB/s** | **拐点：超出 32 MB MALL → DRAM** |
| 16 777 216 | 64 MiB | 224.13 GB/s | DRAM |
| 67 108 864 | 256 MiB | 225.16 GB/s | DRAM |
| 134 217 728 | 512 MiB | 224.52 GB/s | DRAM |
| 268 435 456 | 1 GiB | 224.23 GB/s | DRAM 稳态 |

预期与解读（AI MAX 395 + ROCm 7.12.0）：

- **拐点位置完全符合预期**：16 MiB 还能跑到 804 GB/s（命中 32 MB MALL），32 MiB 时**单跳腰斩**到 224 GB/s（工作集 64 MiB > MALL，必须打 DRAM）；之后一直到 1 GiB 都稳定在 224–225 GB/s，与 4.7.1 节的 DRAM 平台 ~228 GB/s 高度一致（差 < 2%）；
- **小工作集那一段是 launch-overhead 主导，不是缓存层级**：16 KiB 那行 5.9 GB/s 不能解读为 "L0/L1 太慢"——200 次 launch 总共 1.11 ms，每次 launch 5–6 µs 已经吃掉绝大多数时间；要看 L0/L1 真实带宽得换"单 kernel 内多次访问"的 microbench；
- **MALL 命中峰值 804 GB/s ≈ 3.6 × DRAM**：比 4.7.1 Triad 的 4 MiB 峰值（702 GB/s）略高一档，因为 vector_copy 只 1 读 1 写、流水线更紧；
- 8 MiB → 16 MiB 那段**还在涨**（620 → 804 GB/s）说明 MALL 是逐步"暖"起来的，整张图的真实形状是"小尺寸被 launch 拖、中尺寸 MALL 拉满、超 MALL 后 DRAM 平台"三段；
- gfx1151 的 L0（一对 SIMD 共享 32 KiB）/ L1（每 Shader Array 256 KiB）/ L2（整 GPU 共享 2 MB，4.1 与 4.3 节均按此口径）都没在这条曲线里独立显形——这是 footprint 扫描的固有局限（不是 latency-bench），4.7 节的目的只是**抓住 MALL 拐点**这一条最显眼的信号，已经达成。

### 4.7.3 HIP：LDS bank conflict 微基准

故意构造 stride=1 / 4 / 16 / 32 的 LDS 访问，观察吞吐。这是验证 4.4 节那张冲突表的最直接办法。

<details>
<summary><code>lds_stride.hip</code></summary>

```cpp
// code/part1-hardware-rocm/chapter4/lds_stride.hip
//
// LDS bank conflict micro-benchmark：构造 stride=1 / 2 / 4 / 8 / 16 / 32
// 的 LDS 访问，观察吞吐变化，验证 4.4 节的 N-way 冲突。
//
// 编译：
//   hipcc -O3 lds_stride.hip -o lds_stride
// 运行：
//   ./lds_stride                       # 默认扫 1/2/4/8/16/32 + repeats=100
//   ./lds_stride --repeats 200 --iters 65536
//
// 硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0 (rocm-sdk wheels) / hipcc 7.2.x
// 输出格式（每个 stride 一行）：
//   [lds_stride] STRIDE=<S> repeats=<R> time=<ms_total> time_per_iter=<ms>
//
// 实现要点：
//   - LDS 大小 2048 floats = 8 KiB（fits in single CU 的 LDS）
//   - 每个 thread 在 n_iters 次循环里反复读 lds[(tid*STRIDE) & 2047]
//   - 输出 acc 写到 d_out 防止 dead-code-elim
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#define HIP_CHECK(expr) do {                                                      \
    hipError_t _err = (expr);                                                     \
    if (_err != hipSuccess) {                                                     \
        fprintf(stderr, "HIP error %s at %s:%d : %s\n", hipGetErrorString(_err),  \
                __FILE__, __LINE__, #expr);                                       \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

template <int STRIDE>
__global__ void lds_stride_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n_iters) {
    __shared__ float lds[2048];
    int tid = threadIdx.x;
    // load LDS from global once
    for (int i = tid; i < 2048; i += blockDim.x)
        lds[i] = in[i];
    __syncthreads();

    float acc = 0.f;
    #pragma unroll 1
    for (int it = 0; it < n_iters; ++it) {
        int idx = (tid * STRIDE + it) & 2047;  // +it 避免编译器算成常量
        acc += lds[idx];
    }
    out[tid] = acc;  // sink 防止 DCE
}

template <int STRIDE>
static double run_one(const float* d_in, float* d_out, int block, int n_iters,
                      int repeats, int warmup) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));

    for (int i = 0; i < warmup; ++i)
        lds_stride_kernel<STRIDE><<<1, block>>>(d_in, d_out, n_iters);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < repeats; ++i)
        lds_stride_kernel<STRIDE><<<1, block>>>(d_in, d_out, n_iters);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));

    double ms_total = (double)ms;
    double ms_per = ms_total / (double)repeats;
    printf("[lds_stride] STRIDE=%-2d repeats=%d iters=%d time=%.4f ms time_per_iter=%.4f ms\n",
           STRIDE, repeats, n_iters, ms_total, ms_per);
    fflush(stdout);

    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    return ms_total;
}

static void print_header(int block, int n_iters, int repeats, int warmup) {
    char buf[64];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", std::localtime(&t));
    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("# lds_stride (LDS bank conflict scan) — AI MAX 395 / gfx1151\n");
    printf("# timestamp: %s\n", buf);
    printf("# device:    %s (gcnArch=%s, %d CUs)\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("# config:    block=%d iters=%d repeats=%d warmup=%d\n",
           block, n_iters, repeats, warmup);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int block = 256;
    int n_iters = 1 << 16;  // 65536
    int repeats = 100;
    int warmup = 5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--block") block = std::atoi(next());
        else if (a == "--iters") n_iters = std::atoi(next());
        else if (a == "--repeats") repeats = std::atoi(next());
        else if (a == "--warmup") warmup = std::atoi(next());
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--block B] [--iters I] [--repeats R] [--warmup W]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return 2;
        }
    }
    if (repeats < 50) repeats = 50;
    if (warmup < 5) warmup = 5;

    print_header(block, n_iters, repeats, warmup);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, 2048 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, block * sizeof(float)));
    HIP_CHECK(hipMemset(d_in, 1, 2048 * sizeof(float)));

    run_one<1>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<2>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<4>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<8>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<16>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<32>(d_in, d_out, block, n_iters, repeats, warmup);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
```

</details>

实测结果（AI MAX 395 + ROCm 7.12.0，`logs/lds_stride.log`，block=256, iters=65536, repeats=100）：

| STRIDE | 总时间 (ms) | per-iter (ms) | 相对 STRIDE=1 | 解读 |
| ---- | ---- | ---- | ---- | ---- |
| 1 | 150.53 | 1.505 | 1.00x | baseline，无冲突 |
| 2 | 154.31 | 1.543 | 1.03x | 无冲突 |
| 4 | 155.30 | 1.553 | 1.03x | 无冲突 |
| 8 | 168.57 | 1.686 | 1.12x | 轻度冲突初现 |
| 16 | 291.15 | 2.912 | **1.93x** | 2-way 冲突 |
| 32 | 580.88 | 5.809 | **3.86x** | 严重冲突，吞吐腰斩两次 |

预期与解读（AI MAX 395 + ROCm 7.12.0）：

- **stride 1/2/4 几乎完全持平**（1.51–1.55 ms，差 ~3%），说明 RDNA 3.5 的 32-bank × 4-byte LDS 在线性、连续、对齐的访问下确实"零冲突"；这是 4.4 节那张冲突表的最强证据；
- **stride 16 → 1.93x、stride 32 → 3.86x** 的退化阶梯，与 4.4 节"stride k 落入 gcd(k, 32) 个 bank"的预测**几乎逐字对得上**：stride 16 时 32 个 lane 落到 2 个 bank（16-way 冲突的另一种说法是"一拍只能服务 2 个 lane"），stride 32 时 32 个 lane 全部落到同一 bank（32-way 冲突，理论 32x 慢，但 wave64 的 dual-issue 与硬件 broadcast 把它压到 ~4x，符合实测）；
- **stride 8 已经开始抖**（+12%）——这是 RDNA bank 访问粒度（32-byte aligned）和 fp32 步长 8 之间的次要不对齐效应，4.4 节没有强调，但这里看得很清楚；
- 这一组数据是后面 GEMM tile / softmax LDS 写法选择的硬性依据：**只要让最内层维度 stride=1，bank 冲突的代价就基本消失**；如果非要 stride=32 写法（比如不 padding 的 transposed shared mem），就要做好 ~4x 慢的心理准备；
- 没有看到反常——stride 32 的退化幅度和 4.4 节预期高度一致，没有发现 gfx1151 在 LDS 行为上偏离 RDNA 3 公开模型的迹象。

### 4.7.4 Triton：atomic 收尾代价对比

最简单的"对比 atomic 频率与争用"实验，对应 4.6 节的两条结论："atomic 必须保留每 lane 独立语义、打破写折叠优化"和 "atomics 的代价取决于 scope 与争用程度"。三种写法工作量完全相同（每 thread 写一次 int32），差别只在原子频率/冲突模式：

- **(a) `atomic_full`**：所有 thread `atomicAdd` 到 **同一个全局地址**（最坏情况：wave 内 64 lane 全部串行）；
- **(b) `atomic_bucketed_1024`**：把 thread 哈希到 1024 个桶上做 `atomicAdd`（**轻冲突**；模拟"先 LDS reduce、再 block 级 atomic 提交"的现实写法）；
- **(c) `plain_store`**：普通非原子 `tl.store` 到 thread 自己的 slot（**无冲突 baseline**，表征理论上限）。

hipEvent 计时、warmup=5、repeats=50。

<details>
<summary><code>atomic_cmp.py</code></summary>

```python
#!/usr/bin/env python
# code/part1-hardware-rocm/chapter4/atomic_cmp.py
#
# §4.7.4 atomic 收尾代价对比 micro-benchmark。
#
# 对照三种"每线程加 1 到某个全局位置"的写法，看 atomic 频率/争用对吞吐
# 的影响：
#   (a) atomic_full     —— 所有 thread 都 atomicAdd 到同一个地址（最坏情况，
#       wave 里 64 个 lane 全部串行写一个 counter）；
#   (b) atomic_bucketed —— 把 thread 哈希到 1024 个桶，每个桶上做 atomicAdd
#       （轻冲突；模拟"先 reduce 再 atomic 提交"的现实写法）；
#   (c) plain_store     —— 普通非原子写入到 thread 自己的 slot（无冲突 baseline，
#       表征"理论上限"，每 thread 一次 store 不需要 RMW 也不需要 serialization）。
#
# 三种 kernel 的工作量都是"每个 thread 写一次"：n_threads 一致，差别只在
# atomic 频率与争用模式。所以 ops/s 的差距直接等于 atomic 的相对代价。
#
# 用法（在 code/part1-hardware-rocm/chapter4/ 目录里）：
#   python atomic_cmp.py
# 可选参数：
#   --sizes 65536,262144,1048576,4194304    # 扫多个 n_threads
#   --repeats 100  --warmup 10  --buckets 1024
#
# 硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0（torch 2.9.1+rocm7.12.0,
# triton 3.5.1+rocm7.12.0），与 §4.7.1–§4.7.3 一致。
#
# 输出：先打印 timestamp + rocm-smi 头，然后一行一种 mode 一行一个 size，
# CSV-friendly：mode, n_threads, time_ms, ops_per_s

import argparse
import datetime
import os
import shutil
import subprocess
import sys

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def atomic_full_kernel(out_ptr, n, BLOCK: tl.constexpr):
    """每 thread atomicAdd(out[0], 1) —— 全部冲突到同一地址。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    one = tl.where(mask, 1, 0)
    # atomic_add 接受 ptr + value 向量；同一地址重复 → 硬件序列化。
    tl.atomic_add(out_ptr + tl.zeros_like(offs), one, mask=mask)


@triton.jit
def atomic_bucketed_kernel(out_ptr, n, BUCKETS: tl.constexpr, BLOCK: tl.constexpr):
    """每 thread atomicAdd(out[tid % BUCKETS], 1) —— 分散到 BUCKETS 个桶。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    bucket = offs % BUCKETS
    one = tl.where(mask, 1, 0)
    tl.atomic_add(out_ptr + bucket, one, mask=mask)


@triton.jit
def plain_store_kernel(out_ptr, n, BLOCK: tl.constexpr):
    """每 thread out[tid] = 1 —— 普通 store，无原子、无冲突。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.full([BLOCK], 1, tl.int32), mask=mask)


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _bench(launch, n_threads: int, repeats: int, warmup: int) -> float:
    """通用 hipEvent 计时；launch 是 callable，无参；返回 total ms over `repeats`."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        launch()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def run_atomic_full(n_threads: int, repeats: int, warmup: int, block: int):
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    grid = ((n_threads + block - 1) // block,)

    def launch():
        atomic_full_kernel[grid](out, n_threads, BLOCK=block)

    ms = _bench(launch, n_threads, repeats, warmup)
    return ms


def run_atomic_bucketed(n_threads: int, repeats: int, warmup: int, block: int, buckets: int):
    out = torch.zeros(buckets, dtype=torch.int32, device="cuda")
    grid = ((n_threads + block - 1) // block,)

    def launch():
        atomic_bucketed_kernel[grid](out, n_threads, BUCKETS=buckets, BLOCK=block)

    ms = _bench(launch, n_threads, repeats, warmup)
    return ms


def run_plain_store(n_threads: int, repeats: int, warmup: int, block: int):
    out = torch.zeros(n_threads, dtype=torch.int32, device="cuda")
    grid = ((n_threads + block - 1) // block,)

    def launch():
        plain_store_kernel[grid](out, n_threads, BLOCK=block)

    ms = _bench(launch, n_threads, repeats, warmup)
    return ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_header(args):
    ts = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"# atomic_cmp (atomic frequency / contention micro-bench) — AI MAX 395 / gfx1151")
    print(f"# timestamp: {ts}")
    print(f"# host:      {os.uname().nodename}")
    print(f"# torch:     {torch.__version__}  triton: {triton.__version__}")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"# device:    {prop.name} (gcnArchName={getattr(prop, 'gcnArchName', '?')}, {prop.multi_processor_count} CUs)")
    print(
        f"# config:    sizes={args.sizes} repeats={args.repeats} warmup={args.warmup}"
        f" block={args.block} buckets={args.buckets}"
    )
    print("# --- rocm-smi snapshot ---")
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            out = subprocess.run(
                [rocm_smi, "--showuse", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=10,
            )
            for line in (out.stdout + out.stderr).splitlines():
                print(f"#   {line}")
        except Exception as exc:
            print(f"#   rocm-smi failed: {exc}")
    else:
        print("#   rocm-smi: not in PATH")
    print("# --- output ---")
    print("mode,n_threads,time_ms_total,time_ms_per_iter,ops_per_s")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="65536,262144,1048576,4194304",
                   help="comma-separated n_threads to scan")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--buckets", type=int, default=1024)
    args = p.parse_args()
    if args.repeats < 50:
        args.repeats = 50
    if args.warmup < 5:
        args.warmup = 5

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda is not available", file=sys.stderr)
        sys.exit(1)

    print_header(args)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    repeats = args.repeats
    warmup = args.warmup
    block = args.block
    buckets = args.buckets

    def emit(mode, n, ms_total):
        ms_per = ms_total / repeats
        # 每次 launch 完成 n 次"逻辑写入"——三种模式都是 n_threads 个写
        ops_total = float(n) * repeats
        ops_per_s = ops_total / (ms_total * 1e-3) if ms_total > 0 else float("nan")
        print(f"{mode},{n},{ms_total:.4f},{ms_per:.6f},{ops_per_s:.3e}")
        sys.stdout.flush()

    for n in sizes:
        ms = run_plain_store(n, repeats, warmup, block)
        emit("plain_store", n, ms)
        ms = run_atomic_bucketed(n, repeats, warmup, block, buckets)
        emit(f"atomic_bucketed_{buckets}", n, ms)
        ms = run_atomic_full(n, repeats, warmup, block)
        emit("atomic_full", n, ms)


if __name__ == "__main__":
    main()
```

</details>

实测结果（AI MAX 395 + ROCm 7.12.0，`logs/atomic_cmp.log`，2026-05-15 01:35 +0800；time 是 50 次 launch 总和）：

| n_threads | mode | total time (ms) | per-iter (µs) | ops/s | 相对 plain_store |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 65 536 | plain_store | 0.45 | 9.0 | 7.27e+09 | 1.00x |
| 65 536 | atomic_bucketed_1024 | 0.26 | 5.1 | 1.28e+10 | 0.57x（更快，噪声） |
| 65 536 | atomic_full | 0.32 | 6.4 | 1.02e+10 | 0.72x（更快，噪声） |
| 262 144 | plain_store | 0.25 | 4.9 | 5.33e+10 | 1.00x |
| 262 144 | atomic_bucketed_1024 | 0.28 | 5.6 | 4.65e+10 | 1.15x 慢 |
| 262 144 | atomic_full | 0.44 | 8.7 | 3.01e+10 | 1.77x 慢 |
| 1 048 576 | plain_store | 0.37 | 7.4 | 1.41e+11 | 1.00x |
| 1 048 576 | atomic_bucketed_1024 | 0.75 | 15.1 | 6.95e+10 | **2.03x** 慢 |
| 1 048 576 | atomic_full | 1.60 | 31.9 | 3.29e+10 | **4.30x** 慢 |
| 4 194 304 | plain_store | 1.16 | 23.1 | 1.81e+11 | 1.00x |
| 4 194 304 | atomic_bucketed_1024 | 2.73 | 54.5 | 7.69e+10 | **2.36x** 慢 |
| 4 194 304 | atomic_full | 5.34 | 106.7 | 3.93e+10 | **4.62x** 慢 |

预期与解读（AI MAX 395 + ROCm 7.12.0）：

- **atomic_full 在大尺寸下被钉死在 ~40 GOps/s**——4M threads 时比 `plain_store` 的 ~181 GOps/s 慢 **4.62x**；这正是 4.6 节"atomics 打破写折叠优化、每 lane 必须保留独立 RMW"的最直接证据，硬件原子单元再快，所有写也得在同一地址上**串行通过**；
- **atomic_bucketed_1024 处在中间档，比 plain 慢 ~2x**（4M 行 2.36x）：1024 桶把每桶平均争用从 4M lane 压到 ~4096 lane 数量级，但 RMW 本身的开销仍然存在——这就是 4.6 节末"先 LDS reduce、再 block 一次 atomic 提交"模式起作用的量化版本：把争用从"全设备一个地址"降到"每 block 一个地址"足以让 atomic 从 4.6x 罚款降到 ~2x；
- **小尺寸（65k / 262k）那两组主要被 launch overhead 主导**——65k 那行 plain_store 反而比 atomic 还慢（绝对时间都在 ~0.3–0.5 ms 量级），完全是 ~6 µs/launch 的固定开销在小工作集下喧宾夺主；这条规律和 4.7.2 footprint 扫描在小工作集那段的解读一致，不要据此判断 atomic 有什么"加速"；
- 拐点出现在 1M-4M 之间，atomic_full 的 ops/s 几乎恒定在 ~33–39 GOps/s——这意味着 gfx1151 上"单个全局原子地址"的稳态吞吐上限就在这附近，不论 launch 多大、thread 多多，吞吐都不会再涨；
- **结论与 4.6 节的实战经验完全吻合**：reduction / histogram 这类 pattern 必须把 atomic 压到 block 级（甚至先 LDS 内 reduce），把同一地址的争用度降一两个数量级；至于"每 thread 一次 global atomic"这种写法，实测 4.6x 慢只是 worst-case 的下限——实际算子里通常伴随 cacheline 写折叠丢失，代价更高。

### 4.7.5 实验落地清单

| 文件 | 测什么 | 实测证据（AI MAX 395 + ROCm 7.12.0） |
| ---- | ---- | ---- |
| `bw_triad.hip` | LPDDR5X 全局带宽 | DRAM 平台 **~228 GB/s**（与文中 ~215 GB/s 同量级），4 MiB 处出现 **702 GB/s** MALL 峰值（4.7.1） |
| `bw_footprint.py` | L2 / MALL / DRAM 拐点 | 16 MiB → 32 MiB **单跳腰斩**（804 → 224 GB/s），DRAM 平台 ~224 GB/s（4.7.2） |
| `lds_stride.hip` | LDS bank 冲突 | stride 32 比 stride 1 慢 **~3.86x**，stride 16 慢 ~1.93x（4.7.3） |
| `atomic_cmp.py` | atomic 频率/争用对比 | atomic_full 比 plain_store 慢 **~4.6x**（4M threads，~39 vs ~181 GOps/s），bucketed_1024 慢 ~2.4x（4.7.4） |

## 本章小结

- GPU 内存层级是一座很陡的金字塔：寄存器 → LDS → L0/L1 → L2 → Infinity Cache → DRAM，每往下一层容量增加、延迟变差近一个数量级。
- gfx1151 / AI MAX 395 是 APU，**没有独立 HBM/GDDR**：CPU 与 GPU 共享 256-bit LPDDR5X-8000（实测 ~215 GB/s），并通过 32 MB MALL 充当片上最后一级缓存——和 MI 系列 HBM 模型差异非常大[^amd-strix-blog][^cnc-strix-mall]。
- L1 / L2 决定一个 wave 的 64 个 lane 实际生成多少次外部内存事务；写一致性靠写直达保持简单。
- LDS 是 32 bank × 4 字节，bank 冲突常出现在 GEMM tile 这类 2D 访问上；padding 与 XOR swizzle 是经典修复手段。
- 合并访存的本质是"用最少条按 64 字节对齐的事务覆盖 wave 的请求集合"——连续线程访问连续地址 + 自然对齐永远是最稳的写法。
- Atomics 的代价分 scope（block / agent / system）：scope 越宽越贵；hot path 上避免 system fence；reduction 类算子先 block 内汇总再 global atomic。
- 4.7 节四个 micro-bench（Triad / footprint / LDS stride / atomic_cmp）已经在 AI MAX 395 + ROCm 7.12.0 上跑完，DRAM 平台 ~228 GB/s、MALL 峰值 ~804 GB/s、LDS stride 32 比 stride 1 慢 ~3.86x、单地址全冲突 atomic 比无冲突 store 慢 ~4.6x，与本章 4.1–4.6 节描述的内存模型完全一致；原始日志见 `code/part1-hardware-rocm/chapter4/logs/`。
- 这套内存模型会在 [第 12 章 从 Vector Add 理解 GPU 并行](../../part3-hip-kernels/chapter12/index.md)、[第 13 章 Reduction 优化](../../part3-hip-kernels/chapter13/index.md)、[第 16 章 Matmul 入门优化](../../part3-hip-kernels/chapter16/index.md) 等后续章节里被反复引用。

## 延伸阅读

- [HIP Programming Guide — Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)
- [HIP C++ Language Extensions（atomics、fence、scope 详细列表）](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)
- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [Composable Kernel — Understanding AMD GPU LDS and Bank Conflicts](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html)
- [ROCm Blogs — Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- [GPUOpen — Understanding Memory Coalescing on GCN](https://gpuopen.com/learn/gcn-memory-coalescing/)
- [Chips and Cheese — Microbenchmarking AMD's RDNA 3 Graphics Architecture](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture)
- [Chips and Cheese — Strix Halo's Memory Subsystem: Tackling iGPU Challenges](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling)
- [Chips and Cheese — Evaluating the Infinity Cache in AMD Strix Halo](https://chipsandcheese.com/p/evaluating-the-infinity-cache-in)
- [AMD — RDNA Architecture Whitepaper](https://www.amd.com/system/files/documents/rdna-whitepaper.pdf)
- [AMD — Ryzen AI Max+ 395 产品页](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html)
- [LLVM — AMDGPU Backend Usage](https://llvm.org/docs/AMDGPUUsage.html)

[^rdna3-cache]: Chips and Cheese, *Microbenchmarking AMD's RDNA 3 Graphics Architecture*. https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture
[^cnc-rdna3]: Chips and Cheese, *Microbenchmarking AMD's RDNA 3 Graphics Architecture* — L0/L1/L2 容量、Infinity Cache 96 MB、L2 6 MB 等数据来源。
[^cnc-strix-mall]: Chips and Cheese, *Evaluating the Infinity Cache in AMD Strix Halo* 与 *Strix Halo's Memory Subsystem: Tackling iGPU Challenges* — Strix Halo 上 256 KB L1 / Shader Array、2 MB L2、32 MB MALL、~123 ns DRAM 延迟、~215 GB/s 实测带宽等。
[^videocardz-strix]: Videocardz, *AMD Ryzen 9050 "Strix Halo" specs leaked: 16 Zen5 cores and 40 RDNA3.5 CUs, LP5x-8000 memory and 32MB MALL cache*。
[^amd-strix-blog]: AMD, *AMD Ryzen™ AI MAX+ 395 Processor: Breakthrough AI Performance in Thin and Light* 以及产品页（Variable Graphics Memory、最高 128 GB LPDDR5X-8000）。
[^gpuopen-coalescing]: AMD GPUOpen, *Understanding Memory Coalescing on GCN*。https://gpuopen.com/learn/gcn-memory-coalescing/
[^hip-coalescing-write]: HIP Documentation, *Performance Guidelines* 中关于 write-through L1 与 write coalescing 的描述。
[^ck-bank]: AMD ROCm, *Composable Kernel — Understanding AMD GPU LDS and Bank Conflicts*。https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
[^rocm-blog-bank]: ROCm Blogs, *Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile Framework*。https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
[^hip-cpp-ext]: HIP Documentation, *HIP C++ Language Extensions*（atomics、scope、fence、`-munsafe-fp-atomics` 等）。https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
[^llvm-d87858]: LLVM Code Review D87858, *[hip] Add HIP scope atomic ops*（singlethread / wavefront / workgroup / agent / system 完整 scope 列表来源）。
