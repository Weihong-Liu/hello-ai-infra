# Hello-GPU 重构计划（REFACTOR-PLAN）

> 本文件记录 2026-06 重构的设计决策与定稿大纲，供后续协作和 PR 引用。
> 本文件不进 VitePress 站点（站点只构建 `docs/`），仅作为仓库内的规划留档。
> 实施时按「规范层 → 骨架 → 内容」的顺序推进。

## 1. 重构动机

旧版（Alpha，31 章）定位偏「AI Infra 全栈百科」，覆盖硬件→profiling→HIP→Triton→推理→编译器→Agent。问题：

- **篇幅过重**：和 hello-gpu 的「简单入门」定位不符。
- **同质化**：Vector Add / Reduction / Softmax / LayerNorm / Matmul 套餐和市面教程雷同，缺辨识度。
- **目标读者失焦**：同时讲 RDNA vs CDNA、编译器原理、Agent，初学者被劝退。
- **Agent 篇定位尴尬**：README/导航未露面，占 8 章但未融入主线。

## 2. 新定位

**旧**：AI Infra 全栈教程（31 章）
**新**：GPU 算子优化入门 + Agent 自动化（18 章）

**一句话价值主张**：拿到一张 AMD 显卡（9070XT），学会几个经典算子从慢到快的优化思路，最后亲手搭一个能自动做这件事的 Agent。不追求覆盖全栈，追求**一个闭环讲透**。

## 3. 关键设计决策

| # | 决策 | 理由 |
|---|---|---|
| 1 | 硬件完全替换为 9070XT（RDNA4 / gfx12 / ROCm 6.4.x / Linux-only）| 消费卡更易获得，符合入门定位 |
| 2 | Profiling 篇前置到算子篇之前 | 先会「看数据」再学「改代码」，profiling-driven 正道 |
| 3 | Triton 不单设篇，合并进算子篇做 HIP/Triton 对比 | 避免读者学两遍同一算子 |
| 4 | 砍 LayerNorm（和 Softmax 重复），加 Flash Attention 思路章 | 体现现代优化视角 |
| 5 | LeetGPU 刷题章写硬件无关方法论 | 平台现状是 CUDA-only，方法论通用 |
| 5 | Agent 占 6 章（Part 3 算子层 + Part 4 真实模型），为压轴卖点 | 参考 hello-agents 节奏 |
| 6 | Agent 能力边界递进：单 kernel → 真实模型 | 边界差异本身就是教学点 |
| 7 | Vector Add 贯穿 Ch3-6 作为 profiling 固定教学算子 | 不在算子篇重复 |
| 8 | Roofline 出现两次：Ch3 建直觉、Ch6 用真实数据画点 | 概念层 vs 证据层，不重复 |

## 4. 定稿大纲（v5，18 章 5 篇）

| 篇 | 章 | 标题 | 核心内容 | 实操对象 |
|---|---|---|---|---|
| **Part 0 入门** | 0 | 写给读者的话 | 教程定位、为什么选 9070XT、和市面教程差异、学习路线 | — |
| | 1 | 环境准备 | 9070XT + Linux + ROCm 6.4 验证、Windows 劝退、uv 环境、最小 smoke test | 环境 |
| | 2 | GPU 体系结构速通 | CU/Wavefront/LDS/寄存器/显存层次（RDNA4/gfx12 视角，砍 MFMA/CDNA/HBM）| 概念 |
| | 3 | 第一个程序 + Roofline 心智模型 | vector add 跑通、建立「理论上限」直觉 | vector add |
| **Part 1 Profiling 实战** | 4 | benchmark 与可信计时 | 热身、重复、GPU event、避免测量陷阱 | vector add |
| | 5 | rocprof + Omniperf 定位瓶颈 | kernel 耗时、访存/占用率计数器、strided 非合并反例 | vector add |
| | 6 | Roofline 曲线详解 + 性能报告 | 把算子点画到 Roofline、解释差距、报告模板 | vector add |
| **Part 2 算子优化 + 刷题** | 7 | Reduction：naive→LDS→wavefront | 跨线程归约、LDS 协作、多阶段、HIP/Triton 对比 | reduction |
| | 8 | Softmax：数值稳定 + 融合 | 减最大值、block 并行、减少写回、HIP/Triton 对比 | softmax |
| | 9 | GEMM：tiling + LDS | 分块复用、寄存器 blocking、不追 rocBLAS、HIP/Triton 对比 | matmul |
| | 10 | Flash Attention 思路 | 分块 + 在线 softmax、不物化中间矩阵（算子篇压轴）| attention |
| | 11 | 怎么刷 LeetGPU | 平台题型/评分、本地评测器、调试策略、性能闭环（硬件无关方法论）| 题目套路 |
| **Part 3 Agent（算子层）** | 12 | Agent 入门 | 参考 hello-agents、LLM Agent 基本范式、工具调用 | — |
| | 13 | 工具封装 | benchmark/profiling/编译包成 Agent 可调用工具 | 复用 Part1 工具 |
| | 14 | 算子优化 Agent 设计 | 读题→生成 kernel→跑分→反思迭代 | 教学算子 |
| | 15 | 多轮优化实战 | Agent 把 naive kernel 优化 3-5x、失败回退、对比报告 | 教学算子 |
| **Part 4 真实模型 + Agent** | 16 | YOLO 部署 + Agent 自动优化 | ONNX/MIGraphX 部署、Agent profiling 找瓶颈、改配置/算子、对比 | YOLOv8 |
| | 17 | 小模型 LLM 解码 + Agent 自动优化 | Qwen 0.5B/1.8B 量化、decode 算子视角、Agent 优化 KV cache/精度 | Qwen 小模型 |

## 5. Agent 能力边界（重要教学点）

| 层面 | Agent 能自动做 | Agent 做不到（入门范畴）|
|---|---|---|
| **算子层**（Part 3）| 生成/迭代单个 kernel 代码、跑 benchmark、反思改写 | — |
| **模型层**（Part 4，YOLO/LLM）| 自动 profiling、识别瓶颈、给配置建议（batch/精度/算子融合/换 MIGraphX）、改配置跑对比、出报告 | 自动写新算子集成进 ONNX/vLLM、自动改模型结构、自动实现 PagedAttention |

这个边界在 Ch16 开头明确点出，让读者理解「Agent 自动化是有层次的」。

## 6. 三个现实约束

1. **9070XT 显存 16GB GDDR6** → LLM 案例只能量化小模型（Qwen 0.5B/1.8B），Ch17 选型已定。
2. **RDNA4 无 MFMA** → GEMM/Attention 章的硬件论据靠 WMMA + 寄存器复用，不能讲 MFMA 流水线。等机器到手实测后定。
3. **9070XT 带宽约 760 GB/s（GDDR6，非 HBM）** → Roofline 拐点位置和 AI MAX 395 完全不同，profiling 数据形态会大改。

## 7. 和外部生态的边界声明

- **单卡部署/算子/Agent 优化**：留在本书。
- **多卡通信 / 服务化 / 动态 batching / 多副本**：指向 hello-mlsys / hello-ai-infra。
- **LeetGPU 平台现状**：目前仅支持 CUDA/Triton/PyTorch。本书 Ch11 讲方法论，等 AMD 等价平台出现后迁移成本很低。

## 8. 实施分阶段（机器未就绪）

- **阶段 A（现在可做）**：文档骨架 + 代码 + `.docs-rules` 规范更新。性能数字处留 `🚧 待 job:xxx 填充` 占位（复用现有 `scripts/gpu-queue/` 机制）。
- **阶段 B（9070XT 机器就绪后）**：跑 `gpu-queue` 填数字。第一批优先 Ch3-Ch6（vector add + profiling）和 Ch7-Ch10（四个算子）的 baseline + 优化版。

## 9. 规范层改造清单（同步进行）

| 文件 | 改动要点 |
|---|---|
| `.docs-rules/00-overview.md` | 定位改为「算子优化入门 + Agent 自动化」；阶段说明更新；Out of Scope 调整 |
| `.docs-rules/01-writing-style.md` | 硬件上下文示例从 AI MAX 395 改为 9070XT；新增刷题/Agent 内容的写作约定 |
| `.docs-rules/02-experiment-policy.md` | 主线硬件措辞更新；保留 `🚧 待补实验` 占位机制（机器未就绪）|
| `.docs-rules/03-environment.md` | 实验机 host/路径/架构/ROCm 版本/wheel 源全部更新；新增 Linux-only 说明 |
| `.docs-rules/04-hardware-matrix.md` | 主线设备 AI MAX 395 → 9070XT；gfx1151 → gfx12；ROCm 7.12.0 → 6.4.x |
| `.docs-rules/05-repo-layout.md` | 新 5 篇目录约定；旧 part2/part4/part5/part6 目录归档说明 |
| `.docs-rules/README.md` | 各文件一句话总结同步更新 |
| `AGENTS.md` | 远程实验速查里的 host 名、路径 |
| `docs/.vitepress/outline.mjs` | 整张 parts 数组重写（阶段 A 骨架）|
| `README.md` | 目录表重写（由 `npm run docs:sync-outline` 自动生成）|
| `scripts/gpu-queue/jobs/*.json` | 13 个 job 的 code_dir/chapter_doc 重新对应 |
| `code/` 旧目录 | part2/part4/part5/part6 归档或删除 |

> 本清单中前 7 项（`.docs-rules/`）在本次重构的第一步完成；`AGENTS.md`、`outline.mjs`、`README.md`、`scripts/`、`code/` 在后续步骤处理。
