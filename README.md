<div align='center'>
  <h1>Hello AI Infra</h1>
  <h3>从硬件到智能体的 AI 基础设施实践教程</h3>
  <p><em>面向 AMD GPU / ROCm，系统学习算子优化、推理优化、AI 编译器与自动优化 Agent</em></p>
</div>

---

## 项目介绍

随着大模型和多模态模型快速发展，AI Infra 已经成为连接算法、系统与硬件的关键方向。很多学习者会使用模型、会调用框架，却不知道模型为什么跑得慢、瓶颈在哪里、应该如何优化。

本教程希望解决这个问题：
当你拿到一块 AMD GPU 时，如何从硬件出发理解它的执行方式，如何通过 profiling 找到性能瓶颈，如何优化算子和推理 pipeline，并最终构建一个 AI Agent，自动完成性能分析、优化建议、代码修改、benchmark 和报告生成。

本教程不是简单介绍工具，而是希望帮助读者建立 AI Infra 的核心思维：
**以硬件为起点，以 profiling 为证据，以优化为手段，以 Agent 自动化为终点。**

## 创新点

- **AMD First**：不是 CUDA 教程换皮，而是围绕 AMD GPU / ROCm 生态设计
- **Hardware-Aware**：先教硬件能力，再教程序映射与优化原理
- **Profiling-Driven**：每个优化都必须有证据支撑
- **Agent-Driven**：让 AI 参与完整优化闭环
- **Project-Based**：围绕真实项目推进，从单算子到 AutoInfra Agent

## 目标读者

1. 想进入 AI Infra / AI 系统 / 推理优化 / 算子优化方向的学生和开发者
2. 有 Python / Linux 基础，但对 GPU、ROCm、AI 编译器不熟悉的初学者
3. 做过模型训练或部署，但不知道模型为什么跑得慢、怎么优化的工程师
4. 想基于 AI Agent 自动完成 profiling、benchmark、代码调优和报告生成的开发者
5. 希望了解 AMD GPU / ROCm 生态，而不是只学习 CUDA 体系的学习者

> 你不需要有 GPU 编程经验，但最好具备基础 Python、Linux 命令行和深度学习概念。

## 内容导航

### 第 0 篇：前言与学习路线

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 0 章 写给读者的话](./docs/part0-preface/chapter0/index.md) | AI Infra 的重要性、教程特色、学习路线 | 🚧 |

### 第 1 篇：AI Infra 全景与 AMD GPU 基础

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 1 章 AI Infra 全景图](./docs/part1-hardware-rocm/chapter1/index.md) | 从模型到硬件的完整链路、HPOA 方法论 | 🚧 |
| [第 2 章 AMD GPU 与 ROCm 软件栈](./docs/part1-hardware-rocm/chapter2/index.md) | AMD GPU 架构、ROCm 生态、环境检查 | 🚧 |
| [第 3 章 第一个 AMD GPU 程序](./docs/part1-hardware-rocm/chapter3/index.md) | ROCm 安装、PyTorch ROCm、第一个 HIP kernel | 🚧 |

### 第 2 篇：性能分析与瓶颈定位

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 4 章 性能优化的基本方法论](./docs/part2-profiling/chapter4/index.md) | Roofline、Memory-bound / Compute-bound | 🚧 |
| [第 5 章 AMD GPU Profiling 工具链](./docs/part2-profiling/chapter5/index.md) | rocprof、Omniperf、PyTorch Profiler | 🚧 |
| [第 6 章 建立你的第一个性能分析报告](./docs/part2-profiling/chapter6/index.md) | 选择算子、采集数据、生成报告 | 🚧 |

### 第 3 篇：HIP 算子优化实战

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 7 章 HIP 编程基础](./docs/part3-hip-kernels/chapter7/index.md) | Kernel、Thread、Block、Grid、Memory 管理 | 🚧 |
| [第 8 章 从 Vector Add 理解 GPU 并行](./docs/part3-hip-kernels/chapter8/index.md) | 线程映射、访存合并、benchmark | 🚧 |
| [第 9 章 Reduction 优化](./docs/part3-hip-kernels/chapter9/index.md) | LDS 优化、Wavefront 优化、多阶段 Reduction | 🚧 |
| [第 10 章 Softmax 优化](./docs/part3-hip-kernels/chapter10/index.md) | 数值稳定性、访存优化、Block 级并行 | 🚧 |
| [第 11 章 LayerNorm 优化](./docs/part3-hip-kernels/chapter11/index.md) | Reduction + Normalize 融合、向量化读写 | 🚧 |
| [第 12 章 Matmul 入门优化](./docs/part3-hip-kernels/chapter12/index.md) | Tiling、LDS 缓存、Register Blocking | 🚧 |

### 第 4 篇：Triton on AMD 与自动调参

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 13 章 Triton 编程模型](./docs/part4-triton/chapter13/index.md) | Triton vs HIP、block/program model | 🚧 |
| [第 14 章 Triton 实现常见算子](./docs/part4-triton/chapter14/index.md) | Vector Add、Matmul、Softmax、Attention | 🚧 |
| [第 15 章 Triton 自动调参](./docs/part4-triton/chapter15/index.md) | autotune、搜索空间设计、自动 benchmark | 🚧 |

### 第 5 篇：推理优化与模型部署

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 16 章 推理优化全景](./docs/part5-inference/chapter16/index.md) | 延迟与吞吐、精度量化、端到端 pipeline | 🚧 |
| [第 17 章 ONNX Runtime / MIGraphX](./docs/part5-inference/chapter17/index.md) | ONNX 导出、图优化、性能对比 | 🚧 |
| [第 18 章 Triton Inference Server](./docs/part5-inference/chapter18/index.md) | Model Repository、动态 batching | 🚧 |
| [第 19 章 YOLO 推理优化案例](./docs/part5-inference/chapter19/index.md) | 预处理优化、NMS 优化、pipeline profiling | 🚧 |
| [第 20 章 LLM 推理优化案例](./docs/part5-inference/chapter20/index.md) | KV Cache、vLLM on AMD、吞吐优化 | 🚧 |

### 第 6 篇：AI 编译器与自动调优

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 21 章 AI 编译器到底在优化什么](./docs/part6-compiler/chapter21/index.md) | 模型图到计算图到算子到 kernel 到 ISA | 🚧 |
| [第 22 章 图优化基础](./docs/part6-compiler/chapter22/index.md) | 算子融合、常量折叠、布局优化 | 🚧 |
| [第 23 章 Kernel 生成与调度搜索](./docs/part6-compiler/chapter23/index.md) | Schedule 原语、搜索空间、AutoScheduler | 🚧 |
| [第 24 章 TVM / Triton / MIGraphX 对比](./docs/part6-compiler/chapter24/index.md) | 三个工具的定位和选择指南 | 🚧 |

### 第 7 篇：AutoInfra Agent 自动优化系统

| 章节 | 关键内容 | 状态 |
|------|----------|------|
| [第 25 章 为什么 AI Infra 需要 Agent](./docs/part7-agent/chapter25/index.md) | Agent 工作流、LLM 角色、总架构 | 🚧 |
| [第 26 章 Hardware Inspector](./docs/part7-agent/chapter26/index.md) | 自动检测硬件、生成硬件画像 | 🚧 |
| [第 27 章 Benchmark Agent](./docs/part7-agent/chapter27/index.md) | 自动 benchmark、保存结果、对比版本 | 🚧 |
| [第 28 章 Profiler Agent](./docs/part7-agent/chapter28/index.md) | 自动 profiling、提取慢 kernel | 🚧 |
| [第 29 章 Optimization Planner](./docs/part7-agent/chapter29/index.md) | 瓶颈判断、生成优化策略、实验计划 | 🚧 |
| [第 30 章 Code Agent](./docs/part7-agent/chapter30/index.md) | 自动修改参数、生成 kernel、回滚 | 🚧 |
| [第 31 章 Report Agent](./docs/part7-agent/chapter31/index.md) | 自动报告、before/after 对比 | 🚧 |
| [第 32 章 毕业项目：AutoInfra Agent](./docs/part7-agent/chapter32/index.md) | 完整系统、三个端到端案例 | 🚧 |

## 项目结构

```
hello-ai-infra/
├── docs/
│   ├── part0-preface/          # 前言与学习路线
│   │   └── chapter0/
│   │       └── index.md
│   ├── part1-hardware-rocm/    # AI Infra 全景与 AMD GPU 基础
│   │   ├── chapter1/
│   │   ├── chapter2/
│   │   └── chapter3/
│   ├── part2-profiling/        # 性能分析与瓶颈定位
│   │   ├── chapter4/
│   │   ├── chapter5/
│   │   └── chapter6/
│   ├── part3-hip-kernels/      # HIP 算子优化实战
│   │   ├── chapter7/
│   │   └── ... (chapter8-12)
│   ├── part4-triton/           # Triton on AMD 与自动调参
│   │   ├── chapter13/
│   │   └── ... (chapter14-15)
│   ├── part5-inference/        # 推理优化与模型部署
│   │   ├── chapter16/
│   │   └── ... (chapter17-20)
│   ├── part6-compiler/         # AI 编译器与自动调优
│   │   ├── chapter21/
│   │   └── ... (chapter22-24)
│   └── part7-agent/            # AutoInfra Agent
│       ├── chapter25/
│       └── ... (chapter26-32)
├── code/                       # 配套代码（与 docs 同构）
│   ├── part0-preface/
│   ├── part1-hardware-rocm/
│   ├── ...
│   └── part7-agent/
├── assets/                     # 图片、架构图
├── notebooks/                  # Jupyter Notebook 实验
├── Extra-Chapter/              # 扩展章节
├── Co-creation-projects/       # 共创项目
└── reference_repositories/     # 参考资料
```

## 学习方法

本教程推荐以下学习闭环：

**理解 → 测量 → 优化 → 自动化**

1. **理解**：先搞清楚概念和原理
2. **测量**：用 profiling 工具获取真实数据
3. **优化**：基于证据做针对性优化
4. **自动化**：把优化流程交给 Agent

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
