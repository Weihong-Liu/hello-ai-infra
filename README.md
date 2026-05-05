<h1 align="center"> Hello AI Infra ⚠️ Alpha内测版 </h1>

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。

随着大模型和多模态模型快速发展，AI Infra 已经成为连接算法、系统与硬件的关键方向。很多学习者会使用模型、会调用框架，却不知道模型为什么跑得慢、瓶颈在哪里、应该如何优化。

本教程希望解决这个问题：
当你拿到一块 AMD GPU 时，如何从硬件出发理解它的执行方式，如何通过 profiling 找到性能瓶颈，如何优化算子和推理 pipeline，并最终构建一个 AI Agent，自动完成性能分析、优化建议、代码修改、benchmark 和报告生成。

本教程不是简单介绍工具，而是希望帮助读者建立 AI Infra 的核心思维：
**以硬件为起点，以 profiling 为证据，以优化为手段，以 Agent 自动化为终点。**

## 项目受众

- 想进入 AI Infra / AI 系统 / 推理优化 / 算子优化方向的学生和开发者
- 有 Python / Linux 基础，但对 GPU、ROCm、AI 编译器不熟悉的初学者
- 做过模型训练或部署，但不知道模型为什么跑得慢、怎么优化的工程师
- 想基于 AI Agent 自动完成 profiling、benchmark、代码调优和报告生成的开发者
- 希望了解 AMD GPU / ROCm 生态，而不是只学习 CUDA 体系的学习者

> 你不需要有 GPU 编程经验，但最好具备基础 Python、Linux 命令行和深度学习概念。

## 在线阅读

https://weihong-liu.github.io/hello-ai-infra/

## 目录

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| **第 0 篇：前言与学习路线** | | |
| [第 0 章 写给读者的话](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part0-preface/chapter0/index.md) | AI Infra 的重要性、教程特色、学习路线 | 🚧 |
| **第 1 篇：AI Infra 全景与 AMD GPU 基础** | | |
| [第 1 章 AI Infra 全景图](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part1-hardware-rocm/chapter1/index.md) | 从模型到硬件的完整链路、HPOA 方法论 | 🚧 |
| [第 2 章 AMD GPU 与 ROCm 软件栈](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part1-hardware-rocm/chapter2/index.md) | AMD GPU 架构、ROCm 生态、环境检查 | 🚧 |
| [第 3 章 第一个 AMD GPU 程序](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part1-hardware-rocm/chapter3/index.md) | ROCm 安装、PyTorch ROCm、第一个 HIP kernel | 🚧 |
| **第 2 篇：性能分析与瓶颈定位** | | |
| [第 4 章 性能优化的基本方法论](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter4/index.md) | Roofline、Memory-bound / Compute-bound | 🚧 |
| [第 5 章 AMD GPU Profiling 工具链](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter5/index.md) | rocprof、Omniperf、PyTorch Profiler | 🚧 |
| [第 6 章 建立你的第一个性能分析报告](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter6/index.md) | 选择算子、采集数据、生成报告 | 🚧 |
| **第 3 篇：HIP 算子优化实战** | | |
| [第 7 章 HIP 编程基础](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter7/index.md) | Kernel、Thread、Block、Grid、Memory 管理 | 🚧 |
| [第 8 章 从 Vector Add 理解 GPU 并行](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter8/index.md) | 线程映射、访存合并、benchmark | 🚧 |
| [第 9 章 Reduction 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter9/index.md) | LDS 优化、Wavefront 优化、多阶段 Reduction | 🚧 |
| [第 10 章 Softmax 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter10/index.md) | 数值稳定性、访存优化、Block 级并行 | 🚧 |
| [第 11 章 LayerNorm 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter11/index.md) | Reduction + Normalize 融合、向量化读写 | 🚧 |
| [第 12 章 Matmul 入门优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter12/index.md) | Tiling、LDS 缓存、Register Blocking | 🚧 |
| **第 4 篇：Triton on AMD 与自动调参** | | |
| [第 13 章 Triton 编程模型](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter13/index.md) | Triton vs HIP、block/program model | 🚧 |
| [第 14 章 Triton 实现常见算子](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter14/index.md) | Vector Add、Matmul、Softmax、Attention | 🚧 |
| [第 15 章 Triton 自动调参](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter15/index.md) | autotune、搜索空间设计、自动 benchmark | 🚧 |
| **第 5 篇：推理优化与模型部署** | | |
| [第 16 章 推理优化全景](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter16/index.md) | 延迟与吞吐、精度量化、端到端 pipeline | 🚧 |
| [第 17 章 ONNX Runtime / MIGraphX](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter17/index.md) | ONNX 导出、图优化、性能对比 | 🚧 |
| [第 18 章 Triton Inference Server](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter18/index.md) | Model Repository、动态 batching | 🚧 |
| [第 19 章 YOLO 推理优化案例](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter19/index.md) | 预处理优化、NMS 优化、pipeline profiling | 🚧 |
| [第 20 章 LLM 推理优化案例](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter20/index.md) | KV Cache、vLLM on AMD、吞吐优化 | 🚧 |
| **第 6 篇：AI 编译器与自动调优** | | |
| [第 21 章 AI 编译器到底在优化什么](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter21/index.md) | 模型图到计算图到算子到 kernel 到 ISA | 🚧 |
| [第 22 章 图优化基础](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter22/index.md) | 算子融合、常量折叠、布局优化 | 🚧 |
| [第 23 章 Kernel 生成与调度搜索](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter23/index.md) | Schedule 原语、搜索空间、AutoScheduler | 🚧 |
| [第 24 章 TVM / Triton / MIGraphX 对比](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter24/index.md) | 三个工具的定位和选择指南 | 🚧 |
| **第 7 篇：AutoInfra Agent 自动优化系统** | | |
| [第 25 章 为什么 AI Infra 需要 Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter25/index.md) | Agent 工作流、LLM 角色、总架构 | 🚧 |
| [第 26 章 Hardware Inspector](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter26/index.md) | 自动检测硬件、生成硬件画像 | 🚧 |
| [第 27 章 Benchmark Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter27/index.md) | 自动 benchmark、保存结果、对比版本 | 🚧 |
| [第 28 章 Profiler Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter28/index.md) | 自动 profiling、提取慢 kernel | 🚧 |
| [第 29 章 Optimization Planner](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter29/index.md) | 瓶颈判断、生成优化策略、实验计划 | 🚧 |
| [第 30 章 Code Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter30/index.md) | 自动修改参数、生成 kernel、回滚 | 🚧 |
| [第 31 章 Report Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter31/index.md) | 自动报告、before/after 对比 | 🚧 |
| [第 32 章 毕业项目：AutoInfra Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter32/index.md) | 完整系统、三个端到端案例 | 🚧 |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :---- | :---- | :---- |
| Weihong Liu | 项目负责人 | AI Infra 方向，专注 AMD GPU 优化与 AI 编译器 |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
