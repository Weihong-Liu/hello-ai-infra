<h1 align="center"> Hello AI Infra ⚠️ Alpha内测版 </h1>

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。

随着大模型和多模态模型快速发展，AI Infra 已经成为连接算法、系统与硬件的关键方向。很多学习者会使用模型、会调用框架，却不知道模型为什么跑得慢、瓶颈在哪里、应该如何优化。

本教程希望解决这个问题：
当你拿到一块 AMD GPU 时，如何从硬件出发理解它的执行方式，如何通过 profiling 找到性能瓶颈，如何优化算子和推理 pipeline，并最终构建一个 AI Agent，自动完成性能分析、优化建议、代码修改、benchmark 和报告生成。

本教程不是简单介绍工具，而是希望帮助读者建立 AI Infra 的核心思维：
**以硬件为起点，以 profiling 为证据，以优化为手段，以 Agent 自动化为终点。**

> Alpha 阶段所有实验默认以 **AI MAX 395 + ROCm 7.12.0** 为基线。其他 AMD GPU 可以参考方法论，但性能数字和工具可用性需要单独实测确认。

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

> 前言 + 7 篇正文，共 37 章。显示章号由 `docs/.vitepress/outline.mjs` 自动生成，新增章节后运行 `npm run docs:sync-outline` 即可同步 README 与站点导航。

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| **第 0 篇：前言与学习路线** | | |
| [第 0 章 写给读者的话](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part0-preface/chapter0/index.md) | AI Infra 的重要性、教程特色、学习路线 | 🚧 |
| [第 1 章 环境准备与验证](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part0-preface/chapter1/index.md) | uv sync、AI MAX 395 + ROCm 7.12.0 基线、最小环境验证 | 🚧 |
| **第 1 篇：AI Infra 全景与 AMD GPU 基础** | | |
| [第 2 章 AI Infra 全景图](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part1-hardware-rocm/chapter1/index.md) | 从模型到硬件的完整链路、HPOA 方法论 | 🚧 |
| [第 3 章 AMD GPU 与 ROCm 软件栈](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part1-hardware-rocm/chapter2/index.md) | AMD GPU 执行模型、ROCm 层次、上层框架关系 | 🚧 |
| [第 4 章 第一个 AMD GPU 程序](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part1-hardware-rocm/chapter3/index.md) | PyTorch ROCm、最小 HIP kernel、baseline benchmark | 🚧 |
| **第 2 篇：性能分析与瓶颈定位** | | |
| [第 5 章 性能优化的基本方法论](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter4/index.md) | Latency、Throughput、Bandwidth、FLOPS、Roofline、可信 benchmark | 🚧 |
| [第 6 章 用一个慢算子跑通 Profiling 闭环](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter5/index.md) | 同一案例贯穿 benchmark、rocprof、PyTorch Profiler、瓶颈判断 | 🚧 |
| [第 7 章 建立你的第一个性能分析报告](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter6/index.md) | 采集数据、判断瓶颈、提出假设、生成 Markdown 报告 | 🚧 |
| [第 8 章 Omniperf 与硬件计数器进阶](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part2-profiling/chapter7/index.md) | 用进阶计数器解释访存、Occupancy、波前行为和 Roofline 证据 | 🚧 |
| **第 3 篇：HIP 算子优化实战** | | |
| [第 9 章 HIP 编程基础](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter7/index.md) | Kernel、Thread、Block、Grid、Host / Device、内存管理 | 🚧 |
| [第 10 章 从 Vector Add 理解 GPU 并行](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter8/index.md) | CPU baseline、Naive HIP、线程映射、访存合并、benchmark | 🚧 |
| [第 11 章 Reduction 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter9/index.md) | Naive Reduction、LDS、Wavefront、多阶段 Reduction、性能对比 | 🚧 |
| [第 12 章 Softmax 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter10/index.md) | 数值稳定性、访存优化、Block 级并行、PyTorch 对齐 | 🚧 |
| [第 13 章 LayerNorm 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter11/index.md) | 均值方差、Reduction + Normalize 融合、向量化读写、性能分析 | 🚧 |
| [第 14 章 Matmul 入门优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part3-hip-kernels/chapter12/index.md) | Naive GEMM、Tiling、LDS 缓存、Register Blocking、rocBLAS 差距观察 | 🚧 |
| **第 4 篇：Triton on AMD 与自动调参** | | |
| [第 15 章 Triton 编程模型](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter13/index.md) | Triton vs HIP、program model、block 级张量、AMD 环境验证 | 🚧 |
| [第 16 章 Triton Matmul 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter14/index.md) | Triton GEMM、tile 设计、数据复用、benchmark、HIP / rocBLAS 对比 | 🚧 |
| [第 17 章 Triton Softmax 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter15/index.md) | 行级 Softmax、数值稳定、block reduction、访存优化、PyTorch 对齐 | 🚧 |
| [第 18 章 Triton Attention 优化](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter16/index.md) | QK^T、Softmax、PV、分块注意力、显存访问、可复现实验边界 | 🚧 |
| [第 19 章 Triton 自动调参](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part4-triton/chapter17/index.md) | 搜索空间、autotune、自动 benchmark、选择最优 kernel config | 🚧 |
| **第 5 篇：推理优化与模型部署** | | |
| [第 20 章 推理优化全景](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter16/index.md) | 延迟、吞吐、精度、batch、并发、端到端 pipeline | 🚧 |
| [第 21 章 ONNX Runtime / MIGraphX 工具实战](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter17/index.md) | ONNX 导出、ROCm 推理、MIGraphX 运行、工具层性能对比 | 🚧 |
| [第 22 章 Triton Inference Server on AMD](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter18/index.md) | Model Repository、Backend、HTTP / gRPC、动态 batching、端到端测试 | 🚧 |
| [第 23 章 YOLO 推理优化案例](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter19/index.md) | 图像预处理、NMS、batch 推理、pipeline profiling、性能报告 | 🚧 |
| [第 24 章 LLM 推理性能分析入门](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part5-inference/chapter20/index.md) | Prefill、Decode、TTFT、TPOT、KV Cache、显存观测、batch / 并发 | 🚧 |
| **第 6 篇：AI 编译器与自动调优** | | |
| [第 25 章 AI 编译器到底在优化什么](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter21/index.md) | 模型图、计算图、算子、kernel、ISA、手写优化关系 | 🚧 |
| [第 26 章 图优化原理基础](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter22/index.md) | 算子融合、常量折叠、死代码消除、布局优化、Memory Planning 原理 | 🚧 |
| [第 27 章 Kernel 生成与调度搜索](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter23/index.md) | Schedule 原语、搜索空间、Cost Model、AutoScheduler、硬件反馈 | 🚧 |
| [第 28 章 TVM / Triton / MIGraphX 对比](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part6-compiler/chapter24/index.md) | 三个工具的定位、适用问题和选择指南 | 🚧 |
| **第 7 篇：AutoInfra Agent 自动优化系统** | | |
| [第 29 章 为什么 AI Infra 需要 Agent](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter25/index.md) | 人类优化流程、可自动化环节、LLM 角色、AutoInfra 总架构 | 🚧 |
| [第 30 章 实验资产与数据结构](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter26/index.md) | 硬件画像、实验配置、benchmark 结果、profiling 输出、报告数据模型 | 🚧 |
| [第 31 章 自动 Benchmark 与 Profiling 数据管线](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter27/index.md) | 自动运行 benchmark、调用 profiling、保存结果、对比版本 | 🚧 |
| [第 32 章 瓶颈判断与优化计划](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter28/index.md) | Memory-bound / Compute-bound 判断、优化候选、优先级排序、实验计划 | 🚧 |
| [第 33 章 代码修改、运行与回滚](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter29/index.md) | 生成候选代码、自动插入 benchmark、检查编译错误、失败回滚 | 🚧 |
| [第 34 章 报告生成与实验追踪](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter30/index.md) | before / after 表格、失败尝试、证据链、下一步优化方向 | 🚧 |
| [第 35 章 最小 AutoInfra Agent 系统](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter31/index.md) | 把硬件采集、benchmark、profiling、计划、执行和报告串成闭环 | 🚧 |
| [第 36 章 毕业项目：AutoInfra Agent 完整案例](https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/part7-agent/chapter32/index.md) | 完整系统、Triton Softmax、YOLO pipeline、AMD GPU 性能诊断报告 | 🚧 |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :---- | :---- | :---- |
| 刘伟鸿 | 项目负责人 | DataWhale成员 |

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
