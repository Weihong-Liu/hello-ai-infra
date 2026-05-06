---
title: "第3章 AMD GPU 与 ROCm 软件栈"
description: "Hello AI Infra 第3章 · AMD GPU 执行模型、ROCm 层次、上层框架关系"
---

# 第3章 AMD GPU 与 ROCm 软件栈

## 本章导读

> 本章建立后续优化会反复用到的 AMD GPU 最小心智模型，同时解释 ROCm 软件栈如何把上层框架连接到硬件。读完后，你应该能看懂后续章节里 CU、Wavefront、LDS、HIP、MIGraphX 等词的位置。

## 3.1 AMD GPU 基本架构

介绍 CU、Wavefront 和 SIMD 执行模型，建立硬件执行的第一层直觉。

## 3.2 CU、Wavefront、SIMD、LDS、VGPR、SGPR

解释关键硬件单元，并用后续算子优化会遇到的问题来理解它们。

## 3.3 HBM、Cache 与访存层次

理解显存带宽、缓存层次和数据复用为什么影响 AI 算子性能。

## 3.4 ROCm 是什么

用分层图理解驱动、运行时、编译器、库和工具链之间的关系。

## 3.5 HIP、HSA、AMDGPU Driver 的关系

厘清 HIP 程序如何经由运行时和驱动落到设备执行。

## 3.6 PyTorch / Triton / MIGraphX / vLLM 与 ROCm 的关系

说明上层框架如何依赖 ROCm 能力，但不在本章展开使用细节。

## 3.7 如何检查一台机器的 AMD GPU 环境

把环境检查命令放回软件栈语境中，解释每个命令在检查什么。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part1-hardware-rocm/chapter2/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
