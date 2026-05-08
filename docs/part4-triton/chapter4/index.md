---
title: "第18章 Triton Attention 优化"
description: "Hello AI Infra 第18章 · QK^T、Softmax、PV、分块注意力、显存访问、可复现实验边界"
---

# 第18章 Triton Attention 优化

## 本章导读

> 本章用 Attention 作为第三个 Triton 主案例，把 Matmul 和 Softmax 串成一个更接近真实模型的算子。读完后，你应该能理解注意力优化的主要难点来自分块、数值稳定和显存访问，而不是简单把三个算子拼起来。

## 18.1 Attention 计算流程

拆解 QK^T、Softmax 和乘 V 三个阶段。

## 18.2 PyTorch baseline 与输入约束

先固定 batch、head、sequence 和 hidden size，保证实验可复现。

## 18.3 Naive Triton Attention

写出便于理解的基础实现，不一开始追求 FlashAttention 级复杂度。

## 18.4 分块计算与显存压力

理解为什么不能直接物化所有中间矩阵。

## 18.5 数值稳定与 mask

处理 causal mask、padding mask 和 softmax 稳定性。

## 18.6 Benchmark 与 profiling

观察序列长度、head dim 和 batch 对性能的影响。

## 18.7 优化边界与后续方向

说明哪些优化需要更多实测证据，避免未验证承诺。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part4-triton/chapter16/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
