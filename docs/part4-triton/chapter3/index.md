---
title: "第17章 Triton Softmax 优化"
description: "Hello AI Infra 第17章 · 行级 Softmax、数值稳定、block reduction、访存优化、PyTorch 对齐"
---

# 第17章 Triton Softmax 优化

## 本章导读

> 本章用 Softmax 作为第二个 Triton 主案例，重点练习 block 内 reduction、数值稳定性和访存控制。读完后，你应该能把 HIP Softmax 的思路迁移到 Triton 表达。

## 17.1 Softmax 的输入形状与 baseline

确定按行计算的输入规模、输出校验和 PyTorch baseline。

## 17.2 Naive Triton Softmax

写出一行对应一个 program 的基础实现。

## 17.3 数值稳定性

实现减最大值、指数、求和和归一化。

## 17.4 Block reduction 怎么表达

理解 Triton 张量操作如何表达一行内归约。

## 17.5 访存与中间结果优化

减少重复加载和多余写回。

## 17.6 Benchmark 与 profiling

比较不同 block size 和输入形状。

## 17.7 与 HIP Softmax 对比

从实现复杂度和性能瓶颈角度复盘差异。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part4-triton/chapter3/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
