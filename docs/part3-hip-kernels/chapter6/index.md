---
title: "第14章 Matmul 入门优化"
description: "Hello AI Infra 第14章 · Naive GEMM、Tiling、LDS 缓存、Register Blocking、rocBLAS 差距观察"
---

# 第14章 Matmul 入门优化

## 本章导读

> 本章用教学版 GEMM 理解矩阵乘为什么是 AI 计算的核心。目标不是追平 rocBLAS，而是通过 tiling、LDS 和寄存器复用看懂高性能 GEMM 的基本方向。

## 14.1 GEMM 为什么是核心算子

说明矩阵乘在神经网络和注意力计算中的地位。

## 14.2 Naive Matmul

写出最直接的一线程计算一个输出元素的实现。

## 14.3 Tiling

把矩阵拆块，理解数据复用的第一步。

## 14.4 LDS 缓存

用 LDS 缓存 tile，减少全局内存重复读取。

## 14.5 Register Blocking

观察每个线程计算多个输出时的寄存器复用。

## 14.6 简化版高性能 GEMM

组合前面的优化，形成一个教学版优化实现。

## 14.7 与 rocBLAS 对比

只观察差距和方向，不承诺达到库级性能。

## 14.8 思考题

分析 tile size、数据类型和矩阵形状对性能的影响。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-hip-kernels/chapter6/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
