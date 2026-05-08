---
title: "第16章 Triton Matmul 优化"
description: "Hello AI Infra 第16章 · Triton GEMM、tile 设计、数据复用、benchmark、HIP / rocBLAS 对比"
---

# 第16章 Triton Matmul 优化

## 本章导读

> 本章用 Matmul 作为第一个 Triton 主案例，理解 block 级矩阵乘如何表达 tiling 和数据复用。读完后，你应该能写出一个教学版 Triton GEMM，并用 benchmark 观察它和 HIP / rocBLAS 的差距。

## 16.1 Matmul 的计算形状

明确 M、N、K 维度、输入布局和输出 tile 的关系。

## 16.2 PyTorch 与 rocBLAS baseline

建立可对比的库函数基线。

## 16.3 Naive Triton Matmul

用 Triton 写出第一个可运行矩阵乘 kernel。

## 16.4 Tile 大小如何影响性能

观察 BLOCK_M、BLOCK_N、BLOCK_K 对并行度和复用的影响。

## 16.5 数据加载、mask 与边界处理

处理非整除形状和越界访问。

## 16.6 Benchmark 与 profiling

用相同实验流程比较不同 config。

## 16.7 与 HIP 实现对比

从代码复杂度和性能证据两方面比较 Triton 与 HIP。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part4-triton/chapter2/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
