---
title: "第12章 Matmul 入门优化"
description: "Hello AI Infra 第12章"
---

# 第12章 Matmul 入门优化

## 本章导读

本章挑战最重要的 GEMM 算子，逐步优化到接近 rocBLAS 的性能。

## 12.1 GEMM 为什么是核心算子

理解矩阵乘法在神经网络中的核心地位。

## 12.2 Naive Matmul

实现最简单的 GPU 矩阵乘法。

## 12.3 Tiling

学习分块矩阵乘法。

## 12.4 LDS 缓存

将 tile 数据缓存到 LDS。

## 12.5 Register Blocking

利用寄存器缓存进一步减少 LDS 访问。

## 12.6 简化版高性能 GEMM

综合所有技巧实现高性能 GEMM。

## 12.7 与 rocBLAS 对比

与 AMD 官方库做性能对比。

## 思考题

1. 本章的优化中，哪个步骤带来的性能提升最大？为什么？
2. 如果换一个更大的数据规模，优化策略需要调整吗？
3. 本章中哪些步骤可以被 Agent 自动化？
