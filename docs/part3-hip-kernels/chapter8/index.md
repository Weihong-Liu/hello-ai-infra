---
title: "第8章 从 Vector Add 理解 GPU 并行"
description: "Hello AI Infra 第8章"
---

# 第8章 从 Vector Add 理解 GPU 并行

## 本章导读

本章通过最简单的向量加法，带你理解 GPU 并行编程的核心概念。

## 8.1 CPU 版本

先实现一个简单的 CPU 向量加法作为对比基准。

## 8.2 Naive HIP 版本

写一个最简单的 HIP kernel。

## 8.3 线程映射

理解如何将数据索引映射到线程索引。

## 8.4 访存合并

学习 coalesced memory access 原理。

## 8.5 Benchmark 与 profiling

对 CPU 和 GPU 版本进行 benchmark 对比。

## 8.6 优化报告

整理实验数据，生成性能演进报告。

## 思考题

1. 本章的优化中，哪个步骤带来的性能提升最大？为什么？
2. 如果换一个更大的数据规模，优化策略需要调整吗？
3. 本章中哪些步骤可以被 Agent 自动化？
