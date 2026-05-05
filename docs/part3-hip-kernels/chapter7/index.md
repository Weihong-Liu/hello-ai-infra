---
title: "第7章 HIP 编程基础"
description: "Hello AI Infra 第7章"
---

# 第7章 HIP 编程基础

## 本章导读

本章从零开始教你 HIP 编程，为手写优化算子打基础。

## 7.1 HIP 和 CUDA 的关系

理解 HIP 作为 CUDA 的可移植替代。

## 7.2 Kernel、Thread、Block、Grid

掌握 GPU 并行编程的核心概念模型。

## 7.3 Host 与 Device

理解 CPU 和 GPU 的分工与数据传输。

## 7.4 Device Memory 管理

学习 hipMalloc、hipMemcpy、hipFree 等显存管理 API。

## 7.5 Kernel Launch

学习如何定义和启动 HIP kernel。

## 7.6 错误检查与调试

学习 GPU 编程的调试手段。

## 思考题

1. 本章的优化中，哪个步骤带来的性能提升最大？为什么？
2. 如果换一个更大的数据规模，优化策略需要调整吗？
3. 本章中哪些步骤可以被 Agent 自动化？
