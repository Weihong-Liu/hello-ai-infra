---
title: "第9章 Reduction 优化"
description: "Hello AI Infra 第9章"
---

# 第9章 Reduction 优化

## 本章导读

本章通过 Reduction 优化，教你逐步提升 GPU kernel 性能。

## 9.1 Reduction 为什么重要

Reduction 是 Sum、Mean、Max、Min 等操作的基础。

## 9.2 Naive Reduction

实现最简单的 GPU Reduction。

## 9.3 Shared Memory / LDS 优化

利用 LDS 减少全局显存访问次数。

## 9.4 Warp / Wavefront 级优化思路

利用 Wavefront 内的硬件同步机制。

## 9.5 多阶段 Reduction

实现分块 Reduction 的多阶段策略。

## 9.6 性能对比

对比各阶段的性能提升。

## 思考题

1. 本章的优化中，哪个步骤带来的性能提升最大？为什么？
2. 如果换一个更大的数据规模，优化策略需要调整吗？
3. 本章中哪些步骤可以被 Agent 自动化？
