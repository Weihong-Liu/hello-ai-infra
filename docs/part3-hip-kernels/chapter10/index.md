---
title: "第10章 Softmax 优化"
description: "Hello AI Infra 第10章"
---

# 第10章 Softmax 优化

## 本章导读

本章优化 Transformer 中最关键的 Softmax 算子。

## 10.1 Softmax 在 Transformer 中的位置

理解 Softmax 在 Self-Attention 中的核心角色。

## 10.2 Naive Softmax

实现最直接的 GPU Softmax。

## 10.3 数值稳定性

学习 log-sum-exp 技巧。

## 10.4 访存优化

减少全局显存读写次数。

## 10.5 Block 级并行

利用共享内存加速 Softmax 计算。

## 10.6 与 PyTorch 结果对齐

验证优化后输出与 PyTorch 一致。

## 思考题

1. 本章的优化中，哪个步骤带来的性能提升最大？为什么？
2. 如果换一个更大的数据规模，优化策略需要调整吗？
3. 本章中哪些步骤可以被 Agent 自动化？
