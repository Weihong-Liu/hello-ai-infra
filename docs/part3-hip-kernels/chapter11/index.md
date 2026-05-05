---
title: "第11章 LayerNorm 优化"
description: "Hello AI Infra 第11章"
---

# 第11章 LayerNorm 优化

## 本章导读

本章优化 LayerNorm，学习 kernel 融合技巧。

## 11.1 LayerNorm 原理

理解 LayerNorm 的数学公式。

## 11.2 均值与方差计算

将均值和方差拆解为 Reduction 操作。

## 11.3 Reduction + Normalize 融合

将多个 kernel 融合为一个。

## 11.4 向量化读写

利用向量化加载提升访存效率。

## 11.5 性能分析

对比融合与分离 kernel 的性能差异。

## 思考题

1. 本章的优化中，哪个步骤带来的性能提升最大？为什么？
2. 如果换一个更大的数据规模，优化策略需要调整吗？
3. 本章中哪些步骤可以被 Agent 自动化？
