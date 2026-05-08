---
title: "第13章 LayerNorm 优化"
description: "Hello AI Infra 第13章 · 均值方差、Reduction + Normalize 融合、向量化读写、性能分析"
---

# 第13章 LayerNorm 优化

## 本章导读

> 本章用 LayerNorm 继续练习 reduction，并引入融合与向量化读写的思路。读完后，你应该能解释为什么把多步归一化合在一个 kernel 里通常更高效。

## 13.1 LayerNorm 原理

回顾均值、方差、缩放和平移的计算流程。

## 13.2 均值与方差计算

把两个 reduction 放到 GPU 执行模型里分析。

## 13.3 Reduction + Normalize 融合

减少 kernel launch 和中间数据写回。

## 13.4 向量化读写

观察数据对齐和一次处理多个元素对性能的影响。

## 13.5 性能分析

用 profiling 判断瓶颈是否仍然来自访存。

## 13.6 思考题

比较不同 hidden size 和 batch size 下的实现选择。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-hip-kernels/chapter5/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
