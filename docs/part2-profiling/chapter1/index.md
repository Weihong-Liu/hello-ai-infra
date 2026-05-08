---
title: "第5章 性能优化的基本方法论"
description: "Hello AI Infra 第5章 · Latency、Throughput、Bandwidth、FLOPS、Roofline、可信 benchmark"
---

# 第5章 性能优化的基本方法论

## 本章导读

> 本章先不急着打开 profiling 工具，而是建立判断性能问题的基本语言。读完后，你应该能区分延迟、吞吐、带宽、FLOPS，以及为什么不能凭感觉优化。

## 5.1 为什么不能凭感觉优化

用常见误区说明没有数据的优化为什么容易走偏。

## 5.2 Latency、Throughput、Bandwidth、FLOPS

定义最常用的性能指标，并说明它们分别回答什么问题。

## 5.3 Memory-bound 与 Compute-bound

理解访存瓶颈和计算瓶颈的差异。

## 5.4 Roofline 思想入门

用简单图示理解理论上限、实际性能和优化方向之间的关系。

## 5.5 如何设计一个可信的 benchmark

说明热身、重复次数、同步、输入规模和日志记录的基本要求。

## 5.6 如何避免伪优化

识别缓存偶然命中、测量范围错误、数据拷贝遗漏等伪提升。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part2-profiling/chapter1/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
