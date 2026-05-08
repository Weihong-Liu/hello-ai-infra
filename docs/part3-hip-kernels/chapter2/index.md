---
title: "第10章 从 Vector Add 理解 GPU 并行"
description: "Hello AI Infra 第10章 · CPU baseline、Naive HIP、线程映射、访存合并、benchmark"
---

# 第10章 从 Vector Add 理解 GPU 并行

## 本章导读

> 本章用最简单的 Vector Add 连接 CPU 思维和 GPU 并行思维。读完后，你应该能解释线程如何映射到数据，以及为什么看似简单的向量加法也需要严谨 benchmark。

## 10.1 CPU 版本

先写一个清晰的 CPU baseline，明确要搬到 GPU 上的计算是什么。

## 10.2 Naive HIP 版本

写出第一个一线程处理一个元素的 HIP kernel。

## 10.3 线程映射

理解 blockIdx、threadIdx 和全局元素下标的关系。

## 10.4 访存合并

观察连续线程访问连续地址为什么重要。

## 10.5 Benchmark 与 profiling

对比 CPU、GPU 和不同输入规模下的表现。

## 10.6 优化报告

把实验结果整理成一份最小优化报告。

## 10.7 思考题

通过修改输入规模和 block size 理解并行度变化。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-hip-kernels/chapter8/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
