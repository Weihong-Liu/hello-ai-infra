---
title: "第6章 用一个慢算子跑通 Profiling 闭环"
description: "Hello AI Infra 第6章 · 同一案例贯穿 benchmark、rocprof、PyTorch Profiler、瓶颈判断"
---

# 第6章 用一个慢算子跑通 Profiling 闭环

## 本章导读

> 本章用一个固定的慢算子做主线，不把 profiling 工具当清单介绍，而是让每个工具服务同一个问题：它到底慢在哪里。读完后，你应该能完成一次从 benchmark 到优化假设的最小闭环。

## 6.1 选择一个可控的慢算子

确定输入规模、baseline 实现和预期瓶颈，避免一开始就分析复杂模型。

## 6.2 运行 baseline benchmark

用统一脚本记录延迟、吞吐和硬件上下文。

## 6.3 用 rocprof 看 kernel 时间

采集 kernel 级耗时，找出主要开销来自哪里。

## 6.4 用 PyTorch Profiler 关联框架调用

把 Python / framework 层的调用和 GPU kernel 时间线对应起来。

## 6.5 分析 H2D / D2H / Kernel Launch 开销

判断问题来自数据搬运、启动开销还是 kernel 本身。

## 6.6 从 profiling 结果反推优化方向

把观测结果转成下一步实验假设，而不是直接拍脑袋改代码。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part2-profiling/chapter2/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
