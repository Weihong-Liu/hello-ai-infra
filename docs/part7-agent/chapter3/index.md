---
title: "第31章 自动 Benchmark 与 Profiling 数据管线"
description: "Hello AI Infra 第31章 · 自动运行 benchmark、调用 profiling、保存结果、对比版本"
---

# 第31章 自动 Benchmark 与 Profiling 数据管线

## 本章导读

> 本章搭建系统闭环的第一段：让程序能自动运行实验并收集性能证据。读完后，你应该能把手工 benchmark 和 profiling 变成可重复调用的数据管线。

## 31.1 自动运行 benchmark

把固定命令封装成可重复执行的任务。

## 31.2 自动记录 latency / throughput / memory

统一指标格式和日志路径。

## 31.3 自动调用 profiling 工具

根据配置调用 rocprof、PyTorch Profiler 或 Omniperf。

## 31.4 自动保存实验结果

把原始输出和摘要结果一起落盘。

## 31.5 自动对比不同版本

比较 baseline 与候选版本的关键指标。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part7-agent/chapter3/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
