---
title: "第5章 AMD GPU Profiling 工具链"
description: "Hello AI Infra 第5章"
---

# 第5章 AMD GPU Profiling 工具链

## 本章导读

本章介绍 AMD GPU 的 profiling 工具链，学会用数据说话。

## 5.1 rocprof 入门

使用 rocprof 采集 kernel 执行时间、访存量等基础性能数据。

## 5.2 rocprofv2 入门

介绍 rocprofv2 的新功能和改进。

## 5.3 Omniperf 入门

使用 Omniperf 获取更详细的硬件计数器数据。

## 5.4 PyTorch Profiler 与 ROCm 联合分析

结合 PyTorch Profiler 和 ROCm 工具定位性能热点。

## 5.5 Kernel timeline 怎么看

学会阅读 kernel 执行时间线。

## 5.6 H2D / D2H / Kernel Launch 开销分析

分析 Host-Device 数据传输和 kernel launch 的开销占比。

## 5.7 如何从 profiling 结果反推优化方向

建立从 profiling 数据到优化决策的推理框架。
