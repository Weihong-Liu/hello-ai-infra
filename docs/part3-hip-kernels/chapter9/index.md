---
title: "第11章 Reduction 优化"
description: "Hello AI Infra 第11章 · Naive Reduction、LDS、Wavefront、多阶段 Reduction、性能对比"
---

# 第11章 Reduction 优化

## 本章导读

> 本章进入第一个真正体现 GPU 层级协作的算子：Reduction。读完后，你应该能理解为什么跨线程汇总需要 LDS、同步和多阶段设计。

## 11.1 Reduction 为什么重要

说明求和、归约、归一化和注意力中为什么经常出现 reduction。

## 11.2 Naive Reduction

从简单但低效的实现开始，观察瓶颈。

## 11.3 Shared Memory / LDS 优化

使用 LDS 减少全局内存访问并组织 block 内归约。

## 11.4 Warp / Wavefront 级优化思路

理解 wavefront 内协作和分支收敛对 reduction 的影响。

## 11.5 多阶段 Reduction

把大规模输入拆成 block 内归约和跨 block 合并。

## 11.6 性能对比

用 benchmark 和 profiling 对比不同版本。

## 11.7 思考题

分析不同输入长度和 block size 对性能的影响。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-hip-kernels/chapter9/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
