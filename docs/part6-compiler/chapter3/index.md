---
title: "第27章 Kernel 生成与调度搜索"
description: "Hello AI Infra 第27章 · Schedule 原语、搜索空间、Cost Model、AutoScheduler、硬件反馈"
---

# 第27章 Kernel 生成与调度搜索

## 本章导读

> 本章从图优化进入 kernel 生成和调度搜索，解释自动调优为什么需要硬件反馈。读完后，你应该能把 Triton autotune 和编译器 schedule 搜索放到同一张图里理解。

## 27.1 Schedule 是什么

理解同一个计算可以有不同执行计划。

## 27.2 Tile / Split / Reorder / Vectorize / Unroll

介绍常见 schedule 原语对执行方式的影响。

## 27.3 搜索空间设计

说明为什么可选 schedule 太多，需要约束搜索范围。

## 27.4 Cost Model

理解代价模型如何预测候选实现。

## 27.5 AutoScheduler / MetaSchedule 思想

认识自动搜索 schedule 的基本流程。

## 27.6 为什么自动调优需要硬件反馈

回到 AI MAX 395 实测，说明真实硬件数据不可替代。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part6-compiler/chapter23/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
