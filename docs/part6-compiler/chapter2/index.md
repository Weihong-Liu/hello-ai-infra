---
title: "第26章 图优化原理基础"
description: "Hello AI Infra 第26章 · 算子融合、常量折叠、死代码消除、布局优化、Memory Planning 原理"
---

# 第26章 图优化原理基础

## 本章导读

> 本章只讲图优化背后的原理，不重复第 17 章的工具命令。读完后，你应该能解释为什么工具打开某些优化选项后，模型可能更快或更省显存。

## 26.1 Operator Fusion

理解算子融合如何减少中间写回、kernel launch 和内存带宽压力。

## 26.2 Constant Folding

说明常量计算为什么可以提前完成。

## 26.3 Dead Code Elimination

理解无用节点如何被删除。

## 26.4 Common Subexpression Elimination

观察重复计算如何被复用。

## 26.5 Layout Transform

说明数据布局为什么影响后端 kernel 的访问效率。

## 26.6 Memory Planning

理解内存复用和生命周期分析如何降低峰值显存。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part6-compiler/chapter2/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
