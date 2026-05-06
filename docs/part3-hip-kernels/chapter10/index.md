---
title: "第12章 Softmax 优化"
description: "Hello AI Infra 第12章 · 数值稳定性、访存优化、Block 级并行、PyTorch 对齐"
---

# 第12章 Softmax 优化

## 本章导读

> 本章用 Softmax 把 reduction、数值稳定性和访存优化串起来。读完后，你应该能写出一个结果正确、能被 benchmark 和 profiling 验证的教学版 Softmax。

## 12.1 Softmax 在 Transformer 中的位置

说明为什么 Softmax 是理解注意力性能的重要入口。

## 12.2 Naive Softmax

从直接实现开始，观察重复访存和数值问题。

## 12.3 数值稳定性

使用减去最大值的形式避免指数溢出。

## 12.4 访存优化

减少多次读取和写回，理解中间结果如何组织。

## 12.5 Block 级并行

用 block 内协作处理一行或一段数据。

## 12.6 与 PyTorch 结果对齐

确认数值误差、输入范围和边界条件。

## 12.7 思考题

分析不同 hidden size 下实现策略的变化。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-hip-kernels/chapter10/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
