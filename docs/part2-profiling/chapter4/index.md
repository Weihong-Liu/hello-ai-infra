---
title: "第4章 性能优化的基本方法论"
description: "Hello AI Infra 第4章"
---

# 第4章 性能优化的基本方法论

## 本章导读

本章是整个教程的方法论核心，教你建立以 profiling 为证据的优化思维。

## 4.1 为什么不能凭感觉优化

通过反面案例说明没有 profiling 的优化往往是浪费时间。

## 4.2 Latency、Throughput、Bandwidth、FLOPS

定义和区分这四个核心性能指标。

## 4.3 Memory-bound 与 Compute-bound

理解两类瓶颈的本质区别。

## 4.4 Roofline 思想入门

用 Roofline Model 可视化算子的性能上限。

## 4.5 如何设计一个可信的 benchmark

学习预热、多次运行、排除异常值等 benchmark 设计原则。

## 4.6 如何避免伪优化

识别常见的伪优化陷阱。
