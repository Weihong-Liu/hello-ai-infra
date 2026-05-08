---
title: "第19章 Triton 自动调参"
description: "Hello AI Infra 第19章 · 搜索空间、autotune、自动 benchmark、选择最优 kernel config"
---

# 第19章 Triton 自动调参

## 本章导读

> 本章把前面 Matmul、Softmax 和 Attention 中反复出现的 config 选择系统化。读完后，你应该知道如何定义搜索空间、运行自动 benchmark，并用证据选择当前硬件上的最优配置。

## 19.1 BLOCK_SIZE 怎么影响性能

从前面算子的实验结果回顾 block 参数为什么重要。

## 19.2 num_warps / num_stages 的意义

理解并行度、流水和资源占用之间的权衡。

## 19.3 搜索空间设计

避免盲目枚举过大的配置集合。

## 19.4 Triton autotune

使用 Triton 的 autotune 机制管理候选配置。

## 19.5 自动 benchmark

记录每个候选的命令、输入、硬件和结果。

## 19.6 自动选择最优 kernel config

把性能结果转成可复用的配置选择。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part4-triton/chapter17/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
