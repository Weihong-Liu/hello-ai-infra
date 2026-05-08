---
title: "第15章 Triton 编程模型"
description: "Hello AI Infra 第15章 · Triton vs HIP、program model、block 级张量、AMD 环境验证"
---

# 第15章 Triton 编程模型

## 本章导读

> 本章从 HIP 切换到 Triton，重点理解 Triton 为什么让算子开发更接近张量块级编程。读完后，你应该能解释 program、block 和 mask 如何对应到数据。

## 15.1 为什么需要 Triton

说明在手写 HIP 和调用库之间，Triton 提供了什么折中。

## 15.2 Triton 和 HIP 的区别

对比线程级编程和 block 级张量编程的思维差异。

## 15.3 Triton on AMD 环境配置

验证当前 AMD ROCm 环境下 Triton 是否可用。

## 15.4 第一个 Triton kernel

用最小示例理解 program id、block 指针和 mask。

## 15.5 Triton 的 block / program model

把 Triton 抽象映射回 GPU 执行模型。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part4-triton/chapter13/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
