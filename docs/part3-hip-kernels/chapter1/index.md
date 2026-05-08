---
title: "第9章 HIP 编程基础"
description: "Hello AI Infra 第9章 · Kernel、Thread、Block、Grid、Host / Device、内存管理"
---

# 第9章 HIP 编程基础

## 本章导读

> 本章建立 HIP 编程的最小语法和执行模型，为后续手写算子做准备。读完后，你应该能看懂一个 HIP kernel 如何从 Host 端启动并在 Device 上执行。

## 9.1 HIP 和 CUDA 的关系

用迁移视角理解 HIP 的定位，但不把本教程写成 CUDA API 对照表。

## 9.2 Kernel、Thread、Block、Grid

理解 GPU 并行程序的基本层级。

## 9.3 Host 与 Device

区分 CPU 侧控制逻辑和 GPU 侧执行逻辑。

## 9.4 Device Memory 管理

介绍分配、拷贝和释放 device memory 的基本流程。

## 9.5 Kernel Launch

理解启动参数如何影响并行度和数据映射。

## 9.6 错误检查与调试

建立最小错误检查习惯，避免失败时只看到空输出。

## 9.7 思考题

通过小问题确认你是否理解基本执行模型。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-hip-kernels/chapter1/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
