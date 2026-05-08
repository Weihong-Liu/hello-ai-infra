---
title: "第21章 ONNX Runtime / MIGraphX 工具实战"
description: "Hello AI Infra 第21章 · ONNX 导出、ROCm 推理、MIGraphX 运行、工具层性能对比"
---

# 第21章 ONNX Runtime / MIGraphX 工具实战

## 本章导读

> 本章只站在工具使用者角度，带你把模型导出、加载、运行并做性能对比。图优化为什么有效会留到编译器篇讲，这里重点是怎么正确使用和记录结果。

## 21.1 ONNX 模型导出

从 PyTorch 模型导出 ONNX，并检查输入输出签名。

## 21.2 ONNX Runtime on ROCm

在 ROCm 环境下运行 ONNX Runtime，并记录基线性能。

## 21.3 MIGraphX 基础运行

使用 MIGraphX 加载和执行模型，观察工具链差异。

## 21.4 开启工具层优化选项

只介绍如何使用优化选项，不在本章展开优化原理。

## 21.5 模型推理性能对比

用相同输入和指标比较 PyTorch、ONNX Runtime、MIGraphX。

## 21.6 结果记录与风险说明

标注硬件、版本、模型和已验证边界。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part5-inference/chapter2/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
