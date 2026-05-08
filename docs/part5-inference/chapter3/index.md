---
title: "第22章 Triton Inference Server on AMD"
description: "Hello AI Infra 第22章 · Model Repository、Backend、HTTP / gRPC、动态 batching、端到端测试"
---

# 第22章 Triton Inference Server on AMD

## 本章导读

> 本章把单机推理程序放进服务化框架中，观察服务层如何影响端到端性能。读完后，你应该能搭建最小 Model Repository，并用统一请求方式测试服务性能。

## 22.1 Triton Server 是什么

理解推理服务和本地脚本的差异。

## 22.2 Model Repository 结构

组织模型文件、配置和版本目录。

## 22.3 Python Backend / ONNX Backend / 自定义 Backend

比较不同 backend 的定位和适用场景。

## 22.4 gRPC / HTTP 调用

用客户端发起请求并记录响应延迟。

## 22.5 动态 batching

观察动态 batch 对吞吐和延迟的影响。

## 22.6 端到端性能测试

把模型计算和服务开销放在同一份报告里。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part5-inference/chapter3/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
