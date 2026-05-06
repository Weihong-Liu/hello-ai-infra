---
title: "第24章 LLM 推理性能分析入门"
description: "Hello AI Infra 第24章 · Prefill、Decode、TTFT、TPOT、KV Cache、显存观测、batch / 并发"
---

# 第24章 LLM 推理性能分析入门

## 本章导读

> 本章降低对完整 vLLM 优化的承诺，先建立 LLM 推理性能分析的基本框架。读完后，你应该能观察 TTFT、TPOT、KV Cache 和显存占用，并理解后续是否引入 vLLM 需要先实测确认。

## 24.1 LLM 推理流程

拆解 tokenizer、prefill、decode、采样和输出后处理。

## 24.2 Prefill 与 Decode

理解两个阶段的计算形态和性能指标差异。

## 24.3 TTFT、TPOT 与吞吐

定义 LLM 推理中更常用的延迟和吞吐指标。

## 24.4 KV Cache 与显存占用

观察上下文长度、batch 和 cache 对显存的影响。

## 24.5 小模型推理 benchmark

选择能在当前硬件上稳定运行的模型做可复现实验。

## 24.6 Batch 与并发

分析批处理和并发请求对延迟、吞吐和显存的影响。

## 24.7 vLLM on AMD 的验证边界

把 vLLM 作为待实测或可选路线，不在未验证前承诺优化结果。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part5-inference/chapter20/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
