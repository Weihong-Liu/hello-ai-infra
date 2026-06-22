---
title: "第17章 小模型 LLM 解码 + Agent 自动优化"
description: "Hello GPU 第17章 · Qwen 0.5B/1.8B 量化、decode 算子视角、Agent 优化 KV cache/精度"
---

# 第17章 小模型 LLM 解码 + Agent 自动优化

## 本章导读

> 本章把 Agent 应用到 LLM 解码场景。受 9070XT 16GB 显存限制，只能量化小模型（Qwen 0.5B/1.8B）。重点是 decode 阶段的算子视角：Agent 如何优化 KV cache 访问、精度选择等。注意：PagedAttention、多卡 TP 等留给 hello-mlsys。

## 17.1 显存约束下的模型选型

说明 9070XT 16GB 显存为什么选 Qwen 0.5B/1.8B 量化，跑不了 7B+。

## 17.2 LLM 推理流程拆解

拆解 prefill 和 decode 两个阶段，理解 decode 为什么是算子密集型。

## 17.3 建立 decode baseline

测量 TTFT、TPOT、KV cache 显存占用作为 baseline。

## 17.4 decode 的算子视角

把 decode 拆成 attention（KV cache 读取）+ matmul（投影）两个核心算子。

## 17.5 Agent 优化 KV cache 访问

让 Agent 分析 KV cache 的访存模式，给出精度/布局优化建议。

## 17.6 Agent 优化精度选择

让 Agent 对比 fp16/int8/int4 在 decode 性能和显存上的权衡。

## 17.7 单卡边界与下一步

明确 PagedAttention、多卡 TP、并发调度等留给 hello-mlsys。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 Radeon RX 9070 XT + ROCm 6.4.x（Linux）上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part4-models-agent/chapter17/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
