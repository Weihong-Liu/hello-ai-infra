---
title: "第23章 YOLO 推理优化案例"
description: "Hello AI Infra 第23章 · 图像预处理、NMS、batch 推理、pipeline profiling、性能报告"
---

# 第23章 YOLO 推理优化案例

## 本章导读

> 本章选择 YOLO 作为端到端视觉推理案例，练习从模型外部开销到 GPU kernel 的完整分析。读完后，你应该能知道一个推理 pipeline 慢，不一定是模型本身慢。

## 23.1 YOLO 模型部署

准备模型、输入图片和最小推理脚本。

## 23.2 图像预处理优化

分析 resize、normalize、layout transform 等 CPU/GPU 开销。

## 23.3 后处理 NMS 优化

观察 NMS 是否成为端到端瓶颈。

## 23.4 Batch 推理

测试 batch size 对吞吐和延迟的影响。

## 23.5 Pipeline profiling

把预处理、模型、后处理和服务开销拆开记录。

## 23.6 性能报告

输出一份包含瓶颈判断和下一步计划的案例报告。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part5-inference/chapter4/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
