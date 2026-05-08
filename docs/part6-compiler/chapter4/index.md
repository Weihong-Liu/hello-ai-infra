---
title: "第28章 TVM / Triton / MIGraphX 对比"
description: "Hello AI Infra 第28章 · 三个工具的定位、适用问题和选择指南"
---

# 第28章 TVM / Triton / MIGraphX 对比

## 本章导读

> 本章用前面学过的工具和原理做一次定位对比。读完后，你应该能根据问题类型判断是该写 Triton kernel、用 MIGraphX 跑推理，还是研究 TVM 类编译器。

## 28.1 TVM 的定位

理解 TVM 更偏通用编译器和 schedule 搜索平台。

## 28.2 Triton 的定位

理解 Triton 更适合手写和自动调参 GPU kernel。

## 28.3 MIGraphX 的定位

理解 MIGraphX 在 AMD 推理优化链路中的角色。

## 28.4 三者适合解决的问题

用问题类型对比各工具的使用边界。

## 28.5 如何选择工具

给出基于目标、成本和可复现性的选择建议。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part6-compiler/chapter24/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
