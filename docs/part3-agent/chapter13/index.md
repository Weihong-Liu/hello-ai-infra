---
title: "第13章 工具封装"
description: "Hello GPU 第13章 · benchmark/profiling/编译包成 Agent 可调用工具"
---

# 第13章 工具封装

## 本章导读

> 本章把 Part 1 学过的 benchmark、rocprof、Omniperf 以及编译流程，封装成 Agent 能调用的标准化工具。这是让 Agent「能动手」的前提——没有工具的 Agent 只会空谈。

## 13.1 为什么要封装工具

说明 Agent 不能直接操作 shell，需要结构化、可解析的工具接口。

## 13.2 封装 benchmark 工具

把 Part 1 的计时脚本包成输入 kernel → 输出延迟/带宽的标准化工具。

## 13.3 封装 profiling 工具

把 rocprof/Omniperf 包成输入 kernel → 输出瓶颈信号的标准化工具。

## 13.4 封装编译工具

把 hipcc/triton 编译流程包成输入代码 → 输出编译成功/失败的标准化工具。

## 13.5 工具的输入输出 schema

用 JSON schema 定义每个工具的接口，让 Agent 能正确调用。

## 13.6 错误处理与重试

说明工具失败时如何把错误信息回传给 Agent 触发反思。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 Radeon RX 9070 XT + ROCm 6.4.x（Linux）上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-agent/chapter13/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
