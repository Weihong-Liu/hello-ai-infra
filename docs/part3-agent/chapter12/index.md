---
title: "第12章 Agent 入门"
description: "Hello GPU 第12章 · 参考 hello-agents、LLM Agent 基本范式、工具调用"
---

# 第12章 Agent 入门

## 本章导读

> 本章是 Agent 篇的入口，参考 hello-agents 的概念铺垫节奏，讲清楚 LLM Agent 的基本范式。但本书的 Agent 场景是「算子/模型优化」，不是通用智能体——这是和 hello-agents 的关键区别。

## 12.1 什么是 LLM Agent

用最简模型理解 Agent = LLM + 工具 + 循环，参考 hello-agents 第 1 章。

## 12.2 为什么 Agent 适合算子优化

说明算子优化天然适合 Agent：有明确目标（性能）、有可调用工具（编译/跑分/profiling）、有可验证反馈（benchmark 数字）。

## 12.3 ReAct 范式简介

理解 Reason-Act-Observe 循环，这是后续算子优化 Agent 的基本骨架。

## 12.4 工具调用（Tool Use）

理解 Agent 如何通过结构化接口调用外部工具。

## 12.5 本书 Agent 的边界

明确本书 Agent 聚焦算子/模型优化，不做通用代码生成或对话助手。

## 12.6 和 hello-agents 的关系

说明本书 Agent 篇假设你已了解 Agent 基本概念；零基础建议先读 hello-agents。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 Radeon RX 9070 XT + ROCm 6.4.x（Linux）上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part3-agent/chapter12/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
