---
title: "第4章 第一个 AMD GPU 程序"
description: "Hello AI Infra 第4章 · PyTorch ROCm、最小 HIP kernel、baseline benchmark"
---

# 第4章 第一个 AMD GPU 程序

## 本章导读

> 本章在已经验证环境可用的基础上，带你跑通第一个真正的 AMD GPU 程序。重点不是安装百科，而是建立后续实验都会复用的代码、计时和日志习惯。

## 4.1 确认当前章节环境

复用 Part 0 的环境基线，确认当前章节代码目录和运行环境已经就绪。

## 4.2 读取硬件信息

用 rocminfo 和 rocm-smi 记录实验机器的关键上下文。

## 4.3 跑通 PyTorch ROCm

运行最小 tensor 运算，确认 PyTorch 能把计算放到 AMD GPU 上。

## 4.4 跑通第一个 HIP kernel

编写、编译并运行一个最小 HIP kernel。

## 4.5 建立 baseline benchmark

用固定输入、热身、重复运行和日志文件建立可复现的计时 baseline。

## 4.6 留下实验底稿

说明代码、日志和 EXPERIMENT.md 应该如何对应。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 AI MAX 395 + ROCm 7.12.0 上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part1-hardware-rocm/chapter3/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
