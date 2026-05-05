---
title: "第2章 AMD GPU 与 ROCm 软件栈"
description: "Hello AI Infra 第2章"
---

# 第2章 AMD GPU 与 ROCm 软件栈

## 本章导读

本章深入 AMD GPU 硬件架构和 ROCm 软件栈，为后续优化实践奠定基础。

## 2.1 AMD GPU 基本架构

介绍 AMD GPU 的核心概念：CU、Wavefront、SIMD 执行模型。

## 2.2 CU、Wavefront、SIMD、LDS、VGPR、SGPR

逐一解释 AMD GPU 的关键硬件单元，与 NVIDIA GPU 的对应概念做类比。

## 2.3 HBM、Cache 与访存层次

理解显存带宽、缓存层次对 AI 计算性能的决定性影响。

## 2.4 ROCm 是什么

ROCm 生态全景：驱动、运行时、编译器、工具链的层次关系。

## 2.5 HIP、HSA、AMDGPU Driver 的关系

厘清 HIP、HSA、AMDGPU 驱动之间的层次。

## 2.6 PyTorch / Triton / MIGraphX / vLLM 与 ROCm 的关系

展示上层框架如何对接 ROCm 底层。

## 2.7 如何检查一台机器的 AMD GPU 环境

动手实践：用 rocminfo、rocm-smi 等工具检查 GPU 环境。
