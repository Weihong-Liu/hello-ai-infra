---
title: "第11章 怎么刷 LeetGPU"
description: "Hello GPU 第11章 · 平台题型/评分、本地评测器、调试策略、性能闭环（硬件无关方法论）"
---

# 第11章 怎么刷 LeetGPU

## 本章导读

> 本章把前面四个算子积累的经验系统化成「刷题方法论」。重要前提：LeetGPU 目前仅支持 CUDA/Triton/PyTorch，不在 AMD 上提交；但刷题的方法论是硬件无关的——怎么读题、怎么搭本地评测、怎么用 profiling 驱动迭代，这些在哪个平台都通用。

## 11.1 GPU 算子题库长什么样

介绍 LeetGPU / Tensara / GPU MODE 等平台的题目结构和评分机制（读题层，不依赖提交）。

## 11.2 题型套路分类

把题目分成 elementwise / reduction / GEMM-like / 融合型，对应本书 Ch4 / Ch7 / Ch9 / Ch10。

## 11.3 本地评测器怎么搭

复用 gpu-queue 思路，搭一个喂输入、跑 kernel、计时、对答案的本地评测器（跑在 9070XT 上）。

## 11.4 刷题策略：正确性 → 带宽 → 计算强度

说明每一步该用什么 profiling 工具验证，避免一上来就盲目优化。

## 11.5 调试常见问题

列出边界条件、数值误差、bank 冲突、occupancy 不足等常见坑。

## 11.6 关于 AMD 平台的现状

诚实说明 LeetGPU 当前是 CUDA-only；等 AMD 等价平台出现，本章方法论迁移成本很低。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 涉及命令、输出或性能数字的内容，后续必须在 Radeon RX 9070 XT + ROCm 6.4.x（Linux）上实测。
- 与本章相关的代码、日志和实验底稿会放在 `code/part2-kernels/chapter11/`。

## 延伸阅读

- 待补：正式正文完成时补充对应官方文档、论文或工具链接。
