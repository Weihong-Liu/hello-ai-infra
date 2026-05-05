---
title: "第15章 Triton 自动调参"
description: "Hello AI Infra 第15章"
---

# 第15章 Triton 自动调参

## 本章导读

本章学习 Triton 的自动调参能力。

## 15.1 BLOCK_SIZE 怎么影响性能

通过实验展示 BLOCK_SIZE 对性能的影响。

## 15.2 num_warps / num_stages 的意义

解释参数对并行度和流水线深度的影响。

## 15.3 搜索空间设计

学习设计合理的参数搜索空间。

## 15.4 Triton autotune

使用 @triton.autotune 自动搜索最优配置。

## 15.5 自动 benchmark

构建自动化的 benchmark 流水线。

## 15.6 自动选择最优 kernel config

封装自动选择最优配置的 wrapper。
