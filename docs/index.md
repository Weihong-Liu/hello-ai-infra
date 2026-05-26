---
layout: home
hero:
  name: "Hello GPU"
  text: "从硬件到单卡推理的 AMD GPU 实战教程"
  tagline: 三本书架构第一本（hello-gpu 层）。聚焦单卡：硬件 → profiling → HIP/Triton kernel → 编译器 → 单卡推理。多卡分布式见 hello-mlsys，K8s 平台层见 hello-ai-infra-platform。基于 AI MAX 395 + ROCm 7.12.0 实测。
  actions:
    - theme: brand
      text: 开始学习
      link: /part0-preface/chapter0/
    - theme: alt
      text: 单卡毕业项目
      link: /part5-inference/chapter26/

features:
  - title: 🔧 AMD First
    details: 围绕 AMD GPU / ROCm 生态设计，覆盖 HIP、rocprof、Omniperf、MIGraphX、Triton on AMD
  - title: 📊 Profiling-Driven
    details: 每个优化都有 profiling 数据支撑，教会你用证据做决策
  - title: 🧪 实测基线
    details: Alpha 阶段所有实验默认只在 AI MAX 395 + ROCm 7.12.0 上验证，其他设备先参考方法论
  - title: 🎯 单卡聚焦
    details: 不掺多卡通信、不掺 K8s 编排，把单卡硬件视角讲透；后续两本书各自处理多卡与平台
  - title: 🚀 Project-Based
    details: 围绕真实项目推进：单算子 baseline → kernel 优化 → 编译器图优化 → 单卡推理诊断报告
  - title: 📖 前言 + 6 篇正文，共 31 章
    details: 从环境验证到 GPU 体系结构（含 RDNA/CDNA、MFMA/WMMA、HBM/Cache 内存层次）、profiling、HIP/Triton kernel、AI 编译器与单卡推理毕业项目
---

<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
  {
    avatar: 'https://avatars.githubusercontent.com/u/65588374?v=4',
    name: '刘伟鸿',
    title: '项目负责人 · DataWhale成员',
    links: [
      { icon: 'github', link: 'https://github.com/Weihong-Liu' },
    ]
  },
]
</script>

<h2 align="center">Team</h2>
<VPTeamMembers size="small" :members />
