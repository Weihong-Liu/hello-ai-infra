---
layout: home
hero:
  name: "Hello AI Infra"
  text: "从硬件到智能体的 AI 基础设施实践教程"
  tagline: 面向 AMD GPU / ROCm，基于 AI MAX 395 + ROCm 7.12.0 实测，系统学习算子优化、推理优化、AI 编译器与自动优化 Agent
  actions:
    - theme: brand
      text: 开始学习
      link: /part0-preface/chapter0/

features:
  - title: 🔧 AMD First
    details: 围绕 AMD GPU / ROCm 生态设计，覆盖 HIP、rocprof、Omniperf、MIGraphX、Triton on AMD
  - title: 📊 Profiling-Driven
    details: 每个优化都有 profiling 数据支撑，教会你用证据做决策
  - title: 🧪 实测基线
    details: Alpha 阶段所有实验默认只在 AI MAX 395 + ROCm 7.12.0 上验证，其他设备先参考方法论
  - title: 🤖 Agent-Driven
    details: 构建 AutoInfra Agent，让 AI 自动完成性能分析、代码修改、benchmark 和报告生成
  - title: 🚀 Project-Based
    details: 围绕真实项目推进：从单算子优化到推理服务优化到 AutoInfra Agent
  - title: 📖 前言 + 7 篇正文，共 37 章
    details: 从环境验证到 AI Infra 全景图，再到 HIP、Triton、推理优化、AI 编译器和 AutoInfra Agent
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
