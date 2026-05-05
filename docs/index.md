---
layout: home
hero:
  name: "Hello AI Infra"
  text: "从硬件到智能体的 AI 基础设施实践教程"
  tagline: 面向 AMD GPU / ROCm，系统学习算子优化、推理优化、AI 编译器与自动优化 Agent
  actions:
    - theme: brand
      text: 开始学习
      link: /part0-preface/chapter0/

features:
  - title: 🔧 AMD First
    details: 围绕 AMD GPU / ROCm 生态设计，覆盖 HIP、rocprof、Omniperf、MIGraphX、Triton on AMD
  - title: 📊 Profiling-Driven
    details: 每个优化都有 profiling 数据支撑，教会你用证据做决策
  - title: 🖥️ Hardware-Aware
    details: 先理解硬件能力，再理解程序如何映射到硬件，最后才做优化
  - title: 🤖 Agent-Driven
    details: 构建 AutoInfra Agent，让 AI 自动完成性能分析、代码修改、benchmark 和报告生成
  - title: 🚀 Project-Based
    details: 围绕真实项目推进：从单算子优化到推理服务优化到 AutoInfra Agent
  - title: 📖 7 篇 33 章
    details: 从 AI Infra 全景图到 HIP 算子到 Triton 到推理优化到 AI 编译器到 Agent
---

<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
  {
    avatar: 'https://www.github.com/Weihong-Liu.png',
    name: 'Weihong Liu',
    title: '项目负责人',
    links: [
      { icon: 'github', link: 'https://github.com/Weihong-Liu' },
    ]
  },
]
</script>

<h2 align="center">Team</h2>
<VPTeamMembers size="small" :members />
