---
layout: home
hero:
  name: "Hello GPU"
  text: "GPU 算子优化入门 + Agent 自动化"
  tagline: 拿到一张 AMD 显卡，学会几个经典算子从慢到快的优化思路，最后亲手搭一个能自动做这件事的 Agent。基于 AMD Radeon RX 9070 XT + ROCm 6.4.x（Linux）实测。
  actions:
    - theme: brand
      text: 开始学习
      link: /part0-intro/chapter0/

features:
  - title: 🔧 AMD First
    details: 围绕 AMD Radeon RX 9070 XT（RDNA4）/ ROCm 生态设计，覆盖 HIP、rocprof、Omniperf、Triton on AMD
  - title: 📊 Profiling-Driven
    details: 先学会「看数据」再学「改代码」——每个优化都有 profiling 数据支撑
  - title: 🧩 刷题导向
    details: 每个算子都是从 naive 到优化的完整案例，配套刷题方法论，学完能上手 GPU 算子题库
  - title: 🤖 Agent-Driven
    details: 把「人做的优化流程」封装给 Agent 自动做：从单 kernel 到真实模型（YOLO/LLM）
  - title: 🧪 实测基线
    details: 所有实验默认在 Radeon RX 9070 XT + ROCm 6.4.x（Linux）上验证，其他设备先参考方法论
  - title: 📖 入门 + 4 篇正文，共 18 章
    details: 入门与硬件速通 → Profiling 实战 → 算子优化 + 刷题 → Agent（算子层）→ 真实模型 + Agent
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
