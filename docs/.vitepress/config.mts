import { defineConfig } from 'vitepress'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/hello-ai-infra/'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hello AI Infra',
  description: '从硬件到智能体的 AI 基础设施实践教程',
  base: baseConfig,

  cleanUrls: true,

  themeConfig: {
    nav: [
      { text: '前言', link: '/part0-preface/chapter0/' },
      { text: '硬件基础', link: '/part1-hardware-rocm/chapter1/' },
      { text: 'Profiling', link: '/part2-profiling/chapter4/' },
      { text: 'HIP 算子', link: '/part3-hip-kernels/chapter7/' },
      { text: 'Triton', link: '/part4-triton/chapter13/' },
      { text: '推理优化', link: '/part5-inference/chapter16/' },
      { text: '编译器', link: '/part6-compiler/chapter21/' },
      { text: 'Agent', link: '/part7-agent/chapter25/' },
    ],

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档',
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
            },
          },
        },
      },
    },

    sidebar: {
      '/part0-preface/': [
        {
          text: '前言与学习路线',
          items: [
            { text: '第 0 章 写给读者的话', link: '/part0-preface/chapter0/' },
          ],
        },
      ],
      '/part1-hardware-rocm/': [
        {
          text: 'AI Infra 全景与 AMD GPU 基础',
          items: [
            { text: '第 1 章 AI Infra 全景图', link: '/part1-hardware-rocm/chapter1/' },
            { text: '第 2 章 AMD GPU 与 ROCm 软件栈', link: '/part1-hardware-rocm/chapter2/' },
            { text: '第 3 章 第一个 AMD GPU 程序', link: '/part1-hardware-rocm/chapter3/' },
          ],
        },
      ],
      '/part2-profiling/': [
        {
          text: '性能分析与瓶颈定位',
          items: [
            { text: '第 4 章 性能优化的基本方法论', link: '/part2-profiling/chapter4/' },
            { text: '第 5 章 AMD GPU Profiling 工具链', link: '/part2-profiling/chapter5/' },
            { text: '第 6 章 建立你的第一个性能分析报告', link: '/part2-profiling/chapter6/' },
          ],
        },
      ],
      '/part3-hip-kernels/': [
        {
          text: 'HIP 算子优化实战',
          items: [
            { text: '第 7 章 HIP 编程基础', link: '/part3-hip-kernels/chapter7/' },
            { text: '第 8 章 从 Vector Add 理解 GPU 并行', link: '/part3-hip-kernels/chapter8/' },
            { text: '第 9 章 Reduction 优化', link: '/part3-hip-kernels/chapter9/' },
            { text: '第 10 章 Softmax 优化', link: '/part3-hip-kernels/chapter10/' },
            { text: '第 11 章 LayerNorm 优化', link: '/part3-hip-kernels/chapter11/' },
            { text: '第 12 章 Matmul 入门优化', link: '/part3-hip-kernels/chapter12/' },
          ],
        },
      ],
      '/part4-triton/': [
        {
          text: 'Triton on AMD 与自动调参',
          items: [
            { text: '第 13 章 Triton 编程模型', link: '/part4-triton/chapter13/' },
            { text: '第 14 章 Triton 实现常见算子', link: '/part4-triton/chapter14/' },
            { text: '第 15 章 Triton 自动调参', link: '/part4-triton/chapter15/' },
          ],
        },
      ],
      '/part5-inference/': [
        {
          text: '推理优化与模型部署',
          items: [
            { text: '第 16 章 推理优化全景', link: '/part5-inference/chapter16/' },
            { text: '第 17 章 ONNX Runtime / MIGraphX', link: '/part5-inference/chapter17/' },
            { text: '第 18 章 Triton Inference Server', link: '/part5-inference/chapter18/' },
            { text: '第 19 章 YOLO 推理优化案例', link: '/part5-inference/chapter19/' },
            { text: '第 20 章 LLM 推理优化案例', link: '/part5-inference/chapter20/' },
          ],
        },
      ],
      '/part6-compiler/': [
        {
          text: 'AI 编译器与自动调优',
          items: [
            { text: '第 21 章 AI 编译器到底在优化什么', link: '/part6-compiler/chapter21/' },
            { text: '第 22 章 图优化基础', link: '/part6-compiler/chapter22/' },
            { text: '第 23 章 Kernel 生成与调度搜索', link: '/part6-compiler/chapter23/' },
            { text: '第 24 章 TVM / Triton / MIGraphX 对比', link: '/part6-compiler/chapter24/' },
          ],
        },
      ],
      '/part7-agent/': [
        {
          text: 'AutoInfra Agent 自动优化系统',
          items: [
            { text: '第 25 章 为什么 AI Infra 需要 Agent', link: '/part7-agent/chapter25/' },
            { text: '第 26 章 Hardware Inspector', link: '/part7-agent/chapter26/' },
            { text: '第 27 章 Benchmark Agent', link: '/part7-agent/chapter27/' },
            { text: '第 28 章 Profiler Agent', link: '/part7-agent/chapter28/' },
            { text: '第 29 章 Optimization Planner', link: '/part7-agent/chapter29/' },
            { text: '第 30 章 Code Agent', link: '/part7-agent/chapter30/' },
            { text: '第 31 章 Report Agent', link: '/part7-agent/chapter31/' },
            { text: '第 32 章 毕业项目：AutoInfra Agent', link: '/part7-agent/chapter32/' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Weihong-Liu/hello-ai-infra' },
    ],

    editLink: {
      pattern: 'https://github.com/Weihong-Liu/hello-ai-infra/blob/main/docs/:path',
      text: '在 GitHub 上编辑此页',
    },

    outline: {
      level: [1, 3],
      label: '本章目录',
    },

    footer: {
      message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可',
    },
  },
})
