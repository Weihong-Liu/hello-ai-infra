import { defineConfig } from 'vitepress'
import footnote from 'markdown-it-footnote'
import mathjax3 from 'markdown-it-mathjax3'
import figurePlugin from './markdown-figures.mjs'
import { navItems, sidebar } from './outline.mjs'

function encodeMermaid(value: string) {
  return encodeURIComponent(value)
}

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/hello-ai-infra/'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hello GPU',
  description: '从硬件到智能体的 AI 基础设施实践教程（单卡 AMD GPU 视角；多卡见 hello-mlsys，平台层见 hello-ai-infra-platform）',
  base: baseConfig,

  cleanUrls: true,

  srcExclude: ['part7-agent/**'],

  vue: {
    template: {
      compilerOptions: {
        // mathjax3 emits <mjx-container> / <mjx-...> custom elements; keep
        // Vue from trying to resolve them as components.
        isCustomElement: (tag: string) => tag.startsWith('mjx-'),
      },
    },
  },

  markdown: {
    config(md) {
      md.use(footnote)
      md.use(mathjax3)
      md.use(figurePlugin)

      // 同一条脚注被多次引用时，默认会渲染成 [6]、[6:1]、[6:2] …
      // 对读者没有意义，统一只显示脚注序号，让回跳锚点照常工作。
      md.renderer.rules.footnote_caption = (tokens, idx) => {
        const n = Number(tokens[idx].meta.id + 1).toString()
        return `[${n}]`
      }

      const fence = md.renderer.rules.fence!
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const language = token.info.trim().split(/\s+/)[0]

        if (language === 'mermaid') {
          return `<MermaidDiagram code="${encodeMermaid(token.content)}" />`
        }

        return fence(tokens, idx, options, env, self)
      }

      // Rewrite ![](*.png) → ![](*.webp) at render time so authors keep
      // editing PNG locally while the built site serves WebP.
      const defaultImage =
        md.renderer.rules.image ||
        ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options))
      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const srcIndex = token.attrIndex('src')
        if (srcIndex >= 0) {
          const src = token.attrs![srcIndex][1]
          if (/\.png(\?.*)?$/i.test(src)) {
            token.attrs![srcIndex][1] = src.replace(/\.png(\?.*)?$/i, (_, q) => `.webp${q ?? ''}`)
          }
        }
        if (token.attrIndex('loading') < 0) {
          token.attrSet('loading', 'lazy')
        }
        if (token.attrIndex('decoding') < 0) {
          token.attrSet('decoding', 'async')
        }
        return defaultImage(tokens, idx, options, env, self)
      }
    },
  },

  themeConfig: {
    nav: navItems,

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

    sidebar,

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
