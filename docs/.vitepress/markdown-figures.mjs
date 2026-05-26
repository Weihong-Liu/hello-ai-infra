// markdown-it 扩展：::: figure 容器 + @fig-id 自动交叉引用。
//
// 语法示例：
//
//   ::: figure fig-mem-pyramid
//   ![GPU 内存层级金字塔](./images/memory-hierarchy-pyramid.png)
//
//   GPU 内存层级金字塔：离计算单元越近越快、越小
//   :::
//
//   如 @fig-mem-pyramid 所示，访问越快、容量越小。
//
// 渲染结果（简化）：
//
//   <div class="figure" id="fig-mem-pyramid">
//     <img src="..." alt="...">
//     <p class="figure-caption">图 4.2　GPU 内存层级金字塔：…</p>
//   </div>
//   <p>如 <a class="figure-ref" href="#fig-mem-pyramid">图 4.2</a> 所示，…</p>
//
// 设计要点：
// 1. 章号从文件路径 chapter(\d+) 抽取，页内顺序递增。
// 2. 容器内最后一个段落自动变成 figcaption。
// 3. @fig-id 在所有 inline text token 中替换；未定义的 id 原样保留并打 warning。
// 4. 与现有 PNG→WebP / mermaid fence 规则正交，不会互相影响。

import container from 'markdown-it-container'

const FIGURE_NAME = 'figure'
const REF_PATTERN = /@(fig-[a-z0-9][a-z0-9-]*)/g

function extractChapterNumber(envPath) {
  if (!envPath) return null
  const m = /chapter(\d+)/i.exec(envPath)
  return m ? Number(m[1]) : null
}

function findFigureContainerRange(tokens, startIdx) {
  const openTok = tokens[startIdx]
  const expectedLevel = openTok.level
  for (let i = startIdx + 1; i < tokens.length; i++) {
    const t = tokens[i]
    if (
      t.type === `container_${FIGURE_NAME}_close` &&
      t.level === expectedLevel
    ) {
      return i
    }
  }
  return -1
}

// Find the last paragraph_open/paragraph_close pair strictly inside (openIdx, closeIdx).
function findLastInnerParagraph(tokens, openIdx, closeIdx) {
  for (let i = closeIdx - 1; i > openIdx; i--) {
    if (tokens[i].type === 'paragraph_close') {
      // Walk back to its matching paragraph_open at the same level.
      const level = tokens[i].level
      for (let j = i - 1; j > openIdx; j--) {
        if (
          tokens[j].type === 'paragraph_open' &&
          tokens[j].level === level
        ) {
          return { openIdx: j, closeIdx: i }
        }
      }
    }
  }
  return null
}

function plainTextFromInline(inlineToken) {
  if (!inlineToken || !inlineToken.children) return ''
  return inlineToken.children
    .map((c) => (c.type === 'text' ? c.content : c.type === 'code_inline' ? c.content : ''))
    .join('')
    .trim()
}

export default function figurePlugin(md) {
  // 1) Register ::: figure container. Validator accepts `figure <id>`.
  md.use(container, FIGURE_NAME, {
    validate(params) {
      return /^figure\s+([a-z0-9][a-z0-9-]*)\s*$/i.test(params.trim())
    },
    // Renderers are overridden below; provide defaults just in case.
    render(tokens, idx) {
      return tokens[idx].nesting === 1 ? '<div class="figure">\n' : '</div>\n'
    },
  })

  // 2) Numbering pass: assign env._figures = { id -> "N.M" } and stash
  //    metadata on the container open token so the renderer can use it.
  md.core.ruler.after('block', 'figure_numbering', (state) => {
    const env = state.env || {}
    const chapter = extractChapterNumber(env.path || env.relativePath || '')
    const figures = {}
    let seq = 0
    const tokens = state.tokens
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i]
      if (t.type !== `container_${FIGURE_NAME}_open`) continue
      const m = /^figure\s+([a-z0-9][a-z0-9-]*)\s*$/i.exec(t.info.trim())
      if (!m) continue
      const id = m[1].toLowerCase()
      seq += 1
      const num = chapter != null ? `${chapter}.${seq}` : `${seq}`
      figures[id] = num
      t.meta = t.meta || {}
      t.meta.figureId = id
      t.meta.figureNum = num

      // Locate matching close + last inner paragraph; tag them as caption.
      const closeIdx = findFigureContainerRange(tokens, i)
      if (closeIdx < 0) continue
      const closeTok = tokens[closeIdx]
      closeTok.meta = closeTok.meta || {}
      closeTok.meta.figureId = id
      closeTok.meta.figureNum = num
      const inner = findLastInnerParagraph(tokens, i, closeIdx)
      if (inner) {
        tokens[inner.openIdx].meta = tokens[inner.openIdx].meta || {}
        tokens[inner.openIdx].meta.figureCaption = true
        tokens[inner.closeIdx].meta = tokens[inner.closeIdx].meta || {}
        tokens[inner.closeIdx].meta.figureCaption = true
        // Compute plain caption text for later (if needed).
        const inlineTok = tokens[inner.openIdx + 1]
        if (inlineTok && inlineTok.type === 'inline') {
          t.meta.captionText = plainTextFromInline(inlineTok)
        }
      }
    }
    env._figures = figures
  })

  // 3) Reference pass: walk inline tokens and replace @fig-id text tokens
  //    with a clickable link. Done after inline parsing so any @fig-id
  //    inside code spans is left alone (markdown-it tokenizes code_inline
  //    separately from text).
  md.core.ruler.after('inline', 'figure_references', (state) => {
    const env = state.env || {}
    const figures = env._figures || {}
    if (!Object.keys(figures).length) return
    for (const token of state.tokens) {
      if (token.type !== 'inline' || !token.children) continue
      const newChildren = []
      for (const child of token.children) {
        if (child.type !== 'text' || !REF_PATTERN.test(child.content)) {
          newChildren.push(child)
          continue
        }
        REF_PATTERN.lastIndex = 0
        const parts = []
        let lastIndex = 0
        let m
        while ((m = REF_PATTERN.exec(child.content)) !== null) {
          if (m.index > lastIndex) {
            parts.push({ kind: 'text', value: child.content.slice(lastIndex, m.index) })
          }
          parts.push({ kind: 'ref', id: m[1] })
          lastIndex = m.index + m[0].length
        }
        if (lastIndex < child.content.length) {
          parts.push({ kind: 'text', value: child.content.slice(lastIndex) })
        }
        for (const part of parts) {
          if (part.kind === 'text') {
            const t = new state.Token('text', '', 0)
            t.content = part.value
            t.level = child.level
            newChildren.push(t)
          } else {
            const num = figures[part.id]
            if (!num) {
              const t = new state.Token('text', '', 0)
              t.content = `@${part.id}`
              t.level = child.level
              newChildren.push(t)
              continue
            }
            const open = new state.Token('link_open', 'a', 1)
            open.attrSet('href', `#${part.id}`)
            open.attrSet('class', 'figure-ref')
            open.level = child.level
            const text = new state.Token('text', '', 0)
            text.content = `图 ${num}`
            text.level = child.level
            const close = new state.Token('link_close', 'a', -1)
            close.level = child.level
            newChildren.push(open, text, close)
          }
        }
      }
      token.children = newChildren
    }
  })

  // 4) Override container open/close renderers to emit <div class="figure" id="...">,
  //    and override paragraph open/close inside the figure container to emit
  //    a <p class="figure-caption">图 N.M ...</p> instead of a normal <p>.
  md.renderer.rules[`container_${FIGURE_NAME}_open`] = (tokens, idx) => {
    const t = tokens[idx]
    const id = t.meta && t.meta.figureId
    const idAttr = id ? ` id="${id}"` : ''
    return `<div class="figure"${idAttr}>\n`
  }
  md.renderer.rules[`container_${FIGURE_NAME}_close`] = () => `</div>\n`

  const defaultParagraphOpen =
    md.renderer.rules.paragraph_open ||
    ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options))
  const defaultParagraphClose =
    md.renderer.rules.paragraph_close ||
    ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options))

  md.renderer.rules.paragraph_open = (tokens, idx, options, env, self) => {
    const t = tokens[idx]
    if (t.meta && t.meta.figureCaption) {
      // Find the corresponding container_figure_open to retrieve the number.
      let num = null
      for (let i = idx - 1; i >= 0; i--) {
        if (tokens[i].type === `container_${FIGURE_NAME}_open`) {
          num = tokens[i].meta && tokens[i].meta.figureNum
          break
        }
      }
      const prefix = num ? `图 ${num}　` : ''
      // Stash prefix on the token so we can emit it after the opening tag.
      t._captionPrefix = prefix
      return `<p class="figure-caption">${prefix}`
    }
    return defaultParagraphOpen(tokens, idx, options, env, self)
  }
  md.renderer.rules.paragraph_close = (tokens, idx, options, env, self) => {
    const t = tokens[idx]
    if (t.meta && t.meta.figureCaption) {
      return `</p>\n`
    }
    return defaultParagraphClose(tokens, idx, options, env, self)
  }
}
