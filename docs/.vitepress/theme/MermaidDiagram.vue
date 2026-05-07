<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useData } from 'vitepress'
import mermaid from 'mermaid'

const props = defineProps<{
  code: string
}>()

const { isDark } = useData()
const diagram = ref<HTMLElement | null>(null)
const error = ref('')
const source = computed(() => decodeURIComponent(props.code))
let renderCount = 0
let diagramId = 0

async function renderDiagram() {
  if (!diagram.value) return

  if (!diagramId) {
    diagramId = Math.floor(Math.random() * 1_000_000_000)
  }
  renderCount += 1
  error.value = ''
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: isDark.value ? 'dark' : 'default',
  })

  try {
    const { svg } = await mermaid.render(`mermaid-${diagramId}-${renderCount}`, source.value)
    diagram.value.innerHTML = svg
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    diagram.value.textContent = source.value
  }
}

onMounted(renderDiagram)
watch(isDark, renderDiagram)
</script>

<template>
  <div class="mermaid-block">
    <div ref="diagram" />
    <pre v-if="error" class="mermaid-error">{{ error }}</pre>
  </div>
</template>
