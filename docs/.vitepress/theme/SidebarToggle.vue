<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

const STORAGE_KEY = 'sidebar-collapsed'
const collapsed = ref(false)

function apply(value: boolean) {
  if (typeof document === 'undefined') return
  document.body.classList.toggle('sidebar-collapsed', value)
}

function toggle() {
  collapsed.value = !collapsed.value
  apply(collapsed.value)
  try {
    localStorage.setItem(STORAGE_KEY, collapsed.value ? '1' : '0')
  } catch {
    /* ignore */
  }
}

function onKey(e: KeyboardEvent) {
  // Ctrl/Cmd + \ 切换
  if ((e.ctrlKey || e.metaKey) && e.key === '\\') {
    e.preventDefault()
    toggle()
  }
}

onMounted(() => {
  try {
    collapsed.value = localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    /* ignore */
  }
  apply(collapsed.value)
  window.addEventListener('keydown', onKey)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  apply(false)
})
</script>

<template>
  <button
    class="sidebar-toggle"
    :class="{ 'is-collapsed': collapsed }"
    :title="collapsed ? '展开目录 (Ctrl/Cmd+\\)' : '收起目录 (Ctrl/Cmd+\\)'"
    :aria-label="collapsed ? '展开目录' : '收起目录'"
    :aria-pressed="collapsed"
    @click="toggle"
  >
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path
        v-if="!collapsed"
        d="M15 6l-6 6 6 6"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
      <path
        v-else
        d="M9 6l6 6-6 6"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
    </svg>
  </button>
</template>
