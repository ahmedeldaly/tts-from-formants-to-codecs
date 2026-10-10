<script setup lang="ts">
// One thin rounded card inside an SVG diagram: small uppercase step label,
// bold title, muted subtitle, and an accent dot that pulses while "hot".
withDefaults(defineProps<{
  x: number
  y: number
  w?: number
  h?: number
  label?: string
  title: string
  sub?: string
  on?: boolean
  hot?: boolean
  ghost?: boolean
}>(), { w: 230, h: 140, on: true })
</script>

<template>
  <g class="rv" :class="{ on }">
    <rect :x="x" :y="y" :width="w" :height="h" rx="16" class="box" :class="{ ghost }" />
    <text v-if="label" :x="x + 20" :y="y + 32" class="mono">{{ label }}</text>
    <circle v-if="hot" :cx="x + w - 22" :cy="y + 27" r="4" class="dot pulse" />
    <circle v-else-if="label" :cx="x + w - 22" :cy="y + 27" r="4" fill="none" stroke="var(--line)" />
    <text :x="x + 20" :y="y + h - (sub ? 44 : 24)" class="title">{{ title }}</text>
    <text v-if="sub" :x="x + 20" :y="y + h - 20" class="sub">{{ sub }}</text>
  </g>
</template>
