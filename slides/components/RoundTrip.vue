<script setup lang="ts">
// Sequence view of one round trip, then where the latency budget goes.
// step 1: request hops · 2: responses loop back · 3: latency budget
const props = defineProps<{ step: number }>()
const on = (n: number) => props.step >= n

const lanes = [
  { x: 140, t: 'Client' },
  { x: 450, t: 'Gateway' },
  { x: 760, t: 'Backend' },
  { x: 1070, t: 'Database' },
]
const req = [0, 1, 2].map(i => ({ x1: lanes[i].x, x2: lanes[i + 1].x, y: 112 + i * 34 }))
const res = [2, 1, 0].map((i, k) => ({ x1: lanes[i + 1].x, x2: lanes[i].x, y: 228 + k * 34 }))

const budget = [
  { t: 'network', ms: 8 },
  { t: 'gateway', ms: 4 },
  { t: 'backend', ms: 16 },
  { t: 'database', ms: 14 },
]
const total = budget.reduce((s, b) => s + b.ms, 0)
const segs = budget.reduce<{ t: string, ms: number, x: number, w: number }[]>((acc, b) => {
  const x = acc.length ? acc[acc.length - 1].x + acc[acc.length - 1].w : 30
  acc.push({ ...b, x, w: (b.ms / total) * 1160 })
  return acc
}, [])
</script>

<template>
  <svg viewBox="0 0 1220 460" class="dg w-full">
    <DgDefs id="rt" />

    <g v-for="l in lanes" :key="l.t">
      <rect :x="l.x - 90" y="20" width="180" height="50" rx="12" class="box" />
      <text :x="l.x" y="51" text-anchor="middle" style="font-size: 17px; font-weight: 700">{{ l.t }}</text>
      <line :x1="l.x" y1="74" :x2="l.x" y2="340" stroke="var(--line)" stroke-dasharray="3 5" />
    </g>

    <g class="rv" :class="{ on: on(1) }">
      <path v-for="r in req" :key="r.y" :d="`M ${r.x1 + 4} ${r.y} L ${r.x2 - 8} ${r.y}`" class="wire flow" marker-end="url(#rt-ink)" />
    </g>
    <g class="rv" :class="{ on: on(2) }">
      <path v-for="r in res" :key="r.y" :d="`M ${r.x1 - 4} ${r.y} L ${r.x2 + 8} ${r.y}`" class="wire accent flow" marker-end="url(#rt-accent)" />
    </g>

    <g class="rv" :class="{ on: on(3) }">
      <text x="30" y="378" class="mono">LATENCY BUDGET · {{ total }} MS</text>
      <g v-for="(s, i) in segs" :key="s.t">
        <rect :x="s.x" y="392" :width="s.w - 4" height="40" rx="10" :fill="s.t === 'backend' ? 'var(--accent)' : i % 2 ? 'var(--ink)' : 'var(--paper)'" stroke="var(--ink)" stroke-width="1.25" />
        <text :x="s.x + 14" y="417" style="font-size: 14px; font-weight: 600" :style="{ fill: s.t === 'backend' || i % 2 ? 'var(--paper)' : 'var(--ink)' }">{{ s.t }} {{ s.ms }}ms</text>
      </g>
    </g>
  </svg>
</template>
