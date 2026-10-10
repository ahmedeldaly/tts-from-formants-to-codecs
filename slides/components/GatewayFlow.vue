<script setup lang="ts">
// API Gateway: many clients in, checks in order, one routed request out, rejects looped back.
// step 1: auth · 2: rate limit · 3: route → backend · 4: reject loop
const props = defineProps<{ step: number }>()
const on = (n: number) => props.step >= n

const clients = [
  { y: 60, t: 'Web' },
  { y: 189, t: 'iOS / Android' },
  { y: 318, t: 'Partner' },
]
const checks = [
  { k: 'AUTH', v: 'verify token' },
  { k: 'LIMIT', v: '100 req / min' },
  { k: 'ROUTE', v: '/orders → orders-svc' },
]
const reject = 'M 540 40 C 540 -6, 120 -6, 120 52'
</script>

<template>
  <svg viewBox="0 -20 1220 470" class="dg w-full">
    <DgDefs id="gw" />

    <!-- clients converge on the gateway -->
    <g v-for="c in clients" :key="c.t">
      <rect x="30" :y="c.y" width="180" height="72" rx="14" class="box" />
      <text x="50" :y="c.y + 44" class="title" style="font-size: 18px">{{ c.t }}</text>
      <path :d="`M 214 ${c.y + 36} C 270 ${c.y + 36}, 270 225, 322 225`" class="wire flow" marker-end="url(#gw-ink)" />
    </g>

    <!-- gateway shell -->
    <rect x="330" y="40" width="420" height="370" rx="20" class="box" />
    <text x="356" y="76" class="mono">API GATEWAY</text>

    <!-- ordered checks -->
    <g v-for="(c, i) in checks" :key="c.k" class="rv" :class="{ on: on(i + 1) }">
      <rect x="356" :y="108 + i * 92" width="368" height="68" rx="12" class="box" :style="props.step === i + 1 ? 'stroke: var(--accent)' : ''" />
      <text x="378" :y="148 + i * 92" class="mono" :style="props.step === i + 1 ? 'fill: var(--accent)' : ''">0{{ i + 1 }} {{ c.k }}</text>
      <text x="490" :y="148 + i * 92" style="font-size: 18px; font-weight: 600">{{ c.v }}</text>
    </g>

    <!-- routed request out -->
    <g class="rv" :class="{ on: on(3) }">
      <path d="M 754 225 L 852 225" class="wire flow" marker-end="url(#gw-ink)" />
      <DgNode :x="860" :y="155" :w="330" :h="140" label="Upstream" title="App Backend" sub="only sees trusted traffic" :hot="props.step === 3" />
    </g>

    <!-- rejected requests loop straight back to the client -->
    <g class="rv" :class="{ on: on(4) }">
      <path :d="reject" class="wire accent flow" marker-end="url(#gw-accent)" />
      <circle r="5" class="dot"><animateMotion dur="1.8s" repeatCount="indefinite" :path="reject" /></circle>
      <text x="330" y="-2" text-anchor="middle" class="mono" style="fill: var(--accent)">401 · 429 — FAIL FAST</text>
    </g>
  </svg>
</template>
