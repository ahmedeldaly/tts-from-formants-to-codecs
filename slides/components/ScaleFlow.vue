<script setup lang="ts">
// Stateless backend: one replica, then N identical replicas behind the gateway, all sharing one DB.
// step 1: first replica · 2: scale out · 3: shared state lives in the DB
const props = defineProps<{ step: number }>()
const on = (n: number) => props.step >= n
const reps = [
  { y: 30, n: 1, at: 1 },
  { y: 165, n: 2, at: 2 },
  { y: 300, n: 3, at: 2 },
]
</script>

<template>
  <svg viewBox="0 0 1220 450" class="dg w-full">
    <DgDefs id="sc" />

    <DgNode :x="30" :y="155" :w="230" :h="140" label="Edge" title="API Gateway" sub="load balances" />

    <g v-for="r in reps" :key="r.n" class="rv" :class="{ on: on(r.at) }">
      <path :d="`M 266 225 C 340 225, 340 ${r.y + 60}, 412 ${r.y + 60}`" class="wire flow" marker-end="url(#sc-ink)" />
      <rect x="420" :y="r.y" width="340" height="120" rx="16" class="box" />
      <text x="442" :y="r.y + 32" class="mono">REPLICA 0{{ r.n }}</text>
      <text x="442" :y="r.y + 82" class="title">App Backend</text>
      <text x="740" :y="r.y + 82" text-anchor="end" class="mono" style="letter-spacing: .04em">no local state</text>
    </g>

    <g class="rv" :class="{ on: on(3) }">
      <path v-for="r in reps" :key="r.n" :d="`M 766 ${r.y + 60} C 850 ${r.y + 60}, 850 225, 930 225`" class="wire flow" marker-end="url(#sc-ink)" />
      <DgNode :x="940" :y="155" :w="250" :h="140" label="Shared" title="Database" sub="sessions · data · locks" :hot="true" />
    </g>
  </svg>
</template>
