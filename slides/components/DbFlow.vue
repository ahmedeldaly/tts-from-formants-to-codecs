<script setup lang="ts">
// Database tier: pooled connections, writes to primary, async replication, reads from replica.
// step 1: pool · 2: writes → primary · 3: replication · 4: reads → replica
const props = defineProps<{ step: number }>()
const on = (n: number) => props.step >= n
const write = 'M 676 140 L 852 140'
const repl = 'M 1020 214 L 1020 266'
const read = 'M 676 300 C 760 300, 760 340, 852 340'
</script>

<template>
  <svg viewBox="0 0 1220 450" class="dg w-full">
    <DgDefs id="db" />

    <DgNode :x="30" :y="150" :w="230" :h="140" label="Service" title="App Backend" sub="many requests" />

    <!-- connection pool -->
    <g class="rv" :class="{ on: on(1) }">
      <path d="M 266 220 L 352 220" class="wire flow" marker-end="url(#db-ink)" />
      <rect x="360" y="90" width="310" height="260" rx="18" class="box" />
      <text x="382" y="124" class="mono">CONNECTION POOL</text>
      <g v-for="i in 5" :key="i">
        <rect :x="382 + (i - 1) * 56" y="160" width="40" height="120" rx="10" fill="none" stroke="var(--ink)" stroke-width="1.25" />
        <rect :x="382 + (i - 1) * 56" :y="160 + (i % 3) * 30" width="40" :height="120 - (i % 3) * 30" rx="10" :fill="i < 4 ? 'var(--ink)' : 'var(--line)'" opacity="0.9" />
      </g>
      <text x="382" y="326" class="sub">reuse a few long-lived connections</text>
    </g>

    <!-- writes -->
    <g class="rv" :class="{ on: on(2) }">
      <path :d="write" class="wire flow" marker-end="url(#db-ink)" />
      <text x="764" y="126" text-anchor="middle" class="mono">WRITE</text>
      <DgNode :x="860" :y="70" :w="330" :h="140" label="Primary" title="Postgres" sub="INSERT · UPDATE · DELETE" :hot="props.step === 2" />
    </g>

    <!-- replication -->
    <g class="rv" :class="{ on: on(3) }">
      <path :d="repl" class="wire accent flow" marker-end="url(#db-accent)" />
      <text x="1034" y="246" class="mono" style="fill: var(--accent)">ASYNC WAL</text>
      <DgNode :x="860" :y="272" :w="330" :h="140" label="Replica" title="Read copy" sub="ms behind primary" :hot="props.step === 3" />
    </g>

    <!-- reads -->
    <g class="rv" :class="{ on: on(4) }">
      <path :d="read" class="wire flow" marker-end="url(#db-ink)" />
      <text x="764" y="300" text-anchor="middle" class="mono">READ</text>
    </g>
  </svg>
</template>
