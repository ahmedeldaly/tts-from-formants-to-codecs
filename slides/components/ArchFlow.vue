<script setup lang="ts">
// Main diagram: Client → API Gateway → Backend → Database, then the response loop back.
// step 0: client · 1: gateway · 2: backend · 3: database · 4: response loop
const props = defineProps<{ step: number }>()
const on = (n: number) => props.step >= n
const hot = (n: number) => props.step === n || (n === 0 && props.step >= 4)

const Y = 120, W = 230, H = 140
const xs = [30, 340, 650, 960]
const mid = Y + 52 // request lane
const loop = `M ${xs[3] + W / 2} ${Y + H} C ${xs[3] + W / 2} 420, ${xs[0] + W / 2} 420, ${xs[0] + W / 2} ${Y + H + 8}`
</script>

<template>
  <svg viewBox="0 0 1220 450" class="dg w-full">
    <DgDefs id="arch" />

    <!-- lane labels -->
    <text x="30" y="80" class="mono rv" :class="{ on: on(1) }">REQUEST →</text>
    <text x="1190" y="440" text-anchor="end" class="mono rv" :class="{ on: on(4) }" style="fill: var(--accent)">← RESPONSE</text>

    <!-- nodes -->
    <DgNode :x="xs[0]" :y="Y" :w="W" :h="H" label="01 · Client" title="Web / Mobile App" sub="calls the public API" :hot="hot(0)" />
    <DgNode :x="xs[1]" :y="Y" :w="W" :h="H" label="02 · Edge" title="API Gateway" sub="auth · rate limit · route" :on="on(1)" :hot="hot(1)" />
    <DgNode :x="xs[2]" :y="Y" :w="W" :h="H" label="03 · Service" title="App Backend" sub="business logic" :on="on(2)" :hot="hot(2)" />
    <DgNode :x="xs[3]" :y="Y" :w="W" :h="H" label="04 · State" title="Database" sub="source of truth" :on="on(3)" :hot="hot(3)" />

    <!-- forward hops -->
    <g v-for="(lbl, i) in ['HTTPS', 'route', 'SQL']" :key="lbl" class="rv" :class="{ on: on(i + 1) }">
      <path :d="`M ${xs[i] + W + 6} ${mid} L ${xs[i + 1] - 8} ${mid}`" class="wire flow" :marker-end="`url(#arch-ink)`" />
      <text :x="(xs[i] + W + xs[i + 1]) / 2" :y="mid - 14" text-anchor="middle" class="mono" style="letter-spacing: .06em">{{ lbl }}</text>
    </g>

    <!-- response feedback loop -->
    <g class="rv" :class="{ on: on(4) }">
      <path :d="loop" class="wire accent flow slow" marker-end="url(#arch-accent)" />
      <circle r="6" class="dot">
        <animateMotion dur="2.4s" repeatCount="indefinite" :path="loop" />
      </circle>
      <rect x="490" y="372" width="240" height="34" rx="17" fill="var(--paper)" stroke="var(--accent)" />
      <text x="610" y="394" text-anchor="middle" class="mono" style="fill: var(--accent); letter-spacing: .08em">200 OK · JSON · ~40 ms</text>
    </g>
  </svg>
</template>
