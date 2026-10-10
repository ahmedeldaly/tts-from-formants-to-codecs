---
theme: default
title: One Request, Three Hops
info: App → API Gateway → Application Backend → Database
colorSchema: light
aspectRatio: 16/9
canvasWidth: 1280
transition: fade
download: true
exportFilename: api-architecture
fonts:
  sans: Inter
  mono: JetBrains Mono
  weights: '400,600,700,800'
---

<div class="absolute inset-0 px-[72px] py-[56px] flex flex-col justify-between">
  <div class="eyebrow">Architecture · 01</div>
  <div>
    <div class="text-[104px] leading-[0.95] font-extrabold tracking-[-0.045em] text-ink">
      One request,<br/>three hops.
    </div>
    <div class="mt-8 font-mono text-[18px] text-muted flex items-center gap-3">
      <span>App</span><span class="text-accent">→</span>
      <span>API Gateway</span><span class="text-accent">→</span>
      <span>Backend</span><span class="text-accent">→</span>
      <span>Database</span>
    </div>
  </div>
  <div class="flex justify-between eyebrow"><span>Press → to begin</span><span>F · fullscreen</span></div>
</div>

---
clicks: 4
---

<div class="eyebrow">The architecture</div>
<h1 class="mt-3">Every call takes the same path.</h1>

<div class="mt-8">
  <ArchFlow :step="$clicks" />
</div>

---
clicks: 4
---

<div class="eyebrow">02 · API Gateway</div>
<h1 class="mt-3">The front door says no early.</h1>

<div class="mt-6">
  <GatewayFlow :step="$clicks" />
</div>

---
clicks: 3
---

<div class="eyebrow">03 · Application Backend</div>
<h1 class="mt-3">Stateless, so it scales out.</h1>

<div class="mt-8">
  <ScaleFlow :step="$clicks" />
</div>

---
clicks: 4
---

<div class="eyebrow">04 · Database</div>
<h1 class="mt-3">Writes go up, reads fan out.</h1>

<div class="mt-8">
  <DbFlow :step="$clicks" />
</div>

---
clicks: 3
---

<div class="eyebrow">Feedback loop</div>
<h1 class="mt-3">Every request owes a response.</h1>

<div class="mt-6">
  <RoundTrip :step="$clicks" />
</div>

---

<div class="eyebrow">Takeaways</div>
<h1 class="mt-3">Keep each hop boring.</h1>

<div class="mt-14 grid grid-cols-3 gap-6">
  <div v-for="(c, i) in [
      ['Gateway', 'Guard the edge', 'Reject bad traffic before it costs compute.'],
      ['Backend', 'Hold no state', 'Any replica can serve any request.'],
      ['Database', 'Own the truth', 'Pool connections, split reads from writes.'],
    ]" :key="c[0]" v-click="i + 1" class="card p-7 h-[220px] flex flex-col justify-between">
    <div class="eyebrow">0{{ i + 2 }} · {{ c[0] }}</div>
    <div>
      <div class="text-[30px] font-bold tracking-[-0.02em] leading-tight">{{ c[1] }}</div>
      <div class="mt-2 text-[16px] text-muted">{{ c[2] }}</div>
    </div>
  </div>
</div>
