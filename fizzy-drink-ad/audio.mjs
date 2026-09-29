// Synthesises the 10 s soundtrack (soundtrack.wav), cued to the animation timeline.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const SR = 44100, DUR = 10, N = SR * DUR;
const L = new Float32Array(N), Rt = new Float32Array(N);
let seed = 12345;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;

// add a mono voice with optional stereo pan (-1..1)
function add(t0, len, fn, gain = 1, pan = 0) {
  const s0 = Math.max(0, Math.floor(t0 * SR)), s1 = Math.min(N, Math.floor((t0 + len) * SR));
  const gl = gain * Math.cos((pan + 1) * Math.PI / 4) * Math.SQRT2, gr = gain * Math.sin((pan + 1) * Math.PI / 4) * Math.SQRT2;
  for (let i = s0; i < s1; i++) {
    const v = fn((i - s0) / SR);
    L[i] += v * gl; Rt[i] += v * gr;
  }
}

// one-pole filtered noise source
function noise(hp) {
  let lp = 0, prev = 0;
  return () => {
    const n = rand();
    if (hp) { const o = n - prev; prev = n; return o * 0.5; }
    lp += 0.08 * (n - lp); return lp * 3;
  };
}

const kick = (a = 1) => t => a * Math.sin(2 * Math.PI * (45 * t + 90 * (1 - Math.exp(-t * 30)) / 30)) * Math.exp(-t * 9);
function thud(t0, a) { add(t0, 0.35, kick(a)); }
function hat(t0, a = 0.12) { const n = noise(true); add(t0, 0.08, t => n() * Math.exp(-t * 60), a, 0.3); }
function clap(t0, a = 0.35) {
  const n = noise(true);
  add(t0, 0.25, t => n() * (Math.exp(-t * 25) + 0.6 * Math.exp(-Math.max(0, t - 0.012) * 40) * (t > 0.012)), a, -0.15);
}
function impact(t0, a = 0.6) {
  thud(t0, a);
  const n = noise(false);
  add(t0, 0.3, t => n() * Math.exp(-t * 14), a * 0.5);
}

// 1) can drop: bounce thuds (match easeOutBounce contacts at p·0.95 s)
thud(0.346, 0.9); thud(0.69, 0.45); thud(0.864, 0.25);

// 2) shake rattle + riser before the pop
{
  const n = noise(true);
  add(0.95, 1.05, t => { const k = t / 1.05; return n() * k * k * (0.5 + 0.5 * Math.sin(2 * Math.PI * 18 * t)); }, 0.35);
  let ph = 0;
  add(1.0, 1.0, t => { ph += 2 * Math.PI * (200 + 1000 * t * t) / SR; return Math.sin(ph) * t * t; }, 0.12);
}

// 3) the POP: click + boom + pssst
{
  const n = noise(true), n2 = noise(false);
  add(2.0, 0.01, t => rand() * (1 - t / 0.01), 1.0);
  add(2.0, 0.6, kick(1.1));
  add(2.0, 1.2, t => n() * Math.exp(-t * 3.5) * Math.min(1, t * 200), 0.55);
  add(2.0, 1.2, t => n2() * Math.exp(-t * 5), 0.25);
}

// 4) carbonation crackle bed
for (let t = 2.0; t < DUR; ) {
  const rate = 60 + 260 * Math.exp(-(t - 2) * 1.2);
  t += -Math.log((rand() + 1) / 2 * 0.999 + 0.001) / rate;
  const a = 0.05 + 0.08 * Math.abs(rand());
  add(t, 0.004, x => rand() * (1 - x / 0.004), a, rand() * 0.8);
}

// 5) groove at 120 bpm from the pop
const beat = 0.5;
for (let t = 2.0; t < 9.6; t += beat) {
  thud(t, 0.75);
  hat(t + beat / 2);
  if (Math.round((t - 2.0) / beat) % 2 === 1) clap(t);
}
// bass: A – F#m – D – E, eighth-note plucks
const roots = [55, 46.25, 73.42, 82.41];
for (let i = 0; i < 30; i++) {
  const t = 2.0 + i * beat / 2;
  if (t >= 9.5) break;
  const f = roots[Math.floor(i / 8) % 4] * (i % 4 === 3 ? 2 : 1);
  add(t, 0.24, x => {
    let v = 0;
    for (let h = 1; h <= 6; h++) v += Math.sin(2 * Math.PI * f * h * x) / h * Math.exp(-x * (6 + h * 4));
    return v * Math.min(1, x * 400);
  }, 0.35);
}
// sparkly arpeggio on top
const arp = [880, 1108.7, 1318.5, 1760, 1318.5, 1108.7];
for (let i = 0; i < 28; i++) {
  const t = 2.25 + i * beat / 2;
  if (t >= 9.5) break;
  const f = arp[i % arp.length];
  add(t, 0.3, x => Math.sin(2 * Math.PI * f * x) * Math.exp(-x * 14), 0.07, i % 2 ? 0.5 : -0.5);
}

// 6) word slams
for (const t of [2.9, 3.65, 4.4]) {
  impact(t, 0.55);
  const n = noise(false);
  add(t - 0.25, 0.25, x => n() * (x / 0.25) ** 2, 0.25);
}

// 7) soda wave whoosh
{
  const n = noise(false);
  add(5.5, 1.9, t => n() * Math.sin(Math.PI * t / 1.9) ** 2, 0.45);
  const h = noise(true);
  add(5.8, 1.4, t => h() * Math.sin(Math.PI * t / 1.4) ** 2, 0.15);
}

// 8) logo letter blips + CTA shimmer
const blips = [880, 1108.7, 1318.5, 1760, 2217.5, 2637];
blips.forEach((f, i) => {
  const t0 = 7.75 + i * 0.09 + 0.2;
  add(t0, 0.35, x => Math.sin(2 * Math.PI * f * x) * Math.exp(-x * 10), 0.14, (i / 5) * 1.2 - 0.6);
});
for (const f of [440, 554.4, 659.3, 880]) {
  add(8.95, 1.05, x => Math.sin(2 * Math.PI * f * x + Math.sin(2 * Math.PI * 6 * x) * 0.4) * Math.exp(-x * 2.2) * Math.min(1, x * 30), 0.08);
}
impact(8.95, 0.4);

// master: fade in/out, soft clip, normalise, write 16-bit stereo WAV
let peak = 0;
for (let i = 0; i < N; i++) {
  const t = i / SR, g = Math.min(1, t / 0.05) * Math.min(1, (DUR - t) / 0.35);
  L[i] = Math.tanh(L[i] * 1.1) * g; Rt[i] = Math.tanh(Rt[i] * 1.1) * g;
  peak = Math.max(peak, Math.abs(L[i]), Math.abs(Rt[i]));
}
const buf = Buffer.alloc(44 + N * 4);
buf.write("RIFF", 0); buf.writeUInt32LE(36 + N * 4, 4); buf.write("WAVE", 8);
buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22);
buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 4, 28); buf.writeUInt16LE(4, 32); buf.writeUInt16LE(16, 34);
buf.write("data", 36); buf.writeUInt32LE(N * 4, 40);
const k = 0.89 / peak;
for (let i = 0; i < N; i++) {
  buf.writeInt16LE(Math.round(L[i] * k * 32767), 44 + i * 4);
  buf.writeInt16LE(Math.round(Rt[i] * k * 32767), 46 + i * 4);
}
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), "soundtrack.wav");
writeFileSync(out, buf);
console.log("wrote", out);
