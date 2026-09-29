// Renders scene.html frame-by-frame with headless Chromium and pipes the
// frames into ffmpeg together with the mixed audio track.
//
//   node render.mjs --stills 1.6,4.5,7.5   -> PNG stills in out/
//   node render.mjs                         -> out/ahmed_cto.mp4
import { createRequire } from 'module';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('playwright')); }
catch { ({ chromium } = require('/opt/node22/lib/node_modules/playwright')); }

const here = path.dirname(new URL(import.meta.url).pathname);
const out = path.join(here, 'out');
fs.mkdirSync(out, { recursive: true });
const args = process.argv.slice(2);
const FPS = 30, DURATION = 15;
const ffmpeg = process.env.FFMPEG || 'ffmpeg';
const voPath = path.join(out, 'vo_timing.json');
const vo = fs.existsSync(voPath) ? JSON.parse(fs.readFileSync(voPath, 'utf8')) : null;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
if (vo) await page.addInitScript(v => { window.VO = v; }, vo);
await page.goto('file://' + path.join(here, 'scene.html'));
await page.evaluate(() => document.fonts.ready);

const shot = async t => {
  await page.evaluate(t => window.renderFrame(t), t);
  return page.screenshot({ type: 'png' });
};

if (args[0] === '--stills') {
  for (const t of args[1].split(',').map(Number)) {
    fs.writeFileSync(path.join(out, `still_${t.toFixed(2)}.png`), await shot(t));
  }
} else {
  const audio = path.join(out, 'mix.wav');
  const ff = spawn(ffmpeg, [
    '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-i', audio,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart',
    path.join(out, 'ahmed_cto.mp4'),
  ], { stdio: ['pipe', 'inherit', 'inherit'] });
  const n = FPS * DURATION;
  for (let i = 0; i < n; i++) {
    const buf = await shot(i / FPS);
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 60 === 0) process.stderr.write(`frame ${i}/${n}\n`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}
await browser.close();
