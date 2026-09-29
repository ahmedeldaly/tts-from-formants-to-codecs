// Renders index.html frame-by-frame with headless Chromium and encodes an MP4.
import { chromium } from "playwright-core";
import ffmpegPath from "ffmpeg-static";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const dir = path.dirname(fileURLToPath(import.meta.url));
const FPS = 30, DURATION = 10;
const out = path.join(dir, "bubblr-fizz-ad.mp4");
const audio = path.join(dir, "soundtrack.wav");

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined, // e.g. /opt/pw-browsers/chromium-1194/chrome-linux/chrome
});
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto(pathToFileURL(path.join(dir, "index.html")).href + "?capture");
await page.evaluate(() => window.ready);

const args = ["-y", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-"];
if (existsSync(audio)) args.push("-i", audio, "-c:a", "aac", "-b:a", "192k", "-shortest");
args.push("-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out);
const ff = spawn(ffmpegPath, args, { stdio: ["pipe", "inherit", "inherit"] });

for (let i = 0; i < FPS * DURATION; i++) {
  const b64 = await page.evaluate(t => {
    window.renderAt(t);
    return document.getElementById("c").toDataURL("image/jpeg", 0.96).split(",")[1];
  }, i / FPS);
  if (!ff.stdin.write(Buffer.from(b64, "base64"))) await new Promise(r => ff.stdin.once("drain", r));
  if (i % 30 === 0) process.stdout.write(`frame ${i}/${FPS * DURATION}\n`);
}
ff.stdin.end();
await new Promise(r => ff.on("close", r));
await browser.close();
console.log("wrote", out);
