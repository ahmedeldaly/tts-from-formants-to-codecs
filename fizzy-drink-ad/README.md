# Bubblr — 10 s fizzy-drink motion graphic

`bubblr-fizz-ad.mp4` is a 1920×1080, 30 fps, 10 s spot for a fictional blood-orange sparkling soda, with a synthesised soundtrack.

| Time | Beat |
| --- | --- |
| 0.0–0.95 s | Can drops in and bounces on a night-blue background of rising bubbles |
| 0.95–2.0 s | Can shakes as pressure builds ("Something's about to bubble up…") |
| 2.0 s | **POP!** Tab flips, spray, shockwave, flash, tangerine sunburst reveal; the beat drops |
| 2.6–5.5 s | Can spins on the left while "ICE COLD. / ZERO SUGAR. / 100% FIZZ." slam in |
| 5.5–7.4 s | A wave of soda floods the screen; blood-orange slices and a "CITRUS BURST" marquee |
| 7.4–10 s | End card: logo bounces in, "Taste the fizz.", "NOW IN STORES" |

## Files

- `index.html` — the animation (canvas, deterministic `renderAt(t)`); open it in a browser for a looping preview, click to restart.
- `audio.mjs` — synthesises `soundtrack.wav` (bounce thuds, shake rattle, pop, fizz crackle, 120 bpm groove, whooshes, logo blips).
- `render.mjs` — drives headless Chromium frame by frame and encodes the MP4 with ffmpeg.
- `fonts/` — Luckiest Guy and Baloo 2 (SIL Open Font License), bundled so rendering works offline.

## Re-render

```sh
npm install
CHROMIUM_PATH=/path/to/chromium npm run render   # CHROMIUM_PATH optional if Playwright's browser is installed
```
