# Ahmed, CTO: 15-second motion graphics

`ahmed_cto.mp4`: 1920×1080, 30 fps, 15 s, English voice-over with captions.

| Time | Scene | Voice-over |
|---|---|---|
| 0.0–3.0 | Intro, name card, wave | "Meet Ahmed, CTO of Mara7el." |
| 3.0–6.6 | Code, cloud, growth dashboard, rocket | "He builds the technology that moves the company forward," |
| 6.6–8.9 | Confetti, hearts, rising chart | "and he loves every minute of it." |
| 8.9–12.4 | Salad bowl, fork bites, chewing | "His secret fuel? A fresh, crunchy salad." |
| 12.4–15.0 | Wink, thumbs up, tagline and logo | "Built to lead. Powered by greens." |

## How it's made

- `scene.html`: the character (vector SVG drawn from reference photos) and all animation, driven by `renderFrame(t)`.
- `audio.py`: Kokoro TTS voice (`am_michael`), a synthesized music bed with ducking, and whoosh/pop effects. Writes `out/mix.wav` and `out/vo_timing.json`.
- `render.mjs`: steps through every frame in headless Chromium (Playwright) and pipes the frames plus the audio into ffmpeg.

## Rebuild

```bash
pip install kokoro-onnx soundfile numpy imageio-ffmpeg
# download kokoro-v1.0.onnx + voices-v1.0.bin from
# https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
KOKORO_DIR=/path/to/models python3 audio.py
FFMPEG=$(python3 -c "import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())") node render.mjs
# preview stills only:  node render.mjs --stills 1.8,4.8,10.6
```

To change the wording, edit `LINES` in `audio.py` and the scene timings `B` in `scene.html`.
