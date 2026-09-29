"""Builds the soundtrack: Kokoro TTS voice-over + a synthesized music bed + SFX.

Writes out/mix.wav and out/vo_timing.json (caption timing used by scene.html).
Needs kokoro-v1.0.onnx and voices-v1.0.bin (github.com/thewh1teagle/kokoro-onnx
releases) in the directory given by KOKORO_DIR (default: this folder).
"""
import json
import os

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
KDIR = os.environ.get("KOKORO_DIR", HERE)
SR = 24000
DUR = 15.0
VOICE = os.environ.get("VOICE", "am_michael")

# (spoken text, caption text, start time in seconds)
LINES = [
    ("Meet Ahmed, CTO of Marahel.", "Meet Ahmed, CTO of Mara7el.", 0.35),
    ("He builds the technology that moves the company forward,",
     "He builds the technology that moves the company forward,", 3.15),
    ("and he loves every minute of it.", "and he loves every minute of it.", 6.75),
    ("His secret fuel? A fresh, crunchy salad.", "His secret fuel? A fresh, crunchy salad.", 9.05),
    ("Built to lead. Powered by greens.", "Built to lead. Powered by greens.", 12.55),
]
SCENE_CUTS = [3.0, 6.6, 8.9, 12.4]


def trim(a, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    return a[max(0, idx[0] - 240): idx[-1] + 480] if len(idx) else a


def voice_track():
    k = Kokoro(os.path.join(KDIR, "kokoro-v1.0.onnx"), os.path.join(KDIR, "voices-v1.0.bin"))
    track = np.zeros(int(SR * DUR))
    timing = []
    for spoken, caption, start in LINES:
        a, sr = k.create(spoken, voice=VOICE, speed=1.08, lang="en-us")
        assert sr == SR
        a = trim(a)
        i = int(start * SR)
        n = min(len(a), len(track) - i)
        track[i:i + n] += a[:n]
        timing.append({"start": start, "end": round(start + len(a) / SR, 2), "text": caption})
    return track, timing


def env(n, a, r):
    e = np.ones(n)
    a, r = min(a, n), min(r, n)
    e[:a] = np.linspace(0, 1, a)
    e[n - r:] *= np.linspace(1, 0, r)
    return e


def note(freq, dur, amp, kind="pluck"):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == "pluck":
        w = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t) + 0.12 * np.sin(6 * np.pi * freq * t)
        return amp * w * np.exp(-t * 6)
    if kind == "pad":
        w = sum(np.sin(2 * np.pi * freq * d * t) for d in (1.0, 1.003, 0.997)) / 3
        w += 0.2 * np.sin(4 * np.pi * freq * t)
        return amp * w * env(n, int(0.25 * SR), int(0.4 * SR))
    if kind == "bass":
        return amp * np.tanh(2 * np.sin(2 * np.pi * freq * t)) * np.exp(-t * 3)
    raise ValueError(kind)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def music():
    rng = np.random.default_rng(1)
    track = np.zeros(int(SR * DUR))
    bpm = 116
    beat = 60 / bpm

    def add(sig, t):
        i = int(t * SR)
        n = min(len(sig), len(track) - i)
        if n > 0:
            track[i:i + n] += sig[:n]

    # C - G - Am - F, one chord per bar
    chords = [[60, 64, 67, 72], [55, 59, 62, 67], [57, 60, 64, 69], [53, 57, 60, 65]]
    bar = 4 * beat
    nbars = int(DUR / bar) + 1
    for b in range(nbars):
        ch = chords[b % 4]
        t0 = b * bar
        for m in ch[:3]:
            add(note(mtof(m), bar, 0.05, "pad"), t0)
        add(note(mtof(ch[0] - 24), beat * 1.5, 0.18, "bass"), t0)
        add(note(mtof(ch[0] - 24), beat * 1.5, 0.14, "bass"), t0 + 2 * beat)
        arp = [ch[0], ch[1], ch[2], ch[3], ch[2], ch[1], ch[3] + 12 - 12, ch[2]]
        for s, m in enumerate(arp):
            add(note(mtof(m + 12), 0.35, 0.05, "pluck"), t0 + s * beat / 2)
        for q in range(4):
            tt = t0 + q * beat
            # kick
            n = int(0.25 * SR); t = np.arange(n) / SR
            f = 50 + 90 * np.exp(-t * 30)
            add(0.35 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12), tt)
            # hat on off-beats
            n = int(0.05 * SR)
            add(0.035 * rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 80), tt + beat / 2)
            if q in (1, 3):  # clap
                n = int(0.15 * SR)
                add(0.08 * rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 25), tt)
    track *= env(len(track), int(0.3 * SR), int(1.2 * SR))
    return track


def sfx():
    rng = np.random.default_rng(2)
    track = np.zeros(int(SR * DUR))
    for c in SCENE_CUTS:
        n = int(0.6 * SR)
        noise = rng.standard_normal(n)
        # swept band-ish whoosh: filtered noise via moving average with changing width
        out = np.zeros(n)
        acc = np.cumsum(noise)
        for i in range(n):
            w = int(40 - 34 * np.sin(np.pi * i / n))
            j0 = max(0, i - w)
            out[i] = (acc[i] - acc[j0]) / max(1, i - j0)
        out *= np.sin(np.pi * np.arange(n) / n) ** 2
        i = int((c - 0.3) * SR)
        track[i:i + n] += 0.5 * out / (np.abs(out).max() + 1e-9)
    # pops for elements appearing
    for t in [0.5, 0.95, 1.2, 3.2, 3.45, 3.7, 6.95, 9.6, 12.55, 13.9]:
        n = int(0.08 * SR); tt = np.arange(n) / SR
        f = 900 - 500 * tt / 0.08
        pop = 0.18 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 45)
        i = int(t * SR); track[i:i + n] += pop
    return track


def main():
    os.makedirs(OUT, exist_ok=True)
    vo, timing = voice_track()
    mus = music()
    fx = sfx()
    # duck music under the voice
    envl = np.convolve(np.abs(vo), np.ones(2400) / 2400, mode="same")
    duck = 1 - 0.55 * np.clip(envl / (envl.max() * 0.25), 0, 1)
    duck = np.convolve(duck, np.ones(4800) / 4800, mode="same")
    mix = 1.0 * vo / (np.abs(vo).max() + 1e-9) * 0.9 + 0.55 * mus * duck + 0.5 * fx
    mix /= max(1.0, np.abs(mix).max() / 0.97)
    sf.write(os.path.join(OUT, "mix.wav"), np.stack([mix, mix], axis=1), SR)
    with open(os.path.join(OUT, "vo_timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
