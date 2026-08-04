# Text-to-Speech: A Deep Technical Guide

**From formant synthesis to flow matching and codec language models (1970s to 2026)**

This is a self-contained, equation-level guide to how TTS works, how it evolved, and where the field stands as of mid-2026. It is written as a study curriculum: every era is explained with its core math, its representative papers, and what it contributed to the systems we use today. At the end you will find a curated reading list in curriculum order, free open courses, toolkits, and a suggested study plan.

**Who this is for:** ML engineers and researchers who want to go beyond "call the API" and actually understand acoustic models, vocoders, neural codecs, and speech language models deeply enough to train, fine-tune, and debug them.

**Notation:** $x$ is text (characters or phonemes), $y$ is the waveform, $M$ is a mel spectrogram, $z$ is a latent, $c$ is conditioning. Equations use standard forms from the original papers, sometimes lightly simplified.

---

## Table of Contents

1. [Problem Formulation and the Classical Pipeline](#1-problem-formulation)
2. [The Classical Era: Formants, Concatenation, HMMs](#2-the-classical-era)
3. [Signal Processing You Must Know](#3-signal-processing-you-must-know)
4. [The Neural Revolution: WaveNet, Tacotron, FastSpeech](#4-the-neural-revolution-2016-2020)
5. [Neural Vocoders](#5-neural-vocoders)
6. [Flows and VITS: Probabilistic End-to-End TTS](#6-flows-and-vits)
7. [Diffusion and Flow Matching](#7-diffusion-and-flow-matching)
8. [Neural Audio Codecs: Speech as Discrete Tokens](#8-neural-audio-codecs)
9. [LLM-Style TTS and the Modern Scene (2023-2026)](#9-llm-style-tts-and-the-modern-scene)
10. [Evaluation](#10-evaluation)
11. [Arabic TTS Notes](#11-arabic-tts-notes)
12. [Datasets](#12-datasets)
13. [Open Courses and Books](#13-open-courses-and-books)
14. [Toolkits and Codebases](#14-toolkits-and-codebases)
15. [The Reading List (Curriculum Order)](#15-the-reading-list)
16. [Suggested 8-Week Study Plan](#16-suggested-8-week-study-plan)

---

## 1. Problem Formulation

TTS is conditional generation: learn and sample from

$$p(y \mid x)$$

where $y = (y_1, \dots, y_T)$ is a waveform (e.g. $T = 24{,}000$ samples per second) and $x$ is text. Three properties make this hard:

1. **One-to-many mapping.** The same sentence has infinitely many valid renditions (speaker, prosody, emotion, speed, recording conditions). Any model that regresses to a single "average" output produces muffled, oversmoothed speech. Almost every architectural idea in this guide (autoregression, flows, VAEs, GANs, diffusion, discrete tokens) is a different answer to the question "how do we model a multimodal distribution instead of its mean?"
2. **Extreme length mismatch.** A 10-word sentence maps to ~50 phonemes, ~400 mel frames, ~240,000 samples. Something must decide *alignment*: which output region corresponds to which input token. Attention, forced alignment, duration predictors, monotonic alignment search, and implicit alignment via filler tokens are all solutions to this single problem.
3. **Perceptual sensitivity.** Human ears detect millisecond-level glitches, phase artifacts, and unnatural prosody that standard losses (MSE) do not penalize.

**The classical decomposition** that dominated for decades:

```
text ──► Frontend ──► linguistic features ──► Acoustic model ──► acoustic features ──► Vocoder ──► waveform
        (normalize,                            (predicts mel/                          (mel → wav)
         G2P, prosody)                          F0/duration)
```

- **Frontend:** text normalization ("$5.2M in 2026" → words), grapheme-to-phoneme (G2P), phrasing, stress, prosodic annotation.
- **Acoustic model:** linguistic features → acoustic features (mel spectrogram, F0, duration).
- **Vocoder:** acoustic features → waveform.

Modern systems progressively collapse these stages. A rough map of paradigms:

| Paradigm | Alignment | Output space | Representative models |
|---|---|---|---|
| Concatenative | forced alignment + search | waveform units | Festival unit selection |
| Statistical parametric | HMM states | vocoder params | HTS, Merlin |
| AR seq2seq | learned attention | mel frames | Tacotron 2, Transformer TTS |
| NAR duration-based | external durations | mel frames | FastSpeech 2, FastPitch |
| Flow / VAE end-to-end | monotonic alignment search | latent → waveform | Glow-TTS, VITS |
| Diffusion / flow matching | durations or implicit | mel / latents | Grad-TTS, F5-TTS, Voicebox |
| Codec language models | implicit (AR over tokens) | discrete audio tokens | VALL-E, CosyVoice, Llasa, Qwen3-TTS |

---

## 2. The Classical Era

You need this era for three reasons: its vocabulary (formants, diphones, F0, alignment costs) is still the vocabulary of the field, its probabilistic machinery (HMM alignment, maximum likelihood parameter generation) reappears inside modern models, and its failure modes explain why each neural idea was invented.

### 2.1 Formant Synthesis (1960s-1980s)

Speech production is modeled by the **source-filter model**: an excitation signal $e[n]$ (glottal pulses for voiced sounds, noise for unvoiced) passes through the vocal tract, modeled as a linear filter $h[n]$:

$$s[n] = (e * h)[n] \quad\Longleftrightarrow\quad S(z) = E(z)\,H(z)$$

The vocal tract filter is approximated by a cascade or parallel bank of second-order resonators, one per **formant** (resonance frequency $F_k$ with bandwidth $B_k$):

$$H_k(z) = \frac{1}{1 - 2e^{-\pi B_k T}\cos(2\pi F_k T)\,z^{-1} + e^{-2\pi B_k T} z^{-2}}$$

with $T$ the sampling period. Systems like the **Klatt synthesizer** (1980) and DECtalk drove dozens of such parameters with hand-written rules per phoneme and transition. Result: fully intelligible, tiny footprint, unmistakably robotic. Stephen Hawking's voice was this technology.

**Lesson that survives:** speech is excitation plus spectral envelope. LPC vocoders, WORLD, and even the "period" discriminators in HiFi-GAN all exploit this structure.

### 2.2 Concatenative Synthesis and Unit Selection (1990s-2000s)

Instead of generating speech, *reuse* recorded speech.

**Diphone synthesis** stores one example of every phone-to-phone transition (~1500 diphones for English) and concatenates them. To fix prosody, **TD-PSOLA** (pitch-synchronous overlap-add) decomposes the signal into windows centered on glottal epochs $t_i$, then re-spaces and overlap-adds them to change F0 and duration without changing the spectral envelope:

$$\tilde{s}[n] = \sum_i w[n - \tilde{t}_i]\, s_i[n - \tilde{t}_i + t_i]$$

**Unit selection** (Hunt & Black, 1996) scales this up: record hours of speech, cut into units, and at synthesis time search the database for the unit sequence $u_{1:n}$ that best matches target specifications $t_{1:n}$ by minimizing

$$C(t_{1:n}, u_{1:n}) = \sum_{i=1}^{n} C^{t}(t_i, u_i) + \sum_{i=2}^{n} C^{c}(u_{i-1}, u_i)$$

where $C^t$ is the **target cost** (how well unit $u_i$ matches the desired phonetic/prosodic context) and $C^c$ is the **join cost** (spectral/F0/energy discontinuity between consecutive units). The optimal sequence is found with Viterbi search over the unit lattice. This powered Festival, and commercial TTS (Apple, Nuance, Google) into the mid-2010s.

**Why it died:** at its best it is real recorded speech, so extremely natural; but quality collapses when the database lacks a good unit ("join artifacts"), voices cost weeks of studio recording, and there is essentially no control over style. The cost-function-plus-Viterbi structure, however, is the direct ancestor of alignment search in Glow-TTS.

### 2.3 Statistical Parametric Speech Synthesis (HMM-based, 2000s-2015)

**SPSS** (Zen, Tokuda & Black, 2009; the HTS toolkit) replaces the database with a *generative statistical model* of acoustic features.

**Representation.** Each frame is described by vocoder parameters: mel-cepstral coefficients (spectral envelope), $\log F_0$, and aperiodicity, extracted with classical vocoders such as **STRAIGHT** or **WORLD** (F0 by DIO/Harvest, envelope by CheapTrick, aperiodicity by D4C). Synthesis inverts this: parameters → MLSA filter or WORLD synthesis → waveform.

**Model.** For every phoneme *in context* (quinphone identity, position, stress, part of speech, ~50 features), a 5-state left-to-right HMM emits acoustic frames with Gaussian output distributions. Because the context space is astronomically large, states are tied with **decision-tree clustering**. $\log F_0$ is undefined in unvoiced regions, handled by **multi-space distributions (MSD)**. Durations get explicit Gaussian models per state (making it a hidden semi-Markov model).

**The key equation: Maximum Likelihood Parameter Generation (MLPG).** Naively taking each state's mean produces piecewise-constant, discontinuous trajectories. The fix: model each observation as statics plus dynamics,

$$\mathbf{o}_t = \big[\mathbf{c}_t^\top,\ \Delta \mathbf{c}_t^\top,\ \Delta^2 \mathbf{c}_t^\top\big]^\top,\qquad \Delta \mathbf{c}_t = \tfrac{1}{2}(\mathbf{c}_{t+1} - \mathbf{c}_{t-1})$$

so the full observation sequence is a linear function $\mathbf{o} = W\mathbf{c}$ of the static sequence $\mathbf{c}$, where $W$ stacks the delta windows. Given the state sequence $q$ with means $\boldsymbol{\mu}_q$ and covariances $\boldsymbol{\Sigma}_q$, choose $\mathbf{c}$ maximizing $\log \mathcal{N}(W\mathbf{c};\ \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$. Setting the gradient to zero:

$$W^\top \boldsymbol{\Sigma}_q^{-1} W\, \mathbf{c}^{*} = W^\top \boldsymbol{\Sigma}_q^{-1} \boldsymbol{\mu}_q$$

a banded linear system solved in $O(T)$. This yields smooth trajectories consistent with both static and delta statistics. *Too* smooth in fact, so **global variance (GV)** compensation reinflates the variance of generated trajectories.

**DNN-SPSS (2013-2016).** Zen et al. replaced the decision tree + Gaussians with a DNN (then LSTM) mapping frame-level linguistic features directly to acoustic features. The Merlin toolkit is the reference implementation. Same vocoder, same oversmoothing problem, better regression.

**Why it lost:** MSE regression to vocoder parameters averages over the one-to-many mapping (buzzy, muffled), and classical vocoders bound the ceiling. **Why it matters:** HMM forced alignment still produces the durations that train FastSpeech-style models; MLPG is the conceptual ancestor of every "predict a smooth trajectory under a probabilistic model" method; and the frontend engineering (contexts, G2P, normalization) transferred wholesale into neural systems.

---

## 3. Signal Processing You Must Know

**Sampling.** Speech models operate at 16, 22.05, 24, 44.1 or 48 kHz. Nyquist: content up to $f_s/2$.

**STFT.** With window $w$ of length $N$ and hop $H$:

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N}$$

**Mel scale.** Perceptually motivated frequency warping:

$$\mathrm{mel}(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

**Mel spectrogram.** Apply a bank of $B$ triangular mel filters $H_b[k]$ to the (power) spectrogram, then compress:

$$M[m, b] = \log\left( \sum_k H_b[k]\, \lvert X[m,k]\rvert^2 \right)$$

Typical acoustic-model config: 22.05/24 kHz audio, $N = 1024$, $H = 256$ (~11.6 ms per frame), $B = 80$ to $100$ mel bins. The mel spectrogram is the lingua franca between acoustic models and vocoders because it is compact (~86 frames/s × 80 dims vs 24,000 samples/s) and perceptually aligned. It also *discards phase*, which is exactly why the vocoder is a hard generative problem and not just an inverse transform.

**Griffin-Lim** recovers a plausible phase by iterating projections between the magnitude constraint and the set of consistent STFTs:

$$x^{(i+1)} = \mathrm{ISTFT}\!\left( |X| \cdot e^{\,j \angle \mathrm{STFT}(x^{(i)})} \right)$$

It was Tacotron 1's vocoder and remains a useful debugging baseline (metallic but intelligible).

**F0 / pitch:** the fundamental frequency of voiced speech (~60-300 Hz adult speech); harmonics at integer multiples. Prosody is largely carried by F0 contours, energy, and durations, which is why FastSpeech 2 predicts exactly these.

---

## 4. The Neural Revolution (2016-2020)

### 4.1 WaveNet (2016): Autoregression on Raw Audio

**WaveNet** (van den Oord et al., arXiv:1609.03499) discards the source-filter assumption entirely and models the waveform sample-by-sample:

$$p(y) = \prod_{t=1}^{T} p(y_t \mid y_1, \dots, y_{t-1},\ c)$$

Each conditional is a categorical distribution over 256 amplitude levels after **mu-law companding**:

$$f(y_t) = \mathrm{sign}(y_t)\,\frac{\ln(1 + \mu |y_t|)}{\ln(1 + \mu)}, \qquad \mu = 255$$

The network is a stack of **dilated causal convolutions** whose dilation doubles per layer (1, 2, 4, ..., 512, repeated), giving an exponentially large receptive field at linear cost, with the gated residual unit

$$\mathbf{z} = \tanh(W_{f} * \mathbf{x}) \odot \sigma(W_{g} * \mathbf{x})$$

plus residual and skip connections. Conditioning $c$ enters every layer: *global* (speaker embedding) and *local* (upsampled linguistic features, later mel spectrograms).

WaveNet produced the first clearly human-like neural speech, but sampling was autoregressive at audio rate (originally far slower than real time). **Parallel WaveNet** (probability density distillation into an IAF student) fixed speed at Google-scale cost. The lasting legacy: raw-audio likelihood modeling is possible, and mel-conditioned WaveNet defined the modern "neural vocoder" role.

### 4.2 Tacotron and Tacotron 2 (2017): Attention Seq2seq to Mel

**Tacotron 2** (Shen et al., arXiv:1712.05884) is the canonical attention-based acoustic model:

- **Encoder:** character embeddings → 3 conv layers → bidirectional LSTM → memory $h_{1:S}$.
- **Attention (location-sensitive):** at decoder step $t$, energies use content *and* the previous alignment $\alpha_{t-1}$ (via a conv $f_{t} = F * \alpha_{t-1}$), which encourages monotonic forward movement:

$$e_{t,s} = \mathbf{w}^\top \tanh\!\big(W \mathbf{s}_t + V \mathbf{h}_s + U \mathbf{f}_{t,s} + \mathbf{b}\big), \qquad \alpha_t = \mathrm{softmax}(e_t), \qquad \mathbf{c}_t = \sum_s \alpha_{t,s}\mathbf{h}_s$$

- **Decoder:** an autoregressive 2-layer LSTM consumes the previous mel frame through a heavily-dropped-out **pre-net** (the information bottleneck that forces reliance on attention), predicts the next mel frame(s) plus a **stop token**; a 5-layer conv **post-net** adds a residual refinement.
- **Loss:** MSE on mels before and after post-net + BCE on the stop token.

With a mel-conditioned WaveNet vocoder, Tacotron 2 reached MOS ≈ 4.53 vs ≈ 4.58 for ground truth on its single-speaker domain: the "human parity in-domain" moment for neural TTS.

**Failure modes you will still encounter** when running AR attention models: skipped or repeated words, attention collapse on long or out-of-domain inputs, babbling past the end. A whole subfield produced fixes (forward attention, GMM attention, dynamic convolution attention), but the real fix was removing learned soft attention altogether, which leads to:

### 4.3 Transformer TTS (2019)

Replace the LSTMs with Transformer encoder-decoder blocks (Li et al., arXiv:1809.08895): parallel *training*, better long-range context. Inference is still autoregressive and attention is still fragile. Its importance is mostly that it made the next step obvious.

### 4.4 FastSpeech 1/2 (2019-2020): Non-Autoregressive, Duration-Based

**FastSpeech** (Ren et al., arXiv:1905.09263) removes both autoregression and learned attention. The core device is the **length regulator**: predict an integer duration $d_i$ per input token, then repeat each encoder state $d_i$ times to reach frame length, and decode all mel frames *in parallel* with a Transformer stack.

**FastSpeech 2** (arXiv:2006.04558) trains directly on ground-truth targets and adds a **variance adaptor**: predictors for duration, pitch, and energy inserted between encoder and decoder. Ground-truth durations come from forced alignment (in practice: the **Montreal Forced Aligner**, a triphone HMM-GMM system, i.e. the SPSS machinery reused). Losses:

$$\mathcal{L} = \lVert M - \hat{M}\rVert_1 + \lVert \log \mathbf{d} - \log \hat{\mathbf{d}} \rVert_2^2 + \lVert \mathbf{p} - \hat{\mathbf{p}} \rVert_2^2 + \lVert \mathbf{e} - \hat{\mathbf{e}} \rVert_2^2$$

Properties that made it the industry workhorse (with FastPitch as a close sibling): orders-of-magnitude faster mel generation than AR decoding, zero attention failures (robust for products), and explicit control knobs (multiply durations for speed, shift pitch, etc.). Weakness: MSE regression brings back mild oversmoothing, so prosody is flatter than a well-behaved AR model. Fixing *that* properly requires real generative modeling over mels, which is where flows, VAEs, GANs, and diffusion enter.

---

## 5. Neural Vocoders

The vocoder problem: given a mel spectrogram $M$ (no phase, compressed frequency), generate a waveform $\hat{y}$ perceptually indistinguishable from real speech. Every generative-model family has been tried; GANs won on the speed/quality frontier.

### 5.1 Autoregressive

- **WaveNet / WaveRNN** (arXiv:1802.08435): WaveRNN replaces deep convs with a single GRU plus dual softmax (coarse/fine 8 bits each), plus sparsification and batched "subscale" tricks for mobile-speed synthesis.
- **LPCNet:** hybridizes DSP and DL: the network only predicts the LPC *residual* while linear prediction handles the spectral envelope. Still relevant for ultra-low-power devices.

### 5.2 Flow-Based: Exact Likelihood

A **normalizing flow** transforms noise $z \sim \mathcal{N}(0, I)$ through an invertible network $y = f(z)$, with exact log-likelihood via change of variables:

$$\log p_Y(y) = \log p_Z\!\big(f^{-1}(y)\big) + \log \left| \det \frac{\partial f^{-1}(y)}{\partial y} \right|$$

Tractability comes from **affine coupling layers** (RealNVP/Glow): split $x = [x_a, x_b]$,

$$y_a = x_a, \qquad y_b = s(x_a) \odot x_b + t(x_a), \qquad \log|\det J| = \sum \log |s(x_a)|$$

invertible regardless of how complex $s, t$ are. **WaveGlow** (arXiv:1811.00002) conditions such a flow on mels and trains by maximum likelihood only: stable, one-shot parallel sampling, but parameter-hungry and since surpassed.

### 5.3 GAN Vocoders: HiFi-GAN in Depth

**MelGAN** (arXiv:1910.06711) proved adversarial vocoding works: a fully convolutional generator, multi-scale discriminators, and a **feature matching** loss. **Parallel WaveGAN** (arXiv:1910.11480) added a multi-resolution STFT auxiliary loss (spectral convergence + log-magnitude terms) that dramatically stabilizes adversarial vocoder training.

**HiFi-GAN** (Kong et al., arXiv:2010.05646) is the design that stuck; most systems from VITS to XTTS still ship a HiFi-GAN-family decoder:

- **Generator:** transposed convolutions upsample the mel (e.g. 256×) with **Multi-Receptive-Field fusion** blocks (parallel residual conv stacks with different kernel/dilation).
- **Multi-Period Discriminator (MPD):** reshapes the 1D waveform into 2D with periods $p \in \{2,3,5,7,11\}$ and applies 2D convs, so each sub-discriminator sees a distinct periodic structure of speech (the source-filter idea reappearing as an inductive bias).
- **Multi-Scale Discriminator (MSD):** operates on the raw, ×2 and ×4 average-pooled waveform for long-range structure.

With LSGAN objectives, for each sub-discriminator $D_k$:

$$\mathcal{L}_D = \mathbb{E}\big[(D_k(y) - 1)^2\big] + \mathbb{E}\big[D_k(G(M))^2\big], \qquad \mathcal{L}_G^{adv} = \mathbb{E}\big[(D_k(G(M)) - 1)^2\big]$$

plus feature matching over discriminator layers $D_k^{(i)}$ and a mel reconstruction loss with $\phi$ the mel transform:

$$\mathcal{L}_{FM} = \mathbb{E} \sum_{i} \frac{1}{N_i} \big\lVert D_k^{(i)}(y) - D_k^{(i)}(G(M)) \big\rVert_1, \qquad \mathcal{L}_{mel} = \big\lVert \phi(y) - \phi(G(M)) \big\rVert_1$$

$$\mathcal{L}_G = \mathcal{L}_G^{adv} + \lambda_{fm}\mathcal{L}_{FM} + \lambda_{mel}\mathcal{L}_{mel} \quad (\lambda_{fm} = 2,\ \lambda_{mel} = 45)$$

### 5.4 Diffusion Vocoders

**DiffWave** (arXiv:2009.09761) and **WaveGrad** (arXiv:2009.00713) apply denoising diffusion (math in §7) to mel-conditioned waveform generation. Excellent quality, but iterative sampling makes them slower than GANs for the same quality, so they remained a research branch for vocoding while diffusion won elsewhere (acoustic modeling).

### 5.5 Modern Vocoders

- **BigVGAN** (NVIDIA, arXiv:2206.04658): HiFi-GAN plus the periodic **snake activation** $f(x) = x + \frac{1}{\alpha}\sin^2(\alpha x)$ and anti-aliased up/downsampling: the current default "universal" vocoder, robust far out of domain (singing, unseen languages) at up to 44.1 kHz.
- **Vocos** (arXiv:2306.00814): predicts STFT magnitude and phase and inverts with ISTFT instead of transposed convs in time domain: an order of magnitude faster, quality on par; also widely used to decode EnCodec tokens.
- **UnivNet, iSTFTNet:** the light/fast branch; iSTFTNet is inside Kokoro.

**Practical guidance:** for a mel-based stack in 2026, BigVGAN-v2 checkpoints or Vocos are the defaults; train your acoustic model on exactly the mel parameters your vocoder expects (mismatched `hop/win/fmax` is the classic silent quality bug).

---

## 6. Flows and VITS

### 6.1 Glow-TTS (2020): Flows + Monotonic Alignment Search

**Glow-TTS** (Kim et al., arXiv:2005.11129) asks: can we get FastSpeech's robustness *without* an external aligner? It models mels with a normalizing-flow decoder $z = f^{-1}(M)$ and a text-conditioned Gaussian prior whose statistics $(\mu_i, \sigma_i)$ come from the text encoder, connected by an *unknown monotonic alignment* $A$ mapping each frame $j$ to a token $i = A(j)$:

$$\log p(M \mid x) = \log \mathcal{N}\big(f^{-1}(M);\ \mu_{A}, \sigma_{A}\big) + \log\left|\det \frac{\partial f^{-1}(M)}{\partial M}\right|$$

Training alternates: given the current model, find the best monotonic alignment with **Monotonic Alignment Search (MAS)**, a Viterbi-style dynamic program

$$Q_{i,j} = \log \mathcal{N}\big(z_j;\ \mu_i, \sigma_i\big) + \max\big(Q_{i-1,\,j-1},\ Q_{i,\,j-1}\big)$$

then maximize likelihood under that alignment, and train a duration predictor on the alignment's token durations (stop-gradient). At inference: predict durations, expand the prior, sample $z$, run the flow forward. No attention, no external aligner, fully parallel, and a proper likelihood model. This is unit selection's Viterbi cost lattice reborn inside a deep generative model.

### 6.2 VITS (2021): The One-Stage End-to-End Recipe

**VITS** (Kim, Kong & Son, arXiv:2106.06103) fuses a conditional VAE, normalizing flows, MAS, and adversarial training into a single model trained *text → waveform* end to end. It was the open-source state of the art for roughly three years and still powers Piper, MMS, and countless products.

**Structure.** A posterior encoder maps the *linear* spectrogram to latents $z \sim q(z \mid y)$; a HiFi-GAN-style decoder reconstructs the waveform from $z$; the prior $p(z \mid x)$ is the text encoder's Gaussians made expressive by a normalizing flow $f_\theta$, aligned to frames via MAS run in latent space. Training maximizes the conditional ELBO

$$\log p(y \mid x)\ \ge\ \mathbb{E}_{q(z \mid y)}\big[\log p(y \mid z)\big] - D_{KL}\big(q(z \mid y)\ \Vert\ p(z \mid x)\big)$$

where the reconstruction term is realized as an L1 mel loss on the decoded waveform, and the whole thing is additionally trained adversarially (MPD + MSD discriminators, feature matching) exactly as in HiFi-GAN. A **stochastic duration predictor** (a small flow trained with a variational bound and dequantization) lets the model sample varied rhythms instead of one deterministic tempo.

**Why VITS mattered:** (1) removed the acoustic-model/vocoder seam and its error compounding (training the decoder on *real* latents rather than predicted mels); (2) genuinely multimodal outputs via latent sampling + stochastic durations; (3) a complete, reproducible recipe. **VITS2** refined the duration model and flows; **YourTTS** (arXiv:2112.02418) extended VITS to zero-shot multi-speaker and cross-lingual TTS with speaker encoders, the bridge toward the zero-shot era.

---

## 7. Diffusion and Flow Matching

This is the mathematical core of most modern non-autoregressive TTS. Learn it once, properly.

### 7.1 Denoising Diffusion (DDPM)

Fix a variance schedule $\beta_1, \dots, \beta_T$, define $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s\le t}\alpha_s$. The forward process gradually destroys data $x_0$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big) \quad\Rightarrow\quad q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t) I\big)$$

A network $\epsilon_\theta$ learns to reverse it. The ELBO reduces (Ho et al., arXiv:2006.11239) to the famous simple loss

$$\mathcal{L}_{simple} = \mathbb{E}_{t,\,x_0,\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t,\ c\big)\big\rVert^2\Big]$$

and sampling runs the learned reverse chain $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$ with

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\Big)$$

**Score/SDE view** (Song et al., arXiv:2011.13456): the forward process is an SDE $dx = f(x,t)\,dt + g(t)\,dw$; its reverse is

$$dx = \big[f(x,t) - g(t)^2\, \nabla_x \log p_t(x)\big]\,dt + g(t)\,d\bar{w}$$

and learning $\epsilon_\theta$ is equivalent to learning the **score** $\nabla_x \log p_t(x)$, since $\epsilon_\theta(x_t,t) = -\sqrt{1-\bar\alpha_t}\; s_\theta(x_t,t)$. There is also a deterministic probability-flow ODE with the same marginals, which is what fast samplers integrate.

### 7.2 Diffusion in TTS

- **Grad-TTS** (Popov et al., arXiv:2105.06337): a text encoder + MAS (from Glow-TTS) produce an aligned mean $\mu$; the forward SDE is *mean-reverting toward* $\mu$, $dX_t = \tfrac{1}{2}(\mu - X_t)\beta_t\,dt + \sqrt{\beta_t}\,dW_t$, so generation starts from $\mathcal{N}(\mu, I)$, already "roughly correct", and the reverse SDE refines it into a sharp mel. Solved oversmoothing elegantly.
- **NaturalSpeech 2** (arXiv:2304.09116): latent diffusion over *continuous* codec latents with pitch/duration conditioning: strong zero-shot TTS without discrete-token error accumulation.
- The cost: tens to hundreds of network evaluations (NFE). Distillation (progressive, consistency models, CoMoSpeech) attacks this, but the cleaner fix turned out to be changing the objective:

### 7.3 Flow Matching (the current default continuous method)

A **continuous normalizing flow** transports noise to data via an ODE $\frac{dx}{dt} = v_\theta(x, t)$. **Flow Matching** (Lipman et al., arXiv:2210.02747) trains $v_\theta$ by regression onto conditional target velocities, no simulation needed. With the optimal-transport path between a noise sample $x_0 \sim \mathcal{N}(0,I)$ and a data sample $x_1$:

$$x_t = \big(1 - (1-\sigma_{min})\,t\big)\,x_0 + t\,x_1, \qquad u_t(x_t \mid x_1) = x_1 - (1-\sigma_{min})\,x_0$$

$$\mathcal{L}_{CFM} = \mathbb{E}_{t,\,x_0,\,x_1}\Big[\big\lVert v_\theta(x_t, t, c) - u_t(x_t \mid x_1)\big\rVert^2\Big]$$

Because the target paths are *straight lines*, the learned ODE is nearly straight too, so an Euler/midpoint solver needs only ~10-32 steps for high quality (vs. hundreds for vanilla DDPM). Classifier-free guidance transfers directly: $\tilde{v} = v_\theta(x,t,\varnothing) + w\,\big(v_\theta(x,t,c) - v_\theta(x,t,\varnothing)\big)$.

### 7.4 The Flow-Matching TTS Family

- **Matcha-TTS** (arXiv:2309.03199): Grad-TTS's skeleton (encoder + MAS + duration predictor) with the decoder swapped for conditional flow matching: small, fast, excellent single/multi-speaker quality; the best "trainable on one GPU" recipe of its generation.
- **Voicebox** (Meta, arXiv:2306.15687): frames TTS as **speech infilling**: given surrounding audio context and a forced-aligned phoneme sequence, fill in masked mel frames with FM; a separate regression model predicts durations. One model does zero-shot TTS, noise editing, and cross-lingual style transfer.
- **E2-TTS** (Microsoft, arXiv:2406.18009): "embarrassingly easy": drop phonemes, drop the duration model, drop explicit alignment. Input is just the character sequence padded with filler tokens to the target mel length; a big Transformer + FM learns alignment *implicitly*. Total duration is the only knob.
- **F5-TTS** (arXiv:2410.06885): E2's paradigm made practical and open: ConvNeXt blocks preprocess the text stream, a DiT backbone does the infilling, and **sway sampling** re-warps the ODE time grid toward the early (structure-forming) steps, $t' = t + s\big(\cos(\tfrac{\pi}{2}t) - 1 + t\big)$ with $s < 0$, improving quality at low NFE. Trained on ~100k hours (Emilia), fully open, and the base of many community fine-tunes (including Arabic ones).

Flow matching is also the standard *second stage* in modern LM-based systems (CosyVoice's chunk-aware FM, MegaTTS 3, Voxtral TTS): the LM decides *what and when*, FM renders *how it sounds*.

---

## 8. Neural Audio Codecs

The bridge to the LLM era: turn continuous audio into a short sequence of discrete tokens that a language model can predict, and back.

### 8.1 VQ-VAE and Residual Vector Quantization

**VQ-VAE** (van den Oord et al., arXiv:1711.00937): an encoder produces $z_e(x)$, quantized to the nearest entry of a learned codebook $\{e_k\}_{k=1}^{K}$:

$$z_q(x) = e_{k^*}, \qquad k^* = \arg\min_k \lVert z_e(x) - e_k \rVert_2$$

trained with a straight-through estimator and the loss

$$\mathcal{L} = \underbrace{\lVert x - \hat{x}\rVert^2}_{\text{reconstruction}} + \underbrace{\lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2}_{\text{codebook}} + \beta \underbrace{\lVert z_e(x) - \mathrm{sg}[e] \rVert_2^2}_{\text{commitment}}$$

One codebook cannot capture full-fidelity audio, so codecs use **Residual VQ (RVQ)**: quantize, subtract, quantize the residual, repeat over $C$ codebooks:

$$r_0 = z, \qquad q_c = Q_c(r_{c-1}), \qquad r_c = r_{c-1} - q_c, \qquad \hat{z} = \sum_{c=1}^{C} q_c$$

Codebook 1 carries coarse content; later codebooks add fine acoustic detail: a hierarchy that every LM-TTS decoding scheme exploits. **Quantizer dropout** during training makes one model serve multiple bitrates. **FSQ** (finite scalar quantization) replaces learned codebooks entirely: bound each latent dim with $\tanh$ and round it to $L$ levels, giving an implicit codebook of size $\prod_i L_i$ with no codebook-collapse pathologies (adopted by CosyVoice 2's tokenizer).

### 8.2 The Codec Lineage

| Codec | Year | Frame rate | Structure | Notes |
|---|---|---|---|---|
| SoundStream (arXiv:2107.03312) | 2021 | 75 Hz | RVQ | first fully E2E neural codec, 3-18 kbps |
| EnCodec (arXiv:2210.13438) | 2022 | 75 Hz | RVQ | Meta; adversarial + multi-scale spectral losses; streaming |
| DAC (arXiv:2306.06546) | 2023 | 86 Hz | RVQ ×9 | snake activations, ~8 kbps at 44.1 kHz, big fidelity jump |
| Mimi (Kyutai, Moshi report) | 2024 | 12.5 Hz | split RVQ | first level distilled from a self-supervised model (semantic); built for speech LMs; used by Moshi and Sesame CSM |
| WavTokenizer (arXiv:2408.16532) | 2024 | 75/40 Hz | single VQ | one codebook, LM-friendly |
| X-codec2 (Llasa) | 2025 | 50 Hz | single VQ, 65k codes | fuses semantic + acoustic in one stream |
| S3 / FSQ tokenizer (CosyVoice) | 2024-25 | 25 Hz | supervised | tokens from an ASR-encoder trained with multi-task supervision |
| Qwen3-TTS tokenizer | 2026 | 12 Hz | multi-codebook | ultra-low frame rate for cheap LM decoding |

### 8.3 Semantic vs. Acoustic Tokens

- **Semantic tokens** (k-means over HuBERT / w2v-BERT features, or supervised ASR-encoder tokens): capture *content and coarse prosody*, discard speaker identity and fine acoustics; low rate; easy for an LM to predict.
- **Acoustic tokens** (codec RVQ codes): reconstruct the actual audio; high rate; harder to model.

Token-rate arithmetic drives everything: EnCodec at 75 Hz × 8 codebooks = 600 tokens per second of speech, vs. Mimi ≈ 100/s, X-codec2 = 50/s, Qwen3-TTS at 12 Hz. Sequence length is what an LM pays for, so the whole codec trend of 2024-2026 is *lower frame rates with semantics folded in*, letting a single AR stream stay short. Two-stage designs (LM predicts semantic tokens, a flow-matching or NAR model renders acoustics) are the other route to the same goal.

---

## 9. LLM-Style TTS and the Modern Scene

### 9.1 AudioLM and VALL-E: TTS as Language Modeling

**AudioLM** (Google, arXiv:2209.03143) showed that a decoder-only Transformer over audio tokens generates coherent speech continuations: stage 1 models semantic tokens, stage 2 maps semantic → coarse acoustic tokens, stage 3 adds fine codebooks.

**VALL-E** (Microsoft, arXiv:2301.02111) turned this into *zero-shot TTS as in-context learning*. Input: phonemes $x$ plus a 3-second enrollment prompt's EnCodec codes $\tilde{C}$. An **AR model** predicts the first codebook:

$$p\big(c_{t,1} \,\big|\, x,\ \tilde{C},\ c_{<t,1}\big)$$

and a **NAR model** fills codebooks $2 \dots 8$ in parallel, one level at a time, conditioned on all previous levels:

$$p\big(c_{:,j} \,\big|\, x,\ \tilde{C},\ c_{:,<j}\big), \qquad j = 2, \dots, 8$$

Cloning a voice becomes prompting, no fine-tuning, and quality scales with data (60k hours LibriLight). **VALL-E 2** (arXiv:2406.05370) added repetition-aware sampling and grouped code modeling and claimed human parity on LibriSpeech/VCTK benchmarks. The lineage's chronic weaknesses, hallucination, repetition and instability inherited from AR sampling, defined the next two years of research.

### 9.2 Decoding Patterns for Token Stacks

How do you model $C$ codebooks per frame with one sequence model?

- **Flatten:** interleave all codebooks into one stream. Exact but $C\times$ longer sequences.
- **Delay pattern** (MusicGen): predict codebook $k$ at time $t$ during step $t + k$, so one AR pass handles the RVQ hierarchy with only a $C$-step lag. Used by Parler-TTS, Dia.
- **Coarse-first two-stage:** AR over semantic/first-level tokens, then a separate model for the rest (AudioLM, VALL-E, SPEAR-TTS arXiv:2302.03540).
- **MaskGIT-style parallel decoding** (**SoundStorm**, arXiv:2305.09636): start fully masked, iteratively unmask the highest-confidence positions with a cosine schedule; a few parallel iterations per RVQ level generate 30 s of audio in ~0.5 s. **MaskGCT** (arXiv:2409.00750) builds a fully non-autoregressive text → semantic → acoustic TTS on this idea, no duration model, with explicit total-length control.
- **Depth transformer:** a large temporal backbone predicts a per-frame embedding, a small transformer expands it across codebooks *within* the frame (RQ-Transformer; used by Moshi and **Sesame CSM**).

### 9.3 Tortoise, Bark, XTTS: The Bridge Generation

- **Tortoise-TTS** (Betker, arXiv:2305.07243): AR Transformer over discrete mel-VQ tokens + a diffusion decoder to mels + CLVP contrastive reranking of candidates. Slow by design, but its "scale AR + rerank + diffusion-decode" blueprint shaped everything after; XTTS is its production descendant.
- **Bark** (Suno, 2023): GPT stacks for text → semantic → coarse → fine EnCodec tokens; "unconstrained" generation including laughter, music-ish sounds, and multilingual speech, at the cost of controllability.
- **XTTS v2** (Coqui): GPT backbone + HiFi-GAN-style decoder, 3-second voice cloning across 17 languages *including Arabic*, easily fine-tunable. Coqui the company shut down (Jan 2024), but the maintained fork (`idiap/coqui-ai-TTS`) keeps it alive as the most practical "clone + fine-tune tonight" model.

### 9.4 The Open Scene, 2023 → 2026

Selected open-weight systems, in rough chronological order. Always re-check licenses on the model card; several projects changed terms between versions.

| Model | Year | Params | Paradigm | Notes |
|---|---|---|---|---|
| StyleTTS 2 (arXiv:2306.07691) | 2023 | ~150M | style diffusion + GAN, E2E | adversarial training against WavLM (SLM) discriminators; human-level LJSpeech; base of Kokoro |
| MetaVoice, Parler-TTS (arXiv:2402.01912) | 2024 | 1B / 0.6-0.9B | AR codec LM | Parler: control voice via natural-language *descriptions* |
| CosyVoice 1/2 (arXiv:2407.05407, 2412.10117) | 2024 | 0.5B | AR LM over supervised semantic tokens + chunk-aware flow matching | v2: Qwen2.5-0.5B backbone, FSQ tokens, unified streaming/non-streaming, ~150 ms latency |
| Fish Speech (arXiv:2411.01156) | 2024 | ~0.5B | dual-AR codec LM | firefly-GAN codec; → OpenAudio S1 (2025), → Fish Audio S2 (2026) |
| MaskGCT (arXiv:2409.00750) | 2024 | 1B | fully NAR masked | in Amphion; Emilia-trained |
| F5-TTS (arXiv:2410.06885) | 2024 | 0.34B | flow-matching infilling | see §7.4; huge fine-tune ecosystem |
| Kokoro-82M | 2025 | 82M | StyleTTS2-lineage + iSTFTNet | Apache 2.0, CPU-fast, absurd quality-per-parameter; still a top pick in 2026 |
| Llasa (arXiv:2502.04128) | 2025 | 1B/3B/8B | single-stream AR (Llama init) over X-codec2 | the cleanest "TTS is literally an LLM" result; scaling laws for train and inference compute |
| Zonos-v0.1 | 2025 | 1.6B | AR (transformer & SSM hybrid variants) + DAC | Zyphra; cloning + emotion conditioning |
| Spark-TTS (arXiv:2503.01710) | 2025 | 0.5B | AR over BiCodec (semantic + global speaker tokens) | attribute-controllable (pitch, rate) |
| Sesame CSM-1B | 2025 | 1B + depth decoder | AR over Mimi tokens | conversational, context-conditioned; Apache 2.0 |
| Orpheus-3B | 2025 | 3B | Llama-3.2 over SNAC tokens | emotive tags (`<laugh>` etc.), streaming |
| Dia-1.6B | 2025 | 1.6B | AR codec LM (delay pattern) | dialogue-native: `[S1]/[S2]` turns, nonverbals |
| Chatterbox / Chatterbox Multilingual | 2025 | 0.5B | Llama-backbone codec LM | MIT; emotion-exaggeration knob; PerTh watermarking built in; 23 languages incl. Arabic |
| MegaTTS 3 | 2025 | 0.45B | sparse-aligned latent DiT | ByteDance; high-quality cloning |
| IndexTTS2 (arXiv:2506.21619) → 2.5 | 2025-26 | ~1.5B | AR + explicit duration tokens | precise duration control for dubbing; timbre/emotion disentangled |
| Higgs Audio v2 | 2025 | 3B (Llama-3.2) | unified tokenizer, DualFFN adapter | trained on ~10M hours; strong expressiveness |
| VibeVoice (Microsoft) | 2025 | 1.5B/7B | continuous 7.5 Hz latents + next-token diffusion head | long-form: up to ~90 min, 4 speakers |
| Kyutai TTS | 2025 | 1.6B | delayed-streams AR (text and audio streams offset) | true word-level streaming in and out |
| VoxCPM (arXiv:2509.24650) | 2025 | 0.5B | tokenizer-free (continuous) | context-aware generation, no discrete codec |
| NeuTTS Air, Kitten TTS, PocketTTS, Supertonic | 2025-26 | 15M-750M | various | the on-device wave: CPU-first, streaming |
| Qwen3-TTS (open release) | 2026 | 0.6B/1.7B | AR over 12 Hz multi-codebook tokens | Base (cloning), CustomVoice (preset speakers + style prompts), VoiceDesign (create voices from text descriptions); DPO/GSPO-aligned; 10 languages |
| Fun-CosyVoice 3 (arXiv:2505.17589) | 2025-26 | 0.5B-1.5B | AR LM + FM | 1M training hours, multi-task supervised tokenizer, differentiable-reward post-training; 9 languages + 18 Chinese dialects; 0.5B weights Apache 2.0 |
| Voxtral TTS (Mistral) | 2026 | 3.4B + 0.39B FM + codec | semantic AR + flow matching | runs on a single 16 GB GPU via vLLM; strong arena Elo |
| Fish Audio S2 (arXiv:2603.08823) | 2026 | 4B + 0.4B dual-AR | 10-codebook RVQ | top-tier expressiveness; weights under a research (non-commercial) license |
| Step Audio EditX, NVIDIA Magpie, Maya1 | 2025-26 | 1-3B | AR codec LMs | editing-focused / production / expressive niches |

Closed but influential references: ElevenLabs v3, OpenAI `gpt-4o-mini-tts` and the Realtime API, Google Gemini native-audio TTS and NotebookLM, ByteDance Seed-TTS (arXiv:2406.02430, the paper everyone benchmarks against), MiniMax Speech, Cartesia Sonic (SSM-based low latency), Hume Octave, Qwen-Audio-3.0-TTS (API-only flagship above the open Qwen3-TTS line).

### 9.5 Where the Field Is (Trends as of Mid-2026)

1. **The LM + flow-matching split is the dominant architecture:** an AR (or masked) LM over low-rate semantic-ish tokens for content/prosody decisions, plus a flow-matching acoustic renderer. CosyVoice, Voxtral, MegaTTS 3, and many closed systems share this shape.
2. **Single-stream simplification is the counter-trend:** Llasa-style "one Llama, one codebook" and tokenizer-free VoxCPM argue the pipeline can collapse further.
3. **Small got shockingly good:** 82M-750M models (Kokoro, PocketTTS, NeuTTS, Supertonic, Qwen3-TTS-0.6B) now beat multi-billion-parameter 2024 systems in blind tests; on-device and CPU real-time is mainstream.
4. **Post-training arrived:** DPO/GSPO/RL with rewards built from WER, speaker similarity, and MOS predictors (Seed-TTS RL, CosyVoice 3's differentiable reward, Koel-TTS, Qwen3-TTS alignment) now separates top systems more than architecture does.
5. **Instruction-following TTS:** voice design from natural-language descriptions (Parler, Qwen3-TTS VoiceDesign), style/emotion instructions inline, and fine-grained editing (Step Audio EditX).
6. **Conversation-native and full-duplex:** context-conditioned generation (CSM), dialogue models (Dia), streaming-text streaming-audio (Kyutai), and speech folded into omni-LLMs (GPT-4o, Gemini, Qwen-Omni, Moshi).
7. **Safety became a feature:** built-in watermarking (Chatterbox's PerTh, AudioSeal), anti-spoofing benchmarks (ASVspoof), and consent-gated cloning, under real deepfake regulation pressure.
8. **Evaluation shifted** from MOS papers to arenas (TTS Arena, Artificial Analysis Speech Arena Elo) and standardized zero-shot suites (seed-tts-eval, CV3-Eval).

---

## 10. Evaluation

**Subjective (still the gold standard):**
- **MOS**: naturalness rated 1-5 by panels; report with 95% CI. **CMOS**: side-by-side comparison against a baseline (more sensitive). **SMOS**: speaker-similarity MOS. **MUSHRA**: multi-stimulus with anchors. Beware: MOS is not comparable across papers, listener pools, or years, and top systems have saturated it.

**Objective (the modern zero-shot triad):**
- **WER/CER**: transcribe the synthesized audio with a strong ASR (typically Whisper-large-v3 or Paraformer for zh) and compare with the input text: measures intelligibility and hallucination/robustness.
- **SIM**: cosine similarity between speaker embeddings (WavLM-based ECAPA/x-vector models) of the prompt and generated audio: measures cloning fidelity.
- **MOS predictors**: UTMOS(v2), DNSMOS, NISQA estimate perceptual quality without listeners.

**Signal-level (useful for regression tests, not rankings):**

$$\mathrm{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{d=1}^{D} \big(c_d - \hat{c}_d\big)^2}$$

(mel-cepstral distortion, dB, over DTW-aligned frames), F0 RMSE, voiced/unvoiced error rate, log-F0 correlation.

**Systems metrics:** RTF (real-time factor), first-packet latency for streaming, GPU memory.

**Benchmarks worth knowing:** seed-tts-eval (EN/ZH zero-shot WER + SIM), LibriSpeech test-clean continuations (the VALL-E setup), CV3-Eval (multilingual, in-the-wild), EmergentTTS-Eval (emotions, paralinguistics, complex text), and the public arenas. A model report without WER + SIM + a listening test is not a serious TTS report in 2026.

---

## 11. Arabic TTS Notes

A short section, because Arabic is where generic TTS knowledge meets hard frontend reality:

- **Diacritization (التشكيل) is the core problem.** Undiacritized MSA text underspecifies pronunciation (short vowels, case endings, gemination). A production Arabic pipeline is effectively: text normalization → automatic diacritization → near-regular G2P → the acoustic stack. Diacritizer errors become pronunciation errors downstream, so benchmark your diacritizer (CATT and similar neural diacritizers) as seriously as your TTS model.
- **Phonology that stresses models:** emphatics (ص ض ط ظ) and their vowel coloring, gemination (الشدة) as a duration phenomenon, long vs. short vowel contrasts, hamza handling, and assimilation of ال. Duration-explicit models and duration-controllable LM-TTS (IndexTTS2-style) are genuinely useful here.
- **Dialects have no orthographic standard**, TTS-grade dialectal audio is scarce, and Arabic-English code-switching is everywhere in real inputs. Curated recording + annotation is the actual moat in Arabic speech products; public data is mostly MSA or ASR-grade.
- **Data:** ASC / Arabic Speech Corpus (Nawar Halabi, ~3.7 h studio MSA, the classic), ClArTTS (~12 h Classical Arabic), Arabic Common Voice, and large ASR-grade corpora (MASC, QASR, SADA for Saudi dialects) usable for pretraining or tokenizer training but not as clean TTS targets.
- **Models with Arabic support today:** XTTS v2 (17 languages incl. ar, fine-tunable), MMS-TTS (VITS-based, 1100+ languages, CC-BY-NC), Chatterbox Multilingual (23 languages incl. ar, MIT), OpenAudio/Fish line (check per-version cards), plus many community Arabic fine-tunes of F5-TTS and XTTS on Hugging Face. ArTST (MBZUAI's SpeechT5-style Arabic pretrained model) covers TTS + ASR for MSA and dialects. Note that several top 2026 systems (CosyVoice 3's 9 languages, Qwen3-TTS's 10) do **not** list Arabic, so fine-tuning a strong open base on curated Arabic data remains the winning strategy.

---

## 12. Datasets

| Dataset | Size | Notes |
|---|---|---|
| LJSpeech | 24 h, 1 speaker (EN) | the "MNIST of TTS"; every classic recipe uses it |
| VCTK | 44 h, 109 speakers | multi-speaker English, accents |
| LibriTTS / LibriTTS-R | 585 h | multi-speaker; -R is the restored/cleaned version |
| Hi-Fi TTS | 292 h, 10 speakers | high-quality multi-speaker |
| Expresso | 47 h | expressive styles/emotions, conversational |
| GigaSpeech, MLS, Libriheavy | 10k-60k h | ASR-grade; pretraining scale |
| Emilia (+ Emilia-Large) | 101k → 200k+ h | in-the-wild, 6 languages, the modern pretraining default (F5, MaskGCT, ...) |
| WenetSpeech4TTS | 12k+ h (zh) | TTS-filtered Chinese |
| ASC, ClArTTS, Arabic Common Voice | 3.7 h / 12 h / varies | Arabic (see §11) |

Data licensing and speaker consent are now first-class engineering constraints: check terms before commercial training, and keep provenance records.

---

## 13. Open Courses and Books

**Courses (free, complete):**

1. **Hugging Face Audio Course**: hands-on transformers-for-audio; Unit 6 is TTS (SpeechT5, MMS, Bark, fine-tuning for a new language). The fastest route from zero to fine-tuning. → https://huggingface.co/learn/audio-course (TTS unit: https://huggingface.co/learn/audio-course/chapter6/introduction)
2. **Speech Processing** (Simon King et al., University of Edinburgh, full videos + readings + labs online): signals, phonetics, source-filter, the TTS frontend, and classical waveform generation. → https://speech.zone/courses/speech-processing/
3. **Speech Synthesis** (Edinburgh, the sequel): unit selection in depth → HMM synthesis → neural synthesis and the current state of the art, with a build-a-voice practical. The single best from-first-principles TTS course in existence. → https://speech.zone/courses/speech-synthesis/
4. **Stanford CS224S: Spoken Language Processing**: lectures and assignments spanning ASR, TTS, and dialogue. → https://web.stanford.edu/class/cs224s/
5. **CMU speech courses (11-751 and related)**: materials online; strong on recognition-side foundations (alignment, HMMs) that TTS reuses.

**Books:**

- **Paul Taylor, *Text-to-Speech Synthesis*** (Cambridge UP, 2009): the classical bible; frontend and unit selection chapters remain unmatched.
- **Xu Tan, *Neural Text-to-Speech Synthesis*** (Springer, 2023): the neural era, book-length, by the FastSpeech author.
- **Jurafsky & Martin, *Speech and Language Processing*** (3rd ed. draft, free online): the ASR+TTS chapter is a gentle, current on-ramp. → https://web.stanford.edu/~jurafsky/slp3/

**Surveys:**

- Tan et al., *A Survey on Neural Speech Synthesis* (arXiv:2106.15561): the definitive map of 2016-2021.
- Xie et al., *Towards Controllable Speech Synthesis in the Era of Large Language Models* (EMNLP 2025, arXiv:2412.06602): the map of the LLM/diffusion era, with a continuously-updated companion list: https://github.com/imxtx/awesome-controllable-speech-synthesis
- Zen, Tokuda & Black, *Statistical Parametric Speech Synthesis* (Speech Communication, 2009): the SPSS era from its architects.

---

## 14. Toolkits and Codebases

| Tool | Use it for |
|---|---|
| **ESPnet** (github.com/espnet/espnet) | reproducible recipes: Tacotron 2, FastSpeech 2, VITS, JETS on dozens of corpora |
| **Amphion** (github.com/open-mmlab/Amphion) | modern research recipes: MaskGCT, vocoders, VC; ships Emilia tooling |
| **Coqui TTS fork** (github.com/idiap/coqui-ai-TTS) | XTTS v2 inference and fine-tuning, ready-made pipelines |
| **NeMo** (NVIDIA) | production training: FastPitch, HiFi-GAN, BigVGAN, Magpie |
| **SpeechBrain / PaddleSpeech** | batteries-included alternatives with TTS recipes |
| **Montreal Forced Aligner** | phoneme-level alignments for duration-based models |
| **Matcha-TTS, F5-TTS, CosyVoice, Qwen3-TTS repos** | the reference implementations to *read*: each is a compact lesson in its paradigm |
| **descript-audio-codec, encodec, vocos, BigVGAN repos** | codec and vocoder building blocks |

Track the field via: TTS Arena and Artificial Analysis Speech Arena (rankings), Hugging Face `text-to-speech` trending models, Interspeech / ICASSP / SSW proceedings, and arXiv `eess.AS` + `cs.SD`.

---

## 15. The Reading List

Curriculum order. Core spine papers are marked ★: read those twelve carefully and you understand the field; the rest add depth per topic.

**Foundations and classical**
1. ★ Zen, Tokuda, Black: Statistical Parametric Speech Synthesis (2009): the pre-neural synthesis worldview
2. Hunt & Black: Unit Selection in a Concatenative Speech Synthesis System (ICASSP 1996)
3. Tokuda et al.: Speech Synthesis Based on Hidden Markov Models (Proc. IEEE 2013)

**The neural wave**
4. ★ WaveNet: arXiv:1609.03499
5. ★ Tacotron 2: arXiv:1712.05884 (plus Tacotron 1, arXiv:1703.10135)
6. Transformer TTS: arXiv:1809.08895
7. ★ FastSpeech 2: arXiv:2006.04558 (plus FastSpeech, arXiv:1905.09263)

**Vocoders**
8. ★ HiFi-GAN: arXiv:2010.05646
9. MelGAN: arXiv:1910.06711; Parallel WaveGAN: arXiv:1910.11480
10. WaveGlow: arXiv:1811.00002; WaveRNN: arXiv:1802.08435
11. BigVGAN: arXiv:2206.04658; Vocos: arXiv:2306.00814

**Probabilistic end-to-end**
12. ★ Glow-TTS: arXiv:2005.11129
13. ★ VITS: arXiv:2106.06103 (then VITS2: arXiv:2307.16430; YourTTS: arXiv:2112.02418)
14. NaturalSpeech: arXiv:2205.04421

**Diffusion and flow matching**
15. DDPM: arXiv:2006.11239; Score SDE: arXiv:2011.13456
16. ★ Grad-TTS: arXiv:2105.06337 (plus DiffWave: arXiv:2009.09761)
17. ★ Flow Matching: arXiv:2210.02747
18. Matcha-TTS: arXiv:2309.03199; Voicebox: arXiv:2306.15687
19. ★ F5-TTS: arXiv:2410.06885 (plus E2-TTS: arXiv:2406.18009)

**Codecs and speech LMs**
20. VQ-VAE: arXiv:1711.00937
21. ★ SoundStream: arXiv:2107.03312; EnCodec: arXiv:2210.13438; DAC: arXiv:2306.06546
22. AudioLM: arXiv:2209.03143
23. ★ VALL-E: arXiv:2301.02111 (then VALL-E 2: arXiv:2406.05370)
24. SPEAR-TTS: arXiv:2302.03540; SoundStorm: arXiv:2305.09636; MaskGCT: arXiv:2409.00750
25. Tortoise: arXiv:2305.07243; StyleTTS 2: arXiv:2306.07691

**The modern systems (pick per interest)**
26. Seed-TTS: arXiv:2406.02430 (the benchmark-setter)
27. ★ CosyVoice 2: arXiv:2412.10117; CosyVoice 3: arXiv:2505.17589
28. Llasa: arXiv:2502.04128 (scaling laws for LLM-style TTS)
29. Spark-TTS: arXiv:2503.01710; Fish Speech: arXiv:2411.01156; Fish Audio S2: arXiv:2603.08823
30. IndexTTS2: arXiv:2506.21619; VoxCPM: arXiv:2509.24650
31. Surveys: arXiv:2106.15561 and arXiv:2412.06602 (read alongside everything above)

---

## 16. Suggested 8-Week Study Plan

| Week | Topic | Read | Watch | Build |
|---|---|---|---|---|
| 1 | Signals + classical TTS | §1-3 here; Taylor ch. on frontend; paper 1 | speech.zone Speech Processing modules 5-6 | implement STFT → mel → Griffin-Lim from scratch in NumPy |
| 2 | Unit selection + SPSS | papers 2-3 | speech.zone Speech Synthesis modules 1-4 | run Festival; derive the MLPG solution by hand |
| 3 | AR neural TTS | papers 4-6 | HF Audio Course units 1-3 | train Tacotron 2 on LJSpeech (ESPnet); break its attention on long text, observe failure modes |
| 4 | NAR + vocoders | papers 7-11 | HF Audio Course unit 6 | train FastSpeech 2 + HiFi-GAN; A/B against week 3 |
| 5 | Flows + VITS | papers 12-14 | speech.zone state-of-the-art module | fine-tune VITS or Piper on a new voice; inspect MAS alignments |
| 6 | Diffusion + FM | papers 15-19 | | train Matcha-TTS on LJSpeech; then fine-tune F5-TTS on your language and measure WER/SIM |
| 7 | Codecs + speech LMs | papers 20-25 | | encode/decode audio with EnCodec and DAC at several bitrates; plot token-rate vs quality; run CSM or Spark-TTS locally |
| 8 | Modern systems + eval | papers 26-31 | | fine-tune CosyVoice 2 or XTTS on a curated dataset; build an eval harness (Whisper WER + WavLM SIM + UTMOS) and benchmark 3 open models on your own test set |

The pattern of the whole field, worth internalizing: every generation solves the previous one's bottleneck (naturalness → alignment robustness → speed → distribution modeling → zero-shot generality → controllability and latency) and the discarded machinery keeps returning in new clothes. Viterbi search became MAS; HMM forced alignment trains FastSpeech; the source-filter model hides inside HiFi-GAN's period discriminators; and MLPG's "smooth trajectory under a probabilistic model" is, squinting only slightly, what a flow-matching ODE does. Learn the history and the newest paper stops being magic.

---

*Compiled August 2026. Corrections and PRs welcome.*
