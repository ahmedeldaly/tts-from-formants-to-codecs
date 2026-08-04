# دليل تقني معمّق في تحويل النص إلى كلام (TTS)

**رحلة كاملة من الـ formant synthesis إلى الـ flow matching والـ codec language models، من السبعينيات حتى 2026**

كُتب هذا الدليل ليكون منهجاً دراسياً متكاملاً في الـ TTS، لا مجرد استعراض سريع. ستجد فيه شرحاً لكل حقبة في تاريخ المجال: ما المشكلة التي واجهتها، وما الرياضيات التي حلّتها بها، وما الأوراق التي تمثلها، وما الذي تركته للأنظمة التي نستخدمها اليوم. وفي النهاية قائمة قراءة مرتبة ترتيباً منهجياً، ودورات مجانية مفتوحة، وأدوات عملية، وخطة دراسة من ثمانية أسابيع.

**لمن هذا الدليل؟** لمهندسي الـ ML والباحثين الذين يريدون تجاوز مرحلة "استدعاء الـ API" إلى فهم حقيقي للـ acoustic models والـ vocoders والـ neural codecs والـ speech language models، فهماً يكفي لتدريب هذه النماذج وعمل fine-tuning لها وتتبّع أخطائها عند حدوثها.

**الرموز الرياضية:** الرمز $x$ هو النص (حروف أو phonemes)، و $y$ هو الموجة الصوتية (الـ waveform)، و $M$ هو الـ mel spectrogram، و $z$ متغير latent، و $c$ أي conditioning إضافي. المعادلات منقولة من الأوراق الأصلية بصيغها القياسية مع تبسيط طفيف أحياناً.

---

## جدول المحتويات

1. ما المشكلة أصلاً؟ وما شكل الـ pipeline الكلاسيكي؟
2. الحقبة الكلاسيكية: الـ formants والـ unit selection والـ HMMs
3. أساسيات معالجة الإشارات التي تحتاجها
4. الثورة العصبية: WaveNet و Tacotron و FastSpeech
5. الـ neural vocoders
6. الـ flows و VITS
7. الـ diffusion والـ flow matching
8. الـ neural audio codecs: الكلام كـ tokens
9. الـ TTS بعقلية الـ LLM والمشهد الحديث (2023-2026)
10. كيف نقيّم أنظمة TTS؟
11. ملاحظات خاصة بالعربية
12. مجموعات البيانات
13. الدورات المفتوحة والكتب
14. الأدوات والمستودعات
15. قائمة القراءة بترتيب منهجي
16. خطة دراسة من 8 أسابيع

---

## 1. ما المشكلة أصلاً؟

الـ TTS في جوهره مسألة conditional generation: نريد نموذجاً يتعلم التوزيع

$$p(y \mid x)$$

ثم يولّد منه عينات، حيث $y = (y_1, \dots, y_T)$ موجة صوتية (قد تكون $T = 24{,}000$ عينة في الثانية الواحدة) و $x$ هو النص. لماذا هذه المسألة صعبة تحديداً؟ لثلاثة أسباب:

1. **العلاقة one-to-many.** الجملة الواحدة يمكن نطقها بطرق لا نهائية، كلها صحيحة: يختلف المتحدث والـ prosody والانفعال والسرعة وظروف التسجيل. وأي نموذج يعمل regression نحو مخرج "متوسط" واحد سيعطيك صوتاً مكتوماً باهتاً. ولو تأملت ستجد أن معظم الأفكار المعمارية في هذا الدليل، من الـ autoregression إلى الـ flows والـ VAEs والـ GANs والـ diffusion والـ discrete tokens، هي إجابات مختلفة عن سؤال واحد: كيف ننمذج التوزيع كاملاً بدل أن نكتفي بمتوسطه؟
2. **فجوة الأطوال.** جملة من 10 كلمات تعني نحو 50 phoneme، تقابل نحو 400 إطار mel، تقابل نحو 240,000 عينة صوتية. لا بد أن يحسم شيء ما مسألة الـ alignment: أي جزء من الصوت يقابل أي جزء من النص؟ الـ attention والـ forced alignment والـ duration predictors والـ monotonic alignment search وحتى الـ filler tokens كلها حلول مختلفة لهذه المشكلة الواحدة.
3. **حساسية الأذن.** الأذن البشرية تلتقط عيوباً في حدود الميلي ثانية، وتلتقط تشوهات الـ phase والنبرة غير الطبيعية، وهي أشياء لا تعاقب عليها خسائر تقليدية مثل الـ MSE.

**التقسيم الكلاسيكي** الذي حكم المجال لعقود كان ثلاث مراحل:

```
text ──► Frontend ──► linguistic features ──► Acoustic model ──► acoustic features ──► Vocoder ──► waveform
        (normalize,                            (predicts mel/                          (mel → wav)
         G2P, prosody)                          F0/duration)
```

- **الـ frontend:** تطبيع النص (تحويل "$5.2M in 2026" إلى كلمات منطوقة)، ثم G2P أي التحويل من حروف إلى phonemes، ثم تقسيم العبارات وتحديد النبر والـ prosody.
- **الـ acoustic model:** من السمات اللغوية إلى السمات الصوتية: الـ mel spectrogram والـ F0 والـ durations.
- **الـ vocoder:** من السمات الصوتية إلى الموجة الصوتية النهائية.

الأنظمة الحديثة تدمج هذه المراحل شيئاً فشيئاً حتى تكاد تختفي الحدود بينها. وهذه خريطة سريعة للمدارس المختلفة:

| المدرسة | كيف تحل الـ alignment | ماذا تُخرج | أمثلة |
|---|---|---|---|
| Concatenative | forced alignment + بحث | مقاطع waveform حقيقية | Festival unit selection |
| Statistical parametric | حالات HMM | معاملات vocoder | HTS, Merlin |
| AR seq2seq | attention متعلَّم | إطارات mel | Tacotron 2, Transformer TTS |
| NAR duration-based | durations خارجية | إطارات mel | FastSpeech 2, FastPitch |
| Flow / VAE end-to-end | monotonic alignment search | latent ثم waveform | Glow-TTS, VITS |
| Diffusion / flow matching | durations أو alignment ضمني | mel أو latents | Grad-TTS, F5-TTS, Voicebox |
| Codec language models | ضمني (AR فوق الـ tokens) | audio tokens منفصلة | VALL-E, CosyVoice, Llasa, Qwen3-TTS |

---

## 2. الحقبة الكلاسيكية

لماذا تدرس حقبة انتهت؟ لثلاثة أسباب وجيهة: أولاً لأن مفرداتها (الـ formants والـ diphones والـ F0 وتكاليف الـ alignment) ما زالت مفردات المجال حتى اليوم. وثانياً لأن أدواتها الاحتمالية (محاذاة الـ HMM وفكرة الـ MLPG) تعود للظهور داخل النماذج الحديثة بأشكال جديدة. وثالثاً لأن أنماط فشلها هي التي تفسر لماذا اختُرعت كل فكرة عصبية جاءت بعدها.

### 2.1 الـ Formant Synthesis (الستينيات إلى الثمانينيات)

نقطة البداية هي الـ source-filter model: الكلام إشارة إثارة $e[n]$ (نبضات الحبال الصوتية في الأصوات المجهورة، و noise في المهموسة) تمر خلال القناة الصوتية التي نمثلها كـ filter خطي $h[n]$:

$$s[n] = (e * h)[n] \quad\Longleftrightarrow\quad S(z) = E(z)\,H(z)$$

ونقرّب filter القناة الصوتية بمجموعة resonators من الدرجة الثانية، واحد لكل formant (تردد رنين $F_k$ وعرض نطاق $B_k$):

$$H_k(z) = \frac{1}{1 - 2e^{-\pi B_k T}\cos(2\pi F_k T)\,z^{-1} + e^{-2\pi B_k T} z^{-2}}$$

حيث $T$ فترة الـ sampling. أنظمة مثل الـ Klatt synthesizer (عام 1980) و DECtalk كانت تتحكم في عشرات من هذه المعاملات بقواعد مكتوبة يدوياً لكل phoneme ولكل انتقال بين صوتين. النتيجة؟ كلام مفهوم تماماً وخفيف حسابياً إلى أبعد حد، لكنه آلي النبرة بلا أي لبس. صوت Stephen Hawking الشهير كان هذه التقنية بالضبط.

**الدرس الذي بقي:** الكلام يساوي excitation مضروبة في spectral envelope. هذه البنية نفسها تستغلها الـ LPC vocoders و WORLD، وحتى الـ period discriminators في HiFi-GAN كما سنرى.

### 2.2 الـ Unit Selection (التسعينيات إلى العقد الأول)

الفكرة هنا معاكسة تماماً: بدل أن تولّد الكلام، أعد استخدام كلام مسجَّل بالفعل.

بدأ الأمر بالـ diphone synthesis: خزّن مثالاً واحداً لكل انتقال بين صوتين (نحو 1500 diphone للإنجليزية) ثم صِلها ببعضها. ولإصلاح الـ prosody بعد الوصل، جاءت خوارزمية TD-PSOLA: فكّك الإشارة إلى نوافذ متمركزة حول نبضات الحبال الصوتية $t_i$، ثم أعد توزيعها واجمعها بشكل متراكب لتغيّر الـ F0 والمدة دون أن تمس الـ spectral envelope:

$$\tilde{s}[n] = \sum_i w[n - \tilde{t}_i]\, s_i[n - \tilde{t}_i + t_i]$$

ثم وسّع الـ unit selection (بحث Hunt و Black عام 1996) الفكرة إلى أقصاها: سجّل ساعات من الكلام، وقطّعها إلى وحدات، وعند التوليف ابحث في قاعدة البيانات عن سلسلة الوحدات $u_{1:n}$ الأقرب للمواصفات المطلوبة $t_{1:n}$ عن طريق تقليل

$$C(t_{1:n}, u_{1:n}) = \sum_{i=1}^{n} C^{t}(t_i, u_i) + \sum_{i=2}^{n} C^{c}(u_{i-1}, u_i)$$

حيث $C^t$ هي الـ target cost (مدى مطابقة الوحدة للسياق الصوتي والعروضي المطلوب) و $C^c$ هي الـ join cost (مقدار عدم الاستمرارية في الطيف والـ F0 والطاقة بين وحدتين متتاليتين). والسلسلة المثلى نستخرجها ببحث Viterbi فوق شبكة الوحدات. هذا النظام هو الذي شغّل Festival، وشغّل الـ TTS التجاري عند Apple و Nuance و Google حتى منتصف العقد الثاني من الألفية.

**لماذا مات؟** في أفضل حالاته هو كلام بشري حقيقي مسجَّل، ولذلك كان طبيعياً للغاية. لكن الجودة تنهار فجأة عندما لا تجد قاعدة البيانات وحدة مناسبة (وهي الـ join artifacts الشهيرة)، وبناء صوت جديد يكلف أسابيع من التسجيل في الاستوديو، والتحكم في الأسلوب والانفعال شبه مستحيل. ومع ذلك انتبه: بنية "دالة تكلفة زائد Viterbi" هذه هي الجد المباشر لبحث الـ alignment في Glow-TTS الذي سيأتي بعد ربع قرن.

### 2.3 الـ SPSS: التوليف الإحصائي بالـ HMM (من أوائل الألفية حتى 2015)

الـ SPSS اختصار Statistical Parametric Speech Synthesis (ورقة Zen و Tokuda و Black عام 2009، وأداة HTS الشهيرة). الفكرة: استبدل قاعدة البيانات كلها بنموذج إحصائي توليدي يتعلم السمات الصوتية نفسها.

**التمثيل.** كل frame نصفه بمعاملات vocoder: معاملات mel-cepstral تمثل الـ spectral envelope، و $\log F_0$، و aperiodicity. نستخرجها بأدوات كلاسيكية مثل STRAIGHT أو WORLD (الـ F0 عبر DIO أو Harvest، والـ envelope عبر CheapTrick، والـ aperiodicity عبر D4C). والتوليف يعكس الطريق: من المعاملات إلى filter من نوع MLSA أو توليف WORLD ثم إلى الموجة.

**النموذج.** لكل phoneme في سياقه الكامل (هوية الـ quinphone والموقع والنبر ونوع الكلمة، نحو 50 سمة) نبني HMM من 5 حالات باتجاه واحد، يُخرج frames بتوزيعات Gaussian. ولأن عدد السياقات الممكنة فلكي، نربط الحالات المتشابهة معاً عبر decision trees. والـ $\log F_0$ غير معرَّف في المقاطع المهموسة، فيُعالج بحيلة اسمها MSD أي multi-space distributions. والـ durations تحصل على نماذج Gaussian صريحة لكل حالة، فيتحول النموذج فعلياً إلى hidden semi-Markov model.

**المعادلة الأهم في الحقبة كلها: الـ MLPG.** لو اكتفينا بأخذ متوسط كل state سنحصل على مسارات متقطعة تشبه درجات السلم. الحل الذكي: مثّل كل frame بسماته الساكنة زائد مشتقاتها الزمنية (الـ deltas):

$$\mathbf{o}_t = \big[\mathbf{c}_t^\top,\ \Delta \mathbf{c}_t^\top,\ \Delta^2 \mathbf{c}_t^\top\big]^\top,\qquad \Delta \mathbf{c}_t = \tfrac{1}{2}(\mathbf{c}_{t+1} - \mathbf{c}_{t-1})$$

وبهذا تصبح سلسلة الملاحظات كلها دالة خطية $\mathbf{o} = W\mathbf{c}$ في السلسلة الساكنة $\mathbf{c}$، حيث المصفوفة $W$ تكدّس نوافذ الـ delta. ومع معرفة تسلسل الحالات $q$ بمتوسطاته $\boldsymbol{\mu}_q$ وتبايناته $\boldsymbol{\Sigma}_q$، نبحث عن $\mathbf{c}$ التي تعظّم $\log \mathcal{N}(W\mathbf{c};\ \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$. نساوي الـ gradient بالصفر فنحصل على:

$$W^\top \boldsymbol{\Sigma}_q^{-1} W\, \mathbf{c}^{*} = W^\top \boldsymbol{\Sigma}_q^{-1} \boldsymbol{\mu}_q$$

وهذا نظام خطي banded يُحل في زمن $O(T)$، ويعطينا مسارات ناعمة متسقة مع إحصاءات الـ statics والـ deltas معاً. المفارقة أنها خرجت ناعمة أكثر من اللازم، فجاءت تقنية الـ Global Variance لتعيد نفخ التباين في المسارات المولّدة.

**الـ DNN-SPSS (من 2013 إلى 2016).** استبدل Zen وفريقه شجرة القرار والـ Gaussians بشبكة DNN ثم LSTM تحوّل السمات اللغوية مباشرة إلى سمات صوتية على مستوى الـ frame. أداة Merlin هي التنفيذ المرجعي لهذا الجيل. نفس الـ vocoder ونفس مشكلة النعومة الزائدة، لكن regression أفضل.

**لماذا خسرت المدرسة كلها؟** لأن regression بالـ MSE نحو معاملات الـ vocoder يحسب المتوسط فوق علاقة one-to-many فيخرج الصوت مكتوماً وفيه أزيز، ولأن الـ vocoders الكلاسيكية نفسها كانت سقفاً للجودة. **ولماذا ما زالت مهمة؟** لأن محاذاة الـ HMM القسرية هي حتى اليوم مصدر الـ durations التي تتدرب عليها نماذج عائلة FastSpeech، ولأن الـ MLPG هو الجد الفكري لكل طريقة تقول "تنبأ بمسار ناعم تحت نموذج احتمالي"، ولأن هندسة الـ frontend كلها (السياقات والـ G2P والتطبيع) انتقلت كما هي إلى الأنظمة العصبية.

---

## 3. أساسيات معالجة الإشارات التي تحتاجها

**الـ sampling.** نماذج الكلام تعمل عند 16 أو 22.05 أو 24 أو 44.1 أو 48 kHz. وقاعدة Nyquist تقول إن أقصى محتوى ترددي يمكن تمثيله هو $f_s/2$.

**الـ STFT.** بنافذة $w$ طولها $N$ وقفزة $H$ بين النوافذ:

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N}$$

**مقياس mel.** الأذن لا تسمع الترددات بشكل خطي، فنلوي محور التردد ليطابق الإدراك:

$$\mathrm{mel}(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

**الـ mel spectrogram.** طبّق مجموعة من $B$ من الـ mel filters المثلثية $H_b[k]$ على طيف القدرة ثم اضغط لوغاريتمياً:

$$M[m, b] = \log\left( \sum_k H_b[k]\, \lvert X[m,k]\rvert^2 \right)$$

الإعداد المعتاد في الـ acoustic models: صوت 22.05 أو 24 kHz، و $N = 1024$، و $H = 256$ (أي نحو 11.6 ms لكل frame)، و $B$ بين 80 و 100. لماذا صار الـ mel spectrogram هو اللغة المشتركة بين الـ acoustic models والـ vocoders؟ لأنه مضغوط (نحو 86 frame في الثانية في 80 بعداً، بدل 24,000 عينة في الثانية) ومطابق لإدراك الأذن. لكن انتبه لنقطة جوهرية: الـ mel spectrogram يتخلص من الـ phase تماماً. وهذا بالضبط ما يجعل الـ vocoder مسألة توليدية صعبة، لا مجرد تحويل عكسي.

**خوارزمية Griffin-Lim** تستعيد phase معقولاً بالتكرار بين قيد الـ magnitude ومجموعة تحويلات STFT المتسقة:

$$x^{(i+1)} = \mathrm{ISTFT}\!\left( |X| \cdot e^{\,j \angle \mathrm{STFT}(x^{(i)})} \right)$$

كانت هذه هي vocoder نموذج Tacotron الأول، وما زالت baseline مفيداً في الـ debugging: صوتها معدني لكنه مفهوم.

**الـ F0 أو الـ pitch:** التردد الأساسي للكلام المجهور، بين 60 و 300 Hz تقريباً عند البالغين، مع harmonics عند مضاعفاته. الـ prosody كله تقريباً محمول على منحنيات الـ F0 والطاقة والـ durations. ولهذا بالتحديد ستجد أن FastSpeech 2 يتنبأ بهذه الثلاثة بالذات.

---

## 4. الثورة العصبية (2016-2020)

### 4.1 WaveNet (عام 2016): الـ autoregression فوق الصوت الخام

جاء WaveNet (ورقة van den Oord وفريقه، arXiv:1609.03499) وكسر القاعدة الأولى: تخلّى عن فرضية الـ source-filter كلياً وقرر نمذجة الموجة الصوتية عينة عينة:

$$p(y) = \prod_{t=1}^{T} p(y_t \mid y_1, \dots, y_{t-1},\ c)$$

كل توزيع شرطي هنا هو توزيع categorical فوق 256 مستوى، بعد ضغط الإشارة بالـ mu-law:

$$f(y_t) = \mathrm{sign}(y_t)\,\frac{\ln(1 + \mu |y_t|)}{\ln(1 + \mu)}, \qquad \mu = 255$$

الشبكة كومة من الـ dilated causal convolutions يتضاعف فيها الـ dilation مع كل طبقة (1 ثم 2 ثم 4 حتى 512 ثم يتكرر النمط)، فنحصل على receptive field ينمو أسّياً بتكلفة خطية، مع وحدة gated:

$$\mathbf{z} = \tanh(W_{f} * \mathbf{x}) \odot \sigma(W_{g} * \mathbf{x})$$

إضافة إلى وصلات residual و skip. والـ conditioning يدخل في كل طبقة: global مثل embedding المتحدث، و local مثل السمات اللغوية بعد رفع عيّناتها، ولاحقاً الـ mel spectrograms.

كان WaveNet أول نموذج عصبي يخرج كلاماً يشبه البشر بوضوح. المشكلة؟ التوليد autoregressive بمعدل الصوت نفسه، أي أبطأ بكثير من الزمن الحقيقي في نسخته الأولى. جاء Parallel WaveNet بحل عبر تقطير النموذج في student من نوع IAF، لكنه حل بتكلفة لا تتحملها إلا شركة بحجم Google. الإرث الحقيقي المزدوج: أولاً أن نمذجة الاحتمال فوق الصوت الخام ممكنة أصلاً، وثانياً أن WaveNet المشروط بالـ mel هو الذي عرّف وظيفة الـ neural vocoder كما نعرفها.

### 4.2 Tacotron 2 (عام 2017): الـ attention من النص إلى الـ mel

Tacotron 2 (ورقة Shen وفريقه، arXiv:1712.05884) هو النموذج القياسي لفكرة seq2seq بالـ attention في الـ TTS:

- **الـ encoder:** embeddings للحروف، ثم 3 طبقات convolution، ثم LSTM ثنائي الاتجاه، فتخرج الذاكرة $h_{1:S}$.
- **الـ attention من نوع location-sensitive:** عند كل خطوة $t$ من فك الترميز، تُحسب الطاقات من المحتوى ومن الـ alignment السابق $\alpha_{t-1}$ معاً (عبر convolution بالشكل $f_{t} = F * \alpha_{t-1}$)، وهذا ما يشجع الحركة الرتيبة للأمام:

$$e_{t,s} = \mathbf{w}^\top \tanh\!\big(W \mathbf{s}_t + V \mathbf{h}_s + U \mathbf{f}_{t,s} + \mathbf{b}\big), \qquad \alpha_t = \mathrm{softmax}(e_t), \qquad \mathbf{c}_t = \sum_s \alpha_{t,s}\mathbf{h}_s$$

- **الـ decoder:** شبكة LSTM من طبقتين تعمل بشكل autoregressive، تستقبل frame الـ mel السابق عبر pre-net بمعدل dropout عالٍ جداً. هذا الـ dropout ليس تفصيلة: هو عنق الزجاجة الذي يجبر النموذج على الاعتماد على الـ attention بدل نسخ الماضي. ثم يتنبأ الـ decoder بالـ frame التالي مع stop token، ويضيف post-net من 5 طبقات convolution لمسة تحسين نهائية.
- **الخسارة:** MSE على الـ mels قبل الـ post-net وبعده، زائد BCE على الـ stop token.

بربطه مع vocoder من نوع WaveNet مشروط بالـ mel، سجّل Tacotron 2 قيمة MOS بلغت 4.53 مقابل 4.58 للتسجيل البشري الحقيقي في نطاقه أحادي المتحدث. هذه كانت لحظة "التعادل مع البشر داخل النطاق" في تاريخ الـ TTS العصبي.

**أنماط فشل ستقابلها بنفسك** عند تشغيل أي نموذج AR بالـ attention: كلمات تُحذف أو تتكرر، والـ attention ينهار مع النصوص الطويلة أو الخارجة عن التوزيع، والنموذج يهذي بعد نهاية الجملة. ظهر حقل فرعي كامل من الترقيعات (forward attention و GMM attention و dynamic convolution attention)، لكن الحل الجذري كان التخلص من الـ soft attention المتعلَّم أصلاً. وهذا يقودنا للخطوة التالية.

### 4.3 Transformer TTS (عام 2019)

استبدل بشبكات LSTM كتل Transformer encoder-decoder (ورقة Li وفريقه، arXiv:1809.08895). المكسب: تدريب متوازٍ وسياق أطول. لكن الاستدلال بقي autoregressive والـ attention بقي هشاً. أهميته التاريخية أنه جعل الخطوة التالية بديهية لأي قارئ.

### 4.4 FastSpeech 1 و 2 (من 2019 إلى 2020): بلا autoregression وبلا attention

FastSpeech (ورقة Ren وفريقه، arXiv:1905.09263) حذف المشكلتين معاً. الأداة المركزية اسمها الـ length regulator: تنبأ بمدة صحيحة $d_i$ لكل token في المدخل، ثم كرر كل حالة من حالات الـ encoder عدد $d_i$ من المرات حتى تصل لطول الـ frames، ثم فكّ كل إطارات الـ mel دفعة واحدة بالتوازي عبر كومة Transformer.

أما FastSpeech 2 (arXiv:2006.04558) فتدرّب مباشرة على الأهداف الحقيقية وأضاف الـ variance adaptor: ثلاثة predictors للـ duration والـ pitch والـ energy بين الـ encoder والـ decoder. ومن أين تأتي الـ durations الحقيقية للتدريب؟ من الـ forced alignment، وعملياً من أداة Montreal Forced Aligner القائمة على triphone HMM-GMM. نعم، آلية الـ SPSS القديمة نفسها عادت للخدمة. والخسارة الكلية:

$$\mathcal{L} = \lVert M - \hat{M}\rVert_1 + \lVert \log \mathbf{d} - \log \hat{\mathbf{d}} \rVert_2^2 + \lVert \mathbf{p} - \hat{\mathbf{p}} \rVert_2^2 + \lVert \mathbf{e} - \hat{\mathbf{e}} \rVert_2^2$$

لماذا صار FastSpeech 2 (وقريبه FastPitch) حصان العمل في الصناعة لسنوات؟ لأن توليد الـ mel أسرع بمراتب من أي نموذج AR، ولأن أعطال الـ attention اختفت نهائياً وهذه متانة تحتاجها المنتجات، ولأن مقابض التحكم صريحة: اضرب الـ durations لتغيير السرعة، وحرّك الـ pitch كما تشاء. نقطة الضعف الوحيدة أن regression بالـ MSE أعاد شيئاً من النعومة الزائدة، فخرج الـ prosody أكثر تسطحاً من نموذج AR جيد. وإصلاح هذه النقطة تحديداً يتطلب نمذجة توليدية حقيقية فوق الـ mels. وهنا بالضبط يدخل الـ flows والـ VAEs والـ GANs والـ diffusion إلى القصة.

---

## 5. الـ Neural Vocoders

المطلوب من الـ vocoder: معطى mel spectrogram $M$ بلا phase وبترددات مضغوطة، ولّد موجة $\hat{y}$ لا تستطيع الأذن تمييزها عن الكلام الحقيقي. جُرّبت في هذه المهمة كل عائلات النماذج التوليدية بلا استثناء، وفازت الـ GANs في المعادلة الصعبة: الجودة مع السرعة معاً.

### 5.1 العائلة الـ autoregressive

- **WaveNet و WaveRNN** (arXiv:1802.08435): استبدل WaveRNN بالـ convolutions العميقة وحدة GRU واحدة مع softmax مزدوج (8 بت خشنة و 8 بت دقيقة)، وأضاف الـ sparsification وحيل الـ subscale ليصل لسرعة الهواتف.
- **LPCNet:** تهجين ذكي بين الـ DSP والتعلم العميق: الشبكة تتنبأ فقط بالـ residual بينما يتكفل الـ linear prediction بالـ spectral envelope. ما زال خياراً وجيهاً للأجهزة محدودة الطاقة.

### 5.2 عائلة الـ flows: احتمال دقيق قابل للحساب

الـ normalizing flow يحوّل noise من توزيع $z \sim \mathcal{N}(0, I)$ عبر شبكة قابلة للعكس $y = f(z)$، ويعطيك الـ log-likelihood بدقة عبر قانون تغيير المتغيرات:

$$\log p_Y(y) = \log p_Z\!\big(f^{-1}(y)\big) + \log \left| \det \frac{\partial f^{-1}(y)}{\partial y} \right|$$

وسر قابلية الحساب هو طبقات الـ affine coupling (من RealNVP و Glow): قسّم المدخل $x = [x_a, x_b]$ ثم:

$$y_a = x_a, \qquad y_b = s(x_a) \odot x_b + t(x_a), \qquad \log|\det J| = \sum \log |s(x_a)|$$

وهي قابلة للعكس مهما كانت $s$ و $t$ معقدتين. نموذج WaveGlow (arXiv:1811.00002) شرط هذا الـ flow على الـ mels ودرّبه بالـ maximum likelihood فقط: تدريب مستقر وتوليد متوازٍ بخطوة واحدة، لكنه شره في المعاملات وتجاوزه الزمن.

### 5.3 عائلة الـ GANs: وقفة مطولة مع HiFi-GAN

أثبت MelGAN (arXiv:1910.06711) أن الـ adversarial vocoding يعمل فعلاً: generator كله convolutions، مع discriminators متعددة المقاييس، وخسارة feature matching. ثم أضاف Parallel WaveGAN (arXiv:1910.11480) خسارة STFT مساعدة متعددة الدقة جعلت تدريب هذه العائلة مستقراً بشكل ملموس.

ثم جاء HiFi-GAN (ورقة Kong وفريقه، arXiv:2010.05646) وكان التصميم الذي ثبت وبقي. معظم الأنظمة من VITS حتى XTTS ما زالت حتى اليوم تشحن decoder من عائلته:

- **الـ generator:** يقوم بالـ upsampling على الـ mel عبر transposed convolutions (بمعامل 256 مثلاً) مع كتل Multi-Receptive-Field fusion، وهي كومات convolutions متوازية بأحجام kernels و dilations مختلفة.
- **الـ Multi-Period Discriminator أو MPD:** يعيد تشكيل الموجة أحادية البعد إلى ثنائية البعد بدورات $p \in \{2,3,5,7,11\}$ ثم يطبق convolutions ثنائية البعد. النتيجة أن كل discriminator فرعي يرى بنية دورية مختلفة في الكلام. لاحظ الجمال هنا: فكرة الـ source-filter القديمة عادت للظهور بصفتها inductive bias داخل GAN.
- **الـ Multi-Scale Discriminator أو MSD:** يعمل على الموجة الخام وعلى نسختين منها بعد average pooling بمعاملي 2 و 4، لالتقاط البنية بعيدة المدى.

بصيغة LSGAN، لكل discriminator فرعي $D_k$:

$$\mathcal{L}_D = \mathbb{E}\big[(D_k(y) - 1)^2\big] + \mathbb{E}\big[D_k(G(M))^2\big], \qquad \mathcal{L}_G^{adv} = \mathbb{E}\big[(D_k(G(M)) - 1)^2\big]$$

مع feature matching فوق طبقات الـ discriminator وخسارة إعادة بناء الـ mel حيث $\phi$ هو تحويل mel:

$$\mathcal{L}_{FM} = \mathbb{E} \sum_{i} \frac{1}{N_i} \big\lVert D_k^{(i)}(y) - D_k^{(i)}(G(M)) \big\rVert_1, \qquad \mathcal{L}_{mel} = \big\lVert \phi(y) - \phi(G(M)) \big\rVert_1$$

$$\mathcal{L}_G = \mathcal{L}_G^{adv} + \lambda_{fm}\mathcal{L}_{FM} + \lambda_{mel}\mathcal{L}_{mel} \quad (\lambda_{fm} = 2,\ \lambda_{mel} = 45)$$

### 5.4 عائلة الـ diffusion

طبّق DiffWave (arXiv:2009.09761) و WaveGrad (arXiv:2009.00713) فكرة الـ denoising diffusion (رياضياتها كاملة في القسم 7) على توليد الموجة من الـ mel. الجودة ممتازة، لكن التوليد التكراري يجعلها أبطأ من الـ GANs عند نفس الجودة. فبقيت فرعاً بحثياً في الـ vocoding تحديداً، بينما انتصر الـ diffusion في مكان آخر: الـ acoustic modeling.

### 5.5 الجيل الحالي

- **BigVGAN** (من NVIDIA، arXiv:2206.04658): بنية HiFi-GAN مضافاً إليها دالة التفعيل الدورية snake بصيغة $f(x) = x + \frac{1}{\alpha}\sin^2(\alpha x)$ مع upsampling و downsampling مضاد للـ aliasing. هو الـ vocoder "الشامل" الافتراضي حالياً: متين حتى خارج نطاق تدريبه (الغناء ولغات لم يرها) وحتى 44.1 kHz.
- **Vocos** (arXiv:2306.00814): بدل الـ transposed convolutions في المجال الزمني، يتنبأ بالـ magnitude والـ phase للـ STFT مباشرة ثم يعكس بالـ ISTFT. أسرع بعشرة أضعاف بجودة موازية، ويُستخدم كثيراً أيضاً لفك رموز EnCodec.
- **UnivNet و iSTFTNet:** الفرع الخفيف السريع، و iSTFTNet تحديداً يعمل داخل Kokoro.

**نصيحة عملية:** لو تبني كومة قائمة على الـ mel في 2026، فالخيار الافتراضي هو checkpoints نموذج BigVGAN-v2 أو Vocos. والأهم من ذلك: درّب الـ acoustic model على نفس معاملات الـ mel التي يتوقعها الـ vocoder بالضبط، لأن عدم تطابق `hop/win/fmax` هو أشهر خطأ جودة صامت في المجال كله.

---

## 6. الـ Flows و VITS

### 6.1 Glow-TTS (عام 2020): هل نستغني عن الـ aligner الخارجي؟

سؤال Glow-TTS (ورقة Kim وفريقه، arXiv:2005.11129) كان: هل نحصل على متانة FastSpeech بدون aligner خارجي أصلاً؟ الجواب: نعم. النموذج ينمذج الـ mels بـ decoder من نوع normalizing flow بالشكل $z = f^{-1}(M)$، مع prior من توزيعات Gaussian تأتي إحصاءاتها $(\mu_i, \sigma_i)$ من الـ text encoder. والرابط بين الاثنين هو alignment رتيب مجهول $A$ يعيّن كل frame رقم $j$ إلى token رقم $i = A(j)$:

$$\log p(M \mid x) = \log \mathcal{N}\big(f^{-1}(M);\ \mu_{A}, \sigma_{A}\big) + \log\left|\det \frac{\partial f^{-1}(M)}{\partial M}\right|$$

والتدريب يتناوب بين خطوتين. الأولى: مع النموذج الحالي، ابحث عن أفضل alignment رتيب عبر خوارزمية MAS أي Monotonic Alignment Search، وهي dynamic programming على طريقة Viterbi:

$$Q_{i,j} = \log \mathcal{N}\big(z_j;\ \mu_i, \sigma_i\big) + \max\big(Q_{i-1,\,j-1},\ Q_{i,\,j-1}\big)$$

والثانية: عظّم الـ likelihood تحت هذا الـ alignment، ودرّب duration predictor على المدد الناتجة (مع stop-gradient). وعند الاستدلال: تنبأ بالـ durations، مدّد الـ prior، اسحب $z$، وشغّل الـ flow للأمام. النتيجة: لا attention ولا aligner خارجي، توازٍ كامل، ونموذج likelihood سليم رياضياً. وتذكر ما قلناه في القسم 2: هذه هي شبكة تكاليف Viterbi من زمن الـ unit selection، وقد بُعثت من جديد داخل نموذج توليدي عميق.

### 6.2 VITS (عام 2021): الوصفة الكاملة في نموذج واحد

جمع VITS (ورقة Kim و Kong و Son، arXiv:2106.06103) أربع أفكار في نموذج واحد يتدرب من النص إلى الموجة مباشرة: conditional VAE، و normalizing flows، و MAS، وتدريب adversarial. وظل قمة المصادر المفتوحة نحو ثلاث سنوات كاملة، وما زال حتى اليوم يشغّل Piper و MMS وعدداً لا يحصى من المنتجات.

**البنية باختصار.** الـ posterior encoder يحوّل الـ spectrogram الخطي (وليس mel) إلى متغيرات latent بالشكل $z \sim q(z \mid y)$. والـ decoder على طراز HiFi-GAN يعيد بناء الموجة من $z$. والـ prior وهو $p(z \mid x)$ عبارة عن توزيعات Gaussian من الـ text encoder جرى تعميقها بـ normalizing flow $f_\theta$، وتُحاذى مع الـ frames عبر MAS لكن هذه المرة في الفضاء الـ latent. والتدريب يعظّم الـ ELBO الشرطي:

$$\log p(y \mid x)\ \ge\ \mathbb{E}_{q(z \mid y)}\big[\log p(y \mid z)\big] - D_{KL}\big(q(z \mid y)\ \Vert\ p(z \mid x)\big)$$

حيث يتحقق حد إعادة البناء عملياً بخسارة L1 على mel الموجة الناتجة، ويُدرَّب النظام كله فوق ذلك تدريباً adversarial بنفس وصفة HiFi-GAN تماماً: discriminators من نوعي MPD و MSD مع feature matching. وهناك إضافة ذكية أخيرة: الـ stochastic duration predictor، وهو flow صغير يتدرب بحد variational مع dequantization، يتيح للنموذج أن يسحب إيقاعات مختلفة في كل مرة بدل سرعة واحدة حتمية.

**لماذا كان VITS نقلة؟** أولاً لأنه أزال خط اللحام بين الـ acoustic model والـ vocoder وما يتراكم عنده من أخطاء، إذ يتدرب الـ decoder على latents حقيقية لا على mels متوقعة. وثانياً لأنه أعطى مخرجات متعددة فعلاً عبر السحب من الـ latent والمدد العشوائية. وثالثاً لأنه قدّم وصفة كاملة قابلة لإعادة الإنتاج من أول يوم. جاء بعده VITS2 ليحسّن نموذج المدد والـ flows، ثم وسّعه YourTTS (arXiv:2112.02418) إلى zero-shot متعدد المتحدثين وعابر للغات عبر speaker encoders، فكان الجسر نحو المرحلة التالية.

---

## 7. الـ Diffusion والـ Flow Matching

هذا القسم هو القلب الرياضي لمعظم أنظمة الـ TTS الحديثة غير الـ autoregressive. اقرأه مرة واحدة بتركيز وستربح الفهم في كل ما بعده.

### 7.1 الـ DDPM

ثبّت جدول تباين $\beta_1, \dots, \beta_T$ وعرّف $\alpha_t = 1 - \beta_t$ و $\bar{\alpha}_t = \prod_{s\le t}\alpha_s$. العملية الأمامية تدمّر البيانات $x_0$ تدريجياً بإضافة noise:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big) \quad\Rightarrow\quad q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t) I\big)$$

وشبكة $\epsilon_\theta$ تتعلم عكس العملية. والجميل أن الـ ELBO كله يختزل (ورقة Ho وفريقه، arXiv:2006.11239) إلى الخسارة البسيطة الشهيرة:

$$\mathcal{L}_{simple} = \mathbb{E}_{t,\,x_0,\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t,\ c\big)\big\rVert^2\Big]$$

والتوليد يمشي بالسلسلة العكسية المتعلَّمة $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$ حيث:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\Big)$$

**زاوية النظر الثانية، الـ score والـ SDE** (ورقة Song وفريقه، arXiv:2011.13456): العملية الأمامية معادلة تفاضلية عشوائية $dx = f(x,t)\,dt + g(t)\,dw$، وعكسها:

$$dx = \big[f(x,t) - g(t)^2\, \nabla_x \log p_t(x)\big]\,dt + g(t)\,d\bar{w}$$

وتعلُّم $\epsilon_\theta$ يكافئ تعلُّم الـ score أي $\nabla_x \log p_t(x)$، لأن $\epsilon_\theta(x_t,t) = -\sqrt{1-\bar\alpha_t}\; s_\theta(x_t,t)$. وتوجد أيضاً probability-flow ODE حتمية لها نفس التوزيعات الهامشية، وهي التي تكاملها خوارزميات التوليد السريع.

### 7.2 الـ diffusion داخل الـ TTS

- **Grad-TTS** (ورقة Popov وفريقه، arXiv:2105.06337): الـ text encoder مع MAS (المأخوذ من Glow-TTS) يعطيان متوسطاً محاذى $\mu$، والـ SDE الأمامية هنا ترتد نحو هذا المتوسط: $dX_t = \tfrac{1}{2}(\mu - X_t)\beta_t\,dt + \sqrt{\beta_t}\,dW_t$. الذكاء في أن التوليد يبدأ من $\mathcal{N}(\mu, I)$، أي من نقطة صحيحة تقريباً منذ اللحظة الأولى، ثم تصقلها الـ SDE العكسية إلى mel حاد التفاصيل. هكذا حُلت مشكلة النعومة الزائدة بأناقة.
- **NaturalSpeech 2** (arXiv:2304.09116): نقل الفكرة إلى latent diffusion فوق latents متصلة من codec، مع conditioning على الـ pitch والـ durations: نظام zero-shot قوي بلا تراكم أخطاء الـ tokens المنفصلة.
- التكلفة المشتركة: عشرات إلى مئات من تقييمات الشبكة (الـ NFE) لكل عينة. حاول الناس علاجها بالتقطير (progressive و consistency models و CoMoSpeech)، لكن الحل الأنظف جاء من تغيير الهدف نفسه:

### 7.3 الـ Flow Matching: الطريقة المتصلة الافتراضية اليوم

الـ continuous normalizing flow ينقل الـ noise إلى البيانات عبر ODE بالشكل $\frac{dx}{dt} = v_\theta(x, t)$. والفكرة الجوهرية في الـ Flow Matching (ورقة Lipman وفريقه، arXiv:2210.02747) أننا ندرّب $v_\theta$ بمجرد regression نحو سرعات هدف شرطية، بلا أي محاكاة أثناء التدريب. ومع مسار الـ optimal transport بين عينة noise وهي $x_0 \sim \mathcal{N}(0,I)$ وعينة بيانات $x_1$:

$$x_t = \big(1 - (1-\sigma_{min})\,t\big)\,x_0 + t\,x_1, \qquad u_t(x_t \mid x_1) = x_1 - (1-\sigma_{min})\,x_0$$

$$\mathcal{L}_{CFM} = \mathbb{E}_{t,\,x_0,\,x_1}\Big[\big\lVert v_\theta(x_t, t, c) - u_t(x_t \mid x_1)\big\rVert^2\Big]$$

ولأن مسارات الهدف خطوط مستقيمة، تخرج الـ ODE المتعلَّمة شبه مستقيمة أيضاً. النتيجة العملية: حلّال بسيط مثل Euler أو midpoint يكفيه من 10 إلى 32 خطوة لجودة عالية، مقابل مئات الخطوات في DDPM الأصلي. والـ classifier-free guidance ينتقل كما هو: $\tilde{v} = v_\theta(x,t,\varnothing) + w\,\big(v_\theta(x,t,c) - v_\theta(x,t,\varnothing)\big)$.

### 7.4 عائلة الـ TTS القائمة على الـ flow matching

- **Matcha-TTS** (arXiv:2309.03199): نفس هيكل Grad-TTS (الـ encoder مع MAS مع duration predictor) لكن الـ decoder صار conditional flow matching. صغير وسريع وبجودة ممتازة، وكان أفضل وصفة "تتدرب على GPU واحد" في جيله.
- **Voicebox** (من Meta، arXiv:2306.15687): أعاد صياغة الـ TTS كمسألة speech infilling: أعطه سياقاً صوتياً محيطاً وسلسلة phonemes محاذاة قسرياً، وسيملأ إطارات الـ mel المقنّعة بالـ FM. نموذج regression منفصل يتنبأ بالـ durations. والنتيجة نموذج واحد يعمل zero-shot TTS وتنظيف الضجيج ونقل الأسلوب بين اللغات.
- **E2-TTS** (من Microsoft، arXiv:2406.18009): الاسم يعني "سهل إلى حد الإحراج"، والوصفة كذلك فعلاً: احذف الـ phonemes، واحذف نموذج الـ durations، واحذف الـ alignment الصريح كله. المدخل مجرد سلسلة الحروف محشوة بـ filler tokens حتى طول الـ mel المطلوب، و Transformer كبير مع FM يتعلم الـ alignment ضمنياً وحده. المدة الكلية هي مقبض التحكم الوحيد المتبقي.
- **F5-TTS** (arXiv:2410.06885): هو E2 بعد أن أصبح عملياً ومفتوحاً بالكامل: كتل ConvNeXt تجهّز تيار النص، و backbone من نوع DiT يقوم بالـ infilling، وحيلة الـ sway sampling تعيد توزيع شبكة الزمن في الـ ODE نحو الخطوات المبكرة التي تتشكل فيها البنية، بالصيغة $t' = t + s\big(\cos(\tfrac{\pi}{2}t) - 1 + t\big)$ مع $s < 0$، فترتفع الجودة عند NFE منخفض. تدرّب على نحو 100 ألف ساعة من Emilia، وهو اليوم أساس عدد ضخم من الـ fine-tunes المجتمعية ومنها نسخ عربية.

وخذ هذه الملاحظة معك للقسم التاسع: الـ flow matching هو أيضاً المرحلة الثانية القياسية في الأنظمة الحديثة القائمة على الـ LM، مثل الـ chunk-aware FM في CosyVoice و MegaTTS 3 و Voxtral TTS. القسمة صارت واضحة: الـ LM يقرر ماذا يُقال ومتى، والـ FM يرسم كيف يبدو الصوت.

---

## 8. الـ Neural Audio Codecs

هذا هو الجسر إلى عصر الـ LLM: حوّل الصوت المتصل إلى سلسلة قصيرة من tokens منفصلة يستطيع نموذج لغوي التنبؤ بها، ثم أعد بناء الصوت منها.

### 8.1 الـ VQ-VAE والـ Residual Vector Quantization

في الـ VQ-VAE (ورقة van den Oord وفريقه، arXiv:1711.00937) يُخرج الـ encoder المتجه $z_e(x)$، ثم نكمّمه إلى أقرب مدخل في codebook متعلَّم $\{e_k\}_{k=1}^{K}$:

$$z_q(x) = e_{k^*}, \qquad k^* = \arg\min_k \lVert z_e(x) - e_k \rVert_2$$

ويجري التدريب عبر straight-through estimator بالخسارة:

$$\mathcal{L} = \underbrace{\lVert x - \hat{x}\rVert^2}_{\text{reconstruction}} + \underbrace{\lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2}_{\text{codebook}} + \beta \underbrace{\lVert z_e(x) - \mathrm{sg}[e] \rVert_2^2}_{\text{commitment}}$$

لكن codebook واحداً لا يكفي لصوت كامل الدقة. فتلجأ الـ codecs لفكرة الـ RVQ أي Residual VQ: كمّم، اطرح، كمّم الباقي، وكرر على $C$ من الـ codebooks:

$$r_0 = z, \qquad q_c = Q_c(r_{c-1}), \qquad r_c = r_{c-1} - q_c, \qquad \hat{z} = \sum_{c=1}^{C} q_c$$

النتيجة تسلسل هرمي جميل: الـ codebook الأول يحمل المحتوى الخشن، وكل codebook لاحق يضيف تفاصيل صوتية أدق. وهذا الهرم بالذات هو ما تستغله كل خطط فك الترميز في أنظمة LM-TTS كما سنرى. حيلة إضافية: الـ quantizer dropout أثناء التدريب يجعل نموذجاً واحداً يخدم عدة معدلات bitrate. وهناك بديل جذري اسمه FSQ أي finite scalar quantization: بدل codebooks متعلَّمة، قيّد كل بعد latent بدالة $\tanh$ وقرّبه إلى $L$ مستوى، فتحصل على codebook ضمني حجمه $\prod_i L_i$ بلا أي مشاكل codebook collapse. وهذا ما تبنّاه tokenizer نموذج CosyVoice 2.

### 8.2 شجرة عائلة الـ codecs

| الـ codec | السنة | معدل الـ frames | البنية | ملاحظات |
|---|---|---|---|---|
| SoundStream (arXiv:2107.03312) | 2021 | 75 Hz | RVQ | أول codec عصبي كامل end-to-end، من 3 إلى 18 kbps |
| EnCodec (arXiv:2210.13438) | 2022 | 75 Hz | RVQ | من Meta؛ خسائر adversarial وطيفية متعددة المقاييس؛ يدعم البث |
| DAC (arXiv:2306.06546) | 2023 | 86 Hz | RVQ ×9 | تفعيلات snake، نحو 8 kbps عند 44.1 kHz، قفزة واضحة في الدقة |
| Mimi (من Kyutai، تقرير Moshi) | 2024 | 12.5 Hz | RVQ مقسوم | المستوى الأول مقطَّر من نموذج self-supervised فيحمل الدلالة؛ مصمم لنماذج الكلام؛ يستخدمه Moshi و Sesame CSM |
| WavTokenizer (arXiv:2408.16532) | 2024 | 75/40 Hz | VQ مفرد | codebook واحد فقط، مريح جداً للـ LMs |
| X-codec2 (الخاص بـ Llasa) | 2025 | 50 Hz | VQ مفرد بـ 65 ألف رمز | يدمج الدلالي والصوتي في تيار واحد |
| S3 / FSQ tokenizer (الخاص بـ CosyVoice) | 2024-25 | 25 Hz | supervised | الـ tokens تأتي من ASR-encoder مدرَّب بإشراف متعدد المهام |
| Qwen3-TTS tokenizer | 2026 | 12 Hz | متعدد الـ codebooks | معدل frames منخفض جداً يجعل فك ترميز الـ LM رخيصاً |

### 8.3 الـ semantic tokens مقابل الـ acoustic tokens

- **الـ semantic tokens** (عنقدة k-means فوق سمات HuBERT أو w2v-BERT، أو tokens من ASR-encoder مدرَّب بإشراف): تحمل المحتوى والـ prosody الخشن، وتُسقط هوية المتحدث والتفاصيل الدقيقة. معدلها منخفض، والـ LM يتنبأ بها بسهولة.
- **الـ acoustic tokens** (رموز RVQ من الـ codec): تعيد بناء الصوت الفعلي بأمانة. معدلها مرتفع، ونمذجتها أصعب.

وهنا حساب بسيط يفسّر المجال كله: EnCodec عند 75 Hz في 8 codebooks يعني 600 token لكل ثانية كلام. قارن مع نحو 100 في الثانية عند Mimi، و 50 عند X-codec2، و 12 فقط عند Qwen3-TTS. وطول السلسلة هو الفاتورة التي يدفعها الـ LM في كل خطوة. لذلك فإن اتجاه الـ codecs كله بين 2024 و 2026 كان: معدلات frames أقل مع دمج الدلالة في الداخل، حتى يبقى تيار AR واحد قصيراً. والطريق الموازي لنفس الهدف: تصميم من مرحلتين، يتنبأ فيه الـ LM بالـ semantic tokens فقط، ثم يتولى نموذج flow matching أو NAR رسم التفاصيل الصوتية.

---

## 9. الـ TTS بعقلية الـ LLM والمشهد الحديث

### 9.1 AudioLM و VALL-E: التوليف الصوتي كنمذجة لغوية

أثبت AudioLM (من Google، arXiv:2209.03143) أن Transformer من نوع decoder-only فوق tokens صوتية يستطيع توليد استمرارات كلامية متماسكة. الوصفة على ثلاث مراحل: نمذجة الـ semantic tokens، ثم تحويلها إلى acoustic tokens خشنة، ثم إضافة الـ codebooks الدقيقة.

ثم جاء VALL-E (من Microsoft، arXiv:2301.02111) بالنقلة الكبرى: الـ zero-shot TTS بوصفه in-context learning. المدخل: الـ phonemes $x$ زائد رموز EnCodec لمقطع مرجعي مدته 3 ثوانٍ فقط $\tilde{C}$. نموذج AR يتنبأ بالـ codebook الأول:

$$p\big(c_{t,1} \,\big|\, x,\ \tilde{C},\ c_{<t,1}\big)$$

ونموذج NAR يملأ الـ codebooks من 2 إلى 8 بالتوازي، مستوى بعد مستوى، مشروطاً بكل المستويات السابقة:

$$p\big(c_{:,j} \,\big|\, x,\ \tilde{C},\ c_{:,<j}\big), \qquad j = 2, \dots, 8$$

وهكذا صار استنساخ أي صوت مجرد prompting بلا أي fine-tuning، والجودة تتوسع مع حجم البيانات (تدرّب على 60 ألف ساعة من LibriLight). أضاف VALL-E 2 (arXiv:2406.05370) تقنية repetition-aware sampling ونمذجة رموز مجمّعة، وأعلن التعادل مع البشر على معياري LibriSpeech و VCTK. لكن السلالة ورثت أمراض التوليد الـ AR المزمنة: الهلوسة والتكرار وعدم الاستقرار. وهذه الأمراض بالذات هي التي رسمت خريطة البحث في العامين التاليين.

### 9.2 كيف تفك عدة codebooks بنموذج واحد؟

عندك $C$ من الـ codebooks في كل frame ونموذج تسلسلي واحد. الحلول المعروفة:

- **الـ flatten:** اجدل كل الـ codebooks في تيار واحد. دقيق رياضياً لكن السلسلة تطول بمعامل $C$ كاملاً.
- **الـ delay pattern** (من MusicGen): تنبأ بالـ codebook رقم $k$ الخاص بالزمن $t$ في الخطوة $t + k$. مرور AR واحد يغطي هرم الـ RVQ كله بتأخير $C$ خطوات فقط. يستخدمه Parler-TTS و Dia.
- **الخشن أولاً على مرحلتين:** AR فوق الـ semantic tokens أو المستوى الأول، ثم نموذج منفصل للبقية. هذا نهج AudioLM و VALL-E و SPEAR-TTS (arXiv:2302.03540).
- **فك متوازٍ على طريقة MaskGIT** وهو نهج SoundStorm (arXiv:2305.09636): ابدأ بكل المواضع مقنَّعة، ثم اكشف في كل تكرار المواضع الأعلى ثقة بجدول cosine. بضع تكرارات متوازية لكل مستوى RVQ تولّد 30 ثانية صوت في نصف ثانية تقريباً. وبنى MaskGCT (arXiv:2409.00750) على الفكرة نظام TTS كامل غير autoregressive من النص إلى الدلالي إلى الصوتي، بلا نموذج durations، مع تحكم صريح في الطول الكلي.
- **الـ depth transformer:** عمود فقري زمني كبير يتنبأ بـ embedding واحد لكل frame، ثم يوسّعه transformer صغير عبر الـ codebooks داخل الـ frame نفسه. هذه فكرة RQ-Transformer، ويستخدمها Moshi و Sesame CSM.

### 9.3 Tortoise و Bark و XTTS: جيل الجسر

- **Tortoise-TTS** (ورقة Betker، arXiv:2305.07243): نموذج AR فوق رموز mel-VQ، زائد diffusion decoder نحو الـ mels، زائد إعادة ترتيب المرشحين بنموذج تبايني اسمه CLVP. بطيء عمداً، لكن مخططه "كبّر الـ AR ثم أعد الترتيب ثم فك بالـ diffusion" أثّر في كل ما جاء بعده. و XTTS هو سليله الإنتاجي المباشر.
- **Bark** (من Suno، عام 2023): أكوام GPT متتالية من النص إلى الدلالي إلى الخشن إلى رموز EnCodec الدقيقة. توليد حر يشمل الضحك وأصواتاً شبه موسيقية وكلاماً متعدد اللغات، والثمن كان ضعف قابلية التحكم.
- **XTTS v2** (من Coqui): عمود فقري GPT مع decoder على طراز HiFi-GAN، واستنساخ صوت من 3 ثوانٍ فقط عبر 17 لغة من ضمنها العربية، مع سهولة fine-tuning حقيقية. الشركة نفسها أغلقت في يناير 2024، لكن النسخة المصانة في `idiap/coqui-ai-TTS` أبقته حياً بوصفه النموذج الأكثر عملية لمن يريد استنساخاً وضبطاً في ليلة واحدة.

### 9.4 المشهد المفتوح من 2023 إلى 2026

مختارات من الأنظمة مفتوحة الأوزان بترتيب زمني تقريبي. وتذكير مهم: راجع الترخيص على بطاقة النموذج دائماً، فأكثر من مشروع غيّر شروطه بين إصدار وآخر.

| النموذج | السنة | المعاملات | المدرسة | ملاحظات |
|---|---|---|---|---|
| StyleTTS 2 (arXiv:2306.07691) | 2023 | نحو 150M | style diffusion مع GAN | تدريب adversarial ضد discriminators مبنية على WavLM؛ مستوى بشري على LJSpeech؛ وهو أساس Kokoro |
| MetaVoice و Parler-TTS (arXiv:2402.01912) | 2024 | 1B / 0.6-0.9B | AR codec LM | ميزة Parler: تحكم في الصوت عبر وصف بلغة طبيعية |
| CosyVoice 1/2 (arXiv:2407.05407 و 2412.10117) | 2024 | 0.5B | AR LM فوق semantic tokens مع chunk-aware flow matching | النسخة الثانية: backbone من Qwen2.5-0.5B، رموز FSQ، بث ولا بث في نموذج واحد، وكمون نحو 150 ms |
| Fish Speech (arXiv:2411.01156) | 2024 | نحو 0.5B | dual-AR codec LM | يستخدم codec باسم firefly-GAN؛ تلاه OpenAudio S1 في 2025 ثم Fish Audio S2 في 2026 |
| MaskGCT (arXiv:2409.00750) | 2024 | 1B | مقنَّع غير autoregressive بالكامل | موجود في Amphion؛ تدرّب على Emilia |
| F5-TTS (arXiv:2410.06885) | 2024 | 0.34B | flow-matching infilling | راجع القسم 7.4؛ منظومة fine-tunes ضخمة حوله |
| Kokoro-82M | 2025 | 82M | سلالة StyleTTS2 مع iSTFTNet | رخصة Apache 2.0، سريع على CPU، وجودته نسبة لحجمه غير معقولة؛ ما زال خياراً أول في 2026 |
| Llasa (arXiv:2502.04128) | 2025 | 1B/3B/8B | AR بتيار واحد (تهيئة من Llama) فوق X-codec2 | أنقى تجسيد لفكرة "الـ TTS هو حرفياً LLM"؛ مع قوانين scaling للتدريب والاستدلال |
| Zonos-v0.1 | 2025 | 1.6B | AR بنسختي transformer و SSM مع DAC | من Zyphra؛ استنساخ مع تحكم انفعالي |
| Spark-TTS (arXiv:2503.01710) | 2025 | 0.5B | AR فوق BiCodec | رموز دلالية مع رموز متحدث شاملة؛ تحكم في الـ pitch والسرعة |
| Sesame CSM-1B | 2025 | 1B مع depth decoder | AR فوق رموز Mimi | حواري ومشروط بالسياق؛ رخصة Apache 2.0 |
| Orpheus-3B | 2025 | 3B | Llama-3.2 فوق رموز SNAC | وسوم انفعالية مثل `<laugh>` مع بث |
| Dia-1.6B | 2025 | 1.6B | AR codec LM بنمط delay | مصمم للحوار: أدوار `[S1]/[S2]` وأصوات غير كلامية |
| Chatterbox و Chatterbox Multilingual | 2025 | 0.5B | codec LM بعمود فقري Llama | رخصة MIT؛ مقبض مبالغة انفعالية؛ watermarking مدمج باسم PerTh؛ 23 لغة من ضمنها العربية |
| MegaTTS 3 | 2025 | 0.45B | latent DiT بمحاذاة متفرقة | من ByteDance؛ استنساخ عالي الجودة |
| IndexTTS2 (arXiv:2506.21619) ثم 2.5 | 2025-26 | نحو 1.5B | AR مع رموز durations صريحة | تحكم دقيق في المدد يناسب الدبلجة؛ فصل بين الطابع الصوتي والانفعال |
| Higgs Audio v2 | 2025 | 3B على Llama-3.2 | tokenizer موحّد مع محوّل DualFFN | تدرّب على نحو 10 ملايين ساعة؛ تعبيرية قوية |
| VibeVoice (من Microsoft) | 2025 | 1.5B/7B | latents متصلة عند 7.5 Hz مع رأس next-token diffusion | للمحتوى الطويل: حتى 90 دقيقة و 4 متحدثين |
| Kyutai TTS | 2025 | 1.6B | AR بتيارات مؤخَّرة (النص والصوت متزاحان زمنياً) | بث حقيقي على مستوى الكلمة دخولاً وخروجاً |
| VoxCPM (arXiv:2509.24650) | 2025 | 0.5B | بلا tokenizer إطلاقاً (تمثيل متصل) | توليد واعٍ بالسياق بلا codec منفصل |
| NeuTTS Air و Kitten TTS و PocketTTS و Supertonic | 2025-26 | من 15M إلى 750M | متنوعة | موجة النماذج على الجهاز: CPU أولاً مع بث |
| Qwen3-TTS (الإصدار المفتوح) | 2026 | 0.6B/1.7B | AR فوق رموز 12 Hz متعددة الـ codebooks | ثلاث نسخ: Base للاستنساخ، و CustomVoice بأصوات جاهزة مع style prompts، و VoiceDesign لتصميم صوت من وصف نصي؛ محاذاة بالـ DPO/GSPO؛ 10 لغات |
| Fun-CosyVoice 3 (arXiv:2505.17589) | 2025-26 | 0.5B-1.5B | AR LM مع FM | مليون ساعة تدريب، و tokenizer بإشراف متعدد المهام، و post-training بمكافأة قابلة للاشتقاق؛ 9 لغات مع 18 لهجة صينية؛ أوزان 0.5B برخصة Apache 2.0 |
| Voxtral TTS (من Mistral) | 2026 | 3.4B زائد 0.39B للـ FM مع codec | AR دلالي مع flow matching | يعمل على GPU واحدة بذاكرة 16 GB عبر vLLM؛ ترتيب Elo قوي في الساحات |
| Fish Audio S2 (arXiv:2603.08823) | 2026 | 4B زائد 0.4B بنمط dual-AR | RVQ بعشرة codebooks | تعبيرية من الطراز الأول؛ الأوزان برخصة بحثية غير تجارية |
| Step Audio EditX و NVIDIA Magpie و Maya1 | 2025-26 | من 1B إلى 3B | AR codec LMs | تخصصات: التحرير الصوتي والإنتاج والتعبيرية |

ومن المراجع المغلقة المؤثرة التي يجب أن تعرفها: ElevenLabs v3، و `gpt-4o-mini-tts` مع الـ Realtime API من OpenAI، والـ TTS الصوتي الأصيل في Gemini مع NotebookLM من Google، و Seed-TTS من ByteDance (arXiv:2406.02430 وهي الورقة التي يقيس الجميع أنفسهم عليها)، و MiniMax Speech، و Cartesia Sonic بكمونه المنخفض القائم على SSM، و Hume Octave، و Qwen-Audio-3.0-TTS وهو النموذج الرائد المتاح عبر API فقط فوق خط Qwen3-TTS المفتوح.

### 9.5 أين يقف المجال؟ اتجاهات منتصف 2026

1. **المعمارية المهيمنة هي الثنائية: LM مع flow matching.** نموذج AR أو مقنَّع فوق tokens شبه دلالية منخفضة المعدل يتولى قرارات المحتوى والـ prosody، ثم راسم صوتي بالـ flow matching. هذا الشكل تتشاركه CosyVoice و Voxtral و MegaTTS 3 وكثير من الأنظمة المغلقة.
2. **والاتجاه المضاد هو التبسيط لتيار واحد.** فكرة Llasa بنموذج Llama واحد فوق codebook واحد، و VoxCPM بلا tokenizer أصلاً، كلاهما يقول إن الـ pipeline يمكن أن ينهار أكثر.
3. **الصغير صار جيداً بشكل صادم.** نماذج بين 82M و 750M مثل Kokoro و PocketTTS و NeuTTS و Supertonic و Qwen3-TTS-0.6B تتفوق اليوم في اختبارات عمياء على أنظمة 2024 متعددة المليارات. والتشغيل على الجهاز وبالزمن الحقيقي على CPU صار أمراً عادياً.
4. **وصل عصر الـ post-training.** أساليب DPO و GSPO والـ RL بمكافآت مبنية من الـ WER وتشابه المتحدث ومتنبئات الـ MOS (كما في Seed-TTS RL والمكافأة القابلة للاشتقاق في CosyVoice 3 و Koel-TTS ومحاذاة Qwen3-TTS) صارت تفرّق بين الأنظمة العليا أكثر مما تفرّق المعمارية نفسها.
5. **الـ TTS الذي يتبع التعليمات.** تصميم أصوات من وصف نصي (Parler و VoiceDesign في Qwen3-TTS)، وتعليمات أسلوب وانفعال داخل النص، وتحرير صوتي دقيق (Step Audio EditX).
6. **الحواري والـ full-duplex.** توليد مشروط بسياق المحادثة (CSM)، ونماذج حوار كاملة (Dia)، وبث نص مع بث صوت متزامن (Kyutai)، ودمج الكلام داخل الـ omni-LLMs مثل GPT-4o و Gemini و Qwen-Omni و Moshi.
7. **السلامة صارت ميزة في المنتج.** watermarking مدمج (PerTh في Chatterbox و AudioSeal)، ومعايير مضادة للانتحال الصوتي (ASVspoof)، واستنساخ مشروط بالموافقة، وكل ذلك تحت ضغط تشريعي حقيقي بسبب الـ deepfakes.
8. **وتحوّل التقييم** من أوراق MOS إلى الساحات العامة (TTS Arena وتصنيف Elo عند Artificial Analysis) وإلى حزم zero-shot معيارية مثل seed-tts-eval و CV3-Eval.

---

## 10. كيف نقيّم أنظمة TTS؟

**التقييم البشري (ما زال المعيار الذهبي):**
- **MOS**: تقييم الطبيعية من 1 إلى 5 عبر لجان مستمعين، ويُذكر دائماً مع فاصل ثقة 95%. و **CMOS**: مقارنة مباشرة مع baseline وهو أكثر حساسية. و **SMOS**: تقييم تشابه المتحدث. و **MUSHRA**: تقييم متعدد المحفزات مع مراجع. وتحذير مهم: قيم MOS لا تُقارن بين ورقة وأخرى ولا بين لجنة وأخرى ولا بين سنة وأخرى، والأنظمة العليا أشبعت المقياس أصلاً.

**التقييم الآلي (ثلاثية الـ zero-shot الحديثة):**
- **WER أو CER**: فرّغ الصوت المولَّد بنظام ASR قوي (عادة Whisper-large-v3، أو Paraformer للصينية) وقارن بالنص الأصلي. يقيس الوضوح ويكشف الهلوسة.
- **SIM**: تشابه cosine بين speaker embeddings (نماذج ECAPA أو x-vector المبنية على WavLM) للمقطع المرجعي والصوت المولَّد. يقيس دقة الاستنساخ.
- **متنبئات الـ MOS**: أدوات UTMOS(v2) و DNSMOS و NISQA تقدّر الجودة الإدراكية بلا مستمعين.

**على مستوى الإشارة (مفيد لاختبارات regression لا للترتيب النهائي):**

$$\mathrm{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{d=1}^{D} \big(c_d - \hat{c}_d\big)^2}$$

وهو الـ mel-cepstral distortion بالديسيبل فوق frames محاذاة بالـ DTW، ومعه F0 RMSE ومعدل خطأ المجهور/المهموس وارتباط log-F0.

**مقاييس الأنظمة:** الـ RTF أي real-time factor، وكمون أول حزمة صوتية في البث، واستهلاك ذاكرة الـ GPU.

**معايير تستحق الحفظ:** seed-tts-eval للإنجليزية والصينية بالـ WER والـ SIM في وضع zero-shot، واستمرارات LibriSpeech test-clean وهي إعداد VALL-E الأصلي، و CV3-Eval متعدد اللغات وواقعي، و EmergentTTS-Eval للانفعالات وما وراء اللغة والنصوص المعقدة، والساحات العامة. وخذها قاعدة: أي تقرير عن نموذج TTS في 2026 بلا WER و SIM واختبار استماع بشري ليس تقريراً جاداً.

---

## 11. ملاحظات خاصة بالعربية

قسم قصير لكنه مهم، لأن العربية هي المكان الذي تصطدم فيه معرفة الـ TTS العامة بواقع الـ frontend الصعب:

- **التشكيل هو المشكلة المركزية.** النص الفصيح غير المشكول لا يكفي لتحديد النطق: الحركات القصيرة وعلامات الإعراب والشدة كلها غائبة عن الكتابة. لذلك فإن أي pipeline عربي إنتاجي هو عملياً: تطبيع نص، ثم تشكيل آلي، ثم G2P شبه منتظم، ثم بقية الكومة الصوتية. وأي خطأ من المشكِّل يتحول مباشرة إلى خطأ نطق في الصوت النهائي. الخلاصة العملية: قِس جودة المشكِّل عندك (CATT وأمثاله من المشكِّلات العصبية) بنفس الجدية التي تقيس بها نموذج الـ TTS نفسه.
- **فونولوجيا تُتعب النماذج:** الأصوات المطبقة (ص ض ط ظ) وتأثيرها على الحركات المجاورة، والشدة بوصفها ظاهرة مدة زمنية، والفرق بين الحركات الطويلة والقصيرة، والهمزة بأشكالها، وإدغام لام التعريف. النماذج ذات الـ durations الصريحة، وأنظمة LM-TTS القابلة للتحكم في المدد على طريقة IndexTTS2، مفيدة هنا فائدة حقيقية ملموسة.
- **اللهجات بلا معيار كتابي**، والصوت اللهجي بجودة تصلح للـ TTS نادر، والـ code-switching بين العربية والإنجليزية موجود في كل المدخلات الواقعية. ولهذا فإن التسجيل المنسَّق مع الـ annotation الدقيق هو الخندق التنافسي الحقيقي في منتجات الكلام العربية، لأن البيانات العامة معظمها إما فصحى أو بجودة ASR فقط.
- **البيانات المتاحة:** ASC أي Arabic Speech Corpus (من Nawar Halabi، نحو 3.7 ساعة فصحى استوديو، وهو الكلاسيكي)، و ClArTTS (نحو 12 ساعة عربية تراثية)، و Arabic Common Voice، ثم المدونات الكبيرة بجودة ASR مثل MASC و QASR و SADA للهجات السعودية، وهذه الأخيرة تنفع للـ pretraining أو لتدريب الـ tokenizer لكنها ليست أهدافاً نظيفة للـ TTS.
- **النماذج التي تدعم العربية اليوم:** XTTS v2 بسبع عشرة لغة منها العربية وقابل للـ fine-tuning، و MMS-TTS القائم على VITS بأكثر من 1100 لغة برخصة CC-BY-NC، و Chatterbox Multilingual بثلاث وعشرين لغة منها العربية برخصة MIT، وخط OpenAudio/Fish مع مراجعة بطاقة كل إصدار، إضافة إلى عدد كبير من الـ fine-tunes العربية المجتمعية لنموذجي F5-TTS و XTTS على Hugging Face. وهناك ArTST من MBZUAI، وهو نموذج عربي مدرَّب مسبقاً على طراز SpeechT5 يغطي TTS و ASR للفصحى واللهجات. ولاحظ نقطة مهمة: عدة أنظمة عليا في 2026 لا تذكر العربية ضمن لغاتها، فلغات CosyVoice 3 التسع ولغات Qwen3-TTS العشر كلها بلا عربية. لذلك تبقى الاستراتيجية الرابحة هي fine-tuning لقاعدة مفتوحة قوية على بيانات عربية منسَّقة بعناية.

---

## 12. مجموعات البيانات

| المجموعة | الحجم | ملاحظات |
|---|---|---|
| LJSpeech | 24 ساعة، متحدث واحد (إنجليزي) | بمثابة MNIST في عالم الـ TTS؛ كل الوصفات الكلاسيكية تستخدمها |
| VCTK | 44 ساعة، 109 متحدثين | إنجليزية متعددة المتحدثين واللكنات |
| LibriTTS / LibriTTS-R | 585 ساعة | متعددة المتحدثين؛ النسخة -R هي المنظَّفة المرمَّمة |
| Hi-Fi TTS | 292 ساعة، 10 متحدثين | جودة عالية متعددة المتحدثين |
| Expresso | 47 ساعة | أساليب وانفعالات تعبيرية وحوارية |
| GigaSpeech و MLS و Libriheavy | من 10 إلى 60 ألف ساعة | جودة ASR؛ مناسبة لمرحلة الـ pretraining |
| Emilia (مع Emilia-Large) | من 101 ألف إلى أكثر من 200 ألف ساعة | صوت واقعي بست لغات؛ الخيار الافتراضي الحديث للـ pretraining (تدرّب عليها F5 و MaskGCT وغيرهما) |
| WenetSpeech4TTS | أكثر من 12 ألف ساعة (صيني) | صيني مُرشَّح خصيصاً للـ TTS |
| ASC و ClArTTS و Arabic Common Voice | 3.7 ساعة / 12 ساعة / متفاوت | العربية (راجع القسم 11) |

وملاحظة صارت قيداً هندسياً من الدرجة الأولى: ترخيص البيانات وموافقة المتحدثين. راجع الشروط قبل أي تدريب تجاري، واحتفظ بسجلات مصدر كل عينة.

---

## 13. الدورات المفتوحة والكتب

**الدورات (مجانية وكاملة):**

1. **Hugging Face Audio Course**: تطبيق عملي للـ transformers على الصوت، والوحدة السادسة كلها عن الـ TTS (نماذج SpeechT5 و MMS و Bark مع fine-tuning للغة جديدة). أسرع طريق من الصفر إلى أول fine-tuning ناجح. الرابط: https://huggingface.co/learn/audio-course (وحدة الـ TTS: https://huggingface.co/learn/audio-course/chapter6/introduction)
2. **Speech Processing** (بإشراف Simon King وفريقه في جامعة Edinburgh، بفيديوهات وقراءات ومعامل كاملة على الإنترنت): الإشارات والصوتيات ونموذج الـ source-filter والـ frontend وتوليد الموجة الكلاسيكي. الرابط: https://speech.zone/courses/speech-processing/
3. **Speech Synthesis** (من Edinburgh أيضاً وهي الجزء الثاني): الـ unit selection بعمق، ثم التوليف بالـ HMM، ثم التوليف العصبي حتى أحدث ما وصل إليه المجال، مع مشروع عملي تبني فيه صوتاً كاملاً. ببساطة أفضل دورة TTS من المبادئ الأولى موجودة على الإطلاق. الرابط: https://speech.zone/courses/speech-synthesis/
4. **Stanford CS224S بعنوان Spoken Language Processing**: محاضرات وواجبات تغطي الـ ASR والـ TTS والحوار. الرابط: https://web.stanford.edu/class/cs224s/
5. **دورات CMU الكلامية (11-751 وما حولها)**: موادها متاحة، وهي قوية في أسس جانب التعرف (الـ alignment والـ HMMs) التي يعيد الـ TTS استخدامها.

**الكتب:**

- **كتاب Paul Taylor بعنوان Text-to-Speech Synthesis** (مطبعة Cambridge، عام 2009): مرجع الحقبة الكلاسيكية بلا منازع، وفصوله عن الـ frontend والـ unit selection لم يتفوق عليها شيء حتى الآن.
- **كتاب Xu Tan بعنوان Neural Text-to-Speech Synthesis** (دار Springer، عام 2023): الحقبة العصبية بطول كتاب كامل، من مؤلف FastSpeech نفسه.
- **كتاب Jurafsky و Martin بعنوان Speech and Language Processing** (مسودة الطبعة الثالثة، مجانية): فصل الـ ASR والـ TTS فيه مدخل لطيف ومحدَّث. الرابط: https://web.stanford.edu/~jurafsky/slp3/

**المسوح البحثية:**

- ورقة Tan وفريقه بعنوان A Survey on Neural Speech Synthesis (arXiv:2106.15561): الخريطة الكاملة للفترة من 2016 إلى 2021.
- ورقة Xie وفريقه بعنوان Towards Controllable Speech Synthesis in the Era of Large Language Models (منشورة في EMNLP 2025، arXiv:2412.06602): خريطة عصر الـ LLM والـ diffusion، ومعها قائمة مرافقة تُحدَّث باستمرار: https://github.com/imxtx/awesome-controllable-speech-synthesis
- ورقة Zen و Tokuda و Black بعنوان Statistical Parametric Speech Synthesis (مجلة Speech Communication، عام 2009): حقبة الـ SPSS بأقلام من بنوها.

---

## 14. الأدوات والمستودعات

| الأداة | استخدمها من أجل |
|---|---|
| **ESPnet** (github.com/espnet/espnet) | وصفات قابلة لإعادة الإنتاج: Tacotron 2 و FastSpeech 2 و VITS و JETS على عشرات المدونات |
| **Amphion** (github.com/open-mmlab/Amphion) | وصفات بحثية حديثة: MaskGCT والـ vocoders وتحويل الصوت، ومعها أدوات Emilia |
| **نسخة Coqui المصانة** (github.com/idiap/coqui-ai-TTS) | استدلال XTTS v2 والـ fine-tuning بخطوط جاهزة |
| **NeMo** (من NVIDIA) | تدريب على مستوى الإنتاج: FastPitch و HiFi-GAN و BigVGAN و Magpie |
| **SpeechBrain و PaddleSpeech** | بدائل متكاملة التجهيز بوصفات TTS |
| **Montreal Forced Aligner** | محاذاة على مستوى الـ phoneme للنماذج القائمة على الـ durations |
| **مستودعات Matcha-TTS و F5-TTS و CosyVoice و Qwen3-TTS** | التنفيذات المرجعية التي تستحق القراءة سطراً سطراً: كل واحد منها درس مكثف في مدرسته |
| **مستودعات descript-audio-codec و encodec و vocos و BigVGAN** | لبنات بناء الـ codecs والـ vocoders |

ولمتابعة المجال أولاً بأول: ساحتا TTS Arena و Artificial Analysis Speech Arena للترتيب، والنماذج الرائجة تحت وسم `text-to-speech` على Hugging Face، ووقائع مؤتمرات Interspeech و ICASSP و SSW، وقسما `eess.AS` و `cs.SD` على arXiv.

---

## 15. قائمة القراءة

مرتبة ترتيباً منهجياً. أوراق العمود الفقري معلَّمة بنجمة ★: اقرأ هذه الاثنتي عشرة وحدها بعناية وستفهم المجال، والبقية تضيف عمقاً في كل موضوع على حدة.

**الأسس والحقبة الكلاسيكية**
1. ★ ورقة Zen و Tokuda و Black بعنوان Statistical Parametric Speech Synthesis (عام 2009): نظرة العالم التوليفية قبل العصر العصبي
2. ورقة Hunt و Black بعنوان Unit Selection in a Concatenative Speech Synthesis System (مؤتمر ICASSP 1996)
3. ورقة Tokuda وفريقه بعنوان Speech Synthesis Based on Hidden Markov Models (مجلة Proc. IEEE 2013)

**الموجة العصبية**
4. ★ WaveNet: arXiv:1609.03499
5. ★ Tacotron 2: arXiv:1712.05884 (ومعها Tacotron 1 في arXiv:1703.10135)
6. Transformer TTS: arXiv:1809.08895
7. ★ FastSpeech 2: arXiv:2006.04558 (ومعها FastSpeech في arXiv:1905.09263)

**الـ vocoders**
8. ★ HiFi-GAN: arXiv:2010.05646
9. MelGAN: arXiv:1910.06711 و Parallel WaveGAN: arXiv:1910.11480
10. WaveGlow: arXiv:1811.00002 و WaveRNN: arXiv:1802.08435
11. BigVGAN: arXiv:2206.04658 و Vocos: arXiv:2306.00814

**النماذج الاحتمالية end-to-end**
12. ★ Glow-TTS: arXiv:2005.11129
13. ★ VITS: arXiv:2106.06103 (ثم VITS2 في arXiv:2307.16430 و YourTTS في arXiv:2112.02418)
14. NaturalSpeech: arXiv:2205.04421

**الـ diffusion والـ flow matching**
15. DDPM: arXiv:2006.11239 و Score SDE: arXiv:2011.13456
16. ★ Grad-TTS: arXiv:2105.06337 (ومعها DiffWave في arXiv:2009.09761)
17. ★ Flow Matching: arXiv:2210.02747
18. Matcha-TTS: arXiv:2309.03199 و Voicebox: arXiv:2306.15687
19. ★ F5-TTS: arXiv:2410.06885 (ومعها E2-TTS في arXiv:2406.18009)

**الـ codecs ونماذج الكلام اللغوية**
20. VQ-VAE: arXiv:1711.00937
21. ★ SoundStream: arXiv:2107.03312 و EnCodec: arXiv:2210.13438 و DAC: arXiv:2306.06546
22. AudioLM: arXiv:2209.03143
23. ★ VALL-E: arXiv:2301.02111 (ثم VALL-E 2 في arXiv:2406.05370)
24. SPEAR-TTS: arXiv:2302.03540 و SoundStorm: arXiv:2305.09636 و MaskGCT: arXiv:2409.00750
25. Tortoise: arXiv:2305.07243 و StyleTTS 2: arXiv:2306.07691

**الأنظمة الحديثة (اختر حسب اهتمامك)**
26. Seed-TTS: arXiv:2406.02430 (واضع المعايير التي يقيس عليها الجميع)
27. ★ CosyVoice 2: arXiv:2412.10117 و CosyVoice 3: arXiv:2505.17589
28. Llasa: arXiv:2502.04128 (قوانين الـ scaling لأنظمة TTS بعقلية الـ LLM)
29. Spark-TTS: arXiv:2503.01710 و Fish Speech: arXiv:2411.01156 و Fish Audio S2: arXiv:2603.08823
30. IndexTTS2: arXiv:2506.21619 و VoxCPM: arXiv:2509.24650
31. المسوح: arXiv:2106.15561 و arXiv:2412.06602 (تُقرأ بموازاة كل ما سبق)

---

## 16. خطة دراسة من 8 أسابيع

| الأسبوع | الموضوع | اقرأ | شاهد | ابنِ |
|---|---|---|---|---|
| 1 | الإشارات والـ TTS الكلاسيكي | الأقسام 1-3 هنا، وفصل Taylor عن الـ frontend، والورقة 1 | وحدتا 5-6 من Speech Processing على speech.zone | نفّذ STFT ثم mel ثم Griffin-Lim من الصفر بـ NumPy |
| 2 | الـ unit selection والـ SPSS | الورقتان 2-3 | الوحدات 1-4 من Speech Synthesis على speech.zone | شغّل Festival، واشتق حل الـ MLPG بيدك على الورق |
| 3 | الـ TTS العصبي الـ AR | الأوراق 4-6 | الوحدات 1-3 من HF Audio Course | درّب Tacotron 2 على LJSpeech عبر ESPnet، ثم اكسر الـ attention عمداً بنص طويل ولاحظ أنماط الفشل |
| 4 | الـ NAR والـ vocoders | الأوراق 7-11 | الوحدة 6 من HF Audio Course | درّب FastSpeech 2 مع HiFi-GAN وقارن A/B مع نتيجة الأسبوع 3 |
| 5 | الـ flows و VITS | الأوراق 12-14 | وحدة أحدث ما في المجال على speech.zone | اعمل fine-tuning لنموذج VITS أو Piper على صوت جديد وافحص محاذاات الـ MAS بعينك |
| 6 | الـ diffusion والـ FM | الأوراق 15-19 | | درّب Matcha-TTS على LJSpeech، ثم اعمل fine-tuning لنموذج F5-TTS على لغتك وقِس WER و SIM |
| 7 | الـ codecs ونماذج الكلام | الأوراق 20-25 | | رمّز وفك الصوت بـ EnCodec و DAC عند عدة معدلات bitrate، وارسم معدل الـ tokens مقابل الجودة، وشغّل CSM أو Spark-TTS محلياً |
| 8 | الأنظمة الحديثة والتقييم | الأوراق 26-31 | | اعمل fine-tuning لنموذج CosyVoice 2 أو XTTS على بيانات منسَّقة، وابنِ منظومة تقييم كاملة (WER بـ Whisper مع SIM بـ WavLM مع UTMOS)، وقارن 3 نماذج مفتوحة على مجموعة اختبارك الخاصة |

وقبل أن تغلق الدليل، خذ معك نمط المجال كله في فقرة واحدة: كل جيل يحل عنق زجاجة الجيل السابق (الطبيعية، ثم متانة الـ alignment، ثم السرعة، ثم نمذجة التوزيع، ثم عمومية الـ zero-shot، ثم التحكم والكمون)، والأدوات المهجورة تعود دائماً بثياب جديدة. بحث Viterbi صار MAS، ومحاذاة الـ HMM القسرية هي التي تدرّب FastSpeech، ونموذج الـ source-filter مختبئ داخل الـ period discriminators في HiFi-GAN، وفكرة الـ MLPG عن "مسار ناعم تحت نموذج احتمالي" هي، لو تأملتها قليلاً، ما تفعله الـ ODE في الـ flow matching. تعلّم التاريخ ولن تبدو لك أحدث ورقة سحراً بعد اليوم.

---

*جُمع في أغسطس 2026. التصحيحات والمساهمات موضع ترحيب.*
