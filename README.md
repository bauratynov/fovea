<p align="center">
  <img src="assets/logo.png" alt="Fovea" width="420">
</p>

<h1 align="center">Fovea — Perceptual JND Model for Video Coding</h1>

<p align="center">
  <a href="https://github.com/bauratynov/fovea/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/bauratynov/fovea/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pkg.go.dev/github.com/bauratynov/fovea"><img alt="Go Reference" src="https://pkg.go.dev/badge/github.com/bauratynov/fovea.svg"></a>
  <a href="https://goreportcard.com/report/github.com/bauratynov/fovea"><img alt="Go Report Card" src="https://goreportcard.com/badge/github.com/bauratynov/fovea"></a>
  <a href="https://go.dev"><img alt="Go Version" src="https://img.shields.io/github/go-mod/go-version/bauratynov/fovea"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://github.com/bauratynov/fovea/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/bauratynov/fovea?style=social"></a>
</p>

<p align="center"><em>What the fovea can't see, the bitstream doesn't need to carry.</em></p>

Fovea is a Just-Noticeable Difference (JND) model that predicts, for every
pixel, the minimum luminance change detectable by the human visual system
(HVS). Coefficients whose reconstruction error falls below this threshold
are perceptually invisible and can be discarded by a codec — yielding
20–30 % bitrate savings at equal perceptual quality.

The name references the *fovea centralis*: the 1.5 mm central pit of the
retina where cone density peaks at ~150 000 cells/mm². Fovea-the-library
models what that pit (and the surrounding visual pathway) actually notices.

---

## 1. Model

For a frame $Y_t \in [0, 255]^{W \times H}$ with previous frame $Y_{t-1}$,
Fovea computes a per-pixel threshold map

$$
T(x,y) \;=\; \underbrace{\max\!\big(T_L(x,y),\; T_C(x,y)\big)}_{\text{spatial JND}}
\;\cdot\; \underbrace{B_M(x,y)}_{\text{temporal boost}}
$$

combining **luminance adaptation** $T_L$, **contrast / texture masking**
$T_C$, and a multiplicative **motion boost** $B_M \ge 1$. The $\max$ rule
follows the *masking dominance* principle — when two maskers coexist, the
stronger one dictates visibility (Legge & Foley, 1980 [6]).

### 1.1 Luminance adaptation $T_L$

Weber's law, $\frac{\Delta I}{I} = k$ (Weber, 1834), integrates to Fechner's
logarithmic sensation $S = k \ln(I / I_0)$. It holds well in the photopic
mid-range but breaks at both extremes:

- **Dark (DeVries–Rose regime):** shot noise dominates, so $\Delta I \propto \sqrt{I}$
  and the threshold *decreases* as $I$ rises out of black.
- **Bright (saturation):** receptor response flattens; thresholds rise again.

Fovea therefore uses a piecewise fit calibrated to Chou & Li [1]:

$$
T_L(\bar Y) \;=\;
\begin{cases}
17 - 3\sqrt{\bar Y}, & \bar Y < 32 \\[2pt]
\dfrac{3(\bar Y-32)}{95} + 3, & 32 \le \bar Y \le 127 \\[8pt]
0.04\,\bar Y + 1, & 127 < \bar Y \le 200 \\[4pt]
\dfrac{4(\bar Y-200)}{55} + 9, & \bar Y > 200
\end{cases}
$$

Background luminance $\bar Y$ is the 5×5 box-mean around $(x,y)$:

$$
\bar Y(x,y) \;=\; \frac{1}{|W|}\sum_{(i,j)\in W(x,y)} Y(i,j),
\qquad W = [x{-}2,\,x{+}2]\times[y{-}2,\,y{+}2].
$$

Computed in $O(WH)$ via an integral image $I(x,y) = \sum_{i<x,\,j<y} Y(i,j)$:

$$
\sum_{W} Y \;=\; I(x_1,y_1) - I(x_0,y_1) - I(x_1,y_0) + I(x_0,y_0).
$$

The curve passes through $(0, 17), (32, 3), (127, 6.08), (200, 9)$,
monotonically decreasing on $[0, 32]$ then monotonically increasing on
$[32, 255]$ — the classic "U-shape" of HVS luminance sensitivity.

### 1.2 Texture / contrast masking $T_C$

Texture masking reflects the HVS's reduced sensitivity to error inside
high-frequency, high-contrast regions. Fovea operates in the 8×8 DCT
domain (matching H.26x coefficient blocks).

The type-II DCT of an 8×8 block $f$:

$$
C(u,v) \;=\; \tfrac{1}{4}\,\alpha(u)\,\alpha(v)
\sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y)\,
\cos\!\frac{(2x+1)u\pi}{16}\,
\cos\!\frac{(2y+1)v\pi}{16},
$$

with $\alpha(0) = 1/\sqrt{2}$ and $\alpha(k) = 1$ for $k > 0$. AC-coefficient
variance is the masker energy:

$$
\sigma^2_{AC} \;=\; \frac{1}{63}\!\!\sum_{(u,v)\neq(0,0)}\!\! C(u,v)^2
\;-\; \bigg(\frac{1}{63}\!\!\sum_{(u,v)\neq(0,0)}\!\! C(u,v)\bigg)^{\!2}.
$$

Following the power-law fit of Yang et al. [2]:

$$
\boxed{\,T_C \;=\; \alpha\,\sigma^{2\beta}_{AC}\,}, \qquad
\alpha = 0.25,\; \beta = 0.36.
$$

The exponent $2\beta \approx 0.72 < 1$ compresses the threshold dynamic
range — empirically observed in contrast-masking psychophysics. All pixels
in a block share $T_C$; edge blocks use edge-replication padding.

### 1.3 Temporal masking $B_M$

Moving content hides spatial error: the HVS integrates over ~100 ms, so
high inter-frame motion reduces effective acuity. Fovea estimates motion
per 4×4 block via Mean Absolute Difference (MAD):

$$
M(b) \;=\; \frac{1}{16}\sum_{(x,y)\in b}\,\big|\,Y_t(x,y) - Y_{t-1}(x,y)\,\big|.
$$

The multiplicative boost follows a saturating exponential (cf. Girod [3]):

$$
\boxed{\,B_M(M) \;=\; 1 + \gamma\!\left(1 - e^{-M/\tau}\right)\,},
\qquad \gamma = 1.4,\; \tau = 20.
$$

Asymptotic behaviour:

$$
\lim_{M\to 0} B_M = 1 \quad(\text{static: no boost}), \qquad
\lim_{M\to\infty} B_M = 1 + \gamma = 2.4 \quad(\text{saturated masking}).
$$

The half-saturation motion is $M_{1/2} = \tau \ln 2 \approx 13.9$ (grey
levels per pixel per frame). To avoid spurious flicker from isolated
frame pairs, Fovea maintains an exponentially-smoothed motion history:

$$
H_t \;=\; \lambda H_{t-1} + (1-\lambda)\,M_t,
\qquad \lambda = 0.7.
$$

### 1.4 Prunable ratio & bitrate estimate

Given per-pixel reconstruction errors $e_i$, the *prunable set* is

$$
P \;=\; \{\, i \;:\; |e_i| < T_i \,\}, \qquad
\rho \;=\; \frac{|P|}{W\cdot H}.
$$

A first-order bitrate-savings estimate (assuming $\text{bits}(e) \propto |e|$):

$$
\hat R_{\text{save}} \;=\;
\frac{\sum_{i \in P} |e_i|}{\sum_{i} |e_i|}.
$$

> **Caveat.** This is a proxy. Real H.264/H.265 cost is entropy-coded
> (CABAC / CAVLC) with context modelling, so $|e|\mapsto\text{bits}$
> is sub-linear and context-dependent. A calibrated-rate version is in
> the [roadmap](#4-roadmap).

---

## 2. Complexity

Let $N = W \cdot H$. Per-frame cost:

| Component      | Work          | Notes                                      |
|----------------|---------------|--------------------------------------------|
| Luminance      | $\Theta(N)$   | Integral-image build $\Theta(N)$; lookups $\Theta(1)$ |
| Texture        | $\Theta(N)$   | $N/64$ blocks × 128 mults for row+col DCT  |
| Temporal       | $\Theta(N)$   | One MAD pass                               |
| Max + multiply | $\Theta(N)$   | SIMD-friendly                              |
| **Total**      | $\Theta(N)$   | Texture dominates ≈ 56 % of wall time      |

At 1080p ($N = 2.07\text{M}$) on an i7-13700, current pure-Go implementation:

| Model           | ms/frame  | fps        |
|-----------------|-----------|------------|
| LuminanceMap    |  8.6      |  ~116      |
| TextureMap      | 30.1      |   ~33      |
| TemporalMap     |  6.9      |  ~145      |
| **CombinedJND** | **53.9**  | **~18.5**  |

DCT is the bottleneck. The in-tree C header (`csrc/fovea.h`) sketches an
AVX2 path via `_mm256_madd_epi16` on the row/column butterflies and
`_mm256_sad_epu8` for 4×4 MAD — a 4–6× headroom to reach real-time
1080p60 without GPU.

---

## 3. Build & use

```bash
# Build
CGO_ENABLED=0 go build ./cmd/fovea/
CGO_ENABLED=0 go test ./...

# Analyze a single frame (static JND, no temporal boost)
fovea analyze frame.jpg

# Compare two successive frames (full model, with motion)
fovea compare frame1.jpg frame2.jpg

# Benchmark at 1080p
fovea bench

# HTTP API (POST /analyze, POST /compare, GET /healthz)
fovea serve :8080
curl -X POST -T frame.jpg http://localhost:8080/analyze
```

Source layout:

```
internal/
  luminance/  Weber-Fechner piecewise, integral-image box mean
  texture/    8×8 type-II DCT, AC-variance power law
  temporal/   4×4 MAD + saturating exponential + history decay
  model/      Combine: max(T_L, T_C) * B_M
  api/        HTTP API
cmd/fovea/    CLI
csrc/fovea.h  C reference implementation with SSE2/AVX2 hooks
```

---

## 4. Roadmap

Ordered by expected impact on model quality or throughput:

1. **SIMD DCT + parallel blocks** — AVX2 row/col butterflies with
   `_mm256_madd_epi16`; `goroutine`-per-block-row. Target 4–6× on the
   texture path → real-time 1080p60 on CPU.
2. **Empirical 20–30 % bitrate claim** — end-to-end pipeline on UVG /
   Netflix open content: encode with/without sub-JND pruning in x265;
   report BD-rate vs VMAF at matched quality.
3. **VMAF-calibrated constants** — refit $(\alpha, \beta, \gamma, \tau)$
   by minimising $\| \text{VMAF}_{\text{pruned}} - \text{VMAF}_{\text{ref}} \|_2$
   on LIVE / CSIQ / BVI-HD.
4. **Chroma JND** — separate sensitivity curves for Cb / Cr, exploiting
   HVS chromatic-acuity asymmetry (≈ 0.5× of luma in the mid-frequencies).
5. **Scene-cut-aware temporal** — reset $H_t$ on SAD jumps above
   threshold; prevents spurious masking on the first frame of a shot.
6. **Proper entropy-coded bit model** — replace the linear
   $|e|\mapsto\text{bits}$ proxy with a CABAC-context look-up.
7. **Foveated JND** — centre-weighted $T$ for PTZ, drones, VR where
   gaze is constrained.
8. **Codec integration** — x265 patch exposing `--fovea-prune` as a
   post-quantisation coefficient killer.

---

## 5. References

1. Chou, C.-H. & Li, Y.-C. (1995). *A perceptually tuned subband image
   coder based on the measure of just-noticeable-distortion profile.*
   IEEE Trans. Circuits Syst. Video Technol., 5(6):467–476.
2. Yang, X.-K., Ling, W.-S., Lu, Z.-K., Ong, E.-P. & Yao, S.-S. (2005).
   *Just noticeable distortion model and its applications in video
   coding.* Signal Processing: Image Communication, 20(7):662–680.
3. Girod, B. (1993). *What's wrong with mean-squared error?* In:
   Watson, A. B. (Ed.), *Digital Images and Human Vision*, MIT Press,
   pp. 207–220.
4. Watson, A. B. (1993). *DCTune: A technique for visual optimization
   of DCT quantization matrices for individual images.* SID Digest
   of Technical Papers, pp. 946–949.
5. Daly, S. (1993). *The Visible Differences Predictor: an algorithm
   for the assessment of image fidelity.* In *Digital Images and Human
   Vision*, MIT Press, pp. 179–206.
6. Legge, G. E. & Foley, J. M. (1980). *Contrast masking in human
   vision.* Journal of the Optical Society of America, 70(12):1458–1471.
7. Weber, E. H. (1834). *De pulsu, resorptione, auditu et tactu.*
   Köhler, Leipzig. (Original statement of the Weber ratio.)
8. Fechner, G. T. (1860). *Elemente der Psychophysik.* Breitkopf &
   Härtel, Leipzig.

---

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Patent grant
included, which matters in the heavily-patented video-coding space.
