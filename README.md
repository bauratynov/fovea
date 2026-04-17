# Fovea — Perceptual JND Model for Video Coding

Fovea is a Just-Noticeable Difference (JND) model that predicts the minimum
pixel-level change visible to the human eye. Sub-JND errors can be discarded
by a codec, achieving 20–30 % bitrate savings with zero perceived quality
loss.

The name comes from the *fovea centralis* — the tiny central region of the
retina responsible for sharp vision. Fovea the library models what that
region (and the rest of the visual system) actually notices, so codecs can
stop wasting bits on what it doesn't.

## Architecture

```
internal/
  luminance/  — Weber-Fechner luminance adaptation + dark/bright extremes
  texture/    — DCT-based texture masking (8×8 blocks)
  temporal/   — Motion-magnitude temporal masking (4×4 blocks, exp decay)
  model/      — Combined: JND = max(luminance, texture) × temporal_boost
  api/        — HTTP API (POST /analyze, POST /compare, GET /healthz)
cmd/fovea/    — CLI (analyze, compare, bench, serve)
csrc/fovea.h  — C header with SSE2/AVX2 SIMD annotations
```

## Build

```bash
CGO_ENABLED=0 go build ./cmd/fovea/
CGO_ENABLED=0 go test ./...
```

## Usage

```bash
# Analyze single frame
fovea analyze frame.jpg

# Compare two frames (temporal masking included)
fovea compare frame1.jpg frame2.jpg

# Benchmark at 1080p
fovea bench

# HTTP API
fovea serve :8080
curl -X POST -T frame.jpg http://localhost:8080/analyze
```

## Performance (1080p, i7-13700)

| Model         | ms/frame |
|---------------|----------|
| LuminanceMap  |  8.6     |
| TextureMap    | 30.1     |
| TemporalMap   |  6.9     |
| CombinedJND   | 53.9     |

## Models

**Luminance adaptation** — piecewise Weber-Fechner with dark/bright extremes.
5×5 box mean via integral image.

**Texture masking** — full 8×8 DCT, AC coefficient variance. High texture
→ high masking → more prunable bits.

**Temporal masking** — `1 + γ(1 - e^(-motion/τ))` with γ=1.4, τ=20.
4×4 block MAD between frames. MotionHistory with exponential decay.

**Combined** — `max(lum, tex) × temporal_boost`. PrunableRatio and
BitrateReduction computed from per-pixel thresholds vs actual errors.

## Roadmap

- SIMD native backend (AVX2 DCT, parallel block processing)
- Chroma (Cb/Cr) JND with separate sensitivity curves
- VMAF-calibrated threshold tuning on LIVE / CSIQ / BVI-HD datasets
- Scene-cut-aware temporal masking (reset decay on cuts)
- Foveated JND (center-weighted for surveillance PTZ, drones, VR)
- Codec integration: x265 patch with `--fovea-prune` coefficient pass
- End-to-end bench pipeline: VMAF at equal bitrate vs baseline encoder

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
