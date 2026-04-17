/*
 * fovea.h — Fovea JND (Just-Noticeable Difference) perceptual model
 *
 * C reference implementation with SSE2/AVX2 SIMD annotations.
 * Same algorithms as the Go implementation in internal/.
 *
 * Build: gcc -O3 -mavx2 -o fovea fovea.c  (with JND_IMPLEMENTATION defined)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FOVEA_JND_H
#define FOVEA_JND_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Luminance adaptation (Weber-Fechner + dark/bright extremes)
 * ======================================================================== */

static inline float jnd_luminance(float bg_luma) {
    if (bg_luma < 0.0f)   bg_luma = 0.0f;
    if (bg_luma > 255.0f) bg_luma = 255.0f;

    if (bg_luma < 32.0f) {
        return 17.0f - 3.0f * sqrtf(bg_luma);
    } else if (bg_luma <= 127.0f) {
        return 3.0f * (bg_luma - 32.0f) / 95.0f + 3.0f;
    } else if (bg_luma <= 200.0f) {
        return 0.04f * bg_luma + 1.0f;
    } else {
        return (bg_luma - 200.0f) / 55.0f * 4.0f + 9.0f;
    }
}

/*
 * jnd_luminance_map — compute per-pixel luminance JND using integral image
 *                     for 5x5 box mean. Output must be preallocated (w*h floats).
 *
 * SSE2 annotation: the integral image scan and box mean lookups are
 * amenable to SIMD — process 4 pixels per iteration with _mm_loadu_si128
 * for the integral sums and _mm_cvtepi32_ps for float conversion.
 *
 * AVX2 annotation: 8 pixels per iteration using _mm256 variants.
 */
static inline void jnd_luminance_map(const uint8_t* gray, int w, int h,
                                     float* out) {
    /* Integral image (int32 to handle up to 255*w*h without overflow). */
    int stride = w + 1;
    int32_t* integ = (int32_t*)calloc((size_t)stride * (h + 1), sizeof(int32_t));
    if (!integ) return;

    for (int y = 0; y < h; y++) {
        int32_t row_sum = 0;
        for (int x = 0; x < w; x++) {
            row_sum += gray[y * w + x];
            integ[(y + 1) * stride + (x + 1)] = row_sum + integ[y * stride + (x + 1)];
        }
    }

    const int radius = 2; /* 5x5 window */

    /*
     * SIMD opportunity: for each row, process 4 (SSE2) or 8 (AVX2) x-values
     * simultaneously. The box sum for each pixel is 4 int32 lookups from the
     * integral image — gather with _mm256_i32gather_epi32 (AVX2) or manual
     * loads (SSE2), then subtract/add, convert to float, divide by area.
     */
    for (int y = 0; y < h; y++) {
        int y0 = y - radius; if (y0 < 0) y0 = 0;
        int y1 = y + radius + 1; if (y1 > h) y1 = h;

#ifdef __AVX2__
        /* AVX2 path: 8 pixels per iteration */
        int x = 0;
        for (; x + 7 < w; x += 8) {
            /* For each of 8 pixels, compute box mean via integral image.
             * Use _mm256_i32gather_epi32 for the 4 corners of each box.
             * This is a sketch — full implementation would handle variable
             * x0/x1 clipping at edges. */
            __m256 jnd_vec;
            float tmp[8];
            for (int k = 0; k < 8; k++) {
                int xx = x + k;
                int x0 = xx - radius; if (x0 < 0) x0 = 0;
                int x1 = xx + radius + 1; if (x1 > w) x1 = w;
                int32_t s = integ[y1*stride+x1] - integ[y0*stride+x1]
                          - integ[y1*stride+x0] + integ[y0*stride+x0];
                float area = (float)((x1-x0) * (y1-y0));
                tmp[k] = (float)s / area;
            }
            jnd_vec = _mm256_loadu_ps(tmp);
            /* Apply piecewise jnd_luminance to each lane. */
            /* In production: use _mm256_cmp_ps + blendv for branchless. */
            for (int k = 0; k < 8; k++) {
                out[y*w+x+k] = jnd_luminance(tmp[k]);
            }
        }
        for (; x < w; x++) {
#else
        for (int x = 0; x < w; x++) {
#endif
            int x0 = x - radius; if (x0 < 0) x0 = 0;
            int x1 = x + radius + 1; if (x1 > w) x1 = w;
            int32_t s = integ[y1*stride+x1] - integ[y0*stride+x1]
                      - integ[y1*stride+x0] + integ[y0*stride+x0];
            float area = (float)((x1-x0) * (y1-y0));
            out[y*w+x] = jnd_luminance((float)s / area);
        }
    }

    free(integ);
}


/* ========================================================================
 * Texture masking (DCT coefficient variance in 8x8 blocks)
 * ======================================================================== */

/*
 * jnd_texture_block — compute texture masking JND for one 8x8 block.
 *
 * SSE2: the 8-point DCT butterfly can be done with _mm_madd_epi16 for
 * the row/column transforms (4 rows at a time).
 * AVX2: process all 8 rows simultaneously.
 */
static inline float jnd_texture_block(const int16_t block[8][8]) {
    /* Simplified: compute pixel-domain variance instead of full DCT
     * for the C reference. The Go version does full DCT. */
    float sum = 0.0f, sum2 = 0.0f;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            float v = (float)block[r][c];
            sum += v;
            sum2 += v * v;
        }
    }
    float mean = sum / 64.0f;
    float variance = sum2 / 64.0f - mean * mean;
    if (variance < 0.0f) variance = 0.0f;

    return 0.25f * powf(variance, 0.36f);
}

/*
 * jnd_texture_map — per-pixel texture masking for entire frame.
 *
 * AVX2 opportunity: process 4 blocks (32 pixels) per pass by interleaving
 * the DCT row transforms across blocks using 256-bit registers. Each
 * _mm256_madd_epi16 handles 16 multiplications.
 */
static inline void jnd_texture_map(const uint8_t* gray, int w, int h,
                                   float* out) {
    int bw = (w + 7) / 8;
    int bh = (h + 7) / 8;

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            int16_t block[8][8];
            for (int r = 0; r < 8; r++) {
                int y = by * 8 + r; if (y >= h) y = h - 1;
                for (int c = 0; c < 8; c++) {
                    int x = bx * 8 + c; if (x >= w) x = w - 1;
                    block[r][c] = (int16_t)gray[y * w + x];
                }
            }

            float jnd = jnd_texture_block(block);

            for (int r = 0; r < 8; r++) {
                int y = by * 8 + r; if (y >= h) break;
                for (int c = 0; c < 8; c++) {
                    int x = bx * 8 + c; if (x >= w) break;
                    out[y * w + x] = jnd;
                }
            }
        }
    }
}


/* ========================================================================
 * Temporal masking (motion magnitude between frames)
 * ======================================================================== */

static inline float jnd_temporal(float motion_mag) {
    if (motion_mag < 0.0f) motion_mag = 0.0f;
    const float gamma = 1.4f;
    const float tau = 20.0f;
    return 1.0f + gamma * (1.0f - expf(-motion_mag / tau));
}

/*
 * jnd_temporal_map — per-pixel temporal masking from frame difference.
 *
 * SSE2: compute |cur-prev| using _mm_sad_epu8 (sum of absolute differences
 * across 16 bytes at once), then accumulate per 4x4 block.
 *
 * AVX2: _mm256_sad_epu8 processes 32 bytes, covering two 4x4 block rows.
 */
static inline void jnd_temporal_map(const uint8_t* cur, const uint8_t* prev,
                                    int w, int h, float* out) {
    int bw = (w + 3) / 4;
    int bh = (h + 3) / 4;

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            int sum = 0, count = 0;

#ifdef __SSE2__
            /*
             * SSE2 opportunity: if the 4x4 block is contiguous in memory
             * (stride == w), load 16 bytes from cur and prev, compute
             * _mm_sad_epu8, extract the two 64-bit sums. For non-contiguous
             * blocks, fall back to scalar.
             */
#endif
            for (int r = 0; r < 4; r++) {
                int y = by * 4 + r; if (y >= h) break;
                for (int c = 0; c < 4; c++) {
                    int x = bx * 4 + c; if (x >= w) break;
                    int idx = y * w + x;
                    int d = (int)cur[idx] - (int)prev[idx];
                    if (d < 0) d = -d;
                    sum += d;
                    count++;
                }
            }

            float mag = (float)sum / (float)count;
            float jnd = jnd_temporal(mag);

            for (int r = 0; r < 4; r++) {
                int y = by * 4 + r; if (y >= h) break;
                for (int c = 0; c < 4; c++) {
                    int x = bx * 4 + c; if (x >= w) break;
                    out[y * w + x] = jnd;
                }
            }
        }
    }
}


/* ========================================================================
 * Combined JND model
 * ======================================================================== */

/*
 * jnd_combined — compute combined JND map.
 *   JND(x,y) = max(luminance_jnd, texture_jnd) * temporal_boost
 *
 * prev_gray may be NULL for static (single-frame) analysis.
 * out must be preallocated: w*h floats.
 *
 * AVX2 opportunity: the max + multiply can be done with
 * _mm256_max_ps + _mm256_mul_ps, processing 8 pixels per cycle.
 */
static inline void jnd_combined(const uint8_t* gray, const uint8_t* prev_gray,
                                int w, int h, float* out) {
    size_t n = (size_t)w * h;
    float* lum = (float*)malloc(n * sizeof(float));
    float* tex = (float*)malloc(n * sizeof(float));
    if (!lum || !tex) { free(lum); free(tex); return; }

    jnd_luminance_map(gray, w, h, lum);
    jnd_texture_map(gray, w, h, tex);

    /* Spatial JND = max(luminance, texture) */
#ifdef __AVX2__
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vl = _mm256_loadu_ps(&lum[i]);
        __m256 vt = _mm256_loadu_ps(&tex[i]);
        __m256 vm = _mm256_max_ps(vl, vt);
        _mm256_storeu_ps(&out[i], vm);
    }
    for (; i < n; i++) {
        out[i] = lum[i] > tex[i] ? lum[i] : tex[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = lum[i] > tex[i] ? lum[i] : tex[i];
    }
#endif

    /* Temporal boost (multiplicative) */
    if (prev_gray) {
        float* temp = (float*)malloc(n * sizeof(float));
        if (temp) {
            jnd_temporal_map(gray, prev_gray, w, h, temp);
#ifdef __AVX2__
            size_t j = 0;
            for (; j + 7 < n; j += 8) {
                __m256 vo = _mm256_loadu_ps(&out[j]);
                __m256 vt = _mm256_loadu_ps(&temp[j]);
                _mm256_storeu_ps(&out[j], _mm256_mul_ps(vo, vt));
            }
            for (; j < n; j++) {
                out[j] *= temp[j];
            }
#else
            for (size_t j = 0; j < n; j++) {
                out[j] *= temp[j];
            }
#endif
            free(temp);
        }
    }

    free(lum);
    free(tex);
}

/*
 * jnd_prunable_ratio — fraction of errors below JND threshold.
 *
 * SSE2/AVX2: compare 4/8 floats per cycle with _mm_cmplt_ps / _mm256_cmp_ps,
 * accumulate popcount of the resulting mask.
 */
static inline float jnd_prunable_ratio(const float* jnd_map,
                                       const float* errors, size_t n) {
    size_t prunable = 0;

#ifdef __AVX2__
    size_t i = 0;
    __m256i count = _mm256_setzero_si256();
    for (; i + 7 < n; i += 8) {
        __m256 ve = _mm256_loadu_ps(&errors[i]);
        __m256 vj = _mm256_loadu_ps(&jnd_map[i]);
        __m256 cmp = _mm256_cmp_ps(ve, vj, _CMP_LT_OQ);
        int mask = _mm256_movemask_ps(cmp);
        prunable += __builtin_popcount(mask);
    }
    for (; i < n; i++) {
        if (errors[i] < jnd_map[i]) prunable++;
    }
#else
    for (size_t i = 0; i < n; i++) {
        if (errors[i] < jnd_map[i]) prunable++;
    }
#endif

    return n > 0 ? (float)prunable / (float)n : 0.0f;
}

#ifdef __cplusplus
}
#endif

#endif /* FOVEA_JND_H */
