// Package texture implements JND texture masking based on DCT coefficient
// variance in 8×8 blocks. Textured/busy areas mask errors more effectively.
package texture

import "math"

// dctBasis[u][x] = C(u) * cos((2x+1)*u*pi/16), precomputed for 8-point DCT.
var dctBasis [8][8]float64

func init() {
	for u := 0; u < 8; u++ {
		cu := 1.0
		if u == 0 {
			cu = 1.0 / math.Sqrt2
		}
		cu *= 0.5 // normalization factor sqrt(2/N) with N=8
		for x := 0; x < 8; x++ {
			dctBasis[u][x] = cu * math.Cos(float64(2*x+1)*float64(u)*math.Pi/16.0)
		}
	}
}

// dct8x8 computes an 8×8 type-II DCT of block, storing result in out.
func dct8x8(block *[8][8]int16, out *[8][8]float64) {
	// Row transform.
	var tmp [8][8]float64
	for y := 0; y < 8; y++ {
		for u := 0; u < 8; u++ {
			s := 0.0
			for x := 0; x < 8; x++ {
				s += float64(block[y][x]) * dctBasis[u][x]
			}
			tmp[y][u] = s
		}
	}
	// Column transform.
	for u := 0; u < 8; u++ {
		for v := 0; v < 8; v++ {
			s := 0.0
			for y := 0; y < 8; y++ {
				s += tmp[y][v] * dctBasis[u][y]
			}
			out[u][v] = s
		}
	}
}

// TextureJND returns the texture masking JND for an 8×8 block of pixel values
// (int16 to allow signed residuals). The masking strength is derived from the
// variance of AC DCT coefficients: high variance = busy texture = more masking.
//
// The formula:  T_tex = alpha * (variance ^ beta)
// with alpha=0.25, beta=0.36 — empirical fit from Yang/Wu 2005.
func TextureJND(block *[8][8]int16) float64 {
	var dct [8][8]float64
	dct8x8(block, &dct)

	// Compute variance of AC coefficients (skip DC at [0][0]).
	n := 63 // 8*8 - 1
	sum := 0.0
	sum2 := 0.0
	for u := 0; u < 8; u++ {
		for v := 0; v < 8; v++ {
			if u == 0 && v == 0 {
				continue
			}
			c := dct[u][v]
			sum += c
			sum2 += c * c
		}
	}
	mean := sum / float64(n)
	variance := sum2/float64(n) - mean*mean
	if variance < 0 {
		variance = 0
	}

	const alpha = 0.25
	const beta = 0.36
	return alpha * math.Pow(variance, beta)
}

// TextureMap computes per-pixel texture masking JND from a grayscale image.
// Each pixel gets the masking value of its enclosing 8×8 block.
// Pixels in partial edge blocks (if w or h is not divisible by 8) use a
// block padded with the edge value.
func TextureMap(gray []uint8, w, h int) []float64 {
	if w <= 0 || h <= 0 || len(gray) != w*h {
		return nil
	}

	out := make([]float64, w*h)

	bw := (w + 7) / 8
	bh := (h + 7) / 8

	var block [8][8]int16

	for by := 0; by < bh; by++ {
		for bx := 0; bx < bw; bx++ {
			// Extract block, replicating edge pixels for partial blocks.
			for r := 0; r < 8; r++ {
				y := by*8 + r
				if y >= h {
					y = h - 1
				}
				for c := 0; c < 8; c++ {
					x := bx*8 + c
					if x >= w {
						x = w - 1
					}
					block[r][c] = int16(gray[y*w+x])
				}
			}

			jnd := TextureJND(&block)

			// Write JND to all pixels in this block.
			for r := 0; r < 8; r++ {
				y := by*8 + r
				if y >= h {
					break
				}
				for c := 0; c < 8; c++ {
					x := bx*8 + c
					if x >= w {
						break
					}
					out[y*w+x] = jnd
				}
			}
		}
	}

	return out
}
