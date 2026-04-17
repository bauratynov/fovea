// Package temporal implements JND temporal masking based on inter-frame
// motion magnitude. Fast motion masks perception — errors are less visible
// in moving areas.
package temporal

import "math"

// TemporalJND returns the temporal masking JND boost for a given motion
// magnitude (mean absolute difference of pixel values in a local block).
//
// Model: T_temp = 1 + gamma * (1 - exp(-motionMag / tau))
// At zero motion: T_temp = 1 (no boost).
// At high motion: T_temp → 1 + gamma ≈ 2.4 (strong masking).
//
// gamma = 1.4, tau = 20.0 — empirical fit.
func TemporalJND(motionMag float64) float64 {
	if motionMag < 0 {
		motionMag = 0
	}
	const gamma = 1.4
	const tau = 20.0
	return 1.0 + gamma*(1.0-math.Exp(-motionMag/tau))
}

// TemporalMap computes per-pixel temporal masking boost from current and
// previous grayscale frames. Motion magnitude is the mean |cur-prev| in
// each 4×4 block. Each pixel gets the boost of its enclosing block.
//
// Both curGray and prevGray must have length w*h.
// Returns nil if inputs are invalid.
func TemporalMap(curGray, prevGray []uint8, w, h int) []float64 {
	if w <= 0 || h <= 0 {
		return nil
	}
	n := w * h
	if len(curGray) != n || len(prevGray) != n {
		return nil
	}

	out := make([]float64, n)
	bw := (w + 3) / 4
	bh := (h + 3) / 4

	for by := 0; by < bh; by++ {
		for bx := 0; bx < bw; bx++ {
			// Compute mean absolute difference in this 4×4 block.
			sum := 0
			count := 0
			for r := 0; r < 4; r++ {
				y := by*4 + r
				if y >= h {
					break
				}
				for c := 0; c < 4; c++ {
					x := bx*4 + c
					if x >= w {
						break
					}
					idx := y*w + x
					d := int(curGray[idx]) - int(prevGray[idx])
					if d < 0 {
						d = -d
					}
					sum += d
					count++
				}
			}

			mag := float64(sum) / float64(count)
			jnd := TemporalJND(mag)

			// Write to all pixels in block.
			for r := 0; r < 4; r++ {
				y := by*4 + r
				if y >= h {
					break
				}
				for c := 0; c < 4; c++ {
					x := bx*4 + c
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

// MotionHistory maintains an exponentially-decayed motion history for
// temporal masking with memory across multiple frames.
type MotionHistory struct {
	w, h   int
	decay  float64   // exponential decay factor (e.g., 0.8)
	history []float64 // accumulated motion per 4×4 block
}

// NewMotionHistory creates a motion history tracker.
// decay should be in (0, 1); typical value 0.7–0.9.
func NewMotionHistory(w, h int, decay float64) *MotionHistory {
	if decay <= 0 || decay >= 1 {
		decay = 0.8
	}
	bw := (w + 3) / 4
	bh := (h + 3) / 4
	return &MotionHistory{
		w: w, h: h,
		decay:   decay,
		history: make([]float64, bw*bh),
	}
}

// Update feeds a new frame pair and returns the temporal masking map.
// The history is updated with exponential decay.
func (mh *MotionHistory) Update(curGray, prevGray []uint8) []float64 {
	w, h := mh.w, mh.h
	if len(curGray) != w*h || len(prevGray) != w*h {
		return nil
	}

	bw := (w + 3) / 4
	out := make([]float64, w*h)

	bi := 0
	bh := (h + 3) / 4
	for by := 0; by < bh; by++ {
		for bx := 0; bx < bw; bx++ {
			sum := 0
			count := 0
			for r := 0; r < 4; r++ {
				y := by*4 + r
				if y >= h {
					break
				}
				for c := 0; c < 4; c++ {
					x := bx*4 + c
					if x >= w {
						break
					}
					idx := y*w + x
					d := int(curGray[idx]) - int(prevGray[idx])
					if d < 0 {
						d = -d
					}
					sum += d
					count++
				}
			}

			instantMag := float64(sum) / float64(count)
			// Exponential moving average.
			mh.history[bi] = mh.decay*mh.history[bi] + (1-mh.decay)*instantMag
			jnd := TemporalJND(mh.history[bi])

			for r := 0; r < 4; r++ {
				y := by*4 + r
				if y >= h {
					break
				}
				for c := 0; c < 4; c++ {
					x := bx*4 + c
					if x >= w {
						break
					}
					out[y*w+x] = jnd
				}
			}
			bi++
		}
	}

	return out
}
