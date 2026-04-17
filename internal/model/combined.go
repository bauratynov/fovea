// Package model combines luminance, texture, and temporal JND models into
// a single perceptual visibility threshold map.
package model

import (
	"github.com/bauratynov/fovea/internal/luminance"
	"github.com/bauratynov/fovea/internal/temporal"
	"github.com/bauratynov/fovea/internal/texture"
)

// JNDMap holds per-pixel just-noticeable-difference thresholds.
type JNDMap struct {
	W, H      int
	Threshold []float64 // length W*H
}

// ThresholdAt returns the JND at pixel (x, y).
func (m *JNDMap) ThresholdAt(x, y int) float64 {
	if x < 0 || x >= m.W || y < 0 || y >= m.H {
		return 0
	}
	return m.Threshold[y*m.W+x]
}

// CanPrune returns true if the error at (x, y) is below the JND threshold
// (i.e., imperceptible to the human eye).
func (m *JNDMap) CanPrune(x, y int, err float64) bool {
	return err < m.ThresholdAt(x, y)
}

// PrunableRatio returns the fraction of pixel errors that are below their
// respective JND thresholds. errors must have length W*H.
func (m *JNDMap) PrunableRatio(errors []float64) float64 {
	if len(errors) != len(m.Threshold) {
		return 0
	}
	prunable := 0
	for i, e := range errors {
		if e < m.Threshold[i] {
			prunable++
		}
	}
	return float64(prunable) / float64(len(errors))
}

// BitrateReduction estimates the percentage of bitrate savings achievable
// by pruning sub-JND coefficients. The model assumes that each prunable
// pixel's contribution to bitrate is proportional to its error magnitude,
// so savings ≈ sum(error_i for prunable_i) / sum(error_i for all_i).
// Returns a value in [0, 1].
func (m *JNDMap) BitrateReduction(errors []float64) float64 {
	if len(errors) != len(m.Threshold) {
		return 0
	}
	totalBits := 0.0
	savedBits := 0.0
	for i, e := range errors {
		if e < 0 {
			e = -e
		}
		totalBits += e
		if e < m.Threshold[i] {
			savedBits += e
		}
	}
	if totalBits == 0 {
		return 0
	}
	return savedBits / totalBits
}

// CombinedJND computes the combined JND map from all three models:
//   JND(x,y) = max(luminance_jnd, texture_jnd) × temporal_boost
//
// gray: current frame grayscale (w×h).
// prevGray: previous frame grayscale (w×h), or nil for static analysis.
func CombinedJND(gray, prevGray []uint8, w, h int) *JNDMap {
	if w <= 0 || h <= 0 || len(gray) != w*h {
		return nil
	}

	lumMap := luminance.LuminanceMap(gray, w, h)
	texMap := texture.TextureMap(gray, w, h)

	n := w * h
	thresh := make([]float64, n)

	// Spatial JND = max(luminance, texture).
	for i := 0; i < n; i++ {
		lum := lumMap[i]
		tex := texMap[i]
		if tex > lum {
			thresh[i] = tex
		} else {
			thresh[i] = lum
		}
	}

	// Temporal boost (multiplicative).
	if prevGray != nil && len(prevGray) == n {
		tempMap := temporal.TemporalMap(gray, prevGray, w, h)
		for i := 0; i < n; i++ {
			thresh[i] *= tempMap[i]
		}
	}

	return &JNDMap{W: w, H: h, Threshold: thresh}
}

// StaticJND computes the combined JND without temporal masking (single image).
func StaticJND(gray []uint8, w, h int) *JNDMap {
	return CombinedJND(gray, nil, w, h)
}
