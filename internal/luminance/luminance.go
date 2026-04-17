// Package luminance implements a JND luminance adaptation model based on
// the Weber-Fechner law with non-linear corrections at low and high extremes.
package luminance

import "math"

// LuminanceJND returns the just-noticeable-difference threshold for a given
// background luminance value (0–255). The model uses Weber-Fechner in the
// mid-range and adds higher thresholds at extremes where the eye is less
// sensitive (dark adaptation noise, bright saturation).
//
// The piecewise model:
//   bg < 32:  T = 17 - 3*sqrt(bg)             (dark: high threshold, drops as bg rises)
//   32 ≤ bg ≤ 127: T = 3*(bg-32)/95 + 3       (Weber mid-low: linear ramp ~3→6)
//   127 < bg ≤ 200: T = 0.04*bg + 1            (Weber mid-high: ~6→9)
//   bg > 200: T = 9/255*(bg-200)/55*4 + 9      (bright saturation: rises to ~13)
func LuminanceJND(bgLuma float64) float64 {
	if bgLuma < 0 {
		bgLuma = 0
	}
	if bgLuma > 255 {
		bgLuma = 255
	}

	switch {
	case bgLuma < 32:
		// Dark adaptation: noise masks errors, threshold is high and
		// decreases as luminance rises out of near-black.
		return 17.0 - 3.0*math.Sqrt(bgLuma)
	case bgLuma <= 127:
		// Weber-Fechner mid-low range: threshold proportional to bg.
		return 3.0*(bgLuma-32.0)/95.0 + 3.0
	case bgLuma <= 200:
		// Weber-Fechner mid-high range.
		return 0.04*bgLuma + 1.0
	default:
		// Bright saturation: sensitivity drops again.
		return (bgLuma-200.0)/55.0*4.0 + 9.0
	}
}

// LuminanceMap computes per-pixel JND thresholds from a grayscale image
// using a 5×5 local mean as the background luminance estimate.
// gray must have length w*h. The returned slice has the same length.
func LuminanceMap(gray []uint8, w, h int) []float64 {
	if w <= 0 || h <= 0 || len(gray) != w*h {
		return nil
	}

	out := make([]float64, w*h)

	// Compute integral image for fast box-mean (using int32 to avoid overflow).
	// integ[y*stride+x] = sum of gray[0..y-1][0..x-1].
	stride := w + 1
	integ := make([]int32, stride*(h+1))

	for y := 0; y < h; y++ {
		rowSum := int32(0)
		for x := 0; x < w; x++ {
			rowSum += int32(gray[y*w+x])
			integ[(y+1)*stride+(x+1)] = rowSum + integ[y*stride+(x+1)]
		}
	}

	// boxMean returns the mean pixel value in [y0,y1) × [x0,x1).
	boxMean := func(x0, y0, x1, y1 int) float64 {
		s := integ[y1*stride+x1] - integ[y0*stride+x1] -
			integ[y1*stride+x0] + integ[y0*stride+x0]
		n := (x1 - x0) * (y1 - y0)
		return float64(s) / float64(n)
	}

	const radius = 2 // 5×5 window

	for y := 0; y < h; y++ {
		y0 := y - radius
		y1 := y + radius + 1
		if y0 < 0 {
			y0 = 0
		}
		if y1 > h {
			y1 = h
		}
		for x := 0; x < w; x++ {
			x0 := x - radius
			x1 := x + radius + 1
			if x0 < 0 {
				x0 = 0
			}
			if x1 > w {
				x1 = w
			}
			bgLuma := boxMean(x0, y0, x1, y1)
			out[y*w+x] = LuminanceJND(bgLuma)
		}
	}
	return out
}
