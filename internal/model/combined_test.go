package model

import (
	"math"
	"testing"
)

func makeFlat(w, h int, val uint8) []uint8 {
	buf := make([]uint8, w*h)
	for i := range buf {
		buf[i] = val
	}
	return buf
}

func makeNoise(w, h int) []uint8 {
	buf := make([]uint8, w*h)
	for i := range buf {
		buf[i] = uint8((i*137 + 59) % 256)
	}
	return buf
}

func TestCombinedJND_StaticBright(t *testing.T) {
	// Static bright scene → mostly luminance-driven.
	w, h := 64, 64
	gray := makeFlat(w, h, 200)
	m := CombinedJND(gray, nil, w, h)
	if m == nil {
		t.Fatal("nil map")
	}
	if m.W != w || m.H != h {
		t.Fatalf("dims: %dx%d", m.W, m.H)
	}
	// All thresholds should be > 0.
	for i, v := range m.Threshold {
		if v <= 0 {
			t.Fatalf("pixel %d: threshold = %f", i, v)
		}
	}
}

func TestCombinedJND_MovingTextured(t *testing.T) {
	// Moving textured scene → temporal + texture dominate.
	w, h := 64, 64
	cur := makeNoise(w, h)
	prev := makeFlat(w, h, 128)

	withMotion := CombinedJND(cur, prev, w, h)
	withoutMotion := CombinedJND(cur, nil, w, h)

	// With motion should have higher thresholds (temporal boost).
	sumWith := 0.0
	sumWithout := 0.0
	for i := range withMotion.Threshold {
		sumWith += withMotion.Threshold[i]
		sumWithout += withoutMotion.Threshold[i]
	}
	if sumWith <= sumWithout {
		t.Fatalf("motion should boost thresholds: with=%f without=%f", sumWith, sumWithout)
	}
}

func TestPrunableRatio_AllBelow(t *testing.T) {
	w, h := 16, 16
	gray := makeFlat(w, h, 128)
	m := StaticJND(gray, w, h)

	// Errors all zero → all below JND → 100%.
	errors := make([]float64, w*h)
	ratio := m.PrunableRatio(errors)
	if math.Abs(ratio-1.0) > 0.001 {
		t.Fatalf("all-zero errors: ratio = %f, want 1.0", ratio)
	}
}

func TestPrunableRatio_AllAbove(t *testing.T) {
	w, h := 16, 16
	gray := makeFlat(w, h, 128)
	m := StaticJND(gray, w, h)

	// Errors all 255 → all above JND → 0%.
	errors := make([]float64, w*h)
	for i := range errors {
		errors[i] = 255
	}
	ratio := m.PrunableRatio(errors)
	if ratio != 0 {
		t.Fatalf("all-255 errors: ratio = %f, want 0.0", ratio)
	}
}

func TestBitrateReduction_Typical(t *testing.T) {
	// Simulate typical surveillance: noisy frame with small errors.
	w, h := 64, 64
	gray := makeNoise(w, h)
	m := StaticJND(gray, w, h)

	// Generate errors centered around half the typical JND (~2-3).
	errors := make([]float64, w*h)
	for i := range errors {
		errors[i] = float64(i%5) + 0.5 // 0.5, 1.5, 2.5, 3.5, 4.5
	}
	reduction := m.BitrateReduction(errors)
	// Should be between 0 and 1.
	if reduction < 0 || reduction > 1 {
		t.Fatalf("reduction out of range: %f", reduction)
	}
	t.Logf("BitrateReduction on synthetic: %.1f%%", reduction*100)
}

func TestCanPrune(t *testing.T) {
	w, h := 8, 8
	gray := makeFlat(w, h, 128)
	m := StaticJND(gray, w, h)

	thresh := m.ThresholdAt(4, 4)
	if !m.CanPrune(4, 4, thresh*0.5) {
		t.Fatal("error below JND should be prunable")
	}
	if m.CanPrune(4, 4, thresh*2.0) {
		t.Fatal("error above JND should NOT be prunable")
	}
}

func TestCombinedJND_EdgeCases(t *testing.T) {
	// 1×1
	m := CombinedJND([]uint8{128}, nil, 1, 1)
	if m == nil || len(m.Threshold) != 1 {
		t.Fatal("1x1 failed")
	}

	// 2×2
	m = CombinedJND([]uint8{0, 255, 128, 64}, nil, 2, 2)
	if m == nil || len(m.Threshold) != 4 {
		t.Fatal("2x2 failed")
	}

	// nil
	m = CombinedJND(nil, nil, 0, 0)
	if m != nil {
		t.Fatal("empty should return nil")
	}
}

func TestThresholdAt_OutOfBounds(t *testing.T) {
	m := &JNDMap{W: 2, H: 2, Threshold: []float64{1, 2, 3, 4}}
	if m.ThresholdAt(-1, 0) != 0 {
		t.Fatal("OOB should return 0")
	}
	if m.ThresholdAt(2, 0) != 0 {
		t.Fatal("OOB should return 0")
	}
}

func TestBitrateReduction_SurveillanceEstimate(t *testing.T) {
	// Realistic test: surveillance-like frame with quantization-like errors.
	// We expect 20-30% savings on typical content.
	w, h := 128, 128
	gray := make([]uint8, w*h)
	for i := range gray {
		// Simulate outdoor scene: mix of textures and flat areas.
		x := i % w
		y := i / w
		base := uint8(80 + (x*y)%60)
		noise := uint8((i * 31) % 20)
		gray[i] = base + noise
	}
	m := StaticJND(gray, w, h)

	// Quantization-like errors: mostly small, occasionally large.
	errors := make([]float64, w*h)
	for i := range errors {
		// QP-like error distribution: most values 1-4, some 5-10.
		errors[i] = float64(1 + (i*7)%8)
	}
	reduction := m.BitrateReduction(errors)
	t.Logf("Surveillance estimate: %.1f%% bitrate reduction", reduction*100)
	// We check it's in a reasonable range — not 0, not 100.
	if reduction < 0.05 || reduction > 0.80 {
		t.Fatalf("unexpected reduction: %.1f%%", reduction*100)
	}
}

func BenchmarkCombinedJND_1080p(b *testing.B) {
	w, h := 1920, 1080
	gray := makeNoise(w, h)
	prev := makeFlat(w, h, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CombinedJND(gray, prev, w, h)
	}
}

func BenchmarkStaticJND_1080p(b *testing.B) {
	w, h := 1920, 1080
	gray := makeNoise(w, h)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		StaticJND(gray, w, h)
	}
}

func BenchmarkPrunableRatio(b *testing.B) {
	w, h := 1920, 1080
	gray := makeNoise(w, h)
	m := StaticJND(gray, w, h)
	errors := make([]float64, w*h)
	for i := range errors {
		errors[i] = float64(i%10) + 0.5
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.PrunableRatio(errors)
	}
}
