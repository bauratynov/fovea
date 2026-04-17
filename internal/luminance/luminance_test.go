package luminance

import (
	"math"
	"testing"
)

func TestLuminanceJND_DarkHighThreshold(t *testing.T) {
	// Near-black: eye is noisy, threshold should be high.
	jnd0 := LuminanceJND(0)
	if jnd0 < 10 {
		t.Fatalf("LuminanceJND(0) = %f, want >= 10", jnd0)
	}
	jnd10 := LuminanceJND(10)
	if jnd10 >= jnd0 {
		t.Fatalf("threshold should decrease as luma rises from 0: JND(0)=%f JND(10)=%f", jnd0, jnd10)
	}
}

func TestLuminanceJND_MidGrayLow(t *testing.T) {
	// Mid-gray (128): threshold should be moderate (lowest region).
	jnd := LuminanceJND(128)
	if jnd < 2 || jnd > 10 {
		t.Fatalf("LuminanceJND(128) = %f, want 2..10", jnd)
	}
}

func TestLuminanceJND_BrightHighThreshold(t *testing.T) {
	// Near-white: saturation, threshold rises.
	jnd255 := LuminanceJND(255)
	jnd128 := LuminanceJND(128)
	if jnd255 <= jnd128 {
		t.Fatalf("bright should have higher threshold: JND(255)=%f JND(128)=%f", jnd255, jnd128)
	}
}

func TestLuminanceJND_WeberFechnerRange(t *testing.T) {
	// In Weber-Fechner range (32..200) threshold should be roughly proportional
	// to background luminance — monotonically increasing.
	prev := LuminanceJND(32)
	for bg := 33.0; bg <= 200; bg++ {
		cur := LuminanceJND(bg)
		if cur < prev-0.01 {
			t.Fatalf("non-monotonic at bg=%f: prev=%f cur=%f", bg, prev, cur)
		}
		prev = cur
	}
}

func TestLuminanceJND_Clamp(t *testing.T) {
	// Out-of-range inputs should be clamped.
	if v := LuminanceJND(-10); v != LuminanceJND(0) {
		t.Fatalf("negative not clamped: %f", v)
	}
	if v := LuminanceJND(300); v != LuminanceJND(255) {
		t.Fatalf("over-255 not clamped: %f", v)
	}
}

func TestLuminanceMap_Uniform(t *testing.T) {
	// Uniform gray image → all pixels get the same JND.
	w, h := 16, 16
	gray := make([]uint8, w*h)
	for i := range gray {
		gray[i] = 100
	}
	m := LuminanceMap(gray, w, h)
	if len(m) != w*h {
		t.Fatalf("len = %d, want %d", len(m), w*h)
	}
	expected := LuminanceJND(100)
	for i, v := range m {
		if math.Abs(v-expected) > 1.0 {
			t.Fatalf("pixel %d: got %f, want ~%f", i, v, expected)
		}
	}
}

func TestLuminanceMap_EdgeCases(t *testing.T) {
	// 1×1
	m := LuminanceMap([]uint8{128}, 1, 1)
	if len(m) != 1 {
		t.Fatalf("1x1: len = %d", len(m))
	}

	// 2×2
	m = LuminanceMap([]uint8{0, 255, 128, 64}, 2, 2)
	if len(m) != 4 {
		t.Fatalf("2x2: len = %d", len(m))
	}

	// Empty
	m = LuminanceMap(nil, 0, 0)
	if m != nil {
		t.Fatalf("empty should return nil")
	}
}

func TestLuminanceMap_DimMatch(t *testing.T) {
	w, h := 64, 48
	gray := make([]uint8, w*h)
	m := LuminanceMap(gray, w, h)
	if len(m) != w*h {
		t.Fatalf("dimension mismatch: %d vs %d", len(m), w*h)
	}
}

func BenchmarkLuminanceJND(b *testing.B) {
	for i := 0; i < b.N; i++ {
		LuminanceJND(float64(i & 255))
	}
}

func BenchmarkLuminanceMap_1080p(b *testing.B) {
	w, h := 1920, 1080
	gray := make([]uint8, w*h)
	for i := range gray {
		gray[i] = uint8(i % 256)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LuminanceMap(gray, w, h)
	}
}
