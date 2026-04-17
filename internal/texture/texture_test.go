package texture

import (
	"math"
	"testing"
)

func TestTextureJND_FlatBlock(t *testing.T) {
	// All same value → zero AC variance → zero texture masking.
	var block [8][8]int16
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			block[r][c] = 128
		}
	}
	jnd := TextureJND(&block)
	if jnd > 0.01 {
		t.Fatalf("flat block: JND = %f, want ~0", jnd)
	}
}

func TestTextureJND_BusyBlock(t *testing.T) {
	// Checkerboard pattern → high AC variance → high masking.
	var block [8][8]int16
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			if (r+c)%2 == 0 {
				block[r][c] = 200
			} else {
				block[r][c] = 50
			}
		}
	}
	jnd := TextureJND(&block)
	if jnd < 1.0 {
		t.Fatalf("busy block: JND = %f, want >= 1.0", jnd)
	}
}

func TestTextureJND_GradientBlock(t *testing.T) {
	// Smooth gradient → moderate variance.
	var block [8][8]int16
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			block[r][c] = int16(r*32 + c*4)
		}
	}
	flat := 0.0
	{
		var b [8][8]int16
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				b[r][c] = 128
			}
		}
		flat = TextureJND(&b)
	}
	grad := TextureJND(&block)
	if grad <= flat {
		t.Fatalf("gradient should have more masking than flat: grad=%f flat=%f", grad, flat)
	}
}

func TestTextureMap_Uniform(t *testing.T) {
	w, h := 16, 16
	gray := make([]uint8, w*h)
	for i := range gray {
		gray[i] = 100
	}
	m := TextureMap(gray, w, h)
	if len(m) != w*h {
		t.Fatalf("len = %d, want %d", len(m), w*h)
	}
	for i, v := range m {
		if v > 0.01 {
			t.Fatalf("uniform image pixel %d: JND = %f, want ~0", i, v)
		}
	}
}

func TestTextureMap_EdgeCases(t *testing.T) {
	// 1×1
	m := TextureMap([]uint8{128}, 1, 1)
	if len(m) != 1 {
		t.Fatalf("1x1: len = %d", len(m))
	}

	// 2×2
	m = TextureMap([]uint8{0, 255, 128, 64}, 2, 2)
	if len(m) != 4 {
		t.Fatalf("2x2: len = %d", len(m))
	}

	// Empty
	m = TextureMap(nil, 0, 0)
	if m != nil {
		t.Fatal("empty should return nil")
	}
}

func TestTextureMap_NonMultipleOf8(t *testing.T) {
	// 10×10 image — partial blocks at edges.
	w, h := 10, 10
	gray := make([]uint8, w*h)
	for i := range gray {
		gray[i] = uint8(i % 256)
	}
	m := TextureMap(gray, w, h)
	if len(m) != w*h {
		t.Fatalf("len = %d, want %d", len(m), w*h)
	}
	// No NaN or Inf.
	for i, v := range m {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("pixel %d: bad value %f", i, v)
		}
	}
}

func BenchmarkTextureJND(b *testing.B) {
	var block [8][8]int16
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			block[r][c] = int16((r*8 + c) * 3)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TextureJND(&block)
	}
}

func BenchmarkTextureMap_1080p(b *testing.B) {
	w, h := 1920, 1080
	gray := make([]uint8, w*h)
	for i := range gray {
		gray[i] = uint8(i % 256)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TextureMap(gray, w, h)
	}
}
