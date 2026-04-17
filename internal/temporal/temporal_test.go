package temporal

import (
	"math"
	"testing"
)

func TestTemporalJND_NoMotion(t *testing.T) {
	jnd := TemporalJND(0)
	if math.Abs(jnd-1.0) > 0.001 {
		t.Fatalf("no motion: JND = %f, want 1.0", jnd)
	}
}

func TestTemporalJND_HighMotion(t *testing.T) {
	jnd := TemporalJND(100)
	if jnd < 2.0 {
		t.Fatalf("high motion: JND = %f, want >= 2.0", jnd)
	}
}

func TestTemporalJND_Monotonic(t *testing.T) {
	prev := TemporalJND(0)
	for m := 1.0; m <= 200; m++ {
		cur := TemporalJND(m)
		if cur < prev-0.001 {
			t.Fatalf("non-monotonic at mag=%f: prev=%f cur=%f", m, prev, cur)
		}
		prev = cur
	}
}

func TestTemporalJND_NegativeClamped(t *testing.T) {
	if v := TemporalJND(-5); v != TemporalJND(0) {
		t.Fatalf("negative not clamped: %f vs %f", v, TemporalJND(0))
	}
}

func TestTemporalMap_IdenticalFrames(t *testing.T) {
	w, h := 16, 16
	gray := make([]uint8, w*h)
	for i := range gray {
		gray[i] = 100
	}
	m := TemporalMap(gray, gray, w, h)
	if len(m) != w*h {
		t.Fatalf("len = %d, want %d", len(m), w*h)
	}
	for i, v := range m {
		if math.Abs(v-1.0) > 0.001 {
			t.Fatalf("identical frames pixel %d: boost = %f, want 1.0", i, v)
		}
	}
}

func TestTemporalMap_HighMotion(t *testing.T) {
	w, h := 16, 16
	cur := make([]uint8, w*h)
	prev := make([]uint8, w*h)
	for i := range cur {
		cur[i] = 200
		prev[i] = 50
	}
	m := TemporalMap(cur, prev, w, h)
	for i, v := range m {
		if v < 2.0 {
			t.Fatalf("high motion pixel %d: boost = %f, want >= 2.0", i, v)
		}
	}
}

func TestTemporalMap_EdgeCases(t *testing.T) {
	// 1×1
	m := TemporalMap([]uint8{100}, []uint8{100}, 1, 1)
	if len(m) != 1 {
		t.Fatalf("1x1: len = %d", len(m))
	}

	// 2×2
	m = TemporalMap([]uint8{0, 255, 128, 64}, []uint8{128, 128, 128, 128}, 2, 2)
	if len(m) != 4 {
		t.Fatalf("2x2: len = %d", len(m))
	}

	// Empty
	m = TemporalMap(nil, nil, 0, 0)
	if m != nil {
		t.Fatal("empty should return nil")
	}
}

func TestMotionHistory_Decay(t *testing.T) {
	w, h := 8, 8
	mh := NewMotionHistory(w, h, 0.8)

	static := make([]uint8, w*h)
	for i := range static {
		static[i] = 100
	}

	moving := make([]uint8, w*h)
	for i := range moving {
		moving[i] = 200
	}

	// First update: big motion.
	m1 := mh.Update(moving, static)
	if m1 == nil {
		t.Fatal("Update returned nil")
	}
	boost1 := m1[0]

	// Second update: no motion — history should decay.
	m2 := mh.Update(moving, moving)
	boost2 := m2[0]

	// Third update: still no motion.
	m3 := mh.Update(moving, moving)
	boost3 := m3[0]

	if boost2 >= boost1 {
		t.Fatalf("history should decay: boost1=%f boost2=%f", boost1, boost2)
	}
	if boost3 >= boost2 {
		t.Fatalf("history should keep decaying: boost2=%f boost3=%f", boost2, boost3)
	}
}

func BenchmarkTemporalJND(b *testing.B) {
	for i := 0; i < b.N; i++ {
		TemporalJND(float64(i % 200))
	}
}

func BenchmarkTemporalMap_1080p(b *testing.B) {
	w, h := 1920, 1080
	cur := make([]uint8, w*h)
	prev := make([]uint8, w*h)
	for i := range cur {
		cur[i] = uint8((i * 7) % 256)
		prev[i] = uint8((i * 3) % 256)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TemporalMap(cur, prev, w, h)
	}
}
