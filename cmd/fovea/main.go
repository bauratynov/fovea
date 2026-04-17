// Command fovea provides CLI access to the Fovea JND perceptual model.
//
// Usage:
//
//	fovea analyze image.jpg       — compute JND heatmap + stats
//	fovea compare f1.jpg f2.jpg   — pruning analysis between two frames
//	fovea bench                   — benchmark all models at 1080p
//	fovea serve [:port]           — start HTTP API server
package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"time"

	"github.com/bauratynov/fovea/internal/api"
	"github.com/bauratynov/fovea/internal/luminance"
	"github.com/bauratynov/fovea/internal/model"
	"github.com/bauratynov/fovea/internal/temporal"
	"github.com/bauratynov/fovea/internal/texture"
	"net/http"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "analyze":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: fovea analyze image.jpg")
			os.Exit(1)
		}
		cmdAnalyze(os.Args[2])
	case "compare":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "usage: fovea compare frame1.jpg frame2.jpg")
			os.Exit(1)
		}
		cmdCompare(os.Args[2], os.Args[3])
	case "bench":
		cmdBench()
	case "serve":
		addr := ":8080"
		if len(os.Args) >= 3 {
			addr = os.Args[2]
		}
		cmdServe(addr)
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: fovea <analyze|compare|bench|serve> [args]")
	fmt.Fprintln(os.Stderr, "  analyze image.jpg         JND heatmap + stats")
	fmt.Fprintln(os.Stderr, "  compare f1.jpg f2.jpg     pruning analysis")
	fmt.Fprintln(os.Stderr, "  bench                     benchmark 1080p")
	fmt.Fprintln(os.Stderr, "  serve [:port]             HTTP API")
}

func loadGray(path string) ([]uint8, int, int) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open %s: %v\n", path, err)
		os.Exit(1)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "decode %s: %v\n", path, err)
		os.Exit(1)
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	gray := make([]uint8, w*h)

	if g, ok := img.(*image.Gray); ok {
		copy(gray, g.Pix)
		return gray, w, h
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			luma := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
			gray[y*w+x] = uint8(luma)
		}
	}
	return gray, w, h
}

type analyzeOutput struct {
	W    int     `json:"w"`
	H    int     `json:"h"`
	Min  float64 `json:"min_jnd"`
	Max  float64 `json:"max_jnd"`
	Mean float64 `json:"mean_jnd"`
}

func cmdAnalyze(path string) {
	gray, w, h := loadGray(path)
	m := model.StaticJND(gray, w, h)

	out := analyzeOutput{W: w, H: h}
	out.Min = math.MaxFloat64
	for _, v := range m.Threshold {
		if v < out.Min {
			out.Min = v
		}
		if v > out.Max {
			out.Max = v
		}
		out.Mean += v
	}
	out.Mean /= float64(len(m.Threshold))

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(out)
}

type compareOutput struct {
	W                 int     `json:"w"`
	H                 int     `json:"h"`
	PrunableRatio     float64 `json:"prunable_ratio_pct"`
	BitrateSavingsPct float64 `json:"bitrate_savings_pct"`
	MeanJND           float64 `json:"mean_jnd"`
	MeanError         float64 `json:"mean_error"`
}

func cmdCompare(path1, path2 string) {
	gray1, w1, h1 := loadGray(path1)
	gray2, w2, h2 := loadGray(path2)

	if w1 != w2 || h1 != h2 {
		fmt.Fprintf(os.Stderr, "dimension mismatch: %dx%d vs %dx%d\n", w1, h1, w2, h2)
		os.Exit(1)
	}

	jndMap := model.CombinedJND(gray2, gray1, w1, h1)
	n := w1 * h1
	errors := make([]float64, n)
	for i := 0; i < n; i++ {
		d := float64(gray2[i]) - float64(gray1[i])
		if d < 0 {
			d = -d
		}
		errors[i] = d
	}

	out := compareOutput{
		W:                 w1,
		H:                 h1,
		PrunableRatio:     jndMap.PrunableRatio(errors) * 100,
		BitrateSavingsPct: jndMap.BitrateReduction(errors) * 100,
	}
	for _, v := range jndMap.Threshold {
		out.MeanJND += v
	}
	out.MeanJND /= float64(n)
	for _, e := range errors {
		out.MeanError += e
	}
	out.MeanError /= float64(n)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(out)
}

func cmdBench() {
	w, h := 1920, 1080
	n := w * h

	gray := make([]uint8, n)
	prev := make([]uint8, n)
	for i := range gray {
		gray[i] = uint8((i*137 + 59) % 256)
		prev[i] = uint8((i*31 + 17) % 256)
	}

	const iters = 10

	// Luminance
	start := time.Now()
	for i := 0; i < iters; i++ {
		luminance.LuminanceMap(gray, w, h)
	}
	lumMs := float64(time.Since(start).Milliseconds()) / float64(iters)

	// Texture
	start = time.Now()
	for i := 0; i < iters; i++ {
		texture.TextureMap(gray, w, h)
	}
	texMs := float64(time.Since(start).Milliseconds()) / float64(iters)

	// Temporal
	start = time.Now()
	for i := 0; i < iters; i++ {
		temporal.TemporalMap(gray, prev, w, h)
	}
	tempMs := float64(time.Since(start).Milliseconds()) / float64(iters)

	// Combined
	start = time.Now()
	for i := 0; i < iters; i++ {
		model.CombinedJND(gray, prev, w, h)
	}
	combMs := float64(time.Since(start).Milliseconds()) / float64(iters)

	// PrunableRatio
	jndMap := model.CombinedJND(gray, prev, w, h)
	errors := make([]float64, n)
	for i := range errors {
		errors[i] = float64(1 + (i*7)%8)
	}
	start = time.Now()
	for i := 0; i < iters*100; i++ {
		jndMap.PrunableRatio(errors)
	}
	pruneMs := float64(time.Since(start).Microseconds()) / float64(iters*100)

	fmt.Printf("Fovea benchmarks (1080p, %d iterations):\n", iters)
	fmt.Printf("  LuminanceMap:  %6.1f ms/frame\n", lumMs)
	fmt.Printf("  TextureMap:    %6.1f ms/frame\n", texMs)
	fmt.Printf("  TemporalMap:   %6.1f ms/frame\n", tempMs)
	fmt.Printf("  CombinedJND:   %6.1f ms/frame\n", combMs)
	fmt.Printf("  PrunableRatio: %6.1f us/call\n", pruneMs)

	ratio := jndMap.PrunableRatio(errors)
	reduction := jndMap.BitrateReduction(errors)
	fmt.Printf("\nPrunable ratio:     %.1f%%\n", ratio*100)
	fmt.Printf("Bitrate reduction:  %.1f%%\n", reduction*100)
}

func cmdServe(addr string) {
	srv := api.New()
	fmt.Printf("Fovea API listening on %s\n", addr)
	if err := http.ListenAndServe(addr, srv.Handler()); err != nil {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}
}
