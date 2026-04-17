// Package api provides an HTTP server for JND analysis.
package api

import (
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"math"
	"net/http"

	"github.com/bauratynov/fovea/internal/model"
)

// Server is the JND HTTP API server.
type Server struct {
	mux *http.ServeMux
}

// New creates a new API server.
func New() *Server {
	s := &Server{mux: http.NewServeMux()}
	s.mux.HandleFunc("/healthz", s.healthz)
	s.mux.HandleFunc("/analyze", s.analyze)
	s.mux.HandleFunc("/compare", s.compare)
	return s
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) healthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

// AnalyzeResponse is the JSON response for /analyze.
type AnalyzeResponse struct {
	W   int       `json:"w"`
	H   int       `json:"h"`
	JND []float64 `json:"jnd"`

	Stats struct {
		Min  float64 `json:"min"`
		Max  float64 `json:"max"`
		Mean float64 `json:"mean"`
	} `json:"stats"`
}

// CompareResponse is the JSON response for /compare.
type CompareResponse struct {
	W                 int       `json:"w"`
	H                 int       `json:"h"`
	PrunableRatio     float64   `json:"prunable_ratio"`
	BitrateSavingsPct float64   `json:"bitrate_savings_pct"`
	PerBlockJND       []float64 `json:"per_block_jnd"`
}

func decodeJPEGToGray(r io.Reader) ([]uint8, int, int, error) {
	img, err := jpeg.Decode(r)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("jpeg decode: %w", err)
	}
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	gray := make([]uint8, w*h)

	// Check if it's already a Gray image.
	if g, ok := img.(*image.Gray); ok {
		copy(gray, g.Pix)
		return gray, w, h, nil
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			// ITU-R BT.601 luma.
			luma := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
			gray[y*w+x] = uint8(luma)
		}
	}
	return gray, w, h, nil
}

func (s *Server) analyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	gray, imgW, imgH, err := decodeJPEGToGray(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	jndMap := model.StaticJND(gray, imgW, imgH)
	if jndMap == nil {
		http.Error(w, "analysis failed", http.StatusInternalServerError)
		return
	}

	resp := AnalyzeResponse{W: imgW, H: imgH, JND: jndMap.Threshold}
	resp.Stats.Min = math.MaxFloat64
	for _, v := range jndMap.Threshold {
		if v < resp.Stats.Min {
			resp.Stats.Min = v
		}
		if v > resp.Stats.Max {
			resp.Stats.Max = v
		}
		resp.Stats.Mean += v
	}
	resp.Stats.Mean /= float64(len(jndMap.Threshold))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) compare(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(32 << 20) // 32 MB
	if err != nil {
		http.Error(w, "multipart form required with 'frame1' and 'frame2' files", http.StatusBadRequest)
		return
	}

	f1, _, err := r.FormFile("frame1")
	if err != nil {
		http.Error(w, "missing frame1: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer f1.Close()

	f2, _, err := r.FormFile("frame2")
	if err != nil {
		http.Error(w, "missing frame2: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer f2.Close()

	gray1, w1, h1, err := decodeJPEGToGray(f1)
	if err != nil {
		http.Error(w, "frame1: "+err.Error(), http.StatusBadRequest)
		return
	}
	gray2, w2, h2, err := decodeJPEGToGray(f2)
	if err != nil {
		http.Error(w, "frame2: "+err.Error(), http.StatusBadRequest)
		return
	}

	if w1 != w2 || h1 != h2 {
		http.Error(w, "frames must have same dimensions", http.StatusBadRequest)
		return
	}

	jndMap := model.CombinedJND(gray2, gray1, w1, h1)

	// Compute per-pixel errors.
	n := w1 * h1
	errors := make([]float64, n)
	for i := 0; i < n; i++ {
		d := float64(gray2[i]) - float64(gray1[i])
		if d < 0 {
			d = -d
		}
		errors[i] = d
	}

	resp := CompareResponse{
		W:                 w1,
		H:                 h1,
		PrunableRatio:     jndMap.PrunableRatio(errors),
		BitrateSavingsPct: jndMap.BitrateReduction(errors) * 100,
		PerBlockJND:       jndMap.Threshold,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
