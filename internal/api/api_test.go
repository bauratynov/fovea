package api

import (
	"bytes"
	"encoding/json"
	"image"
	"image/jpeg"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
)

func makeJPEG(w, h int, val uint8) []byte {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = val
	}
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	return buf.Bytes()
}

func TestHealthz(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if string(body) != `{"status":"ok"}` {
		t.Fatalf("body: %s", body)
	}
}

func TestAnalyze(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	jpgData := makeJPEG(32, 32, 128)
	resp, err := http.Post(ts.URL+"/analyze", "image/jpeg", bytes.NewReader(jpgData))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}

	var result AnalyzeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.W != 32 || result.H != 32 {
		t.Fatalf("dims: %dx%d", result.W, result.H)
	}
	if len(result.JND) != 32*32 {
		t.Fatalf("JND len: %d", len(result.JND))
	}
	if result.Stats.Min <= 0 || result.Stats.Max <= 0 {
		t.Fatalf("stats: min=%f max=%f", result.Stats.Min, result.Stats.Max)
	}
}

func TestAnalyze_MethodNotAllowed(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/analyze")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", resp.StatusCode)
	}
}

func TestCompare(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	jpg1 := makeJPEG(32, 32, 100)
	jpg2 := makeJPEG(32, 32, 110)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	p1, _ := writer.CreateFormFile("frame1", "frame1.jpg")
	p1.Write(jpg1)
	p2, _ := writer.CreateFormFile("frame2", "frame2.jpg")
	p2.Write(jpg2)
	writer.Close()

	resp, err := http.Post(ts.URL+"/compare", writer.FormDataContentType(), &body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, b)
	}

	var result CompareResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.W != 32 || result.H != 32 {
		t.Fatalf("dims: %dx%d", result.W, result.H)
	}
	t.Logf("PrunableRatio=%.1f%% BitrateSavings=%.1f%%", result.PrunableRatio*100, result.BitrateSavingsPct)
}

func TestHeatmapDimensions(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, dim := range [][2]int{{16, 16}, {33, 17}, {64, 48}} {
		w, h := dim[0], dim[1]
		jpgData := makeJPEG(w, h, 128)
		resp, err := http.Post(ts.URL+"/analyze", "image/jpeg", bytes.NewReader(jpgData))
		if err != nil {
			t.Fatal(err)
		}
		var result AnalyzeResponse
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()

		if result.W != w || result.H != h {
			t.Fatalf("input %dx%d → output %dx%d", w, h, result.W, result.H)
		}
		if len(result.JND) != w*h {
			t.Fatalf("JND len %d, want %d", len(result.JND), w*h)
		}
	}
}
