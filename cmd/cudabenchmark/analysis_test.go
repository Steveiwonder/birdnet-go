package cudabenchmark

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/go-wav"

	"github.com/tphakala/birdnet-go/internal/inference/onnx"
)

func TestSummarizeLatencies(t *testing.T) {
	t.Parallel()

	assert.Equal(t, latencyStats{}, summarizeLatencies(nil))

	in := []time.Duration{5 * time.Millisecond, 1 * time.Millisecond, 3 * time.Millisecond, 2 * time.Millisecond, 4 * time.Millisecond}
	s := summarizeLatencies(in)
	assert.Equal(t, 5, s.count)
	assert.Equal(t, 3*time.Millisecond, s.mean)
	assert.Equal(t, 3*time.Millisecond, s.median)
	assert.Equal(t, 5*time.Millisecond, s.p95)
	assert.Equal(t, 5*time.Millisecond, in[0], "input must not be reordered")
}

func TestChunkSamples(t *testing.T) {
	t.Parallel()

	assert.Nil(t, chunkSamples(nil, 3))
	assert.Nil(t, chunkSamples([]float32{1}, 0))

	chunks := chunkSamples([]float32{1, 2, 3, 4, 5, 6, 7}, 3)
	require.Len(t, chunks, 3)
	assert.Equal(t, []float32{1, 2, 3}, chunks[0])
	assert.Equal(t, []float32{7, 0, 0}, chunks[2], "last window is zero-padded")
}

func TestTopClassesAndDetections(t *testing.T) {
	t.Parallel()

	scores := []float32{0.01, 0.95, 0.5, 0.7}
	top := topClasses(scores, 2)
	require.Len(t, top, 2)
	assert.Equal(t, 1, top[0].index)
	assert.Equal(t, 3, top[1].index)

	assert.Len(t, topClasses(scores, 10), 4, "k is clamped to the class count")
	assert.Equal(t, []int{1, 2, 3}, detections(scores, 0.5))
	assert.Equal(t, 1, argmax(scores))
	assert.Equal(t, -1, argmax(nil))
}

// window builds a window output whose scores are the BirdNET v2.4 activation
// (sigmoid) of raw, as the benchmark does through onnx.Activate.
func window(raw ...float32) windowOutput {
	return windowOutput{raw: raw, scores: onnx.Activate(onnx.BirdNETv24, raw)}
}

func TestCompareOutputs(t *testing.T) {
	t.Parallel()

	cpu := []windowOutput{window(3, -2, 0.5), window(-1, 2, -3)}
	identical := compareOutputs(cpu, cpu, 0.5)
	assert.Equal(t, 2, identical.windows)
	assert.Equal(t, 2, identical.top1Agree)
	assert.Equal(t, 2, identical.detectionSetsAgree)
	assert.Zero(t, identical.maxAbsRawDiff)

	// Tiny numeric noise keeps the decisions; a flipped class does not.
	gpu := []windowOutput{window(3.001, -2, 0.5), window(2.5, 2, -3)}
	c := compareOutputs(cpu, gpu, 0.5)
	assert.Equal(t, 1, c.top1Agree)
	assert.Equal(t, 1, c.detectionSetsAgree)
	assert.InDelta(t, 3.5, c.maxAbsRawDiff, 1e-6)
	assert.Positive(t, c.meanAbsRawDiff)
	assert.Positive(t, c.maxAbsScoreDiff)
}

// BirdNET v3.0 outputs probabilities; comparing them must not re-apply a
// sigmoid, which would push every class above a 0.5 threshold.
func TestCompareOutputs_V30ScoresNotResquashed(t *testing.T) {
	t.Parallel()

	raw := []float32{0.9, 0.1, 0.2}
	w := windowOutput{raw: raw, scores: onnx.Activate(onnx.BirdNETv30, raw)}
	assert.Equal(t, []int{0}, detections(w.scores, 0.5))
}

func TestSyntheticAudio(t *testing.T) {
	t.Parallel()

	a := syntheticAudio(480, 48000)
	b := syntheticAudio(480, 48000)
	require.Len(t, a, 480)
	assert.Equal(t, a, b, "the synthetic signal is deterministic")
	for _, v := range a {
		assert.LessOrEqual(t, math.Abs(float64(v)), synthAmplitude+synthNoise)
	}
}

func TestParseNvidiaSmiLine(t *testing.T) {
	t.Parallel()

	s, ok := parseNvidiaSmiLine("37, 1024")
	require.True(t, ok)
	assert.InDelta(t, 37.0, s.utilizationPct, 0)
	assert.InDelta(t, 1024.0, s.memoryUsedMiB, 0)

	for _, bad := range []string{"", "[N/A], 100", "37", "a, b, c"} {
		_, ok := parseNvidiaSmiLine(bad)
		assert.Falsef(t, ok, "line %q", bad)
	}
}

func TestSummarizeGPU(t *testing.T) {
	t.Parallel()

	s := summarizeGPU(strings.NewReader("10, 500\n[N/A], 0\n30, 700\n"))
	assert.Equal(t, 2, s.samples)
	assert.InDelta(t, 20.0, s.avgUtilization, 1e-9)
	assert.InDelta(t, 30.0, s.maxUtilization, 0)
	assert.InDelta(t, 700.0, s.maxMemoryMiB, 0)

	assert.Zero(t, summarizeGPU(strings.NewReader("")).samples)
}

func TestMonoFloat(t *testing.T) {
	t.Parallel()

	t.Run("stereo keeps first channel", func(t *testing.T) {
		t.Parallel()
		data := make([]byte, 16)
		binary.LittleEndian.PutUint32(data[0:], uint32(1<<30)) // frame 0, left: 0.5
		binary.LittleEndian.PutUint32(data[4:], 1)             // frame 0, right (ignored)
		v := int32(math.MinInt32)
		binary.LittleEndian.PutUint32(data[8:], uint32(v)) // frame 1, left: -1
		out, err := monoFloat(data, &wav.StreamInfo{Channels: 2, BitDepth: 32, Format: wav.SampleFormatPCM})
		require.NoError(t, err)
		assert.InDeltaSlice(t, []float32{0.5, -1}, out, 1e-6)
	})

	t.Run("rejects unconverted width and no channels", func(t *testing.T) {
		t.Parallel()
		_, err := monoFloat(make([]byte, 4), &wav.StreamInfo{Channels: 1, BitDepth: 16, Format: wav.SampleFormatPCM})
		require.Error(t, err)
		_, err = monoFloat(nil, &wav.StreamInfo{})
		require.Error(t, err)
	})
}

func TestClassName(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "#3", className([]string{placeholderLabel}, 3))
	assert.Equal(t, "b", className([]string{"a", "b"}, 1))
}
