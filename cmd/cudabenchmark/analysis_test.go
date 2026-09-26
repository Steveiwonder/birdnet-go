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
	assert.Equal(t, time.Millisecond, s.min)
	assert.Equal(t, 5*time.Millisecond, s.max)
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

	logits := []float32{-4, 3, 0, 1}
	top := topClasses(logits, 2)
	require.Len(t, top, 2)
	assert.Equal(t, 1, top[0].index)
	assert.Equal(t, 3, top[1].index)
	assert.InDelta(t, 0.9526, top[0].score, 1e-3)

	assert.Len(t, topClasses(logits, 10), 4, "k is clamped to the class count")
	assert.Equal(t, []int{1, 2, 3}, detections(logits, 0.5))
}

func TestCompareOutputs(t *testing.T) {
	t.Parallel()

	cpu := [][]float32{{3, -2, 0.5}, {-1, 2, -3}}
	identical := compareOutputs(cpu, cpu, 0.5)
	assert.Equal(t, 2, identical.windows)
	assert.Equal(t, 2, identical.top1Agree)
	assert.Equal(t, 2, identical.detectionSetsAgree)
	assert.Zero(t, identical.maxAbsLogitDiff)

	// Tiny numeric noise keeps the decisions; a flipped class does not.
	gpu := [][]float32{{3.001, -2, 0.5}, {2.5, 2, -3}}
	c := compareOutputs(cpu, gpu, 0.5)
	assert.Equal(t, 1, c.top1Agree)
	assert.Equal(t, 1, c.detectionSetsAgree)
	assert.InDelta(t, 3.5, c.maxAbsLogitDiff, 1e-6)
	assert.Positive(t, c.meanAbsLogitDiff)
	assert.Positive(t, c.maxAbsScoreDiff)
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

	t.Run("pcm16 stereo keeps first channel", func(t *testing.T) {
		t.Parallel()
		data := make([]byte, 8)
		binary.LittleEndian.PutUint16(data[0:], uint16(16384)) // frame 0, left: 0.5
		binary.LittleEndian.PutUint16(data[2:], uint16(1))     // frame 0, right (ignored)
		v := int16(-32768)
		binary.LittleEndian.PutUint16(data[4:], uint16(v)) // frame 1, left: -1
		out, err := monoFloat(data, &wav.StreamInfo{Channels: 2, BitDepth: 16, Format: wav.SampleFormatPCM})
		require.NoError(t, err)
		assert.InDeltaSlice(t, []float32{0.5, -1}, out, 1e-6)
	})

	t.Run("pcm32", func(t *testing.T) {
		t.Parallel()
		data := make([]byte, 4)
		binary.LittleEndian.PutUint32(data, uint32(1<<30)) // 0.5
		out, err := monoFloat(data, &wav.StreamInfo{Channels: 1, BitDepth: 32, Format: wav.SampleFormatPCM})
		require.NoError(t, err)
		assert.InDeltaSlice(t, []float32{0.5}, out, 1e-6)
	})

	t.Run("float32", func(t *testing.T) {
		t.Parallel()
		data := make([]byte, 4)
		binary.LittleEndian.PutUint32(data, math.Float32bits(-0.25))
		out, err := monoFloat(data, &wav.StreamInfo{Channels: 1, BitDepth: 32, Format: wav.SampleFormatFloat})
		require.NoError(t, err)
		assert.Equal(t, []float32{-0.25}, out)
	})

	t.Run("unsupported", func(t *testing.T) {
		t.Parallel()
		_, err := monoFloat(make([]byte, 3), &wav.StreamInfo{Channels: 1, BitDepth: 24, Format: wav.SampleFormatPCM})
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
