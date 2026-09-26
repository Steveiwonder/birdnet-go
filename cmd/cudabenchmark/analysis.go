package cudabenchmark

import (
	"bufio"
	"cmp"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
)

// latencyStats summarizes per-inference latencies.
type latencyStats struct {
	count  int
	mean   time.Duration
	median time.Duration
	p95    time.Duration
}

// percentile95 is the latency percentile reported alongside the median.
const percentile95 = 0.95

// summarizeLatencies computes latency statistics; the input is not modified.
func summarizeLatencies(durations []time.Duration) latencyStats {
	if len(durations) == 0 {
		return latencyStats{}
	}
	sorted := slices.Clone(durations)
	slices.Sort(sorted)
	var total time.Duration
	for _, d := range sorted {
		total += d
	}
	p95Index := min(int(math.Ceil(percentile95*float64(len(sorted))))-1, len(sorted)-1)
	return latencyStats{
		count:  len(sorted),
		mean:   total / time.Duration(len(sorted)),
		median: sorted[len(sorted)/2],
		p95:    sorted[max(p95Index, 0)],
	}
}

// chunkSamples splits audio into consecutive windows of size samples. The last
// partial window is zero-padded; empty audio yields no windows.
func chunkSamples(audio []float32, size int) [][]float32 {
	if size <= 0 || len(audio) == 0 {
		return nil
	}
	chunks := make([][]float32, 0, (len(audio)+size-1)/size)
	for start := 0; start < len(audio); start += size {
		chunk := make([]float32, size)
		copy(chunk, audio[start:min(start+size, len(audio))])
		chunks = append(chunks, chunk)
	}
	return chunks
}

// scoredClass is one class index with its confidence.
type scoredClass struct {
	index int
	score float32
}

// topClasses returns the k highest-scoring classes, highest first.
func topClasses(scores []float32, k int) []scoredClass {
	classes := make([]scoredClass, len(scores))
	for i, sc := range scores {
		classes[i] = scoredClass{index: i, score: sc}
	}
	slices.SortStableFunc(classes, func(a, b scoredClass) int {
		return cmp.Compare(b.score, a.score)
	})
	return classes[:min(k, len(classes))]
}

// argmax returns the index of the highest score, or -1 for an empty slice.
func argmax(scores []float32) int {
	best := -1
	for i, sc := range scores {
		if best < 0 || sc > scores[best] {
			best = i
		}
	}
	return best
}

// detections returns the sorted class indices scoring at or above minScore.
func detections(scores []float32, minScore float32) []int {
	var out []int
	for i, sc := range scores {
		if sc >= minScore {
			out = append(out, i)
		}
	}
	return out
}

// windowOutput is one analysis window's raw model output and the activated
// per-class scores derived from it (see onnx.Activate).
type windowOutput struct {
	raw    []float32
	scores []float32
}

// comparison describes how closely two providers' outputs agree on the same
// audio windows.
type comparison struct {
	windows            int
	maxAbsRawDiff      float64
	meanAbsRawDiff     float64
	maxAbsScoreDiff    float64
	top1Agree          int
	detectionSetsAgree int
}

// compareOutputs compares per-window outputs from a reference (CPU) run and a
// candidate (CUDA) run. Windows are compared up to the shorter of the two.
func compareOutputs(reference, candidate []windowOutput, minScore float32) comparison {
	c := comparison{windows: min(len(reference), len(candidate))}
	var sum float64
	var n int
	for w := range c.windows {
		ref, cand := reference[w], candidate[w]
		for i := range min(len(ref.raw), len(cand.raw)) {
			d := math.Abs(float64(ref.raw[i]) - float64(cand.raw[i]))
			c.maxAbsRawDiff = max(c.maxAbsRawDiff, d)
			sum += d
			n++
		}
		for i := range min(len(ref.scores), len(cand.scores)) {
			c.maxAbsScoreDiff = max(c.maxAbsScoreDiff, math.Abs(float64(ref.scores[i])-float64(cand.scores[i])))
		}
		if top := argmax(ref.scores); top >= 0 && top == argmax(cand.scores) {
			c.top1Agree++
		}
		if slices.Equal(detections(ref.scores, minScore), detections(cand.scores, minScore)) {
			c.detectionSetsAgree++
		}
	}
	if n > 0 {
		c.meanAbsRawDiff = sum / float64(n)
	}
	return c
}

// gpuSample is one nvidia-smi reading.
type gpuSample struct {
	utilizationPct float64
	memoryUsedMiB  float64
}

// nvidiaSmiFields is the number of CSV fields requested from nvidia-smi.
const nvidiaSmiFields = 2

// parseNvidiaSmiLine parses one "utilization.gpu, memory.used" CSV line
// (nounits). It reports false for anything else, such as "[N/A]" values.
func parseNvidiaSmiLine(line string) (gpuSample, bool) {
	fields := strings.Split(line, ",")
	if len(fields) != nvidiaSmiFields {
		return gpuSample{}, false
	}
	util, err := strconv.ParseFloat(strings.TrimSpace(fields[0]), 64)
	if err != nil {
		return gpuSample{}, false
	}
	mem, err := strconv.ParseFloat(strings.TrimSpace(fields[1]), 64)
	if err != nil {
		return gpuSample{}, false
	}
	return gpuSample{utilizationPct: util, memoryUsedMiB: mem}, true
}

// gpuSummary aggregates nvidia-smi readings taken during a run.
type gpuSummary struct {
	samples        int
	maxUtilization float64
	avgUtilization float64
	maxMemoryMiB   float64
}

// summarizeGPU reads nvidia-smi CSV lines and aggregates the valid samples.
func summarizeGPU(r io.Reader) gpuSummary {
	var s gpuSummary
	var total float64
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		sample, ok := parseNvidiaSmiLine(scanner.Text())
		if !ok {
			continue
		}
		s.samples++
		total += sample.utilizationPct
		s.maxUtilization = max(s.maxUtilization, sample.utilizationPct)
		s.maxMemoryMiB = max(s.maxMemoryMiB, sample.memoryUsedMiB)
	}
	if s.samples > 0 {
		s.avgUtilization = total / float64(s.samples)
	}
	return s
}
