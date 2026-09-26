package cudabenchmark

import (
	"bufio"
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
	min    time.Duration
	max    time.Duration
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
		min:    sorted[0],
		max:    sorted[len(sorted)-1],
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

// sigmoid maps a raw logit to a confidence score.
func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// scoredClass is one class index with its confidence.
type scoredClass struct {
	index int
	score float32
}

// topClasses returns the k highest-confidence classes of a logit vector.
func topClasses(logits []float32, k int) []scoredClass {
	classes := make([]scoredClass, len(logits))
	for i, l := range logits {
		classes[i] = scoredClass{index: i, score: sigmoid(l)}
	}
	slices.SortStableFunc(classes, func(a, b scoredClass) int {
		switch {
		case a.score > b.score:
			return -1
		case a.score < b.score:
			return 1
		default:
			return 0
		}
	})
	return classes[:min(k, len(classes))]
}

// detections returns the sorted class indices scoring at or above minScore.
func detections(logits []float32, minScore float32) []int {
	var out []int
	for i, l := range logits {
		if sigmoid(l) >= minScore {
			out = append(out, i)
		}
	}
	return out
}

// comparison describes how closely two providers' outputs agree on the same
// audio windows.
type comparison struct {
	windows            int
	maxAbsLogitDiff    float64
	meanAbsLogitDiff   float64
	maxAbsScoreDiff    float64
	top1Agree          int
	detectionSetsAgree int
}

// compareOutputs compares per-window logits from a reference (CPU) run and a
// candidate (CUDA) run. Windows are compared up to the shorter of the two.
func compareOutputs(reference, candidate [][]float32, minScore float32) comparison {
	c := comparison{windows: min(len(reference), len(candidate))}
	var sum float64
	var n int
	for w := range c.windows {
		ref, cand := reference[w], candidate[w]
		for i := range min(len(ref), len(cand)) {
			d := math.Abs(float64(ref[i]) - float64(cand[i]))
			c.maxAbsLogitDiff = max(c.maxAbsLogitDiff, d)
			sum += d
			n++
			sd := math.Abs(float64(sigmoid(ref[i])) - float64(sigmoid(cand[i])))
			c.maxAbsScoreDiff = max(c.maxAbsScoreDiff, sd)
		}
		if len(ref) > 0 && len(cand) > 0 && topClasses(ref, 1)[0].index == topClasses(cand, 1)[0].index {
			c.top1Agree++
		}
		if slices.Equal(detections(ref, minScore), detections(cand, minScore)) {
			c.detectionSetsAgree++
		}
	}
	if n > 0 {
		c.meanAbsLogitDiff = sum / float64(n)
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
