// Package cudabenchmark implements the "cuda-benchmark" command, which runs the
// same audio through an ONNX classifier on the CPU and on the CUDA execution
// provider and reports latency, CPU use, GPU activity and whether the two
// providers produce the same detections.
package cudabenchmark

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"
	"github.com/tphakala/go-wav"
	"github.com/tphakala/go-wav/pcm"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/errors"
	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/inference/onnx"
)

// Defaults for the command flags.
const (
	defaultIterations = 3
	defaultTopK       = 5
	defaultMinScore   = 0.5
	// synthWindows is how many model windows of synthetic audio are generated
	// when no audio file is given.
	synthWindows = 10
	warmupRuns   = 3
	// gpuSampleIntervalMs is the nvidia-smi sampling period during the CUDA run.
	gpuSampleIntervalMs = 200
	// placeholderLabel stands in for species names when no label file is given.
	placeholderLabel = "?"
	// percentScale converts a ratio to a percentage.
	percentScale = 100
	// pcmBitDepth is the width every WAV input is converted to before scaling;
	// pcm32Scale normalizes signed 32-bit PCM to [-1, 1].
	pcmBitDepth = 32
	pcm32Scale  = 2147483648.0
	// synthAmplitude, synthNoise and synthToneHz shape the synthetic test
	// signal used when no audio file is given: a quiet 3 kHz tone in low-level
	// noise, generated at the model's sample rate.
	synthAmplitude = 0.2
	synthNoise     = 0.02
	synthToneHz    = 3000
	synthSeed      = 42
)

// flagCUDADevice is the flag selecting the CUDA device.
const flagCUDADevice = "cuda-device"

// options holds the command flags.
type options struct {
	modelPath  string
	labelsPath string
	audioPath  string
	iterations int
	deviceID   int
	topK       int
	minScore   float64
	threads    int
	ortPath    string
}

// providerRun is the result of benchmarking one execution provider.
type providerRun struct {
	provider  string
	loadTime  time.Duration
	latency   latencyStats
	wall      time.Duration
	cpuTime   time.Duration
	cpuKnown  bool
	outputs   []windowOutput
	gpu       gpuSummary
	gpuStatus string
	// placement is the measured operator placement of the CUDA session.
	placement *onnx.NodePlacement
	err       error
}

// Command returns the cuda-benchmark command.
func Command(settings *conf.Settings) *cobra.Command {
	opts := options{}
	cmd := &cobra.Command{
		Use:   "cuda-benchmark",
		Short: "Compare ONNX model inference on the CPU and on an NVIDIA GPU (CUDA)",
		Long: `Runs the same audio through an ONNX classifier twice, first on the CPU
execution provider and then on the CUDA execution provider, and reports load
time, per-window latency, process CPU use, GPU utilization (sampled with
nvidia-smi) and how closely the detections agree.

The CUDA run uses the same code path as the application: it fails, rather than
falling back to the CPU, when CUDA cannot be initialized or when no model
operator runs on the GPU.`,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, _ []string) error {
			opts.threads = settings.BirdNET.Threads
			if !cmd.Flags().Changed(flagCUDADevice) {
				opts.deviceID = settings.BirdNET.CUDADeviceID
			}
			if opts.ortPath == "" {
				opts.ortPath = settings.BirdNET.ONNXRuntimePath
			}
			return run(cmd.Context(), cmd.OutOrStdout(), &opts)
		},
	}
	f := cmd.Flags()
	f.StringVar(&opts.modelPath, "model", "", "ONNX classifier model to benchmark (required)")
	f.StringVar(&opts.labelsPath, "labels", "", "label file for the model (optional; enables species names and label validation)")
	f.StringVar(&opts.audioPath, "audio", "", "mono WAV file at the model's sample rate (default: synthetic signal)")
	f.IntVar(&opts.iterations, "iterations", defaultIterations, "passes over the audio per provider")
	f.IntVar(&opts.deviceID, flagCUDADevice, 0, "CUDA device ordinal (default: birdnet.cudadeviceid)")
	f.IntVar(&opts.topK, "top", defaultTopK, "classes listed per window in the comparison")
	f.Float64Var(&opts.minScore, "min-score", defaultMinScore, "confidence at which a class counts as detected")
	f.StringVar(&opts.ortPath, "onnxruntime", "", "path to libonnxruntime (default: birdnet.onnxruntimepath or the system search path)")
	_ = cmd.MarkFlagRequired("model")
	return cmd
}

// run executes the benchmark and writes the report to out.
func run(ctx context.Context, out io.Writer, opts *options) error {
	if opts.iterations < 1 {
		return errors.NewStd("--iterations must be at least 1")
	}
	if err := inference.InitONNXRuntime(opts.ortPath); err != nil {
		return err
	}
	modelCfg, err := onnx.DetectModelConfig(opts.modelPath)
	if err != nil {
		return err
	}
	sampleCount := modelCfg.SampleCount
	labels, skipValidation, err := loadLabels(opts.labelsPath)
	if err != nil {
		return err
	}
	audio, audioDesc, err := loadAudio(opts.audioPath, &modelCfg)
	if err != nil {
		return err
	}
	windows := chunkSamples(audio, sampleCount)
	if len(windows) == 0 {
		return errors.NewStd("audio contains no samples")
	}

	_, _ = fmt.Fprintf(out, "Model:      %s (%s)\n", filepath.Base(opts.modelPath), modelCfg.Type)
	_, _ = fmt.Fprintf(out, "Audio:      %s, %d windows of %d samples, %d passes per provider\n",
		audioDesc, len(windows), sampleCount, opts.iterations)
	_, _ = fmt.Fprintf(out, "Threads:    %d (0 = all CPUs)\n\n", opts.threads)

	cpu := benchmarkProvider(ctx, opts, labels, skipValidation, windows, modelCfg.Type,
		inference.ExecutionProviderOptions{Provider: inference.ExecutionProviderCPU})
	cuda := benchmarkProvider(ctx, opts, labels, skipValidation, windows, modelCfg.Type,
		inference.ExecutionProviderOptions{Provider: inference.ExecutionProviderCUDA, DeviceID: opts.deviceID})

	printReport(out, opts, labels, &cpu, &cuda)
	if cpu.err != nil {
		return cpu.err
	}
	return cuda.err
}

// benchmarkProvider loads the model on one provider and runs every window
// opts.iterations times, recording the first pass's outputs for comparison.
func benchmarkProvider(ctx context.Context, opts *options, labels []string, skipValidation bool, windows [][]float32, modelType onnx.ModelType, ep inference.ExecutionProviderOptions) providerRun {
	res := providerRun{provider: ep.Provider}
	start := time.Now()
	classifier, err := inference.NewONNXClassifier(opts.modelPath, inference.ONNXClassifierOptions{
		Labels:              labels,
		Threads:             opts.threads,
		SkipLabelValidation: skipValidation,
		ExecutionProvider:   ep,
	})
	if err != nil {
		res.err = fmt.Errorf("%s: %w", ep.Provider, err)
		return res
	}
	defer classifier.Close()
	res.loadTime = time.Since(start)
	if p, ok := classifier.(placementReporter); ok {
		res.placement = p.Placement()
	}

	for range warmupRuns {
		if _, err := classifier.Predict(windows[0]); err != nil {
			res.err = fmt.Errorf("%s warm-up: %w", ep.Provider, err)
			return res
		}
	}

	var stopGPU func() (gpuSummary, string)
	if ep.UsesCUDA() {
		stopGPU = startGPUMonitor(ctx, ep.DeviceID)
		// Stop the sampler on every return path; the success path below reads
		// its summary first, and stopping twice is a no-op.
		defer stopGPU()
	}

	cpuBefore, cpuOK := processCPUTime()
	durations := make([]time.Duration, 0, len(windows)*opts.iterations)
	res.outputs = make([]windowOutput, 0, len(windows))
	runStart := time.Now()
	for pass := range opts.iterations {
		for _, w := range windows {
			t0 := time.Now()
			logits, err := classifier.Predict(w)
			durations = append(durations, time.Since(t0))
			if err != nil {
				res.err = fmt.Errorf("%s inference: %w", ep.Provider, err)
				return res
			}
			if pass == 0 {
				res.outputs = append(res.outputs, windowOutput{raw: logits, scores: onnx.Activate(modelType, logits)})
			}
		}
	}
	res.wall = time.Since(runStart)
	cpuAfter, cpuAfterOK := processCPUTime()
	res.cpuKnown = cpuOK && cpuAfterOK
	res.cpuTime = cpuAfter - cpuBefore
	res.latency = summarizeLatencies(durations)
	if stopGPU != nil {
		res.gpu, res.gpuStatus = stopGPU()
	}
	return res
}

// placementReporter is implemented by ONNX classifiers that measured their
// operator placement at load time (CUDA sessions).
type placementReporter interface {
	Placement() *onnx.NodePlacement
}

// startGPUMonitor samples GPU utilization and memory with nvidia-smi until the
// returned stop function is called. The status string explains a missing
// measurement (for example when nvidia-smi is not installed).
func startGPUMonitor(ctx context.Context, deviceID int) func() (gpuSummary, string) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return func() (gpuSummary, string) { return gpuSummary{}, "nvidia-smi not found" }
	}
	monitorCtx, cancel := context.WithCancel(ctx)
	var buf bytes.Buffer
	cmd := exec.CommandContext(monitorCtx, path,
		"--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
		"-i", strconv.Itoa(deviceID), "-lms", strconv.Itoa(gpuSampleIntervalMs))
	cmd.Stdout = &buf
	if err := cmd.Start(); err != nil {
		cancel()
		return func() (gpuSummary, string) { return gpuSummary{}, "nvidia-smi failed: " + err.Error() }
	}
	var once sync.Once
	var summary gpuSummary
	var status string
	return func() (gpuSummary, string) {
		once.Do(func() {
			cancel()
			_ = cmd.Wait()
			summary = summarizeGPU(&buf)
			if summary.samples == 0 {
				status = "nvidia-smi returned no samples"
			}
		})
		return summary, status
	}
}

// loadLabels reads the label file, or returns a placeholder list and asks the
// classifier to skip label validation when none is given.
func loadLabels(path string) (labels []string, skipValidation bool, err error) {
	if path == "" {
		return []string{placeholderLabel}, true, nil
	}
	labels, err = onnx.LoadLabels(path)
	if err != nil {
		return nil, false, err
	}
	return labels, false, nil
}

// loadAudio returns mono float samples from a WAV file, or a deterministic
// synthetic signal when path is empty.
func loadAudio(path string, modelCfg *onnx.ModelConfig) (samples []float32, desc string, err error) {
	if path == "" {
		return syntheticAudio(synthWindows*modelCfg.SampleCount, modelCfg.SampleRate), "synthetic tone + noise", nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer func() { _ = f.Close() }()
	data, info, err := pcm.DecodeInterleaved(f, pcm.WithConvertTo(pcmBitDepth))
	if err != nil {
		return nil, "", fmt.Errorf("decode %s: %w", filepath.Base(path), err)
	}
	if info.SampleRate != modelCfg.SampleRate {
		return nil, "", fmt.Errorf("%s is %d Hz but the model expects %d Hz; resample it first (for example with sox or ffmpeg)",
			filepath.Base(path), info.SampleRate, modelCfg.SampleRate)
	}
	samples, err = monoFloat(data, &info)
	if err != nil {
		return nil, "", fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	return samples, fmt.Sprintf("%s (%d Hz)", filepath.Base(path), info.SampleRate), nil
}

// monoFloat converts decoded little-endian signed 32-bit PCM (every input is
// decoded with pcm.WithConvertTo(pcmBitDepth)) to mono float32 samples,
// keeping the first channel.
func monoFloat(data []byte, info *wav.StreamInfo) ([]float32, error) {
	if info.Channels < 1 {
		return nil, errors.NewStd("stream has no channels")
	}
	if info.BitDepth != pcmBitDepth {
		return nil, fmt.Errorf("unexpected decoded width %d-bit", info.BitDepth)
	}
	frame := info.BytesPerFrame()
	frames := len(data) / frame
	out := make([]float32, frames)
	for i := range frames {
		out[i] = float32(float64(int32(binary.LittleEndian.Uint32(data[i*frame:]))) / pcm32Scale)
	}
	return out, nil
}

// syntheticAudio builds a reproducible test signal: a quiet tone in low-level
// noise. It is not birdsong, but it is identical for both providers, which is
// all a CPU-versus-GPU comparison needs; pass --audio for real detections.
func syntheticAudio(n, sampleRate int) []float32 {
	rng := rand.New(rand.NewPCG(synthSeed, synthSeed)) //nolint:gosec // deterministic test signal, not security relevant
	out := make([]float32, n)
	for i := range out {
		tone := synthAmplitude * math.Sin(2*math.Pi*synthToneHz*float64(i)/float64(sampleRate))
		out[i] = float32(tone + synthNoise*(rng.Float64()*2-1))
	}
	return out
}

// printReport writes the side-by-side results.
func printReport(out io.Writer, opts *options, labels []string, cpu, cuda *providerRun) {
	_, _ = fmt.Fprintln(out, "Provider  Load       Mean       Median     p95        Windows/s  CPU use")
	_, _ = fmt.Fprintln(out, "--------  ---------  ---------  ---------  ---------  ---------  --------")
	for _, r := range []*providerRun{cpu, cuda} {
		if r.err != nil {
			_, _ = fmt.Fprintf(out, "%-8s  FAILED: %v\n", r.provider, r.err)
			continue
		}
		throughput := float64(r.latency.count) / r.wall.Seconds()
		cpuUse := "n/a"
		if r.cpuKnown && r.wall > 0 {
			cpuUse = fmt.Sprintf("%.0f%%", percentScale*r.cpuTime.Seconds()/r.wall.Seconds())
		}
		_, _ = fmt.Fprintf(out, "%-8s  %-9s  %-9s  %-9s  %-9s  %-9.1f  %s\n", r.provider,
			fmtMs(r.loadTime), fmtMs(r.latency.mean), fmtMs(r.latency.median), fmtMs(r.latency.p95), throughput, cpuUse)
	}
	_, _ = fmt.Fprintln(out, "(CPU use is process CPU time over wall time; 100% = one core busy.)")

	if cuda.err == nil {
		if cuda.gpuStatus != "" {
			_, _ = fmt.Fprintf(out, "\nGPU activity: %s\n", cuda.gpuStatus)
		} else {
			_, _ = fmt.Fprintf(out, "\nGPU activity (device %d, %d samples): avg %.0f%%, peak %.0f%% utilization, peak %.0f MiB memory\n",
				opts.deviceID, cuda.gpu.samples, cuda.gpu.avgUtilization, cuda.gpu.maxUtilization, cuda.gpu.maxMemoryMiB)
		}
		if p := cuda.placement; p != nil {
			_, _ = fmt.Fprintf(out, "CUDA operator placement: %d of %d operators on the GPU", p.Count(onnx.ProviderCUDA), p.Total)
			if cpuOps := p.Ops(onnx.ProviderCPU); len(cpuOps) > 0 {
				_, _ = fmt.Fprintf(out, " (CPU: %s)", strings.Join(cpuOps, ", "))
			}
			_, _ = fmt.Fprintln(out)
		}
	}

	if cpu.err != nil || cuda.err != nil {
		return
	}
	c := compareOutputs(cpu.outputs, cuda.outputs, float32(opts.minScore))
	_, _ = fmt.Fprintf(out, "\nDetection comparison over %d windows (CPU is the reference):\n", c.windows)
	_, _ = fmt.Fprintf(out, "  top-1 class identical:          %d/%d\n", c.top1Agree, c.windows)
	_, _ = fmt.Fprintf(out, "  detections >= %.2f identical:   %d/%d\n", opts.minScore, c.detectionSetsAgree, c.windows)
	_, _ = fmt.Fprintf(out, "  max |raw output diff|:          %.6f (mean %.6f)\n", c.maxAbsRawDiff, c.meanAbsRawDiff)
	_, _ = fmt.Fprintf(out, "  max |confidence diff|:          %.6f\n", c.maxAbsScoreDiff)

	_, _ = fmt.Fprintf(out, "\nTop %d per window (CPU | CUDA):\n", opts.topK)
	for w := range c.windows {
		_, _ = fmt.Fprintf(out, "  window %d\n", w)
		cpuTop := topClasses(cpu.outputs[w].scores, opts.topK)
		cudaTop := topClasses(cuda.outputs[w].scores, opts.topK)
		for i := range min(len(cpuTop), len(cudaTop)) {
			_, _ = fmt.Fprintf(out, "    %-40s %.4f | %-40s %.4f\n",
				className(labels, cpuTop[i].index), cpuTop[i].score,
				className(labels, cudaTop[i].index), cudaTop[i].score)
		}
	}
}

// className names a class index, falling back to "#<index>" without labels.
func className(labels []string, index int) string {
	if len(labels) > 1 && index < len(labels) {
		return labels[index]
	}
	return "#" + strconv.Itoa(index)
}

// fmtMs formats a duration in milliseconds with one decimal.
func fmtMs(d time.Duration) string {
	return fmt.Sprintf("%.1f ms", float64(d.Microseconds())/float64(time.Millisecond/time.Microsecond))
}
