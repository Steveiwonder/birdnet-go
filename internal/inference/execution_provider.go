package inference

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tphakala/birdnet-go/internal/errors"
	ort "github.com/tphakala/birdnet-go/internal/inference/onnx"
	ortlib "github.com/yalue/onnxruntime_go"
)

// ONNX Runtime execution providers selectable for species classifiers. CPU is
// the default and matches every existing installation; CUDA runs the model on
// an NVIDIA GPU and needs the GPU build of ONNX Runtime.
const (
	ExecutionProviderCPU  = "cpu"
	ExecutionProviderCUDA = "cuda"
)

// DeviceCUDA is the device label reported for a model running on the CUDA
// execution provider. CUDADeviceLabel appends the device ordinal.
const DeviceCUDA = "CUDA"

// cudaDeviceIDKey is the ONNX Runtime CUDA provider option selecting the GPU.
const cudaDeviceIDKey = "device_id"

// cudaProviderLibrary is the file name of the CUDA execution provider library
// shipped next to libonnxruntime in the GPU build of ONNX Runtime.
func cudaProviderLibrary() string {
	if runtime.GOOS == "windows" {
		return "onnxruntime_providers_cuda.dll"
	}
	return "libonnxruntime_providers_cuda.so"
}

// Sentinel errors for execution provider selection. Callers match them with
// errors.Is; the wrapped message carries the ONNX Runtime detail and a hint.
var (
	// ErrUnknownExecutionProvider is returned for a provider name that is
	// neither "cpu" nor "cuda".
	ErrUnknownExecutionProvider = errors.NewStd("unknown ONNX Runtime execution provider")
	// ErrCUDAUnavailable is returned when CUDA was selected but the CUDA
	// execution provider could not be attached or initialized. There is no
	// CPU fallback: the model is not loaded.
	ErrCUDAUnavailable = errors.NewStd("CUDA execution provider unavailable")
	// ErrCUDANotUsed is returned when the CUDA provider attached but the load
	// probe measured no model operator running on it.
	ErrCUDANotUsed = errors.NewStd("model did not run on the CUDA execution provider")
)

// ExecutionProviderOptions selects the ONNX Runtime execution provider for a
// classifier session.
type ExecutionProviderOptions struct {
	// Provider is ExecutionProviderCPU or ExecutionProviderCUDA. Empty means CPU.
	Provider string
	// DeviceID is the CUDA device ordinal; ignored for CPU.
	DeviceID int
}

// UsesCUDA reports whether the options select the CUDA execution provider.
func (o ExecutionProviderOptions) UsesCUDA() bool {
	return o.Provider == ExecutionProviderCUDA
}

// NormalizeExecutionProvider canonicalizes a configured provider name: case
// and surrounding space are ignored and empty means CPU. Any other value is
// rejected with ErrUnknownExecutionProvider.
func NormalizeExecutionProvider(name string) (string, error) {
	switch p := strings.ToLower(strings.TrimSpace(name)); p {
	case "", ExecutionProviderCPU:
		return ExecutionProviderCPU, nil
	case ExecutionProviderCUDA:
		return ExecutionProviderCUDA, nil
	default:
		return "", fmt.Errorf("%w %q: valid values are %q and %q",
			ErrUnknownExecutionProvider, name, ExecutionProviderCPU, ExecutionProviderCUDA)
	}
}

// CUDADeviceLabel returns the device label for a CUDA device ordinal, e.g. "CUDA:0".
func CUDADeviceLabel(deviceID int) string {
	return DeviceCUDA + ":" + strconv.Itoa(deviceID)
}

// appendCUDAProvider registers the CUDA execution provider on the session
// options. It is a variable so tests can simulate provider failures without a
// GPU build of ONNX Runtime.
var appendCUDAProvider = func(so *ortlib.SessionOptions, deviceID int) error {
	cudaOpts, err := ortlib.NewCUDAProviderOptions()
	if err != nil {
		return err
	}
	defer func() { _ = cudaOpts.Destroy() }()
	if err := cudaOpts.Update(map[string]string{cudaDeviceIDKey: strconv.Itoa(deviceID)}); err != nil {
		return err
	}
	return so.AppendExecutionProviderCUDA(cudaOpts)
}

// Troubleshooting hints attached to CUDA initialization errors.
const (
	cudaHintBuild = "the ONNX Runtime library in use has no CUDA execution provider: " +
		"use the BirdNET-Go CUDA container image (a -cuda tag) or install the ONNX Runtime GPU package " +
		"(onnxruntime-linux-x64-gpu " + RequiredORTAPIMajor + ".x) and point onnxruntimepath at it"
	cudaHintLibraries = "CUDA 12 and cuDNN 9 runtime libraries could not be loaded: " +
		"use the BirdNET-Go CUDA container image, or install the CUDA 12 runtime, cuBLAS, cuFFT, cuRAND and cuDNN 9 " +
		"and make them visible to the dynamic loader (ldconfig or LD_LIBRARY_PATH)"
	cudaHintDevice = "no usable NVIDIA GPU is visible to this process: check nvidia-smi on the host, " +
		"start the container with --gpus all (requires the NVIDIA Container Toolkit), " +
		"make sure the driver supports CUDA 12, and that cudadeviceid is below the number of visible GPUs"
	cudaHintGeneric = "check the NVIDIA driver (nvidia-smi), the container GPU access (--gpus all), " +
		"and the CUDA 12 / cuDNN 9 libraries; set birdnet.onnxprovider to cpu to run without a GPU"
)

// cudaHint picks a troubleshooting hint from the ONNX Runtime error text. The
// order matters: a failure to load the CUDA provider library names that library
// even when the real cause is a CUDA dependency it links, so the dependency
// names are checked before the provider library name.
func cudaHint(msg string) string {
	m := strings.ToLower(msg)
	switch {
	case strings.Contains(m, "in this build"):
		return cudaHintBuild
	case strings.Contains(m, "libcudnn"),
		strings.Contains(m, "libcublas"),
		strings.Contains(m, "libcudart"),
		strings.Contains(m, "libcufft"),
		strings.Contains(m, "libcurand"),
		strings.Contains(m, "libnvrtc"),
		strings.Contains(m, "libnvjitlink"):
		return cudaHintLibraries
	case strings.Contains(m, "providers_cuda"),
		strings.Contains(m, "providers_shared"):
		return cudaHintBuild
	case strings.Contains(m, "cannot open shared object"):
		return cudaHintLibraries
	case strings.Contains(m, "no cuda-capable device"),
		strings.Contains(m, "cudaerrornodevice"),
		strings.Contains(m, "driver version is insufficient"),
		strings.Contains(m, "cudaerrorinsufficientdriver"),
		strings.Contains(m, "invalid device ordinal"),
		strings.Contains(m, "cudaerrorinvaliddevice"):
		return cudaHintDevice
	default:
		return cudaHintGeneric
	}
}

// cudaInitError wraps a CUDA failure at the given stage with ErrCUDAUnavailable
// and a troubleshooting hint, and records it for the status API.
func cudaInitError(stage string, deviceID int, err error) error {
	wrapped := fmt.Errorf("%w (device %d) during %s: %w; hint: %s",
		ErrCUDAUnavailable, deviceID, stage, err, cudaHint(err.Error()))
	cudaRegistry.recordError(wrapped)
	return wrapped
}

// verifyCUDAPlacement checks the measured operator placement of a CUDA session.
// At least one model operator must have run on the CUDA provider; operators the
// CUDA provider cannot run stay on the CPU (normal ONNX Runtime partitioning)
// and are reported, not rejected.
func verifyCUDAPlacement(modelPath string, deviceID int, placement *ort.NodePlacement) error {
	if placement == nil || placement.Total == 0 {
		err := fmt.Errorf("%w: %s: the load probe recorded no executed operators",
			ErrCUDANotUsed, filepath.Base(modelPath))
		cudaRegistry.recordError(err)
		return err
	}
	if placement.Count(ort.ProviderCUDA) == 0 {
		err := fmt.Errorf("%w: %s: all %d operators ran on %s (device %d); "+
			"the model's operators are not supported by the CUDA provider, use a FP32 ONNX model or set onnxprovider to cpu",
			ErrCUDANotUsed, filepath.Base(modelPath), placement.Total, ort.ProviderCPU, deviceID)
		cudaRegistry.recordError(err)
		return err
	}
	return nil
}

// CUDASessionInfo describes one live classifier session on the CUDA provider,
// with the operator placement measured when it was loaded.
type CUDASessionInfo struct {
	// Model is the model file name (base name only, never the full path).
	Model string `json:"model"`
	// DeviceID is the CUDA device ordinal the session runs on.
	DeviceID int `json:"deviceId"`
	// CUDANodes is the number of model operators that ran on the CUDA provider.
	CUDANodes int `json:"cudaNodes"`
	// TotalNodes is the number of model operators executed in the probe.
	TotalNodes int `json:"totalNodes"`
	// CPUOps lists operator types ONNX Runtime left on the CPU provider.
	CPUOps []string `json:"cpuOps,omitempty"`
}

// CUDAStatus is the process-wide CUDA execution provider status for the
// inference-status API.
type CUDAStatus struct {
	// LibraryPresent reports whether the CUDA provider library sits next to the
	// ONNX Runtime library, i.e. whether the installed runtime is a GPU build.
	// It says nothing about whether a GPU is usable; Sessions does.
	LibraryPresent bool `json:"libraryPresent"`
	// Sessions lists the classifier sessions currently running on CUDA.
	Sessions []CUDASessionInfo `json:"sessions"`
	// LastError is the most recent CUDA initialization failure, if any.
	LastError string `json:"lastError,omitempty"`
	// LastErrorAtUnix is when LastError happened (Unix seconds).
	LastErrorAtUnix int64 `json:"lastErrorAtUnix,omitempty"`
}

// cudaSessionRegistry tracks live CUDA sessions and the last CUDA failure.
type cudaSessionRegistry struct {
	mu        sync.Mutex
	nextID    uint64
	sessions  map[uint64]CUDASessionInfo
	lastErr   string
	lastErrAt time.Time
}

//nolint:gochecknoglobals // process-wide registry of native CUDA sessions
var cudaRegistry = &cudaSessionRegistry{sessions: make(map[uint64]CUDASessionInfo)}

func (r *cudaSessionRegistry) add(info CUDASessionInfo) uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nextID++
	r.sessions[r.nextID] = info
	// A successful CUDA load supersedes an earlier failure.
	r.lastErr, r.lastErrAt = "", time.Time{}
	return r.nextID
}

func (r *cudaSessionRegistry) remove(id uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.sessions, id)
}

func (r *cudaSessionRegistry) recordError(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastErr, r.lastErrAt = err.Error(), time.Now()
}

func (r *cudaSessionRegistry) snapshot() (sessions []CUDASessionInfo, lastErr string, lastErrAt time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := slices.Sorted(maps.Keys(r.sessions))
	sessions = make([]CUDASessionInfo, 0, len(ids))
	for _, id := range ids {
		s := r.sessions[id]
		s.CPUOps = slices.Clone(s.CPUOps)
		sessions = append(sessions, s)
	}
	return sessions, r.lastErr, r.lastErrAt
}

// GetCUDAStatus returns the CUDA execution provider status. configuredPath is
// the configured ONNX Runtime library path (may be empty for the default search).
func GetCUDAStatus(configuredPath string) CUDAStatus {
	sessions, lastErr, lastErrAt := cudaRegistry.snapshot()
	status := CUDAStatus{
		LibraryPresent: cudaProviderLibraryPresent(configuredPath),
		Sessions:       sessions,
		LastError:      lastErr,
	}
	if !lastErrAt.IsZero() {
		status.LastErrorAtUnix = lastErrAt.Unix()
	}
	return status
}

// cudaProviderLibraryPresent reports whether the CUDA provider library exists
// in the same directory as the resolved ONNX Runtime library.
func cudaProviderLibraryPresent(configuredPath string) bool {
	libPath := resolvedORTPath(configuredPath)
	if !libraryFileExists(libPath) {
		return false
	}
	info, err := os.Stat(filepath.Join(filepath.Dir(libPath), cudaProviderLibrary()))
	return err == nil && !info.IsDir()
}
