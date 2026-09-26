package inference

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/birdnet-go/internal/errors"
	ort "github.com/tphakala/birdnet-go/internal/inference/onnx"
	ortlib "github.com/yalue/onnxruntime_go"
)

// envTestORTLib points the real-runtime tests at a libonnxruntime; when unset
// the default search paths are used, and the tests skip if none is found.
const envTestORTLib = "BIRDNET_TEST_ORT_LIB"

// testClassifierModel is a stock ONNX classifier shipped in the repository.
const testClassifierModel = "../classifier/data/BirdNET_INT8_ARM.onnx"

func TestNormalizeExecutionProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{in: "", want: ExecutionProviderCPU},
		{in: "cpu", want: ExecutionProviderCPU},
		{in: " CPU ", want: ExecutionProviderCPU},
		{in: "cuda", want: ExecutionProviderCUDA},
		{in: "CUDA", want: ExecutionProviderCUDA},
		{in: "gpu", wantErr: true},
		{in: "tensorrt", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			t.Parallel()
			got, err := NormalizeExecutionProvider(tt.in)
			if tt.wantErr {
				require.ErrorIs(t, err, ErrUnknownExecutionProvider)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCUDADeviceLabel(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "CUDA:0", CUDADeviceLabel(0))
	assert.Equal(t, "CUDA:2", CUDADeviceLabel(2))
}

func TestCUDAHint(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  string
		want string
	}{
		{"cpu build", "CUDA execution provider is not enabled in this build.", cudaHintBuild},
		{"cpu build legacy", "CUDA execution provider is not supported in this build.", cudaHintBuild},
		{"provider lib missing", "Failed to load library /usr/lib/libonnxruntime_providers_cuda.so with error: libonnxruntime_providers_cuda.so: cannot open shared object file", cudaHintBuild},
		// Real ONNX Runtime 1.25.1 message with the GPU build but no CUDA libraries.
		{"provider dependency missing", "Failed to load library /usr/lib/libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file: No such file or directory", cudaHintLibraries},
		{"cudnn missing", "libcudnn.so.9: cannot open shared object file: No such file or directory", cudaHintLibraries},
		{"no device", "CUDA failure 100: no CUDA-capable device is detected", cudaHintDevice},
		{"old driver", "CUDA driver version is insufficient for CUDA runtime version", cudaHintDevice},
		{"bad ordinal", "CUDA failure 101: invalid device ordinal", cudaHintDevice},
		{"other", "something unexpected", cudaHintGeneric},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, cudaHint(tt.msg))
		})
	}
}

// resetCUDARegistry isolates tests that touch the process-wide CUDA registry.
func resetCUDARegistry(t *testing.T) {
	t.Helper()
	saved := cudaRegistry
	cudaRegistry = &cudaSessionRegistry{sessions: make(map[uint64]CUDASessionInfo)}
	t.Cleanup(func() { cudaRegistry = saved })
}

func TestCUDARegistry_Lifecycle(t *testing.T) {
	resetCUDARegistry(t)

	err := cudaInitError("provider registration", 0, errors.NewStd("boom"))
	require.ErrorIs(t, err, ErrCUDAUnavailable)
	sessions, lastErr, lastAt := cudaRegistry.snapshot()
	assert.Empty(t, sessions)
	assert.Contains(t, lastErr, "boom")
	assert.False(t, lastAt.IsZero())

	id := cudaRegistry.add(CUDASessionInfo{Model: "m.onnx", CUDANodes: 3, TotalNodes: 4, CPUOps: []string{"Shape"}})
	sessions, lastErr, _ = cudaRegistry.snapshot()
	require.Len(t, sessions, 1)
	assert.Equal(t, "m.onnx", sessions[0].Model)
	assert.Contains(t, lastErr, "boom", "a later CUDA success must not hide another model's failure")

	cudaRegistry.remove(id)
	sessions, _, _ = cudaRegistry.snapshot()
	assert.Empty(t, sessions)
}

func TestGetCUDAStatus_LibraryPresence(t *testing.T) {
	resetCUDARegistry(t)

	dir := t.TempDir()
	lib := filepath.Join(dir, "libonnxruntime.so.1.25.1")
	require.NoError(t, os.WriteFile(lib, nil, 0o600))
	assert.False(t, GetCUDAStatus(lib).LibraryPresent, "CPU build has no CUDA provider library")

	require.NoError(t, os.WriteFile(filepath.Join(dir, cudaProviderLibrary()), nil, 0o600))
	status := GetCUDAStatus(lib)
	assert.True(t, status.LibraryPresent)
	assert.NotNil(t, status.Sessions, "sessions is always a list for the API")
}

func TestVerifyCUDAPlacement(t *testing.T) {
	resetCUDARegistry(t)

	require.ErrorIs(t, verifyCUDAPlacement("m.onnx", 0, nil), ErrCUDANotUsed)

	cpuOnly := &ort.NodePlacement{Total: 5, ByProvider: map[string]int{"CPUExecutionProvider": 5}}
	err := verifyCUDAPlacement("/models/m.onnx", 0, cpuOnly)
	require.ErrorIs(t, err, ErrCUDANotUsed)
	assert.Contains(t, err.Error(), "m.onnx")
	assert.NotContains(t, err.Error(), "/models/", "only the base name is reported")

	partial := &ort.NodePlacement{Total: 5, ByProvider: map[string]int{"CUDAExecutionProvider": 4, "CPUExecutionProvider": 1}}
	require.NoError(t, verifyCUDAPlacement("m.onnx", 0, partial))
}

// requireORT initializes the real ONNX Runtime or skips the test.
func requireORT(t *testing.T) {
	t.Helper()
	path := os.Getenv(envTestORTLib)
	if !libraryFileExists(resolvedORTPath(path)) {
		t.Skipf("ONNX Runtime library not found (set %s)", envTestORTLib)
	}
	if _, err := os.Stat(testClassifierModel); err != nil {
		t.Skipf("test model not found: %v", err)
	}
	require.NoError(t, InitONNXRuntime(path))
}

// stubCUDAProvider replaces the CUDA provider registration for one test.
func stubCUDAProvider(t *testing.T, fn func(*ortlib.SessionOptions, int) error) {
	t.Helper()
	saved := appendCUDAProvider
	appendCUDAProvider = fn
	t.Cleanup(func() { appendCUDAProvider = saved })
}

// newTestClassifier builds a classifier on the stock model; the label list is
// not validated because the tests only care about session construction.
func newTestClassifier(ep ExecutionProviderOptions) (Classifier, error) {
	return NewONNXClassifier(testClassifierModel, ONNXClassifierOptions{
		Labels:              []string{"placeholder"},
		Threads:             1,
		SkipLabelValidation: true,
		ExecutionProvider:   ep,
	})
}

func TestNewONNXClassifier_CPUDefault(t *testing.T) {
	requireORT(t)
	resetCUDARegistry(t)

	c, err := newTestClassifier(ExecutionProviderOptions{})
	require.NoError(t, err)
	defer c.Close()

	oc, ok := c.(*onnxClassifier)
	require.True(t, ok)
	assert.Zero(t, oc.cudaSession)
	assert.Nil(t, oc.Placement(), "CPU sessions are not probed")
	assert.Empty(t, GetCUDAStatus("").Sessions)
}

func TestNewONNXClassifier_UnknownProvider(t *testing.T) {
	requireORT(t)

	_, err := newTestClassifier(ExecutionProviderOptions{Provider: "rocm"})
	require.ErrorIs(t, err, ErrUnknownExecutionProvider)
}

// With a CPU build of ONNX Runtime the real CUDA registration fails; the
// classifier must not be created and nothing may fall back to CPU.
func TestNewONNXClassifier_CUDAFailsOnCPUBuild(t *testing.T) {
	requireORT(t)
	resetCUDARegistry(t)
	if GetCUDAStatus(os.Getenv(envTestORTLib)).LibraryPresent {
		t.Skip("GPU build of ONNX Runtime in use; this test needs the CPU build")
	}

	c, err := newTestClassifier(ExecutionProviderOptions{Provider: ExecutionProviderCUDA})
	require.ErrorIs(t, err, ErrCUDAUnavailable)
	assert.Nil(t, c)
	assert.Contains(t, err.Error(), cudaHintBuild)

	status := GetCUDAStatus("")
	assert.Empty(t, status.Sessions)
	assert.NotEmpty(t, status.LastError)
}

func TestNewONNXClassifier_CUDADeviceError(t *testing.T) {
	requireORT(t)
	resetCUDARegistry(t)
	stubCUDAProvider(t, func(*ortlib.SessionOptions, int) error {
		return errors.NewStd("CUDA failure 100: no CUDA-capable device is detected")
	})

	c, err := newTestClassifier(ExecutionProviderOptions{Provider: ExecutionProviderCUDA, DeviceID: 1})
	require.ErrorIs(t, err, ErrCUDAUnavailable)
	assert.Nil(t, c)
	assert.Contains(t, err.Error(), cudaHintDevice)
	assert.Contains(t, err.Error(), "device 1")
}

// A provider that registers without error but places no operator on CUDA is
// exactly the silent-CPU case: the placement probe must reject it.
func TestNewONNXClassifier_CUDARegisteredButUnused(t *testing.T) {
	requireORT(t)
	resetCUDARegistry(t)
	stubCUDAProvider(t, func(*ortlib.SessionOptions, int) error { return nil })

	c, err := newTestClassifier(ExecutionProviderOptions{Provider: ExecutionProviderCUDA})
	require.ErrorIs(t, err, ErrCUDANotUsed)
	assert.Nil(t, c)
	assert.Empty(t, GetCUDAStatus("").Sessions)
}
