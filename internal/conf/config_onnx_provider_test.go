package conf

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// hasProviderError reports whether a validation result carries an error about
// the ONNX Runtime execution provider settings.
func hasProviderError(r ValidationResult) bool {
	for _, e := range r.Errors {
		if strings.Contains(e, "onnxprovider") || strings.Contains(e, "cudadeviceid") {
			return true
		}
	}
	return false
}

// TestValidateONNXProvider verifies the provider selection: CPU (and empty, the
// default for existing configs) and CUDA are accepted and canonicalized, while
// an unknown provider is an error rather than a warning, because a user who
// asked for a GPU must not be run on the CPU silently.
func TestValidateONNXProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		cfg      BirdNETConfig
		wantErr  bool
		wantNorm string
	}{
		{name: "empty defaults to cpu", cfg: BirdNETConfig{}, wantNorm: ""},
		{name: "cpu", cfg: BirdNETConfig{ONNXProvider: ONNXProviderCPU}, wantNorm: ONNXProviderCPU},
		{name: "cuda", cfg: BirdNETConfig{ONNXProvider: ONNXProviderCUDA}, wantNorm: ONNXProviderCUDA},
		{name: "uppercase cuda is canonicalized", cfg: BirdNETConfig{ONNXProvider: " CUDA "}, wantNorm: ONNXProviderCUDA},
		{name: "cuda with device 1", cfg: BirdNETConfig{ONNXProvider: ONNXProviderCUDA, CUDADeviceID: 1}, wantNorm: ONNXProviderCUDA},
		{name: "cuda with onnx backend", cfg: BirdNETConfig{ONNXProvider: ONNXProviderCUDA, Backend: BackendPrefONNX}, wantNorm: ONNXProviderCUDA},
		{name: "unknown provider", cfg: BirdNETConfig{ONNXProvider: "gpu"}, wantErr: true},
		{name: "negative device", cfg: BirdNETConfig{ONNXProvider: ONNXProviderCUDA, CUDADeviceID: -1}, wantErr: true},
		{name: "cuda with forced openvino", cfg: BirdNETConfig{ONNXProvider: ONNXProviderCUDA, Backend: BackendPrefOpenVINO}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			res := ValidateBirdNETSettings(&tt.cfg)
			assert.Equal(t, tt.wantErr, hasProviderError(res), "errors: %v", res.Errors)
			if tt.wantErr {
				assert.False(t, res.Valid)
				return
			}
			normalized, ok := res.Normalized.(*BirdNETConfig)
			require.True(t, ok)
			assert.Equal(t, tt.wantNorm, normalized.ONNXProvider)
		})
	}
}

func TestValidateEnvONNXProvider(t *testing.T) {
	t.Parallel()
	for _, v := range []string{"", "cpu", "cuda", "CUDA", " cuda "} {
		require.NoErrorf(t, validateEnvONNXProvider(v), "value %q", v)
	}
	for _, v := range []string{"gpu", "rocm", "tensorrt"} {
		assert.Errorf(t, validateEnvONNXProvider(v), "value %q", v)
	}
}

func TestValidateEnvCUDADeviceID(t *testing.T) {
	t.Parallel()
	for _, v := range []string{"0", "1", " 3 "} {
		require.NoErrorf(t, validateEnvCUDADeviceID(v), "value %q", v)
	}
	for _, v := range []string{"-1", "abc", ""} {
		assert.Errorf(t, validateEnvCUDADeviceID(v), "value %q", v)
	}
}
