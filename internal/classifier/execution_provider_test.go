package classifier

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/errors"
	"github.com/tphakala/birdnet-go/internal/inference"
)

// TestONNXProviderConstantsDrift guards the literals conf keeps to avoid an
// inference dependency.
func TestONNXProviderConstantsDrift(t *testing.T) {
	t.Parallel()
	assert.Equal(t, inference.ExecutionProviderCPU, conf.ONNXProviderCPU)
	assert.Equal(t, inference.ExecutionProviderCUDA, conf.ONNXProviderCUDA)
}

func TestExecutionProviderFor(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		cfg        conf.BirdNETConfig
		want       inference.ExecutionProviderOptions
		wantCUDA   bool
		wantDevice string
	}{
		{
			name:       "default is cpu",
			cfg:        conf.BirdNETConfig{},
			want:       inference.ExecutionProviderOptions{Provider: inference.ExecutionProviderCPU},
			wantDevice: deviceCPU,
		},
		{
			name:       "cuda device 0",
			cfg:        conf.BirdNETConfig{ONNXProvider: "cuda"},
			want:       inference.ExecutionProviderOptions{Provider: inference.ExecutionProviderCUDA},
			wantCUDA:   true,
			wantDevice: "CUDA:0",
		},
		{
			name:       "uppercase cuda on device 1",
			cfg:        conf.BirdNETConfig{ONNXProvider: "CUDA", CUDADeviceID: 1},
			want:       inference.ExecutionProviderOptions{Provider: inference.ExecutionProviderCUDA, DeviceID: 1},
			wantCUDA:   true,
			wantDevice: "CUDA:1",
		},
		{
			// An unknown value is passed through so the classifier rejects it
			// instead of it quietly becoming CPU.
			name:       "unknown passes through",
			cfg:        conf.BirdNETConfig{ONNXProvider: "gpu"},
			want:       inference.ExecutionProviderOptions{Provider: "gpu"},
			wantDevice: deviceCPU,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := executionProviderFor(&tt.cfg)
			assert.Equal(t, tt.want, got)
			assert.Equal(t, tt.wantCUDA, got.UsesCUDA())
			assert.Equal(t, tt.wantDevice, onnxDeviceLabel(got))
		})
	}
}

// TestBirdNETOpenVINOPlan_CUDADeclines verifies an explicit CUDA selection
// keeps the primary model off OpenVINO even when the OpenVINO plan would accept
// it (e.g. an Intel iGPU host with backend "auto"). Not parallel: stubs package
// state.
func TestBirdNETOpenVINOPlan_CUDADeclines(t *testing.T) {
	stubBirdNETV24BasePlan(t, openVINOPlan{device: inference.OVDeviceGPU})

	cfg := conf.BirdNETConfig{ONNXProvider: conf.ONNXProviderCUDA}
	_, ok, reason := birdnetV24OpenVINOPlan(&cfg, QuantizationFP32)
	assert.False(t, ok)
	assert.Equal(t, ovReasonCUDASelected, reason)

	cfg.ONNXProvider = conf.ONNXProviderCPU
	_, ok, _ = birdnetV24OpenVINOPlan(&cfg, QuantizationFP32)
	assert.True(t, ok, "CPU provider must leave the OpenVINO plan untouched")
}

// TestSecondaryOpenVINOAttempts_CUDADeclines verifies the secondary models skip
// OpenVINO when CUDA is selected, so their ORT path runs on CUDA.
func TestSecondaryOpenVINOAttempts_CUDADeclines(t *testing.T) {
	t.Parallel()
	ep := inference.ExecutionProviderOptions{Provider: inference.ExecutionProviderCUDA}

	_, _, _, ok := tryBirdNETV3OpenVINO(&BirdNETV3Config{Backend: conf.BackendPrefAuto, ExecutionProvider: ep}, []string{"a"})
	assert.False(t, ok)
	_, _, _, ok = tryPerchOpenVINO(&PerchConfig{Backend: conf.BackendPrefAuto, ExecutionProvider: ep, ModelPath: "perch_v2_no_dft.onnx"}, []string{"a"})
	assert.False(t, ok)
	_, _, _, ok = tryBatOpenVINO(&BatModelConfig{Backend: conf.BackendPrefAuto, ExecutionProvider: ep}, 1024)
	assert.False(t, ok)
}

// TestInitializeModel_CUDARejectsTFLite verifies that selecting CUDA with a
// TFLite primary model fails with an actionable error instead of running the
// TFLite model on the CPU.
func TestInitializeModel_CUDARejectsTFLite(t *testing.T) {
	t.Parallel()

	bn := birdNETWithModelPath("/models/model.tflite", &ModelInfo{ID: DefaultModelVersion, Backend: BackendTFLite})
	bn.Settings.BirdNET.ONNXProvider = conf.ONNXProviderCUDA

	err := bn.initializeModel()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "TFLite")
	assert.Contains(t, err.Error(), "onnxprovider")
	enhanced, ok := errors.AsType[*errors.EnhancedError](err)
	require.True(t, ok)
	assert.Equal(t, errors.CategoryModelInit, enhanced.Category)
}

// TestSecondaryTripletFor_ProviderChange verifies a provider or CUDA device
// change alters the secondary rebuild key, so ReloadSecondaryModels moves the
// secondary models to the new provider without a restart.
func TestSecondaryTripletFor_ProviderChange(t *testing.T) {
	t.Parallel()

	base := &conf.Settings{}
	cuda := &conf.Settings{}
	cuda.BirdNET.ONNXProvider = conf.ONNXProviderCUDA
	cuda1 := &conf.Settings{}
	cuda1.BirdNET.ONNXProvider = conf.ONNXProviderCUDA
	cuda1.BirdNET.CUDADeviceID = 1

	assert.NotEqual(t, secondaryTripletFor(base), secondaryTripletFor(cuda))
	assert.NotEqual(t, secondaryTripletFor(cuda), secondaryTripletFor(cuda1))

	sameCUDA := &conf.Settings{}
	sameCUDA.BirdNET.ONNXProvider = conf.ONNXProviderCUDA
	assert.Equal(t, secondaryTripletFor(cuda), secondaryTripletFor(sameCUDA), "an unchanged provider must not rebuild")
}
