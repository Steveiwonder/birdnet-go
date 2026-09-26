package system

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/hwprofile"
	"github.com/tphakala/birdnet-go/internal/inference"
)

func TestBuildCUDABackendStatus(t *testing.T) {
	t.Parallel()

	session := inference.CUDASessionInfo{Model: "BirdNET_v2.4_fp32_dfttrunc.onnx", CUDANodes: 90, TotalNodes: 96, CPUOps: []string{"Shape"}}
	tests := []struct {
		name string
		cfg  conf.BirdNETConfig
		cuda inference.CUDAStatus
		want CUDABackendStatus
	}{
		{
			name: "default cpu, cpu build",
			cfg:  conf.BirdNETConfig{},
			cuda: inference.CUDAStatus{},
			want: CUDABackendStatus{Provider: "cpu", Sessions: []inference.CUDASessionInfo{}},
		},
		{
			// Requested but failed: not active, error surfaced.
			name: "cuda requested, init failed",
			cfg:  conf.BirdNETConfig{ONNXProvider: "CUDA", CUDADeviceID: 1},
			cuda: inference.CUDAStatus{LastError: "CUDA execution provider unavailable", LastErrorAtUnix: 42},
			want: CUDABackendStatus{
				Provider: "cuda", Requested: true, DeviceID: 1,
				Sessions: []inference.CUDASessionInfo{}, LastError: "CUDA execution provider unavailable", LastErrorAtUnix: 42,
			},
		},
		{
			name: "cuda active",
			cfg:  conf.BirdNETConfig{ONNXProvider: "cuda"},
			cuda: inference.CUDAStatus{LibraryPresent: true, Sessions: []inference.CUDASessionInfo{session}},
			want: CUDABackendStatus{
				Provider: "cuda", Requested: true, LibraryPresent: true, Active: true,
				Sessions: []inference.CUDASessionInfo{session},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, buildCUDABackendStatus(&tt.cfg, &tt.cuda))
		})
	}
}

// TestCUDABackendStatus_JSON pins the wire shape the frontend reads.
func TestCUDABackendStatus_JSON(t *testing.T) {
	t.Parallel()

	status := buildCUDABackendStatus(&conf.BirdNETConfig{ONNXProvider: "cuda"}, &inference.CUDAStatus{
		LibraryPresent: true,
		Sessions:       []inference.CUDASessionInfo{{Model: "m.onnx", DeviceID: 0, CUDANodes: 3, TotalNodes: 4, CPUOps: []string{"Shape"}}},
	})
	raw, err := json.Marshal(BackendsInfo{CUDA: status})
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	cuda, ok := decoded["cuda"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "cuda", cuda["provider"])
	assert.Equal(t, true, cuda["active"])
	assert.Equal(t, true, cuda["libraryPresent"])
	sessions, ok := cuda["sessions"].([]any)
	require.True(t, ok)
	require.Len(t, sessions, 1)
	s0, ok := sessions[0].(map[string]any)
	require.True(t, ok)
	assert.InDelta(t, 3, s0["cudaNodes"], 0)
	assert.InDelta(t, 4, s0["totalNodes"], 0)
}

func TestApplyCUDAToAccelerators(t *testing.T) {
	t.Parallel()

	nvidia := func() []AcceleratorInfo {
		return []AcceleratorInfo{
			{Vendor: hwprofile.VendorNVIDIA, Reasons: []string{hwprofile.ReasonNoRuntime, hwprofile.ReasonRenderNodeUnavailable}},
			{Vendor: hwprofile.VendorIntel, Reasons: []string{hwprofile.ReasonRenderNodeUnavailable}},
		}
	}

	t.Run("cpu build keeps no-runtime", func(t *testing.T) {
		t.Parallel()
		accs := nvidia()
		applyCUDAToAccelerators(accs, &CUDABackendStatus{})
		assert.Equal(t, nvidia(), accs)
	})

	t.Run("gpu build drops no-runtime only", func(t *testing.T) {
		t.Parallel()
		accs := nvidia()
		applyCUDAToAccelerators(accs, &CUDABackendStatus{LibraryPresent: true})
		assert.Equal(t, []string{hwprofile.ReasonRenderNodeUnavailable}, accs[0].Reasons)
		assert.False(t, accs[0].Accessible, "a present library alone does not prove the GPU is reachable")
		assert.Equal(t, nvidia()[1], accs[1], "non-NVIDIA rows are untouched")
	})

	t.Run("active cuda session marks the card reachable", func(t *testing.T) {
		t.Parallel()
		accs := nvidia()
		applyCUDAToAccelerators(accs, &CUDABackendStatus{LibraryPresent: true, Active: true})
		assert.Nil(t, accs[0].Reasons)
		assert.True(t, accs[0].Accessible)
		assert.Equal(t, nvidia()[1], accs[1])
	})
}
