package api

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tphakala/birdnet-go/internal/conf"
)

// TestBirdnetSettingsChanged_ONNXProvider verifies that switching the ONNX
// Runtime execution provider or CUDA device triggers a model reload, so the
// change takes effect without a restart.
func TestBirdnetSettingsChanged_ONNXProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*conf.Settings)
		want   bool
	}{
		{name: "unchanged", mutate: func(*conf.Settings) {}, want: false},
		{name: "cpu to cuda", mutate: func(s *conf.Settings) { s.BirdNET.ONNXProvider = conf.ONNXProviderCUDA }, want: true},
		{name: "cuda device", mutate: func(s *conf.Settings) { s.BirdNET.CUDADeviceID = 1 }, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			oldSettings := &conf.Settings{}
			newSettings := &conf.Settings{}
			tt.mutate(newSettings)
			assert.Equal(t, tt.want, birdnetSettingsChanged(oldSettings, newSettings))
		})
	}
}
