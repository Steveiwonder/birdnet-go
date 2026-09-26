package onnx

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sampleProfile mimics an ONNX Runtime profile for a partially offloaded CUDA
// session run twice: fence events, a repeated run, and inserted memcpy nodes.
const sampleProfile = `[
 {"cat":"Session","name":"model_loading_uri","args":{}},
 {"cat":"Node","name":"conv1_fence_before","args":{"op_name":"Conv"}},
 {"cat":"Node","name":"conv1_kernel_time","args":{"op_name":"Conv","provider":"CUDAExecutionProvider"}},
 {"cat":"Node","name":"conv1_fence_after","args":{"op_name":"Conv"}},
 {"cat":"Node","name":"relu1_kernel_time","args":{"op_name":"Relu","provider":"CUDAExecutionProvider"}},
 {"cat":"Node","name":"shape1_kernel_time","args":{"op_name":"Shape","provider":"CPUExecutionProvider"}},
 {"cat":"Node","name":"dft1_kernel_time","args":{"op_name":"DFT","provider":"CPUExecutionProvider"}},
 {"cat":"Node","name":"memcpy_in_kernel_time","args":{"op_name":"MemcpyFromHost","provider":"CUDAExecutionProvider"}},
 {"cat":"Node","name":"conv1_kernel_time","args":{"op_name":"Conv","provider":"CUDAExecutionProvider"}}
]`

func TestParseProfilePlacement(t *testing.T) {
	t.Parallel()

	p, err := ParseProfilePlacement(strings.NewReader(sampleProfile))
	require.NoError(t, err)

	assert.Equal(t, 4, p.Total, "memcpy, fence and repeated events must not be counted")
	assert.Equal(t, 2, p.Count(ProviderCUDA))
	assert.Equal(t, 2, p.Count(ProviderCPU))
	assert.Equal(t, []string{"DFT", "Shape"}, p.Ops(ProviderCPU))
	assert.Equal(t, []string{"Conv", "Relu"}, p.Ops(ProviderCUDA))
}

func TestParseProfilePlacement_CPUOnly(t *testing.T) {
	t.Parallel()

	const cpuOnly = `[{"cat":"Node","name":"a_kernel_time","args":{"op_name":"Conv","provider":"CPUExecutionProvider"}}]`
	p, err := ParseProfilePlacement(strings.NewReader(cpuOnly))
	require.NoError(t, err)
	assert.Equal(t, 1, p.Total)
	assert.Zero(t, p.Count(ProviderCUDA))
}

func TestParseProfilePlacement_Invalid(t *testing.T) {
	t.Parallel()

	_, err := ParseProfilePlacement(strings.NewReader("not json"))
	require.Error(t, err)
}

func TestNodePlacement_NilSafe(t *testing.T) {
	t.Parallel()

	var p *NodePlacement
	assert.Zero(t, p.Count(ProviderCUDA))
	assert.Nil(t, p.Ops(ProviderCPU))
}
