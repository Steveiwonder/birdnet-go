package onnx

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tphakala/birdnet-go/internal/errors"
	ort "github.com/yalue/onnxruntime_go"
)

// ONNX Runtime execution provider names as they appear in the session profile
// ("args.provider" on each node event).
const (
	ProviderCPU  = "CPUExecutionProvider"
	ProviderCUDA = "CUDAExecutionProvider"
)

const (
	// profileCategoryNode is the "cat" value ONNX Runtime writes for per-node
	// profile events.
	profileCategoryNode = "Node"
	// profileKernelSuffix marks the kernel-execution event of a node. Each node
	// also gets fence events; only the kernel event carries the real placement.
	profileKernelSuffix = "_kernel_time"
	// profileMemcpyPrefix marks the host<->device copy nodes ONNX Runtime inserts
	// around a device partition. They are plumbing, not model operators, so they
	// are left out of the placement counts.
	profileMemcpyPrefix = "Memcpy"
	// profileFilePrefix names the temporary profile file the probe session writes.
	profileFilePrefix = "placement"
	// profileTempDirPattern names the temporary directory holding the profile.
	profileTempDirPattern = "birdnet-ort-profile-"
)

// ErrNoProfile is returned when a placement probe ran but ONNX Runtime wrote no
// profile file to read the node placement from.
var ErrNoProfile = errors.NewStd("onnx: placement probe produced no profile")

// NodePlacement records which execution provider ONNX Runtime ran each model
// operator on during one real inference. It is measured from the session
// profile, not predicted, so it is the evidence that a model really executes on
// a device rather than merely having had the provider registered.
type NodePlacement struct {
	// Total is the number of model operators executed (memcpy nodes excluded).
	Total int
	// ByProvider counts executed operators per execution provider name.
	ByProvider map[string]int
	// OpsByProvider lists the distinct operator types per provider, sorted,
	// so a partial offload can name the operators left on the CPU.
	OpsByProvider map[string][]string
}

// Count returns how many operators ran on the named execution provider.
func (p *NodePlacement) Count(provider string) int {
	if p == nil {
		return 0
	}
	return p.ByProvider[provider]
}

// Ops returns the sorted distinct operator types that ran on provider.
func (p *NodePlacement) Ops(provider string) []string {
	if p == nil {
		return nil
	}
	return slices.Clone(p.OpsByProvider[provider])
}

// profileEvent is the subset of an ONNX Runtime profile event the parser reads.
type profileEvent struct {
	Cat  string `json:"cat"`
	Name string `json:"name"`
	Args struct {
		OpName   string `json:"op_name"`
		Provider string `json:"provider"`
	} `json:"args"`
}

// ParseProfilePlacement reads an ONNX Runtime profile (the JSON array written
// when profiling is enabled on a session) and returns the per-provider operator
// placement. Each node is counted once even when the profile holds several runs.
func ParseProfilePlacement(r io.Reader) (*NodePlacement, error) {
	var events []profileEvent
	if err := json.NewDecoder(r).Decode(&events); err != nil {
		return nil, fmt.Errorf("onnx: failed to parse ONNX Runtime profile: %w", err)
	}

	placement := &NodePlacement{
		ByProvider:    make(map[string]int),
		OpsByProvider: make(map[string][]string),
	}
	seen := make(map[string]struct{}, len(events))
	for i := range events {
		ev := &events[i]
		if ev.Cat != profileCategoryNode || !strings.HasSuffix(ev.Name, profileKernelSuffix) {
			continue
		}
		if ev.Args.Provider == "" || strings.HasPrefix(ev.Args.OpName, profileMemcpyPrefix) {
			continue
		}
		if _, dup := seen[ev.Name]; dup {
			continue
		}
		seen[ev.Name] = struct{}{}
		placement.Total++
		placement.ByProvider[ev.Args.Provider]++
		if ops := placement.OpsByProvider[ev.Args.Provider]; !slices.Contains(ops, ev.Args.OpName) {
			placement.OpsByProvider[ev.Args.Provider] = append(ops, ev.Args.OpName)
		}
	}
	for provider := range placement.OpsByProvider {
		slices.Sort(placement.OpsByProvider[provider])
	}
	return placement, nil
}

// probeNodePlacement builds a throwaway session with the caller's session
// options plus profiling, runs one inference on silence, and reads back which
// execution provider each operator ran on. The serving session is built
// separately without profiling, so the probe adds only load-time cost.
func probeNodePlacement(modelPath string, inputNames, outputNames []string, modelCfg *ModelConfig, sessionOptsFn sessionConfigurer) (*NodePlacement, error) {
	dir, err := os.MkdirTemp("", profileTempDirPattern)
	if err != nil {
		return nil, fmt.Errorf("onnx: failed to create profile directory: %w", err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	prefix := filepath.Join(dir, profileFilePrefix)
	session, err := createSession(modelPath, inputNames, outputNames, func(so *ort.SessionOptions) error {
		if sessionOptsFn != nil {
			if err := sessionOptsFn(so); err != nil {
				return err
			}
		}
		if err := so.EnableProfiling(prefix); err != nil {
			return fmt.Errorf("onnx: failed to enable profiling: %w", err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	probe := &Classifier{session: session, config: *modelCfg}

	_, _, runErr := probe.PredictRawWithEmbeddings(make([]float32, modelCfg.SampleCount))
	// Destroying the session flushes the profile to disk.
	closeErr := probe.Close()
	if runErr != nil {
		return nil, fmt.Errorf("onnx: placement probe inference failed: %w", runErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("onnx: failed to close placement probe session: %w", closeErr)
	}

	matches, err := filepath.Glob(prefix + "*.json")
	if err != nil {
		return nil, fmt.Errorf("onnx: failed to locate profile: %w", err)
	}
	if len(matches) == 0 {
		return nil, ErrNoProfile
	}
	f, err := os.Open(matches[0])
	if err != nil {
		return nil, fmt.Errorf("onnx: failed to open profile: %w", err)
	}
	defer func() { _ = f.Close() }()
	return ParseProfilePlacement(f)
}
