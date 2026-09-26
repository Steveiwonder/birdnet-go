package inference

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	ort "github.com/tphakala/birdnet-go/internal/inference/onnx"
	"github.com/tphakala/birdnet-go/internal/logger"
	ortlib "github.com/yalue/onnxruntime_go"
)

var (
	ortInitMu      sync.Mutex
	ortInitialized bool
)

// IsORTInitialized reports whether the ONNX Runtime has been successfully initialized. Thread-safe.
func IsORTInitialized() bool {
	ortInitMu.Lock()
	defer ortInitMu.Unlock()
	return ortInitialized
}

// ONNXClassifierOptions configures the ONNX species classifier.
type ONNXClassifierOptions struct {
	// Labels is the species label list. Required.
	Labels []string
	// Threads is the number of CPU threads for ONNX inference. 0 = use ONNX defaults.
	Threads int
	// SkipLabelValidation disables the label-count-vs-model-output check.
	// Use when the model is loaded only for embedding extraction and the
	// caller's label list may not match the model's logits dimension.
	SkipLabelValidation bool
	// ExecutionProvider selects CPU (default) or CUDA. When CUDA is selected
	// and cannot be used, construction fails; it never falls back to CPU.
	ExecutionProvider ExecutionProviderOptions
}

// onnxClassifier implements Classifier using an ONNX Runtime session.
type onnxClassifier struct {
	classifier *ort.Classifier
	numSpecies int
	// cudaSession is the CUDA registry id, 0 for a CPU session.
	cudaSession uint64
}

// NewONNXClassifier creates a Classifier backed by an ONNX Runtime model.
// The ONNX Runtime must be initialized via InitONNXRuntime before calling this.
func NewONNXClassifier(modelPath string, opts ONNXClassifierOptions) (Classifier, error) {
	if len(opts.Labels) == 0 {
		return nil, fmt.Errorf("ONNX classifier requires labels")
	}

	classifierOpts := []ort.ClassifierOption{
		ort.WithLabels(opts.Labels),
		ort.WithTopK(0),          // We handle topK in BirdNET-Go's post-processing
		ort.WithMinConfidence(0), // No filtering, return all raw scores
	}
	if opts.SkipLabelValidation {
		classifierOpts = append(classifierOpts, ort.WithSkipLabelValidation())
	}
	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	ep := opts.ExecutionProvider
	provider, err := NormalizeExecutionProvider(ep.Provider)
	if err != nil {
		return nil, err
	}
	ep.Provider = provider

	var configErr error
	// providerErr records a failure to register the CUDA provider. It aborts
	// session creation, so a CUDA request never yields a CPU session.
	var providerErr error
	classifierOpts = append(classifierOpts, ort.WithSessionConfigurer(func(so *ortlib.SessionOptions) error {
		if err := so.SetIntraOpNumThreads(threads); err != nil && configErr == nil {
			configErr = fmt.Errorf("failed to set IntraOpNumThreads to %d: %w", threads, err)
		}
		if err := so.SetInterOpNumThreads(threads); err != nil && configErr == nil {
			configErr = fmt.Errorf("failed to set InterOpNumThreads to %d: %w", threads, err)
		}
		if ep.UsesCUDA() {
			if err := appendCUDAProvider(so, ep.DeviceID); err != nil {
				providerErr = cudaInitError("provider registration", ep.DeviceID, err)
				return providerErr
			}
		}
		return nil
	}))
	if ep.UsesCUDA() {
		// Measure where the operators really run; registering the provider alone
		// does not prove the model executes on the GPU.
		classifierOpts = append(classifierOpts, ort.WithPlacementProbe())
	}
	classifier, err := ort.NewClassifier(modelPath, classifierOpts...)
	if err != nil {
		switch {
		case providerErr != nil:
			return nil, providerErr
		case ep.UsesCUDA() && isCUDARuntimeError(modelPath, err):
			return nil, cudaInitError("session creation", ep.DeviceID, err)
		case ep.UsesCUDA():
			// Not a CUDA stack error (for example a bad model file or a failed
			// placement probe), but the CUDA load still failed: report it.
			wrapped := fmt.Errorf("failed to create ONNX classifier on CUDA device %d: %w", ep.DeviceID, err)
			cudaRegistry.recordError(wrapped)
			return nil, wrapped
		}
		return nil, fmt.Errorf("failed to create ONNX classifier: %w", err)
	}
	if configErr != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("failed to configure ONNX session options: %w", configErr)
	}

	c := &onnxClassifier{
		classifier: classifier,
		numSpecies: len(opts.Labels),
	}
	if ep.UsesCUDA() {
		placement := classifier.Placement()
		if err := verifyCUDAPlacement(modelPath, ep.DeviceID, placement); err != nil {
			_ = classifier.Close()
			return nil, err
		}
		info := CUDASessionInfo{
			Model:      filepath.Base(modelPath),
			DeviceID:   ep.DeviceID,
			CUDANodes:  placement.Count(ort.ProviderCUDA),
			TotalNodes: placement.Total,
			CPUOps:     placement.Ops(ort.ProviderCPU),
		}
		c.cudaSession = cudaRegistry.add(info)
		logger.Global().Module("inference").Info("ONNX Runtime CUDA execution provider active",
			logger.String("model", info.Model),
			logger.String("device", CUDADeviceLabel(info.DeviceID)),
			logger.Int("cuda_nodes", info.CUDANodes),
			logger.Int("total_nodes", info.TotalNodes),
			logger.String("cpu_ops", strings.Join(info.CPUOps, ",")))
	}
	return c, nil
}

// Placement returns the operator placement measured when the classifier was
// loaded, or nil for a CPU session (which is not probed).
func (c *onnxClassifier) Placement() *ort.NodePlacement {
	if c.classifier == nil {
		return nil
	}
	return c.classifier.Placement()
}

// isCUDARuntimeError reports whether an ONNX Runtime session error came from
// the CUDA stack (driver, runtime, cuDNN or cuBLAS) rather than the model. The
// model path is removed first so a file or directory named "cuda" cannot match.
func isCUDARuntimeError(modelPath string, err error) bool {
	m := strings.ToLower(strings.ReplaceAll(err.Error(), modelPath, ""))
	return strings.Contains(m, "cuda") || strings.Contains(m, "cudnn") || strings.Contains(m, "cublas")
}

// Predict runs ONNX inference, returning raw logits (pre-activation).
func (c *onnxClassifier) Predict(samples []float32) ([]float32, error) {
	if c.classifier == nil {
		return nil, ort.ErrSessionClosed
	}
	return c.classifier.PredictRaw(samples)
}

// PredictWithEmbeddings runs ONNX inference, returning both raw logits and embedding vector.
func (c *onnxClassifier) PredictWithEmbeddings(samples []float32) (logits, embeddings []float32, err error) {
	if c.classifier == nil {
		return nil, nil, ort.ErrSessionClosed
	}
	return c.classifier.PredictRawWithEmbeddings(samples)
}

// NumSpecies returns the number of species in the model output.
func (c *onnxClassifier) NumSpecies() int {
	return c.numSpecies
}

// Close releases the ONNX session resources.
func (c *onnxClassifier) Close() {
	if c.classifier != nil {
		_ = c.classifier.Close()
		c.classifier = nil
		if c.cudaSession != 0 {
			cudaRegistry.remove(c.cudaSession)
			c.cudaSession = 0
		}
	}
}

// DetectEmbeddingOutput returns the output-port index and size of a model's BirdNET
// v2.4 embedding output ([.,1024]): index 1 for the 2-output backbone (logits@0 +
// embedding@1), index 0 for the head-pruned, embedding-only model. The bat OpenVINO
// path uses it to bind the embedding extractor to the correct port before inference.
// Returns an error when the model has no 1024-dim output. The ONNX Runtime must be
// initialized via InitONNXRuntime before calling.
func DetectEmbeddingOutput(modelPath string) (index, size int, err error) {
	return ort.DetectEmbeddingOutput(modelPath)
}

// DetectPredictionsOutput returns the output-port index of a model's
// species-predictions output: the output whose last dimension equals numClasses.
// BirdNET v3.0 also exposes a 1280-dim embeddings output whose position varies by
// export, so the OpenVINO path uses this to bind the classifier to the correct
// port before inference. Returns an error when no output matches numClasses.
func DetectPredictionsOutput(modelPath string, numClasses int) (index int, err error) {
	return ort.DetectPredictionsOutput(modelPath, numClasses)
}

// ONNXCustomClassifierOptions configures the ONNX custom classifier.
type ONNXCustomClassifierOptions struct {
	Labels     []string // Provide labels directly (takes priority over LabelsPath)
	LabelsPath string   // Load labels from file (text, CSV, or JSON)
	Threads    int
}

type onnxCustomClassifier struct {
	classifier *ort.CustomClassifier
}

// NewONNXCustomClassifier creates a CustomClassifier backed by an ONNX Runtime model.
func NewONNXCustomClassifier(modelPath string, opts ONNXCustomClassifierOptions) (CustomClassifier, error) {
	builder := ort.NewCustomClassifierBuilder().
		ModelPath(modelPath).
		TopK(0).
		MinConfidence(0)

	switch {
	case len(opts.Labels) > 0:
		builder = builder.Labels(opts.Labels)
	case opts.LabelsPath != "":
		builder = builder.LabelsPath(opts.LabelsPath)
	default:
		return nil, fmt.Errorf("ONNX custom classifier requires labels or labels path")
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	var configErr error
	builder = builder.SessionOptions(func(so *ortlib.SessionOptions) {
		if err := so.SetIntraOpNumThreads(threads); err != nil && configErr == nil {
			configErr = fmt.Errorf("failed to set IntraOpNumThreads to %d: %w", threads, err)
		}
		if err := so.SetInterOpNumThreads(threads); err != nil && configErr == nil {
			configErr = fmt.Errorf("failed to set InterOpNumThreads to %d: %w", threads, err)
		}
	})

	cc, err := builder.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX custom classifier: %w", err)
	}
	if configErr != nil {
		_ = cc.Close()
		return nil, fmt.Errorf("failed to configure ONNX custom classifier session: %w", configErr)
	}

	return &onnxCustomClassifier{
		classifier: cc,
	}, nil
}

// PredictEmbedding runs inference on an embedding vector.
func (c *onnxCustomClassifier) PredictEmbedding(embeddings []float32) ([]float32, error) {
	if c.classifier == nil {
		return nil, ort.ErrSessionClosed
	}
	return c.classifier.PredictRaw(embeddings)
}

// NumClasses returns the number of output classes.
func (c *onnxCustomClassifier) NumClasses() int {
	if c.classifier == nil {
		return 0
	}
	return c.classifier.NumClasses()
}

// InputDim returns the embedding vector length the classifier expects as input.
func (c *onnxCustomClassifier) InputDim() int {
	if c.classifier == nil {
		return 0
	}
	return c.classifier.InputDim()
}

// Labels returns the classification labels.
func (c *onnxCustomClassifier) Labels() []string {
	if c.classifier == nil {
		return nil
	}
	return c.classifier.Labels()
}

// Close releases the ONNX custom classifier session.
func (c *onnxCustomClassifier) Close() {
	if c.classifier != nil {
		_ = c.classifier.Close()
		c.classifier = nil
	}
}

// ONNXRangeFilterOptions configures the ONNX range filter.
type ONNXRangeFilterOptions struct {
	// Labels is the species label list. Required.
	Labels []string
}

// onnxRangeFilter implements BatchRangeFilter using an ONNX Runtime session.
type onnxRangeFilter struct {
	filter     *ort.RangeFilter
	numSpecies int
}

// NewONNXRangeFilter creates a BatchRangeFilter backed by an ONNX Runtime meta model.
func NewONNXRangeFilter(modelPath string, opts ONNXRangeFilterOptions) (BatchRangeFilter, error) {
	if len(opts.Labels) == 0 {
		return nil, fmt.Errorf("ONNX range filter requires labels")
	}

	filter, err := ort.NewRangeFilter(modelPath,
		ort.WithRangeFilterLabels(opts.Labels),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX range filter: %w", err)
	}

	return &onnxRangeFilter{
		filter:     filter,
		numSpecies: len(opts.Labels),
	}, nil
}

// Predict returns species occurrence scores for a geographic location and week.
func (r *onnxRangeFilter) Predict(latitude, longitude, week float32) ([]float32, error) {
	if r.filter == nil {
		return nil, ort.ErrSessionClosed
	}
	return r.filter.PredictRaw(latitude, longitude, week)
}

// PredictBatch runs batch inference on multiple location/week inputs.
// inputs is a flat slice of [lat, lon, week] triples: len(inputs) must equal batchSize * 3.
// Returns a flat slice of [batchSize * numSpecies] scores in row-major order.
func (r *onnxRangeFilter) PredictBatch(inputs []float32, batchSize int) ([]float32, error) {
	if r.filter == nil {
		return nil, ort.ErrSessionClosed
	}
	return r.filter.PredictBatchRaw(inputs, batchSize)
}

// NumSpecies returns the number of species in the range filter model output.
func (r *onnxRangeFilter) NumSpecies() int {
	return r.numSpecies
}

// Close releases the ONNX range filter session resources.
func (r *onnxRangeFilter) Close() {
	if r.filter != nil {
		_ = r.filter.Close()
		r.filter = nil
	}
}

// InitONNXRuntime initializes the ONNX Runtime with the given shared library path.
// When libraryPath is empty, searches standard system library paths for
// libonnxruntime.so (Linux) or onnxruntime.dll (Windows).
// Safe to call multiple times; skips if already initialized successfully.
// On failure, allows retry with a corrected path (supports hot-reload recovery).
func InitONNXRuntime(libraryPath string) (err error) {
	ortInitMu.Lock()
	defer ortInitMu.Unlock()

	if ortInitialized {
		return nil
	}

	if libraryPath == "" {
		libraryPath = findONNXRuntimeLibrary()
	}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("failed to initialize ONNX Runtime: %v; install guide: %s", r, ORTInstallGuideURL)
		}
	}()

	ort.MustInitORT(libraryPath)
	ortInitialized = true
	return nil
}

// findONNXRuntimeLibrary searches standard system paths for the ONNX Runtime
// shared library. Returns the first path found, or "onnxruntime" as a
// fallback for dlopen's default search.
func findONNXRuntimeLibrary() string {
	var candidates []string
	switch runtime.GOOS {
	case "windows":
		candidates = []string{"onnxruntime.dll"}
		if exePath, err := os.Executable(); err == nil {
			candidates = append([]string{filepath.Join(filepath.Dir(exePath), "onnxruntime.dll")}, candidates...)
		}
		for i, c := range candidates {
			if abs, err := filepath.Abs(c); err == nil {
				candidates[i] = abs
			}
		}
	case "darwin":
		candidates = []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
			"libonnxruntime.dylib",
		}
	default:
		candidates = []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"libonnxruntime.so",
		}
	}
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	if runtime.GOOS == "darwin" {
		return "libonnxruntime.dylib"
	}
	return "onnxruntime"
}

// DestroyONNXRuntime tears down the ONNX Runtime environment.
// Resets initialization state so InitONNXRuntime can be called again. It is a
// no-op (returns nil) when the runtime was never initialized, mirroring
// DestroyOpenVINO, so a shutdown teardown can call it unconditionally without
// ort.DestroyORT reporting "InitializeRuntime has not been called".
func DestroyONNXRuntime() error {
	ortInitMu.Lock()
	defer ortInitMu.Unlock()
	if !ortInitialized {
		return nil
	}
	if err := ort.DestroyORT(); err != nil {
		return err
	}
	ortInitialized = false
	return nil
}
