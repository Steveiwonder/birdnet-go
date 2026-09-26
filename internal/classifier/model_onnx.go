package classifier

import (
	"os"
	"time"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/errors"
	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/logger"
)

// onnxModelPath resolves the ONNX classifier model file: the explicit config
// model path when set, otherwise ModelInfo.CustomPath (set by the arm64 default
// resolver, defaultClassifierModelInfo). The first value is the RESOLVED
// configured path (see BirdNET.configuredModelPath), not the raw setting.
// Returns "" when neither is set. Keeping
// the default in CustomPath avoids mutating settings.BirdNET.ModelPath, which
// would make the default indistinguishable from a user override.
func (bn *BirdNET) onnxModelPath() string {
	if p := bn.configuredModelPath(); p != "" {
		return p
	}
	return bn.ModelInfo.CustomPath
}

// initializeONNXModel loads and initializes an ONNX model as the classifier backend.
func (bn *BirdNET) initializeONNXModel() error {
	start := time.Now()
	log := GetLogger()
	settings := bn.Settings

	modelPath := bn.onnxModelPath()
	if modelPath == "" {
		return errors.Newf("ONNX classifier model path is empty").
			Category(errors.CategoryModelInit).
			Context("model_id", bn.ModelInfo.ID).
			Build()
	}

	// Expand environment variables and the ~ prefix so dispatch and loading agree:
	// usesONNXBackend env-expands the path before checking the .onnx extension, so a
	// configured $VAR/~ ONNX path would dispatch here yet fail to open if left raw.
	rawPath := modelPath
	modelPath = os.ExpandEnv(modelPath)
	modelPath, err := conf.ExpandTildePath(modelPath)
	if err != nil {
		return errors.New(err).
			Category(errors.CategoryFileIO).
			Context("path", rawPath).
			Build()
	}

	if err := checkORTOrFail(settings.BirdNET.ONNXRuntimePath, "ONNX classifier", "onnx_classifier", ""); err != nil {
		return err
	}

	// Initialize ONNX Runtime if not already done
	if err := inference.InitONNXRuntime(settings.BirdNET.ONNXRuntimePath); err != nil {
		return errors.New(err).
			Category(errors.CategoryModelInit).
			Context("onnx_runtime_path", settings.BirdNET.ONNXRuntimePath).
			Timing("onnx-init", time.Since(start)).
			Build()
	}

	ep := executionProviderFor(&settings.BirdNET)
	classifier, err := inference.NewONNXClassifier(modelPath, inference.ONNXClassifierOptions{
		Labels:            settings.BirdNET.Labels,
		Threads:           settings.BirdNET.Threads,
		ExecutionProvider: ep,
	})
	if err != nil {
		return errors.New(err).
			Category(errors.CategoryModelInit).
			ModelContext(modelPath, bn.ModelInfo.ID).
			Timing("onnx-model-init", time.Since(start)).
			Build()
	}

	bn.classifier = classifier
	// The device is the execution provider the session was built on: the CPU
	// provider, or CUDA:<id> once NewONNXClassifier has verified that operators
	// really ran on the GPU (a CUDA request never yields a CPU session). ONNX
	// Runtime executes the model file as-is, so the runtime precision is the
	// weight precision recorded in ModelInfo.Quantization (e.g. INT8 for the arm64
	// int8 variant, FP32 for an fp32 ONNX model).
	device := onnxDeviceLabel(ep)
	bn.setRuntimeInfo(device, BackendONNX, string(bn.ModelInfo.Quantization))

	log.Info("ONNX model initialized",
		logger.String("model", modelPath),
		logger.String("execution_provider", ep.Provider),
		logger.String("device", device),
		logger.Int("species", classifier.NumSpecies()))

	return nil
}

// executionProviderFor returns the ONNX Runtime execution provider selected in
// a settings snapshot. The provider name is canonicalized; an unrecognised
// value is passed through unchanged so NewONNXClassifier rejects it with a clear
// error rather than it silently becoming CPU.
func executionProviderFor(cfg *conf.BirdNETConfig) inference.ExecutionProviderOptions {
	provider, err := inference.NormalizeExecutionProvider(cfg.ONNXProvider)
	if err != nil {
		provider = cfg.ONNXProvider
	}
	return inference.ExecutionProviderOptions{Provider: provider, DeviceID: cfg.CUDADeviceID}
}

// onnxDeviceLabel is the runtime device reported for an ONNX Runtime session
// built with ep: "CUDA:<id>" for CUDA, otherwise "CPU".
func onnxDeviceLabel(ep inference.ExecutionProviderOptions) string {
	if ep.UsesCUDA() {
		return inference.CUDADeviceLabel(ep.DeviceID)
	}
	return deviceCPU
}

// errCUDARequiresONNX builds the error for a CUDA selection with a TFLite
// primary model, which has no CUDA path: running it would silently use the CPU.
func errCUDARequiresONNX(modelID, modelPath string) error {
	return errors.Newf("onnxprovider is set to cuda but the %s model is a TFLite model, which cannot run on CUDA; "+
		"install an ONNX variant (for example the BirdNET v2.4 FP32 ONNX model) or set birdnet.onnxprovider to cpu", modelID).
		Component("classifier").
		Category(errors.CategoryModelInit).
		Context("operation", "select_execution_provider").
		ModelContext(modelPath, modelID).
		Build()
}
