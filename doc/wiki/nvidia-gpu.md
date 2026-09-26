# NVIDIA GPU (CUDA) Inference

BirdNET-Go can run its ONNX species classifiers on an NVIDIA GPU through the
ONNX Runtime CUDA execution provider. Audio capture, the dashboard, the API,
detections and every integration work exactly as on the CPU; only the neural
network inference moves to the GPU.

CPU remains the default. Nothing changes for an existing installation until you
explicitly select CUDA.

> **No silent fallback.** When you select CUDA and it cannot be used (no GPU
> visible, missing libraries, a TFLite model, or a model whose operators the
> CUDA provider cannot run), the model fails to load with an error that says
> why. BirdNET-Go never quietly runs the model on the CPU instead. The status
> page and API report the provider that is really in use.

## Do You Need This Guide?

**You want this if** you run BirdNET-Go on an x86-64 (amd64) Linux host with an
NVIDIA GPU and want to take inference load off the CPU, for example when running
many RTSP streams or the large Perch v2 model.

**You can skip this if** you have no NVIDIA GPU, run on a Raspberry Pi or other
ARM board, or are happy with CPU performance. For Intel iGPUs see the
[OpenVINO Acceleration Guide](openvino-acceleration.md).

## Requirements

| Requirement      | Details                                                                                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Host             | Linux, x86-64 (amd64). The CUDA image is not built for arm64 (Jetson) or Windows/macOS.                                                                                                                                        |
| GPU              | An NVIDIA GPU with compute capability **7.0 or newer**: GeForce GTX 16 / RTX 20 series (Turing) or later, Volta, Ampere, Ada and Hopper cards. ONNX Runtime 1.25 ships no kernels for older cards (GTX 10 series and earlier). |
| NVIDIA driver    | A host driver for CUDA 12: **525.60 or newer**; 575 or newer is recommended (it supports the bundled CUDA 12.9 natively). `nvidia-smi` on the host must list the GPU.                                                          |
| Container access | The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), so `docker run --gpus all` works.                                                                 |
| Image            | The CUDA image variant (a `cuda` tag). The standard images ship the CPU build of ONNX Runtime and cannot use CUDA.                                                                                                             |
| Model            | An **ONNX** classifier in FP32 or FP16, for example the BirdNET v2.4 optimized FP32 build (`fp32-dfttrunc`), BirdNET v3.0, or Perch v2. The built-in amd64 model is TFLite (below).                                            |
| GPU memory       | Depends on the model; the larger Perch v2 and BirdNET v3.0 need more than BirdNET v2.4. `nvidia-smi` and `cuda-benchmark` show the actual use.                                                                                 |

Check the driver and toolkit before going further:

```bash
nvidia-smi                                                 # on the host
docker run --rm --gpus all ubuntu nvidia-smi               # through the toolkit
```

Both must list your GPU. If the second command fails, fix the NVIDIA Container
Toolkit installation first; BirdNET-Go cannot reach the GPU without it.

## What Runs on the GPU

| Model                                     | On CUDA | Notes                                                                                                      |
| ----------------------------------------- | :-----: | ---------------------------------------------------------------------------------------------------------- |
| BirdNET v2.4, ONNX FP32 (`fp32-dfttrunc`) |   Yes   | Recommended. Install it from the model gallery (below).                                                    |
| BirdNET v2.4, built-in TFLite             |   No    | TFLite has no CUDA path. Selecting CUDA with this model is an error, not a CPU fallback.                   |
| BirdNET v2.4 INT8 (arm64 default)         | Partly  | Quantized operators mostly stay on the CPU; use the FP32 build on a GPU.                                   |
| BirdNET v3.0 (ONNX)                       |   Yes   |                                                                                                            |
| Perch v2 (ONNX)                           |   Yes   | Operators the CUDA provider lacks stay on the CPU (reported, see [Verifying](#verifying-the-gpu-is-used)). |
| BattyBirdNET                              | Partly  | The heavy embedding model runs on CUDA; the tiny classifier head stays on the CPU.                         |
| Range filter, privacy/voice filter        |   No    | Small models that always run on the CPU.                                                                   |

ONNX Runtime assigns each operator of a model to the CUDA provider when it has a
CUDA kernel for it, and leaves the rest (typically shape bookkeeping) on the CPU.
This is normal. At load time BirdNET-Go runs one profiled inference and counts
where every operator actually ran; if **no** operator ran on the GPU, the load
fails. The counts are logged and shown on the status page.

When CUDA is selected it takes precedence over OpenVINO: models run on ONNX
Runtime's CUDA provider even where the `auto` backend would otherwise pick
OpenVINO. Forcing `backend: openvino` together with `onnxprovider: cuda` is
rejected as a configuration error.

## Configuration

Two settings control the provider. Both are in the `birdnet` section of
`config.yaml` and have environment variable equivalents.

| Setting                | Environment variable   | Values                  | Meaning                                                                     |
| ---------------------- | ---------------------- | ----------------------- | --------------------------------------------------------------------------- |
| `birdnet.onnxprovider` | `BIRDNET_ONNXPROVIDER` | `cpu` (default), `cuda` | ONNX Runtime execution provider for species classifiers.                    |
| `birdnet.cudadeviceid` | `BIRDNET_CUDADEVICEID` | `0` (default), `1`, ... | Which GPU to use when there is more than one (as numbered by `nvidia-smi`). |

```yaml
birdnet:
  onnxprovider: cuda
  cudadeviceid: 0
```

Both settings hot-reload: changing them rebuilds the loaded models on the new
provider without a restart. The new model is built before the old one is
released, so if CUDA fails to initialize, the previous (CPU) model keeps
running and the error is reported.

An unrecognised value such as `gpu` is rejected at startup. It is never
treated as `cpu`.

## Running the CUDA Image with Docker

The CUDA images are built by the `docker-build-cuda` workflow and published to
the GitHub Container Registry of the repository that runs it. For this fork the
image is `ghcr.io/steveiwonder/birdnet-go`:

| Tag                | Built from                                   |
| ------------------ | -------------------------------------------- |
| `cuda`             | the latest `main`                            |
| `cuda-latest`      | the latest release tag                       |
| `cuda-<version>`   | a release tag, e.g. `cuda-v1.2.3`            |
| `cuda-<branch>`    | a branch whose name contains `gpu` or `cuda` |
| `cuda-<short sha>` | a specific commit                            |

The image is the standard BirdNET-Go image plus the GPU build of ONNX Runtime
and the CUDA 12 and cuDNN 9 runtime libraries, so it is about 2.5 GB larger.
The NVIDIA driver is not in the image; the Container Toolkit mounts the host's
driver when the container starts with GPU access.

### docker run

```bash
docker run -d --name birdnet-go \
  --gpus all \
  -p 8080:8080 \
  -v ./config:/config -v ./data:/data \
  --device /dev/snd \
  -e BIRDNET_ONNXPROVIDER=cuda \
  ghcr.io/steveiwonder/birdnet-go:cuda
```

`--gpus all` exposes every GPU; `--gpus device=1` exposes only the second one
(inside the container it is then device 0).

### Docker Compose

```yaml
services:
  birdnet-go:
    image: ghcr.io/steveiwonder/birdnet-go:cuda
    environment:
      - BIRDNET_ONNXPROVIDER=cuda
      # - BIRDNET_CUDADEVICEID=0
    devices:
      - /dev/snd
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # ... your other settings (ports, volumes)
```

### Select an ONNX model

The built-in amd64 model is TFLite, which cannot run on CUDA, so with
`onnxprovider: cuda` BirdNET-Go refuses to start the built-in model and tells
you to install an ONNX build. In the web UI open **Settings > Analysis >
Models**, choose **BirdNET v2.4** and install the optimized **FP32** build
(`fp32-dfttrunc`). Perch v2 and
BirdNET v3.0 are ONNX models and run on CUDA as installed.

To start directly on CUDA, set CPU first, install the FP32 build, then switch
`onnxprovider` to `cuda`. Alternatively download
`BirdNET_v2.4_fp32_dfttrunc.onnx` from the
[tphakala/BirdNET-v2.4](https://huggingface.co/tphakala/BirdNET-v2.4) model
repository into `data/models` and point `BIRDNET_MODELPATH` at it.

## Native (Non-Container) Installs

1. Install the NVIDIA driver (575 or newer) and check `nvidia-smi`.
2. Install the CUDA 12 runtime libraries (cudart, cuBLAS, cuFFT, cuRAND, NVRTC)
   and cuDNN 9, from NVIDIA's CUDA repository (`cuda-libraries-12-9` and
   `libcudnn9-cuda-12`) or the CUDA installer.
3. Replace the CPU ONNX Runtime with the GPU build of the **same version**
   BirdNET-Go requires (1.25.x):

   ```bash
   curl -fsSLO https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-gpu-1.25.1.tgz
   tar -xzf onnxruntime-linux-x64-gpu-1.25.1.tgz
   sudo cp -a onnxruntime-linux-x64-gpu-1.25.1/lib/libonnxruntime*.so* /usr/local/lib/
   sudo ldconfig
   ldd /usr/local/lib/libonnxruntime_providers_cuda.so | grep "not found"   # must print nothing
   ```

   `libonnxruntime_providers_cuda.so` and `libonnxruntime_providers_shared.so`
   must sit in the same directory as `libonnxruntime.so`. If that is not a
   default library path, set `birdnet.onnxruntimepath` to the full path of
   `libonnxruntime.so`.

4. Set `birdnet.onnxprovider: cuda`, select an ONNX model and restart.

## Verifying the GPU Is Used

Do not assume it worked. Check at least one of the following.

### 1. The inference status API

```bash
curl -s http://localhost:8080/api/v2/system/inference \
  | jq '.backends.cuda, (.models[] | {name, backend, device})'
```

- `backends.cuda.provider` is the configured provider (`cuda`).
- `backends.cuda.libraryPresent: true` means the image has the GPU build of
  ONNX Runtime. It does not prove a GPU is in use.
- `backends.cuda.active: true` means at least one model session is **running on
  CUDA**. Each entry in `backends.cuda.sessions` gives the model, device and
  `cudaNodes` of `totalNodes`: how many operators ran on the GPU in the load
  probe. `cpuOps` lists the operator types left on the CPU.
- `backends.cuda.lastError` holds the most recent CUDA initialization failure.
- Each model reports `backend: "ONNX"` and `device: "CUDA:0"` (the GPU number)
  when it runs on the GPU, and `device: "CPU"` otherwise.

### 2. The web UI

Open **System > AI Models & Inference**. The Inference Backends card has an
**NVIDIA CUDA** row: **Active** with an "N of M operators on GPU" badge per
model, **Failed to initialize** with the error text, or **Selected, but no
model is running on CUDA**. Each model card shows its device (`CUDA:0`).

### 3. Logs

A model on CUDA logs lines like these:

```text
INFO  [inference] ONNX Runtime CUDA execution provider active model=BirdNET_v2.4_fp32_dfttrunc.onnx device=CUDA:0 cuda_nodes=... total_nodes=... cpu_ops=...
INFO  [classifier] ONNX model initialized ... execution_provider=cuda device=CUDA:0
```

With CUDA selected, OpenVINO also logs that it was declined because
`onnxprovider is set to cuda`.

### 4. Watch the GPU

While audio is being analyzed, `nvidia-smi` on the host lists the `birdnet-go`
process under **Processes** with GPU memory in use, and **GPU-Util** rises
during inference:

```bash
watch -n 1 nvidia-smi
```

## Benchmark: CPU vs CUDA

The `cuda-benchmark` command runs the same audio through a model on the CPU and
then on CUDA, and reports load time, latency, throughput, process CPU use, GPU
utilization and memory (sampled with `nvidia-smi`), the measured operator
placement, and whether both providers produce the same detections.

```bash
docker exec birdnet-go birdnet-go cuda-benchmark \
  --model /data/models/birdnet-v2.4/BirdNET_v2.4_fp32_dfttrunc.onnx \
  --audio /data/clips/some-recording.wav \
  --iterations 5
```

- `--audio` takes a mono WAV at the model's sample rate (48 kHz for BirdNET
  v2.4, 32 kHz for v3.0 and Perch); without it a synthetic signal is used.
- `--model` is the path of the installed ONNX file (the example is where the
  gallery installs the BirdNET v2.4 FP32 build).
- `--labels` is optional; with a label file the comparison lists species names.
- `--cuda-device` selects the GPU; `--min-score` sets the confidence at which a
  class counts as detected in the comparison (default 0.5).
- The command exits non-zero when CUDA fails, so it doubles as a CUDA check.

Example output layout:

```text
Provider  Load       Mean       Median     p95        Windows/s  CPU use
--------  ---------  ---------  ---------  ---------  ---------  --------
cpu       ...        ...        ...        ...        ...        ...
cuda      ...        ...        ...        ...        ...        ...

GPU activity (device 0, N samples): avg ..%, peak ..% utilization, peak .. MiB memory
CUDA operator placement: X of Y operators on the GPU (CPU: Shape, ...)

Detection comparison over N windows (CPU is the reference):
  top-1 class identical:          N/N
  detections >= 0.50 identical:   N/N
  max |logit diff|:               ...
```

Small differences in the last decimal places between CPU and GPU are expected
(different kernels and summation order); the top classes and the detections at
your threshold should match.

## Troubleshooting

The error text always ends with a `hint:` naming the likely fix.

### `CUDA execution provider is not enabled in this build`

The ONNX Runtime library in use is the CPU build. Use a `cuda` image tag, or on
a native install replace ONNX Runtime with the GPU build (see
[Native Installs](#native-non-container-installs)).

### `libcublasLt.so.12: cannot open shared object file` (or `libcudnn.so.9`, `libcudart.so.12`, ...)

The GPU build of ONNX Runtime is present but the CUDA or cuDNN libraries are
not. In the container this should not happen; on a native install, install the
CUDA 12 runtime libraries and cuDNN 9 and run `sudo ldconfig`.

### `CUDA driver version is insufficient for CUDA runtime version`

The container cannot see a driver, or the host driver is too old:

- The container was started without `--gpus all` (or the Compose `deploy`
  block), or the NVIDIA Container Toolkit is not installed or configured
  (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
- The host driver is older than CUDA 12 supports (525.60). Update it; 575 or
  newer is recommended.

### `no CUDA-capable device is detected` / `invalid device ordinal`

No GPU is visible, or `cudadeviceid` is higher than the number of GPUs the
container sees. With `--gpus device=1`, the container sees a single GPU,
numbered 0.

### `onnxprovider is set to cuda but the ... model is a TFLite model`

The built-in amd64 model is TFLite. Install the BirdNET v2.4 FP32 ONNX build
(see [Select an ONNX model](#select-an-onnx-model)) or set `onnxprovider: cpu`.

### `model did not run on the CUDA execution provider`

The CUDA provider attached, but none of the model's operators ran on the GPU
(typically an INT8-quantized model). Use the FP32 or FP16 build of the model.

### `onnxprovider 'cuda' cannot be combined with backend 'openvino'`

Set `birdnet.backend` to `auto` (the default) or `onnx`.

### Pulling the image fails with `denied` or `unauthorized`

New GitHub Container Registry packages can be private. Either
`docker login ghcr.io` with a personal access token that has `read:packages`,
or make the package public under **GitHub > Your profile > Packages >
birdnet-go > Package settings**.

### High CPU use even on CUDA

Audio capture, resampling, the spectrogram and clip exports still run on the
CPU, as do the range filter and any operators listed under `cpuOps`. Use
`cuda-benchmark` to compare the inference share before and after.

## Licenses

The CUDA image redistributes NVIDIA's CUDA runtime, cuBLAS, cuFFT, cuRAND,
NVRTC, nvJitLink and cuDNN libraries under their NVIDIA license terms, and
Microsoft's ONNX Runtime under the MIT license. Their license and copyright
files are in the image under `/usr/share/doc/nvidia-cuda/` and
`/usr/share/doc/onnxruntime-gpu/`. BirdNET-Go's own license is unchanged.
