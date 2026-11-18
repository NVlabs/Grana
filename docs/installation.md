# Install Grana

These instructions apply to a source checkout or an unpacked source release.
Installation and environment checks require no robot recordings, task-specific
configuration, model checkpoint or existing virtual environment. Project data
and recipes are supplied separately after installation.

| Intended use | Environment | Hardware |
|---|---|---|
| Human labeling, annotation review and cache construction | CPU environment | Linux CPU; no GPU required |
| Model inference and training | Full CUDA environment | Linux x86_64 or aarch64 with a compatible NVIDIA GPU |

The CUDA installer selects native wheels and a GPU backend from the local CPU
architecture and GPU capabilities. See [hardware guidance](#hardware-and-capacity)
for memory planning and [training](training.md) for selecting devices and batches.

Both installation paths pin Pillow to **12.0.0**; the full-runtime environment
check verifies its version and import location. This pin is for reproducibility;
follow the [image-decoding restrictions](../SECURITY.md#image-decoding-and-rendering)
when selecting inputs.

## System prerequisites

- Linux with Git, Bash and [uv](https://docs.astral.sh/uv/getting-started/installation/).
  Ubuntu 22.04 is a tested starting point; use equivalent packages on other
  compatible distributions.
- `ffmpeg` and `ffprobe` when using video workflows. Python/runtime installation
  does not require an FFmpeg self-test or a particular version, path or build.
  Decoding and encoding capabilities are checked by the commands that use them.
- A C/C++ compiler for the full runtime's model compilation.
- For the pinned CUDA 13 runtime, a compatible NVIDIA driver in the
  [580 or newer family](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
  Verify GPU access with `nvidia-smi` before installing the full environment.

For Debian/Ubuntu, system packages can be installed with:

```bash
sudo apt-get update
sudo apt-get install git ffmpeg build-essential
uv --version
```

Install uv using its linked instructions first if it is not on `PATH`.
Normal labeling does not need Node.js or Chrome on the server. The default GPU
installation uses prebuilt CUDA wheels, including the causal-conv1d kernel; a
separate CUDA toolkit, FlashAttention, Unsloth and TileLang are not installation
prerequisites.

## Obtain the source

Use the repository URL published with the release. For a Git checkout, set
`GRANA_REPOSITORY_URL` to that URL, then run:

```bash
: "${GRANA_REPOSITORY_URL:?Set the repository URL published with the release}"
git clone "$GRANA_REPOSITORY_URL" grana
cd grana
```

Select the published release tag or the branch you intend to use before
installing. Alternatively, extract the source archive and enter the directory
containing `pyproject.toml`, `grana/`, `scripts/` and `requirements/`; no `.git`
directory is needed to install from an archive. Git is still needed to fetch
the pinned Transformers dependency.

Run the remaining commands from that source root. Start in a shell without
another activated venv or a `PYTHONPATH` pointing to an older checkout. Setup
needs initial network access to the Python/package indexes and GitHub.

## CPU labeling and cache construction

```bash
uv python install 3.12.12
uv venv --python 3.12.12 .venv-annotation
uv pip install --python .venv-annotation/bin/python \
  -r requirements/labeling-cpu.txt
source .venv-annotation/bin/activate
python -m grana.cli label --help
```

This installs the pinned CPU dependencies, including the parquet reader and
cache writer. It uses the source tree directly and does not create a `grana`
console command: run `python -m grana.cli` from the source root. The system
Python version does not matter because uv selects Python 3.12.12 explicitly.

`requirements/annotation.txt` alone adds the API client and schema validator
to an existing full environment; it is not a complete standalone CPU install.
Likewise, `pip install -e .` requests the full model dependencies. Keep the CPU
pins and full-runtime pins in separate environments.

At this point installation is complete. To use your own data, follow
[Configure a project](project-configuration.md). That guide defines standalone
recipes without copying or inheriting repository example configurations.

## Full CUDA environment

Use a fresh shell for this path. The training launcher and runtime ownership
checks require the environment at `.venv-qwen35-unsloth` under the source root.
This historical directory name does not imply an Unsloth dependency.

The same installer automatically selects CPU-specific native wheels and the
GPU backend. Shared Python, Torch and CUDA pins stay the same across these
platforms; only the architecture-specific wheel and Hopper additions differ.

| CPU architecture | GPU | Compute capability | Native wheels | FLA profile |
|---|---|---|---|---|
| x86_64 | H100 | 9.0 | Linux x86_64 | `hopper-tilelang` (adds TileLang and its compiler) |
| x86_64 | RTX 4090 | 8.9 | Linux x86_64 | `default` (Triton) |
| x86_64 | RTX 5090 | 12.0 | Linux x86_64 | `default` (Triton) |
| aarch64 | GB300 | 10.3 | Linux aarch64 | `default` (Triton) |

Preview the detected CPU architecture, local GPU compute capabilities and
selected requirements with `bash scripts/setup_env.sh --plan`. This needs only
Python 3 and does not install packages or create a venv. The regular installer
prints the same plan before installing packages. Detection uses capabilities,
not hostnames or GPU counts; any local Hopper GPU selects the Hopper profile.
An empty GPU list means detection was unavailable; the CUDA runtime check still
requires a usable GPU. `GRANA_FLA_TILELANG=1|0` remains an explicit override.

```bash
uv python install 3.12.3
bash scripts/setup_env.sh
source .venv-qwen35-unsloth/bin/activate
grana --help
```

The installer copies its Python runtime and installs native Transformers, PEFT,
FLA and Ray with pinned dependencies. It neither reads another venv nor needs a
task recipe.

Setup runs `check_env.py` to check interpreter ownership, package versions, the Transformers
source revision, imports and a CUDA reduction. Use its `--output` option with a
path of your choice to retain a receipt including CPU architecture, GPUs and
the selected FLA profile. If you also use API annotation/review,
install its optional dependencies in this environment:

```bash
python -m pip install -r requirements/annotation.txt
```

Base-model files and Grana checkpoints are separate inputs, not installation
artifacts. Keep `HF_HOME`, if configured, outside the venv; enable
`HF_HUB_OFFLINE=1` only after the required files are present. Configure logging
credentials only when enabling online logging. Use the repository's pinned
dependencies for the supported runtime.
Remove a stale `PYTHON_BIN` override when switching an inference launcher to
this environment, or point it to the new interpreter.

Inference requires the full CUDA environment, including FLA and causal-conv1d;
`pip install -e .` alone does not install all required kernels. See
[inference runtime requirements](inference.md#runtime-requirements).

## Ray runtime and online logging

Grana pins **Ray 2.55.0 with Train V1** (`RAY_TRAIN_V2_ENABLED=0`). Setup
configures this default, the training and checkpoint-evaluation launchers enforce
it, and `check_env.py` verifies it. Keep the pinned version and Train V1 together.

Keep Ray's default JSON-only tensor metadata reader enabled.
`check_env.py` rejects `RAY_DATA_AUTOLOAD_CLOUDPICKLE_TENSOR_METADATA=1`;
this unsafe compatibility mode is outside the supported runtime. See
[cached-data restrictions](../SECURITY.md#ray-jobs-and-data-readers).

The launchers start a single-host Ray instance with control services configured
for `127.0.0.1` and the dashboard disabled. Worker and agent sockets can still
listen on other interfaces; this is not complete network isolation. See
[service boundaries](../SECURITY.md#ray-jobs-and-data-readers).

Outbound access remains available for online W&B. Run `wandb login` in the
activated environment, then set `logging.mode: online` and your W&B project/entity
in the training recipe. See [training and logging](training.md#wb-and-local-evidence)
for launch commands and outputs. Multi-GPU training uses devices on one host;
the supplied launchers do not configure a multi-host Ray cluster.

<a id="rtx-4090-recommendations"></a>
<a id="rtx-5090-recommendations"></a>
## Hardware and capacity

- DDP holds a complete model replica on each GPU; device memory is not pooled.
  Match `training.num_workers` to the intended allocation.
- Start with 0.8B and a small physical batch when qualifying new data or hardware.
  Account for image grids, camera count, sequence lengths, evaluation and CUDA
  reserved memory.
- Gradient checkpointing is enabled by default to reduce activation memory.
  Accumulation can preserve the update budget with a smaller physical batch,
  subject to [role packing constraints](TRAINING_CONFIGS.md#global-batch-and-actual-stage-supervision).
- Validate the resolved recipe and caches before launch. Measure the intended
  training and evaluation paths before committing to a long run.

### H100 and other Hopper GPUs

FLA 0.5.2's Triton gated delta-rule backward does not support the pinned Hopper
training path. The installer detects Hopper and selects `hopper-tilelang`,
installing TileLang and its CUDA compiler from
[hopper-tilelang.txt](../requirements/hopper-tilelang.txt). FLA dispatches the
required backward kernel to TileLang on Hopper. Other GPUs keep the default
Triton profile.

The selected profile is recorded in
`.venv-qwen35-unsloth/grana-fla-profile.json`. `GRANA_FLA_TILELANG=1|0` explicitly
controls this profile on the next setup run. `check_env.py` checks Hopper
forward/backward dispatch and agreement with an FP32 reference; the first run
compiles the kernel. For application regression tests, see the
[contributor guide](../CONTRIBUTING.md#optional-local-checks).

## Reinstalling or moving a deployment

Save unfinished browser edits before shutting down a labeling server. Preserve
project data separately from disposable environments: the active labeling YAML,
its current labels, the complete adjacent `.stage-versions/` history, source
recordings, training caches, model files and checkpoints. **Apply stages** can
move the active labels into that hidden history directory; the original
`labels/` alone may not contain the current annotations.

Git does not restore these files. Preserve their paths when restoring an
existing project because label source bindings and version recipes contain
paths. Rebuild a moved venv at its destination instead of assuming its absolute
interpreter and launcher paths remain valid. Update any service's working
directory, interpreter, project config and environment variables, then restart
it and refresh the browser to load the new UI assets.

For an optional persistent service, set its working directory to the source
root and use the environment's absolute Python path. Declare recipe variables
in the service environment; it does not inherit interactive shell exports.
Stop the old server before reusing its port or project. For foreground startup
and SSH forwarding, see the [labeling guide](human-labeling.md#configure-and-start).
