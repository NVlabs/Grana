# Install Grana

These instructions apply to a source checkout or an unpacked source release.
Installation and environment checks require no robot recordings, task-specific
configuration, model checkpoint or existing virtual environment. Project data
and recipes are supplied separately after installation.

| Intended use | Environment | Hardware |
|---|---|---|
| Human labeling, annotation review and cache construction | CPU environment | Linux CPU; no GPU required |
| Model inference and training | Full CUDA environment | Linux x86_64 with H100 / RTX 4090 / RTX 5090, or Linux aarch64 with GB300 |

The full runtime has been installed and checked on Linux x86_64 with RTX 4090,
RTX 5090 and H100.
Full-training qualification on GB300 is documented separately in the
[training guide](training.md). Installation checks do not establish training
capacity or model quality for every hardware/configuration combination.

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
needs initial network access to the Python/package indexes and GitHub. Preserve
the source release version with your environment verification results.

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
python scripts/check_env.py
grana --help
```

The installer copies its Python runtime and installs native Transformers, PEFT,
FLA and Ray with pinned dependencies. It neither reads another venv nor needs a
task recipe. Keep `requirements/runtime-gb300.txt` and
`requirements/constraints-gb300.txt` with source releases: their historical
filenames identify the baseline dependency pins, not a required machine name.

`check_env.py` checks interpreter ownership, package versions, the Transformers
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

The installer also adds the causal-conv1d 1.7.0 kernel, part of the measured
inference latency stack, from hash-pinned prebuilt x86_64/aarch64 wheels
([inference-fast.txt](../requirements/inference-fast.txt)), with no CUDA
toolkit or source build, and checks it on the GPU. Transformers also uses this
kernel during training.

Model inference requires this full CUDA environment, including FLA and
causal-conv1d. A bare `pip install -e .` does not establish that the required CUDA
kernels are present. Missing kernels or unsupported hardware produce an error;
there is no CPU inference mode or silent Torch-kernel replacement. CPU labeling,
cache construction and the training/evaluation recipes retain their own
separate requirements.

Every inference entrypoint uses GPU resize and the same required acceleration
stack, with eight Torch CPU threads. HTTP clients send binary RGB and the server
reuses one inference worker across connections. See [inference](inference.md)
for the API and [benchmark methodology](benchmarks.md) for latency and throughput.

<a id="rtx-4090-recommendations"></a>
<a id="rtx-5090-recommendations"></a>
## Hardware and capacity

Installation checks establish that the pinned runtime loads and executes its
checks. They do not establish convergence, memory capacity for every recipe,
or a portable latency number. Reference training completed on GB300; H100,
RTX 4090 and RTX 5090 checks have narrower scopes. See
[execution validation](results.md#execution-validation).

- DDP holds a complete model replica on each GPU; device memory is not pooled.
  Match `training.num_workers` to the intended allocation.
- Start with 0.8B and a small physical batch when qualifying new data or hardware.
  Account for image grids, camera count, sequence lengths, evaluation and CUDA
  reserved memory. RTX 4090 has 24 GB and RTX 5090 has 32 GB per device.
- Keep `model.gradient_checkpointing: true` for training, smoke tests, profiling
  and batch-size comparisons. Accumulation can preserve the update budget with
  a smaller physical batch, subject to role packing constraints.
- Validate the resolved recipe and caches before launch. Measure the intended
  training and evaluation paths before committing to a long run.
- Warm representative inference requests before timing. New prompt lengths or
  image shapes may trigger compilation.

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
compiles the kernel. These kernel checks do not qualify a full training campaign.

## Verify without private data

The installation checks above do not load datasets or checkpoints. Optional
annotation regression checks generate synthetic videos locally:

```bash
uv pip install --python .venv-annotation/bin/python 'pytest==9.2.0'
.venv-annotation/bin/python -m pytest \
  tests/test_labeling.py tests/test_label_cli.py tests/test_label_ui.py
GRANA_BROWSER_SMOKE=1 .venv-annotation/bin/python -m pytest -s \
  tests/test_label_ui_browser.py
```

The JavaScript checks need Node.js 18+; the browser test needs Node.js 22+,
Chrome/Chromium and FFmpeg with libx264. Set `CHROME_BIN` if needed. No private
recordings or npm packages are required. Check the summary: a skipped browser
test is not a successful browser verification. Set `GRANA_BROWSER_OUTPUT` to
an external directory if you want to retain screenshots and receipts.

The following optional CUDA check uses a small synthetic model and does not
download model weights. Select an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-qwen35-unsloth/bin/python -m pytest \
  tests/test_inference_acceleration.py::test_real_compile_on_cuda_matches_eager_closely
```

For development, run the complete test suite as described in
[CONTRIBUTING.md](../CONTRIBUTING.md). Development and packaging tests also
exercise the fixtures shipped with that release; they are not prerequisites
for installing the application or configuring a new task.

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

For a persistent labeling service on a systemd-based host, create a user unit
such as `~/.config/systemd/user/grana-label.service`. Replace `@SOURCE_ROOT@`
and `@LABEL_CONFIG@` below with your own absolute paths before enabling it:

```ini
[Unit]
Description=Grana local labeling

[Service]
WorkingDirectory=@SOURCE_ROOT@
ExecStart="@SOURCE_ROOT@/.venv-annotation/bin/python" -E -s -m grana.cli label --config "@LABEL_CONFIG@" --port 8790
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

Use resolved paths in the labeling YAML, or declare its variables with
`Environment=` in the unit. A service does not read shell exports. Stop the
previous server before reusing its port or project, then reload and inspect:

```bash
systemctl --user daemon-reload
systemctl --user enable --now grana-label.service
systemctl --user status grana-label.service
journalctl --user -u grana-label.service -n 50 --no-pager
```

Configure systemd user lingering with the host administrator if the service
must survive logout or start at boot. Foreground startup and SSH forwarding
remain sufficient for interactive review; see the
[labeling guide](human-labeling.md#configure-and-start).
