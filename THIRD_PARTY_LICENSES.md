# Third-party licenses

Grana's [Apache-2.0 license](LICENSE) covers its original contributions. It does
not replace third-party licenses or grant rights in separately obtained model
weights, datasets, services, tools, or dependencies. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices.

This inventory covers Grana's Python training, inference, cache, and annotation
source release. The public website maintains its own
[software, icon, and font inventory](https://github.com/reward-grana/reward-grana.github.io/blob/347feba36104f83bb9cd3f5bcd20dd2033df88b6/THIRD_PARTY_LICENSES.md)
for the assets distributed or loaded by that website.

## Dependencies installed separately

The source release contains dependency declarations, setup scripts, and Grana
code. It does not bundle the Python packages, CUDA libraries, interpreter,
model weights, fonts, or FFmpeg binaries listed below. Installing them creates
a separate local runtime; each component retains its own terms.

The following tables identify the direct dependencies declared in `pyproject.toml`,
`requirements/runtime.txt`, `requirements/annotation.txt`,
`requirements/hopper-tilelang.txt`, `requirements/inference-fast.txt`, and
`scripts/setup_env.sh`, plus PyArrow and packaging, which source and tests
import directly and obtain through other packages. The resolved environments
are recorded separately in `requirements/constraints.txt` and
`requirements/labeling-cpu.txt`; their transitive packages are not all expanded
into license rows below. Versions describe the declared environment where an
exact pin exists; `pyproject.toml` also permits broader version ranges. A
different resolved version must be checked against its own license files.

“License” identifies the upstream project license or package license
expression. It does not summarize every component embedded in a dependency's
binary wheel. The linked license and the actual installed artifact's license
and notice files are the evidence for that artifact.

### Training, inference, and cache packages

| Component | Qualified version or source revision | License | Upstream license |
|---|---|---|---|
| PyTorch (torch) | `2.10.0+cu130` | BSD-3-Clause | [License](https://github.com/pytorch/pytorch/blob/v2.10.0/LICENSE) |
| TorchVision | `0.25.0+cu130` | BSD-3-Clause | [License](https://github.com/pytorch/vision/blob/v0.25.0/LICENSE) |
| triton | `3.6.0` | MIT | [License](https://github.com/triton-lang/triton/blob/v3.6.0/LICENSE) |
| Transformers | `a29df2d916e3b820aecd19d3b5a877abc523ba3c` | Apache-2.0 | [License](https://github.com/huggingface/transformers/blob/a29df2d916e3b820aecd19d3b5a877abc523ba3c/LICENSE) |
| accelerate | `1.14.0` | Apache-2.0 | [License](https://github.com/huggingface/accelerate/blob/v1.14.0/LICENSE) |
| PEFT | `0.18.1` | Apache-2.0 | [License](https://github.com/huggingface/peft/blob/v0.18.1/LICENSE) |
| flash-linear-attention | `0.5.2` | MIT | [License](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/LICENSE) |
| fla-core | `0.5.2` | MIT | [License](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/LICENSE) |
| ray | `2.55.0` | Apache-2.0 | [License](https://github.com/ray-project/ray/blob/ray-2.55.0/LICENSE) |
| datasets | `4.1.1` | Apache-2.0 | [License](https://github.com/huggingface/datasets/blob/4.1.1/LICENSE) |
| PyArrow | `26.0.0` | Apache-2.0 | [License](https://github.com/apache/arrow/blob/apache-arrow-26.0.0/LICENSE.txt) |
| NumPy | `2.5.0` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [License](https://github.com/numpy/numpy/blob/v2.5.0/LICENSE.txt) |
| Pillow | `12.0.0` | MIT-CMU | [License](https://github.com/python-pillow/Pillow/blob/12.0.0/LICENSE) |
| PyYAML | `6.0.3` | MIT | [License](https://github.com/yaml/pyyaml/blob/6.0.3/LICENSE) |
| einops | `0.8.2` | MIT | [License](https://github.com/arogozhnikov/einops/blob/v0.8.2/LICENSE) |
| safetensors | `0.9.0rc1` | Apache-2.0 | [License](https://github.com/huggingface/safetensors/blob/659534d9cbd0f40b8463b2e090aee6d979ee8351/LICENSE) |
| huggingface-hub | `1.33.0` | Apache-2.0 | [License](https://github.com/huggingface/huggingface_hub/blob/v1.33.0/LICENSE) |
| tokenizers | `0.22.2` | Apache-2.0 | [License](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE) |
| protobuf | `5.29.6` | BSD-3-Clause | [License](https://github.com/protocolbuffers/protobuf/blob/v29.6/LICENSE) |
| W&B SDK (wandb) | `0.27.2` | MIT | [License](https://github.com/wandb/wandb/blob/v0.27.2/LICENSE) |

### Native inference kernel and its build-tool dependency

`requirements/inference-fast.txt` selects hash-pinned upstream wheels by CPU
architecture. `scripts/setup_env.sh` installs this profile; the package extra
in `pyproject.toml` declares the broader `>=1.7,<1.8` kernel range. These wheel
files and the code they contain are downloaded separately, not included in the
source release.

| Component | Qualified version or declared requirement | License | Upstream license |
|---|---|---|---|
| causal-conv1d | `1.7.0+cu13torch2.10cxx11abiTRUE` (CPython 3.12, x86_64 or aarch64) | BSD-3-Clause | [License](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/LICENSE) |
| Ninja Python distribution (ninja) | `1.13.2` | Apache-2.0 | [License](https://github.com/scikit-build/ninja-python-distributions/blob/1.13.2/LICENSE_Apache_20) |

Ninja is retained to satisfy the kernel wheel's dependency metadata; the
installer uses prebuilt causal-conv1d wheels rather than building that kernel.

### Optional annotation packages

| Component | Qualified version or declared requirement | License | Upstream license |
|---|---|---|---|
| httpx | `0.28.1` | BSD-3-Clause | [License](https://github.com/encode/httpx/blob/0.28.1/LICENSE.md) |
| jsonschema | `4.26.0` | MIT | [License](https://github.com/python-jsonschema/jsonschema/blob/v4.26.0/COPYING) |

### Hopper-only FLA backward packages

`scripts/setup_env.sh` installs these only for the `hopper-tilelang` profile.
The NVIDIA CUDA compiler wheels follow the NVIDIA CUDA-related wheel terms
described below.

| Component | Qualified version or declared requirement | License | Upstream license |
|---|---|---|---|
| TileLang (tilelang) | `0.1.15` | MIT | [License](https://github.com/tile-ai/tilelang/blob/v0.1.15/LICENSE) |
| apache-tvm-ffi | `0.1.12` | Apache-2.0 | [License](https://github.com/apache/tvm-ffi/blob/v0.1.12/LICENSE) |
| torch_c_dlpack_ext | `0.1.5` | Apache-2.0 | [Package](https://pypi.org/project/torch-c-dlpack-ext/0.1.5/) |
| cloudpickle | `3.1.2` | BSD-3-Clause | [License](https://github.com/cloudpipe/cloudpickle/blob/v3.1.2/LICENSE) |
| ml_dtypes | `0.6.0` | Apache-2.0 | [License](https://github.com/jax-ml/ml_dtypes/blob/v0.6.0/LICENSE) |
| z3-solver | `4.15.4.0` | MIT | [License](https://github.com/Z3Prover/z3/blob/z3-4.15.4/LICENSE.txt) |
| nvidia-cuda-nvcc, nvidia-cuda-crt, nvidia-nvvm | `13.0.88` | LicenseRef-NVIDIA-Proprietary | [CUDA Toolkit agreement](https://docs.nvidia.com/cuda/eula/index.html) |
| nvidia-cuda-cccl | `13.0.85` | LicenseRef-NVIDIA-Proprietary | [CUDA Toolkit agreement](https://docs.nvidia.com/cuda/eula/index.html) |

### Setup, build, and test tools

| Component | Qualified version or declared requirement | License | Upstream license |
|---|---|---|---|
| pip | `26.2.1` | MIT | [License](https://github.com/pypa/pip/blob/26.2.1/LICENSE.txt) |
| setuptools | `84.0.0` | MIT | [License](https://github.com/pypa/setuptools/blob/v84.0.0/LICENSE) |
| wheel | `0.48.0` | MIT | [License](https://github.com/pypa/wheel/blob/0.48.0/LICENSE.txt) |
| hatchling | `1.32.4` | MIT | [License](https://github.com/pypa/hatch/blob/hatchling-v1.32.4/backend/LICENSE.txt) |
| pytest | `9.1.1` | MIT | [License](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE) |
| build | `>=1 (optional; not pinned)` | MIT | [License](https://github.com/pypa/build/blob/main/LICENSE) |
| packaging | `26.3` (direct test import; resolved dependency) | Apache-2.0 OR BSD-2-Clause | [License selection](https://github.com/pypa/packaging/blob/26.3/LICENSE), [Apache-2.0](https://github.com/pypa/packaging/blob/26.3/LICENSE.APACHE), [BSD-2-Clause](https://github.com/pypa/packaging/blob/26.3/LICENSE.BSD) |

For Safetensors, the package version `0.9.0rc1` corresponds to upstream tag
`v0.9.0-rc.1`; the license link uses its immutable commit. The optional `build`
requirement has no exact pin, so its link identifies the current upstream
license rather than a qualified installed release.

### CPU labeling profile

`requirements/labeling-cpu.txt` records a separate CPU-only resolved
environment. In particular, it selects different versions for three packages
listed in the training table and adds the following HTTP and certificate
packages. These rows describe the published package license metadata for those
exact versions; they do not replace the licenses and bundled-component notices
inside an installed wheel.

| Component | CPU profile version | License | Upstream package evidence |
|---|---|---|---|
| NumPy | `2.5.4` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [Package](https://pypi.org/project/numpy/2.5.4/) |
| PyArrow | `26.0.1` | Apache-2.0 | [Package](https://pypi.org/project/pyarrow/26.0.1/) |
| huggingface-hub | `2.4.0` | Apache-2.0 | [Package](https://pypi.org/project/huggingface-hub/2.4.0/) |
| httpcore2 | `2.16.0` | BSD-3-Clause | [Package](https://pypi.org/project/httpcore2/2.16.0/) |
| httpx2 | `2.16.0` | BSD-3-Clause | [Package](https://pypi.org/project/httpx2/2.16.0/) |
| truststore | `0.10.5` | MIT | [License](https://github.com/sethmlarson/truststore/blob/v0.10.5/LICENSE) |

The CPU profile also retains `httpx==0.28.1` and `httpcore==1.0.9`; HTTPX2 and
HTTPCore2 are distinct package names. Other resolved dependencies and version
differences remain recorded in the CPU manifest.

### Transitive packages and binary components

`requirements/constraints.txt` and `requirements/labeling-cpu.txt` record
additional resolved dependencies. Their presence in those files does not mean
their contents are distributed with
Grana. The table above is not a complete binary distribution license inventory.
For a redistributed environment or container, retain and review the license
and notice files from the actual resolved artifacts, including components
embedded in wheels.

Particular distinctions in the qualified environment include:

- **NumPy 2.5.0:** the package metadata contains the compound expression shown
  above. Its binary license file also covers bundled OpenBLAS, LAPACK, GCC
  runtime, and libquadmath components under their respective terms.
- **Pillow 12.0.0:** the project license is MIT-CMU. The wheel additionally
  contains notices for its bundled image and font libraries; MIT-CMU alone is
  not a license inventory for the whole binary wheel.
- **PyTorch, Ray, PyArrow, Triton, and packaging tools:** these may contain
  additional third-party code and notices beyond their project-level license.
- **tqdm 4.70.1:** its [license](https://github.com/tqdm/tqdm/blob/v4.70.1/LICENCE)
  identifies MPL-2.0 and MIT portions. **certifi 2026.10.23** uses
  [MPL-2.0](https://github.com/certifi/python-certifi/blob/2026.10.23/LICENSE).
- **NVIDIA CUDA-related wheels:** the exact wheel's terms govern. Most of the
  qualified CUDA binary packages identify NVIDIA proprietary software terms;
  this must not be inferred from the license of a similarly named source
  repository. In particular, `nvidia-cusparselt-cu13==0.8.0` identifies NVIDIA
  Proprietary Software; see the [cuSPARSELt software agreement](https://docs.nvidia.com/cuda/cusparselt/license.html).
  `cuda-bindings==13.0.3` also carries NVIDIA software license terms. By
  contrast, `cuda-pathfinder==1.8.5` and `nvidia-nvtx==13.0.85` identify
  Apache-2.0. See also the [CUDA Toolkit agreement](https://docs.nvidia.com/cuda/eula/index.html)
  and each installed wheel's license files.

## Models, system tools, and services obtained separately

| Component | Terms and evidence | How Grana uses it |
|---|---|---|
| Qwen/Qwen3.5-0.8B | [Apache-2.0 model license](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE) | Downloaded by repository ID or loaded from a user-provided model directory. Weights are not bundled. |
| Qwen/Qwen3.5-2B | [Apache-2.0 model license](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE) | Separate model-loading workflow; this license link records the inspected immutable revision. Weights are not bundled. |
| Qwen/Qwen3.5-4B | [Apache-2.0 model license](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE) | Same separate model-loading workflow. Weights are not bundled. |
| Qwen/Qwen3.5-9B | [Apache-2.0 model license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE) | Separate model-loading workflow; this license link records the inspected immutable revision. Weights are not bundled. |
| Qwen/Qwen3.5-27B | [Apache-2.0 model license](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/LICENSE) | Separate model-loading workflow; this license link records the inspected immutable revision. Weights are not bundled. |
| FFmpeg and ffprobe | [FFmpeg licensing information](https://ffmpeg.org/legal.html): LGPL-2.1-or-later by default; enabling GPL components changes the applicable binary license. | Separately installed executables invoked as subprocesses for video decoding and rendering. The renderers require libx264, so their required FFmpeg build must not be described as unconditionally LGPL-only. |
| x264 / libx264 | [x264 licensing information](https://www.videolan.org/developers/x264.html): GPL, with commercial licensing available separately. | Codec supplied by the user's FFmpeg installation; no encoder binary is bundled. |
| DejaVu Sans | [Bitstream Vera, DejaVu, and Arev font notices](https://github.com/dejavu-fonts/dejavu-fonts/blob/version_2_37/LICENSE) | Inference video rendering reads a system font; `--font` can select a user-provided font under its own terms. No font files are bundled. |
| CPython 3.12.3 (CUDA) / 3.12.12 (CPU labeling) | [Python license and included-component notices](https://docs.python.org/3.12/license.html) | Setup copies an existing interpreter into the CUDA environment; the CPU labeling guide selects its own interpreter version. Neither interpreter is part of the source release. |
| uv | [MIT](https://github.com/astral-sh/uv/blob/main/LICENSE-MIT) OR [Apache-2.0](https://github.com/astral-sh/uv/blob/main/LICENSE-APACHE) | User-installed tool used to locate a managed Python interpreter. uv is not bundled. |
| Node.js | [MIT and bundled-component notices](https://github.com/nodejs/node/blob/main/LICENSE) | User-installed Node.js runs the JavaScript tests; browser smoke tests use Node.js 22+ built-in APIs and require no npm packages. Node.js is not bundled. |
| Chrome or Chromium | The selected browser's distribution terms and third-party notices apply; see the [Chromium source license](https://chromium.googlesource.com/chromium/src/+/main/LICENSE) for Chromium's project license. | User-installed browser launched by optional UI smoke tests. Chrome and Chromium binaries are not bundled, and Chromium's source license does not state all terms of a Chrome binary. |
| W&B service and NVIDIA inference API | The respective provider's service and access terms apply independently of client-library licenses. | Optional remote logging and annotation services. Grana does not distribute these services or the remote annotation model. |

The public configs do not pin immutable Qwen model revisions. Check the terms
attached to the exact revision downloaded, and any other model selected by the
user. Likewise, datasets and human annotations supplied to Grana retain their
own access and redistribution conditions; this source license does not grant
rights to them.
