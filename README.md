# Grana

Grana is a compositional reward model for long-horizon robot manipulation
([project page](https://reward-grana.github.io/)). This repository trains and
runs Qwen3.5 reward models from **cached multi-camera RGB trajectories**:
pretrain across tasks, then adapt to a target task. Use existing Grana caches,
or build them from annotated **LeRobot v2** episodes. Each model size trains
independently and loads only same-size checkpoints.

Model configurations cover Qwen3.5 0.8B, 2B, 4B, 9B and 27B. Training and
hardware validation have different scopes; see the
[model guide](docs/training.md#model-sizes-and-training-recipes) and
[reference results](docs/results.md).

## Install

Follow the [installation guide](docs/installation.md) for a source checkout or
release archive. It covers system prerequisites, CPU labeling, the full CUDA
runtime, RTX 4090 recommendations and checks that need no private data or task
configuration. For labeling alone, use the
[CPU setup](docs/installation.md#cpu-labeling-and-cache-construction); no GPU,
model checkpoint or API key is needed.

The qualified full-training environment is Linux aarch64, Python 3.12,
CUDA 13, and NVIDIA GB300.
The same pinned runtime has installation and CUDA-check validation on RTX 4090
and RTX 5090.
After installing the system prerequisites and uv, run from the source root:

```bash
uv python install 3.12.3
bash scripts/setup_env.sh
source .venv-qwen35-unsloth/bin/activate
python scripts/check_env.py
```

The checkout owns its environment; runtime packages and resolved dependencies
are pinned in `requirements/`. The inherited environment name contains
`unsloth`; execution uses native Transformers, PEFT, and Ray.
[Hardware guidance](docs/installation.md#hardware-and-capacity) distinguishes
installation checks from training capacity and deployment performance.

Cache building needs `ffmpeg` and `ffprobe` on `PATH`; training recipes that
render MP4 evaluations need ffmpeg with libx264. Run `wandb login` for online
logging, or set `logging.mode=disabled`. The [annotation guide](docs/ANNOTATION.md#install-and-configure)
covers its optional dependencies and API setup.

## Workflow

The CPU-only environment provides `python -m grana.cli` in place of the
`grana` console command.

| Step | Command | Guide |
|---|---|---|
| Label subtasks, outcomes and action quality by hand | `grana label` | [Human labeling](docs/human-labeling.md) |
| Propose and review stages (optional) | `grana annotate`, `grana review-annotations`, `grana export-stages` | [Annotation](docs/ANNOTATION.md) |
| Build a cache from reviewed annotations | `grana build-cache` | [Data and target construction](docs/data.md) |
| Validate and train | `grana validate`, `bash scripts/train.sh` | [Training](docs/training.md), [phase configuration](docs/TRAINING_CONFIGS.md) |
| Run and monitor explicit training recipes | `bash scripts/training/run_campaign.sh` | [Training scripts](docs/TRAINING_SCRIPTS.md) |
| Score a checkpoint locally or over HTTP | `grana infer`, `python -m grana.inference.server` | [Inference](docs/inference.md) |
| Compare frame readouts across evaluation trajectories | `scripts/inference/evaluate_frames.py` | [All GPUs, batch 64 per GPU](docs/inference.md#compare-poi-and-last-across-an-evaluation-cohort) |

Stage proposals alone are not reward supervision: cache construction also needs
explicit action-quality and outcome labels. The [annotation skill](skills/grana-robot-annotation/SKILL.md)
provides proposal methods and synthetic examples; task definitions need review.

For unannotated recordings, create a
[standalone labeling recipe](docs/project-configuration.md#labeling-your-own-recordings)
outside the source tree. Supply your own task, stages, cameras and data paths;
open the local editor, submit reviewed episodes, then build a cache with the
same active recipe. The [human-labeling guide](docs/human-labeling.md) covers
review controls. No bundled dataset or example YAML is required.

## Quick start

After installation, follow [Configure a project](docs/project-configuration.md)
to create a private training recipe and cache manifest. The guide contains
standalone schemas, so neither installation nor this workflow depends on the
repository's example task configurations. Set `GRANA_TRAIN_CONFIG` to your
recipe's absolute path and use the full CUDA environment:

```bash
: "${GRANA_TRAIN_CONFIG:?Set the absolute path to your training recipe}"
grana validate --config "$GRANA_TRAIN_CONFIG" --check-data
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" --dry-run

# After reviewing the settings and qualifying capacity on your GPU:
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG"
```

Match `data.views` to the camera aliases in your caches and use new output
directories. Configure disjoint held-out data before quality experiments.
Choose pretraining or posttraining in your own recipe; posttraining can start
from a same-size checkpoint or the base backbone with
`training.init_checkpoint: null`. See [training](docs/training.md) for the
semantics and [inference](docs/inference.md) to score your resulting checkpoint.
Keep gradient checkpointing enabled for training, smoke tests, canaries,
profiling, memory/timing benchmarks, and batch-size/accumulation comparisons.
Checkpointing-off comparisons require an explicit user request; see the
[training guide](docs/training.md).

## Prediction roles

Each request supplies eight ordered observations per camera. The local and
HTTP client CLIs default to **POI (slot 5, third from last)**,
unless input JSON explicitly selects a frame. Direct Python requests, the
HTTP benchmark and the review browser use the same default. Use
`--prediction-frame last` to select slot 7. JSON always includes all eight slots.

| Role | Text input | Prediction |
|---|---|---|
| A | Whole-task description | Whole-task progress in [0,1] |
| B | Task, ordered stages, explicit focus stage | Progress relative to that focus |
| C | Task and ordered stages | Current stage |
| D | Task and ordered stages | C's stage at the selected input slot, then progress relative to it |

Primary deployment quality uses **C→D accumulated progress RMSE**, reported in
raw and normalized units. B/D focus-relative RMSE remains diagnostic; see
[evaluation semantics](docs/training.md#evaluation-and-checkpoint-semantics).

Inference has one [supported runtime](docs/inference_optimizations.md): CUDA
image resize, the required FLA/SDPA acceleration stack, and eight Torch CPU
threads. HTTP requests use lossless binary RGB; JSON transport and runtime
mode switches are not supported. Missing kernels or unsupported inputs raise
errors instead of selecting a slower inference implementation. The HTTP
server keeps one inference worker alive across connections.

C,D are the default outputs across the Python API, CLIs and review browser.
Explicit A or B outputs add language forwards. C and D run independent language
forwards and share only request-local image work.

Use the [benchmark guide](docs/benchmarks.md) to measure your checkpoint and
request workload. Latency comparisons require matching roles, frame readout,
input geometry, warmup, and timing boundaries; hardware names alone do not
make two results comparable.

## Package layout

```text
grana/
├── annotation/       # Human labels, evidence, API proposals, review and export
├── data/             # Cache builder, reader, role sampling and target policies
├── modeling/         # Qwen/LoRA model, prompts, inputs and checkpoints
├── training/         # Ray worker, losses, held-out evaluation and logging
├── inference/        # Predictor, request CLI and HTTP client/server
├── cli.py            # The grana command
├── config.py         # Recipe loading and validation
└── runtime.py        # Checkout environment and Ray worker checks
```

Historical flat imports and module entry points remain supported. See
[package organization](CONTRIBUTING.md#package-organization) for domain APIs
and compatibility details.

## Development

```bash
python -m pip install -r requirements/annotation.txt
python -m pytest tests
python scripts/package_release.py --output dist/grana-source.tar.gz
```

The source archive excludes datasets, model weights, environments, credentials,
and local run artifacts. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete
workflow, source-header conventions, and required DCO sign-off.

## License and contributions

Original Grana contributions are licensed under [Apache-2.0](LICENSE),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Project attribution is in
[NOTICE](NOTICE). Third-party material retains its own terms:
[third-party notices](THIRD_PARTY_NOTICES.md) document attribution, and
[third-party licenses](THIRD_PARTY_LICENSES.md) identify the training, inference,
and annotation dependencies and their scope.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.
