# Train Grana reward models

Run commands from the source root in the [full CUDA environment](installation.md#full-cuda-environment).
Use the [project configuration guide](project-configuration.md#training-with-your-own-cache)
to create a recipe and cache manifest for your data. For new recordings, first
[build a cache](data.md). The [configuration reference](TRAINING_CONFIGS.md)
defines phase defaults, supervision, sampling and metrics.

## Validate and launch

Choose multitask pretraining (`pretrain`, A/B/C supervision) or task adaptation
(`posttrain`, B/C/D supervision). Adaptation can initialize from a same-size
Grana checkpoint or from the base backbone with `training.init_checkpoint: null`.
Set `GRANA_TRAIN_CONFIG` to your recipe's absolute path:

```bash
source .venv-qwen35-unsloth/bin/activate
: "${GRANA_TRAIN_CONFIG:?Set the absolute path to your training recipe}"
grana validate --config "$GRANA_TRAIN_CONFIG" --check-data
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" --dry-run

CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
  --config "$GRANA_TRAIN_CONFIG" \
  --set training.num_workers=1 \
  --set training.output_dir=/runs/my-task-001 \
  --set logging.mode=disabled
```

Review the cache inventory, role budget, initialization, held-out split and
evaluation call count before launching. Match `data.views` to the cache camera
aliases. Use a new output directory for each attempt and keep the foreground
command alive until it completes.

For two GPUs, select `CUDA_VISIBLE_DEVICES=0,1` and set
`training.num_workers=2`. Each DDP worker holds a complete model replica;
additional GPUs do not pool model memory. Automatic accumulation chooses the
forward count needed to meet the configured
[global role-row budget](TRAINING_CONFIGS.md#global-batch-and-actual-stage-supervision)
across the selected workers. Set both visible devices and worker count explicitly.
The [campaign launcher](TRAINING_SCRIPTS.md#run-a-campaign) coordinates explicit
pretraining and adaptation recipes across model sizes and GPU groups.

Each invocation starts a private local Ray cluster. The launcher selects
Ray Train V1, requests loopback addresses and disables the dashboard.
See [Ray runtime and online logging](installation.md#ray-runtime-and-online-logging)
for the runtime version and service boundaries.

For configuration-only checks in the CPU labeling environment:

```bash
python -m grana.cli validate --config "$GRANA_TRAIN_CONFIG"
python scripts/ray_migration/run_specs_cli.py \
  --config "$GRANA_TRAIN_CONFIG" --dry-run
```

These checks read configuration and manifests without loading cache payloads,
model weights or CUDA. Add `--check-data` to validate actual caches when available.

## Model sizes and training recipes

Qwen3.5 model-only overlays are available for
[0.8B](../configs/models/qwen35/0.8b.yaml),
[2B](../configs/models/qwen35/2b.yaml),
[4B](../configs/models/qwen35/4b.yaml),
[9B](../configs/models/qwen35/9b.yaml) and
[27B](../configs/models/qwen35/27b.yaml).
For Qwen3.8, use the separate [27B overlay](../configs/models/qwen38/27b.yaml).
Its official weights use the same native `Qwen3_5ForConditionalGeneration`
architecture; it retains its own tokenizer, model identity and checkpoint source.
Compose an overlay with your phase/data recipe. `model.name_or_path` and
optional `model.revision` select the backbone; a Grana checkpoint must match
its architecture. Choose a physical batch that fits the model, image grids,
sequence lengths and evaluation workload on each allocated GPU.

Both 27B training overlays select `model.delta_rule_backend: torch`.
Keep gradient checkpointing enabled and set `model.revision` to an immutable
model revision when freezing an experiment. A Qwen3.5 Grana checkpoint is not
a substitute for one trained on Qwen3.8, even when tensor shapes match.
Inference has separate [runtime requirements](inference.md#runtime-requirements).

## W&B and local evidence

Set `logging.mode` to `disabled`, `offline` or `online`. Disabled logging keeps
local records; offline logging writes local W&B data for later `wandb sync`.
For online logging, configure your entity/project and authenticate with
`wandb login` or `WANDB_API_KEY`, then replace the launch override with
`--set logging.mode=online`. Keep credentials out of recipes. Online logging
errors fail the run rather than changing its mode.

W&B uses `optimizer_steps` as its x-axis. Updates report losses, role
contributions, learning rate, gradient norm, role-row counts, timing and memory.
Evaluation logs compact quality/coverage/timing curves and an
`evaluation-details` artifact containing full diagnostics and predictions;
see [metric definitions](TRAINING_CONFIGS.md#evaluation-interpretation).
`logging.upload_checkpoint_weights` defaults to `false`: checkpoint artifacts
contain metadata, and uploading weight tensors requires explicitly enabling it.

| Local output | Contents |
|---|---|
| `config.json`, `resolved_config.json` | Effective configuration |
| `supervision_budget.json`, `data_summary.json`, `initialization.json` | Role budget, data inventory and initialization |
| `training_metrics.jsonl`, `sampling_coverage.json` | Update metrics and actual independent A/B/C exposure |
| `evaluation_plan.json`, `evaluation-XXXXXX.json`, `evaluation_metrics.jsonl` | Fixed evaluation inputs, predictions, metrics and timing |
| `checkpoint-XXXXXX/` | Adapter/head/marker weights and metadata |
| `result.json`, `ray_run.json`, `wandb_run.json` | Completion, runtime and enabled W&B records |

Optional render and dense plans, results and media are saved beside these
records. Keep data, weights and run outputs outside the source checkout.
Use the [status and report commands](TRAINING_SCRIPTS.md#watch-and-wait) to
inspect a running campaign or verify completed local and online outputs.

## Evaluation and checkpoint semantics

Configure disjoint held-out caches and enable evaluation in your recipe.
The selected plan stays fixed throughout the run. Choose sampled or dense
evaluation, its batch size and media settings using the
[evaluation reference](TRAINING_CONFIGS.md#evaluation-interpretation).
An A-only holdout cannot assess C stage accuracy; sparse POIs may miss stages.

Train-time evaluation uses all allocated workers. Rank 0 combines records,
renders and logs before training continues. Set
`training.distributed_timeout_seconds` (default `3600`) to accommodate the
configured evaluation event. MP4 rendering requires FFmpeg with libx264;
`GRANA_FFMPEG` can select the executable.

Checkpoints contain language LoRA, complete progress/temporal heads, and the
four added input/output marker embedding rows. Loading initializes **weights
only**: optimizer, scheduler, RNG, sampler and step counters start fresh.
Architecture-incompatible loads fail. Complete matching heads load even when
loss coefficients change; `training.allow_fresh_heads` applies only to
missing, incomplete or shape-incompatible heads. A run records success only
after its automatic parameter-update and checkpoint-restore checks pass.

To evaluate existing weights without optimizer updates, use
[single-checkpoint evaluation](TRAINING_SCRIPTS.md#evaluate-a-saved-checkpoint)
or [multi-GPU checkpoint suites](checkpoint_evaluation_suites.md). Both write
separate outputs.

## Runtime settings and timing

`model.gradient_checkpointing` defaults to `true`. Its non-reentrant
implementation lowers activation memory by recomputing intermediates during
backward. Setting it to `false` avoids this recomputation and needs more
activation memory. Saved recipes retain their explicit setting.

`training.prefetch_samples` defaults to `true`. One CPU thread per worker
prepares the next update's independent A/B/C examples while the current update
runs. The bounded queue holds at most one future local update; allow host RAM
for its raw RGB images in addition to the current update. This buffer uses the
source image sizes, before the model's pixel-budget resize.

Prefetch performs no CUDA work and preserves sampling order, pixels and targets.
D is constructed on the training thread from its matching C prediction and the
same observations. Set `training.prefetch_samples=false` to disable it, for
example when host memory is limited. GPU preprocessing and vision attention
are separate opt-in settings in the
[configuration reference](TRAINING_CONFIGS.md#shared-gpu-image-resizing).

Optional profiling records the first N real optimizer updates:

```bash
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" \
  --set training.output_dir=/runs/my-profile-001 \
  --set training.profile_steps=10
```

`training.profile_steps` defaults to `0`. It neither changes
`training.max_steps` nor discards warmup updates.

| Metric | Timing boundary |
|---|---|
| `time/training_step` / `update_seconds` | One synchronized optimizer update: sampling, preprocessing, transfers, all accumulation forwards/backwards, optimizer and distributed reductions |
| `time/training_seconds` / `training_compute_seconds` | Sum of those update durations |
| `time/elapsed_seconds` / `engine_elapsed_seconds` | Worker training-loop elapsed time, including previous evaluation and persistence; model startup is separate |
| `time/checkpoint_and_logging_seconds` | Checkpoint save, log submission and distributed checkpoint barrier, outside update time |
| `eval/elapsed_seconds` | Evaluation scoring wall time |
| `eval/evaluation_and_persistence_seconds` | Scoring, rendering, local writes and media preparation; excludes final artifact enqueue, history commit and asynchronous cloud upload |
| `time/profile/sample_host_seconds` | Synchronous sampling/RGB materialization, or consumer wait/dequeue when prefetch is enabled |
| `time/profile/sample_prepare_host_seconds` | Prefetch producer work overlapping model execution |
| `time/profile/collate_and_transfer_host_seconds` | Serialization, processor work and host-to-device transfer |
| Other `time/profile/*` fields | Host durations and CUDA event spans for forward, backward, routing, optimizer and metric reduction |

Update time excludes the subsequent checkpoint, evaluation, W&B and Ray report
work. Host durations and CUDA spans overlap; do not add them. CUDA event spans
include dispatch gaps and are not pure kernel time. Prefetch preparation also
overlaps update work. Parallel worker/job durations do not sum to elapsed wall
time.
