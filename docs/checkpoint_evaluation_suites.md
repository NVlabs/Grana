# Evaluate a saved checkpoint on multiple GPUs

`scripts/training/evaluate_checkpoint_suites.py` loads saved adapters and heads
once per GPU, then evaluates explicit held-out cohorts with zero optimizer
updates. Each suite gets separate outputs and, when online logging is enabled,
a new W&B run. An unfinished training run is a valid source after its
checkpoint save receipt exists.

Use the [full CUDA environment](installation.md#full-cuda-environment).
The source run needs `checkpoint-STEP/metadata.json`, `weights.safetensors`
and its save receipt in `training_metrics.jsonl`. Supply a resolved config with
the same backbone revision and adapter architecture, the training inventory
for held-out overlap checks, and your logging settings. Online mode needs W&B
authentication; media rendering needs FFmpeg with libx264.

## Define the evaluation suites

Save a JSON list such as `/runs/suites.json`:

```json
[
  {
    "name": "full_A",
    "caches": [{"path": "/data/holdout_A", "roles": ["A"]}],
    "roles": ["A"],
    "max_trajectories_per_task": null,
    "batch_size": 64,
    "frame_stride": 10
  },
  {
    "name": "bounded_CD",
    "caches": [{"path": "/data/holdout_staged", "roles": ["B", "C"]}],
    "roles": ["C", "D"],
    "max_trajectories_per_task": 2,
    "batch_size": 64,
    "points_per_trajectory": 32
  }
]
```

- `max_trajectories_per_task: null` selects all eligible held-out trajectories;
  an integer caps the deterministic cohort **per task**.
- `frame_stride` scores real POIs at that source-frame interval plus the final
  frame. Use `1` to score every source frame.
- `points_per_trajectory` instead selects evenly spaced real POIs including
  endpoints. Use at least `2` and omit `frame_stride`; short trajectories use
  every frame once.
- `batch_size` is a physical window count **per GPU**, default `64`. Set it for
  your workload and memory capacity. C and its conditioned D stay together.
- `seed` and `stride_pairs` inherit the evaluation config and can be overridden.
  Evaluation uses regular/forward geometry and native stages, independently
  of training sampling.
- Roles default to C+D. A-only tasks require `roles: ["A"]`. Include B and set
  `include_oracle: true` for oracle diagnostics. Media require an A or C+D
  progress route.

Cache entries accept the standard `include_ids` and `exclude_ids` fields.
Planning rejects overlap with the training inventory and incompatible role
coverage. The same inputs and settings reproduce the cohort and POIs.

## Prepare, review, then execute

Select GPUs explicitly. Without `CUDA_VISIBLE_DEVICES`, the tool uses all
visible GPUs. `--num-workers` limits their count for this evaluation,
independently of the training worker count.

```bash
source .venv-qwen35-unsloth/bin/activate
export CUDA_VISIBLE_DEVICES=0,1

python scripts/training/evaluate_checkpoint_suites.py \
  --config /runs/source/resolved_config.json \
  --source-run /runs/source --step 250 \
  --specs /runs/suites.json --prepared /runs/prepared.json \
  --output /runs/checkpoint250-evaluation --num-workers 2 --prepare-only
```

Preparation reads metadata without model loading, inference or a W&B run.
An explicit worker count also permits preparation on a CPU-only host. Inspect
the task, trajectory, window and role-row counts, then run the same command
without `--prepare-only`. Use a fresh output directory and keep both it and
the prepared file outside the source run. Prepare again after changing data,
cohort settings or worker count.

Workers score disjoint batches and aggregate records before computing global
metrics; they do not average per-worker RMSE. Rendering reuses saved
predictions. Metrics and media carry the source checkpoint's optimizer step.
See the [evaluation definitions](TRAINING_CONFIGS.md#evaluation-interpretation)
for C→D accumulated progress, A-only progress and conditional diagnostics.

Each suite saves `evaluation.json`, media, logging records and `report.json`;
`suite_reports.json` and `driver.json` summarize the attempt. Reports separate
scoring/merge wall time, rendering and cloud completion. Summed worker compute
is not elapsed wall time.

A successful command requires every suite to finish evaluation and logging.
Cancellation/failure returns nonzero and leaves partial records for inspection;
these are not completed evaluation results. A new attempt needs a new output
directory.
