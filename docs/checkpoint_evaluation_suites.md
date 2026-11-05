# Evaluate a saved checkpoint on multiple GPUs

`scripts/training/evaluate_checkpoint_suites.py` loads the same saved adapters
and heads once per GPU, then evaluates one or more explicit held-out cohorts.
It makes no optimizer updates. Each suite gets its own new W&B run; the source
training run and checkpoint stay unchanged. An unfinished training run is a
valid source once its checkpoint save receipt exists.

Use the checkout-owned `.venv-qwen35-unsloth` environment. The source run needs
`checkpoint-STEP/metadata.json`, `weights.safetensors`, and the corresponding
save receipt in `training_metrics.jsonl`. Its W&B receipt is optional. Supply
a resolved training config with the same backbone revision and adapter
architecture, the training inventory for held-out overlap checks, and the
desired `logging.mode`, entity and project. Online mode requires W&B login;
media rendering requires ffmpeg with libx264.

## Define the evaluation suites

Save a JSON list such as `suites.json`:

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
  an integer selects a deterministic, capped cohort **per task**.
- `frame_stride` sweeps real POIs at that source-frame interval and includes
  the final frame. Full evaluation means all trajectories at the stated stride;
  use `frame_stride: 1` to score every source frame.
- `points_per_trajectory` instead selects evenly spaced real POIs including
  both endpoints. Short trajectories use every frame once. Specify a budget
  of at least 2, and omit `frame_stride` for this mode.
- `batch_size` is a physical window count **per GPU**, default 64. C and its
  predicted-stage-conditioned D stay together on the same worker. Set this
  explicitly for other sizes; this tool does not run a batch-size benchmark.
- `seed` and `stride_pairs` inherit the evaluation config and can be overridden
  per suite. Training sampling/augmentation does not select evaluation frames.
  Evaluation uses the canonical forward order. Task-level prohibitions on D
  still apply; choose an A progress route for those tasks.
- Roles default to C+D. A-only tasks require `roles: ["A"]`. Include B in
  `roles` and set `include_oracle: true` to add the oracle diagnostic. Every
  trajectory needs an A or C+D progress route for media generation.

Cache entries use the regular data-cache schema, including `include_ids` and
`exclude_ids`. The planner checks overlap against the supplied training
inventory and rejects incompatible role coverage rather than silently skipping
tasks. The same settings and inputs reproduce the same cohort and POIs.

## Prepare, review, then execute

The example selects GPU 2/3. Without `CUDA_VISIBLE_DEVICES`, all visible GPUs
are used. `--num-workers` can limit their count; it overrides the training
worker count only for this evaluation. Workers are isolated by UUID and each
uses local `cuda:0`. Batch sizes, parent logical IDs and UUIDs are recorded.

```bash
source .venv-qwen35-unsloth/bin/activate
export CUDA_VISIBLE_DEVICES=2,3

python scripts/training/evaluate_checkpoint_suites.py \
  --config /runs/source/resolved_config.json \
  --source-run /runs/source --step 250 \
  --specs /runs/suites.json --prepared /runs/prepared.json \
  --output /runs/checkpoint250-evaluation --num-workers 2 --prepare-only
```

Preparation reads metadata only: no model loading, inference or W&B run.
Explicit `--num-workers` also permits preparation on a CPU-only host. Inspect
the printed task, trajectory, window, role-row and shard counts. Execute the
same command **without `--prepare-only`**. Use a fresh output directory outside
the source run. Keep the prepared file outside the source run too. Changes to
cohort settings, data or worker count require preparing again.

All ranks score disjoint batches, including a possible final partial batch.
If there are fewer batches than ranks, surplus ranks contribute no rows.
Rank zero merges ordered records, recomputes aggregate metrics, renders from
saved predictions and uploads media. Metrics and media share the source
checkpoint's explicit optimizer step. Progress uses the existing compact
logging schema, including `predicted_stage_route_accumulated_progress_poi_micro_raw_rmse`
for C→D routes; A metrics describe whole-task progress instead.

Outputs include resolved configs/specs, plans, GPU setup receipts, per-rank
progress and shards, each suite's `evaluation.json`, media, `wandb_run.json`
and `report.json`, plus the aggregate `suite_reports.json` and `driver.json`.
Reports separate scoring/merge wall time, rendering and cloud completion;
summed worker compute spans are not elapsed wall time.

Success requires every suite's completed worker report and finished logging
receipt. Ctrl-C/SIGTERM stop this attempt's private Ray workers and return a
nonzero exit. The driver records `cancelled`, `incomplete` or `failed` rather
than treating a normal Ray return as proof of completion. After shutdown it
tries to close unfinished online runs, marking evaluation incomplete; W&B
shows a failed run for cancellation. Cloud cleanup errors remain in
`driver.json`. Partial records stay available, but are not full-evaluation
results. A new attempt uses a new output directory.
