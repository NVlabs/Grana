# Train Grana reward models

Run commands from the source root after [installation](installation.md).
Use the [project configuration guide](project-configuration.md#training-with-your-own-cache)
to create a standalone recipe and cache manifest for your data. For new
recordings, first [build a cache](data.md). The
[configuration guide](TRAINING_CONFIGS.md) defines phase defaults and valid keys.

## Run the complete workflow

Choose **multitask pretraining** (`pretrain`, A/B/C supervision) or
**task adaptation** (`posttrain`, B/C/D supervision). The phase selects
supervision defaults. Posttraining can load a same-size Grana checkpoint or
start from the base backbone with `training.init_checkpoint: null`.
`model.name_or_path` and optional `model.revision` select the base model.

Set `GRANA_TRAIN_CONFIG` to your recipe's absolute path. Review the cache
inventory, held-out split, role coverage and evaluation call count before launch:

```bash
: "${GRANA_TRAIN_CONFIG:?Set the absolute path to your training recipe}"
grana validate --config "$GRANA_TRAIN_CONFIG" --check-data
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" --dry-run

CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
  --config "$GRANA_TRAIN_CONFIG" \
  --set training.num_workers=1 \
  --set training.output_dir=/runs/my-task-001 \
  --set logging.mode=disabled
```

Use a new output directory for each attempt. Match `data.views` to your cache
camera aliases and `training.num_workers` to the GPUs allocated by
`CUDA_VISIBLE_DEVICES`. Automatic accumulation preserves the configured global
role budget when worker count changes. Keep the foreground command alive until
it completes. Use [training scripts](TRAINING_SCRIPTS.md) to coordinate explicit
pretraining and adaptation recipes across model sizes.

For configuration-only checks in the CPU labeling environment, use the module
CLI and direct dry-run entrypoint from the source root:

```bash
python -m grana.cli validate --config "$GRANA_TRAIN_CONFIG"
python scripts/ray_migration/run_specs_cli.py \
  --config "$GRANA_TRAIN_CONFIG" --dry-run
```

These checks read configuration and manifests without loading cache payloads,
model weights or CUDA. Add `--check-data` to `validate` when the actual caches
are available. Executing training still requires the full CUDA environment.

Gradient checkpointing is **enabled by default**
(`model.gradient_checkpointing: true`) for training, smoke tests, canaries,
profiling, memory measurements and throughput benchmarks. Keep it enabled for
physical-batch and accumulation comparisons. Checkpointing trades recomputation
for lower activation memory; disable it only for an explicitly requested
comparison. See [training efficiency](training_efficiency.md#gradient-checkpointing-default-and-tradeoff).

### Model sizes and training recipes

Model-only overlays are available for
[0.8B](../configs/models/qwen35/0.8b.yaml),
[2B](../configs/models/qwen35/2b.yaml),
[4B](../configs/models/qwen35/4b.yaml),
[9B](../configs/models/qwen35/9b.yaml) and
[27B](../configs/models/qwen35/27b.yaml).
Compose the appropriate overlay with your phase/data recipe; model sizes train
independently and Grana checkpoints must match the selected architecture.

The 27B training overlay selects `model.delta_rule_backend: torch`. Keep this
selection until its accelerated backward path is separately qualified.
Inference uses its own required guarded-FLA runtime, regardless of the training
backend; see [inference runtime](inference_optimizations.md).

Reference 0.8B/4B runs completed 400 pretraining and 200 adaptation updates;
2B/9B/27B have short execution checks. Those runs do not validate the current
5,000/200-update defaults, convergence on new data, or every physical batch.
See [reference results](results.md#reference-training-results) for their scope.
Each DDP worker holds a complete model replica. Qualify memory using the actual
image grids, sequence lengths, evaluation and target GPU.

### Phase budgets

The phase defaults use learning rate **2e-5**, target global batch **128**,
and automatic accumulation. The H100-oriented pretraining recommendation uses
physical batch **32** per device; adaptation uses **8**. These are starting
recipes, not validated optima for a new task.

| Default | Pretraining | Adaptation |
|---|---:|---:|
| Optimizer updates | 5,000 | 200 |
| Global A / B / C / D role rows per update | 40 / 40 / 48 / 0 | 0 / 80 / 24 / 24 |
| Independent C windows per update | 48 | 24 |
| Warmup updates | 100 | 10 |
| Save interval | 500 | 25 |
| Evaluation interval, when enabled | 500 | 50 |

D reuses C observations, so role rows do not count unique windows. Each window
has eight supervised slots that may include repeated boundary frames.
`grana validate`, `--dry-run` and `supervision_budget.json` expose the actual
denominators. At global batch 128, pretraining accumulation is 4/2 for 1/2
workers; adaptation accumulation is 16/8/4/2 for 1/2/4/8 workers. For BS32
pretraining on 4/8 workers, set global batch to 256/512 for accumulation 2.
The final update is saved/evaluated once when those outputs are enabled.

Use [custom pretraining](../configs/pretraining/custom.yaml) or
[custom adaptation](../configs/posttraining/custom.yaml) as small examples, or
write a standalone recipe. Supply your own cache inventory and disjoint
held-out split. Evaluation and online logging must be explicitly configured.
Task sampling weights are separate from role batch counts; inspect
`sampling_coverage.json` for actual exposure. Unknown configuration keys fail.

## What Ray Train owns

The standard training entrypoints use this path:

```text
scripts/train.sh → run_specs_cli.py → submit_local_ray_job.py
                → train_ray.py → TorchTrainer workers → grana.training.engine
```

Each invocation starts its own local Ray cluster and one CUDA worker per GPU.
Distributed workers synchronize model gradients with DDP. The driver never
attaches to or stops another Ray cluster. All ranks evaluate disjoint whole
batches using their existing unwrapped models. Rank 0 writes checkpoints,
merges evaluation records, produces media, and logs W&B before all workers
continue at the same update boundary. The process-group
timeout is configurable with `training.distributed_timeout_seconds` (3600).
Measure the full configured evaluation event before a long distributed run so
that its waiting period and overall runtime fit your budget.

`ray_run.json` records the driver, runtime directory, process ownership,
code hashes, and terminal status. `worker-*.json` records worker process and GPU
identity. Keep the foreground driver/session alive. An interrupted observer is
not proof that training failed: inspect these exact processes and the latest
metrics before restarting. Runs require a **new output directory**; existing
checkpoints are never overwritten. Repeating `--config` executes independent
recipes sequentially with the supplied overrides applied to every recipe.

## W&B and local evidence

Set `logging.mode` to `disabled`, `offline` or `online`. Disabled logging keeps
local records; offline logging writes local W&B data for later `wandb sync`.
For online logging, configure your own entity/project and authenticate with
`wandb login` or `WANDB_API_KEY`. Credentials do not belong in recipes. Online
logging errors fail the run rather than changing its mode.

The W&B x-axis is `optimizer_steps`. Updates report loss, per-role weighted
contributions, learning rate, gradient norm, role-row counts, timing and memory.
Evaluation emits compact quality/coverage/timing curves and an
`evaluation-details` artifact containing full predictions, diagnostics and
provenance. The primary metric names and aggregation are defined below.

Optional `evaluation.deploy_dense` diagnostics score a fixed held-out cohort
at `frame_stride` intervals plus its final frame. Configure its trajectory cap,
`include_oracle`, `max_frames`, `replay_fps` and `group_by_cache` explicitly.
A null trajectory cap includes every trajectory admitted by the plan. Best/worst
selection is within the recorded group at that evaluation step. Media use
actual predictions without smoothing; sampled-POI playback is not original-speed
video. MP4 encoding requires FFmpeg with libx264 (`GRANA_FFMPEG` can select it).
Optional `evaluation.renders` provides a separate PNG/GIF diagnostic cohort.

For a saved checkpoint, use [checkpoint evaluation](TRAINING_SCRIPTS.md#evaluate-a-saved-checkpoint)
or [checkpoint suites](checkpoint_evaluation_suites.md). These perform no
optimizer updates and write separate outputs.

Timing fields describe distinct boundaries:

- `time/training_step` / `update_seconds`: one synchronized optimizer update,
  including data preparation and forward/backward work.
- `time/training_seconds` / `training_compute_seconds`: sum of update times.
- `time/elapsed_seconds` / `engine_elapsed_seconds`: worker elapsed time,
  including evaluation and persistence; model startup has a separate cost.
- `eval/elapsed_seconds`: evaluation scoring wall time.
- `eval/evaluation_and_persistence_seconds`: scoring, rendering and persistence,
  excluding final artifact enqueue, history commit and asynchronous upload.

Parallel job durations do not sum to campaign wall time. To diagnose update
cost, set `training.profile_steps=10`; the default `0` disables detailed spans.
`training.max_steps` still determines the run budget. Optional
`training.prefetch_samples=true` overlaps next-update CPU sampling with GPU
work. Keep checkpointing enabled and measure the actual workload; see
[training efficiency](training_efficiency.md) and
[benchmark methodology](benchmarks.md#evaluation-cost).

Keep these local records with each run:

| File | Contents |
|---|---|
| `config.json`, `resolved_config.json` | Effective configuration |
| `supervision_budget.json` | Global role rows and independent windows |
| `data_summary.json`, `initialization.json` | Data inventory and initialization provenance |
| `training_metrics.jsonl` | Per-update losses and timing |
| `evaluation_plan.json`, `evaluation-XXXXXX.json` | Fixed inputs, predictions, targets, metrics and timing |
| `evaluation_metrics.jsonl` | Compact evaluation events |
| `render_plan.json`, `render-evaluation-XXXXXX.json`, `eval_renders/` | Optional render cohort and media |
| `deploy_dense_plan.json`, `deploy-dense-evaluation-XXXXXX.json`, `deploy_dense/` | Optional dense predictions, media and selection provenance |
| `sampling_coverage.json` | Actual independent A/B/C exposure; D inherits C |
| `checkpoint-XXXXXX/` | Adapter/head/marker weights and metadata |
| `result.json` | Completion, evaluation events and checkpoint verification |
| `wandb_run.json` | W&B identity and artifact/upload receipts, when enabled |

Keep data, weights, runtime outputs and credentials outside source releases.

## Evaluation and checkpoint semantics

Held-out identities and resolved RGB payload paths must be disjoint from
training before any trajectory cap is applied. Every held-out task participates.
Trajectories are selected deterministically. With `evaluation.mode: sampled`
(the generic default), POIs are evenly spaced interior positions: two per
pretraining trajectory and eight for adaptation. Sparse POIs do not guarantee
coverage of every stage. The selected plan stays fixed throughout the run.

`evaluation.mode: dense` makes the dense plan the primary evaluation plan and
reuses its predictions for dense metrics and media. It requires enabled
evaluation and `evaluation.deploy_dense`.
Compare results using their recorded population, mode, and frame selection.

`evaluation.batch_size` limits held-out windows scored together per role. It
defaults to 1; set it for your workload and GPU capacity. Batching preserves
planned windows, eight-slot
records, and C-to-D focus routing. `model_calls` counts logical role rows;
`forward_calls` counts actual batched model invocations. Evaluation batching
does not change the training batch or gradient accumulation.
`training.batched_vision_sdpa` and `evaluation.batched_vision_sdpa` are independent
flags, both false by default. Their guarded frozen-vision path
and fallbacks are described in [batch size and vision attention](TRAINING_CONFIGS.md#batch-size-and-vision-attention).
They do not change the gradient-checkpointing policy.

During training, all `training.num_workers` ranks score disjoint whole batches.
Batch membership, C-to-D pairing, the final partial batch, and record order are
preserved; surplus ranks receive empty shards. Rank 0 recomputes global metrics
from the merged records, rather than averaging rank-level RMSE or task means.
This applies to primary, optional render, and dense scoring. Shards are stored
under `evaluation_shards/step-STEP/PURPOSE/rank-RANK.json` in the shared output
directory; collectives carry small status/digest messages. Execution-world-size,
timing, and memory receipts describe the actual run. See
[evaluation distribution](TRAINING_CONFIGS.md#batch-size-and-vision-attention).

Primary deployment quality is **predicted-stage C→D accumulated progress RMSE**
at POI slot 5: compare `predicted C stage + D progress` with
`true stage + true within-stage progress`. Report both raw stage units and
normalized RMSE, dividing each POI error by its trajectory's stage count before
aggregation. The W&B curves explicitly name this route:
`eval/overall/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse` and
`eval/overall/predicted_stage_route_accumulated_progress_poi_micro_normalized_rmse`.
Full evaluation JSON preserves the original `accumulated/overall/raw_rmse`
and `accumulated/overall/normalized_rmse` keys; their values are unchanged,
and the dashboard does not emit generic or redundant scalar aliases.
`primary_evaluation` in the W&B summary and `final_evaluation_report` in campaign
reports identify this route and the evaluated scope. A-only deployment uses
whole-task A progress on `[0,1]`, with equal raw and normalized RMSE and B/C/D
unassessed. Its curves use `compressed_whole_task_progress_*`; scopes containing
both A and C→D use `mixed_route_accumulated_progress_*`. Route labels are
determined separately for each group/task. The `poi_micro` RMSE curves pool all
POI errors. A-only evaluation also logs
`eval/overall/compressed_whole_task_progress_task_macro_raw_rmse`: the arithmetic
mean of per-task POI RMSEs, recommended for comparing multi-task pretraining
runs. Neither aggregation gives each trajectory equal weight. Compact schema
`grana-compact-eval-v3` names the aggregation explicitly; older runs retain
their keys, values and audit support. Reports explicitly mark accumulated quality
unavailable if the evaluation did not compute it; they never substitute a
conditional-head RMSE.

The `role_B_poi_rmse` and `role_D_poi_rmse` remain in full evaluation
artifacts as **diagnostics**. B uses the true stage; D uses C's predicted stage on exactly
the same images. D's local target is 1 for an earlier predicted stage, true
within-stage progress for the correct stage, and 0 for a later predicted stage.
Its focus-relative RMSE therefore does not measure accumulated progress error.
New runs do not emit duplicate B/D scalar aliases. The separate oracle
accumulated comparison uses `true stage + B progress`. Artifacts also retain
the eight-slot weighted objective, per-task results, per-stage diagnostics,
baselines and task-macro diagnostics. Evaluation preserves training RNG and model
mode. The phase defaults do not perform step-zero evaluation.

Checkpoints contain language LoRA, complete progress/temporal heads, and the
four added input/output marker embedding rows. They initialize **weights only**:
optimizer, scheduler, RNG, sampler, and step counters start fresh. A posttrain
run starts at update 1. This is not interruption recovery with optimizer-state
continuation. Architecture-incompatible loads fail. Complete matching heads load
even when loss coefficients change. Explicit `training.allow_fresh_heads` applies only
to missing/incomplete/shape-incompatible heads.

At the end of every run, each worker confirms parameters changed, deliberately
perturbs a trainable parameter, reloads the saved checkpoint, and compares both
tensor digests and fixed real C/D predictions for ABCD runs (an active example
when D is disabled). All workers must pass before
success is recorded. These checks establish training and restore behavior;
held-out metrics must be interpreted separately from convergence claims.

## ABCD supervision

| Role | Input and supervision |
|---|---|
| A | Merged task description → cached whole-task progress |
| B | Ordered stages and GT stage at the POI as focus → focus-relative progress |
| C | Ordered stages without a focus or GT stage → original-vocabulary letter classification |
| D | C's exact observations and detached prediction at its designated POI slot as focus → focus-relative progress |

For any observation, a requested focus earlier than / equal to / later than its
GT stage has target **1 / within-stage progress / 0**. Progress uses a temporal
refiner and a 10-bin distribution head. C bypasses that refiner and reads raw
marker hidden states against the original letter-token output weights.

Each row contains eight observation groups. Each cache selects an ordered camera
list, and rows with different camera counts can share a physical batch.
All eight observations, including repeated boundary frames, are supervised with
`[0.1, 0.3, 0.5, 1, 1.5, 2, 2, 2]` in selector base order. Query ordering moves
weights with their observations. Edit `loss.frame_weights` in
`configs/pretraining/default.yaml` or `configs/posttraining/default.yaml` to
change that phase's weights for both stage and progress losses. Both heads
use **CE + normalized squared CDF error**: C uses hard stage labels;
A/B/D retain the ten-bin C51 soft targets. Each CDF coefficient defaults to
1.0, and the optional low-progress ceiling defaults to zero. Edit `loss.stage`
and `loss.progress` in the same phase YAMLs; see the
[unified loss definition and baseline recipes](TRAINING_CONFIGS.md#unified-loss).
Checkpoint loading accepts weights independently of the source loss.
Pretrain's C loss coefficient is 0.25; posttrain's is 10.0.

Pretraining defaults to `roles: {A: 10, B: 10, C: 12, D: 0}` per forward,
totaling 32 independent windows per device.
With D active, default posttraining uses `roles: {A: 0, B: 10, C: 3, D: 3}`
**per pair**: B5+C3, then
fresh B5+D3 with C’s observations. GA2 completes one pair per worker; the
default GA16 on one worker completes eight pairs per update. A/B counts must
be even and C>=D. Each role is normalized independently across
the complete optimizer window and all workers before its coefficient is
applied. D=0 enables independent A/B/C batches. A=0 enables B/C/D training.
Recipes can override this packing while retaining the same role-loss
normalization.

The [recommended pretraining configuration](../configs/pretraining/default.yaml)
also enables stage augmentation for pretraining tasks: training C rows use ten
candidate stage answers, with distractors drawn from other training tasks.
Use per-task overrides when the stage count or donor inventory cannot support
this setting. A/B supervision, evaluation and inference keep their canonical
semantics. Posttraining defaults to augmentation off. See
[stage augmentation](stage_augmentation.md) for the flags and task exclusions.

Training draws uniformly from `data.stride_pairs`, which defaults to
`[[10,2],[16,4],[24,6],[32,8]]`: each pair has probability 25% for each
independent A/B/C example. Task selection is weighted within a role;
trajectory and POI selection are uniform. All training uses the
[two-layer sampling configuration](TRAINING_CONFIGS.md#two-layer-training-frame-sampling):
`data.train_frame_sampling.frame_selection_profile_weights` chooses regular or
irregular-stratified observations, and `query_order_profile_weights`
independently chooses forward, backward or tail rewind. Both pretraining and
posttraining default to 60% regular CTSP / 40% irregular selection and
50% forward / 20% backward / 30% tail rewind. These are independent draw
probabilities, not per-batch quotas. D reuses C's stride pair, observations,
order and POI without an independent draw.

New evaluation plans use `evaluation.stride_pairs`, fixed to `[[16,4]]` by
default, with regular/forward geometry. Changing training strides or either
mixture does not affect those evaluation inputs. Dense and render plans inherit
the base plan's strides. A main evaluation plan stays fixed once created;
historical frozen plans retain their explicit stride pairs and exact inputs.
One-stage rows in mixed-stage caches are supported: C has one
valid class and zero CE; D receives stage 0. Preference objectives
are excluded. HTTP transport uses explicit windows; it does not add a streaming
sampler.
