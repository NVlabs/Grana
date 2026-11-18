# Training configuration reference

`training.stage: pretrain` learns progress and stage recognition across tasks;
`posttrain` adapts to a target task or several related tasks. The number of
caches does not select the phase. Start with a
[project recipe](project-configuration.md#training-with-your-own-cache), then
follow the [training workflow](training.md).

## Phase defaults

| Setting | Pretraining | Adaptation (`posttrain`) |
|---|---|---|
| Phase recipe | `configs/pretraining/default.yaml` | `configs/posttraining/default.yaml` |
| Initialization | Base VLM, new adapters and heads | Same-size Grana weights, or base VLM with `training.init_checkpoint: null` |
| YAML role counts | A10 / B10 / C12 / D0 per forward | A0 / B10 / C3 / D3 per two-forward pair |
| Physical rows per forward per worker | 32 | 8 |
| Global role rows per update | 128 | 128 |
| A / B / C / D loss coefficients | 1 / 1 / 0.25 / 1; D inactive | 1 / 1 / 10 / 1; A inactive |
| Learning rate / schedule | 2e-5 / cosine | 2e-5 / cosine |
| Optimizer updates / warmup updates | 5,000 / 100 | 200 / 10 |
| Save / evaluate every | 500 / 500 updates | 25 / 50 updates |
| Sampled evaluation | Up to 2 trajectories/task, 2 POIs each | All trajectories, 8 POIs each |

These are starting recipes. Enable and configure held-out evaluation and
logging explicitly; choose the update budget using your task's coverage and
quality. The final update is saved and, when evaluation is enabled, evaluated
once. Both phases share the role semantics, frame sampling and loss definitions
below. [Runtime settings](training.md#runtime-settings-and-timing) cover
checkpointing, CPU sample prefetch and profiling.

## What each role learns

| Role | Text input to the model | Source of the focus stage | Supervision |
|---|---|---|---|
| A | Whole-task description | — | Cached whole-task progress |
| B | Task, ordered stages, focus stage | Ground-truth stage at the POI | Focus-relative progress |
| C | Task and ordered stages | — | First unfinished stage |
| D | Task, ordered stages, focus stage | C's detached argmax at its paired POI | Focus-relative progress |

B and D share the same prompt format and progress head. B's focus comes from
the cached stage label during training and held-out evaluation; at inference,
the caller supplies it. D's focus comes from C in all three paths.
C identifies the first unfinished stage in task order, including failed or
incomplete prerequisites; if all stages are complete, it selects the final stage.

For B/D, a focus earlier than / equal to / later than an observation's true
stage has target **1 / within-stage progress / 0**. Progress uses a temporal
refiner and a ten-bin distribution head. C reads raw marker hidden states
against the original letter-token output weights, masked to the task's K
chronological stages. Stage IDs and descriptions retain their native order.

Each independent A/B/C row selects a task using per-role manifest weights,
then a trajectory and POI uniformly. A supports caches without stage labels;
B/C require eligible staged caches. D reuses its matching C's exact
observations, camera views, order and designated POI. Its focus is the
**detached C argmax**: D adds no independent C labels or gradient through that
selection.

Every row has eight supervised observation groups, with the cache's ordered
camera views. Boundary clipping can repeat a physical frame in separate slots.
Frame weights bind to selector-base observations and move with query ordering:

```yaml
loss:
  frame_weights: [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0]
```

These weights apply to both heads. Override them in a run recipe or the phase
default YAML; saved configurations retain their explicit values.

## Global batch and actual stage supervision

`training.global_batch_size` counts **all role rows across workers and forwards
in one optimizer update**, including D rows reusing C observations. It does
not count distinct trajectories or frames. `grana validate` and `--dry-run`
report `supervision_budget`; executed runs save `supervision_budget.json`.

With D inactive, `roles` counts rows per forward. With D active, it counts a
**two-forward pair**: default adaptation uses B5+C3, then fresh B5+D3 on C's
observations. A/B counts must be even and C>=D. Extra C rows when C>D are
independent windows distributed across the pair.

Set `training.gradient_accumulation_steps: null` to derive accumulation:

```text
D inactive: global_batch_size = sum(roles) × num_workers × accumulation
D active:   global_batch_size = sum(roles) × num_workers × accumulation / 2
```

Accumulation must be even. For the symmetric defaults, global batch also equals
physical rows per worker × workers × accumulation. With an odd number of extra
independent C rows, the two forwards have different sizes: use the reported
`forward_batch_sizes_per_worker` and total role rows, not the peak batch size.

At the default global batch of 128:

| Quantity per optimizer update, across all workers | Pretraining | Adaptation |
|---|---:|---:|
| A rows | 40 | 0 |
| B rows | 40 | 80 |
| Independent C windows | 48 | 24 |
| C-derived D rows | 0 | 24 |
| C label slots, including repeated boundary frames | 384 | 192 |

Pretraining with physical batch 32 uses accumulation 4 / 2 for 1 / 2 workers.
Adaptation with physical batch 8 uses accumulation 16 / 8 / 4 / 2 for
1 / 2 / 4 / 8 workers. Invalid combinations fail before launch. For
pretraining on 4 / 8 workers at physical batch 32 and accumulation 2, explicitly
set global batch to 256 / 512. `training.num_workers` defaults to one and must
match the allocated GPUs; each worker retains a complete model replica.

To specify accumulation without a global target:

```yaml
training:
  global_batch_size: null
  gradient_accumulation_steps: 2
```

An explicit accumulation that conflicts with an inherited global target is
rejected. Recipes that specify accumulation and omit a global target use
count-based behavior. Check the resolved budget after changing role counts or
worker count.

## Unified loss

Both heads use cross-entropy (CE) plus normalized squared CDF error:

```yaml
loss:
  stage:
    cdf_weight: 1.0
  progress:
    cdf_weight: 1.0
    ceiling_weight: 0.0
```

For predicted probabilities `p` and target distribution `q`, per-slot loss is:

```text
CE(q, p) + cdf_weight × sum((CDF(p)[j] - CDF(q)[j])², j=0..K-2) / (K-1)
```

C uses a one-hot target and the task's actual stage count K; padded letters do
not affect the denominator. Single-stage rows have zero stage loss and gradient.
A/B/D use K=10 and C51-style interpolation between neighboring progress bins,
preserving the continuous target's mean. CDF penalizes distance across ordered
bins; it does not impose monotonic predictions over time.

Setting a head's CDF coefficient to zero gives ordinary hard/soft CE. The
optional progress ceiling adds
`ceiling_weight × relu(predicted_mean - target - 1/9)` for targets below 0.05;
its default coefficient is zero. The focus-relative 0/1 targets retain the
ordinary CE + CDF objective.

After frame weighting, **each role is normalized independently over its frames
and rows across the complete optimizer update and all workers**, then multiplied
by its role coefficient:

```text
Pretraining: L_A + L_B + 0.25 × L_C
Adaptation:  L_B + 10.0 × L_C + L_D
```

`role_X_ce_loss`, `role_X_cdf_loss` and `role_X_ceiling_loss` include internal
coefficients and frame/update normalization; they sum to `role_X_loss` before
the outer role coefficient. More rows change exposure without simply
multiplying that coefficient. Training and evaluation use the same losses;
objective values and progress RMSE measure different quantities.

Loss settings belong to the run configuration, not checkpoint tensor
compatibility. Matching weights can initialize a run with different loss
coefficients without a fresh-head allowance; see
[checkpoint semantics](training.md#evaluation-and-checkpoint-semantics).

## Two-layer training frame sampling

Both phases use these defaults:

```yaml
data:
  stride_pairs: [[10, 2], [16, 4], [24, 6], [32, 8]]
  train_frame_sampling:
    frame_selection_profile_weights:
      regular_ctsp: 0.6
      irregular_stratified: 0.4
    query_order_profile_weights:
      forward: 0.5
      backward: 0.2
      tail_rewind: 0.3
evaluation:
  stride_pairs: [[16, 4]]
```

Each independent A/B/C row draws a training stride pair uniformly, then a
selection profile and an independent query order. These are probabilities,
not per-batch quotas. D inherits C's choices without another draw. Weights
must be finite, nonnegative and have positive total mass per layer; zero
disables a choice. Set every weight when changing a distribution because
recipe inheritance merges mapping keys.

For POI `p`, cache length `N` and strides `(c,t)`, the two selection profiles
produce three context observations followed by five local observations:

- `regular_ctsp`: context is `[p-3*c,p-2*c,p-c]` when `p >= 3*c`, otherwise
  `[0,p//3,2*p//3]`. Local observations are `[p-2*t,p-t,p,p+t,p+2*t]`,
  clipped to `[0,N-1]`.
- `irregular_stratified`: the inclusive window is
  `[max(0,p-max(3*c,2*t+3)), min(N-1,p+2*t)]`. Three context strata precede
  `p-2*t` when sufficient history exists, otherwise the same context fallback
  applies. Each side draws two distinct distances from `1..2*t`, clips to the
  trajectory, and retains the designated POI.

All distances use cache frames. POI is not redrawn at boundaries; repeated
physical frames remain distinct supervised occurrences. Query order is a pure
permutation of those base observations:

| Query order | Base indices | Designated `poi_slot` |
|---|---|---:|
| `forward` | `[0,1,2,3,4,5,6,7]` | 5 |
| `backward` | `[7,6,5,4,3,2,1,0]` | 2 |
| `tail_rewind` | `[0,1,2,7,6,5,4,3]` | 5 |

Context/local overlap can make even the forward base order locally
nonmonotonic. Rewind reverses the selected tail, not a new time interval.
Images, labels and frame weights move together; backward does not apply
`1-progress`. D's detached focus comes from C's designated POI slot after
permutation.

Evaluation uses its own `evaluation.stride_pairs` with regular/forward
geometry. At the default `[16,4]`, away from boundaries its observations are
`[p-48,p-32,p-16,p-8,p-4,p,p+4,p+8]`. Dense and render plans inherit those
evaluation strides. Training sampling settings do not change a fixed
evaluation plan. To use regular/forward-only training, explicitly set selection
weights to `1/0` and query-order weights to `1/0/0` in the YAML order above.

## Shared GPU image resizing

`data.gpu_image_resize` defaults to `false`. Enable it with
`--set data.gpu_image_resize=true` or the
[GPU resize overlay](../configs/data/gpu_image_resize.yaml).
It applies to training and evaluation: RGB uint8 images transfer to each
worker's device, then resize, normalize and patchify on CUDA. Resize geometry,
image order and `data.max_pixels` are preserved; source caches are unchanged.

The resolved configuration and checkpoint record the preprocessing choice.
CPU and CUDA resize can round differently. Inference has its own effective
preprocessing defaults and reports whether they match the checkpoint; see
[inference runtime requirements](inference.md#runtime-requirements).

## Evaluation interpretation

Training and held-out trajectory identities and resolved RGB paths must be
disjoint before any cap is applied. Every held-out task participates; trajectory
selection is deterministic and the plan stays fixed throughout a run.

`evaluation.mode: sampled` is the default. It selects evenly spaced interior
POIs using the phase's budgets above. Sparse POIs may miss short stages;
A-only holdouts cannot assess C stage accuracy. Optional `evaluation.renders`
uses a separate PNG/GIF diagnostic cohort.

`evaluation.deploy_dense` is opt-in. Its defaults score every 10 source frames
plus the final frame, capped at four trajectories per task; a null cap includes
all trajectories admitted by the held-out plan. With `evaluation.mode: dense`,
enabled dense evaluation becomes the primary plan and its predictions are
reused for metrics and media. With sampled mode, dense diagnostics describe
a separate population. `max_frames` (96), `replay_fps` (8), `group_by_cache`
and `include_oracle` control media and optional oracle diagnostics.
Media show actual unsmoothed predictions; sampled-POI playback is not
original-speed video.

Primary staged quality is **C→D accumulated progress RMSE**:
`predicted C stage + D progress` against
`true stage + true within-stage progress` at the POI. Raw RMSE uses stage
units. Normalized RMSE divides each POI error by its trajectory's stage count
before squaring and pooling. W&B names these:

- `eval/overall/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse`
- `eval/overall/predicted_stage_route_accumulated_progress_poi_micro_normalized_rmse`

`poi_micro` gives every evaluated POI equal weight. Group/task scopes use
`eval/group/<group>/` and `eval/task/<task>/`. A-only scopes use
`compressed_whole_task_progress_*`, with raw and normalized scales both 1;
mixed A/C→D scopes use `mixed_route_accumulated_progress_*`.

For multitask A-only evaluation, prefer
`eval/overall/compressed_whole_task_progress_task_macro_raw_rmse`: the arithmetic
mean of per-task POI RMSEs. Within each task, POIs pool across trajectories.
Neither this task-macro metric nor POI-micro gives each trajectory equal weight.

`role_B_poi_rmse` and `role_D_poi_rmse` are conditional-head diagnostics, not
accumulated progress quality. D's focus-relative target changes with C's
predicted stage. The optional `oracle_stage_route_accumulated_progress_*`
comparison uses `true stage + B progress` where B is available. Reports cannot
substitute these diagnostics for uncomputed accumulated or staged metrics.
Compact W&B curves provide primary quality, coverage and timing; full event
JSON artifacts retain predictions, conditional diagnostics, per-stage metrics
and baselines.

### Batch size and vision attention

| Setting | Default |
|---|---:|
| `evaluation.batch_size` | `1` |
| `evaluation.batched_vision_sdpa` | `false` |
| `training.batched_vision_sdpa` | `false` |

Evaluation batch size limits windows scored together for one role,
independently of training batch and accumulation. Batched C predictions
condition D on the same windows in the same order. `model_calls` counts logical
role rows; `forward_calls` counts actual batched model invocations.

Train-time evaluation uses all training ranks, distributing whole batches
while preserving C→D pairing, the final partial batch and output order.
Surplus ranks can receive empty shards. Rank 0 recomputes global metrics from
merged records rather than averaging rank-level RMSE, then logs/renders before
training resumes. Checkpoint saving precedes evaluation when both are due.

The two vision SDPA flags independently batch equal-length image attention,
keeping images separate. The training path requires frozen vision weights,
inputs without gradients, zero attention dropout and the SDPA backend;
unsupported calls use the original implementation. These flags preserve
autograd and do not change gradient checkpointing.

## Config layout and composition

Phase defaults live in `configs/pretraining/default.yaml` and
`configs/posttraining/default.yaml`. Their `custom.yaml` recipes contain
placeholders for user caches and checkpoints; `configs/pretrain.yaml` and
`configs/posttrain.yaml` are short entrypoints to those recipes.
`configs/models/qwen35/` and `configs/models/qwen38/` contain model-only overlays
(Qwen3.8 currently provides 27B). Unspecified values use
the defaults selected by `training.stage`, including a CLI stage override.

`extends` accepts one path or an ordered list. Later parents override earlier
parents, the child overrides its parents, and `--set` wins last. Mappings merge
recursively; lists replace. Unknown keys and inheritance cycles fail.

```yaml
extends:
  - /path/to/grana/configs/posttraining/default.yaml
  - /path/to/grana/configs/models/qwen35/4b.yaml
data:
  caches:
    - path: /data/my-task/train_cached
      roles: [B, C]
training:
  init_checkpoint: /runs/pretrain-4b/checkpoint-005000
  output_dir: /runs/my-task-4b
```

Add a disjoint held-out inventory and enable evaluation/logging as needed.
An `extends` path is relative to its declaring recipe. Manifest paths also
resolve relative to the recipe declaring them; entries inside a manifest
resolve relative to that manifest. Inline cache paths resolve against the
planning working directory; shell wrappers use the checkout root. When
replacing a manifest with inline caches, also set `manifest: null`.

Exported `config.json` retains resolved caches and manifest provenance.
To reuse the frozen caches, set both `data.manifest=null` and
`evaluation.manifest=null` and choose a new output directory. Explicit saved
values override defaults. See [training](training.md) for launch commands and
[training scripts](TRAINING_SCRIPTS.md) for campaigns and saved-checkpoint
evaluation.
