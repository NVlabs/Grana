# Multitask pretraining and task adaptation

Choose the training phase first, then the dataset/task and model size. The
[Grana paper, §3.3 and Appendix B](https://reward-grana.github.io/assets/paper/reward_Grana_arxiv_v1.pdf)
uses **Pretraining** and **Adaptation**. This repository calls them **multitask
pretraining** and **task adaptation (post-training)**. Runtime stage names stay
`pretrain` and `posttrain` for existing scripts and checkpoints.

Pretraining learns reusable progress and stage recognition across multiple
tasks. Adaptation specializes those weights for a target task, particularly a
long task with ordered stages. Adaptation can also train several related tasks
in one run; the number of caches does not select the training phase.

## Phase defaults

| Setting | Multitask pretraining | Task adaptation / post-training |
|---|---|---|
| Canonical directory | `configs/pretraining/` | `configs/posttraining/` |
| Typical scope | Multiple tasks | One target task; multiple supported |
| Active roles | A / B / C | B / C / D |
| Initialization | Base VLM with new adapters and heads | Same-size Grana checkpoint, weights only; or base VLM with `training.init_checkpoint: null` |
| Role counts in YAML | A10 / B10 / C12 / D0 per forward | A0 / B10 / C3 / D3 per two-forward pair |
| C loss coefficient | 0.25 | 10.0 |
| A / B / D loss coefficients | 1 / 1 / 1; D inactive | 1 / 1 / 1; A inactive |
| Learning rate | 2e-5 | 2e-5 |
| Target global batch | 128 role rows/update | 128 role rows/update |
| Optimizer updates | 5,000 | 200 |
| Warmup updates | 100 | 10 |
| Schedule | Cosine | Cosine |
| Gradient checkpointing | On, non-reentrant | On, non-reentrant |
| Training Role C stage augmentation | Enabled for pretraining tasks; 10 candidates | Off |
| Training stride pairs | `[[10, 2], [16, 4], [24, 6], [32, 8]]`, each 25% | Same four pairs, each 25% |
| Base evaluation stride pairs | `[[16, 4]]` | `[[16, 4]]` |
| Save / evaluate every | 500 / 500 updates | 25 / 50 updates |
| Regular held-out sampling | Up to 2 trajectories/task, 2 POIs each | All trajectories, 8 POIs each |
| Dense diagnostics | Opt-in; default cap 4 trajectories/task, stride 10 | Same; optional B oracle diagnostic |

The paper specifies learning rate 2e-5 and batch size 128. The role-row
interpretation of batch size, role mix, C coefficients, step counts and
evaluation cadence above are **engineering starting presets**, not a claim of
exact paper reproduction or demonstrated convergence. The longer pretraining
budget supplies more task exposure; inspect actual sampling coverage and
held-out quality when choosing the final budget. Both phases retain the same
LoRA, temporal/progress architecture, eight-slot targets and loss definitions.

Configuration and CPU tests cover all five model sizes. Select devices and
qualify memory use and model quality for your own workload before a full run.

Gradient checkpointing defaults to **on** for training, canaries, smoke tests,
profiling, memory benchmarks, and throughput/timing measurements. Keep it on
when comparing physical batch sizes or accumulation. Test
`model.gradient_checkpointing=false` only on an explicit user request.
Checkpointing saves activation memory by recomputing activations during
backward. See the [benchmark summary](benchmarks.md) for the scope and limits
of published measurements.
Deployment optimizations and FLA forward measurements do not change this
training default or qualify the training backward path.

The recommended pretraining recipe explicitly enables both stage-augmentation
switches and `default_task_enabled`. It uses other training tasks' stage
descriptions as distractors for Role C. Use `task_overrides` to exclude tasks
whose stage count or donor inventory cannot support ten candidates; evaluation
and inference keep canonical stage choices. See
[stage augmentation](stage_augmentation.md) for the configuration and donor
requirements. Standalone recipes must inherit or explicitly supply this section.

## Shared GPU image resizing

Use one data setting for training, train-time evaluation, standalone checkpoint
evaluation, matching the default CUDA inference preprocessing:

```yaml
data:
  gpu_image_resize: true
model:
  gradient_checkpointing: true
```

Set it in the run recipe, use `--set data.gpu_image_resize=true`, or include
`configs/data/gpu_image_resize.yaml` after the base recipe in `extends`.
Each rank uses its own CUDA device; the implementation does not assume a GPU
count or model. The shared processor transfers RGB uint8 images, then performs
resize, normalization and patchification on CUDA, retaining the same resize
geometry, interpolation options, image order and `data.max_pixels` budget.
This applies to model preprocessing; it does not alter source cache files.

The default remains false for existing recipes. True is recorded in the resolved
config, training runtime, evaluation plan/results and checkpoint preprocessing
descriptor (version 2, `resize_backend: torchvision_cuda`). CUDA inference uses
GPU resize by default, including for version 1 and legacy CPU-trained checkpoints.
`inference_runtime.resize_matches_checkpoint` reports whether its effective
preprocessing matches training. A GPU-trained checkpoint requires supported CUDA
preprocessing in automatic mode; `--no-gpu-image-preprocessing` explicitly selects
full host preprocessing and records the difference. No separate inference resize
switch is needed.

Training and inference share preprocessing, without sharing training activations
or retaining image batches across optimizer updates. CPU and CUDA resize can
round differently; alignment requires the same RGB inputs, processor settings
and pixel budget, and does not guarantee identical predictions under other
inference-only optimizations.

Published preprocessing measurements and their limits are summarized in
[benchmarks](benchmarks.md). Validate tensor alignment and the full training,
evaluation and checkpoint-inference path when changing preprocessing.

## What each role learns

| Role | Supervision | Why it is used |
|---|---|---|
| A | Whole-task description → whole-task progress | Broad pretraining, including caches without usable stage labels |
| B | Annotated focus stage → focus-relative progress | Grounded progress learning from stage annotations in both phases |
| C | Ordered stage descriptions → current stage | Direct stage recognition in both phases |
| D | C-selected focus stage → focus-relative progress on the same observations | Adaptation to the model-selected focus used at deployment |

B and D targets are 1 / within-stage progress / 0 when the requested focus is
earlier than / equal to / later than the observation's GT stage. C receives
direct stage-classification supervision at all eight observation slots. D
reuses C's observations and detached predicted focus; it supplies no additional
independent C labels or direct gradient through the stage argmax.

A-only caches participate in pretraining through A. B/C sampling uses eligible
staged caches, with per-role task weights from the manifest. Post-training
defaults to A=0; B/C select staged examples and D derives from C. A can be
enabled explicitly when whole-task supervision is part of the experiment.
Role coefficients weight each independently normalized role loss; increasing
its row count improves exposure without simply multiplying that coefficient.

Both stage and progress losses use the eight selector-base observation weights below.
Query ordering moves these weights with their observations:

```yaml
loss:
  frame_weights: [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0]
```

Edit this field in [pretraining/default.yaml](../configs/pretraining/default.yaml)
or [posttraining/default.yaml](../configs/posttraining/default.yaml) to adjust
that phase's defaults across every task and model size. A task recipe or
`--set 'loss.frame_weights=[...]'` can override it for a specific experiment.
Each role is normalized by the sum of its frame weights across the optimizer
window, so these values control the relative emphasis of selected observations.
Training and evaluation receive the same resolved weights. Saved run configs
retain their explicit weights when reloaded.

## Two-layer training frame sampling

Every training recipe uses one configuration interface. Both pretraining and
posttraining default to 60% regular CTSP / 40% irregular selection and,
independently, 50% forward / 20% backward / 30% tail rewind. The phase defaults
expose the same values as Python defaults; omitted fields inherit these values.
There is no enable switch or top-level profile selector:

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

Training draws one of the four `data.stride_pairs` uniformly for each
independent A/B/C example: `(10,2)`, `(16,4)`, `(24,6)` or `(32,8)`, each with
probability 25%. Both training selection profiles use the selected pair. D
inherits C's pair and observations without another draw.

New evaluation plans use the separate `evaluation.stride_pairs`, fixed to
`[[16,4]]` by default, and regular/forward geometry. At this evaluation scale,
away from boundaries, the eight observations are
`[p-48,p-32,p-16,p-8,p-4,p,p+4,p+8]`, spanning 57 cache frames. Dense and render
plans inherit their base evaluation plan's strides.

An explicit `data.stride_pairs` override changes training only. An explicit
`evaluation.stride_pairs` override changes newly built evaluation plans.
Training strides, selection weights and query-order weights do not change the
evaluation windows or plan hash. Frozen historical plans and comparison
manifests retain their recorded stride pairs and exact inputs; replay does not
replace them with today's defaults.

After the existing task/trajectory/uniform-POI and stride-pair draws, each
independent A/B/C row draws one selection profile and one query-order profile.
The distributions are independent, not per-batch quotas. Weights must be finite,
nonnegative numbers with positive total mass per layer; each layer normalizes
independently. Zero disables a choice. Specify all weights when switching a
distribution, since recipe inheritance merges individual keys.

| Frame selection | Eight observations before query ordering |
|---|---|
| `regular_ctsp` | Existing three sparse context plus five dense local observations, with the original clipping and fallback order |
| `irregular_stratified` | Three stratified context observations, two randomly spaced observations on each side of the POI, and the designated POI |

For POI `p`, cache length `N` and strides `(c,t)`, regular context is
`[p-3*c,p-2*c,p-c]` when `p >= 3*c`, otherwise `[0,p//3,2*p//3]`.
The five local observations are `[p-2*t,p-t,p,p+t,p+2*t]`, clipped to
`[0,N-1]`. The designated POI is base occurrence 5, even when boundary
clipping duplicates the same physical frame in another slot. Its `poi_slot`
is 5 for forward/tail-rewind and 2 for backward.

The irregular window is
`[max(0,p-max(3*c,2*t+3)), min(N-1,p+2*t)]`, inclusive. Irregular spacing uses
three strata before `p-2*t` when sufficient history exists, otherwise the
existing `[0,p//3,2*p//3]` context fallback; each side draws two distinct
distances from `1..2*t` and clips to the trajectory. All units are cache frames.
POI is never redrawn to avoid a boundary. Repeated physical frames remain
distinct supervised observations.

| Query order | Indices into the selected base observations |
|---|---|
| `forward` | `[0,1,2,3,4,5,6,7]` |
| `backward` | `[7,6,5,4,3,2,1,0]` |
| `tail_rewind` | `[0,1,2,7,6,5,4,3]` |

Both selection profiles use context then local as their base order. `forward`
preserves that base order. In particular, regular context/local overlap can be locally
nonmonotonic: retaining it makes regular/forward reproduce the original inputs
and weights exactly for the same explicit stride pair. Rewind reverses the last five selected observations, not
a separately selected physical-time interval. Every order is a pure permutation.

The defaults activate all six selection/order combinations, with joint
probabilities of 30% / 12% / 18% for regular CTSP and 20% / 8% / 12% for
irregular selection, in forward / backward / tail-rewind order. Source/POI,
stride, selection and order random streams remain separate. Changing order
weights preserves the chosen eight observations for a fixed seed and draw
sequence. D inherits C's exact selection, order, views and designated `poi_slot`;
the detached C prediction at that slot supplies D's focus. Images, labels and
loss weights move together, and backward never applies `1-progress`.

Training mixtures do not alter the fixed main evaluation windows or plan hash.
The frozen evaluation geometry still queries slot 5. Gradient checkpointing and
the compact W&B metric/step contract are unchanged. Mixture quality requires a
separate controlled training comparison. Training applies the default mixture
across all four stride pairs. For a regular/forward-only experiment, explicitly
set selection weights to `1/0` and order weights to `1/0/0` in the YAML order above.

Only these two selection profiles are supported. The removed `uniform_window`
key is rejected even with weight zero; omit it from new recipes. Historical run
configs and receipts retain their recorded semantics. Independent per-profile
window settings are not accepted training configuration fields. Explore the
geometry and pure permutations in the offline
[frame sampling explorer](visualizations/frame-sampling-explorer.html).

## Unified loss

Both phases use **CE + normalized squared CDF error** for both heads. This is
one production objective; the config and Python API have no loss-profile
selector. Edit the coefficients in the phase's `default.yaml`:

```yaml
loss:
  stage:
    cdf_weight: 1.0
  progress:
    cdf_weight: 1.0
    ceiling_weight: 0.0
```

For predicted probabilities `p` and target distribution `q`, the per-slot loss
is:

```text
CE(q, p) + cdf_weight × sum((CDF(p)[j] - CDF(q)[j])², j=0..K-2) / (K-1)
```

C uses a one-hot target and the task's actual number of stages `K`; padded
answer letters do not affect the denominator. A single-stage example has zero
stage loss and gradient. A/B/D use `K=10` and C51 interpolation between
neighboring progress bins, preserving the continuous target's mean. CDF
penalizes misplaced probability mass across ordered boundaries; it does not
impose monotonic predictions over time.

The CDF coefficient is independent for each head. Setting it to zero gives
ordinary hard/soft CE. The optional progress ceiling contributes
`ceiling_weight × relu(predicted_mean - target - 1/9)` when the target is below
0.05. Its default is **0.0**, so both heads use exactly CE + CDF. The 0/1
focus-relative targets retain their ordinary CE + CDF supervision. The
coefficient can be changed for an explicit experiment without changing the
base formula or the phase defaults. These coefficients are engineering
starting values, not demonstrated optima.

The objective, temporal weights and role coefficients remain separate choices.
After applying the eight frame weights, each role is normalized over its own
frames and rows across the complete optimizer update and all workers. The
phase then combines these normalized role losses:

```text
Multitask pretraining: L_A + L_B + 0.25 × L_C
Task adaptation:      L_B + 10.0 × L_C + L_D
```

Training logs `role_X_ce_loss`, `role_X_cdf_loss` and `role_X_ceiling_loss`.
Components include their internal coefficients and the same frame/update
normalization; their sum is `role_X_loss` before the outer role coefficient.
Training, held-out evaluation and dense replay use the same loss functions.
Stage accuracy, stage MAE and errors of at least two stages are reported at
the POI and over all eight slots, with per-task and task-macro metrics.
Deployment quality uses accumulated progress RMSE at the POI; objective values
do not measure accuracy. See [evaluation interpretation](#evaluation-interpretation).

Loss settings belong to the run's `config.json` and W&B configuration.
Checkpoint saving omits the loss section, and weights-only loading never
consults the source training objective. Any compatible architecture's
checkpoint can initialize training with new loss coefficients; no loss
migration or allowance is required. Adapter/head tensor shapes, tokenizer
identity, LoRA scaling and weight integrity still determine whether weights
can be represented correctly.

## Controlled comparisons

For loss experiments, keep initialization, seeds, frame/role weights, batch
and evaluation plans fixed, and change only the intended coefficient. Inspect
per-stage support and accuracy: fixed temporal POIs can miss a short stage, so
report stage-balanced diagnostics separately. D's target depends on each
model's predicted stage; its RMSE alone is not a common-target comparison.
Keep prediction-error images and videos with numerical results outside the
source checkout. Published bounded results are in [benchmarks](benchmarks.md).

The synthetic `configs/experiments/stage_choice/` recipes illustrate a paired
baseline and stage-choice augmentation experiment. Replace their cache paths
and task identifiers; donor stages come only from other training tasks and
must provide enough distinct descriptions for `candidate_count`. A baseline
with all task overrides disabled does not need donor augmentation. See
[stage augmentation](stage_augmentation.md) for the mapping and C/D rules.

## Global batch and actual stage supervision

`training.global_batch_size` counts **all role rows across workers and forwards
in one optimizer update**, including D rows that reuse C observations. It does
not count distinct trajectories or distinct frames. `grana validate`, training
`--dry-run`, and campaign plans expose `supervision_budget`; workers also save
`supervision_budget.json` beside the resolved config.

Pretraining recommends a physical batch of 32 per device for H100 training,
with `roles: {A: 10, B: 10, C: 12, D: 0}`. With the default global target of 128:

| Quantity per optimizer update, across all workers | Pretraining | Post-training |
|---|---:|---:|
| A windows | 40 | 0 |
| B windows | 40 | 80 |
| Independent C windows | 48 | 24 |
| C-derived D rows | 0 | 24 |
| C label slots, including repeated boundary frames | 384 | 192 |
| Physical batch per forward per worker | 32 | 8 |

With D off, `roles` counts rows per forward. With D on, it counts the whole
pair: default post-training runs B5+C3, then fresh B5+D3 using C's observations.
A/B counts must be even and C>=D. When C>D, the remaining C rows are
independent windows distributed across the two forwards. An even number of
accumulation forwards preserves complete pairs. The optimizer still normalizes losses across the
full update window and all workers.

Set `training.gradient_accumulation_steps: null` to derive it from:

```text
D inactive: global_batch_size = sum(roles) × num_workers × accumulation
D active:   global_batch_size = sum(roles) × num_workers × accumulation / 2
```

For the symmetric defaults, this is also physical batch per worker × workers
× accumulation. With an odd number of independent C rows, the two forwards
have different physical sizes; use the reported `forward_batch_sizes_per_worker`
and global supervision counts rather than multiplying by the peak size.

Pretraining's default physical batch of 32 gives accumulation 4 / 2 for 1 / 2
workers at global batch 128. Posttraining's physical batch of 8 gives
accumulation 16 / 8 / 4 / 2 for 1 / 2 / 4 / 8 workers. Within these supported
combinations, changing the GPU count preserves each role's global supervision
budget. Combinations that cannot produce an even accumulation count fail before
launch. For pretraining on 4 / 8 workers with BS32 and accumulation 2, explicitly
set global batch to 256 / 512. Set `training.num_workers` explicitly to the
assigned GPU count; exposing more CUDA devices alone does not change it. The
default is one worker. DDP retains a full model replica and physical batch on
each GPU.

To use explicit counts instead, set both:

```yaml
training:
  global_batch_size: null
  gradient_accumulation_steps: 2
```

This is useful for short workflow checks and controlled comparisons. Existing
configs that explicitly set accumulation and omit `global_batch_size` retain
this count-based behavior. An explicit accumulation that conflicts with an
inherited global target is rejected; disable the target or set accumulation to
null. Fully specified historical role, optimizer and evaluation values remain
authoritative over new phase defaults.

## Config layout and composition

```text
configs/
  pretrain.yaml, posttrain.yaml      # short entrypoints to custom recipes
  pretraining/
    default.yaml                    # phase defaults and automatic global batch
    custom.yaml                     # user-provided cache placeholder
    multitask.yaml                  # small synthetic two-task manifest example
  posttraining/
    default.yaml                    # phase defaults and checkpoint initialization
    custom.yaml                     # cache and checkpoint placeholders
  models/qwen35/
    {0.8b,2b,4b,9b,27b}.yaml         # model-only overlays
  data/
    example_train.yaml, example_eval.yaml
    gpu_image_resize.yaml            # optional preprocessing overlay
  experiments/stage_choice/          # synthetic controlled-comparison example
  cache.yaml, cache_legacy.yaml      # synthetic cache-construction schemas
  labeling_example.yaml             # synthetic human-labeling schema
  annotation_example.yaml, annotation_task_cards.json
```

Fallback phase defaults are centralized in `grana/config.py` and selected using
`training.stage`, including a stage selected with `--set`. The phase's
`default.yaml` provides editable overrides, including `loss.frame_weights`.
User-owned recipes select data, evaluation and run labels. Model overlays own
model identity and backend settings; 27B retains the qualified native Torch delta-rule backend. All five
sizes share the phase defaults, without independent copies of role and
optimizer settings.

`extends` accepts one relative path or an ordered list. Later parents override
earlier parents, the child overrides its parents, and `--set` overrides the
composed values. Mappings merge recursively; lists replace. Unknown keys and
inheritance cycles fail. An inherited manifest path remains relative to the
file declaring it; paths inside that manifest remain relative to the manifest.
Changing a manifest to an inline cache list also requires `manifest: null`.

For example, a related-task adaptation recipe can contain:

```yaml
extends:
  - /path/to/reward-grana/configs/posttraining/default.yaml
  - /path/to/reward-grana/configs/models/qwen35/4b.yaml
data:
  caches:
    - path: /datasets/task-one/train_cached
      roles: [B, C]
    - path: /datasets/task-two/train_cached
      roles: [B, C]
training:
  init_checkpoint: /runs/pretrain-4b/checkpoint-005000
  output_dir: /runs/related-tasks-4b
```

Add disjoint held-out caches and enable `evaluation` and `logging` for quality
experiments. The `custom.yaml` examples contain placeholders; replace them
before training. The multitask example enables evaluation with synthetic
manifest locations under `${GRANA_DATA_ROOT}`. No data payload or private task
inventory is included. Logging remains disabled until explicitly configured.
`configs/pretrain.yaml` and `configs/posttrain.yaml` are short entrypoints to
the custom recipes. Compose phase and model overlays in your own project;
per-task model aliases are not required.

A run's exported `config.json` retains both resolved caches and the original
manifest locations for provenance. To reuse it, select those frozen caches by
adding `--set data.manifest=null --set evaluation.manifest=null`, and set a new
output directory. Campaign runtime YAML already clears its manifest fields.
This keeps recorded training values without re-reading a changed inventory.

## Validate and run

Create a project recipe with your own data, following
[project configuration](project-configuration.md). A structural dry-run can
load the synthetic example with any explicit cache root, without opening data:

```bash
export GRANA_DATA_ROOT=/path/to/your/caches
python scripts/ray_migration/run_specs_cli.py \
  --config configs/pretraining/multitask.yaml --dry-run
```

This validates configuration and supervision budgets; it does not establish
that a cache exists. Before actual training, validate the real cache contents:

```bash
grana validate --config "$GRANA_TRAIN_CONFIG" --check-data
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" --dry-run
```

Review the resolved roles, role rows per update, initialization, split and
evaluation call counts before launching with a fresh output directory. Remove
`--dry-run` and assign GPUs to execute training. Adaptation loads matching
pretrained adapters, complete heads and marker weights; optimizer, scheduler,
sampler, RNG and update counters start fresh.

The [campaign launcher](TRAINING_SCRIPTS.md) requires `--pretrain-config PATH`
and accepts repeated `--posttrain-config NAME=PATH` options. It reads your
recipes' update budgets and assigns only the devices named in `--lane`.
Use `--pretrain-set training.max_steps=...` and `--posttrain-set ...` for phase
overrides. Its post-training dependency points to the actual final pretraining
step, including when that budget changes. Mandatory transfer requires matching
resolved model location, revision, LoRA rank and LoRA alpha within each lane;
planning rejects differences before launch. Roles, losses, preprocessing and
explicit fresh-head allowances can differ. The checkpoint loader still checks
actual token IDs, tensor coverage, shapes and integrity at runtime.

Campaign model size selects the backbone identity unless `--model SIZE=LOCATION`
is supplied. Other model-overlay settings are defaults: an explicit backend in
an inherited recipe or phase override wins; an unspecified 27B backend remains
`torch`. Inline cache paths resolve against the planning working directory,
while manifest entries resolve against their manifest. Campaigns freeze these
cache paths and explicit local model paths before execution changes to the
checkout directory. `./`, `../`, expanded home paths and existing model
directories identify local models; other `org/name` values remain Hugging Face
IDs. Shell wrappers plan from the checkout root.

## Evaluation interpretation

Primary quality is accumulated progress RMSE on the deployment route. For
staged tasks, C predicts a stage and D predicts progress conditioned on that
stage: the POI prediction is `C stage + D progress`, compared with
`true stage + true within-stage progress`. Report raw RMSE and normalized RMSE;
normalization divides each error by that trajectory's stage count before
squaring and pooling POIs. W&B names the pair
`eval/overall/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse` and
`eval/overall/predicted_stage_route_accumulated_progress_poi_micro_normalized_rmse`.
Groups use `eval/group/<group>/`; multiple tasks retain pairs under
`eval/task/<task>/`. The full JSON preserves `accumulated/overall/raw_rmse`
and `accumulated/overall/normalized_rmse` without changing their values.
A-only scopes use `compressed_whole_task_progress_*` with scale 1;
mixed A/C→D scopes use `mixed_route_accumulated_progress_*`. Names follow the
actual route of each scope, not the overall recipe. The `poi_micro` curves pool
squared errors over POIs and then take the square root; each POI has equal
weight, including within task and group scopes. The W&B `primary_evaluation`
summary and campaign `final_evaluation_report` explicitly name this aggregation,
the route and evaluation scope.

A-only evaluation additionally exposes the evaluator's existing equal-task
average as
`eval/overall/compressed_whole_task_progress_task_macro_raw_rmse`.
**Prefer this task-macro metric for multi-task pretraining comparisons.** It
takes the arithmetic mean of per-task POI RMSEs; within each task, all evaluated
POIs are pooled across trajectories. Neither the task-macro nor POI-micro metric
gives each trajectory equal weight. A-only progress uses scale 1, so its raw
and normalized RMSE are equal. These names apply to new runs under compact history
schema `grana-compact-eval-v3`; existing v2 runs retain their original keys and
schema-aware auditing. No metric formula or archived payload is changed, and
new runs do not emit old-name scalar aliases.

`role_B_poi_rmse` and `role_D_poi_rmse` remain conditional diagnostics in the
full evaluation artifact; they are not emitted as scalar aliases. B uses the true
stage; D uses C's predicted stage and a focus-relative target of 1, true
within-stage progress, or 0 when that stage is earlier, equal, or later than
the true stage. These values must not substitute for predicted-stage accumulated
RMSE. The separate
`eval/task/<task>/oracle_stage_route_accumulated_progress_poi_micro_raw_rmse` comparison
uses `true stage + B progress`. Reports do not fabricate accumulated metrics for a
sampled evaluation that has not computed them, or invent staged metrics for
an A-only holdout. Archived metric payloads and their mathematical definitions
remain unchanged; new charts use explicit route names without duplicate aliases.

### Batch size and vision attention

| Setting | Default |
|---|---:|
| `evaluation.batch_size` | `1` |
| `evaluation.batched_vision_sdpa` | `false` |
| `training.batched_vision_sdpa` | `false` |

`evaluation.batch_size` is a positive integer limiting the number of held-out
windows scored together for one role. Batched C predictions supply D's focus
on the same windows, in the same order. Batching preserves the planned cohort,
eight-slot record structure and logical role-row count (`model_calls`);
`forward_calls` records the actual batched model invocations. Evaluation batching is
independent of the training batch and accumulation settings.

During training, evaluation automatically uses all `training.num_workers` ranks.
Contiguous whole batches are split evenly by batch count, preserving batch
membership, C-to-D pairing, the final partial batch and the original output order.
If there are fewer batches than ranks, surplus ranks have empty shards. Each
rank uses its existing unwrapped model; rank 0 recomputes global metrics from
all records, rather than averaging per-rank RMSE or task means. The same
distribution applies to optional render and dense scoring passes; dense mode
continues to reuse its primary predictions.

Shard records are saved under `evaluation_shards/step-STEP/PURPOSE/rank-RANK.json`
in the training job's shared output directory. Collectives carry only small
digests/status messages, avoiding GPU staging of large serialized results.
`execution_world_size` and per-worker memory/timing record actual execution;
combined elapsed time includes scoring, shard persistence and aggregation.
Rank 0 performs final logging/rendering before all ranks continue training.
Checkpoint saving still precedes evaluation when both occur on the same step.

New W&B runs use the compact evaluation contract `grana-compact-eval-v3`.
Core curves contain accumulated raw/normalized RMSE overall and per group,
per-task pairs when multiple tasks exist, stage accuracy, oracle accumulated
comparisons, coverage, scoring/full-event time, throughput, and peak reserved
memory/free memory per actual worker. The number of curves follows the task,
group and worker inventories. Conditional-head diagnostics, per-stage metrics,
baselines and detailed profiles are retained in full per-event JSON artifacts;
they are not emitted again under legacy scalar names. Images, videos and
confusion/support/selection tables remain available. Full final summaries have
a separate completion artifact. No compatibility switch restores dense alias
namespaces or every diagnostic as a chart.

`evaluation_and_persistence_seconds` covers scoring, rendering, local result
writes and media preparation. It excludes the final evaluation artifact enqueue,
unified history commit and asynchronous cloud upload completion. All original
timing fields remain in the per-event artifact for independent verification.

The two vision SDPA flags independently control evaluation and training.
They batch attention over equal-length images while keeping images separate.
The Qwen3.5 training path checks the entire vision encoder is frozen on every
forward and requires inputs/position embeddings without gradients, zero
attention dropout and the SDPA backend. Unsupported calls use the original
attention implementation. Enabling it preserves module modes and autograd;
keep `model.gradient_checkpointing: true` for training, profiling and timing.
Enable these flags explicitly after qualifying the selected runtime and workload.

In the default sampled mode, metrics use fixed, evenly spaced interior POIs: two per trajectory for
pretraining and eight for adaptation. Eight POIs do not guarantee every stage
or transition appears. An A-only holdout does not assess C stage accuracy;
a stage-labelled holdout is required for that question. Task adaptation
assesses B/C/D on staged holdouts.

With the default `evaluation.mode: sampled`, dense diagnostics separately score
every 10 source frames and the final frame on a fixed cohort. With
`evaluation.mode: dense`, the dense plan is the primary evaluation plan and its
predictions are reused for primary accumulated metrics and media. Dense mode
requires enabled evaluation and `evaluation.deploy_dense`; a null trajectory
cap includes every trajectory admitted by the held-out plan. The staged route
is C→D, with an optional B oracle curve;
A-only tasks use A. `deploy_dense_videos`, `deploy_dense_samples`, `overlay_acc`
and `eval_quality` retain the real predictions, stage errors and provenance.
Best/worst selection is within that cohort, with at most 96 measured points
shown in a video. Optional dense RMSE and sampled POI diagnostics have different
populations; compare each metric against the same recorded population and its
own baseline.
