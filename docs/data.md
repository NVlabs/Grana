# Build a training cache

Use an existing Grana cache, or convert annotated LeRobot v2 episodes into one.
Run the commands below from the checkout after [installation](../README.md#install).

## Human annotation

For recordings without reward labels, use the [human-labeling editor](human-labeling.md)
to mark subtasks, Completed/Failed outcomes and action-quality intervals. Its
recipe keeps labels separate from the original recordings and generated cache.
After reviewing and submitting episodes, run `grana build-cache` with that same
recipe. Drafts cannot become training supervision.

## Optional stage annotation

For unannotated LeRobot v2 recordings, the optional
[RGB stage-annotation workflow](ANNOTATION.md) covers bounded API proposals,
resumption, local video review and stage export. Reward-cache construction then
requires explicit action-quality and outcome supervision; stage predictions
alone do not supply those labels.

The portable [robot annotation skill](../skills/grana-robot-annotation/SKILL.md)
proposes stages from a task goal and demonstrations, then derives an AQ rubric
for each objective. It provides methods and synthetic examples; real task
vocabularies remain private. The source workflow includes RGB evidence preparation,
a Responses driver, bounded local resampling, strict candidate validation, and
event/normal-recovery diagnostics. Proposed rules and combined AQ/outcome
candidates require task calibration and review. This does not change the
separate stage-only API command.

## Build a cache from LeRobot and annotations

This optional CPU step needs `ffmpeg` and `ffprobe` on `PATH`. It reads native
LeRobot v2 parquet, annotations and camera videos; it does not require the
LeRobot Python package. Start with the synthetic two-stage token/tray recipe;
its target parameters are explicitly **illustrative examples**:

```bash
export GRANA_LEROBOT_SOURCE=/data/raw/my_lerobot_dataset
export GRANA_CACHE_OUTPUT=/data/grana/my_task_train
grana build-cache --config configs/cache.yaml
```

Edit `episodes`, `task`, `compressed_task_prompt`, ordered `stages`, and the
`views` feature names for your input. Annotations are already inside parquet:

| Input | Meaning |
|---|---|
| `meta/info.json` | LeRobot v2 path templates, source FPS and feature metadata |
| `meta/episodes.jsonl` | Episode index, length and optional trajectory identity |
| `meta/tasks.jsonl` | Integer annotation IDs mapped to text |
| `data/.../episode_*.parquet` | Aligned `frame_index`, `episode_index`, `timestamp`, `annotation.sub_goal`, `annotation.sub_goal_score`, `annotation.frame_type` |
| `videos/.../episode_*.mp4` | One aligned RGB video per selected camera |

`annotation.sub_goal` decodes to configured stage labels such as `"1"`, `"2"`,
`"3"`; ordinary unscored labels are `leading`, `trailing`, or an empty string.
Sparse stage scores decode to `high`, `mid`, `low`, or `empty`. The LeRobot
adapter maps outcomes to 1, 0.5, 0. Quality comes from `annotation.frame_type`:
`normal`/`nominal`, `suboptimal`, `failure`/`regressed`; unmarked `empty`/`""`
is nominal. Each labeled subtask run needs an explicit outcome, and conflicting
outcomes are rejected. An existing `rl_info.action_quality_mask` is checked and
disagreements recorded, but cannot override these native annotations under the
new policy. Existing `rl_info.abs_progress` is not used as a target.

The builder validates frame counts, timestamps, annotations and all cameras.
It computes targets at source FPS, applies the target policy's frame projection,
then selects uniformly spaced frames at the requested output FPS. All views
and labels use the same original frame indices. `image_size: [width, height]`
explicitly resizes; omit it to retain native resolution. Output contains a
Hugging Face `processed_dataset/`, RGB uint8 `frames/*.npy`, and
`provenance.json` with the resolved recipe, method versions and input identities.
Output must be a new directory outside the code checkout and raw source; the
complete staged cache is validated before it becomes the final output.
Keep the same FFmpeg build on `PATH` for pixel-identical regeneration; different
decoder versions can produce slightly different RGB values from the same video.

`configs/cache.yaml` selects **`grana_target_composition`**, implemented in
[Grana's target-composition module](../grana/data/targets/composition.py).
The construction follows these rules:

- Split each annotated subtask at regressed intervals; nominal/suboptimal
  changes remain inside one chunk. Nominal updates have weight `rho`,
  suboptimal updates have weight 1.
- For a chunk interrupted by regression, set its endpoint to
  `p_cap * min(1, T_chunk / T_min)`. For a chunk reaching the subtask boundary,
  use its annotated outcome.
- Let `N = sum(weight * delta_t)` over the chunk's non-regressed updates.
  Each update receives `endpoint * cumulative_weighted_time / N`.
  Regressed frames always receive local progress 0, including a final failure.
- Compose whole-task progress as `(zero_based_stage_id + local_progress) / K`.
  A local reset retains the current stage's earlier-stage credit.

Under the discrete time convention, a subtask starts at its first frame's
timestamp, where progress is zero. After regression, the last regressed frame
is the recovery reset anchor. The next regressed frame's timestamp determines
an interrupted chunk's duration; its
last non-regressed update receives the endpoint. A positive endpoint with zero
weighted duration is rejected rather than inventing time. This is a native-FPS
construction: downsampling happens only after targets are complete.

Supply **each stage's** `rho > 1`, `0 < p_cap < 1`, and `t_min_seconds > 0`, in
the same order as `stages`. The public recipe is a synthetic token/tray example
with **50 / 0.95 / 15s** marked `parameter_status: example`. Replace the proposed
stage vocabulary, camera keys and parameters with your reviewed private task
configuration. The annotation skill explains how to propose those definitions;
the public configuration is not a catalogue of our task stages.

The old algorithm's pre-regression ceiling was **0.90**, and its
`progress_cap_seconds` limited a future suffix-weight denominator window.
That window could include failure and later attempts, so it is not equivalent
to the new `T_min`. Parameters from another task or target policy do not
establish calibration for your data. The independent screw-insertion regression
example uses a 9s value; that example does not establish calibration for other
tasks.

Record `parameter_status` (`example`, `configured`, or `calibrated`) and
`parameter_source` explicitly. `configured` means intentionally selected
hyperparameters, while `calibrated` records values supported by task
measurements. The builder accepts both without a calibration gate. All
parameters and their provenance remain in cache rows, together with each
native chunk's construction details.

This adapter keeps exact annotated stage boundaries, drops only unscored
prefix/suffix frames, and rejects unscored interior gaps. It does not merge short
stages, absorb trailing frames, or infer missing outcomes. These strict input
conventions and the high/mid/low mapping are LeRobot adapter choices,
separate from the target-composition formula. Stage identities follow the
annotation, including regression; the composition assumes earlier stage credit
from that identity, not an independently tracked completion ledger.

For historical reproduction, `configs/cache_legacy.yaml` selects
`legacy_concat_quality_v1` in `grana/data/targets/legacy.py`; this retains the original
quality-mask precedence, missing-outcome heuristics, short-run merge and trailing
hold. Older recipes without `progress.method` still select legacy. Previously
generated caches and trained checkpoints retain their original target policy.
Changing a recipe does not relabel a cache or retrain a checkpoint. Both policies
return aligned `source_indices`, `stage_ids`, `within_stage_progress` and
`progress`, so the cache reader and ABCD training flow are shared.
Frame selection remains `linspace_v1`. Failure-aware oversampling, affine stage
ranges, per-episode remapping and LeRobot v3 are outside this compact builder.

Build a separate cache from disjoint held-out episodes, then feed the new cache
to the existing training flow. Set `data.views` to the cache's ordered camera
names in validation and training; the synthetic recipe uses `[top,wrist]`,
while the training defaults use `[top,left,right,bottom]`. For example, with
the synthetic camera names and new output directories:

```bash
grana validate --config configs/pretrain.yaml \
  --set 'data.views=[top,wrist]' \
  --set "data.caches=[$GRANA_CACHE_OUTPUT]" --check-data
bash scripts/train.sh --config configs/pretrain.yaml \
  --set 'data.views=[top,wrist]' \
  --set "data.caches=[$GRANA_CACHE_OUTPUT]" \
  --set training.output_dir=/runs/grana/new-data-pretrain
bash scripts/train.sh --config configs/posttrain.yaml \
  --set 'data.views=[top,wrist]' \
  --set "data.caches=[$GRANA_CACHE_OUTPUT]" \
  --set training.init_checkpoint=/runs/grana/new-data-pretrain/checkpoint-005000 \
  --set training.output_dir=/runs/grana/new-data-posttrain
```

These generic recipes select multitask ABC pretraining (5,000 updates) and
BCD task adaptation (200 updates); configure [held-out evaluation](training.md#evaluation-and-checkpoint-semantics)
and [logging](training.md#wb-and-local-evidence) for quality experiments. Both use learning rate 2e-5 and target global batch 128. See the
[training configuration guide](TRAINING_CONFIGS.md) for role budgets,
phase defaults and composition. For inference, use the new cache's RGB paths
and task text in the [eight-observation request](inference.md#inference-from-a-trained-checkpoint). Ground-truth annotation arrays
are never part of that inference request.

## Cache format and source layout

A cache contains a Hugging Face `processed_dataset/`, one row per trajectory:
`id`, `task`, `num_frames`, camera paths such as `frames_top`, `frames_left`,
`frames_right`, `frames_bottom`, `progress`, `compressed_task_prompt`,
`stage_ids`, `within_stage_progress`, and ordered `stage_prompts`.
Camera payloads are RGB uint8 `.npy` arrays `[T,H,W,3]`; selected cameras share
`frames_shape`, and labels align with the stored frame count. Existing absolute
paths are used directly; stale paths may resolve only to the same cache's
`frames/<basename>`. Payloads remain external and read-only.

The public `configs/data/example_train.yaml` and `example_eval.yaml` show a
minimal two-task layout using `${GRANA_DATA_ROOT}`. They contain synthetic
folder names, with no cache payloads or task inventory. Replace those locations
and the camera names with your own data before validation or training.

A manifest contains only a `caches` list. Entries may supply `prompt`, eligible
`roles`, positive per-role `task_weights`, an ordered `views` override, and an explicit `include_ids` or
`exclude_ids` list. Alternatively, use `hash_split: {seed: my-split-v1, eval_count: 7, subset: train}`
(or `subset: eval`). SHA256(`seed|trajectory_id`) ranks the full cache inventory;
the lowest `eval_count` hashes form evaluation and the remainder form training.
Both subsets retain source row order. This selector requires unique IDs, a
nonempty seed, and a positive integer evaluation count smaller than the source
inventory, and cannot be combined with explicit ID selectors.
D derives from C; never declare a separate D data lane.
Manifest paths resolve relative to the recipe declaring them, including
inherited recipes; cache paths resolve relative to the manifest. Environment variables and `~` are supported in manifest paths,
model paths, output directories and initialization paths.

```yaml
caches:
  - path: ${MY_CACHE_ROOT}/multistage_cached
    roles: [A, B, C]
    task_weights: {A: 0.9, B: 0.1, C: 0.1}
    views: [exterior, wrist]
```

Without an override, a cache uses `data.views`. The example above requires
`frames_exterior` and `frames_wrist` in each selected HF row. Another cache
entry can use four cameras in the same run. Requested cameras must exist;
missing views are never filled by copying another camera. Evaluation manifests
record each window's actual camera list.

Other paired RGB NPY stores can be used directly for label-free inference by
putting their camera paths in a request JSON. A native RoboLab manifest is not
a Grana HF training cache: its stage/completion labels need an explicit,
validated conversion to the progress and within-stage targets required by the
selected roles. Dynamic image inputs do not perform that label conversion.
