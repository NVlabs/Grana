# Build a training cache

Use an existing Grana cache or convert native **LeRobot v2.x or v3.0** recordings
with explicit reward labels. Cache construction runs on CPU; use the
[installation guide](installation.md#cpu-labeling-and-cache-construction) and
keep `ffmpeg` and `ffprobe` on `PATH`. The LeRobot Python package is not required.

## Human annotation

The [human-labeling editor](human-labeling.md) records subtasks,
Completed/Failed outcomes and action-quality intervals separately from the
source recordings. Submit reviewed episodes, then build with the same recipe.
Drafts cannot become training supervision.

## Optional stage annotation

The [API workflow](ANNOTATION.md) proposes stages and optionally action quality
from RGB evidence. Import candidates for review in the human editor, or supply
an accepted stage revision with an independent AQ/outcome supplement. Stage
predictions alone do not establish reward outcomes or nominal action quality.
The [stage-design guide](../skills/grana-robot-annotation/references/stage-task-cards.md)
helps define a task-specific vocabulary.

## Build a cache from LeRobot and annotations

Create a [standalone recipe](project-configuration.md#labeling-your-own-recordings)
with your source, task text, ordered stages, camera mapping and fresh output
directory. Run from the checkout:

```bash
python -m grana.cli build-cache --config /data/my-project/cache.yaml
```

Choose one supervision input:

| Recipe input | Required supervision |
|---|---|
| `labels` | Submitted human-label files; `episodes: all` selects submitted episodes |
| `annotations` | Accepted stage revisions and explicit AQ/outcome supplements covering the selected episodes; see the [sidecar contract](ANNOTATION.md#add-explicit-reward-supervision-and-build-a-cache) |
| Neither | Native parquet annotation columns, decoded through the dataset's task dictionary |

Without `labels`, `episodes` must be an explicit nonempty list. `labels` and
`annotations` cannot appear together. Native annotated data can use
[configs/cache.yaml](../configs/cache.yaml) as a schema example; replace its
synthetic task, camera names and target parameters.

The reader uses the dataset metadata to locate each selected episode:

| Input | LeRobot v2.x | LeRobot v3.0 |
|---|---|---|
| Source FPS, camera shapes and path templates | `meta/info.json` | `meta/info.json` |
| Episode length and optional trajectory ID | `meta/episodes.jsonl` | `meta/episodes/chunk-*/file-*.parquet` |
| Integer annotation IDs mapped to text | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| Native frame rows | One episode parquet | Episode-filtered rows in its declared shared parquet |
| Camera RGB | One video per episode and camera | Each camera's declared file and `[from_timestamp, to_timestamp)` range |

Each v3 episode must have one parquet location and one video range per camera;
different episodes and cameras may use different files. V3 and external-label
inputs require at least two consecutive native frames to establish the clock.
Frame indices must be contiguous, timestamps must match the source cadence, and every
camera must align with the episode. Selected cameras must have matching native
image dimensions, including when an output resize is requested.

Native supervision requires `frame_index`, `episode_index`, `timestamp`,
`annotation.sub_goal`, `annotation.sub_goal_score` and `annotation.frame_type`.
Under `grana_target_composition`, the decoded values are:

| Column | Values |
|---|---|
| `annotation.sub_goal` | Configured stage labels; `leading`, `trailing`, `empty` or `""` for unscored frames |
| `annotation.sub_goal_score` | Sparse `high`, `mid`, `low` outcomes, mapped to 1, 0.5, 0; `empty`/`""` elsewhere |
| `annotation.frame_type` | `normal`/`nominal`, `suboptimal`, `failure`/`regressed`; `empty`/`""` means nominal |

Each contiguous stage run needs one consistent explicit outcome. The builder
checks any `rl_info.action_quality_mask` and records disagreements, but the
composition policy uses `annotation.frame_type`. It ignores existing
`rl_info.abs_progress` values.

Targets are computed at source FPS before frame selection. `linspace_v1`
selects uniformly spaced frames from the retained native indices; RGB and all
labels share those indices. Output `fps` cannot exceed source FPS.
`image_size: [width, height]` applies area resizing; omit it to retain native
resolution. Use the same FFmpeg build for pixel-identical regeneration.

The output contains `processed_dataset/`, RGB uint8 `frames/*.npy`, and
`provenance.json` with the resolved recipe, method versions and input identities.
It must be a new directory outside Git checkouts and the raw source. The builder
validates the staged cache before publishing it and never changes source files.

### Target composition

Select `progress.method: grana_target_composition` explicitly, as in
[configs/cache.yaml](../configs/cache.yaml). Its construction is:

- Split each stage run at regressed intervals. Within a chunk, nominal updates
  have weight `rho`; suboptimal updates have weight 1.
- A chunk interrupted by regression ends at
  `p_cap * min(1, T_chunk / t_min_seconds)`. A chunk reaching the stage boundary
  uses that run's annotated outcome.
- With `N = sum(weight * delta_t)` over non-regressed updates, local progress
  is `endpoint * cumulative_weighted_time / N`. Regressed frames have local
  progress 0, including a final failure.
- Whole-task progress is `(zero_based_stage_id + local_progress) / K`, where
  `K` is the number of configured stages. A reset retains earlier-stage credit
  from the current stage identity; it does not track a separate completion ledger.

A stage starts at its first frame's timestamp with zero progress. After
regression, the last regressed timestamp anchors recovery. The next regressed
frame's timestamp determines an interrupted chunk's duration, and its last
non-regressed update receives the endpoint. A positive endpoint with zero
observed weighted duration is rejected.

Supply one parameter row per ordered stage: `rho > 1`, `0 < p_cap < 1` and
`t_min_seconds > 0`. Record `parameter_status` (`example`, `configured` or
`calibrated`) and `parameter_source`. `configured` means intentionally selected;
`calibrated` means supported by task measurements. The builder accepts either
without a calibration gate. Public example values do not establish calibration
for your task. Parameters and their provenance are retained in the cache.

This policy preserves annotated boundaries, drops only unscored prefix/suffix
frames and rejects interior gaps. It does not merge short stages, absorb
trailing frames or infer missing outcomes. See the
[target implementation](../grana/data/targets/composition.py) for the discrete formula.

### Progress-method compatibility

**Recipes without `progress.method` select `legacy_concat_quality_v1`.**
[configs/cache_legacy.yaml](../configs/cache_legacy.yaml) selects it explicitly.
It retains quality-mask precedence, missing-outcome heuristics, short-run
merging and trailing hold. Its `progress_cap_seconds` limits a future
suffix-weight denominator window and is not interchangeable with the
composition policy's `t_min_seconds`.

Changing a recipe does not relabel existing caches or retrain checkpoints.
Both policies return aligned `source_indices`, `stage_ids`,
`within_stage_progress` and `progress`, and use `linspace_v1` frame selection.
Failure-aware oversampling, affine stage ranges and per-episode remapping are
outside this builder.

Build disjoint training and held-out caches, then follow
[training with your own cache](project-configuration.md#training-with-your-own-cache).
Set the training camera names to the cache's ordered views.

## Cache format and source layout

A cache's Hugging Face `processed_dataset/` contains one row per trajectory:
`id`, `task`, `num_frames`, camera paths such as `frames_top`,
`compressed_task_prompt`, ordered `stage_prompts`, and aligned `stage_ids`,
`within_stage_progress` and `progress`. Camera payloads are RGB uint8 `.npy`
arrays `[T,H,W,3]`; selected cameras share `frames_shape`. Existing absolute
paths are used directly; stale paths may resolve only to the same cache's
`frames/<basename>`. Payloads remain external and read-only.

A manifest contains a `caches` list. Entries may supply `prompt`, eligible
`roles`, positive per-role `task_weights`, ordered `views`, and `include_ids`
or `exclude_ids`. D derives from C; never declare a separate D data lane.

```yaml
caches:
  - path: ${MY_CACHE_ROOT}/multistage_cached
    roles: [A, B, C]
    task_weights: {A: 0.9, B: 0.1, C: 0.1}
    views: [exterior, wrist]
```

Without an override, a cache uses `data.views`. The example requires
`frames_exterior` and `frames_wrist`; another cache entry can use four cameras
in the same run. Missing views are never filled by copying another camera.
Evaluation records each window's actual camera list.

For a reproducible split, use
`hash_split: {seed: my-split-v1, eval_count: 7, subset: train}` or `subset: eval`.
SHA256(`seed|trajectory_id`) ranks the full cache inventory; the lowest
`eval_count` hashes form evaluation. Both subsets retain row order. IDs must be
unique, the seed nonempty, and `eval_count` a positive integer smaller than the
inventory. Hash splitting cannot be combined with explicit ID selectors.

Manifest paths resolve relative to the declaring recipe, including inherited
recipes; cache paths resolve relative to the manifest. These paths support
environment variables and `~`. [Example manifests](../configs/data/example_train.yaml)
contain synthetic paths only; replace them with your own data.

Other RGB NPY stores can supply [label-free inference requests](inference.md#inference-from-a-trained-checkpoint).
Their labels need an explicit conversion to Grana's target schema before they
can serve as training caches.
