# Configure a project

After [installation](installation.md), create your own recipes outside the
source tree. The examples below are standalone: they do not inherit bundled
task recipes or require a particular robot or dataset. Recordings and model
weights are supplied separately.

Run commands from the source root. Choose a workspace outside Git checkouts
and raw datasets; the location below is an example:

```bash
export GRANA_WORKSPACE="${GRANA_WORKSPACE:-$HOME/grana-workspace/project}"
mkdir -p "$GRANA_WORKSPACE"
export GRANA_LABEL_CONFIG="$GRANA_WORKSPACE/labeling.yaml"
export GRANA_TRAIN_CONFIG="$GRANA_WORKSPACE/training.yaml"
```

Export the workspace and input variables again in new shells. Save the YAML
blocks as files in this workspace. For an existing labeling project, keep using
its active YAML and stage history instead of recreating these templates.

## Labeling your own recordings

The reader supports native **LeRobot v2.x and v3.0** datasets, including v3
shared parquet and video ranges. Episodes need at least two frames and fixed
cadence. V2 video timestamps must start at zero; v3 uses an episode clip.
See the [source requirements](data.md#build-a-cache-from-lerobot-and-annotations)
for camera and clock constraints.

Set `GRANA_LEROBOT_SOURCE` to the absolute root of your dataset, containing its
`meta/`, parquet and video files. Read the metadata before choosing cameras:

```bash
: "${GRANA_LEROBOT_SOURCE:?Set the absolute path to your LeRobot dataset}"
export GRANA_LEROBOT_SOURCE
python - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ["GRANA_LEROBOT_SOURCE"])
info = json.loads((source / "meta/info.json").read_text())
print("format:", info["codebase_version"], "native fps:", info["fps"])
print("video features:", [key for key, feature in info["features"].items()
                          if feature.get("dtype") == "video"])
PY
```

Save this as the file named by `GRANA_LABEL_CONFIG`. Replace the task text,
stage descriptions, camera feature key and cache frame rate before use. Add
one `views` entry per actual camera; names such as `camera_1` are your chosen
aliases and must match the training recipe later.

```yaml
source: ${GRANA_LEROBOT_SOURCE}
labels: ${GRANA_WORKSPACE}/labels
output: ${GRANA_WORKSPACE}/cache-v1
episodes: all
task: your-task-id
compressed_task_prompt: Describe the complete task in English.
stages:
  - label: '1'
    prompt: Describe the first observable subtask in English.
  - label: '2'
    prompt: Describe the second observable subtask in English.
views:
  camera_1: REPLACE_WITH_A_VIDEO_FEATURE_KEY_FROM_METADATA
fps: 5
progress:
  method: grana_target_composition
  parameter_status: example
  parameter_source: Preview values only; calibrate for the actual task before training.
  stage_parameters:
    - {rho: 50, p_cap: 0.95, t_min_seconds: 15}
    - {rho: 50, p_cap: 0.95, t_min_seconds: 15}
```

Choose 1–26 stages at the granularity appropriate for your task, with one
parameter row per stage. Review the [target parameters](data.md#target-composition)
before training. `fps` controls cache sampling and cannot exceed source FPS;
the labeling timeline uses native frames. `episodes: all` includes submitted
episodes when building; use an explicit index list to require those episodes.

Start the editor:

```bash
python -m grana.cli label --config "$GRANA_LABEL_CONFIG"
```

Open **http://127.0.0.1:8790/**. The [labeling guide](human-labeling.md) covers
SSH forwarding, stage editing, language switching and review/submission.
After submitting reviewed episodes, build their training cache:

```bash
python -m grana.cli build-cache --config "$GRANA_LABEL_CONFIG"
```

The cache destination must be new and outside the raw source and Git checkout.
Draft episodes are not included. Stage editing updates the active YAML and may
select a new version's `labels` and `output` directories, so read the **current**
YAML or build report when setting the training cache path. Keep that YAML and
the complete `.stage-versions/` directory with your project backups.

## Training with your own cache

Use the full CUDA environment and an existing Grana cache. Training does not
need the source recordings. Set the cache directory from the build report:

```bash
: "${GRANA_TRAIN_CACHE:?Set the absolute path to your training cache}"
export GRANA_TRAIN_CACHE
```

Save `training-data.yaml` in `GRANA_WORKSPACE`. The manifest expands environment
variables in cache paths; add entries to train on more caches.

```yaml
caches:
  - ${GRANA_TRAIN_CACHE}
```

Save this as the file named by `GRANA_TRAIN_CONFIG`. Set `data.views` to the
exact camera aliases present in your caches:

```yaml
model:
  name_or_path: Qwen/Qwen3.5-0.8B
  gradient_checkpointing: true
data:
  manifest: ${GRANA_WORKSPACE}/training-data.yaml
  views: [camera_1]
training:
  stage: pretrain
  output_dir: ${GRANA_WORKSPACE}/runs/pretrain
  num_workers: 1
logging:
  mode: disabled
evaluation:
  enabled: false
```

Use this model ID or a compatible local base-model directory. Unspecified
settings use the [phase defaults](TRAINING_CONFIGS.md), including role allocation,
update budget and image processing. Review them before launching:

```bash
grana validate --config "$GRANA_TRAIN_CONFIG" --check-data
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" --dry-run
```

Validation checks the cache and camera schema; dry-run prints resolved settings
and supervision counts. Neither starts training or measures GPU memory. Choose
the image budget and physical batch size for your GPU, keeping gradient
checkpointing enabled. Launch with an unused output directory:

```bash
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG"
```

For posttraining, set `training.stage: posttrain` in a separate recipe with a
new output directory. Set `training.init_checkpoint` to a same-size Grana
checkpoint, or `null` to initialize from the backbone. Add disjoint held-out
data and configure evaluation and online logging as described in
[training](training.md).

For inference, use your checkpoint and a request following the
[inference guide](inference.md).

## Compose model and phase recipes

If you prefer inheritance, combine `configs/pretraining/default.yaml` or
`configs/posttraining/default.yaml` with one of the model-only overlays in
`configs/models/qwen35/{0.8b,2b,4b,9b,27b}.yaml` or
`configs/models/qwen38/27b.yaml`. Use absolute paths from your
external recipe or a `${GRANA_SOURCE_ROOT}` variable that you set to the source
checkout. The 27B overlay selects its qualified Torch delta-rule backend.
Add your own data manifest, evaluation and output paths to the child recipe.
Later parents override earlier parents; child fields and then `--set` win.

For staged pretraining and adaptation, pass these user-owned recipes to the
[campaign launcher](TRAINING_SCRIPTS.md). It accepts explicit device lanes and
initializes each adaptation from its lane's final pretraining checkpoint.
