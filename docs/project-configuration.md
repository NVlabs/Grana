# Configure a project

After [installation](installation.md), create your own recipes outside the
source tree. These instructions do not depend on a specific robot, a bundled
dataset, or any YAML in the repository's `configs/` directory. The bundled
configs are small synthetic schema examples; they do not ship
recordings, checkpoints or a private task inventory.

Commands run from the source root. Use absolute locations for external data
and outputs. Choose a workspace outside every Git checkout and raw dataset;
the home-directory location below is a portable default, not a required path:

```bash
export GRANA_WORKSPACE="${GRANA_WORKSPACE:-$HOME/grana-workspace/project}"
mkdir -p "$GRANA_WORKSPACE"
export GRANA_LABEL_CONFIG="$GRANA_WORKSPACE/labeling.yaml"
export GRANA_TRAIN_CONFIG="$GRANA_WORKSPACE/training.yaml"
```

Export the workspace and input variables again when opening a new shell. They
are deployment settings, not values embedded in the source release.

The YAML blocks below are file contents to save in that workspace. They are
not commands that overwrite an existing project. For an existing labeling
project, use its retained active YAML and stage history directly instead of
recreating the template.

## Labeling your own recordings

The current reader supports native **LeRobot v2** datasets. This is a file-format
requirement, not a requirement to use a particular robot or dataset. LeRobot v3
and unrelated directory layouts need an explicit conversion before labeling.
Videos must have fixed frame cadence and start at presentation timestamp zero.

Set `GRANA_LEROBOT_SOURCE` to the absolute root of your dataset, containing its
`meta/`, parquet and video files. Read the metadata before choosing cameras:

```bash
: "${GRANA_LEROBOT_SOURCE:?Set the absolute path to your LeRobot v2 dataset}"
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

The two stage rows illustrate the schema, not a required task decomposition.
Use the number and granularity appropriate for your task; the editor supports
1–26 stages. Each stage needs a corresponding parameter row. Review the
progress parameters for the actual task before using the targets for training.
`fps` controls the generated cache cadence and must not exceed the source
frame rate; the labeling timeline always uses native source frames. Replace
`episodes: all` with an explicit list of episode indices to restrict selection.

In the CPU environment, start the editor from the source root:

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

Use the full CUDA environment. Supply an existing Grana cache, either built
above or prepared separately; training does not require the source recordings.
Set `GRANA_TRAIN_CACHE` to the absolute cache directory from the build report:

```bash
: "${GRANA_TRAIN_CACHE:?Set the absolute path to your training cache}"
export GRANA_TRAIN_CACHE
```

Save the following as `training-data.yaml` in `GRANA_WORKSPACE`. A separate
manifest allows environment variables in cache paths to be expanded by the
configuration loader. Additional cache entries can be added to the same list.

```yaml
caches:
  - ${GRANA_TRAIN_CACHE}
```

Save this standalone recipe as the file named by `GRANA_TRAIN_CONFIG`. It has
no `extends` dependency on shipped examples. Set `data.views` to the exact
camera aliases present in your caches:

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

The model ID selects a supported small backbone; you can use a compatible local
base-model directory instead. Model files are downloaded or supplied separately.
Unspecified settings come from the configuration loader's phase defaults,
including role allocation, update budget and image processing. Review the
resolved configuration and set your intended budgets before running a job:

```bash
grana validate --config "$GRANA_TRAIN_CONFIG" --check-data
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG" --dry-run
```

Neither command starts training. Data validation checks the actual cache and
camera schema; dry-run prints resolved settings and supervision counts. Neither
is a GPU memory or model-quality test. Qualify the image budget and physical
batch size on the selected GPU, keeping gradient checkpointing enabled. When
ready, launch with an unused output directory:

```bash
bash scripts/train.sh --config "$GRANA_TRAIN_CONFIG"
```

For posttraining, create a separate user-owned recipe with `training.stage:
posttrain` and a new output directory. Choose `training.init_checkpoint` as a
same-size Grana checkpoint or `null` for base-backbone initialization. This
selects posttraining defaults without a repository preset. Supply task data and
disjoint held-out data for a quality experiment. Enable evaluation and online
logging explicitly when they are configured; see [training](training.md) and
[configuration semantics](TRAINING_CONFIGS.md).

For inference, use your own same-size checkpoint and request file following
the [request schema](inference.md). Dataset, checkpoint, output and serving
locations are deployment choices, not paths built into the installation.

## Compose model and phase recipes

If you prefer inheritance, combine `configs/pretraining/default.yaml` or
`configs/posttraining/default.yaml` with one of the model-only overlays in
`configs/models/qwen35/{0.8b,2b,4b,9b,27b}.yaml`. Use absolute paths from your
external recipe or a `${GRANA_SOURCE_ROOT}` variable that you set to the source
checkout. The 27B overlay selects its qualified Torch delta-rule backend.
Add your own data manifest, evaluation and output paths to the child recipe.
Later parents override earlier parents; child fields and then `--set` win.

For a staged campaign, supply your pretraining recipe and named adaptation
recipes explicitly. Only the devices listed in each lane are assigned:

```bash
bash scripts/training/run_campaign.sh --pretrain-config "$GRANA_TRAIN_CONFIG" \
  --posttrain-config task-a="$GRANA_WORKSPACE/posttraining.yaml" \
  --lane 0.8B=0 --output-root "$GRANA_WORKSPACE/campaign-preview" --dry-run
```

A dry-run writes nothing, opens no cache payloads, and downloads no model.
Use `--check-data` to validate your actual caches and held-out separation.
The campaign makes each adaptation initialize from its lane's final
pretraining checkpoint. Repeated adaptation recipes run independently,
with jobs sharing a selected device queued in order. See
[training scripts](TRAINING_SCRIPTS.md) for execution and status commands.
