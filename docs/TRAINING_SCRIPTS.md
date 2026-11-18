# Training and evaluation scripts

Run commands from the source root in the [full CUDA environment](installation.md#full-cuda-environment).
The wrappers resolve the checkout and its environment; relative wrapper
arguments resolve against the checkout root. `GRANA_PYTHON` can select an
interpreter for read-only commands; training uses the checkout-owned environment.

| Command | Purpose |
|---|---|
| `scripts/training/run_campaign.sh` | Pretrain and adapt explicit recipes on assigned GPU groups |
| `scripts/training/status.sh` | Show state, update, training time and log paths |
| `scripts/training/wait.sh` | Observe pretraining or campaign completion |
| `scripts/training/report.sh` | Verify local metrics, checkpoints and media |
| `scripts/training/audit_wandb.sh` | Verify existing online runs and uploaded artifacts |
| `python -m grana.training.checkpoint_evaluation` | Evaluate one saved checkpoint |
| [`scripts/training/evaluate_checkpoint_suites.py`](checkpoint_evaluation_suites.md) | Evaluate explicit held-out cohorts on multiple GPUs |

For a single training recipe, use the [training workflow](training.md).

## Run a campaign

Supply a pretraining recipe and zero or more named adaptation recipes.
Each adaptation starts from its model size's final pretrain checkpoint with
[fresh optimizer state](training.md#evaluation-and-checkpoint-semantics):

```bash
bash scripts/training/run_campaign.sh \
  --pretrain-config /projects/my-task/pretrain.yaml \
  --posttrain-config task-a=/projects/my-task/task-a.yaml \
  --posttrain-config task-b=/projects/my-task/task-b.yaml \
  --lane 0.8B=0,1 --lane 4B=2,3 \
  --output-root /runs/my-campaign-001 \
  --logging-mode disabled --check-data --dry-run
```

Review the resolved jobs, split checks and supervision budgets, then remove
`--dry-run`. Dry runs allocate no GPUs and write no campaign files. Recipes
supply update budgets; adaptation loads the actual final pretrain checkpoint.

Pretraining uses every GPU in its lane. Adaptations use one GPU each,
assigned round-robin and serialized when sharing a GPU. Lanes must use
disjoint devices. CUDA indices and GPU/MIG UUIDs are accepted; do not name
the same device through different aliases. Each worker holds a full model
replica, so additional GPUs do not pool memory.

Repeat `--pretrain-set KEY=VALUE` or `--posttrain-set KEY=VALUE` for phase
overrides. For example, `--pretrain-set training.max_steps=750` makes adaptations
load `checkpoint-000750`. The launcher owns model identity, output paths,
devices, checkpoint dependencies and campaign logging; overrides cannot change
the phase. Recheck the [global budget](TRAINING_CONFIGS.md#global-batch-and-actual-stage-supervision)
after changing batches or worker counts.

Use `--model '4B=/models/Qwen3.5-4B'` for a local snapshot; otherwise model sizes
select official Qwen identifiers. `--model-family qwen35` is the default.
For Qwen3.8-27B, select `--model-family qwen38 --lane 27B=0,1`; its pretraining
and adaptation jobs use Qwen3.8 weights, checkpoint sources and logging tags.
Other sizes are rejected for that family. A local Qwen3.8 snapshot can be
selected with `--model '27B=/models/Qwen3.8-27B'`.
Logging follows the recipe unless
`--logging-mode online|offline|disabled` is supplied. Online mode needs separate
authentication; `--entity`, `--project`, `--group` and `--name` set campaign labels.

`--smoke` selects short training/evaluation budgets to check execution; it
does not establish convergence or full-batch capacity. Smoke logging is disabled
unless explicitly selected. Use a fresh output root and keep the foreground
command alive. Ctrl-C cancels this campaign's jobs. `--timeout-seconds` limits
campaign lifetime, including cleanup; the default is eight hours.

## Watch and wait

Use another terminal while the foreground campaign runs:

```bash
bash scripts/training/status.sh --control-dir /runs/my-campaign-001
bash scripts/training/wait.sh --control-dir /runs/my-campaign-001 \
  --until pretrain --timeout-seconds 7200 --poll-seconds 15
bash scripts/training/wait.sh --control-dir /runs/my-campaign-001 \
  --until all --timeout-seconds 28800 --poll-seconds 15
```

Observers accept the campaign root or its `control/` directory. Exit codes
are 0 for completion, 1 for failure/cancellation or a dead coordinator, and
124 for the observer's own timeout. An observer timeout leaves training
running; pretrain completion can be reported while adaptations continue.

## Verify results

Optional verification commands read completed outputs:

```bash
bash scripts/training/report.sh --control-dir /runs/my-campaign-001 \
  --output /reports/my-campaign-local.json
bash scripts/training/audit_wandb.sh --control-dir /runs/my-campaign-001 \
  --output /reports/my-campaign-wandb.json
```

Both require new report paths and return nonzero when checks fail. Local
reporting loads no model and makes no cloud requests. The online audit requires
W&B access; it reads existing runs and downloads artifacts to compare metrics,
timing and media digests. It does not modify runs. Offline/disabled runs can
use local reporting.

Interpret durations using the [timing boundaries](training.md#runtime-settings-and-timing);
parallel job durations do not sum to wall time. Progress metrics and conditional
diagnostics have distinct [definitions and populations](TRAINING_CONFIGS.md#evaluation-interpretation).
Campaign state and logs live under `control/`; each run retains its own
configuration, checkpoints, metrics, evaluation plans and optional media.
Keep outputs outside the source checkout.

## Evaluate a saved checkpoint

A saved checkpoint from an unfinished or completed training run is usable once
its save receipt exists. Evaluation checks checkpoint provenance and the
original held-out plan, then performs zero optimizer updates. Use a new output
directory outside the source run.

Restore the source experiment's model-cache and attention-backend environment
settings first: this direct module does not apply `scripts/train.sh` exports.
Campaign settings are recorded in `control/campaign.json`.

```bash
: "${GRANA_SOURCE_RUN:?Set the source training run directory}"
: "${GRANA_CHECKPOINT_STEP:?Set an update with a committed checkpoint}"
: "${GRANA_EVAL_OUTPUT:?Set a new evaluation output directory}"
source .venv-qwen35-unsloth/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m grana.training.checkpoint_evaluation \
  --source-run "$GRANA_SOURCE_RUN" --step "$GRANA_CHECKPOINT_STEP" \
  --output "$GRANA_EVAL_OUTPUT" --batch-size 32 \
  --device cuda:0 --logging-mode disabled --dry-run
```

`--dry-run` checks provenance and prints counts without loading a model or
creating output. Remove it to evaluate. `--batch-size` is required and is
independent of training accumulation. `--batched-vision-sdpa` is opt-in; see
[batch size and vision attention](TRAINING_CONFIGS.md#batch-size-and-vision-attention).

Online logging creates a fresh evaluation W&B run using the source
entity/project/group and a checkpoint-specific name. Offline and disabled
modes are supported. Outputs include `checkpoint_evaluation.json`,
`evaluation_plan.json`, `evaluation-STEP.json` and logging records; enabled
dense diagnostics add media under `deploy_dense/`. Completion follows
evaluation, rendering and logger finalization. For new cohorts or multiple
GPUs, use [checkpoint suites](checkpoint_evaluation_suites.md).
