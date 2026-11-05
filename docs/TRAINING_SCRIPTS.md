# Training and evaluation scripts

Run commands from the source root in the [full CUDA environment](installation.md#full-cuda-environment).
The wrappers in `scripts/training/` resolve the checkout and its environment;
relative wrapper arguments resolve against the checkout root. `GRANA_PYTHON`
can select an explicit interpreter for read-only commands; training requires
the checkout-owned environment.

| Command | Purpose |
|---|---|
| `scripts/training/run_campaign.sh` | Run explicit pretraining and adaptation recipes on assigned GPU lanes |
| `scripts/training/status.sh` | Show run state, update, training time and log paths |
| `scripts/training/wait.sh` | Observe pretrain or campaign completion with a bounded timeout |
| `scripts/training/report.sh` | Verify local metrics, timing, checkpoint metadata and media digests |
| `scripts/training/audit_wandb.sh` | Read existing online runs and verify metrics and uploaded media |
| `python -m grana.training.checkpoint_evaluation` | Evaluate one saved checkpoint into a new output directory |
| [`scripts/training/evaluate_checkpoint_suites.py`](checkpoint_evaluation_suites.md) | Evaluate explicit checkpoint cohorts and render saved results |

Actual optimizer updates use `scripts/train.sh` and the private local Ray
launcher. See [training](training.md) for supervision, checkpoint semantics and
[configuration](TRAINING_CONFIGS.md) for recipe keys.

## Run a campaign

Supply your own pretraining recipe and zero or more named adaptation recipes.
Each adaptation starts from its model size's final pretrain checkpoint with
fresh optimizer state. For example, after creating three standalone recipes:

```bash
bash scripts/training/run_campaign.sh \
  --pretrain-config /projects/my-task/pretrain.yaml \
  --posttrain-config task-a=/projects/my-task/task-a.yaml \
  --posttrain-config task-b=/projects/my-task/task-b.yaml \
  --lane 0.8B=0,1 --lane 4B=2,3 \
  --output-root /runs/my-campaign-001 \
  --logging-mode disabled --check-data --dry-run
```

Review the resolved jobs, split checks and `supervision_budget` before removing
`--dry-run`. The dry run does not allocate GPUs or write campaign files. Recipes
supply update budgets; phase defaults are 5,000 pretrain and 200 adaptation
updates. The checkpoint dependency follows the actual pretrain budget.

Pretraining uses every GPU assigned to its lane. Adaptations use one GPU each,
are assigned round-robin and serialize when they share a GPU. Devices must be
disjoint between lanes. CUDA indices and GPU/MIG UUIDs are accepted; do not use
two aliases for the same physical device. Each worker stores a complete model
replica, so additional GPUs do not pool memory.

Repeat `--pretrain-set KEY=VALUE` or `--posttrain-set KEY=VALUE` for phase
overrides. For example, `--pretrain-set training.max_steps=750` makes adaptations
load `checkpoint-000750`. The launcher controls model identity, output paths,
GPU assignment, checkpoint dependencies and campaign logging. Overrides cannot
change the phase. Inspect resolved accumulation and role budgets after any
batch or worker-count change.

Use `--model '4B=/models/Qwen3.5-4B'` for a local backbone snapshot. Otherwise
model sizes select their official Qwen identifiers. Logging follows the recipe
unless `--logging-mode online|offline|disabled` is supplied. For online runs,
select your own entity/project and authenticate separately; `--entity`,
`--project`, `--group` and `--name` provide explicit campaign labels.

`--smoke` selects short update and evaluation budgets to check execution;
it does not qualify convergence or full-batch memory. Smoke logging is disabled
unless selected explicitly. Keep gradient checkpointing enabled for smokes,
profiling and timing, as for normal training.

Use a fresh output root for every attempt and keep the foreground command
alive. Ctrl-C cancels its owned descendants using process identity checks.
`--timeout-seconds` limits the campaign lifetime, including cleanup; choose it
from measured startup, training and evaluation costs. The default is eight
hours with a cleanup reserve. It is not a completion-time estimate. The
launcher does not stop shared Ray clusters or unrelated services.

## Watch and wait

Use another terminal while the foreground campaign runs:

```bash
bash scripts/training/status.sh --control-dir /runs/my-campaign-001
bash scripts/training/wait.sh --control-dir /runs/my-campaign-001 \
  --until pretrain --timeout-seconds 7200 --poll-seconds 15
bash scripts/training/wait.sh --control-dir /runs/my-campaign-001 \
  --until all --timeout-seconds 28800 --poll-seconds 15
```

The observer accepts the campaign root or its `control/` directory. It returns
0 for verified completion, 1 for failure/cancellation or a dead coordinator,
and 124 when its own timeout expires. An observer timeout leaves training
running. Pretrain completion can be reported while adaptations continue.

## Verify results

```bash
bash scripts/training/report.sh --control-dir /runs/my-campaign-001 \
  --output /reports/my-campaign-local.json
bash scripts/training/audit_wandb.sh --control-dir /runs/my-campaign-001 \
  --output /reports/my-campaign-wandb.json
```

Both commands return nonzero when checks fail; output paths must be new.
The local report loads no model and makes no W&B request. The online audit
reads existing runs and downloads their artifacts to compare metrics, timings
and SHA-256 media digests. It does not start runs, upload files or change
credentials. Offline/disabled runs can use the local report, but cannot pass
an online upload audit.

Read timing fields according to their boundaries in the
[training guide](training.md#wb-and-local-evidence). Parallel job durations do
not add up to campaign wall time. Primary C→D accumulated progress RMSE,
conditional-head diagnostics and render-subset metrics describe different
quantities and populations; see
[evaluation semantics](training.md#evaluation-and-checkpoint-semantics).

The campaign retains resolved YAML, logs, process receipts, `campaign.json`,
`status.json` and `runtime_source_fingerprint.json` under `control/`. Per-run
outputs retain checkpoints, metrics, evaluation plans and optional media.
Keep these outputs, data and weights outside source releases.

## Evaluate a saved checkpoint

`grana.training.checkpoint_evaluation` accepts a committed checkpoint from an
unfinished or completed training run. It verifies the saved configuration,
save receipt and original held-out plan, then performs zero optimizer updates.
Use a new output directory outside the source run.

Restore the source experiment's model-cache and attention-backend environment
settings first. This direct module does not apply `scripts/train.sh` exports;
campaign settings are recorded in `control/campaign.json`.

```bash
: "${GRANA_SOURCE_RUN:?Set the source training run directory}"
: "${GRANA_CHECKPOINT_STEP:?Set an update with a committed checkpoint}"
: "${GRANA_EVAL_OUTPUT:?Set a new evaluation output directory}"
source .venv-qwen35-unsloth/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m grana.training.checkpoint_evaluation \
  --source-run "$GRANA_SOURCE_RUN" --step "$GRANA_CHECKPOINT_STEP" \
  --output "$GRANA_EVAL_OUTPUT" \
  --batch-size 32 --batched-vision-sdpa \
  --device cuda:0 --logging-mode disabled --dry-run
```

`--dry-run` checks provenance and prints counts without loading the model or
creating output. Remove it to evaluate. `--batch-size` is required; vision
batching defaults to off. These settings are independent of training's
physical batch and accumulation; see
[batch size and vision attention](TRAINING_CONFIGS.md#batch-size-and-vision-attention).

Online logging creates a fresh checkpoint-evaluation W&B run using the source
entity/project/group and a checkpoint-specific name. Offline and disabled
modes are also supported. Outputs include `checkpoint_evaluation.json`,
`evaluation_plan.json`, `evaluation-STEP.json` and logging receipts; enabled
dense diagnostics also produce PNG/MP4 media under `deploy_dense/`. Completion
is recorded after evaluation, rendering and logger finalization.
