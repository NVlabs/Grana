# Task-specific stage answer augmentation

The [recommended pretraining recipe](../configs/pretraining/default.yaml)
enables stage augmentation for pretraining tasks. It applies only to training
Role C; posttraining defaults to off. A task's ordered real stages move together
inside a full candidate list (A–J by default).
Remaining positions contain unique stage descriptions from other tasks in the
selected training inventory. Evaluation and inference use canonical stage
choices. The method is independent of
[training frame sampling](TRAINING_CONFIGS.md#two-layer-training-frame-sampling).

## Configuration

Pretraining recipes that extend `configs/pretraining/default.yaml` inherit:

```yaml
data:
  train_stage_augmentation:
    enabled: true
    contiguous_shift:
      enabled: true
      candidate_count: 10
    default_task_enabled: true
    task_overrides: {}
```

To exclude a task, add an override to your
[project recipe](project-configuration.md), using its exact cached
`Trajectory.task` name:

```yaml
data:
  train_stage_augmentation:
    task_overrides:
      task_to_exclude:
        enabled: false
```

Both `enabled` flags and the resolved task flag must be true. The recommended
pretraining recipe explicitly enables all three. Standalone recipes that do not
inherit this section retain false flags, `candidate_count: 10` and an empty
override mapping; selecting `training.stage: pretrain` alone does not enable it.
`default_task_enabled: true` enables tasks except those explicitly disabled.
The same task name across caches shares one policy. To disable augmentation for
a comparison run, set `data.train_stage_augmentation.enabled: false`.

Flags must be booleans, overrides accept only `enabled`, and `candidate_count`
must be an integer from 1 to 26. Unknown override tasks, duplicate normalized
real-stage descriptions, insufficient distractors and stage counts greater than
the candidate capacity fail before model initialization. The ten-candidate
recommendation requires enough distinct descriptions from other training tasks;
adjust the candidate count or exclude tasks whose inventory cannot support it.
Keep `model.gradient_checkpointing: true` for training.

## Candidate construction and supervision

Let $K$ be the real stage count and $N$ the candidate count. For each independent
training C draw, the sampler uniformly chooses an integer offset from
$0$ through $N-K$, inclusive. Canonical stage $i$ occupies candidate position
$\text{offset}+i$. Real stages stay contiguous and ordered; distinct distractors
fill every other position. One mapping is shared by all eight observations,
including repeated physical frames. For $K=N$, the offset is zero and no
distractors are needed. Uniform offsets do not imply uniform target-letter
frequencies.

The donor pool uses stage text from `dataset.records` after training inventory
selection, including manifest filters and splits. It excludes the current task
and descriptions matching its real stages. Text equality uses Unicode NFC and
collapsed whitespace; displayed text retains a validated donor description.
Sampling is uniform without replacement over unique descriptions, so repeated
trajectories do not increase a description's probability. Held-out records and
evaluation caches do not supply donors. Single-task training therefore needs
$K=N$ to satisfy the donor requirement.

The prompt presents the task's ordered stages separately from the answer list.
It asks for the first unfinished task stage, or the final stage if all stages
are complete. It includes descriptions, but no target labels, sampled offset,
inverse mapping or donor provenance.

The public data contract is implemented by
[`StageChoiceMapping`](../grana/data/stage_augmentation.py) and
[`RoleExample`](../grana/data/sampling.py):

| Field | Meaning |
|---|---|
| `num_stages`, `stage_prompts`, `stage_ids` | Real count $K$, canonical descriptions and unchanged per-frame stage IDs |
| `targets` | For augmented C, `stage_to_choice[stage_id]` for each observation |
| `stage_choices.candidate_prompts` | All $N$ descriptions in answer-letter order |
| `stage_choices.stage_to_choice` | The $K$ consecutive positions `offset + i` |
| `stage_choices.choice_to_stage` | Canonical stage IDs, with `-1` for distractors |
| `stage_choices.offset` | Start of the real-stage block |
| `stage_choices.pool_digest`, `algorithm_version`, `draw_id` | Donor inventory identity and replay provenance |
| `stage_choices.candidate_source_tasks` | Source tasks for distractors; empty tuples for real stages |

Unaugmented rows have `stage_choices=None`. `stage_answer_count(example)` returns
$N$ for augmented C and $K$ otherwise. Batches can mix these capacities using
the model's native A–Z logits and a per-row candidate mask. Images, frame order,
POI, canonical stage labels and progress targets retain their original meaning.

Both CE and CDF use the same softmax over all $N$ candidates, including
distractors. CDF averages the $N-1$ internal boundaries; $N=1$ has zero CDF.
For example, $K=1,N=10$ still supplies ten-choice supervision and nine CDF
boundaries. Loss coefficients, observation weights and role normalization are
unchanged. A distractor prediction decodes to `-1`, never a valid stage.

## Mixed C and D training

An enabled task cannot supply Role D, including during evaluation in the same
job. Other tasks can still use C→D. A new independent job resolves its own
explicit policy; loading checkpoint weights does not permanently disable D or
enable augmentation.

When `D > 0`, role quotas describe one two-forward cycle per worker. They require
`C >= D`, even A/B quotas and complete cycles in gradient accumulation:

- `C_paired` has quota `D` and samples only unaugmented C-eligible tasks, using
  their renormalized task weights. Each row produces exactly one D row from
  the detached C argmax at its designated POI, retaining the same observations.
- `C_independent` has quota `U = C - D` and samples all C-eligible tasks. It never
  produces D. Augmented C tasks require a positive independent quota.

The first forward contains `A/2` A rows, `B/2` B rows, `D` paired C rows and
`ceil(U/2)` independent C rows. The second replaces paired C with D and contains
`floor(U/2)` independent C rows. For quotas A16/B8/C24/D16:

| Forward | A | B | Paired C | Independent C | D | Total |
|---|---:|---:|---:|---:|---:|---:|
| First | 8 | 4 | 16 | 4 | 0 | 32 |
| Second | 8 | 4 | 0 | 4 | 16 | 32 |

Two workers and one cycle per update give global batch size 128 and gradient
accumulation 2. More generally, $W$ workers and $M$ complete cycles give global
batch size $W M (A+B+C+D)$ and accumulation $2M$. Odd `U` makes the two physical
batches differ by one; the global budget still counts actual supervised rows.
`D=0` uses independent ABC sampling with quotas per forward and global batch
size `workers × accumulation × sum(roles)`.

## Evaluation, replay and diagnostics

Evaluation always uses canonical prompts, $K$ candidates and canonical targets.
Its role plan records task-specific D exclusions. Dense progress evaluation
uses a supported route: an augmented task may use A when available. An augmented
task with only B/C cannot use a C→D dense route. Missing D metrics are not
zero-valued results. Frozen evaluation plans retain their recorded inputs.

Offset, donor selection and donor placement use separate random streams from
task, trajectory, POI, stride and frame sampling. The same inventory, policy,
seed, worker, lane and draw ordinal reproduce the mapping; ground-truth stage
labels do not determine it. Changing the donor pool does not perturb the offset
stream. Changing a task policy can change the paired-C task inventory and its
sample sequence. Sampler snapshots validate inventory and configuration before
restore; they are replay state, not optimizer or learning-rate resume state.

The resolved runtime summary and `stage_augmentation.json` record task policies,
candidate capacity, algorithm version and donor-pool digest. Supervision
receipts record paired and independent budgets. Detailed task counts, offsets
and bounded replay examples go to `stage_augmentation_artifacts/` and, when
enabled, W&B JSON artifacts. Two optional scalar curves use global consumed-row
counts summed across workers:

- `stage_augmentation/augmented_c_fraction`: augmented C rows / all C rows.
- `stage_augmentation/poi_distractor_fraction`: distractor predictions at the
  POI / augmented C rows; absent when its denominator is zero.

Training, evaluation and media use the optimizer step. Regression coverage
includes [policy and budget validation](../tests/test_stage_augmentation_config.py),
[mapping and routing](../tests/test_stage_augmentation.py),
[replay integrity](../tests/test_stage_replay_integrity.py),
[model integration](../tests/test_stage_choice_model.py) and
[canonical evaluation](../tests/test_checkpoint_evaluation.py).
These checks establish implementation contracts; training-quality claims
require a separate controlled comparison.
