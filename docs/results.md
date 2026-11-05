# Reference results

These summaries describe measured workloads and checkpoints, with no guarantee
for every supported model or new training recipe. Timing and memory measurements
are in [benchmarks](benchmarks.md); measurement guidance is in
[training efficiency](training_efficiency.md). Private recordings, checkpoint
payloads and raw execution receipts are not included in the source release.

## Reference training results

Qwen3.5-0.8B and 4B each completed 400 pretrain updates followed by two
independent 200-update posttrain tasks, initialized from that model's final
pretrain checkpoint. The campaign used four GB300 GPUs, two workers per
pretrain run and one per posttrain run, A/B/C/D roles, eight time slots, four
camera views and gradient accumulation 2. This budget does not qualify a
5,000-update pretrain recipe.

Pretrain assessment was A-only across 49 tasks. Posttrain used fixed held-out
sets with 100 windows for Tractor and 14 for Spatula. The conditional metrics
below score POI slot 5: B uses the oracle stage focus and D uses C's predicted
focus. D RMSE here is conditional progress error, not accumulated C+D error.

| Model | Run | A whole-task RMSE | B oracle-focus RMSE | C stage accuracy | D predicted-focus RMSE |
|---|---|---:|---:|---:|---:|
| 0.8B | Pretrain | 0.2155 | — | — | — |
| 0.8B | Tractor posttrain | 0.1692 | 0.3076 | 0.3800 | 0.4424 |
| 0.8B | Spatula posttrain | 0.1617 | 0.3110 | 0.8571 | 0.3595 |
| 4B | Pretrain | 0.2136 | — | — | — |
| 4B | Tractor posttrain | 0.1629 | 0.3029 | 0.3800 | 0.4588 |
| 4B | Spatula posttrain | 0.1959 | 0.3023 | 0.9286 | 0.2938 |

The six runs completed 1,600 optimizer updates, 32 full evaluations and 64
checkpoints. Checkpoint tensors were finite, complete heads and adapters
loaded across phases, and final checkpoint restores reproduced the checked
predictions exactly. Posttrain held-out trajectories had no overlap with the
upstream pretrain inventory. These are results at the stated budgets, with
limited held-out coverage; they do not establish convergence.

## POI and LAST evaluation

A separate HotWheels D100 Qwen3.5-4B checkpoint at step 1000 was evaluated on
24 trajectories with three sampling seeds (11, 29, 47), up to 128 paired
targets per trajectory and round. Each request contained four cameras and
eight time slots using CTSP(16,4). POI selects slot 5 and LAST selects slot 7;
D uses the C stage at the selected slot. POI can observe up to eight later
source frames (0.8 seconds at 10 FPS), whereas LAST uses no future frames.
The modes also change the selected slot and history, so the comparison does
not isolate the effect of lookahead.

For each target, normalized accumulated error is
`(predicted_C + predicted_D - GT_stage - GT_within_stage) / stage_count`.
Mean trajectory RMSE weights trajectories equally. Micro RMSE pools the
squared errors of all sampled observations before taking the square root.

| Normalized metric | POI | LAST |
|---|---:|---:|
| Mean trajectory RMSE | 0.032417199 | 0.032538478 |
| Macro MSE (mean trajectory MSE) | 0.001237795 | 0.001245424 |
| Square root of macro MSE | 0.035182314 | 0.035290564 |
| Micro MSE | 0.001210797 | 0.001218610 |
| Micro RMSE | 0.034796505 | 0.034908593 |

The micro result includes 8,889 quality observations per mode across all
rounds, excluding repeatability anchors. Repeated draws count repeatedly.
Deduplicating to 7,896 targets gives micro RMSE 0.035087395 / 0.035203267
(POI/LAST). Those targets cover 9.7% of 81,417 source frames; neither result
is an exhaustive or duration-weighted full-video evaluation.

LAST wins on 13 of 24 trajectories and in two of three sampling rounds.
The paired trajectory bootstrap 95% interval for LAST minus POI mean
trajectory RMSE is [-0.000570, +0.000909], including zero. There is no
consistent winner. Error analysis found that rare stage-confusion events
dominated several large POI advantages; it did not show uniformly better
within-stage progress.

All 18,930 binary HTTP requests succeeded on three H100 servers, one server
per sampling round, at physical batch size 1. All 384 repeatability anchor
groups reproduced their eight-slot C IDs, D scores and D focus exactly.
These changing-window accuracy runs are separate from the fixed-input
[latency benchmark](benchmarks.md#http-latency).

## Objective comparison

A one-seed Tractor experiment compared CE+CDF with the legacy
distance-weighted objective for 100 posttrain updates using Qwen3.5-0.8B.
Both arms started from the same step-400 pretrain weights with fresh
optimizer, scheduler and sampling state. Each used one GB300, checkpointing,
100 training trajectories, 50 disjoint held-out trajectories, physical batch
32 and 128 global role rows per update (B80/C24/D24).

CE+CDF added a unit-weight CDF term to both stage CE and progress C51 soft
CE, and removed the legacy progress ceiling. The control's
prediction-dependent distance weighting and ceiling also differ, so this
does not isolate the causal effect of the CDF term.

All metrics below score POI slot 5. B uses the ground-truth stage focus.
Accumulated C+D compares `predicted_stage + D_progress` with the same
`GT_stage + GT_within_stage` targets in both arms. Both the prediction and
target are divided by Tractor's eight stages. This measures a common target
even when the predicted D focus differs between models.

| Final metric | CE+CDF | Legacy distance |
|---|---:|---:|
| Regular C accuracy, 200 POIs | 0.7000 | 0.6700 |
| Regular B progress RMSE | 0.3352 | 0.3383 |
| Regular normalized accumulated C+D RMSE | 0.1002 | 0.1120 |
| Dense C accuracy, 359 POIs | 0.6128 | 0.5627 |
| Dense normalized accumulated C+D RMSE | 0.0982 | 0.1016 |
| Stage-balanced C accuracy, 400 POIs | 0.6700 | 0.6575 |

Regular evaluation uses four interior quantiles per trajectory and has no
stage-7 POIs. Dense evaluation covers only two held-out trajectories at
stride 20 plus the final frame. The separate balanced diagnostic has 50
POIs per stage across all eight stages. Compare arms within a population;
do not pool these populations.

The candidate trailed the control at 50 updates and was modestly ahead on
stage and accumulated metrics at 100. Its balanced advantage is five correct
predictions out of 400. Within-stage progress remained weak: regular B RMSE
was 0.3352 versus 0.3458 for constant 0.5, and only 15.5% of B predictions
were within 0.1 of the target (control: 18.0%). These results do not establish
general superiority across tasks, model sizes, seeds or training budgets.

## Execution validation

These bounded checks establish integration behavior for tested inputs and
checkpoints; they do not extend the model-quality claims above.

- A new Linux x86_64 H100 environment passed dependency and interpreter
  checks, real causal-conv1d CUDA execution, Hopper TileLang dispatch and
  numerical checks, and CUDA compilation. A short real 4B batch-1 HTTP smoke
  also completed. Package download caches were reused and compiler caches
  were not explicitly cleared. This qualifies installation and execution,
  not cold-machine setup time, training capacity or model quality.
- Raw-to-cache construction was checked on two disjoint four-camera LeRobot
  episodes: all 536 cached target rows and 24 RGB samples matched independent
  calculations and native decoding. A 0.8B four-update pretrain followed by
  two-update posttrain completed with exact checkpoint reload parity.
- Target-composition construction was checked independently across 788 cached
  targets, 24 RGB samples and ABCD inputs for both the 9-second example and
  configured 15-second minimum duration. No model was retrained for those
  target checks, and they do not establish task calibration.
- Public inference replay across six final checkpoints matched all 112 checked
  progress values and 32 stage IDs, including C-to-D focus routing. Real HTTP
  requests to a 0.8B posttrain checkpoint reproduced 48 A/D values and 24 C
  stage IDs from offline inference exactly.
- Stage-choice augmentation completed four-update Qwen3.5-4B checks on two
  GB300 workers from the same raw initialization, with global batch 128,
  checkpointing, GPU preprocessing and evaluation at updates 2 and 4. Finite
  gradients, parameter updates, checkpoint reload parity, evaluation on both
  ranks and canonical unaugmented evaluation passed. This was an engineering
  qualification, not a quality or isolated throughput comparison.

The validation counts describe the original checks; run the
[current test suite](../CONTRIBUTING.md#development-workflow) when changing
the implementation.
