# Propose stage definitions, optionally build AQ cards

A stage vocabulary is an annotation design decision. Propose it from the
researcher's intended goal and demonstrations, then review it independently of
locating stages or AQ events in individual recordings. A model-produced list is
proposed, even if it appears coherent or compiles successfully.

## Inputs and goal coverage

Use the supplied overall goal, explicit constraints, task variants, initial/final
state descriptions, camera meanings, and available examples. Record whether
examples show successful, partial, failed, or recovering executions. If only a
textual goal is provided, the first vocabulary is a textual proposal; do not
claim visual validation or infer exact frame boundaries.

Separate requirements from observed strategies. A handoff seen in one example
may be optional; an explicitly required final placement remains part of the goal
when a failed example never reaches it. Do not assume two recordings share the
same registry because they share a task name or objects. Conversely, different
arm choices do not necessarily create different objectives.

## Stage proposal instructions for a vision-capable API

Supply the following instructions together with the actual task information and
examples. Do not send private task details in a reusable public skill.

Respect a supplied vocabulary or requested granularity. The objective grouping
below is a default for an unspecified decomposition, not a reason to merge the
user's explicit grasp/transport/placement stages. Stage-only work can stop after
the vocabulary and temporal stage annotation; an AQ rubric is not required.

> Propose an active-subtask vocabulary for the supplied goal. Identify meaningful
> objectives and their observable completion conditions. Group approach, grasp,
> transport, handoff, alignment, insertion adjustments, and recovery under the
> objective they serve unless the task explicitly makes one a separate goal.
> Do not create a stage for every motor action or assume the demonstrated order
> is mandatory. Include required goals absent from failed examples, but distinguish
> those goal-based proposals from objectives observed in the examples. State
> ambiguities and alternative decompositions rather than silently resolving them.
> Return a proposed profile and separate design notes; do not output temporal
> annotations, accepted labels, or a claim that the vocabulary is verified.

Return a `proposed_profile` with these compiler-compatible fields:

- `tasks`: explicitly scoped task/variant names supplied by the researcher.
- `task_instruction`: the intended overall result and relevant constraints.
- `registry`: ordered objects containing stable `id`, distinct `label`, and a
  short objective `prompt`; order is vocabulary organization, not proof of events.
- `cards`: one object per registry ID with exactly `id`, `objective`, `entry`,
  `completion`, `normal_operations`, and `visibility_limits`.

Keep `proposal_status: proposed` and `design_notes` outside that profile. Notes
should identify each objective's basis (explicit goal or example evidence),
meaningful alternative decompositions, task-variant distinctions, and unresolved
questions. Evidence notes cite supplied example observations without importing
private identifiers into the reusable policy. The compiler is a formatting
operation and does not approve those notes or the vocabulary.

## What makes a useful stage card

`objective` describes the state change being pursued. `entry` is the first
visibly committed motion toward that objective, including a purposeful approach;
do not wait for attainment. `completion` defines the observable end state used
to judge outcome. `normal_operations` identifies necessary ways to pursue it,
including appropriate recovery. `visibility_limits` states distinctions the
available views cannot establish, such as hidden retention, contact, or torque.

For easily confused neighboring stages, state the observable distinction and a
counterexample in the existing `entry`/`visibility_limits` text or scoped
`common_rules`; do not add unsupported fields to the card schema. When writing
settings directly, include these distinctions in `task_policy.rules`. For
example, if acquisition and transport are explicitly separate, moving an empty
gripper does not establish transport of the target. Mark hypothetical contrasts
as synthetic; they are decision examples, not evidence of events in a recording.

Keep observation guidance (which visible object relationships to inspect) distinct
from decision criteria (which label or transition they support). Define target
attributes and visibility limits so a distractor is not silently substituted for
the requested object. Use the existing fields or `common_rules`, as described in
[evidence-review.md](evidence-review.md#keep-observations-separate-from-criteria).

A shared exit convention is usually sufficient: leave an active objective when
the dominant purposeful action commits to another objective or the attempt is
abandoned. Necessary bimanual support stays with the current objective. An AQ
change does not itself change the stage. Replacing a displaced prerequisite may
revisit an earlier objective, while recovering the currently manipulated part
may remain in the current one. Distinguish these by the pursued goal.

Review whether the proposal covers all explicit goals, separates neighboring
objectives consistently, supports alternate successful strategies, and describes
retries without inventing new IDs. Where two decompositions remain plausible,
record the choice as a task convention. Do not manufacture a universal stage list.

## Optionally derive AQ within the selected stages

For each stage, compare normal handling with ineffective attempts, achieved-state
loss, intentional release/reorientation, and successful recovery. Use before,
action, and after observations; action names such as “adjustment” do not determine
quality. Write both the positive criterion and the nearest counterexample.

The optional [causal AQ contract](causal-aq-policy.json) supplies a draft starting
point: necessary handling and recovery are normal; a missed acquisition without
state loss is suboptimal; causing loss of an achieved task-relevant state is a
regressed candidate. Specify any task-specific constraints separately. If the
examples do not distinguish a necessary adjustment from an ineffective attempt,
retain unknown rather than inventing a speed, retry-count, or force threshold.

Define interval conventions explicitly. For the causal contract, the bad span
starts with the visibly supported causing action and ends when normal recovery
starts, even before physical state is restored. A later miss is a new error.
Stage outcome is separate: define observable full and partial completion only
when the task supports them; otherwise leave partial outcome unspecified.

## Compile and test without implying acceptance

[annotation_task_cards.json](../../../configs/annotation_task_cards.json) contains
one **invented** token-and-tray example. It includes no real task registry,
recording, or measured result. Replace its proposed profile with the researcher's
selected vocabulary; version the shared `common_rules` and `aq_contract`, and
keep the full contract's status `draft` until its declared review process is met.

From the source checkout:

```bash
python scripts/compile_annotation_settings.py \
  --cards configs/annotation_task_cards.json \
  --profile synthetic-token-tray-2 \
  --task synthetic-token-and-lid \
  --views views.json \
  --output settings.json
```

`views.json` maps view names to actual LeRobot camera features. The compiler
serializes the common rules, AQ contract, and selected cards into
`task_policy.rules`, alongside the matching registry and task instruction. Use
that settings JSON with `prepare_annotation_evidence.py`. Inspect the actual
bundle to verify the intended text was included; links alone are not API context.

Omit `task_owner_override` when no scoped owner policy is needed. An optional
override declares exact `task`, `profile`, and `reference` fields and requires
the actual policy through `--owner-policy`. A policy without declared scope is
rejected; do not carry a different task's rubric into the new one.

Freeze the vocabulary, rubric, and sampling procedure before an evaluation.
Judge proposals against independently reviewed objective boundaries and AQ
before accepting training labels. For exploratory refinement, request denser
frames around API-visible uncertainty/events and sample some apparently normal
windows; do not use hidden reference boundaries to select evidence. A joint
stage/AQ call with explicit cards and a separate stage-first orchestration are
different experiments and should be compared as such.
