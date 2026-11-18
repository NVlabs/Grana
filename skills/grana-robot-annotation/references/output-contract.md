# Portable request and candidate contract

Run commands from the Grana source checkout with its annotation dependencies
installed. All files written by this helper must use fresh output paths.

```bash
python scripts/annotation_skill.py prepare \
  --context /data/my-annotation/context.json \
  --output /data/my-annotation/prompt-bundle.json
python scripts/annotation_skill.py validate \
  --bundle /data/my-annotation/prompt-bundle.json \
  --candidate /data/my-annotation/model-answer.json \
  --output /data/my-annotation/validation.json
```

Both commands are offline. They do not read videos, sample frames, send API
requests, accept labels, or create training caches. Keep private context,
candidate JSON, and bundles outside the public source release.

For a legacy combined format-only smoke check, add `--mode combined` to preparation and use [synthetic-context.json](synthetic-context.json)
and [synthetic-candidate.json](synthetic-candidate.json) as the two inputs above.
They are hand-written fixtures with **no media**, not real annotation examples.
The listed evidence indices demonstrate the format only; validation cannot
verify actual image delivery. Do not submit the fixture as a real recording.
With no visual evidence available, derive a draft policy/intake first and defer
visual annotation; the real context requires a nonempty declared evidence set.

## Context

The evidence helper creates this context from your settings and recording.
Normally, edit the task settings and use the helper instead of assembling
these fields yourself. The context object has these exact fields:

| Field | Contract |
|---|---|
| `schema_version` | `grana.annotation_context.v1` |
| `task`, `task_instruction` | Nonempty task identity and observable objective |
| `stage_semantics` | `active_subtask` or `attained_state` |
| `source_identity` | Nonempty object identifying the exact recording/version; use content identity from the source adapter when available |
| `time_edges` | N+1 finite, strictly increasing native times for N source frames |
| `registry` | Ordered `{id,label,prompt}` objects; IDs are vocabulary, not asserted execution order |
| `evidence_frames` | Distinct integer frame IDs actually supplied to the model, in the declared recording or clip coordinates |
| `task_policy` | `{version,status,rules}`, with nonempty text fields and status `draft` or `calibrated` |

Task policy rules define stage identity and transitions. For the optional AQ round,
also supply the task's AQ distinctions; stages alone do not require an AQ rubric.
Rules may refine bootstrap heuristics, but do not override source-clock, evidence,
coverage, or candidate-only invariants. Declared source identity and evidence
indices are not a proof that the right images were supplied: the caller must
maintain and check that mapping.

For a clipped request, frame IDs are `0..N-1` within the clip. The caller keeps
the explicit mapping `native_frame = candidate_frame + source_start_frame` in
the private source receipt and identity; `time_edges` retain native timestamps.
Use only the supplied frame IDs rather than guessing the original offset.
Coverage applies to this clip, not unseen parts of the trajectory.

The helper prepares the instructions, task context, response schema and requested
mode. An AQ bundle also contains the saved first-round result. Supply instructions as model
instructions, context as user text, and each actual visual frame with its declared
candidate frame ID, native timestamp, and view identities. Adapt this packaging
to the selected vision API.
The skill files do not make a text-only model vision-capable or make different
providers' request formats compatible.

The policy name is `grana-robot-annotation`. The helpers manage file metadata;
users choose the task, recording and whether to annotate AQ.

The Responses driver replaces private `source_identity` with an opaque
`input_id` when sending context. The original context remains in
local provenance. This redaction does not establish that every camera named in
a mosaic declaration was independently verified by the provider.

## First-round candidate (default)

The model returns exactly `{"stages": {...}}`. The stage object contains
`segments`, `boundaries`, `omitted_stages`, `invalid_intervals` and `summary`, using
`grana/annotation/schema.py`. It partitions the clip completely, retains uncertain
identities and boundary brackets, and cites only supplied evidence frames.

There are no AQ or outcome fields. Validation records
`action_quality_status: default_normal_as_not_requested` and `outcome_status: not_requested`.
No empty quality array or nominal default is inserted. No AQ request is sent
unless the caller explicitly prepares and submits a quality bundle.

## Optional second-round candidate

Prepare it offline from the saved first round:

```bash
python scripts/annotation_skill.py prepare-quality \
  --stage-bundle /data/my-annotation/prompt-bundle.json \
  --stage-candidate /data/my-annotation/model-answer.json \
  --output /data/my-annotation/quality-bundle.json
```

Optionally pass `--evidence-bundle` with expanded observations and an AQ rubric.
The source identity, native clock, task, stage semantics and registry must match
round one exactly; all original evidence frames must remain available. The
parent's exact bundle and candidate stay in local provenance. Only the frozen
stage result is sent to the provider; private parent metadata stays local.

The helper supplies the complete response schema. The AQ annotations contain:

- `stage_annotations`: one object per first-round segment, in the same order.
  Each contains `stage_segment_index` (zero-based), `aq_intervals` and
  `aq_boundaries`. Revisited stages have their own indices.

Each AQ interval has `start_frame`, `end_frame`, `quality`, `reason` and
`evidence_frames`. Intervals must partition that frozen segment exactly, without
crossing its start/end or leaving gaps. Each internal AQ cut has one boundary
with `frame`, `lower_frame`, `upper_frame`, `reason`, `evidence_frames`. Brackets
contain the cut and remain within the segment. There is no AQ boundary entry
just for the frozen stage endpoints. Frame IDs remain clip-local, not reset
per stage. Context images from neighboring segments can support causal judgments.

Quality is `nominal`, `suboptimal`, `failure` or `unknown`. Uncertain/unresolved
stage identities and source-invalid spans require unknown AQ. Known AQ needs
cited evidence outside invalid spans. That check is not proof of visual accuracy.
The response cannot replace stages or add outcomes. Changing first-round decisions
requires a new bound AQ result; it does not mutate previously completed artifacts.

Validation reports `action_quality_status: annotated` even if all judged intervals
are unknown. This is distinct from skipping the round (`default_normal_as_not_requested`). Outcome
remains `not_requested`. No reward curve is inferred from either state.

Prepare evidence from the source using the supplied helpers. The
[workflow guide](api-workflow.md) gives executable API commands.

The skipped-AQ default applies Normal to valid unmarked time for downstream use;
it is not a model prediction and must be excluded from measured AQ accuracy.
Bounded campaigns preserve this status in their aggregate delivery and imported
draft provenance. Optional AQ preserves the original stage result.

## Legacy combined candidate (`--mode combined`)

This explicit compatibility mode uses `grana.annotation_bundle.v1`. Existing
joint stage/AQ/outcome evaluation scripts continue to use this contract. It is
not a fixed-stage second round.


The model returns exactly four fields:

- `stages`: the existing Grana stage-candidate object (`segments`, `boundaries`,
  `omitted_stages`, `invalid_intervals`, `summary`). See the generated schema and
  `grana/annotation/schema.py` for exact fields and identity statuses.
- `aq_intervals`: `{start_frame,end_frame,quality,reason,evidence_frames}` rows
  partitioning `[0,N)`. Quality is `nominal`, `suboptimal`, `failure`, or `unknown`.
- `aq_boundaries`: one `{frame,lower_frame,upper_frame,reason,evidence_frames}`
  for every internal AQ cut, in order. Inclusive brackets contain the nominal
  cut and remain within `[0,N]`.
- `outcomes`: `{start_frame,end_frame,stage_id,outcome,reason,evidence_frames}`
  for each maximal contiguous stage run. Outcome is `high`, `mid`, `low`, or
  `unknown`; unresolved/uncertain run identity requires unknown outcome.

All evidence references must belong to the supplied evidence set. Intervals are
half-open in the declared frame coordinates; reasons describe visible evidence or why
the decision is unknown. The JSON schema and local semantic validator both apply.
They reject invalid outputs instead of reordering, clamping, or filling labels.
Known AQ and outcome labels must cite at least one frame outside declared invalid
intervals. This is a structural check; one such frame does not prove the label.

## Meaning of validation

Validation checks the declared contract, policy/context integrity, complete
coverage, allowed evidence references, boundaries, and run alignment. It does
not authenticate the data or reviewer, assess visual accuracy, certify a
calibrated policy, or verify that a provider actually consumed the declared
images. Retain the original bundle and exact policy version when reviewing old
results; a later policy revision must not silently reinterpret them.

Candidates remain distinct from the accepted revision and explicit reward
supplement required by `grana/annotation/bridge.py`. In particular `failure`
is not automatically `regressed`, and unknown intervals are not usable nominal
training labels. Review, mapping calibration, and explicit acceptance remain
separate steps before the existing cache workflow.
