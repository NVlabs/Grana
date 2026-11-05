# Pure API annotation: stages first, optional AQ

Run from the Grana source checkout using its annotation Python environment.
This workflow needs a task definition and RGB recordings; no human seed is needed.
The configured endpoint must implement the driver's Responses/JSON-schema
contract. Changing the model name alone does not adapt another provider's API.
Grana selects no endpoint or model. Before live requests, set
`GRANA_ANNOTATION_BASE_URL` and `GRANA_ANNOTATION_MODEL`, or pass explicit
`--base-url` and `--model` (CLI values take precedence). The default credential
variable is `GRANA_ANNOTATION_API_KEY`; `--api-key-env` selects another variable
name. Never put credential values in settings, prompts or checked-in files.
Offline planning, preparation and human labeling need no provider or key.
Completed requests can replay without a key using the same endpoint/model;
saved receipts are validated against their explicit recorded settings.
The endpoint must support HTTPS POST to `<base_url>/responses`, image inputs,
strict JSON-schema output and the requested reasoning effort. There is no
automatic provider, protocol, model or DNS fallback.

## Set the task and granularity

Create a private settings JSON with `views`, `task`, `task_instruction`,
`stage_semantics`, `registry` and `task_policy`. The
[synthetic settings](synthetic-settings.json) show the format, not real task rules.
Replace camera keys, task wording and all stage definitions. Each registry entry
contains `id`, `label` and `prompt`. Use `active_subtask` for the currently pursued
objective. A stage-only policy needs entry/transition rules, not invented AQ rules.

Choose the granularity here: for example, one `place_object` objective, or explicit
`acquire_object`, `transport_object`, `release_object` stages. Each needs an
observable definition. Reuse the same IDs across recordings and retries. If no
vocabulary exists, propose it using [stage-task-cards.md](stage-task-cards.md)
before running temporal annotation. Do not infer it silently in every API request.
The task-card compiler is available for richer profiles; hand-authored settings
are sufficient for stages and do not require an AQ rubric.

For task-specific distinctions, counterexamples, bounded visual review and an
honest comparison plan, read [quality-improvement.md](quality-improvement.md).
Those checks complement the commands below; schema validation alone cannot
establish that a stage or action-quality judgment matches the images.

## Bounded two-round campaign (recommended)

Plan offline first. Paths are placeholders; use a private output outside the
checkout and raw source. Include the task's AQ rubric in settings if AQ will be
requested. The endpoint, settings, policy and budget are frozen per campaign.

```bash
python scripts/annotate_episode.py \
  --source /data/recordings/my_task --episode 0 \
  --settings /data/annotation/settings.json \
  --output /data/annotation/episode-0 \
  --api-dir /data/annotation/campaign --request-name episode-0 \
  --max-requests 64
```

Add `--run` to that command to submit **stage-only** requests. No AQ or outcome
fields are requested in round one. The controller bisects the native timeline
until each stage request fits both 120 seconds and 96 sampled images by default;
`--stage-seconds`, `--stage-stride` (default 2 seconds) and `--max-images` control
these bounds. Coverage is complete; sampling is sparse and does not establish
frame-perfect boundaries. Equal adjacent stages are coalesced and request seams
retain boundary uncertainty.

The independent first-round result is `candidate/candidate.json`, paired with
`evidence/bundle.json`. Validation records
`action_quality_status: default_normal_as_not_requested` and
`outcome_status: not_requested`. Unmarked valid time defaults to Normal when AQ
is skipped. That default is not an API judgment and is excluded from measured
AQ accuracy. Stop here for stage-only/all-normal datasets.

## Optional second round: AQ within frozen stages

Use the same command and output, adding **`--run --with-quality`**. This can be
requested from the beginning or after inspecting the completed stage result.
Completed stage requests are reused from receipts before any AQ request starts.

Each AQ request contains only one frozen stage occurrence. Longer occurrences
are bisected again: defaults are at most **10 seconds**, sampled every **0.2
seconds**, and at most **96 images**. Adjust `--quality-seconds` and
`--quality-stride` in the original campaign arguments when needed. Requests use
actual RGB mosaics from each clip; descriptions do not replace images. Repeated
stage IDs remain separate occurrences. AQ returns only
`nominal/suboptimal/failure/unknown` spans; it cannot replace stages or invent
outcomes. Source-invalid or unresolved stage spans require unknown AQ.

The final `quality/` directory contains its bundle, candidate and validation.
Its bundle locks the original full-stage candidate; each child request locks an
exact local projection of one occurrence. Local composition verifies every
child API receipt and maps native frames back into the episode. The aggregate
is explicitly a local-composition receipt, not a fictitious single API response.
All first-round evidence remains in the aggregate alongside the denser AQ frames.

All stage/AQ requests share `--api-dir` and the persisted HTTP-attempt budget;
failed attempts count. AQ request count depends on the observed stages, so it is
not known exactly at planning time. A failed optional round preserves the stage
result and completed chunks. Re-running identical arguments reuses completed
requests. Exhausted or uncertain requests require inspection; the controller
never silently repeats an uncertain paid call or enlarges a saved budget. Use a
new campaign when changing frozen inputs, and retain the original receipts.

Review stage identities and boundary brackets before trusting the result. Human
review is not a mandatory manual gate for pure API execution. If stages need
correction, preserve a separate validated candidate and create a new AQ run;
never edit delivery receipts to claim a human correction came from the API.
Splitting limits request size, not semantic error or all possible timeouts.

## Lower-level evidence and refinement

`scripts/prepare_annotation_evidence.py` and `scripts/annotation_skill_api.py`
remain available for individual clips, and `annotation_skill.py prepare-quality`
constructs a frozen AQ bundle offline. Use the bounded campaign above for
multi-stage/long-video AQ rather than submitting a whole recording in one call.
For ambiguous objects or conflicting candidates, follow the optional
[evidence review procedure](evidence-review.md).

## Add evidence without changing the stage result

For first-round boundary refinement, `scripts/refine_annotation_evidence.py`
accepts the stage evidence and completed stage candidate. It copies original
mosaics and adds bounded dense observations around proposed transitions, without
AQ labels or hidden human labels. Send the expanded bundle for a revised stage
candidate if needed. A stage-only run does not fabricate a nominal-AQ audit.

For AQ, use the expanded evidence context with the original or revised stage
result you choose to freeze:

```bash
python scripts/annotation_skill.py prepare-quality \
  --stage-bundle /data/annotation/episode-0/evidence/bundle.json \
  --stage-candidate /data/annotation/episode-0/stages/candidate.json \
  --evidence-bundle /data/annotation/episode-0/dense-evidence/bundle.json \
  --output /data/annotation/episode-0/dense-quality-bundle.json
```

Submit this bundle with `dense-evidence/evidence.json`. It must retain every
original sample. You can also use the evidence preparer's `--extra-frames` list
(native frame IDs) to add observations around contact, uncertainty or recovery
and audit apparently routine behavior. These are candidate-driven evidence
requests, not guarantees of improved accuracy. Stay within the declared budget.

## Resume, batch and validate

Completed identical API requests reuse their receipts. Use fresh output paths for
changed stages, task rules, evidence or model settings. Keep the same campaign
directory and total request limit when resuming; a timeout may have been billed.
Do not routinely enable `--retry-uncertain`.

`scripts/annotation_windows_api.py` can run a manifest of prepared stage bundles
or prepared quality bundles with bounded workers and a shared request budget.
Finish stage requests before preparing dependent quality requests. No provider
calls occur during planning or validation.

Use `scripts/annotation_skill.py validate --bundle ... --candidate ...` for either
round. Preserve the full skill folder with `--policy-dir` when freezing a policy
snapshot; use the same files through preparation, submission and refinement.
The two-round bundles bind evidence to their exact context. Regenerate
incompatible evidence with the helpers rather than editing its schema identifier. Legacy combined experiments use
`--mode combined` and their original contract and policy snapshot.

After updating the skill, retain old bundles with their original policy snapshot.
Prepare a fresh bundle and result directory to use revised visual instructions;
keep previous results and their request records unchanged.
