# Annotate raw LeRobot recordings

Grana can propose task stages from unannotated **LeRobot v2 RGB recordings**.
The workflow is `raw → plan → API probe → draft/refine/review → local video
review → stage JSONL`. Reward-cache construction is a separate final step:
it additionally requires explicit action quality (AQ) and stage outcomes.
The stage annotator does **not** generate those reward labels.

Annotation runs on CPU and sends selected camera mosaics to the configured
inference API. It does not start training, Ray, W&B, or the checkpoint server.
The 2026-11-01 qualification completed the real structured-output preflight and
all six visual calls on two DROID Spatula recordings (535 and 490 frames).
These recordings have whole-episode language but no temporal stage/AQ labels.
Both produced three stage candidates; their boundary-uncertainty durations were
1.933 and 1.667 seconds. Real no-key/no-network resume, synchronized browser
playback, revision save, JSONL export and full-frame MP4 rendering passed.
This establishes the workflow, not measured annotation accuracy or human label
acceptance. The reward bridge passed synthetic end-to-end tests; it has not
created reward caches for these pilot recordings, which still need independent
AQ/outcome supervision. A bounded audit of five existing annotated Spatula
recordings found unlabeled trailing frames and correctly declined conversion
without inventing their stages.

## Pure API annotation: two rounds

Use [grana-robot-annotation](../skills/grana-robot-annotation/SKILL.md) for the
portable API workflow. **Sub-stage annotation is the default first round.**
Action quality is an optional second round over the saved stages. It cannot
change their identities or boundaries. Human seed templates are not required.
Neither round requires stage outcomes or numerical progress.

Evidence preparation for this workflow supports native LeRobot v2 and v3
recordings with the same commands. For v3, it selects the episode's rows and
camera time ranges from shared files, then decodes each requested frame by its
presentation timestamp. Clip-local frame IDs retain their native source offset.
Evidence preparation reads the episode's clock and camera data. It does not
use human labels, robot state or actions to determine what the model sees.

Follow the [executable workflow guide](../skills/grana-robot-annotation/references/api-workflow.md)
for settings, evidence preparation, API calls, optional AQ, denser evidence and
resume. The [output contract](../skills/grana-robot-annotation/references/output-contract.md)
defines both rounds. `default_normal_as_not_requested` means AQ was skipped and valid unmarked time
defaults to Normal; it is not a model-derived nominal
label or an unknown judgment. Results remain candidates until separately reviewed
and accepted for their intended use. Local schema tests do not measure accuracy.

The [quality improvement guide](../skills/grana-robot-annotation/references/quality-improvement.md)
explains how to define task-specific decision criteria and counterexamples, add
visual evidence, and compare results. It distinguishes current tools from
agent-guided review: automatic visual adjudication and a dedicated two-round
accuracy benchmark are not yet implemented. A frozen stage or valid citation is
not proof that the visual judgment is correct.

For difficult clips, the [evidence review procedure](../skills/grana-robot-annotation/references/evidence-review.md)
separates visible object observations from stage/AQ decisions and shows how to
request and compare bounded repeat candidates. Review notes are maintained by
the agent; no extra observation call, human seed or new output schema is required.
The stage and AQ policies include object-identity and visibility checks in the
instructions actually sent to the API. Accuracy gains still require evaluation.

The source checkout supplies the skill, scripts and schemas. Retain a full policy
snapshot with `--policy-dir` for reproducible requests. The Python wheel alone
does not contain the complete executable workflow.

#
### Bounded campaigns and the local playground

Use `python scripts/annotate_episode.py --help` for the stage-first campaign.
Without `--run` it plans offline. With `--run`, it saves stage annotations first;
`--with-quality` then asks AQ separately inside each stage (10-second clips by
default). Reuse identical arguments to add AQ later or reuse completed requests.
The [skill workflow](../skills/grana-robot-annotation/references/api-workflow.md)
describes bounds, receipts, shared budgets and failure recovery.

For the local human/API/seed-example interface, run:

```bash
python -m tools.annotation_playground.launcher \
  --config /data/annotation/recipe.yaml \
  --workspace /data/annotation/playground --port 8840 --max-requests 64
```

Open the printed loopback home URL. The service runs in the foreground and stops
its child services on exit. API generation starts only after an explicit click.
Generate stages by default; select “Also annotate AQ after stages” or use “Add AQ”
on a completed stage result. Imported drafts never overwrite saved human labels.
The seed workflow enables targets only after every selected held-out episode has
a completed comparison against unchanged labels/results. This checks comparison
completeness, not an accuracy threshold. No key or paid call is needed to inspect
the pages or annotate manually. All canonical data and provenance remain English;
the UI language switch changes display text only.

To import completed results from a command-line campaign:

```bash
python -m grana.cli draft-labels --config /data/annotation/recipe.yaml \
  --from /data/annotation/episode-0/candidate --dry-run
```

Use `quality/` instead of `candidate/` when the optional round completed. Review
the dry-run, then remove `--dry-run` to create drafts. Outcomes remain unset until
separately reviewed for full training labels. Human draft submission and cache
construction retain their existing review requirements.

## Propose the vocabulary, then optionally define stage-specific AQ

Use the portable [stage-design skill](../skills/grana-robot-annotation/references/stage-task-cards.md)
with a task goal, initial/final-state descriptions and representative chronological
examples. The skill proposes active objectives, entry/completion criteria,
normal operations, failure counterexamples and visibility limits. Inspect the
proposal before assigning stable stage IDs. Acquisition, handoff, alignment and
recovery usually belong to the objective they serve. A user-declared vocabulary
may explicitly separate these operations; preserve that granularity and define
observable distinctions. Proposals can remain unresolved. Stage-only settings
do not require the richer task-card compiler or its AQ contract.

Store real task definitions privately. The shipped
[task-card example](../configs/annotation_task_cards.json) is entirely synthetic.
Copy its schema, replace its profile with your reviewed proposal, and provide a
JSON mapping of view names to your recording's camera feature keys. Compile it
from the **source checkout**:

```bash
python scripts/compile_annotation_settings.py \
  --cards /data/private/my-task-cards.json --profile my-profile --task my-task \
  --views /data/private/views.json --output /data/private/settings.json
```

Compilation binds the selected objective cards and AQ contract into the text
actually sent to the model; links alone do not transmit rules. It validates the
format and explicit task/profile binding, not the correctness of proposed stages.
The optional `--owner-policy` applies an explicitly declared scoped override.
Freeze the vocabulary and policy before comparison. Stage identity is returned
in round one. Only request AQ after saving that result; vocabulary design is a
separate explicit task, not a hidden API call.

The compiler, refinement and evaluation scripts, JSON templates and skill files
are distributed in the source archive. The Python wheel alone is insufficient
for this annotation workflow; `grana.annotation.refinement` also depends on the
source-checkout validator. The release contains methods and synthetic examples,
not the private task catalogues used in local experiments.

### Legacy combined stage and AQ experiments

The following older joint workflow is explicit compatibility mode. For the default
two-round workflow use the guide above. After installing the dependencies below, copy
[synthetic-settings.json](../skills/grana-robot-annotation/references/synthetic-settings.json)
to a private settings file. Replace its camera feature keys, task, stage vocabulary,
and draft task rules with the recording's actual definitions. This file contains
invented examples; it does not establish a task's human annotation standard.

```bash
python scripts/prepare_annotation_evidence.py \
  --source /data/raw/my_lerobot_v2 --episode 0 \
  --settings /data/private/settings.json --output /data/private/aq-pilot/evidence --mode combined \
  --stride-seconds 0.5 --max-images 96
python scripts/annotation_skill_api.py \
  --bundle /data/private/aq-pilot/evidence/bundle.json \
  --evidence /data/private/aq-pilot/evidence/evidence.json \
  --output-dir /data/private/aq-pilot/candidate \
  --api-dir /data/private/aq-pilot/campaign --request-name episode-0 --max-requests 24
python scripts/evaluate_annotation_aq.py \
  --input /data/private/aq-pilot/comparison.json --output /data/private/aq-pilot/metrics.json
```

Preparation validates every declared camera's native clock and samples RGB without
reading annotation, action or progress columns. Optional `--start-frame` and
`--end-frame` select a half-open native-frame clip. Candidate frame IDs start at
zero within that clip; `source_receipt.json` retains the offset and native times.
Image limits fail explicitly instead of dropping frames. A sparse pilot can miss
brief contact events: use denser observations before assigning temporal precision.
For a focused second pass, `--extra-frames /data/private/extra-frames.json` accepts
a JSON array of distinct native frame IDs within the selected clip. These are
added to the uniform grid without removing existing samples; the combined set
must fit `--max-images`. Record why the extra frames were selected. In a blinded
evaluation, select them from model uncertainty or a fixed sampling protocol,
not from withheld human event boundaries.

The Responses driver checks the saved instructions, selected frames, timestamps
and image contents before sending. It removes private source paths and
trajectory identities from request context, while retaining local provenance.
Mosaic delivery receipts establish the submitted bytes, not the provider's actual
visual understanding. The configurable model/endpoint defaults match the existing
stage workflow below; another provider must support the same Responses contract.
All cases in a pilot should share `--api-dir` to enforce one HTTP-attempt budget.
Only candidates are written; no labels are accepted or converted to cache targets.

The comparison JSON contains `schema_version: "grana.aq_evaluation_input.v1"`,
the bundle's `time_edges`, `predicted_intervals` copied from candidate
`aq_intervals`, and `human_intervals` with `label` and half-open frame or native-time
bounds. Check source alignment and endpoint conventions before constructing it.
The scorer's `--help` documents the exact format. Report `scored_seconds` and
`scored_fraction` alongside time-weighted agreement. Human gaps, unknowns and
overlaps are excluded; predicted unknowns still count as missed reference labels.
This measures agreement on explicitly labeled time, not full-video detection
accuracy. This older interval scorer does not measure event boundaries; use the
explicit-window event diagnostics below when those references are available.
Automatic human-label import is not provided.

For a bounded automatic second pass, use the completed coarse candidate:

```bash
python scripts/refine_annotation_evidence.py \
  --baseline-dir /data/private/aq-pilot/evidence \
  --candidate-dir /data/private/aq-pilot/candidate \
  --settings /data/private/settings.json --output /data/private/aq-pilot/dense-evidence \
  --max-extra-frames 64 --max-total-images 256 --radius-seconds 1 --dense-hz 10
python scripts/annotation_skill_api.py \
  --bundle /data/private/aq-pilot/dense-evidence/bundle.json \
  --evidence /data/private/aq-pilot/dense-evidence/evidence.json \
  --output-dir /data/private/aq-pilot/dense-candidate \
  --api-dir /data/private/aq-pilot/campaign --request-name episode-0-dense --max-requests 24
```

The planner uses model boundary uncertainty, non-nominal endpoints, stage changes
and a deterministic nominal audit window. It reads no human labels. Original
sampled frames and mosaic bytes remain unchanged; whole local windows are added
within the declared budget. `refinement_plan.json` records selected/skipped
windows and actual overlap coverage. Unselected events can still be missed.
Use the same frozen `--policy-dir` on every preparation/submission/refinement
command when comparing versions. Refinement itself makes no network request.
`--resume` verifies exact completed inputs and artifacts; incomplete outputs
require investigation rather than overwrite.

### Query shorter stage windows in parallel

A long recording can exceed a provider's input limits or take longer than a
gateway permits. An HTTP failure alone does not identify the cause: inspect the
receipt before changing sampling or retrying. A gateway timeout may still have
consumed a billed request. Authentication or schema errors need their own fixes.

With a completed coarse candidate, the offline
[window preparer](../scripts/prepare_annotation_windows.py) groups adjacent
predictions of the same stage and splits long runs into bounded cores. The
defaults are at most 30 seconds per core, 2 seconds of context on each side,
0.5-second sampling and a hard 128-mosaic limit per request. Cores partition the
original clip; context can overlap. A revisited stage stays a separate run.
The predicted stage selects evidence only: the new request retains the complete
vocabulary and can disagree with that prediction.

```bash
window_args=(
  --baseline-dir /data/private/aq-pilot/evidence
  --candidate-dir /data/private/aq-pilot/candidate
  --settings /data/private/settings.json
  --output /data/private/aq-pilot/windows
  --policy-dir /data/private/frozen-policy
)
python scripts/prepare_annotation_windows.py "${window_args[@]}"
# Choose actual IDs from the resulting window_plan.json.
python scripts/prepare_annotation_windows.py "${window_args[@]}" --resume --window-id window_0000
python scripts/prepare_annotation_windows.py "${window_args[@]}" --resume --window-id window_0001
```

Preparation reads no annotation columns and makes no API calls. It binds the
coarse result, raw clocks, camera order, settings and frozen policy. A limit
violation fails explicitly; choose smaller cores or another declared sampling
protocol instead of silently dropping images. If no coarse request has succeeded,
start with bounded native-frame clips using `prepare_annotation_evidence.py
--start-frame ... --end-frame ...`; a stage prediction is not required for that
first pass.

Create a private JSON manifest of prepared inputs and distinct output directories:

```json
[
  {
    "name": "episode-0-window-0000",
    "bundle": "/data/private/aq-pilot/windows/windows/window_0000/bundle.json",
    "evidence": "/data/private/aq-pilot/windows/windows/window_0000/evidence.json",
    "output_dir": "/data/private/aq-pilot/candidates/window_0000"
  },
  {
    "name": "episode-0-window-0001",
    "bundle": "/data/private/aq-pilot/windows/windows/window_0001/bundle.json",
    "evidence": "/data/private/aq-pilot/windows/windows/window_0001/evidence.json",
    "output_dir": "/data/private/aq-pilot/candidates/window_0001"
  }
]
```

```bash
python scripts/annotation_windows_api.py \
  --manifest /data/private/aq-pilot/window-jobs.json \
  --api-dir /data/private/aq-pilot/campaign --max-requests 24 --workers 16 \
  --policy-dir /data/private/frozen-policy --max-attempts 1 --timeout-seconds 480
```

The runner validates all inputs before starting, defaults to 16 workers
(configurable from 1–32), and shares one persisted **HTTP-attempt budget**,
including previous invocations and retries. Group independent recordings or
windows into one manifest; the same runner also accepts full-clip bundles.
Requests that depend on a coarse result belong in a subsequent batch. The worker
count controls concurrency, not the total request allowance or a provider's
capacity. Reduce it if the endpoint rate-limits the batch.
Completed identical requests are reused without a new call. A failed job is
reported separately while other jobs finish. Uncertain transport failures are
not retried automatically. With `--max-attempts 1`, that request's allowance is
exhausted after one attempt; `--retry-uncertain` cannot increase it. After deciding
to retry, preserve the failed record and use a new request name and output
directory within the remaining shared campaign budget. The flag only permits a
same-request retry when its originally declared attempt allowance has room.

Each response uses clip-local frame IDs. `window_plan.json`, `source_receipt.json`
and `window_receipt.json` retain the source offset and core bounds. Map intervals,
boundary brackets and evidence citations back to native frames; compare only
the core to avoid double-counting overlap. Context is needed to distinguish a
miss from loss after retention and to see normal recovery. A clipped beginning
or end may leave the event or outcome unknown; widen the relevant window if
needed. These helpers do not stitch responses, infer a full-episode outcome,
accept labels, or generate cache targets. Keep disagreements at shared boundaries
visible during review.

For event diagnostics, prepare the explicit reference schema documented in
[scripts/evaluate_annotation_events.py](../scripts/evaluate_annotation_events.py)
and run `python scripts/evaluate_annotation_events.py --input comparison.json
--output event-metrics.json`. It reports known-window AQ/stage agreement, event
matches, boundary distances, false-alarm seconds on reviewed normal windows,
and continuation into explicitly normal recovery. One predicted event cannot
match multiple reference events. Unreviewed gaps are never negative examples;
predicted unknown and uncertain reference brackets remain visible. Declare
`independent_agent_candidate` or `human_reviewed` honestly: the scorer does not
verify that provenance, establish accuracy, or accept labels for training.

For policy comparisons, freeze the entire skill directory before calibration and
pass that snapshot via `--policy-dir` to preparation, submission and validation.
Keep physical trajectories disjoint between calibration and holdout. Human
reason text is not a class dictionary: the same activity can have different
grades. Imported `normal` defaults, prediction-origin tracks and annotation export
success markers do not establish human-reviewed nominal labels or completed RGB.

The generic [causal AQ draft](../skills/grana-robot-annotation/references/causal-aq-policy.json)
provides distinctions to adapt to each proposed objective: necessary handling,
ineffective acquisition, loss of an achieved state, intentional release, and
normal recovery. Specify the timing convention and visibility limits in the
private task policy. It is not a learned task catalogue or an automatic mapping
from historical `failure` labels to progress resets.

A subsequent private pilot exercised the combined driver on ten paired raw-video
trajectories, with six for policy development and four separate trajectories for
comparison after freezing the candidate policy. This qualified native-clock
alignment, delivery, structural validation, reuse and time-weighted comparison.
The candidate improved recall on explicit abnormal intervals but left many
intervals unknown; the labels contained no explicit nominal comparison cohort.
It does not establish reliable automatic AQ annotation. Local attempt boundaries,
parallel-arm grading and small-object retention still need task-specific examples
and review. Keep the default policy's bootstrap status until broader evaluation.

## Install and configure

Run from the Grana source checkout. For a standalone CPU annotation environment,
follow the [CPU installation guide](installation.md#cpu-labeling-and-cache-construction):

```bash
uv python install 3.12.12
uv venv --python 3.12.12 .venv-annotation
uv pip install --python .venv-annotation/bin/python -r requirements/labeling-cpu.txt
source .venv-annotation/bin/activate
```

Reuse an already qualified annotation environment when available. The CPU pins
include the API client and schema validator; no model checkpoint or GPU runtime
is required. For an existing full Grana environment, `requirements/annotation.txt`
adds the API dependencies without creating a separate CPU environment.

`ffmpeg` and `ffprobe` must be on `PATH`. Credentials are read only from the
process environment variable `GRANA_ANNOTATION_API_KEY` by default; the recipe, receipts
and commands do not contain its value. Provision that environment variable
through your normal credential mechanism before making API calls.

Start with [configs/annotation_example.yaml](../configs/annotation_example.yaml).
It contains an invented two-objective token/tray task, episode 0 and placeholder
camera keys. Use the skill to propose your task's objectives, then replace the
example with the reviewed private vocabulary, episode IDs and actual camera
mapping. Each stage has an immutable string `id`, a downstream cache `label`,
and an observable objective `prompt`. Stages may be omitted or revisited; the
ordered vocabulary does not assert that every stage occurred. The release does
not distribute private task-to-stage catalogues.

```bash
export GRANA_ANNOTATION_SOURCE=/data/raw/my_lerobot_v2
export GRANA_ANNOTATION_OUTPUT=/data/grana/annotation-runs/my-task-pilot
grana annotate --config configs/annotation_example.yaml --plan
```

Paths in annotation YAML resolve relative to that YAML file, after expanding
environment variables and `~`. Unresolved variables fail. Output must be
outside code checkouts and the raw source. Use a new output directory for a
different source, registry, camera order or recipe. A matching existing run is
reserved for recovery and cache reuse.

The source adapter reads only `frame_index`, `episode_index`, and `timestamp`
from parquet, plus source metadata and the configured RGB videos. It does not
read human annotation columns, action quality, progress, robot state or actions.
Native v3 is unsupported. Every camera must have the declared shape, matching
frame count, and video PTS aligned to the recorded source timestamps after an
explicit initial offset. This establishes the recorded mapping, not an
unrecorded physical synchronization guarantee.

The source clock has **N+1 time edges** for N frames. The last edge is the last
timestamp plus the last observed positive delta, recorded in the clock receipt.
All segments use half-open source-frame intervals `[start_frame,end_frame)`;
times come from the source clock, not an assumed playback FPS.

## Probe, annotate and resume

No endpoint or model is selected by Grana. Before a live request, set
`GRANA_ANNOTATION_BASE_URL` and `GRANA_ANNOTATION_MODEL`, or put explicit
`api.base_url` and `api.model` values in your private YAML. Explicit YAML values
take precedence over environment values; null or omitted values use the
environment. Planning and human labeling need neither these settings nor a key.
The driver scripts accept `--base-url` and `--model` with the same precedence.
Use `api.api_key_env` or `--api-key-env` to name a different credential variable;
never put the credential value in configuration or command arguments.

The provider must implement HTTPS `POST <base_url>/responses`, multimodal image
inputs, strict JSON-schema output, and the selected reasoning effort. An
OpenAI-compatible chat-completions endpoint alone is insufficient. Placeholder
endpoints/models are not usable configuration. No DNS rewriting or provider
fallback is performed. For example, provision your chosen provider settings
through your shell or secret manager before using the commands above.

```bash
grana annotate --config configs/annotation_example.yaml --preflight
grana annotate --config configs/annotation_example.yaml
```

The small probe requests `high` reasoning and 4,096
output tokens. Visual draft, boundary refinement, and independent review each
request `xhigh` with the configured token limit. The API must acknowledge the
exact model and reasoning effort and return completed, schema-valid JSON.
Refusals, incomplete responses and invalid outputs fail without a lower-effort
or alternative-model fallback. Requests specify `store: false`.

Each window gets a sparse draft, denser images around proposed boundaries,
then a separate review request using the same model. This is independent API
review, **not human acceptance**. Windows are at most 60 seconds. Each request
must fit `max_images_per_request`; exceeding it stops the run without silently
dropping images. Plan shorter windows in a new run when necessary. Candidate
citations must refer to frames actually sent in that request. Different-stage
window joins retain explicit boundary uncertainty instead of claiming exact
temporal refinement. `tile_size` resizes only the API evidence mosaic; raw
videos and later cache RGB are unchanged by annotation.

The plan reports three annotation calls per window plus one probe. The campaign
`api.max_requests` budget counts reserved HTTP attempts, including retries;
it is not a dollar or token-price estimate. Each logical request also has
`max_attempts`, `max_output_tokens`, and `timeout_seconds`. Limits are persisted
and cannot be raised by restarting with larger values. HTTP 429 may retry
within those limits. A timeout, connection failure, HTTP 408/5xx, nonterminal
response or interrupted process leaves an uncertain attempt and stops.

Resume with the same command and unchanged recipe/source:

```bash
grana annotate --config configs/annotation_example.yaml
```

Completed responses are checked and reused automatically; a fully completed run
can resume without a key and makes no new HTTP requests. Keep the explicit
endpoint/model configuration used by that run: request identity includes both.
Saved candidate validation uses its recorded values, independent of current
provider environment settings. Changed evidence images,
source data, timestamps, configuration or camera
order are rejected. Do not delete receipts or the budget to bypass this record.
If a previous request has an uncertain outcome, inspect its receipt first.
Only an explicit decision to risk duplicate billing permits:

```bash
grana annotate --config configs/annotation_example.yaml --retry-uncertain
```

Do not make that flag a routine resume default. An uncertain request may have
completed and been billed by the provider even when no response reached Grana.
The explicit retry still obeys the original attempt and campaign limits.

Run artifacts include `config.json`, `manifest.json`, per-window evidence and
draft/refine/review JSON, per-episode `final.json` and `report.json`, and
`api/budget.json` with request records. Records retain request status and
available usage, without input base64 images, credentials or model
reasoning text. Evidence PNGs are separate local artifacts. Keep these outputs
outside the source archive.

## Review the original videos

```bash
grana review-annotations --run "$GRANA_ANNOTATION_OUTPUT" --port 8788
```

Open `http://127.0.0.1:8788/`. For a remote host, forward that port manually
using the app's port-forwarding UI or, from your local machine:

```bash
ssh -N -L 8788:127.0.0.1:8788 YOUR_REMOTE_HOST
```

The server binds to loopback. It is independent of checkpoint inference on
port 8787. Its page plays the original allowlisted camera videos using the
recorded offsets and provides frame navigation, segment/boundary tables,
candidate JSON editing and immutable revision saves. Browser seeking is for
review; exact extracted frame mosaics remain available in the evidence records.

For a standalone MP4 with synchronized cameras, stage identity, source time,
uncertainty brackets and the pending-human-review banner:

```bash
python scripts/render_annotation_video.py --run-dir "$GRANA_ANNOTATION_OUTPUT" \
  --episode 0 --output /data/grana/rendered/episode_000000.mp4
```

The CPU renderer supports one to four cameras, requires uniform native time
edges and verifies every camera's frame timestamps. Nonuniform clocks fail
explicitly rather than being converted to an invented FPS. Every source frame
is rendered. The output must be a fresh path outside the run, raw source and
checkout. The script prefers `/usr/bin/ffmpeg` with `libx264`; override it with
`--ffmpeg` if needed. It renders the original machine candidate, not a selected
revision; edited revisions remain reviewable in the HTML page.

Keep these axes separate when reviewing:

| Field or status | Meaning |
|---|---|
| `stage_id: null`, `identity_status: unresolved` | No stage identity is supported |
| A registry ID with `identity_status: uncertain` | Plausible identity remains ambiguous |
| Boundary `[lower_frame,upper_frame]` | Inclusive possible-cut bracket around the nominal cut |
| `invalid_intervals` | Invalid source evidence, independent of unknown identity |
| `pending_human_review` | Machine output has not acquired human acceptance |

Each internal segment cut needs one boundary entry. Coverage reports use union
durations separately for unresolved identity, uncertain identity, boundary
uncertainty and source-invalid spans; overlapping axes must not be added as
disjoint error rates. Omitted stages must name exactly the absent registry IDs.

Saving requires an explicit reviewer, `kind` (`human` or `agent`), decision
(`accepted` or `needs_changes`), notes, and descriptions of inspected evidence.
This is a self-declared review record, not identity authentication. Agent review
does not become human acceptance. Inspect pilot videos and approve the stage
standard with a human before expanding to bulk annotation.

Save the review in the page. It records your changes, decision, inspected evidence
and recording information automatically. Original candidates remain under
`episodes/`; saved revisions are under `revisions/episode_<index>/`.
Use the saved revision path in the export command below.

Export stage-only JSONL, either preserving pending machine status or selecting
an explicit saved revision:

```bash
grana export-stages --run "$GRANA_ANNOTATION_OUTPUT" \
  --output /data/grana/exports/my-task-machine-candidates.jsonl

export GRANA_STAGE_REVISION=/data/grana/annotation-runs/my-task-pilot/revisions/episode_000000/YOUR_SAVED_REVISION.json
grana export-stages --run "$GRANA_ANNOTATION_OUTPUT" \
  --revision "0=$GRANA_STAGE_REVISION" \
  --output /data/grana/exports/my-task-reviewed-stages.jsonl
```

Output must be a new file. All run episodes are exported; episodes without a
selected revision remain pending machine candidates. Rows retain stage
occurrence counts, source seconds, boundary brackets, invalid spans, registry,
recording information and the review identity/decision. This JSONL is not itself
a reward-cache input.

## Add explicit reward supervision and build a cache

Stage identity alone does not establish action quality or successful completion.
Never translate “stage predicted” into “all nominal/high.” A reward cache needs
an accepted human **or explicitly identified agent** revision plus an independent
AQ/outcome supplement derived from inspected evidence. Unresolved/uncertain
stage identities and source-invalid spans are rejected. This bridge currently
supports only an explicit `nominal_dense` boundary policy, not boundary masking.

Create an intentionally incomplete supplement from the chosen revision:

```bash
export GRANA_REWARD_SUPPLEMENT=/data/grana/reward-labels/episode_000000.json
python - <<'PY'
import json, os
from pathlib import Path
revision = json.loads(Path(os.environ["GRANA_STAGE_REVISION"]).read_text())
supplement = {
    "source_identity": revision["source_identity"],
    "reviewer": "",
    "provenance": "",
    "boundary_policy": "nominal_dense",
    "aq_intervals": [],
    "outcomes": []
}
output = Path(os.environ["GRANA_REWARD_SUPPLEMENT"])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x") as stream:
    json.dump(supplement, stream, indent=2)
PY
```

Fill the reviewer, evidence provenance and both arrays after inspection. Empty
arrays deliberately fail validation. AQ intervals must partition every native
frame, using `nominal`, `suboptimal`, or `regressed`. Outcomes must cover each
maximal contiguous stage run exactly, in order, using explicit `high`, `mid`,
or `low` (mapped to 1, 0.5, 0). A stage repeated after another stage needs a
separate outcome. Every interval/outcome needs an evidence reason.

For shape only, this **synthetic six-frame, two-stage example** shows mixed
labels. Its boundaries and values are not defaults for your recording:

```json
{
  "aq_intervals": [
    {"start_frame": 0, "end_frame": 1, "quality": "nominal", "reason": "Synthetic example only"},
    {"start_frame": 1, "end_frame": 2, "quality": "suboptimal", "reason": "Synthetic example only"},
    {"start_frame": 2, "end_frame": 3, "quality": "regressed", "reason": "Synthetic example only"},
    {"start_frame": 3, "end_frame": 6, "quality": "nominal", "reason": "Synthetic example only"}
  ],
  "outcomes": [
    {"start_frame": 0, "end_frame": 3, "stage_id": "1", "outcome": "mid", "reason": "Synthetic partial outcome only"},
    {"start_frame": 3, "end_frame": 6, "stage_id": "2", "outcome": "low", "reason": "Synthetic failed outcome only"}
  ]
}
```

Generate a one-episode cache recipe that preserves the reviewed registry and
camera mapping. This example selects episode 0. The 50 / 0.95 / 15s values in
`configs/cache.yaml` are illustrative. Choose your own stage parameters and
record their provenance; copying the synthetic example is not calibration.

```bash
export GRANA_CACHE_OUTPUT=/data/grana/caches/my-task-reviewed-000000
export GRANA_REVIEWED_CACHE_CONFIG=/data/grana/annotation-configs/cache-reviewed.yaml
python - <<'PY'
import os
from pathlib import Path
import yaml
from grana.annotation.workflow import load_annotation_config
a = load_annotation_config("configs/annotation_example.yaml")
target = yaml.safe_load(Path("configs/cache.yaml").read_text())["progress"]
cfg = {
    "source": a["source"], "output": os.environ["GRANA_CACHE_OUTPUT"],
    "episodes": [0], "task": a["task"],
    "compressed_task_prompt": a["task_instruction"],
    "stages": [{k: stage[k] for k in ("label", "prompt")} for stage in a["stages"]],
    "views": a["views"], "fps": 10, "progress": target,
    "annotations": {"0": {
        "revision": os.environ["GRANA_STAGE_REVISION"],
        "supplement": os.environ["GRANA_REWARD_SUPPLEMENT"]
    }}
}
path = Path(os.environ["GRANA_REVIEWED_CACHE_CONFIG"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x") as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
PY
grana build-cache --config "$GRANA_REVIEWED_CACHE_CONFIG"
```

Cache YAML paths, including revision/supplement paths, resolve relative to the
YAML file with environment expansion. `annotations` must cover exactly the
selected episodes. The bridge checks that the source, clock, stage definitions
and reviewed result still match before generating targets. Cache
construction additionally requires a source clock consistent with its declared
fixed FPS. Native-frame targets are computed before shared RGB/label sampling.
No raw parquet or video is modified. The resulting ordinary grana cache feeds
the training and inference workflow in the [README](../README.md). Set training
`data.views` to the new cache's ordered camera names; this pilot has three
cameras, unlike the four-camera training recipes. For example:

```bash
grana validate --config configs/pretrain.yaml \
  --set "data.caches=[$GRANA_CACHE_OUTPUT]" \
  --set 'data.views=[exterior_1,exterior_2,wrist]' --check-data
```
