# Annotate raw recordings

Grana proposes stages from RGB recordings, then optionally annotates action
quality (AQ) within the saved stages. Annotation runs on CPU and sends selected
camera mosaics to your configured inference API. Results are candidates; neither
round supplies the stage outcomes required to build a reward-training cache.

| Workflow | Input support | Result |
|---|---|---|
| Recommended two-round API workflow | Native LeRobot v2.x/v3.0 | Stage candidates, optional AQ |
| [Human labeling](human-labeling.md) and API draft import | Native LeRobot v2.x/v3.0 | Reviewed labels with outcomes |
| Existing `grana annotate` projects | Native LeRobot v2.x | Stage candidates and reviewed stage JSONL |
| [Cache construction](data.md) | Native LeRobot v2.x/v3.0 | Training cache from explicit reward labels |

## Install and configure

Use the source checkout and the [CPU annotation environment](installation.md#cpu-labeling-and-cache-construction),
with `ffmpeg` and `ffprobe` on `PATH`. That guide also covers adding API
dependencies to a full environment. The wheel alone does not include the
workflow scripts and policy files.

Create a private settings JSON following the [settings reference](../skills/grana-robot-annotation/references/api-workflow.md#set-the-task-and-granularity).
Replace the synthetic task, camera feature keys and stage definitions with your
own. Give each stage a stable ID and an observable objective; review the
[stage-design guide](../skills/grana-robot-annotation/references/stage-task-cards.md)
if you need to define a vocabulary. Stage-only annotation needs no AQ rubric;
add one when requesting the optional second round.

Before live requests, set `GRANA_ANNOTATION_BASE_URL` and
`GRANA_ANNOTATION_MODEL`, or pass `--base-url` and `--model` to the driver.
CLI values take precedence. The provider must support HTTPS
`POST <base_url>/responses`, image inputs, strict JSON-schema output and the
requested reasoning effort. A chat-completions endpoint alone is insufficient;
Grana selects no provider or model and performs no automatic fallback.

Provision `GRANA_ANNOTATION_API_KEY` through your usual credential mechanism.
`--api-key-env` selects a different variable **name**. Never put credential
values in recipes or command arguments. Offline planning, evidence preparation
and human labeling require no provider or key. Keep settings, sampled images
and outputs outside the checkout and raw recording directory.

<a id="pure-api-annotation-two-rounds"></a>

## Plan, run and resume

Run from the checkout root. The following command plans an episode without
making API calls; replace the paths and choose the request budget:

```bash
annotation_args=(
  --source /data/recordings/my_task --episode 0
  --settings /data/annotation/settings.json
  --output /data/annotation/episode-0
  --api-dir /data/annotation/campaign --request-name episode-0
  --max-requests 24
)
python scripts/annotate_episode.py "${annotation_args[@]}"
```

Inspect the plan, then explicitly submit stage-only requests:

```bash
python scripts/annotate_episode.py "${annotation_args[@]}" --run
```

The controller splits long episodes into bounded requests. Sparse sampling does
not establish exact frame boundaries; inspect the saved boundary uncertainty.
The first-round result is `candidate/candidate.json`, paired with
`evidence/bundle.json` under the chosen output directory.

To add AQ, reuse the same arguments and saved stage result:

```bash
python scripts/annotate_episode.py "${annotation_args[@]}" --run --with-quality
```

AQ is requested separately inside each frozen stage occurrence and saved under
`quality/`. It cannot replace stages or infer outcomes. When AQ is skipped,
`default_normal_as_not_requested` records that valid unmarked time defaults to
Normal; this is not a model judgment or reviewed nominal supervision. See the
[output contract](../skills/grana-robot-annotation/references/output-contract.md).

All requests share the persisted HTTP-attempt budget in `--api-dir`; failed
attempts count, and concurrency does not increase the allowance. Repeating
identical arguments reuses completed requests. AQ failure preserves the stage
result and completed chunks. Completed requests can be reused without a key
using the same endpoint/model settings.

Inspect uncertain or exhausted requests before continuing. A timeout may have
been billed even if Grana received no response. Drivers that offer
`--retry-uncertain` require an explicit decision to risk duplicate billing;
it is not a routine resume option and cannot enlarge the saved allowance.
Keep receipts intact. Use a new campaign when changing frozen source, settings,
evidence, policy or provider configuration.

For request bounds, denser evidence, batch execution and policy snapshots, see
the [workflow reference](../skills/grana-robot-annotation/references/api-workflow.md).
The older combined stage/AQ mode remains available through `--mode combined`
in `scripts/prepare_annotation_evidence.py` and its [original contract](../skills/grana-robot-annotation/references/output-contract.md#legacy-combined-candidate---mode-combined);
retain the policy snapshot used by those results.

## Review candidates and import drafts

Inspect stage identities, boundaries and AQ against the source videos. Schema
validation checks structure and provenance, not annotation accuracy. The
[evidence review guide](../skills/grana-robot-annotation/references/evidence-review.md)
describes how to inspect ambiguous candidates and request more evidence.

To review a two-round result in the human editor, create a matching
[labeling recipe](project-configuration.md#labeling-your-own-recordings), then
preview the import:

```bash
python -m grana.cli draft-labels --config /data/annotation/labeling.yaml \
  --from /data/annotation/episode-0/candidate --dry-run
```

Use `quality/` instead of `candidate/` after the optional AQ round. If AQ
contains `failure`, the importer requires the explicit task-policy mapping
`--failure-quality regressed`; use it only when that mapping fits your policy.
Inspect the dry-run, then remove `--dry-run` to write drafts. Existing saved
labels are never overwritten. Open the same recipe with `python -m grana.cli label` to review
the drafts, supply outcomes and submit reviewed episodes. **Export saved JSON**
in that editor exports its canonical label document.

Import does not accept machine output for training. Drafts require review and
submission; stage predictions alone do not establish successful completion or
nominal action quality.

## Bounded campaigns and the local playground

For a local interface to human labeling, API drafts and seed examples:

```bash
python -m tools.annotation_playground.launcher \
  --config /data/annotation/labeling.yaml \
  --workspace /data/annotation/playground --port 8840 --max-requests 24
```

Open the printed loopback URL. API calls start only after an explicit click;
manual annotation needs no key. The service stays in the foreground and stops
its child services on exit. Seed transfer requires completed comparisons on
the selected held-out episodes before applying the template to targets; that
checks comparison completeness, not an accuracy threshold. See the
[playground guide](../tools/annotation_playground/README.md) for its controls.

## Existing `grana annotate` projects

This separate LeRobot v2 workflow uses YAML and produces runs consumed by
`review-annotations` and `export-stages`. Those commands do not consume the
two-round result directories described above.

Copy [annotation_example.yaml](../configs/annotation_example.yaml) to a private
recipe and replace its source/output paths, episodes, camera mapping and stage
registry. Each stage has an `id`, downstream cache `label` and objective
`prompt`. Paths resolve relative to the recipe after environment expansion.
Use a new output directory when changing the source or configuration.

The same provider environment variables apply. Explicit `api.base_url` and
`api.model` in the YAML take precedence; `api.api_key_env` names the credential
variable. This workflow requires the provider to acknowledge its requested
model and reasoning effort (`high` for the probe, `xhigh` for visual calls).

### Probe, annotate and resume

```bash
python -m grana.cli annotate --config /data/annotation/stages.yaml --plan
```

After checking the offline plan and budget, run the API probe and annotation:

```bash
python -m grana.cli annotate --config /data/annotation/stages.yaml --preflight
python -m grana.cli annotate --config /data/annotation/stages.yaml
```

Each window receives draft, boundary-refinement and review requests. This API
review is separate from human acceptance. `api.max_requests` bounds all HTTP
attempts, including retries; each request also has an attempt, token and timeout
limit. Saved limits do not grow when restarting. Repeating the unchanged command
reuses completed responses. An uncertain attempt stops the run; inspect its
receipt before explicitly choosing `--retry-uncertain` and its billing risk.

### Review the original videos

Set `GRANA_ANNOTATION_OUTPUT` to the run directory from your recipe:

```bash
python -m grana.cli review-annotations --run "$GRANA_ANNOTATION_OUTPUT" --port 8788
```

Open **http://127.0.0.1:8788/**. For a remote host, forward the loopback port
from your local computer:

```bash
ssh -N -L 8788:127.0.0.1:8788 user@annotation-host
```

The page plays the source videos and saves immutable revisions alongside the
original candidates. Browser seeking is approximate; the source clock and
evidence frames determine annotation bounds. Segments use half-open native
frame intervals `[start_frame,end_frame)`; boundary uncertainty is an inclusive
possible-cut bracket. Unresolved identity, uncertain identity and invalid source
evidence are separate conditions.

Saving a revision records the reviewer, `kind` (`human` or `agent`), decision
(`accepted` or `needs_changes`), notes and inspected evidence. This is
self-declared attribution; agent review is not human acceptance. Use the saved
revision path to export reviewed stage JSONL:

```bash
python -m grana.cli export-stages --run "$GRANA_ANNOTATION_OUTPUT" \
  --revision "0=/data/annotation/revisions/episode-0.json" \
  --output /data/annotation/exports/reviewed-stages.jsonl
```

The output must be a new file. Every run episode is exported; those without an
explicitly selected revision remain pending machine candidates. Omitting
`--revision` exports the original candidates. Stage JSONL is not a reward-cache
input. For a rendered video of the original candidate, see
`python scripts/render_annotation_video.py --help`; rendering does not apply
selected review revisions.

### Add explicit reward supervision and build a cache

For this workflow, cache construction requires an accepted human or explicitly
identified agent revision plus an independent AQ/outcome supplement derived
from inspected evidence. Unresolved/uncertain identities and source-invalid
spans are rejected. The supported boundary policy is `nominal_dense`, which uses
the nominal cuts; boundary masking is unsupported.

The supplement has these exact top-level fields:

| Field | Required content |
|---|---|
| `source_identity` | Copy the complete value from the selected revision |
| `reviewer`, `provenance` | Nonempty reviewer and evidence descriptions |
| `boundary_policy` | `nominal_dense` |
| `aq_intervals` | Ordered intervals partitioning every native frame without gaps or overlap |
| `outcomes` | One entry for each maximal contiguous stage run, in order |

Each AQ entry has `start_frame`, `end_frame`, `quality` (`nominal`, `suboptimal`
or `regressed`) and an evidence `reason`. Each outcome has `start_frame`,
`end_frame`, `stage_id`, `outcome` (`high`, `mid` or `low`, mapped to 1, 0.5 or 0)
and an evidence `reason`. Bounds are half-open. Revisited stages need separate
outcomes; empty arrays fail validation. Never infer all-nominal/high labels
from a stage prediction.

Create a [cache recipe](data.md#build-a-cache-from-lerobot-and-annotations) with
the same source, camera mapping and ordered stage labels/prompts as the revision.
Choose task-specific progress parameters. Use `annotations` instead of `labels`
and cover exactly the selected episodes:

```yaml
episodes: [0]
annotations:
  '0':
    revision: /data/annotation/revisions/episode-0.json
    supplement: /data/annotation/reward-labels/episode-0.json
```

```bash
python -m grana.cli build-cache --config /data/annotation/reviewed-cache.yaml
```

The builder checks that the source, clock, registry and reviewed result still
match. It requires fixed-cadence input and a fresh cache destination; raw videos
and parquet remain unchanged. Then follow
[training with your own cache](project-configuration.md#training-with-your-own-cache).
