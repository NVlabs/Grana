# Inspect evidence and resolve annotation disagreements

Use this procedure for ambiguous stage identity, poorly supported boundaries,
or disputed optional AQ. It is an agent-guided use of Grana's existing tools,
not an automatic observation extractor or visual verifier. Ordinary clips still
use one stage request and, only when requested, a separate AQ request.

## Keep observations separate from criteria

Before annotation, identify the task's target objects, relevant fixtures, stage
granularity and transition convention. Keep two kinds of guidance separate in
the existing English `task_policy.rules` text; these are headings, not new JSON
keys or a new settings schema:

- **Observation guidance:** what to inspect, such as target-to-gripper motion,
  continued table support, release, occlusion or agreement across camera views.
- **Decision criteria:** which observed distinction supports a stage or AQ class,
  including the closest alternative and what remains unknown without evidence.

For an explicitly separate acquisition/transport vocabulary, "inspect whether
the target moves with the gripper" is observation guidance. "Transport begins
when the retained target starts moving toward its destination" is a possible
task-specific transition criterion. Do not impose it on a task whose declared
entry convention differs. A useful contrast is an empty gripper moving while the
target remains on the table; it does not establish transport of the target.

Do not convert a missing observation into an absent object. Keep target identity
uncertain when attributes or views cannot establish it. Do not substitute a
distractor to make the task instruction fit the scene. An object mismatch does
not by itself determine the intended objective or an AQ class.

## Record only facts needed to resolve the issue

For ordinary candidates, use the existing short `reason` and `evidence_frames`
fields. When a result needs closer inspection, write a separate private review
note alongside the candidate, identifying its result folder and evidence files.
Keep each candidate's note distinct.
Record which agent or model produced observations; they are fallible claims,
not an independent sensor channel or human ground truth.

An optional observation row could look like this **synthetic** example:

```json
{
  "frame_id": 120,
  "view_names": ["front"],
  "object_id": "target_block",
  "observation": "The gripper closes while the target remains supported by the table.",
  "visibility": "clear"
}
```

This is a note convention, not a new model-response contract. Notes are written
by the reviewing agent; the API driver does not generate, validate or ingest
them automatically. Use only supplied clip-local frame IDs and actual view names.
Refer to the reviewed images by filename and frame ID; use the source receipt to
recover native times and offsets. Object IDs are local aliases for visibly tracked
entities, not new stage IDs or a verified cross-recording tracking system.

Record before/transition/after relationships where available, plus the missing
view or observation that prevents a decision. Prefer facts such as "the block
stays on the table" to inferred labels such as "the robot failed". Wrist-camera
motion, gripper closure and disappearance under occlusion are insufficient on
their own to prove transport, retention or loss. Do not describe unobserved
intermediate frames as if they were supplied.

Keep any proposed stage/AQ judgment and its supporting task criterion separate
from the observations. For AQ state loss, identify the previously achieved state,
the causing action, visible loss and onset of normal recovery. A known active
stage can contain a failed attempt; nominal recovery does not imply increasing
numerical progress. Do not add these notes to `candidate.json` or replace images
with generated descriptions in a later request.

## Use disagreement to choose a bounded review

Start with the original candidate. When repeated annotation is within the user's
requested scope and budget, one additional candidate can expose unstable stage
identities or boundaries. Use the same source/clock, vocabulary, policy snapshot,
images and model settings for a repeatability check. Changing those inputs is a
separate comparison, not a repeat. Do not show the first candidate to a supposedly
independent repeat. Repeated calls to one model can share the same mistakes.

For a new stage sample, reuse the original bundle and evidence with a distinct
request name and fresh result directory:

```bash
python scripts/annotation_skill_api.py \
  --bundle /data/annotation/episode-0/evidence/bundle.json \
  --evidence /data/annotation/episode-0/evidence/evidence.json \
  --output-dir /data/annotation/episode-0/stages-repeat-1 \
  --api-dir /data/annotation/campaign --request-name episode-0-stages-repeat-1 \
  --max-requests 24
```

Use the same `--policy-dir` snapshot as the original request when one was supplied.
This new name requests another billable sample; rerunning a completed request
with its original name reuses its receipt and is not independent evidence. Keep
the campaign's existing request limit; do not raise it to obtain agreement. The
limit counts HTTP attempts across stages, AQ, repeats and retries, not episodes.
An unchanged input can legitimately produce an identical output.

Compare candidates on their native clock and declared vocabulary:

| Disagreement | Review action |
|---|---|
| Different target or object state | Reinspect the original views and object identity before changing task rules. |
| Same observation, different stage | Check entry/transition definitions and granularity; fix an ambiguous rule explicitly and version it. |
| Different boundary or bracket | Inspect the evidence around the disputed span; request denser frames where useful. Do not average cut positions or invent precision. |
| Different AQ with the same frozen stages | Check the task criterion and before/action/after evidence; keep causing action and normal recovery separate. |
| Different stage candidates used for AQ | Resolve the stage comparison first. AQ disagreement mixes two inputs and is not an isolated quality comparison. |

Do not average stage IDs or ordinal quality codes. Majority agreement is a
review signal, not grounds to erase unknowns. Audit a bounded sample of apparently
resolved spans as well; agreement can hide a consistently missed transition.
Select review windows without hidden reference labels. For evidence expansion,
follow [api-workflow.md](api-workflow.md#add-evidence-without-changing-the-stage-result).

Preserve each raw candidate, validation and API receipt. Record the disputed span,
evidence inspected, chosen decision and remaining uncertainty in the private note.
A manually corrected candidate has separate provenance; it is not a provider
response. Revise round one and rebind AQ if stages change. If only AQ evidence
changes, prepare a new quality bundle against the same stage result.

Stop after the planned review pass, when no useful observations remain, or when
the budget is exhausted. Retain unresolved results rather than repeat until a
preferred label appears. A small reviewed pilot or stable repeated output does
not establish accuracy; use the comparison guidance in
[quality-improvement.md](quality-improvement.md#measure-improvement-without-changing-the-standard-midway).
