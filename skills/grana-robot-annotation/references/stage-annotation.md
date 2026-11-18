# First round: sub-stage annotation

Identify which declared sub-stage is being pursued at each time. Do not judge
action quality, success, failure or numerical progress in this round. Use the
supplied registry and task rules as definitions, never as evidence of timing.
Write reasons and summaries in English and preserve the supplied registry IDs.

## Observe before assigning identity

Follow relevant objects across chronological views: their locations, support,
retention, contact, and visible state changes. Distinguish observations from
interpretations of the robot's objective. A short scene description may organize
evidence, but the actual images remain authoritative. No extra description API
call is required. Occlusion and wrist-camera motion can make state ambiguous.

Bind task nouns and defining attributes to the same visible entities across views
and time. Do not rename a distractor, substitute a convenient object, or assume a
target exists because the task mentions it. An occluded or out-of-view target is
not proven absent. Keep ambiguous object matches explicit; do not manufacture a
stage transition, failure label or zero progress from missing visibility.

For a proposed transition, inspect the relevant before/after observations and
the object-to-gripper or object-to-fixture relationship. Gripper closure does not
by itself prove retention; camera motion does not prove object transport. Use
available views to resolve the distinction, and retain uncertainty if they
disagree. Do not infer events between sparse samples as though they were observed.

Match the active objective to a registry entry. A failed grasp, pause, adjustment
or recovery usually stays in the objective it serves. An AQ change alone is not
a stage change. If the declared vocabulary intentionally separates grasping,
transport and placement, follow that granularity instead of merging it away.
For `attained_state`, require visible attainment instead of attempted intent.

Apply each stage's observable entry and transition criteria, including any
task-specific counterexamples. For neighboring stages that could both explain
the scene, identify the visible distinction before choosing one. For example,
under an explicitly declared acquire/transport vocabulary, empty gripper motion
does not establish transport of the target object. This is a scoped contrast,
not a universal requirement to split acquisition and transport into two stages.
Use the existing `reason` and `evidence_frames` fields for a short observable
justification; if the distinguishing evidence is absent, retain uncertainty.
Distinguish the observation from the rule it supports: for example, "The gripper
closes while the target stays on the table; the acquisition attempt continues."
This is a synthetic explanation pattern, not an observation to copy. A supplied
description or earlier candidate is a hypothesis to check against the images;
it cannot override contradictory visual evidence. Do not add observation tables
or new fields to the candidate schema.

## Bound the temporal segments

Partition the entire requested clip with half-open `[start_frame,end_frame)`
segments. Stages may be skipped, repeated or revisited, preserving their IDs.
Do not infer a fixed order, duration or stage count from a successful example.
Do not require the previous objective to succeed before recognizing the next
attempted objective. Clips can begin or end partway through a stage.

Use `identity_status=resolved` only when the identity is supported. Use a registry
ID with `uncertain` for a plausible ambiguous identity, or `stage_id=null` with
`unresolved` when no identity is supported. Source-invalid intervals are distinct
from unknown stage identity. Do not create a new vocabulary item during annotation.

Every internal cut has one inclusive possible-cut bracket and citations to
supplied frame IDs. Sparse samples do not establish precise contact times.
Use denser observations around uncertain transitions when available; otherwise
retain the bracket. Reasons should state observable evidence and the ambiguity,
not a speculative account of intent. The omitted-stage list is exactly the
registry IDs absent from the returned segments.

## Review this round independently

Check complete coverage, the meaning of neighboring stages, retries/revisits,
and whether the cited evidence supports each cut. Revise stage candidates before
starting the optional AQ round. If later observations reveal a stage error,
create a revised first-round result and rerun dependent AQ using that result;
do not silently edit a frozen result through an AQ response.

When reviewing alternatives, distinguish an object/visibility disagreement from
a different interpretation of the stage definition or timing convention. Compare
the disputed images and declared criteria rather than averaging stage IDs or
cut positions. Agreement alone does not justify `resolved` identity or a narrower
boundary bracket. Return only this round's requested schema.
