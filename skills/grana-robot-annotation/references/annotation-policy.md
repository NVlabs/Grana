# Visual annotation policy

Policy: `grana-robot-annotation`. Status: **task-dependent draft**. This
policy explains how to apply a supplied vocabulary and rubric; it does not
establish a universal registry or reproduce every annotator's conventions.

## Establish the annotation contract

Require a declared task goal, task variant, stage registry, stage semantics, and
rules for the requested judgment. Quality-only requests need AQ criteria but no
outcome rubric; outcomes are requested only in legacy combined mode. If the
vocabulary is being designed, use the separate proposal
workflow in [stage-task-cards.md](stage-task-cards.md). Keep proposed definitions
and their review status distinct from temporal annotation results. A syntactically
valid registry is not a verified decomposition of the task.

Only rules actually supplied in `task_policy.rules` are task refinements. Do not
assume an API can follow local file links. State which version and temporal
convention the annotation uses, and preserve unknowns where the contract itself
cannot resolve a distinction.
Write canonical reasons and summaries in English and preserve supplied stage IDs.

## Use the declared stage identity

In quality-only mode, the supplied stage segments are fixed. Use their objectives
to interpret local actions; do not reassign identities or boundaries. If the
images contradict a stage, record the conflict in the AQ reason and use unknown
where it prevents a quality judgment. Stage correction belongs to a revised
first-round result. The identity rules below also guide legacy combined mode,
where stages are part of the requested output.

For `active_subtask`, identify the objective currently being pursued, even if
its execution fails. For `attained_state`, identify only a state supported by
visible evidence; intending a transition does not attain it. If the registry
contradicts the declared ontology, report the conflict rather than silently
translating between them.

An active objective normally includes its necessary acquisition, transport,
handoff, support, alignment, manipulation and recovery. These are not separate
stages merely because they are distinct motor actions. Use the declared entry
convention; the proposed causal workflow starts at the first visibly committed
motion toward the objective, including its purposeful approach. Completion is
judged separately and is not a mandatory gate before recognizing a next attempted
objective. Necessary support by the other arm remains with the current objective
until the dominant purposeful action commits elsewhere.

An explicit task vocabulary can separate acquisition, transport or release.
Respect that declared granularity instead of applying the default grouping to it.

Read the registry as vocabulary, not a script. Stages can be absent or revisited;
repeated objectives retain their IDs. An AQ change alone is not a stage boundary.
Use `null` plus `unresolved` when no identity is supported and a registry ID plus
`uncertain` for a plausible but ambiguous identity. Adjacent same-ID segments
form one contiguous run for outcome reporting.

## Judge local AQ from causal evidence

Apply the task's declared distinctions. The optional causal rubric proposes the
following useful contrasts, subject to explicit task scope:

| Candidate | Evidence required | Insufficient on its own |
|---|---|---|
| `nominal` | Necessary and appropriately executed work toward the active objective, including normal preparation or recovery | Smooth motion, eventual success, no bad-event annotation |
| `suboptimal` | A visibly committed missed acquisition without losing an achieved task state; other ineffective actions only under an explicit task criterion | A purposeful approach, necessary alignment, any retry, slow motion |
| `failure` as a proposed regressed event | Prior achieved task-relevant state, a robot action causing its loss, and the resulting loss | Low episode completion, occlusion, external perturbation, intentional release or reorientation |
| `unknown` | Available evidence or the supplied criterion cannot distinguish the relevant alternatives | A reason to default to normal or assign the worst class |

The regressed interpretation is an explicit task convention for new candidates,
not a global mapping from historical failure labels. Tasks may need different
failure criteria, such as an observable constraint violation. State those criteria
before annotation rather than inferring them from a label name.

Track the object or assembly before, during and after the action. Closed empty
jaws with the object still at its original location can support missed acquisition;
an object previously retained and then visibly lost supports a different event.
Disappearance behind a gripper proves neither. Distinguish intentional appropriate
release from unintended loss and external disturbance from robot-caused change.
Use object-to-gripper or object-to-fixture relationships across available views,
including wrist-camera motion. Drill proximity or visible motion does not establish
engagement, torque or tightness.

Keep the task's object identities and defining attributes stable. Do not relabel
a distractor to satisfy the instruction; not seeing the target is not proof of
its absence, a failed action or a loss of state. A wrong-object action requires
the task's explicit AQ criterion before assigning a known quality class. Treat
generated descriptions and earlier candidate reasons as claims to check against
the actual images, not independent confirmation of an event.

In each short reason, connect the visible before/action/after evidence to the
declared AQ distinction. For state loss, establish what was retained or achieved,
what action changed it, and what loss is visible. If that distinction remains
hidden, use unknown rather than infer the worst class. Keep normal recovery
separate from the earlier causing action, even if the lost state is not yet
restored. Use existing reasons and frame citations, without new response fields.

An activity description such as “adjustment” or “regrasp” does not determine AQ.
Establish the objective, observed state change and declared criterion, then explain
why the nearest competing label is unsupported. Cite short observable evidence;
avoid speculative accounts of intent or internal reasoning. No normal label
asserts that numerical task progress rises at every frame.

## Bound the error and recovery separately

State the task's temporal unit. Under the causal contract, label the visibly
supported causing-action span for state loss and the committed ineffective
attempt for a missed acquisition. Do not include unrelated earlier preparation.
End the bad span when necessary, appropriately executed recovery starts, even
if the pre-error state has not yet been restored. The recovery role itself does
not lower AQ; a new visibly bad action during it creates a new bad interval.

Do not carry an error through a whole stage, retry sequence, or eventual
restoration merely because the earlier error happened. Conversely, do not erase
the observed causing action because later recovery succeeds. If the task chooses
a different interval convention, state it explicitly and do not silently compare
durations across conventions.

For each proposed cut, preserve an inclusive possible-cut bracket supported by
neighboring observations. Sparse mosaics are not continuous evidence between
samples. Request denser observations around visible uncertainty, contact, loss or
recovery when possible, and audit some apparently normal spans for missed errors.
In a blind evaluation, extra evidence must not be selected from hidden human
boundaries. More images or fewer unknown labels alone do not prove improvement.

## Keep outcome independent (legacy combined mode only)

Skip this judgment in quality-only mode; do not require completion evidence as
a prerequisite for identifying local AQ.

Judge each contiguous stage run using its declared completion criterion. `high`,
`mid`, and `low` are categories, not probabilities. High requires the whole
objective visibly completed; mid requires an explicit observable partial-result
criterion. Low requires visible noncompletion when the observed attempt ends;
clipped or ambiguous results remain unknown. The chosen next stage, local AQ,
or smooth tool motion does not establish outcome. Uncertain or unresolved stage
identity requires unknown outcome.

## Preserve evidence and output integrity

Return the supplied schema for the active mode. In the optional quality round,
keep the supplied stages frozen, partition each stage segment separately, and
return only its AQ annotations using the supplied response schema. Do not infer
outcomes or replace stage decisions in that round. Uncertain stage identity and
source-invalid spans require unknown AQ.

In legacy combined mode, stage segments and AQ intervals each partition `[0,N)`
completely without overlap. Use explicit unknowns rather than
filling gaps with normal. Every internal cut has exactly one ordered boundary
entry with a bracket containing the nominal cut. Omitted stages are exactly the
registry IDs absent from the segments. Outcomes match maximal contiguous stage
runs. AQ boundaries and stage boundaries are separate.

Cite only supplied native frame ranks. Known AQ and outcome labels require at
least one cited frame outside declared invalid intervals; that minimum is not
proof that evidence is sufficient. Unknown may cite invalid evidence to explain
why a decision is unsupported. Missing annotations, explicit normal, unknown and
invalid source evidence remain distinct through review.

This output is a candidate. It neither accepts a training supplement nor changes
cache progress/reset logic. Record the vocabulary/rubric version and declared
reference-review status when evaluating agreement; do not present agent proposals
or legacy label matches as newly verified human ground truth.
