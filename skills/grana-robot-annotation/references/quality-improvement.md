# Improve pure API annotation quality

Use this guide when defining a task's annotation standard or investigating poor
stage/AQ candidates. Human seed annotations are optional. The target is faithful
application of the declared standard to visible events, not an expected success
story or a fixed stage sequence.

## What the implementation provides

| Mechanism | Current scope |
|---|---|
| Task definitions and round-specific instructions | The API receives `task_policy.rules`, registry, ordered images and the selected visual policy. The whole operational skill is not pasted into two-round requests. |
| Separate stages and optional AQ | The AQ schema and validator bind the exact first-round candidate and prevent boundary/identity changes. A frozen stage can still be visually wrong. |
| Evidence checks | The driver checks that the supplied evidence matches the declared recording and clock. The validator checks that cited frame IDs were supplied, not that those frames support the label. |
| Denser evidence | `refine_annotation_evidence.py` can expand first-round stage boundaries; `prepare_annotation_evidence.py --extra-frames` can add specified native frames. These tools do not independently judge images or submit a review automatically. |
| Visual review and comparison | The procedure below is agent-guided. An automatic second visual reviewer, conflict resolution loop and two-round accuracy benchmark are not implemented. |

The [evidence review procedure](evidence-review.md) gives a concrete way to record
object observations separately from decisions and compare repeated candidates.
Its private notes are agent-maintained; the driver does not emit or ingest them.

## Make the desired decision explicit

For each easily confused stage pair, define the pursued objective, observable
entry/transition criteria, closest alternative and visibility limits. Keep
operational granularity consistent with the user's registry. A stage named
`transport` is not a sufficient rule by itself.

A synthetic contrast for an explicit acquire/transport vocabulary:

- Moving an empty gripper while attempting to grasp remains acquisition.
- Moving the retained target toward its destination can support transport under
  the task's declared transition convention.
- If retention is hidden and views do not distinguish these situations, preserve
  uncertain identity instead of using elapsed time to choose a stage.

For optional AQ, separately define the distinguishing evidence for each quality
class. A missed acquisition, loss of an already held object, deliberate release
and normal recovery are different candidate events. Use the task's criteria;
do not infer a universal quality class from the word "retry".

Separate observation guidance from decision criteria in the task rules: first
say what visible relationship matters, then what that observation means under
this task's convention. Inspect the images before changing a rule to fit a model
answer. Keep object identity and visibility uncertainty explicit.

Write these rules and counterexamples in English in `task_policy.rules`, or the
existing card fields and scoped `common_rules` when using the task-card compiler.
Direct settings are sufficient for stage-only work; the richer card compiler
also expects an AQ contract. See [stage-task-cards.md](stage-task-cards.md).
Inspect the prepared context to confirm the actual rules are included. Links to
local documents do not send their contents to the API. Distinguish synthetic
contrasts, model-proposed rules and independently checked examples.

## Revisit the visual evidence when needed

When visual review is requested, freeze the initial candidate and policy snapshot.
Select ambiguous boundaries or confused identities, and also a bounded sample
of apparently resolved spans so confidently wrong labels are not excluded.
Choose these using model-visible evidence or a declared sampling rule, not hidden
evaluation labels. The current stage refinement planner targets reported
boundaries; it does not automatically sample resolved interiors or discover
missing transitions.

Use the [API workflow](api-workflow.md#add-evidence-without-changing-the-stage-result)
to add observations. Preserve the original frames and native clock mapping;
`--extra-frames` takes native source IDs, while candidates use clip-local IDs.
Reinspect the actual before/transition/after images against the task's criteria.
A generated scene description may help organize evidence, but is not a substitute
for the images and can introduce its own errors.

For a further stage API pass, submit the expanded stage bundle as a separate
request, compare identities and boundary brackets, and retain both candidates
and receipts. This is a new annotation pass, not an independent verifier. A true
reviewer should assess the visual evidence and competing definitions rather than
merely endorse the first answer; model agreement alone is not correctness.
If stage decisions change, prepare dependent AQ again from the chosen new result.
For AQ evidence changes, prepare a new quality bundle with the expanded context.

Keep extra calls within the requested campaign budget. Stop when the planned
review pass completes, no useful new observations are available, or the budget
is exhausted. Retain unresolved cases; do not rerun until a preferred label appears.
Record the disputed interval, evidence used, decision and remaining uncertainty
in a separate private review note without adding fields to the candidate schema.
Use the [bounded repeat procedure](evidence-review.md#use-disagreement-to-choose-a-bounded-review)
when repeatability is the question. It preserves each candidate and uses temporal
disagreement to target review instead of averaging stage IDs or AQ labels.

## Measure improvement without changing the standard midway

Freeze task definitions, model settings, recording set and reference judgments.
Compare one intervention at a time: stage-specific instructions, task contrast
rules, or additional evidence. Keep sampling fixed for prompt comparisons; when
sampling is the intervention, report the extra images, requests, cost and latency.

Assess stage confusion and boundary error/uncertainty. If AQ was requested, also
assess false alarms on normal actions, missed bad events, and bad labels extending
into normal recovery. Report unknown coverage separately. Split development and
evaluation recordings by physical trajectory, and do not use reference boundaries
to choose evaluation frames.

Without independent reference judgments, report consistency and reviewed examples,
not accuracy improvement. Existing combined AQ/event evaluators expect the legacy
contract and are not drop-in evaluators for two-round candidates. Do not invent
AQ or outcomes to make stage-only results fit them. Unit tests, schema validity,
model self-review and fewer unknowns do not establish semantic accuracy.
