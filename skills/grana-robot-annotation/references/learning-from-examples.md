# Derive a policy from human examples

The deliverable is a portable annotation policy, supported by a **private**
evidence ledger. Preserve the difference between original human decisions,
adapter defaults, model suggestions, and new adjudications. Source code's numeric
label mapping explains how labels are consumed; it does not explain what a human
saw when assigning them.

## Intake and inspection

Use the supplied task and examples; a small representative pilot is preferable
to an uncontrolled scan. Record the following privately when available, and
mark missing provenance explicitly rather than inventing it:

| Record | Needed distinction |
|---|---|
| Physical recording identity, views, native frames/time edges | Multiple clips from one trajectory are not independent examples |
| Task instruction, registry, ontology/version, stage semantics | Active subtask versus attained state; inferred versus original stage descriptions |
| Original human interval and raw label, endpoint convention | Frame-rank versus seconds; inclusive versus half-open; transformed IDs versus raw decisions |
| Semantic meaning of the labeled span | Entire attempted action versus visible adverse event or persistent state; recovery onset versus restored object state |
| Annotation coverage and annotator/version | Explicit nominal versus unlabeled; unknown versus a negative label |
| Stage, AQ, and outcome with separate provenance | One axis cannot be used to fabricate another |
| Before/event/after visual evidence | A failed attempt, a retry, recovery, and later success need distinct observations |
| Human comments and accepted corrections, when supplied | A model explanation is not the original annotator's rationale |

Review the original context on both sides of each labeled span, with synchronized
views where available. Inspect the media rather than inferring a rule from a
filename, annotation category, or extracted target curve. Read the human label
as evidence about annotation practice, not as proof that the visual event is
unambiguous. Preserve disagreements instead of overwriting the source labels.

## Extract contrasts, not stories

For each candidate rule, privately record:

1. An observable condition and the task scope where it applies.
2. The human class it supports and the closest competing class.
3. Supporting examples **and counterexamples**, with their local time spans.
4. What distinguishes them: object state, constraint violation, loss of attained
   progress, necessary versus redundant correction, or missing evidence.
5. Whether the explanation came from the annotator, a later human adjudication,
   or the reviewing model's hypothesis.
6. Unresolved cases and whether more temporal detail, a different view, or a
   human decision would resolve them.

Do not call a rule universal because several clips from one trajectory agree.
When two similar events carry different labels, check task objectives, endpoint
conventions, annotation coverage, and annotator versions before merging them.
Do not resolve the conflict simply by majority vote or by copying a third-party
model's class definitions.

## Validate before promoting a rule

Separate calibration and held-out examples by physical trajectory, including
failure, suboptimal, and explicitly reviewed nominal/unknown cases. Freeze the
policy before labeling the held-out set, and withhold original human labels from
the model performing that pass. Record model/provider, policy and task versions,
visual sampling, and review status. Report disagreement by class and task,
failure precision/recall, unknown coverage, and native-clock boundary error.
Separate sampling limitations from a policy/model classification disagreement.

If the held-out results motivate changes, version the rule and disclose that the
set was used for development; reserve new episodes for the next blind check.
Sparse-contact-sheet agreement is not a full-video accuracy measurement.

## Publish the abstraction

Publish confirmed conditions, scope, relevant exceptions, remaining uncertainty,
and explicitly synthetic contrasts. A rule supported only by agent visual review
remains a hypothesis until the intended human standard accepts it. Include a
concise qualification statement identifying what was and was not tested.

Keep raw videos, frames, original label rows, private IDs/paths, per-example
corrections, supporting-example mappings, and request receipts outside the source
release. Check the **actual archive manifest**, not just `.gitignore`: the source
packager uses its own allowlist. Generalized descriptions should retain the
decision criterion without reconstructing a private recording.

Task profiles supplied by other researchers should state their own version,
calibration status, stage ontology, AQ distinctions, and outcome criteria. The
public skill transports a method and rules; it does not certify that one
institution's private labels are the right standard for every robot or task.
