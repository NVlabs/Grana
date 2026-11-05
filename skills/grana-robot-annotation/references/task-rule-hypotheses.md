# Cross-task hypotheses to test with examples

These are methodological starting points, not measured rules, a private task
registry, or a claim to reproduce a particular annotator. Use them to choose
contrasting examples and draft a task-specific rubric. Record the evidence and
unresolved cases privately; publish only the abstract rule and its scope.

## Objectives are larger than motor actions

**Hypothesis:** approach, acquisition, handoff, transport, alignment and recovery
usually belong to the objective they serve. A change of arm or motion type need
not be a stage change.

**Test:** compare alternate successful strategies and failed attempts at the same
goal. Check whether proposed stages preserve that identity while distinguishing
an actual objective change. Counterexamples include tasks that explicitly make
handoff, inspection or a particular final placement a required separate goal.
Do not force a fixed stage granularity across tasks.

## Failure to acquire differs from loss after acquisition

**Hypothesis:** a missed acquisition without achieved-state loss can be suboptimal,
while causing loss of a retained or assembled state can be regressed under a
causal progress-based rubric.

**Test:** inspect the object before closure, after closure, during transport and
after the event. Include ordinary approaches, occluded successful grasps,
intentional releases and externally disturbed objects as counterexamples.
An object leaving a view does not establish either category. Declare different
criteria explicitly for tasks where state loss is not the relevant AQ distinction.

## Necessary correction and recovery can be normal

**Hypothesis:** recovery is an activity role, not an AQ class. Necessary,
appropriately executed recovery can be normal before physical state is restored.

**Test:** compare ordinary alignment, recovery after an earlier error, and a new
miss during recovery. Under the causal interval convention, the earlier bad span
ends at normal recovery onset; the new miss is a separate event. Counterexamples
include explicitly constrained tasks where the recovery maneuver itself violates
a declared requirement. Do not call every repeated motion inefficient.

## Visible evidence must support the claimed distinction

**Hypothesis:** multiple views and local temporal sampling can distinguish retained
objects, unintended loss and normal handling better than isolated images.

**Test:** review the same event under fixed sparse observations and extra local
observations selected from model-visible uncertainty. Also inspect normal control
windows. Preserve unknown when hidden engagement, force or geometry remains
unobservable. A confidence increase or smaller unknown fraction is not evidence
of improved accuracy without an independent reference.

## Local AQ and stage outcome can disagree

**Hypothesis:** a stage can contain a bad local action and still finish successfully;
a normal local action can occur in an incomplete stage.

**Test:** inspect the declared completion state independently of local AQ, final
stage ID, or the robot's decision to move on. Include clips ending mid-attempt and
partially completed objectives. Define partial outcome only when the task has an
observable partial-completion criterion; otherwise use unknown rather than
inventing an intermediate score.

## Turn a hypothesis into a scoped rule

For each candidate rule, retain:

- The task goal/variant and stage scope where it is intended to apply.
- The visible criterion and the closest normal, inefficient, regressed and unknown
  alternatives, without requiring all four to be observable in every example.
- The interval unit, causing-action onset, recovery endpoint and boundary limits.
- Supporting and contradicting examples, plus a reference's actual review status.
- What would falsify or narrow the rule on a new recording.

Use a versioned draft during development. Freeze changes before comparing on
untouched recordings. Agreement with existing labels and a model's self-review
are useful diagnostics, but neither automatically validates the rule or accepts
its output for training.
