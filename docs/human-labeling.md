# Human labeling in the browser

`grana label` opens a local editor for native **LeRobot v2.x and v3.0**
recordings. Mark subtasks, their Completed/Failed outcomes and action-quality
intervals, then confirm that the whole episode was reviewed. The editor runs
on CPU without a model checkpoint, GPU or annotation API key.

## Configure and start

Use the [CPU labeling environment](installation.md#cpu-labeling-and-cache-construction)
or the full Grana environment, with `ffmpeg` and `ffprobe` on `PATH`. Run the
commands below from the source checkout. Create a private recipe using the
[standalone labeling schema](project-configuration.md#labeling-your-own-recordings).
For an existing project, retain its active YAML and complete `.stage-versions/`
history; see [restoring project data](installation.md#reinstalling-or-moving-a-deployment).

Episodes need at least two frames and a fixed frame cadence. Source timestamps
may have a constant offset. V2 videos must start at presentation timestamp zero;
for v3, the editor serves each camera's episode range as a clip starting at zero.
See the [source layout and cache requirements](data.md#build-a-cache-from-lerobot-and-annotations).

```bash
: "${GRANA_LABEL_CONFIG:?Set the absolute path to your labeling recipe}"
python -m grana.cli label --config "$GRANA_LABEL_CONFIG"
```

The recipe separates `source` (read-only recordings), `labels` (editable label
JSON files) and `output` (a new training cache). Keep labels and caches outside
the source and Git checkouts. Use either `labels` or the reviewed
`annotations` revision/supplement input in a recipe, never both.

Open the printed URL, normally **http://127.0.0.1:8790/**. The server stays in
the foreground until Ctrl+C; `--port 9012` selects another port. For a remote
machine, forward the port from your own computer, then open the same URL:

```bash
ssh -N -L 8790:127.0.0.1:8790 user@label-host
```

## Edit the sub-stage definitions

Use **Add stage** or **Edit stages** in the Subtask timeline card to add,
describe, reorder or remove stages. Each needs a distinct nonempty English
description; a project supports 1–26 stages. Definitions apply to every episode
in the recipe.

**Apply stages** saves current edits as a draft before updating the project.
Finish or cancel an open span first. Stable stage IDs preserve intervals,
action-quality spans and outcomes when descriptions or ordering change.
Migrated labels become drafts and need review before submission. A stage used
by a saved episode cannot be removed until its intervals have been reassigned
or deleted. A failed migration leaves the active recipe and saved labels intact.

The active YAML is updated atomically. Previous and new recipes are retained in
`.stage-versions/version-*/` next to the original labels directory; each new
version has separate `labels/` and `cache/` destinations. To return to a version,
start the editor with its saved recipe. Build the cache with the active recipe
to use its current stages and labels. Stale tabs cannot save old definitions
into a new version; export unsaved edits before reloading them.

Progress parameters follow retained stage IDs; added stages inherit the last
stage's parameters for preview. Edited recipes are marked
`parameter_status: example`. Review these parameters before training. Restart
the server after manually editing the recipe.

## Interface language

English is the default. The **English / Chinese** control switches interface
text and remembers the choice in this browser without saving labels or
resetting playback and edits. Task descriptions, annotator names, API values
and exported labels remain unchanged; use English for training definitions.

## Label an episode

1. Enter your name and choose an episode. The name is self-declared attribution,
   not an authenticated account. Watch the synchronized camera views; Space
   plays/pauses and Left/Right step by one frame.
2. Select a subtask brush and drag on its track. Cover every native frame.
   Repeated and reordered subtasks are allowed; touching intervals of the same
   subtask form one run. Adjust boundaries by dragging or entering exact times.
3. Review every run's outcome. **C / Completed** is the initial choice; change
   it to **F / Failed** where appropriate. These map to `high` (1) and `low` (0)
   at the run's last frame.
4. Paint **Suboptimal** or **Regressed** quality spans. The **Nominal** brush
   erases exceptions. Unmarked time becomes Nominal supervision only when you
   confirm that the whole episode has been reviewed.
5. Inspect the progress preview. It uses the same target function and recipe
   settings as cache construction; it previews your labels, not model output.
6. **Save draft** to return later, or confirm whole-episode review and
   **Submit**. Submission requires full subtask coverage, valid outcomes, a
   computable target and your name. Use **Reopen for editing** to revise a
   submitted episode, then review and submit it again.

### Navigate long episodes

Use **+ / − / Fit** or **Ctrl/Cmd+wheel** to zoom. Both tracks zoom and scroll
together; playback keeps the playhead visible. Zoom in or use the Subtasks list
to read descriptions hidden at overview scale. Double-click a camera or use
its enlarge button to enlarge and restore it. Cameras pause together while
buffering; a failed camera is reported while other playable views can continue.

### Correct or remove labels

- **Erase part of a quality span:** choose **Nominal** (key **0**) and drag over
  the range. Subtask labels and quality outside that range stay unchanged.
- **Remove a whole span:** put down the brush, select the span and press
  **Delete/Backspace**, or use **×** in the quality-span list.
- **Start over:** **Reset draft** asks for confirmation, clears both tracks and
  resets review confirmation. Undo remains available until saving.

**Esc never deletes completed spans.** It cancels an active gesture or edit,
or closes the active modal, enlargement or selection. Ctrl/Cmd+Z undoes label
and outcome edits. Finish or cancel open intervals before saving or changing
episodes. **Export saved JSON** downloads the saved canonical label document,
including outcomes.

There is one shared label file per episode. If another tab or person saves
after you loaded it, your save reports a conflict and preserves local edits.
Export unsaved labels if needed; **Reload saved labels** explicitly discards
local edits and loads the current file. The editor does not keep separate
annotator submissions or compute inter-annotator agreement.

## Build a cache from submitted labels

Use the same recipe after submitting its selected episodes:

```bash
python -m grana.cli build-cache --config "$GRANA_LABEL_CONFIG"
```

With an explicit list such as `[0, 1]`, every selected episode must have
finished labels; a missing file or draft stops the build. `episodes: all`
displays all source episodes in the editor and builds from submitted ones
only. It fails when none are submitted. Choose a fresh `output` directory;
existing caches are never overwritten.

Files use `labels/episode_000000.json` and the `grana.labels.v1` schema, with
half-open native-frame spans `[start_frame, end_frame)`. Current files bind
labels to the source episode, clock, videos and ordered stage vocabulary.
Changed evidence or stages are rejected. A dataset can move when its validated
episode identity remains unchanged. Cache provenance records the saved file
and annotator.

Recognized earlier 11-key `grana.labels.v1` files remain readable. Cache
construction checks their episode index, trajectory identity, native clock and
stage labels; the editor upgrades them with current binding and review fields
on their next save. Other legacy formats have no automatic migration.

## Existing annotation review

Use [draft import](ANNOTATION.md#review-candidates-and-import-drafts) to review
two-round API proposals in this editor, add outcomes and submit them. Imports
never overwrite saved labels or accept candidates for training automatically.

The older [grana annotate workflow](ANNOTATION.md#existing-grana-annotate-projects)
is separate and supports LeRobot v2 only. Its `review-annotations` and
`export-stages` commands produce reviewed stage revisions and stage-only JSONL.
That JSONL is not a human-label file or a complete reward-cache input.
