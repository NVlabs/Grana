# Human labeling in the browser

`grana label` opens a local editor for **native LeRobot v2** recordings. A human
marks subtasks, their Completed/Failed outcomes and action-quality intervals,
then explicitly confirms the whole episode was reviewed. The command serves
the page on CPU; it needs no model checkpoint, GPU or annotation API key.

## Configure and start

Use the [CPU labeling environment](installation.md#cpu-labeling-and-cache-construction)
or the [full project environment](../README.md#install), with `ffmpeg` and
`ffprobe` on `PATH`. In the CPU environment, replace `grana` in the commands
below with `python -m grana.cli`, running from the checkout root.
For an existing project, retain its active YAML and complete `.stage-versions/`
history; see [restoring project data](installation.md#reinstalling-or-moving-a-deployment).
For a new project, use the [standalone labeling schema](project-configuration.md#labeling-your-own-recordings)
to create a private recipe with your task, ordered stage descriptions, camera
feature keys and episode selection. No bundled task recipe or recording is
required. Choose and document task-specific target parameters before training.

The editor requires a fixed frame cadence and videos whose presentation timestamps
start at zero. Source parquet timestamps may have a constant offset. Other video
clocks and LeRobot v3 input are rejected before labeling.

```bash
: "${GRANA_LABEL_CONFIG:?Set the absolute path to your labeling recipe}"
grana label --config "$GRANA_LABEL_CONFIG"
```

The recipe uses three separate locations: `source` is the original recording,
`labels` holds editable human label JSON files, and `output` is a new training
cache directory. Keep labels and caches outside the source and Git checkouts.
The original recordings are read only. A human-label recipe uses `labels`;
it cannot also use the existing `annotations` revision/supplement input.

Open the printed URL, normally **http://127.0.0.1:8790/**. The server stays in
the foreground until Ctrl+C. Use `--port 9012` to choose another local port.
For a remote machine, run this on your own computer while the server runs there:

```bash
ssh -N -L 8790:127.0.0.1:8790 user@label-host
```

Then open http://127.0.0.1:8790/ locally. No browser is launched automatically.

## Edit the sub-stage definitions

The **Subtask timeline** card directly below the videos contains an **Add stage**
button and a collapsible **Edit stages** panel. Add stage opens the panel and
focuses a new description. Edit descriptions, remove stages, or move them up and
down without leaving the video. Each stage needs a distinct nonempty English
description; a project supports 1–26 stages. Definitions apply to every episode
in this recipe. The right-hand **Edit stages** shortcut opens the same panel.

**Apply stages** saves unfinished episode edits as a draft first, then updates
the brushes, task descriptions and target preview without a server restart.
Finish or cancel an open span before applying. Existing intervals, action-quality
spans, and completed/failed outcomes carry into the new version for every saved
episode. Stable stage IDs keep intervals attached when descriptions or ordering
change. The current video position, selected brushes and valid Undo history stay
available. Migrated labels become drafts and need review before submitting.

A stage used by any episode cannot be removed until its intervals have been
reassigned or deleted. Rejected changes leave the current annotations available.
The previous version stays untouched, and a failed migration leaves the active
recipe and saved labels unchanged.

The active YAML recipe is updated atomically. The previous recipe and the new
recipe are recorded in `.stage-versions/version-*/` next to the original labels
directory; each new version has separate `labels/` and `cache/` destinations.
To return to a previous version, start `grana label --config` with its saved
recipe. `grana build-cache --config` with the active recipe uses the new stage
definitions and their new labels. Stale browser tabs cannot save old definitions
into the new version; export unsaved edits before reloading a stale tab.

Progress parameters follow retained stage IDs; added stages inherit the last
stage's parameters for preview. These edited configurations are marked
`parameter_status: example`. Review the parameters for the new stage granularity
before using the resulting targets for training. Manually editing the recipe
while the server is running requires restarting the server.

## Interface language

English is the default, including in browsers configured for another language.
The top-right **English / Chinese** control changes interface text immediately
and remembers the explicit preference in this browser. It is available before
sign-in and throughout annotation. Switching does not reload the page, save
labels, change video position, reset form inputs, or alter undo history.

English is the source language for code, configuration, API values, logs and
backend errors. Chinese translations live only in the UI message catalog.
Task/stage descriptions and annotator names are user data, remain unchanged by
language switching, and should use English for training definitions. API stage
IDs, action-quality values, outcome values and exported labels are never
localized. Add new interface copy to `grana-locales.js` and bind dynamic text
through `GRANA_I18N.text/attr` so it also changes without re-rendering the editor.

## Label an episode

1. Enter your name and choose an episode. The name is a self-declared label
   attribution stored in this browser, not an authenticated account. Watch the
   synchronized camera views; Space plays/pauses and Left/Right step by one frame.
2. Select a subtask brush and drag on the subtask track. Cover every native
   frame. Repeated and reordered subtasks are allowed; touching intervals of the
   same subtask form one run. Drag boundaries or use exact times to adjust them.
3. Inspect each run's outcome. **C / Completed** is the initial choice; click
   the outcome control to set **F / Failed** where appropriate. Completed maps
   to `high` (1) and Failed to `low` (0) at the run's last frame.
4. Paint **Suboptimal** or **Regressed** spans on the quality track. The Nominal
   brush erases these exceptions. Unmarked time becomes **Nominal** only as part
   of the human's confirmation that the whole episode has been reviewed.
5. Inspect the converted progress preview. It uses the same target function and
   recipe settings as cache construction, including the selected progress method.
   The curve also previews a held quality boundary drag without saving the candidate
   interval. Releasing the pointer commits it; Escape or Undo restores the
   corresponding curve. Window resizing does not interrupt target calculation,
   and delayed responses cannot replace a newer annotation preview.
   It previews labels, not a model prediction.
6. **Save draft** to return later, or tick the whole-episode review confirmation
   and **Submit**. Submission requires full subtask coverage, valid outcomes,
   a computable target, a name and explicit review confirmation. Use
   **Reopen for editing** to revise a submitted episode; submit it again when
   review is complete.

### Navigate long episodes

Use **+ / − / Fit** above the subtask track, or **Ctrl/Cmd+wheel** over either
track, to zoom. Both tracks zoom and scroll horizontally together; playback
keeps the playhead in view. Fit shows the whole episode. Short spans may have
shortened or hidden labels at overview scale; zoom in or use the Subtasks list
for their full descriptions.

Double-click a camera, or use its enlarge button, to enlarge or restore it.
Loading indicators show views still loading. During playback, cameras pause
together while a view buffers and resume in sync. A failed camera is reported
explicitly; other playable views can continue.

### Correct or remove labels

- **Erase part of a quality span:** choose **Nominal** (key **0**) and drag over
  the range to erase. Only that range returns to Nominal; the remaining parts
  of Suboptimal/Regressed spans stay labeled. Subtask labels are unchanged.
- **Remove a whole span:** put down an active brush by clicking it again, select
  the span, then press **Delete/Backspace**. The quality-span list's **×** removes
  that span directly.
- **Start over:** **Reset draft** asks for confirmation, then clears both the
  subtask and quality tracks and resets review confirmation. Undo is available
  until saving.

**Esc never deletes completed spans.** During painting, span creation or boundary
resizing, it cancels only that gesture, keeping the brush and camera enlargement.
Otherwise it handles the first applicable state in this order: close an open
modal; cancel a pending span or exact-time edit and put down brushes; restore
an enlarged camera; close an open on-page guide; deselect and put down brushes.

Ctrl/Cmd+Z undoes label and outcome edits. Finish an open interval with its end
boundary, or cancel it before saving or changing episodes. **Export saved JSON**
downloads the saved canonical label document, including the subtask outcomes.

There is one shared label file per episode. Saves check the loaded revision UUID;
if another tab or person has saved it since you opened it, the editor reports a
conflict and preserves your local edits. Export the unsaved labels if needed,
then use **Reload saved labels** to explicitly discard your local edits and
load the current file. This editor does
not maintain separate annotator submissions or compute inter-annotator agreement.

## Build a cache from submitted labels

Use the same recipe after submitting its selected episodes:

```bash
grana build-cache --config "$GRANA_LABEL_CONFIG"
```

With an explicit episode list such as `[0, 1]`, every selected episode must have
finished, reviewed labels; a missing file or draft stops cache construction.
For human-label recipes, `episodes: all` displays all source episodes in the UI
and builds a cache from only those with submitted labels. It fails if none are
submitted. Drafts never become training supervision.

Files are stored as `labels/episode_000000.json`, using `grana.labels.v1` with
half-open native-frame spans `[start_frame, end_frame)`. They bind the labels to
the source episode, clock, videos and stage vocabulary. Reusing labels after
changing these inputs is rejected. Cache provenance records the saved label
file and annotator. Choose a fresh `output` directory for each cache build;
existing caches are never overwritten.

Older donor-branch label files without source binding and explicit review metadata
are rejected. Keep those files separate, choose a new `labels` directory and
re-label/review the recordings in this editor. Import and migration of those
older files are not supported.

## Existing annotation review

The [API annotation workflow](ANNOTATION.md) remains separate. Set
`GRANA_ANNOTATION_RUN` to your existing API run directory and
`GRANA_STAGE_EXPORT` to the desired JSONL output file:

```bash
grana review-annotations --run "$GRANA_ANNOTATION_RUN" --port 8788
grana export-stages --run "$GRANA_ANNOTATION_RUN" --output "$GRANA_STAGE_EXPORT"
```

That workflow reviews machine stage candidates, writes immutable revisions with
reviewer decisions and evidence, and exports stage-only JSONL. Its export needs
explicit `--revision EPISODE=PATH` arguments to select reviewed revisions; without
them it exports the original candidates with pending-human-review provenance.
Human label files from this editor are a separate direct input to `build-cache`.
The editor does not automatically accept or import machine candidates, and stage
exports alone do not provide reward outcomes or action quality.

## Developer verification

From the source checkout, run the focused backend, CLI, UI and packaging checks.
The JavaScript checks need Node.js 18+. The optional browser smoke test needs
Node.js 22+ and Chrome/Chromium on `PATH` (or set `CHROME_BIN` to its executable).
Its synthetic recordings require `ffmpeg` with libx264 and `ffprobe` on `PATH`.
No npm packages are needed; checks skip when their prerequisites are missing.
Set `GRANA_BROWSER_OUTPUT=/path/to/results` to retain browser screenshots and
receipts in a chosen directory.

```bash
python -m pytest tests/test_labeling.py tests/test_label_cli.py \
  tests/test_label_ui.py tests/test_label_ui_package.py
GRANA_BROWSER_SMOKE=1 python -m pytest -s tests/test_label_ui_browser.py
```
