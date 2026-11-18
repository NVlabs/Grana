/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* lab-interact.js — direct manipulation for the Annotation Lab (annotation/index.html), loaded after agreement.js.
 *
 * Everything goes through the lab's own closure, exposed by the api() patch as window.__agreementCtl (state, seek,
 * remember, changed, renderPoints, validateClosedInterval, validateStageInterval, addPoint, addStageInterval, ...), so
 * Save / Submit / Undo / autosave keep working exactly as before. Nothing here writes records directly.
 *
 *   drag on a track          scrub (the playhead follows on every track); Shift-drag always scrubs
 *   drag with a label picked create a span (ghost preview, frame-snapped, min 2 frames, clamped to the free gap; the
 *                            start/end snap to the gap's edges within ~0.75 s, so touching subtasks need no precision).
 *                            Subtask track: a new interval that touches a neighbour with the SAME subtask extends that
 *                            neighbour instead ("Extended S1 to 0.00–12.30 s") — touching intervals of one subtask are one
 *                            segment in the paper's target; the merged interval keeps the outcome of its last part. The
 *                            same rule runs after S/E, the Add button, double-click and Fill gaps (one undo step each).
 *   drag an edge handle      move a boundary (snaps to frames, neighbours, clip start/end; validated by the lab). With a
 *                            label picked and the pointer on the GAP side of a boundary, create wins over resize.
 *   drag with nothing picked scrubs, and says why nothing was marked (#annotation-feedback) + pulses the picker
 *   click a span             select (click empty track to deselect) · Delete/Backspace removes it (Undo restores); Esc deselects
 *   double-click a gap       subtask track: fill the gap with the picked subtask · Fill gaps button fills them all
 *   keyboard                 Space play · ←/→ frame · Shift+←/→ 1 s · Home/End · S/E start/end at playhead · 1–9 labels
 *                            (Shift = subtask n, Shift+0 = S10) · P (Direct points) time = playhead · ⌘/Ctrl+Z undo ·
 *                            Esc cancels a drag first, otherwise cancels pending edits, deselects and puts brushes down ·
 *                            ? shortcuts sheet (#lab-shortcuts)
 *   hover                    time / frame tooltip (#lab-track-tip)
 *   armed brush              the track a picked brush paints on gets an outline in the brush colour and a pencil cursor
 *   zoom                     − / + / Fit in the subtask track header, or Ctrl/⌘+wheel over a track (around the pointer):
 *                            both tracks zoom and scroll sideways together; while playing, the view keeps the playhead;
 *                            a stroke or an edge drag stops at the view's edge
 *   #lab-transport           sticky play / ±1f / time / seek bar, shown while the video card is scrolled out of view
 *                            (over the tracks' column, clear of the side panel)
 *   #lab-submit-reason       one live line under Submit saying why it is disabled (also the button's title)
 *   #lab-fill-gaps           subtask track: fill every gap with the picked subtask; small gaps get an amber marker
 *   outcome                  Subtask + AQ / Subtask only: every subtask interval gets a C / F control in the Subtasks
 *                            list (Completed default · Failed) and a letter mark at its end on the track (click
 *                            cycles). Stored in GRANA_LAB_API.outcomes (side store keyed by start/end); the shim writes the
 *                            companion Subtask outcome record on save. A standalone Subtask outcome record wins (note shown).
 *   autosave                 drafts every 30 s while dirty (undo history is kept across the autosave)
 * Geometry mirrors agreement.js makeTimeline: own tracks left 8 / plot 784 × zoom (viewBox width read from the track), curve
 * plots left 48 / 736 of 800, row top 11, row height read from the background rect. */
(() => {
  "use strict";
  const I18N = window.GRANA_I18N || {text: (n, s) => { n.textContent = s; }, attr: (n, k, s) => n.setAttribute(k, s), node: s => document.createTextNode(s), source: n => n.textContent, t: s => s};
  const ctl = window.__agreementCtl, API = window.GRANA_LAB_API;
  if (!ctl || !API) return;
  const state = ctl.state;
  const $ = (id) => document.getElementById(id);
  const SVGNS = "http://www.w3.org/2000/svg";
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fps = () => (state.episode && state.episode.fps) || 30;
  const duration = () => (state.episode && state.episode.duration_seconds) || 0;
  const GEO = {own: {left: 8, plot: 784}, curve: {left: 48, plot: 736}};
  const TOP = 11;
  const CREATE_SNAP = 0.75;   // seconds: a new span's ends snap to the free gap's edges within this (or the pixel tolerance)
  const SNAP_MAX = 0.25;      // seconds: the furthest a painted end or dragged edge jumps to a boundary; separate events stay separate
  const viewW = (svg) => Number((svg.getAttribute("viewBox") || "0 0 800 84").split(/\s+/)[2]) || 800;
  const geo = (svg) => (svg.classList.contains("reward-curve-svg") ? GEO.curve : {left: GEO.own.left, plot: viewW(svg) - 16});   // 784 × zoom
  const viewH = (svg) => Number((svg.getAttribute("viewBox") || "0 0 800 84").split(/\s+/)[3]) || 84;
  const rowH = (svg) => { const bg = svg.querySelector("rect.unlabelled-background"); return bg ? Number(bg.getAttribute("height")) || 23 : 23; };
  const signedIn = () => Boolean(state.session && state.session.annotator_id);

  /* ---------------- feedback: the lab's green box, with an amber "info" tone for hints and warnings ---------------- */
  let feedbackKind = null;
  const INFO_PATTERNS = /do not cover the clip yet|unconfirmed|^Choose |^Pick |^Sign in|^Enter a name|first\.$|^No pending|^Finish or cancel|^Cancelled|cannot|^Draft restored/i;
  (function watchFeedback() {
    const node = $("annotation-feedback"); if (!node) return;
    new MutationObserver(() => {
      const kind = feedbackKind || (INFO_PATTERNS.test(I18N.source(node).trim()) ? "info" : null);
      node.classList.toggle("info", kind === "info"); feedbackKind = null;
    }).observe(node, {childList: true, characterData: true, subtree: true});
  })();
  const feedback = (message, kind = null) => { feedbackKind = kind; ctl.annotationFeedback(message); };
  /** Briefly highlight the control the visitor should use next. */
  function pulse(node) {
    if (!node) return;
    node.classList.remove("lab-pulse"); void node.offsetWidth; node.classList.add("lab-pulse");
    setTimeout(() => node.classList.remove("lab-pulse"), 1400);
  }

  /* ---------------- geometry ---------------- */
  function timeAt(svg, clientX) {
    const b = svg.getBoundingClientRect(); if (!b.width) return 0;
    const x = (clientX - b.left) / b.width * viewW(svg), g = geo(svg);
    return clamp((x - g.left) / g.plot, 0, 1) * duration();
  }
  const xOf = (svg, t) => { const g = geo(svg); return g.left + (duration() ? t / duration() : 0) * g.plot; };
  /** Frame grid, 6 decimals like the lab's boundaryTime; within half a frame of the clip end snaps to the end. */
  function snap(t) {
    const f = fps(), d = duration();
    let v = Math.round(Math.round(clamp(t, 0, d) * f) / f * 1e6) / 1e6;
    if (v > d || Math.abs(v - d) <= 0.5 / f + 1e-9) v = d;
    return v;
  }
  /** Seconds worth of ~6 px on this track, at least 3 frames, capped at SNAP_MAX (or one frame, if longer): the snapping
   *  tolerance for neighbours / clip ends. On a long clip 6 px is over a second; zooming in shrinks it. */
  const tolerance = (svg) => { const b = svg.getBoundingClientRect(); const px = b.width ? 6 / b.width * viewW(svg) / geo(svg).plot * duration() : 0; return Math.min(Math.max(3 / fps(), px), Math.max(SNAP_MAX, 1 / fps())); };
  /** The closest of `targets` within `tol` of v, else v (snapNear takes the first match instead). */
  const nearest = (v, targets, tol) => {
    let best = v, gap = tol + 1e-9;
    for (const u of targets) { const d = Math.abs(v - u); if (d <= gap) { best = u; gap = d; } }
    return best;
  };
  const snapNear = (v, targets, tol) => { for (const u of targets) if (u !== null && u !== undefined && Math.abs(v - u) <= tol) return u; return v; };

  /* ---------------- the two editable tracks ---------------- */
  const trackOf = (host) => (host.id === "own-stage-timeline" ? "stage" : "aq");
  const items = (track) => (track === "stage" ? state.stageIntervals : state.intervals);
  const indexOf = (track, key) => items(track).findIndex((iv) => Math.abs(iv.start - key.start) < 1e-6 && Math.abs(iv.end - key.end) < 1e-6);
  function canCreate(track) {
    if (!ctl.canEdit()) return false;
    if (track === "stage") return ctl.independentStages() && Number.isInteger(state.selectedSemanticStage) && !state.pendingStageInterval && state.editingStageInterval === null;
    if (ctl.isCurve() || ctl.isSubtaskOnly() || state.annotationFormat === "legacy-points") return false;
    if (state.pendingInterval || state.editingInterval !== null || !state.selectedLabel) return false;
    if (ctl.requiresExceptionStage() && state.selectedStage === null) return false;
    if (ctl.defaultNormal() && state.selectedLabel === "normal") return false;
    return true;
  }
  /** Grana brushes: with a subtask or a quality label picked (Nominal erases), a drag paints that stretch, replacing what was
   *  there; touching spans with the same label merge. Without a brush the Lab's select / resize / scrub behaviour applies. */
  function canPaint(track) {
    if (!ctl.canEdit() || ctl.isCurve()) return false;
    if (track === "stage") return ctl.independentStages() && Number.isInteger(state.selectedSemanticStage) && !state.pendingStageInterval && state.editingStageInterval === null;
    if (ctl.isSubtaskOnly()) return false;
    return !state.pendingInterval && state.editingInterval === null && ["normal", "suboptimal", "failure"].includes(state.selectedLabel);
  }
  function paintColor(track) {
    if (track === "stage") return stageColor(state.selectedSemanticStage);
    return state.selectedLabel === "normal" ? "#22503f" : ctl.labelColor(state.scheme, state.selectedLabel);
  }
  function commitPaint(track, a, b) {
    const key = track === "stage" ? "stage" : "label";
    const value = track === "stage" ? state.selectedSemanticStage : state.selectedLabel;
    const kept = [];
    for (const iv of items(track)) {
      if (iv.end <= a + 1e-9 || iv.start >= b - 1e-9) { kept.push({...iv}); continue; }
      if (iv.start < a - 1e-9) kept.push({...iv, end: a});
      if (iv.end > b + 1e-9) kept.push({...iv, start: b});
    }
    const painted = track === "stage" ? {start: a, end: b, stage: value} : {start: a, end: b, label: value, stage: null};
    if (!(track === "aq" && value === "normal")) kept.push(painted);
    kept.sort((x, y) => x.start - y.start);
    const merged = [];
    for (const iv of kept) {
      const last = merged[merged.length - 1];
      if (track === "aq" && last && last[key] === iv[key] && touching(last, iv)) last.end = Math.max(last.end, iv.end);
      else merged.push(iv);
    }
    const list = merged.filter((iv) => iv.end - iv.start >= 1 / fps() - 1e-9);
    ctl.remember();
    if (track === "stage") state.stageIntervals = list; else state.intervals = list;
    const m = track === "stage" ? mergeTouchingStages(painted) : null;
    const shown = m ? m.iv : list.find((iv) => iv.start <= a + 1e-9 && iv.end >= b - 1e-9);
    selection = shown ? {track, start: shown.start, end: shown.end} : null;
    ctl.changed(); ctl.showNotice("");
    const name = track === "stage" ? `S${value + 1}` : value === "normal" ? "Nominal" : ((ctl.schemeLabel(state.scheme, value) || {}).name || value);
    const left = track === "stage" && ctl.independentStages() ? ctl.stageCoverage().missing : 0;
    feedback(`Painted ${name} over ${ctl.timeRange(a, b)}${left > 1e-6 ? ` · ${left.toFixed(1)} s still without a subtask` : ""} · unsaved.`);
  }
  /** Why a drag on this track cannot create anything right now (null when it can). */
  function blocker(track) {
    if (!signedIn()) return {text: "Enter a name to mark spans · drag scrubs the video.", focus: $("account-button")};
    if (ctl.isSubmitted()) return {text: "Submitted record · Reopen for editing to change it.", focus: $("reopen-button")};
    if (!ctl.canEdit()) return null;
    if (track === "stage") {
      if (!ctl.independentStages()) return null;
      if (state.pendingStageInterval) return {text: "End or cancel the pending subtask first.", focus: $("add-stage-interval")};
      if (state.editingStageInterval !== null) return {text: "Finish or cancel the subtask edit first.", focus: $("add-stage-interval")};
      return {text: "Pick a subtask first · Shift-drag scrubs.", focus: $("independent-stage-select")};
    }
    if (ctl.isCurve() || ctl.isSubtaskOnly() || state.annotationFormat === "legacy-points") return null;
    if (state.pendingInterval) return {text: "End or cancel the pending span first.", focus: $("add-point")};
    if (state.editingInterval !== null) return {text: "Finish or cancel the interval edit first.", focus: $("add-point")};
    if (ctl.requiresExceptionStage() && state.selectedStage === null) return {text: "Pick a subtask and High / Mid / Low first.", focus: $("stage-select")};
    if (!state.selectedLabel || (ctl.defaultNormal() && state.selectedLabel === "normal")) return {text: ctl.defaultNormal() ? "Pick Suboptimal or Regressed first · Shift-drag scrubs." : "Pick a label first · Shift-drag scrubs.", focus: $("quality-buttons")};
    return null;
  }
  /** The free gap [lo, hi] of a track around time t, or null when t lies inside a closed interval. */
  function gapAround(track, t) {
    let lo = 0, hi = duration();
    for (const iv of items(track)) {
      if (iv.start < t && t < iv.end) return null;
      if (iv.end <= t + 1e-9 && iv.end > lo) lo = iv.end;
      if (iv.start >= t - 1e-9 && iv.start < hi) hi = iv.start;
    }
    return hi - lo > 1e-9 ? [lo, hi] : null;
  }
  /** Every free gap of a track, in order. */
  function gaps(track) {
    const out = []; let cursor = 0;
    for (const iv of [...items(track)].sort((a, b) => a.start - b.start)) { if (iv.start > cursor + 1e-9) out.push([cursor, iv.start]); cursor = Math.max(cursor, iv.end); }
    if (duration() > cursor + 1e-9) out.push([cursor, duration()]);
    return out;
  }
  const makeInterval = (track, start, end) => (track === "stage" ? {start, end, stage: state.selectedSemanticStage}
    : {start, end, label: state.selectedLabel, stage: ctl.requiresExceptionStage() ? state.selectedStage : null});
  /** Subtask descriptions run to a full sentence; feedback lines show "S3 Drill both wheel screws…" (the list keeps the full text). */
  const shortStage = (k) => { const n = String(ctl.stageName(k) || "").replace(/^\[[^\]]*\]\s*/, ""); return n.length > 44 ? `${n.slice(0, 42).trimEnd()}…` : n; };
  const labelName = (track, iv) => (track === "stage" ? `S${iv.stage + 1} ${shortStage(iv.stage)}` : ((ctl.schemeLabel(state.scheme, iv.label) || {}).name || iv.label));
  const stageColor = (k) => ctl.stageColor(k);   // the one subtask palette (agreement.js)

  let selection = null, selectionLoad = 0;   // {track, start, end, clicked?} · the episode load (state.contextSerial) it was made in
  /** A selection never outlives its load: the next episode (or a reload) opens with nothing selected, so Delete is inert. */
  function dropStaleSelection() { if (selectionLoad !== state.contextSerial) { selectionLoad = state.contextSerial; selection = null; } }
  /** Both tracks and pending starts, to the µs: a drag commits only onto the labels it began on. */
  const us = (iv) => iv && [Math.round(iv.start * 1e6), Math.round(iv.end * 1e6), iv.stage, iv.label];
  const labelsSig = () => JSON.stringify([state.stageIntervals.map(us), state.intervals.map(us), us(state.pendingStageInterval), us(state.pendingInterval)]);

  /* ---------------- segments: touching intervals of the same subtask are ONE segment (paper: find_subgoal_segments) ---------------- */
  const sameSpan = (a, b) => Math.abs(a.start - b.start) < 1e-6 && Math.abs(a.end - b.end) < 1e-6;
  const touching = (a, b) => Math.abs(b.start - a.end) <= 1 / fps() + 1e-6;   // at most one frame apart
  /** Merge every run of touching same-subtask intervals of the subtask track (the caller has called ctl.remember(), so the
   *  merge shares its undo step). The merged interval takes the outcome of its LAST part — how the segment ends; the parts'
   *  own non-Complete entries stay in the side store so Undo brings them back. Returns {iv, parts, outcome} for the merged
   *  interval that contains `focus` (or the first merged one), null when nothing touched. */
  function mergeTouchingStages(focus = null) {
    const sorted = [...state.stageIntervals].sort((a, b) => a.start - b.start); if (sorted.length < 2) return null;
    const runs = [];
    for (const iv of sorted) {
      const last = runs[runs.length - 1];
      if (last && last.stage === iv.stage && touching(last, iv)) { last.end = Math.max(last.end, iv.end); last.parts.push(iv); }
      else runs.push({start: iv.start, end: iv.end, stage: iv.stage, parts: [iv]});
    }
    const mergedRuns = runs.filter((r) => r.parts.length > 1); if (!mergedRuns.length) return null;
    const c = outcomeCtx(), before = c ? outcomeList() : [];
    const outcomeOf = (r) => (c ? r.parts.some(iv => OUT.lookup(before, iv) === "unknown") ? "unknown" : OUT.lookup(before, r.parts[r.parts.length - 1]) : "complete");
    state.stageIntervals = runs.map(({start, end, stage}) => ({start, end, stage}));
    if (c) {
      const list = before.filter((o) => !mergedRuns.some((r) => sameSpan(o, r)));
      for (const r of mergedRuns) list.unshift({start: r.start, end: r.end, stage: r.stage, outcome: outcomeOf(r)});
      OUT.set(...c, list);
    }
    const hit = (focus && mergedRuns.find((r) => r.parts.some((pt) => sameSpan(pt, focus)))) || mergedRuns[0];
    return {iv: {start: hit.start, end: hit.end, stage: hit.stage}, parts: hit.parts.length, outcome: outcomeOf(hit)};
  }
  const extendedText = (m) => `Extended S${m.iv.stage + 1} to ${ctl.timeRange(m.iv.start, m.iv.end)}${m.outcome !== "complete" ? ` · ${OUT_NAME[m.outcome]}` : ""} · touching S${m.iv.stage + 1} intervals form one segment`;
  /** Run ctl.addStageInterval (S / E, the Add button) and merge the interval it closed with touching same-subtask neighbours. */
  function addStageMerged() {
    const snapshot = state.stageIntervals.map((iv) => ({...iv}));
    ctl.addStageInterval();
    afterStageAdd(snapshot);
  }
  function afterStageAdd(snapshot) {
    const added = state.stageIntervals.find((iv) => !snapshot.some((o) => sameSpan(o, iv))); if (!added) return;
    const m = mergeTouchingStages(added); if (!m) return;
    selection = {track: "stage", start: m.iv.start, end: m.iv.end};
    ctl.changed(); ctl.showNotice("");
    const left = ctl.independentStages() ? ctl.stageCoverage().missing : 0;
    feedback(`${extendedText(m)}${left > 1e-6 ? ` · ${left.toFixed(1)} s still uncovered` : ""}.`);
  }
  let addSnapshot = null;   // subtask track before a click on the lab's Add button (capture phase) → merged after it ran (bubble phase)
  document.addEventListener("click", (e) => { addSnapshot = e.target.closest && e.target.closest("#add-stage-interval") ? state.stageIntervals.map((iv) => ({...iv})) : null; }, true);
  document.addEventListener("click", (e) => { if (addSnapshot && e.target.closest && e.target.closest("#add-stage-interval")) { const snap = addSnapshot; addSnapshot = null; afterStageAdd(snap); } });

  function commitCreate(track, a, b) {
    const iv = makeInterval(track, a, b);
    try {
      if (track === "stage") ctl.validateStageInterval(iv); else ctl.validateClosedInterval(iv);
      ctl.remember(); items(track).push(iv); items(track).sort((x, y) => x.start - y.start);
      const m = track === "stage" ? mergeTouchingStages(iv) : null, shown = m ? m.iv : iv;
      selection = {track, start: shown.start, end: shown.end};
      ctl.changed(); ctl.showNotice("");
      const left = track === "stage" && ctl.independentStages() ? ctl.stageCoverage().missing : 0;
      feedback(`${m ? extendedText(m) : `Marked ${labelName(track, iv)} ${ctl.timeRange(a, b)}`} · drag an edge to adjust${left > 1e-6 ? ` · ${left.toFixed(1)} s still uncovered` : ""}.`);
      return true;
    } catch (error) { ctl.showNotice(error.message, "error"); ctl.renderPoints(); return false; }
  }
  function commitResize(track, index, iv) {
    try {
      if (track === "stage") ctl.validateStageInterval(iv, index); else ctl.validateClosedInterval(iv, index);
      ctl.remember(); items(track)[index] = iv; items(track).sort((x, y) => x.start - y.start);
      selection = {track, start: iv.start, end: iv.end};
      ctl.changed(); ctl.showNotice("");
      feedback(`Adjusted ${labelName(track, iv)} to ${ctl.timeRange(iv.start, iv.end)} · unsaved.`);
    } catch (error) { ctl.showNotice(error.message, "error"); ctl.renderPoints(); }
  }
  /** Move the boundary shared by two touching spans (one undo step). */
  function commitBoundary(track, i, iv, j, peer) {
    ctl.remember();
    const list = items(track); list[i] = iv; list[j] = peer; list.sort((x, y) => x.start - y.start);
    selection = {track, start: iv.start, end: iv.end};
    ctl.changed(); ctl.showNotice("");
    const at = Math.abs(iv.start - peer.end) < 1e-6 ? iv.start : iv.end;
    feedback(`Moved the boundary between ${labelName(track, peer)} and ${labelName(track, iv)} to ${at.toFixed(2)} s · unsaved.`);
  }
  function removeSelected() {
    dropStaleSelection();
    if (!selection || !ctl.canEdit()) return false;
    const {track} = selection, i = indexOf(track, selection); if (i < 0) { selection = null; return false; }
    const iv = items(track)[i];
    ctl.remember(); items(track).splice(i, 1);
    if (track === "stage") state.editingStageInterval = null; else state.editingInterval = null;
    selection = null; ctl.changed(); ctl.showNotice("");
    feedback(`Removed ${labelName(track, iv)} ${ctl.timeRange(iv.start, iv.end)} · Undo restores it.`);
    return true;
  }
  /** Subtask track: fill every gap with the picked subtask (one undo step). */
  function fillGaps() {
    if (!ctl.canEdit() || !ctl.independentStages()) return false;
    if (!Number.isInteger(state.selectedSemanticStage)) { feedback("Pick a subtask first, then Fill gaps.", "info"); pulse($("independent-stage-select")); return false; }
    if (state.pendingStageInterval || state.editingStageInterval !== null) { feedback("End or cancel the pending subtask first.", "info"); return false; }
    const free = gaps("stage"); if (!free.length) { feedback("No gaps left; the subtask track covers the clip."); return false; }
    ctl.remember();
    for (const [lo, hi] of free) state.stageIntervals.push(makeInterval("stage", lo, hi));
    state.stageIntervals.sort((x, y) => x.start - y.start);
    const merged = mergeTouchingStages();
    selection = null; ctl.changed(); ctl.showNotice("");
    feedback(`Filled ${free.length} gap${free.length === 1 ? "" : "s"} with S${state.selectedSemanticStage + 1} ${shortStage(state.selectedSemanticStage)}${merged ? " · joined with the touching S" + (state.selectedSemanticStage + 1) + " intervals (one segment)" : ""} · Undo restores them.`);
    return true;
  }
  function refreshSelection() {
    dropStaleSelection();
    document.querySelectorAll("#own-timeline .lab-body, #own-stage-timeline .lab-body").forEach((r) => {
      const s = Number(r.dataset.start), e = Number(r.dataset.end);
      r.classList.toggle("lab-selected", Boolean(selection && selection.track === r.dataset.track && Math.abs(selection.start - s) < 1e-6 && Math.abs(selection.end - e) < 1e-6));
    });
  }

  /* ---------------- per-subtask outcome (paper: o_k = completed 1 / failed 0) ---------------- */
  // Grana: binary subtask outcomes as in the paper (completed 1 / failed 0).
  const OUT = API.outcomes, OUTCOMES = ["complete", "failed"], OUT_LETTER = {complete: "C", failed: "F", unknown: "?"}, OUT_NAME = {complete: "Completed", failed: "Failed", unknown: "Needs review"};
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) I18N.text(n, text); return n; };
  const outcomeCtx = () => (OUT && state.episode && state.scheme && signedIn() && ctl.independentStages() ? [state.episode.id, state.session.annotator_id, state.scheme.id] : null);
  const standaloneOutcome = () => { const c = outcomeCtx(); return c ? OUT.standalone(c[0], c[1]) : null; };
  /** Side-store list for the current record, kept in step with state.stageIntervals: seeded from the stored Subtask outcome
   *  record when the record (re)loads, matched by start/end, an edited interval keeps its outcome while it overlaps the old
   *  span and has the same subtask, new intervals are Complete. Non-complete entries of removed intervals stay (Undo). */
  function outcomeList() {
    const c = outcomeCtx(); if (!c) return [];
    let list = OUT.get(...c);
    const used = new Set();
    const next = state.stageIntervals.map((iv) => {
      let hit = list.find((o) => o.stage === iv.stage && OUT.sameSpan(o, iv));
      if (!hit) hit = list.find((o) => o.outcome === "unknown" && o.stage === iv.stage && o.start < iv.end && o.end > iv.start);
      if (!hit) hit = list.find((o) => !used.has(o) && o.stage === iv.stage && o.start < iv.end && o.end > iv.start && !state.stageIntervals.some((x) => OUT.sameSpan(x, o)));
      if (hit) used.add(hit);
      return {start: iv.start, end: iv.end, stage: iv.stage, outcome: hit ? hit.outcome : "complete"};
    });
    const stale = list.filter((o) => !used.has(o) && o.outcome !== "complete" && !next.some((n) => OUT.sameSpan(n, o))).slice(-40);
    const merged = [...next, ...stale];
    if (JSON.stringify(merged) !== JSON.stringify(list)) OUT.set(...c, merged);
    return merged;
  }
  function setOutcome(index, outcome) {
    const c = outcomeCtx(), iv = state.stageIntervals[index];
    if (!c || !iv || !ctl.canEdit() || !OUTCOMES.includes(outcome) || standaloneOutcome()) return false;
    ctl.remember();
    const list = outcomeList().filter((o) => !OUT.sameSpan(o, iv) && !(o.outcome === "unknown" && o.stage === iv.stage && o.start < iv.end && o.end > iv.start));
    list.unshift({start: iv.start, end: iv.end, stage: iv.stage, outcome});
    OUT.set(...c, list);
    ctl.changed();
    decorateOutcomes(true); for (const host of hosts) decorate(host, true);
    feedback(`S${iv.stage + 1} ${ctl.timeRange(iv.start, iv.end)} · ${OUT_NAME[outcome]}${outcome === "failed" ? " · its last attempt targets 0" : ""} · unsaved.`);
    return true;
  }
  let outcomeOpened = null;
  /** C / P / F control on every row of the Subtasks list (#stage-interval-list, re-rendered by the lab on each change). */
  function decorateOutcomes(force = false) {
    const listHost = $("stage-interval-list"), details = listHost && listHost.closest("details"); if (!listHost) return;
    const c = outcomeCtx(), standalone = c ? standaloneOutcome() : null, editable = ctl.canEdit();
    const rows = [...listHost.querySelectorAll(".interval-row")];
    const list = c && !standalone ? outcomeList() : [];
    const sig = JSON.stringify([c, Boolean(standalone), editable, rows.length, state.stageIntervals.map((iv) => [iv.start, iv.end, iv.stage, OUT ? OUT.lookup(list, iv) : ""])]);
    if (!force && listHost.dataset.labSig === sig && (!rows.length || rows[0].querySelector(".lab-outcome") || standalone)) return;
    listHost.dataset.labSig = sig;
    listHost.querySelectorAll(".lab-outcome, .lab-outcome-note").forEach((n) => n.remove());
    let help = details && details.querySelector(".lab-outcome-help"), sum = details && details.querySelector(".lab-outcome-sum");
    if (details && !help) { help = el("p", "small muted lab-outcome-help", "How the subtask ended; feeds the target's endpoint."); details.querySelector("summary").after(help); }
    if (details && !sum) { sum = el("span", "lab-outcome-sum"); details.querySelector("summary").append(sum); }
    if (help) help.hidden = !c || standalone || !rows.length;
    if (sum) I18N.text(sum, "");
    if (!c) return;
    if (standalone) { if (rows.length) listHost.prepend(el("p", "small muted lab-outcome-note", "Using your Subtask outcome record for outcomes.")); return; }
    let failed = 0, unknown = 0;
    rows.forEach((row, index) => {
      const iv = state.stageIntervals[index]; if (!iv) return;
      const cur = OUT.lookup(list, iv); if (cur === "failed") failed += 1; else if (cur === "unknown") unknown += 1;
      const group = el("span", "lab-outcome"); I18N.attr(group, "data-outcome-label", "Outcome"); group.setAttribute("role", "group"); I18N.attr(group, "aria-label", `Outcome of S${iv.stage + 1} ${ctl.timeRange(iv.start, iv.end)}`); group.dataset.outcome = cur;
      if (cur === "unknown") group.append(el("span", "lab-outcome-unknown", "Needs review"));
      for (const o of OUTCOMES) {
        const b = el("button", `lab-oc ${o}${o === cur ? " on" : ""}`, OUT_LETTER[o]); b.type = "button"; I18N.attr(b, "title", `${OUT_NAME[o]}${o === "complete" ? " (default)" : ""}`);
        I18N.attr(b, "aria-label", OUT_NAME[o]); b.setAttribute("aria-pressed", String(o === cur)); b.disabled = !editable; b.dataset.index = String(index); b.dataset.outcome = o; group.append(b);
      }
      row.append(group);
    });
    if (sum) I18N.text(sum, [failed ? `${failed} failed` : "", unknown ? `${unknown} need review` : ""].filter(Boolean).join(", "));
    const ctxKey = c.join("/");
    if (details && rows.length && editable && outcomeOpened !== ctxKey) { outcomeOpened = ctxKey; details.open = true; }
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest(".lab-oc"); if (!b || b.disabled) return;
    e.preventDefault(); setOutcome(Number(b.dataset.index), b.dataset.outcome);
    if (e.detail === 0 && !b.isConnected) {   // Enter / Space: keep the focus on the re-rendered control
      const again = $("stage-interval-list").querySelector(`.lab-oc[data-index="${b.dataset.index}"][data-outcome="${b.dataset.outcome}"]`);
      if (again) again.focus();
    }
  });
  // a click on a track mark must not also seek (the lab's svg click handler)
  document.addEventListener("click", (e) => { if (e.target.closest && e.target.closest(".lab-outcome-mark")) e.stopPropagation(); }, true);
  /** Letter mark (C muted · P · F) at the end of each subtask segment on the own subtask track; click cycles the outcome.
   *  A Failed segment also gets a red cap above it, and every segment's tooltip reads "S5 · Failed · range". A segment too
   *  narrow for the mark shows F after its S# under the row (agreement.js), so the outcome shows without hovering; the
   *  Subtasks list still sets it. */
  function markOutcomes(svg, rh, editable) {
    if (!outcomeCtx() || standaloneOutcome()) return;
    const list = outcomeList();
    const node = (tag, attrs, text) => { const n = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) I18N.attr(n, k, String(v)); if (text != null) I18N.text(n, text); return n; };
    svg.querySelectorAll("rect.stage-timeline-segment.lab-body").forEach((rect) => {
      const index = Number(rect.dataset.index), iv = state.stageIntervals[index]; if (!iv) return;
      const x = Number(rect.getAttribute("x")), w = Number(rect.getAttribute("width")), oc = OUT.lookup(list, iv);
      const description = `S${iv.stage + 1} · ${OUT_NAME[oc]} · ${ctl.timeRange(iv.start, iv.end)}\n${ctl.stageName(iv.stage)}`;
      const tip = rect.querySelector("title"); if (tip) I18N.text(tip, description);
      I18N.attr(rect, "aria-label", description);
      if (oc === "failed") svg.append(node("rect", {x, y: TOP - 5, width: Math.max(w, 2), height: 3, rx: 1, class: "lab-failed-edge", "pointer-events": "none"}));
      if (w < 34) {   // agreement.js put this segment's S# under the row
        const label = oc === "failed" && [...svg.querySelectorAll("text.stage-narrow-label")].find((n) => n.dataset.start === rect.dataset.start && n.dataset.end === rect.dataset.end);
        if (label) label.append(node("tspan", {class: "lab-narrow-failed"}, " F"));
        return;
      }
      const g = document.createElementNS(SVGNS, "g");
      g.setAttribute("class", `lab-outcome-mark ${oc}`); g.dataset.index = String(index); g.dataset.outcome = oc;
      const box = document.createElementNS(SVGNS, "rect");
      for (const [k, v] of Object.entries({x: x + w - 22, y: TOP + rh - 17, width: 15, height: 14, rx: 3})) box.setAttribute(k, String(v));
      const t = document.createElementNS(SVGNS, "text"); t.setAttribute("x", String(x + w - 14.5)); t.setAttribute("y", String(TOP + rh - 6.5)); t.setAttribute("text-anchor", "middle"); I18N.text(t, OUT_LETTER[oc]);
      const title = document.createElementNS(SVGNS, "title"); I18N.text(title, `Outcome: ${OUT_NAME[oc]}${editable ? " · click to change" : ""}`);
      g.append(box, t, title); svg.append(g);
    });
  }

  /* ---------------- decorate every freshly rendered own track: bodies, edge handles, gap markers, selection ---------------- */
  function decorate(host, force = false) {
    const svg = host.querySelector("svg.timeline-svg"); if (!svg) return;
    if (svg.dataset.labDecorated && !force) return;
    if (force) svg.querySelectorAll(".lab-handle, .lab-gap, .lab-outcome-mark, .lab-failed-edge, .lab-narrow-failed").forEach((n) => n.remove());
    const editable = ctl.canEdit();
    svg.dataset.labDecorated = "1"; svg.dataset.labEditable = String(editable);
    if (svg.classList.contains("reward-curve-svg")) return;          // the lab's own anchors: click selects, double-click adds
    const track = trackOf(host), rh = rowH(svg);
    const rects = svg.querySelectorAll(track === "stage" ? "rect.stage-timeline-segment" : "rect.labelled-interval:not(.default-normal-interval)");
    rects.forEach((rect) => {
      const start = Number(rect.dataset.start), end = Number(rect.dataset.end), index = indexOf(track, {start, end});
      if (index < 0) return;
      rect.classList.add("lab-body"); rect.dataset.index = String(index); rect.dataset.track = track;
      if (!editable) return;
      for (const edge of ["start", "end"]) {
        const h = document.createElementNS(SVGNS, "rect");
        const attrs = {x: xOf(svg, edge === "start" ? start : end) - 5, y: TOP - 4, width: 10, height: rh + 8, class: "lab-handle", "data-edge": edge, "data-index": index, "data-track": track, fill: "transparent"};
        for (const [k, v] of Object.entries(attrs)) h.setAttribute(k, String(v));
        const title = document.createElementNS(SVGNS, "title"); I18N.text(title, `Drag to move the ${edge} (${(edge === "start" ? start : end).toFixed(2)} s)`); h.append(title);
        svg.append(h);
      }
    });
    // subtask track: the gaps that block Submit, drawn wide enough to see (an 11-px gap is invisible on a hatched background)
    if (track === "stage" && editable && ctl.independentStages() && state.stageIntervals.length) {
      for (const [lo, hi] of gaps("stage")) {
        const g = document.createElementNS(SVGNS, "rect"), x1 = xOf(svg, lo), x2 = xOf(svg, hi), w = Math.max(6, x2 - x1);
        for (const [k, v] of Object.entries({x: x2 - x1 < 6 ? (x1 + x2) / 2 - 3 : x1, y: TOP - 3, width: w, height: rh + 6, rx: 2, class: "lab-gap", "pointer-events": "none"})) g.setAttribute(k, String(v));
        const title = document.createElementNS(SVGNS, "title"); I18N.text(title, `Gap ${ctl.timeRange(lo, hi)} · double-click to fill it`); g.append(title);
        svg.append(g);
      }
    }
    if (track === "stage") markOutcomes(svg, rh, editable);
    refreshSelection();
  }
  const hosts = ["own-timeline", "own-stage-timeline"].map($).filter(Boolean);
  // a redrawn track starts unscrolled: put a zoomed view back where it was (zoom section below)
  for (const host of hosts) { new MutationObserver(() => { decorate(host); zoomUi(); applyView(); edgeTicks(host); }).observe(host, {childList: true}); decorate(host); }
  const stageList = $("stage-interval-list");
  if (stageList) { new MutationObserver(() => decorateOutcomes()).observe(stageList, {childList: true}); decorateOutcomes(); }
  // editability changes without a re-render (Reopen, draft load, sign-in, busy): re-decorate so the handles appear / vanish
  setInterval(() => {
    const editable = String(ctl.canEdit());
    for (const host of hosts) { const svg = host.querySelector("svg.timeline-svg"); if (svg && svg.dataset.labDecorated && svg.dataset.labEditable !== editable) decorate(host, true); }
    if (stageList && stageList.dataset.labEditable !== editable) { stageList.dataset.labEditable = editable; decorateOutcomes(true); }
    armedSync();
  }, 250);
  /** While a brush is picked, the track it paints on shows an outline in the brush colour and a brush cursor (.lab-armed),
   *  so a drag that will paint looks different from one that scrubs. */
  function armedSync() {
    for (const host of hosts) {
      const track = trackOf(host), armed = canPaint(track);
      if (host.classList.contains("lab-armed") !== armed) host.classList.toggle("lab-armed", armed);
      const brush = armed ? (track === "aq" && state.selectedLabel === "normal" ? "#30c5a5" : paintColor(track)) : null;   // the erase brush's chip colour
      if (brush && host.style.getPropertyValue("--lab-brush") !== brush) host.style.setProperty("--lab-brush", brush);
    }
  }
  ["click", "change", "keydown"].forEach((name) => document.addEventListener(name, () => setTimeout(armedSync), true));

  /* ---------------- pointer: scrub · create · resize · select ---------------- */
  let drag = null, raf = 0, queuedSeek = null;
  const SAVING = "Saving… try the stroke again.", BUSY = "Busy… try again in a moment.";
  const paintChoice = (track) => JSON.stringify(track === "stage" ? state.selectedSemanticStage : [state.selectedLabel, state.selectedStage]);
  const seekSoon = (t) => {
    queuedSeek = t;
    if (raf) return;
    const context = state.contextSerial;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const next = queuedSeek; queuedSeek = null;
      if (drag && context === state.contextSerial && next !== null) ctl.seek(next);
    });
  };
  function ghostFor(d) {
    const r = document.createElementNS(SVGNS, "rect");
    const color = d.track === "stage" ? stageColor(state.selectedSemanticStage) : ctl.labelColor(state.scheme, state.selectedLabel);
    for (const [k, v] of Object.entries({x: xOf(d.svg, d.t0), y: TOP, width: 0, height: d.rh, rx: 2, fill: color, class: "lab-ghost"})) r.setAttribute(k, String(v));
    d.svg.append(r); return r;
  }
  function onDown(e) {
    if (e.button !== 0 || drag) return;
    const svg = e.target.closest && e.target.closest("svg.timeline-svg"); if (!svg) return;
    const host = svg.closest("#own-timeline, #own-stage-timeline"); if (!host || !state.episode) return;
    const mark = e.target.closest(".lab-outcome-mark");
    if (mark && !e.shiftKey) { e.preventDefault(); if (ctl.canEdit()) setOutcome(Number(mark.dataset.index), OUTCOMES[(OUTCOMES.indexOf(mark.dataset.outcome) + 1) % OUTCOMES.length]); return; }
    const track = trackOf(host), isCurve = svg.classList.contains("reward-curve-svg"), rh = rowH(svg);
    const t0 = timeAt(svg, e.clientX), handle = e.target.closest(".lab-handle"), body = e.target.closest(".lab-body");
    const b = svg.getBoundingClientRect(), ySvg = b.height ? (e.clientY - b.top) / b.height * viewH(svg) : 0;
    const inRow = ySvg >= TOP - 6 && ySvg <= TOP + rh + 6;
    // A picked label + pointer on the free side of a boundary = the next touching span, not a resize of the previous one.
    const gap = !isCurve && !e.shiftKey && canCreate(track) ? gapAround(track, t0) : null;
    let mode = "scrub";
    if (!isCurve && !e.shiftKey && inRow && canPaint(track)) mode = "maybe-paint";
    else if (handle && ctl.canEdit() && !e.shiftKey && !(gap && inRow)) mode = "resize";
    else if (gap && inRow) mode = "maybe-create";
    else if (body && !e.shiftKey) mode = "select";
    drag = {mode, track, svg, host, rh, pointerId: e.pointerId, context: state.contextSerial, brush: paintChoice(track),
      x0: e.clientX, t0, moved: false, tol: tolerance(svg), inRow, shift: e.shiftKey, isCurve, sig: labelsSig()};
    // a brush stroke while the page is busy (a save in flight, an episode loading) only scrubs: say so instead of dropping it silently
    drag.held = (mode === "scrub" || mode === "select") && state.busy && inRow && !e.shiftKey && !isCurve && (track === "stage" ? Number.isInteger(state.selectedSemanticStage) : Boolean(state.selectedLabel))
      ? (API.saving && API.saving() ? SAVING : BUSY) : null;
    if (mode === "resize") {
      drag.index = Number(handle.dataset.index); drag.edge = handle.dataset.edge; drag.orig = {...items(track)[drag.index]}; drag.rect = svg.querySelector(`.lab-body[data-index="${drag.index}"]`);
      const at = drag.edge === "start" ? drag.orig.start : drag.orig.end;
      const j = items(track).findIndex((iv, k) => k !== drag.index && Math.abs((drag.edge === "start" ? iv.end : iv.start) - at) <= 1 / fps() + 1e-6);
      if (j >= 0) drag.peer = {index: j, orig: {...items(track)[j]}, rect: svg.querySelector(`.lab-body[data-index="${j}"]`)};
    }
    if (mode === "select") drag.index = Number(body.dataset.index);
    if (mode === "maybe-create") { drag.gap = gap; drag.ctol = Math.max(drag.tol, CREATE_SNAP); }
    ctl.pauseAll(); ctl.markInteraction();
    if (mode === "scrub") ctl.seek(t0);
    svg.classList.add("lab-dragging");
    e.preventDefault();
    window.addEventListener("pointermove", onMove); window.addEventListener("pointerup", onUp); window.addEventListener("pointercancel", onCancel);
  }
  function onMove(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    if (drag.context !== state.contextSerial || drag.sig !== labelsSig() || drag.svg.isConnected === false) { cancelDrag(); return; }
    if (!drag.moved && Math.abs(e.clientX - drag.x0) < 3) return;
    drag.moved = true;
    if (drag.mode === "scrub" || drag.mode === "select") { const t = timeAt(drag.svg, e.clientX); seekSoon(t); showTip(drag.svg, e.clientX, t, drag.held || undefined); return; }
    // a stroke or an edge stops at the track's visible edge, and snaps only to boundaries in view: zoomed in, it never
    // changes labels out of sight
    const box = drag.host.getBoundingClientRect(), left = box.left + drag.host.clientLeft, right = left + drag.host.clientWidth, f = fps();
    const v0 = Math.ceil(timeAt(drag.svg, left) * f - 1e-6) / f, v1 = Math.floor(timeAt(drag.svg, right) * f + 1e-6) / f;   // frames in view
    const x = clamp(e.clientX, left, right), t = clamp(timeAt(drag.svg, x), v0, v1), seen = (u) => u >= v0 - 1e-9 && u <= v1 + 1e-9;
    if (drag.mode === "maybe-paint") {
      drag.mode = "paint"; drag.ghost = ghostFor(drag); drag.ghost.setAttribute("fill", paintColor(drag.track)); drag.ghost.setAttribute("fill-opacity", "0.75");
    }
    if (drag.mode === "paint") {
      const edges = [0, duration(), ...items(drag.track).flatMap((iv) => [iv.start, iv.end])].filter(seen);
      const a = nearest(snap(Math.min(drag.t0, t)), edges, drag.tol), b = nearest(snap(Math.max(drag.t0, t)), edges, drag.tol);
      drag.a = a; drag.b = b;
      const x1 = xOf(drag.svg, a), x2 = xOf(drag.svg, b);
      drag.ghost.setAttribute("x", String(x1)); drag.ghost.setAttribute("width", String(Math.max(0, x2 - x1)));
      seekSoon(t); showTip(drag.svg, x, t, ctl.timeRange(a, b)); return;
    }
    if (drag.mode === "maybe-create") { drag.mode = "create"; drag.ghost = ghostFor(drag); }
    if (drag.mode === "create") {
      const [lo, hi] = drag.gap;
      const a = snapNear(snap(clamp(Math.min(drag.t0, t), lo, hi)), [lo], drag.ctol), b = snapNear(snap(clamp(Math.max(drag.t0, t), lo, hi)), [hi], drag.ctol);
      drag.a = a; drag.b = b;
      const x1 = xOf(drag.svg, a), x2 = xOf(drag.svg, b);
      drag.ghost.setAttribute("x", String(x1)); drag.ghost.setAttribute("width", String(Math.max(0, x2 - x1)));
      seekSoon(t); showTip(drag.svg, x, t, ctl.timeRange(a, b)); return;
    }
    if (drag.mode === "resize") {
      const list = items(drag.track), i = drag.index, o = drag.orig, minLen = 2 / fps();
      const prevEnd = list.reduce((m, iv, j) => (j !== i && iv.end <= o.start + 1e-9 ? Math.max(m, iv.end) : m), 0);
      const nextStart = list.reduce((m, iv, j) => (j !== i && iv.start >= o.end - 1e-9 ? Math.min(m, iv.start) : m), duration());
      let v = snap(t);
      if (drag.peer) {                                             // shared boundary: both neighbours follow
        const q = drag.peer.orig;
        if (drag.edge === "start") { v = clamp(v, q.start + minLen, o.end - minLen); drag.cand = {...o, start: v}; drag.peerCand = {...q, end: v}; }
        else { v = clamp(v, o.start + minLen, q.end - minLen); drag.cand = {...o, end: v}; drag.peerCand = {...q, start: v}; }
        for (const [rect, iv] of [[drag.rect, drag.cand], [drag.peer.rect, drag.peerCand]]) {
          if (rect) { const x1 = xOf(drag.svg, iv.start), x2 = xOf(drag.svg, iv.end); rect.setAttribute("x", String(x1)); rect.setAttribute("width", String(Math.max(0, x2 - x1))); }
        }
        seekSoon(v); showTip(drag.svg, x, v); return;
      }
      if (drag.edge === "start") { v = snapNear(clamp(v, prevEnd, o.end - minLen), [prevEnd, 0].filter(seen), drag.tol); drag.cand = {...o, start: v}; }
      else { v = snapNear(clamp(v, o.start + minLen, nextStart), [nextStart, duration()].filter(seen), drag.tol); drag.cand = {...o, end: v}; }
      if (drag.rect) { const x1 = xOf(drag.svg, drag.cand.start), x2 = xOf(drag.svg, drag.cand.end); drag.rect.setAttribute("x", String(x1)); drag.rect.setAttribute("width", String(Math.max(0, x2 - x1))); }
      seekSoon(v); showTip(drag.svg, x, v);
    }
  }
  /** A read-only target-preview snapshot while a quality edge is held. The draft and
   * Undo history change only on pointerup; Escape restores the original preview. */
  function previewIntervals() {
    const d = drag;
    if (!d || d.track === "stage" || d.mode !== "resize" || !d.cand || d.context !== state.contextSerial
        || d.sig !== labelsSig() || !d.svg.isConnected || !ctl.canEdit()
        || d.brush !== paintChoice(d.track) || document.querySelector("dialog[open]")) return null;
    const intervals = state.intervals.map(iv => ({...iv}));
    intervals[d.index] = {...d.cand};
    if (d.peer && d.peerCand) intervals[d.peer.index] = {...d.peerCand};
    return {intervals};
  }
  function endDrag() {
    const d = drag; drag = null; followAfter = performance.now() + 800;   // a stroke that ends at the view's edge keeps the view
    window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp); window.removeEventListener("pointercancel", onCancel);
    if (raf) cancelAnimationFrame(raf);
    raf = 0; queuedSeek = null;
    d.svg.classList.remove("lab-dragging"); hideTip();
    if (d.ghost) d.ghost.remove();
    return d;
  }
  /** Esc while the pointer still holds a drag: nothing is marked or moved, and the brush and the selection stay. */
  function cancelDrag() {
    if (!drag) return;
    const d = endDrag();
    if (d.mode === "resize" && d.moved) ctl.renderPoints();   // the dragged edges had moved on screen only
    if (d.moved && ["paint", "create", "resize"].includes(d.mode)) feedback("Cancelled · nothing changed.", "info");
  }
  function onCancel(e) { if (drag && e.pointerId === drag.pointerId) cancelDrag(); }
  function onUp(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    const finalSeek = queuedSeek;
    const d = endDrag();
    // the labels changed under the drag (a key pressed meanwhile): its indices and times no longer hold, so it does nothing
    if (d.context !== state.contextSerial || d.sig !== labelsSig()) { ctl.renderPoints(); return; }
    if (["paint", "create", "resize"].includes(d.mode) && (!ctl.canEdit() || document.querySelector("dialog[open]") || d.brush !== paintChoice(d.track))) { ctl.renderPoints(); return; }
    if (finalSeek !== null) ctl.seek(finalSeek);
    if (d.mode === "paint") { if (d.b - d.a >= 1 / fps() - 1e-9 && canPaint(d.track)) commitPaint(d.track, d.a, d.b); else ctl.renderPoints(); return; }   // the brush may be down by now
    if (d.mode === "maybe-paint") {
      ctl.seek(d.t0);
      const i = items(d.track).findIndex((iv) => iv.start <= d.t0 && d.t0 < iv.end);
      selection = i >= 0 ? {track: d.track, start: items(d.track)[i].start, end: items(d.track)[i].end, clicked: labelsSig()} : null; refreshSelection(); return;
    }
    if (d.mode === "create") { if (d.a !== undefined && d.b - d.a >= 2 / fps() - 1e-9) commitCreate(d.track, d.a, d.b); else ctl.renderPoints(); return; }
    if (d.mode === "resize" && d.peer && d.moved && d.peerCand) { commitBoundary(d.track, d.index, d.cand, d.peer.index, d.peerCand); return; }
    if (d.mode === "resize") { if (d.moved && d.cand) commitResize(d.track, d.index, d.cand); else ctl.renderPoints(); return; }
    if (d.mode === "select" && !d.moved) {                                       // click a span: select it (an empty-track click deselects)
      const iv = items(d.track)[d.index];
      if (iv) { selection = {track: d.track, start: iv.start, end: iv.end, clicked: labelsSig()}; refreshSelection(); }
      return;
    }
    if ((d.mode === "scrub" || d.mode === "maybe-create") && !d.moved && selection) { selection = null; refreshSelection(); }
    // a real drag on a track that only scrubbed: say why nothing was marked (Shift-drag is the deliberate scrub)
    if ((d.mode === "scrub" || d.held) && d.moved && d.inRow && !d.shift && !d.isCurve) {
      const why = d.held ? {text: d.held} : blocker(d.track);
      if (why) { feedback(why.text, "info"); pulse(why.focus); }
    }
  }
  document.addEventListener("pointerdown", onDown);
  window.addEventListener("blur", () => { if (drag) cancelDrag(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden && drag) cancelDrag(); });
  document.addEventListener("dblclick", (e) => {
    const svg = e.target.closest && e.target.closest("#own-stage-timeline svg.timeline-svg");
    if (!svg || !canCreate("stage")) return;
    const gap = gapAround("stage", timeAt(svg, e.clientX)); if (!gap) return;
    e.preventDefault(); commitCreate("stage", gap[0], gap[1]);
  });
  document.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("[data-action='fill-gaps']"); if (b) { e.preventDefault(); fillGaps(); } });

  /* ---------------- zoom: one zoom for both own tracks (agreement.js draws state.trackZoom); zoomed, they scroll together ---------------- */
  const zoomBox = $("lab-zoom");
  let viewStart = 0, zoomTarget = null, zoomAnchor = null, zoomRaf = 0;   // viewStart: the time at the tracks' left edge
  let pinch = null, followAfter = 0;   // {raw, shown}: the wheel gesture's unrounded zoom · no paging to the playhead before this time
  const maxZoom = () => clamp(duration() / 2, 1, 64);                      // down to about 2 s across the track
  const currentZoom = () => (zoomTarget === null ? state.trackZoom : zoomTarget);
  const zoomHosts = () => hosts.filter((h) => h.offsetParent !== null && h.querySelector("svg.timeline-svg:not(.reward-curve-svg)"));
  /** px per second on a host's drawn track (one viewBox unit is host width / 800 px at every zoom). */
  const pxPerSecond = (host) => { const svg = host.querySelector("svg.timeline-svg"); return svg.getBoundingClientRect().width / viewW(svg) * geo(svg).plot / (duration() || 1); };
  function applyView(except = null) {
    if (state.trackZoom <= 1) { viewStart = 0; return; }
    for (const host of zoomHosts()) {
      if (host === except) continue;
      const x = viewStart * pxPerSecond(host);
      if (Math.abs(host.scrollLeft - x) > 1) host.scrollLeft = x;
    }
  }
  /** Zoomed, a track hides the time labels its view's edges would cut (the labels repeat every quarter of the view). */
  function edgeTicks(host) {
    const svg = host.querySelector("svg.timeline-svg"), ticks = svg ? svg.querySelectorAll("text.timeline-tick") : []; if (!ticks.length) return;
    const unit = svg.getBoundingClientRect().width / viewW(svg), lo = host.scrollLeft - 0.5, hi = host.scrollLeft + host.clientWidth + 0.5;
    const em = parseFloat(getComputedStyle(ticks[0]).fontSize) || 9;
    for (const t of ticks) {
      const w = t.textContent.length * em * 0.56, x = Number(t.getAttribute("x")), a = t.getAttribute("text-anchor");
      const l = (a === "start" ? x : a === "end" ? x - w : x - w / 2) * unit, cut = state.trackZoom > 1 && (l < lo || l + w * unit > hi);
      if (t.style.visibility !== (cut ? "hidden" : "")) t.style.visibility = cut ? "hidden" : "";
    }
  }
  function zoomUi() {
    const z = currentZoom(), zoomed = state.trackZoom > 1;
    for (const host of hosts) if (host.classList.contains("lab-zoomed") !== zoomed) host.classList.toggle("lab-zoomed", zoomed);
    if (!zoomBox) return;
    const level = `${z < 10 ? Math.round(z * 10) / 10 : Math.round(z)}×`, node = zoomBox.querySelector(".lab-zoom-level");
    if (node && node.textContent !== level) I18N.text(node, level);
    let refocus = false;   // a key press on Fit / − / + that disables it keeps the focus in the group
    for (const b of zoomBox.querySelectorAll("button[data-zoom]")) {
      const off = !state.episode || (b.dataset.zoom === "in" ? z >= maxZoom() - 1e-6 : z <= 1);
      if (b.disabled !== off) { refocus = refocus || (off && document.activeElement === b); b.disabled = off; }
    }
    const next = refocus && zoomBox.querySelector("button[data-zoom]:not(:disabled)");
    if (next) next.focus();
  }
  /** Zoom both tracks to z, keeping one moment where it is on screen: the time under clientX on `host`, else the playhead
   *  (or the middle of the view when the playhead is out of view). Several wheel steps in one frame draw once. */
  function zoomTo(z, clientX = null, host = zoomHosts()[0]) {
    if (!host || !state.episode || drag) return;
    const svg = host.querySelector("svg.timeline-svg"), unit = svg.getBoundingClientRect().width / viewW(svg), d = duration(), drawn = state.trackZoom;
    let time, x;
    if (clientX !== null) { time = timeAt(svg, clientX); x = clientX - host.getBoundingClientRect().left; }
    else {
      const start = drawn > 1 ? viewStart : 0, span = d / drawn, t = ctl.currentTime();
      time = t >= start && t <= start + span ? t : start + span / 2; x = (8 + 784 * drawn * (time - start) / d) * unit;
    }
    const target = clamp(z, 1, maxZoom());
    zoomTarget = target < 1.05 ? 1 : target; zoomAnchor = {time, x, unit};
    if (!zoomRaf) zoomRaf = requestAnimationFrame(applyZoom);
  }
  function applyZoom() {
    zoomRaf = 0; if (zoomTarget === null) return;
    const z = zoomTarget, {time, x, unit} = zoomAnchor; zoomTarget = zoomAnchor = null;
    viewStart = z > 1 ? clamp(time - (x - 8 * unit) * duration() / (784 * z * unit), 0, duration()) : 0;
    if (z !== state.trackZoom) { state.trackZoom = z; zoomUi(); ctl.renderTracks(); }
    applyView(); zoomUi();
  }
  for (const host of hosts) {
    host.addEventListener("scroll", () => {   // a sideways scroll of one track moves the other with it
      if (state.trackZoom <= 1 || !zoomHosts().includes(host)) return;
      viewStart = host.scrollLeft / pxPerSecond(host); applyView(host); edgeTicks(host);
    }, {passive: true});
    host.addEventListener("wheel", (e) => {   // Ctrl / ⌘ + wheel (or a trackpad pinch): zoom around the pointer
      if (!(e.ctrlKey || e.metaKey) || !zoomHosts().includes(host)) return;
      e.preventDefault();
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 800 : e.deltaY;
      // a pinch arrives as many tiny steps that zoomTo would each round back to 1×: build on the unrounded zoom of this gesture
      const z = currentZoom(), raw = clamp((pinch && pinch.shown === z ? pinch.raw : z) * Math.exp(-clamp(dy, -60, 60) * 0.006), 1, maxZoom());
      zoomTo(raw, e.clientX, host); pinch = {raw, shown: currentZoom()};
    }, {passive: false});
  }
  if (zoomBox) zoomBox.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-zoom]"); if (!b || b.disabled) return;
    zoomTo(b.dataset.zoom === "in" ? currentZoom() * 2 : b.dataset.zoom === "out" ? currentZoom() / 2 : 1);
  });
  // while playing (and after a seek) a zoomed view pages along so the playhead stays in sight
  document.addEventListener("timeupdate", (e) => {
    if (e.target !== state.videos[0] || state.trackZoom <= 1 || drag || performance.now() < followAfter) return;
    const t = ctl.currentTime(), span = duration() / state.trackZoom;
    if (t >= viewStart && t <= viewStart + span * 0.94) return;
    viewStart = clamp(t - span * 0.06, 0, duration()); applyView();
  }, true);
  window.addEventListener("resize", () => { applyView(); hosts.forEach(edgeTicks); });

  /* ---------------- hover tooltip ---------------- */
  const tip = $("lab-track-tip");
  function showTip(svg, clientX, t, extra) {
    if (!tip) return;
    const b = svg.getBoundingClientRect();
    I18N.text(tip, extra || `${t.toFixed(2)} s · f${Math.round(t * fps())}`);
    tip.hidden = false; tip.style.left = `${clamp(clientX, b.left + 40, b.right - 40)}px`; tip.style.top = `${Math.max(8, b.top - 6)}px`;
  }
  const hideTip = () => { if (tip) tip.hidden = true; };
  document.addEventListener("pointermove", (e) => {
    if (drag || !tip) return;
    const svg = e.target.closest && e.target.closest("#own-timeline svg.timeline-svg, #own-stage-timeline svg.timeline-svg");
    if (!svg) { if (!tip.hidden) hideTip(); return; }
    showTip(svg, e.clientX, timeAt(svg, e.clientX));
  }, {passive: true});
  document.addEventListener("pointerleave", hideTip, {passive: true});

  /* ---------------- keyboard ---------------- */
  function pick(n, subtask) {
    if (!ctl.canEdit()) return;
    const qb = $("quality-buttons"), buttons = qb ? [...qb.querySelectorAll("button")] : [];
    if (!subtask && qb && !qb.hidden && buttons.length) {
      const b = buttons[n - 1]; if (!b || b.disabled) return;
      b.click(); feedback(`${I18N.source(b.querySelector(".quality-name"))} picked · drag on the track or press S.`); return;
    }
    const select = ctl.independentStages() ? $("independent-stage-select") : (!$("stage-field").hidden ? $("stage-select") : null);
    if (!select || select.disabled) return;
    const option = select.options[n]; if (!option) return;                         // options[0] is the placeholder
    select.value = option.value; select.dispatchEvent(new Event("change", {bubbles: true}));
    // with a quality brush picked (or a span pending) S starts a quality span; Shift+S is the subtask's (startEnd)
    feedback(`${I18N.source(option)} picked · drag on the subtask track or press ${state.selectedLabel || state.pendingInterval ? "Shift+S" : "S"}.`);
  }
  function startEnd(kind, forceStage) {
    if (!ctl.canEdit()) { feedback(signedIn() ? "Reopen the submitted episode first." : "Enter a name first.", "info"); return; }
    if (ctl.isCurve()) {
      if (kind !== "start") return;
      ctl.pauseAll(); state.curveTimePinned = true; $("curve-time").value = String(ctl.currentTime()); $("curve-value").focus();
      feedback(`Time ${ctl.currentTime().toFixed(2)} s · type a reward 0–${ctl.scoreMax()}, then Add reward anchor.`); return;
    }
    if (state.editingInterval !== null || state.editingStageInterval !== null) { feedback("Finish or cancel the current edit first.", "info"); return; }
    const t = ctl.currentTime();
    const wantStage = forceStage || ctl.isSubtaskOnly() || (kind === "end" ? Boolean(state.pendingStageInterval) && !state.pendingInterval
      : Boolean(state.pendingStageInterval) || (!state.selectedLabel && !state.pendingInterval && Number.isInteger(state.selectedSemanticStage)));
    if (wantStage) {
      if (!ctl.independentStages()) return;
      if (kind === "end" && !state.pendingStageInterval) { feedback("No pending subtask. Press S first.", "info"); return; }
      $("stage-boundary-time").value = String(t); state.stageTimePinned = true; addStageMerged(); return;
    }
    if (kind === "end" && !state.pendingInterval) { feedback("No pending span. Pick a label, then press S.", "info"); return; }
    $("interval-time").value = String(t); state.intervalTimePinned = true; ctl.addPoint();
  }
  const toggleShortcuts = () => { const d = $("lab-shortcuts"); if (!d) return; if (d.open) d.close(); else d.showModal(); };
  document.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("[data-open='lab-shortcuts']"); if (b) { e.preventDefault(); toggleShortcuts(); } });
  const guideShown = () => { const g = $("lab-guide"), r = g && g.open && g.getBoundingClientRect(); return Boolean(r && r.bottom > 64 && r.top < window.innerHeight); };
  function cancelInteraction() {
    if (drag) cancelDrag();
    if (ctl.canEdit()) {
      for (const [active, id] of [[state.pendingStageInterval, "cancel-pending-stage"], [state.pendingInterval, "interval-cancel-pending"],
        [state.editingStageInterval !== null, "cancel-stage-edit"], [state.editingInterval !== null, "interval-cancel-edit"]]) {
        if (active) $(id).click();
      }
      state.selectedLabel = null; state.selectedStage = null; state.selectedSemanticStage = null;
      $("independent-stage-select").value = "";
      ctl.renderSelectedLabel(); ctl.updateControls();
    }
    selection = null; refreshSelection();
  }
  window.addEventListener("keydown", (e) => {
    if (e.defaultPrevented) return;
    const target = e.target || document.body;
    // A held stroke or resize always cancels first, including with an enlarged camera or a focused text field.
    if (drag && e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancelDrag(); return; }
    if (document.querySelector("dialog[open]")) { if (e.key === "?" || (e.key === "Escape" && $("lab-shortcuts") && $("lab-shortcuts").open)) { e.preventDefault(); e.stopPropagation(); toggleShortcuts(); } return; }
    if (e.key === "Escape" && state.episode) {
      e.preventDefault(); e.stopPropagation();
      if (state.pendingInterval || state.pendingStageInterval || state.editingInterval !== null || state.editingStageInterval !== null) cancelInteraction();
      else if (ctl.restoreEnlargedView()) { /* restore only; the next Escape can deselect */ }
      else if (guideShown()) $("lab-guide").open = false;
      else cancelInteraction();
      return;
    }
    if (!state.episode || /INPUT|TEXTAREA|SELECT/.test(target.tagName) || target.isContentEditable) return;
    if (target.tagName === "BUTTON" && (e.key === " " || e.key === "Enter")) return;
    const meta = e.metaKey || e.ctrlKey;
    if (meta && !e.altKey && (e.key === "z" || e.key === "Z")) { if (e.shiftKey) return; e.preventDefault(); const u = $("undo-button"); if (u && !u.disabled) u.click(); return; }
    if (meta || e.altKey) return;
    switch (e.key) {
      case "ArrowLeft": case "ArrowRight":
        if (!e.shiftKey) return;                                                     // one frame: the lab's own handler
        e.preventDefault(); e.stopPropagation(); ctl.pauseAll(); ctl.seek(ctl.currentTime() + (e.key === "ArrowLeft" ? -1 : 1)); return;
      case "Home": e.preventDefault(); ctl.pauseAll(); ctl.seek(0); return;
      case "End": e.preventDefault(); ctl.pauseAll(); ctl.seek(duration()); return;
      case "s": case "S": e.preventDefault(); startEnd("start", e.shiftKey); return;
      case "e": case "E": e.preventDefault(); startEnd("end", e.shiftKey); return;
      case "p": case "P": if (ctl.isCurve()) { e.preventDefault(); e.stopPropagation(); startEnd("start", false); } return;
      case "Delete": case "Backspace": if (removeSelected()) e.preventDefault(); return;
      case "?": e.preventDefault(); toggleShortcuts(); return;
      default: break;
    }
    // 1–9 (Shift = subtask n, Shift+0 = S10) on the number row (any layout) or the numpad with Num Lock on; 0 alone is the
    // Nominal brush (grana-brushes.js)
    const digit = /^(Digit|Numpad)(\d)$/.exec(e.code || "");
    if (digit && (digit[1] === "Digit" || e.key === digit[2]) && (digit[2] !== "0" || e.shiftKey)) { e.preventDefault(); pick(Number(digit[2]) || 10, e.shiftKey); }
  }, true);
  // buttons, selects, checkboxes and seek bars used with the mouse give the focus back so Space / S / E / ← → keep working
  // (keyboard focus is kept)
  let lastPointer = 0, seekPress = null;
  document.addEventListener("pointerdown", (e) => { lastPointer = Date.now(); seekPress = e.target.type === "range" && e.target.closest(".playback-bar, #lab-transport") ? e.target : null; }, {passive: true, capture: true});
  const releaseSeek = () => { const s = seekPress; seekPress = null; if (s) setTimeout(() => { if (document.activeElement === s) s.blur(); }); };
  window.addEventListener("pointerup", releaseSeek, true); window.addEventListener("pointercancel", releaseSeek, true);
  document.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("button"); if (b && e.detail > 0 && b.closest("#annotation-workspace, #lab-transport") && !b.closest("dialog")) b.blur(); });
  document.addEventListener("change", (e) => { if ((e.target.tagName === "SELECT" || e.target.type === "checkbox") && e.target.closest("#annotation-workspace") && Date.now() - lastPointer < 800) e.target.blur(); });

  /* ---------------- sticky transport ---------------- */
  (function transport() {
    const bar = $("lab-transport"); if (!bar) return;
    const play = $("lab-transport-play"), seekBar = $("lab-transport-seek"), time = $("lab-transport-time"), video = document.querySelector(".video-card"), column = document.querySelector(".observation-column");
    let videoVisible = true;
    const put = (node, key, value) => { if (key === "textContent") I18N.text(node, value); else if (node[key] !== value) node[key] = value; };   // idle ticks write nothing
    const update = () => {
      const ws = $("annotation-workspace"), show = Boolean(!videoVisible && ws && !ws.hidden && state.episode);
      put(bar, "hidden", !show); if (!show) return;
      // centred on the tracks' column so it never covers Save / Submit in the side panel; narrow screens keep the CSS width
      const col = column && column.getBoundingClientRect(), wide = col && col.width > 0;
      const left = wide ? `${Math.round(col.left + col.width / 2)}px` : "", width = wide ? `${Math.round(Math.min(720, col.width - 24))}px` : "";
      if (bar.style.left !== left) bar.style.left = left;
      if (bar.style.width !== width) bar.style.width = width;
      const t = ctl.currentTime(), d = duration(), label = `${t.toFixed(2)} s`;
      put(seekBar, "max", String(Math.round(d * fps()))); put(seekBar, "value", String(ctl.frameAt(t)));   // the frame on screen, like #video-seek
      if (seekBar.getAttribute("aria-valuetext") !== label) I18N.attr(seekBar, "aria-valuetext", label);
      put(time, "textContent", `${t.toFixed(2)} / ${d.toFixed(2)} s`);
      put(play, "textContent", state.playbackRequested ? "Pause" : "Play");   // also while a camera holds playback (agreement.js)
    };
    $("lab-transport-prev").addEventListener("click", () => { ctl.pauseAll(); ctl.seek(ctl.currentTime() - 1 / fps()); });
    $("lab-transport-next").addEventListener("click", () => { ctl.pauseAll(); ctl.seek(ctl.currentTime() + 1 / fps()); });
    play.addEventListener("click", () => ctl.playPause());
    seekBar.addEventListener("input", (e) => ctl.seek(Number(e.target.value) / fps()));
    $("lab-transport-top").addEventListener("click", () => { if (video) video.scrollIntoView({behavior: "smooth", block: "center"}); });
    if (video && "IntersectionObserver" in window) new IntersectionObserver((entries) => { videoVisible = entries.some((en) => en.isIntersecting); update(); }, {threshold: 0.05}).observe(video);
    setInterval(update, 120);
  })();

  /* ---------------- side panel: the review tick, Save and Submit (.lab-commit) stay on screen ---------------- */
  // Above 760 px the panel is sticky and scrolls inside. Until it sticks (the page near its top) it starts lower down, so its
  // height is capped to the window below its top; scroll-padding keeps a control reached with Tab clear of the footer. When
  // that room holds the footer but not the panel's padding too (a short window), or not even the footer (a short window with
  // the guide open), the footer floats at the same spot in the panel's column while any of the panel is on screen.
  (function sidePanel() {
    const card = document.querySelector(".annotation-card"), foot = card && card.querySelector(".lab-commit"); if (!foot) return;
    const wide = window.matchMedia("(min-width: 761px)");
    let raf = 0;
    const fit = () => {
      raf = 0;
      const on = wide.matches && card.offsetParent !== null, box = card.getBoundingClientRect(), viewH = document.documentElement.clientHeight;
      const room = on ? Math.floor(viewH - Math.max(box.top, parseFloat(getComputedStyle(card).top) || 0) - 16) : 0, need = foot.offsetHeight;
      const fits = on && room >= need, float = on && box.top < viewH && room < need + 48;
      const height = fits ? `${room}px` : "", pad = on ? `${need + 8}px` : "";
      const left = float ? `${Math.round(box.left)}px` : "", width = float ? `${Math.round(box.width)}px` : "";
      if (card.style.maxHeight !== height) card.style.maxHeight = height;
      if (card.style.scrollPaddingBottom !== pad) card.style.scrollPaddingBottom = pad;
      if (foot.classList.contains("lab-float") !== float) foot.classList.toggle("lab-float", float);
      if (foot.style.left !== left) foot.style.left = left;
      if (foot.style.width !== width) foot.style.width = width;
      // 760 px and below the page scrolls and the footer floats over its bottom (raised above the transport bar when that
      // shows): the page's scroll-padding keeps a control reached with Tab clear of it
      const page = !wide.matches && card.offsetParent !== null ? `${need + (parseFloat(getComputedStyle(foot).bottom) || 0) + 8}px` : "";
      if (document.documentElement.style.scrollPaddingBottom !== page) document.documentElement.style.scrollPaddingBottom = page;
    };
    const soon = () => { if (!raf) raf = requestAnimationFrame(fit); };
    window.addEventListener("scroll", soon, {passive: true}); window.addEventListener("resize", soon);
    setInterval(fit, 250);   // the notice, the guide or the videos above the panel move it without a scroll
  })();

  /* ---------------- why Submit is disabled: one live line under the button (+ its title) and the Fill gaps button ---------------- */
  function submitReason() {
    if (!state.episode || !state.scheme) return "";
    if (!signedIn()) return "Enter a name to submit.";
    if (state.sessionConflict) return "Restore your editing name before submitting.";
    if (ctl.isSubmitted() || state.legacy) return "";
    if (ctl.isCurve()) {
      const pts = state.points || [], d = duration();
      if (!pts.length) return "Add anchors at 0 s and the clip end first.";
      const has0 = pts[0].time === 0, hasEnd = Math.abs(pts[pts.length - 1].time - d) <= 1e-6;
      if (!has0 && !hasEnd) return "Anchors at 0 s and the clip end are required.";
      if (!has0) return "An anchor at 0 s is required.";
      if (!hasEnd) return "An anchor at the clip end is required.";
      return "";
    }
    if (state.pendingStageInterval) return "End or cancel the pending subtask (E).";
    if (state.pendingInterval) return "End or cancel the pending span (E).";
    if (ctl.independentStages()) {
      const cov = ctl.stageCoverage();
      if (!cov.complete) return state.stageIntervals.length ? `Cover ${cov.missing.toFixed(1)} s more of the subtask track.` : "Cover the clip with subtask intervals first.";
      if (ctl.unresolvedOutcomes()) return "Choose Completed or Failed for every subtask marked Needs review before submitting.";
    }
    if (ctl.defaultNormal() && !state.reviewConfirmed) return "Tick the review box above.";
    if (!ctl.defaultNormal() && !ctl.isCurve() && !state.intervals.length) return "Mark at least one interval first.";
    return "";
  }
  setInterval(() => {
    const line = $("lab-submit-reason"), submit = $("submit-button"), fill = $("lab-fill-gaps");
    if (line && submit) {
      const reason = submit.hidden ? "" : submitReason();
      const showFill = Boolean(reason && /subtask track|subtask intervals/.test(reason) && ctl.canEdit());
      if (line.dataset.reason !== reason || line.dataset.fill !== String(showFill)) {
        line.dataset.reason = reason; line.dataset.fill = String(showFill); line.replaceChildren();
        if (reason) { line.append(I18N.node(reason)); if (showFill) { const b = document.createElement("button"); b.type = "button"; b.className = "text-button"; b.dataset.action = "fill-gaps"; I18N.text(b, state.stageIntervals.length ? "Fill gaps" : "Fill with the picked subtask"); line.append(I18N.node(" "), b); } }
        line.hidden = !reason; I18N.attr(submit, "title", reason);
      }
    }
    const hideFill = !(ctl.canEdit() && ctl.independentStages() && !ctl.stageCoverage().complete && (state.stageIntervals.length || Number.isInteger(state.selectedSemanticStage)));
    if (fill && fill.hidden !== hideFill) fill.hidden = hideFill;
  }, 250);

  /* ---------------- protocol-dependent chrome + autosave ---------------- */
  const TRACK_HINT = {
    default: "Pick a brush above a track, then drag to paint · Esc cancels or deselects · click a span, then Delete removes it · without a brush: drag an edge to adjust",
    aq_only: "Drag on the track to mark · drag an edge to adjust · S / E at the playhead",
    stage_score: "Drag on the track to mark · drag an edge to adjust · S / E at the playhead",
    reward_curve: "Double-click the curve to add a point · click a point to edit · P = time at playhead",
  };
  let lastScheme = null;
  setInterval(() => {
    const id = state.scheme && state.scheme.id;
    if (id === lastScheme) return;
    lastScheme = id; selection = null;
    if (id) document.documentElement.dataset.scheme = id;
    const th = $("lab-track-hint-text");
    if (th) I18N.text(th, TRACK_HINT[id] || TRACK_HINT.default);
  }, 250);
  setInterval(async () => {
    if (!state.dirty || state.busy || state.saveConflict || state.sessionConflict || state.pendingInterval || state.pendingStageInterval || drag || !ctl.canEdit() || ctl.isSubmitted() || state.legacy) return;
    if (state.editingInterval !== null || state.editingStageInterval !== null || document.querySelector("dialog[open]")) return;
    if (!(state.points.length || state.intervals.length || state.pendingInterval || state.stageIntervals.length || state.pendingStageInterval || state.record)) return;
    const history = state.history.slice();
    const ok = await ctl.withBusy(async () => { await ctl.saveRecord("draft"); });
    if (!ok) return;
    state.history = history; ctl.updateControls();
    const now = new Date();
    feedback(`Autosaved draft ${now.getHours()}:${String(now.getMinutes()).padStart(2, "0")} · not counted until submitted.`);
  }, 30000);

  window.GRANA_LAB_INTERACT = {snap, timeAt, xOf, gapAround, gaps, commitCreate, commitResize, removeSelected, fillGaps, startEnd, pick, canPaint, commitPaint, stageColor, toggleShortcuts, submitReason, blocker, setOutcome, outcomeList, mergeTouchingStages, cancelInteraction, cancelDrag, previewIntervals, hasActiveDrag: () => Boolean(drag),
    getSelection: () => { dropStaleSelection(); return selection ? {track: selection.track, start: selection.start, end: selection.end} : null; },
    select: (track, start, end) => { dropStaleSelection(); selection = {track, start, end}; refreshSelection(); }};
})();
