/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
"use strict";

(() => {
  const I18N = window.GRANA_I18N || {text: (n, s) => { n.textContent = s; }, attr: (n, k, s) => n.setAttribute(k, s), node: s => document.createTextNode(s), source: n => n.textContent, t: s => s};
  const $ = (id) => document.getElementById(id);
  const state = {
    schemes: [], episodes: [], session: {}, access: {requires_reviewer_key: true, can_manage: false}, episode: null, scheme: null,
    record: null, points: [], selectedLabel: null, selectedStage: null,
    activeSeconds: 0, dirty: false, busy: false, saveConflict: false, sessionConflict: false, history: [], videos: [],
    panel: "annotate", lastInteraction: Date.now(), lastTick: Date.now(),
    reportEpisode: null, playbackRequested: false,
    interpolation: "pchip", curve: null, curveRequest: 0, curveTimer: null,
    curveTimePinned: false, editingTime: null,
    intervals: [], pendingInterval: null, editingInterval: null, intervalTimePinned: false,
    legacy: false, migratedFromRevision: null,
    annotationFormat: null, reviewConfirmed: false,
    stageIntervals: [], pendingStageInterval: null, editingStageInterval: null,
    selectedSemanticStage: null, stageTimePinned: false,
    trackZoom: 1, contextSerial: 0,   // zoom of the two own tracks (lab-interact.js) · counts loadContext calls
  };
  const SVG = "http://www.w3.org/2000/svg";
  let plotSerial = 0;
  const fallbackColors = {normal: "#199e70", suboptimal: "#e3a72f", failure: "#ec5b52", high: "#199e70", mid: "#e3a72f", low: "#ec5b52"};
  // One subtask palette for brushes, tracks, the Subtasks list and the target plot; adjacent pairs validated on #171614.
  // It cycles after S10, so the S# label always names the subtask.
  const STAGE_COLORS = ["#3987e5", "#199e70", "#9085e9", "#5f9c00", "#d55181", "#8749a4", "#6d6e0a", "#07719f", "#a83b6c", "#0b9db3"];
  function stageColor(index) { return STAGE_COLORS[Math.max(0, Number(index) || 0) % STAGE_COLORS.length]; }
  /** Dark or light text on a filled colour, whichever reads better (the darker subtask colours take light text). */
  function inkOn(color) {
    const rgb = /^#[0-9a-f]{6}$/i.test(color || "") ? [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16) / 255) : null;
    if (!rgb) return "#0f0e0d";
    const [r, g, b] = rgb.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.18 ? "#ebe7df" : "#0f0e0d";
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) I18N.text(node, String(text));
    return node;
  }
  function svgEl(tag, attrs, text) {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => I18N.attr(node, key, String(value)));
    if (text !== undefined) I18N.text(node, String(text));
    return node;
  }
  let videoProblems = [];   // {name, detail} per camera of the open episode that failed to load, by tile (videoFailed)
  /** "top, left: why" once per reason, in tile order; detail null = the browser cannot decode what the server sent. */
  function videoProblemText() {
    const reasons = new Map();
    videoProblems.forEach((p) => { if (p) reasons.set(p.detail, [...(reasons.get(p.detail) || []), p.name]); });
    return [...reasons].map(([detail, names]) => (detail === null ? `Cannot play ${names.join(", ")}. Check that this browser supports its video codec.` : `${names.join(", ")}: ${detail}`));
  }
  function showNotice(message, kind = "success") {
    const raised = kind === "error", problems = videoProblemText();
    // A camera that failed to load stays in the notice until the episode changes; later notices (sign-in, saves) join it.
    if (problems.length && !raised) { message = [message, ...problems].filter(Boolean).join(" · "); kind = "error"; }
    I18N.text($("notice"), message);
    $("notice").className = `notice ${kind}`;
    $("notice").hidden = !message;
    if (raised && state.panel === "annotate") annotationFeedback(message);
  }
  async function api(path, options = {}) {
    // Every call goes to window.GRANA_LAB_API.handle (grana-api.js), which reads and writes the grana label
    // server's label files. Errors propagate as Error(message) and surface through showNotice.
    if (!window.GRANA_LAB_API) throw new Error("grana-api.js did not load; grana label cannot work without it.");
    const method = (options.method || "GET").toUpperCase();
    try { return await window.GRANA_LAB_API.handle(method, path, options.body); }
    catch (error) { if (error.sessionConflict) { state.sessionConflict = true; updateControls(); } throw error; }
  }
  window.__agreementSeek = (time) => seek(time);
  window.__agreementRenderReport = (report, episode) => renderReport(report, episode);
  window.__agreementState = state;
  // lab-interact.js / lab-stats.js drive the editor through the lab's own functions (function declarations are hoisted;
  // consts declared later in the closure are read through getters to avoid the temporal dead zone).
  window.__agreementCtl = {
    state, seek, frameAt, pauseAll, playPause, currentTime, markInteraction, remember, changed, renderPoints, renderProtocol, restoreEnlargedView,
    renderSelectedLabel, renderTracks, updateControls, annotationFeedback, showNotice, validateClosedInterval, validateStageInterval,
    boundaryTime, addPoint, addStageInterval, saveRecord, saveIfDirty, loadContext, withBusy, refreshBootstrap,
    canEdit, isSubmitted, isCurve, isAQ, isSubtaskOnly, independentStages, defaultNormal, requiresExceptionStage, stageCoverage, unresolvedOutcomes,
    schemeLabel, labelColor, stageName, stageColor, inkOn, scoreMax, trackIntervals, timeRange, layoutStageLabels,
    get annotatorColors() { return annotatorColors; },
  };
  function ownPath(episode = state.episode, scheme = state.scheme) {
    return `/api/annotations/${encodeURIComponent(episode.id)}/${encodeURIComponent(scheme.id)}`;
  }
  function isSubmitted() { return state.record?.status === "submitted"; }
  function canManage() { return Boolean(state.access.can_manage); }
  function protectedStudy() { return Boolean(state.access.requires_reviewer_key); }
  function isCurve(scheme = state.scheme) { return scheme?.kind === "continuous" || scheme?.id === "reward_curve"; }
  function isAQ(scheme = state.scheme) { return scheme?.id === "aq_only" || scheme?.id === "stage_aq" || scheme?.id === "subtask_only"; }
  function isSubtaskOnly(scheme = state.scheme) { return scheme?.id === "subtask_only"; }
  function independentStages() { return (state.scheme?.id === "stage_aq" || isSubtaskOnly()) && !state.legacy && state.annotationFormat === "stage-aq-independent-v1"; }
  function defaultNormal() { return isAQ() && !state.legacy && ["aq-default-normal-v1", "stage-aq-independent-v1"].includes(state.annotationFormat); }
  function skippedQuality() { return state.record?.machine_import?.action_quality_status === "default_normal_as_not_requested"; }
  function machinePending() { return Boolean(state.record?.machine_import && !skippedQuality() && !state.reviewConfirmed); }
  function defaultQualityName() { return skippedQuality() ? "Default Normal (AQ not requested)" : machinePending() ? "Unreviewed quality" : "Default Nominal"; }
  function defaultQualityColor() { return machinePending() ? "#4b4d53" : "#22503f"; }
  function unresolvedOutcomes() {
    if (!state.episode || !state.session.annotator_id || !independentStages()) return false;
    const outcomes = window.GRANA_LAB_API.outcomes;
    const list = window.GRANA_LAB_INTERACT?.outcomeList() || outcomes.get(state.episode.id, state.session.annotator_id, state.scheme.id);
    return state.stageIntervals.some(iv => outcomes.lookup(list, iv) === "unknown");
  }
  function renderMachineImport() {
    const imported = state.record?.machine_import, panel = $("machine-import-notice");
    panel.hidden = !imported;
    if (!imported) return;
    const notes = $("machine-import-notes"); notes.replaceChildren();
    for (const text of imported.review_notes || []) { const item = document.createElement("li"); item.textContent = text; notes.append(item); }
    I18N.text($("machine-import-status"), isSubmitted() ? "Human review submitted. Original machine notes remain below as provenance." : "Machine proposal · human review pending. Check every suggested label and the source video before submitting.");
  }
  function requiresExceptionStage() { return Boolean(state.scheme?.requires_stage && !independentStages()); }
  function preferredFormat(scheme = state.scheme) { return scheme?.id === "stage_aq" || isSubtaskOnly(scheme) ? "stage-aq-independent-v1" : isAQ(scheme) ? "aq-default-normal-v1" : "intervals-v1"; }
  function scoreMax(episode = state.episode) { return Math.max(1, episode?.stages?.length || 0); }
  function canEdit() { return Boolean(state.session.annotator_id && state.episode && state.scheme && !isSubmitted() && !state.busy && !state.legacy); }
  function cloneIntervals(intervals) { return (intervals || []).map((interval) => ({...interval})); }
  function trackIntervals(track, episode) {
    if (track.effective_intervals) return cloneIntervals(track.effective_intervals);
    if (["aq-default-normal-v1", "stage-aq-independent-v1"].includes(track.annotation_format)) {
      const output = []; let cursor = 0;
      (track.intervals || []).forEach((interval) => { if (interval.start > cursor) output.push({start: cursor, end: interval.start, label: "normal", stage: null, source: "default"}); output.push({...interval, source: "explicit"}); cursor = interval.end; });
      if (cursor < episode.duration_seconds) output.push({start: cursor, end: episode.duration_seconds, label: "normal", stage: null, source: "default"});
      return output;
    }
    if (track.annotation_format === "intervals-v1" || Array.isArray(track.intervals)) return cloneIntervals(track.intervals);
    return (track.points || []).map((point, index, points) => ({start: point.time, end: points[index + 1]?.time ?? episode.duration_seconds, label: point.label, stage: point.stage ?? null, legacy: true}));
  }
  function ownTrack() { return isCurve() || state.annotationFormat === "legacy-points" ? {points: state.points} : {annotation_format: state.annotationFormat || "intervals-v1", default_label: defaultNormal() ? "normal" : undefined, intervals: state.intervals, pending_interval: state.pendingInterval}; }
  function countCaption() { return isCurve() ? `${state.points.length} explicit anchors` : state.legacy ? `${state.annotationFormat === "legacy-points" ? state.points.length : state.intervals.length} legacy labels · read-only` : `${state.intervals.length ? `${state.intervals.length} ${defaultNormal() ? "span" : "interval"}${state.intervals.length === 1 ? "" : "s"}` : defaultNormal() ? "No spans yet" : "No intervals yet"}${state.pendingInterval ? " · 1 pending" : ""}${defaultNormal() ? ` · ${state.reviewConfirmed ? "review confirmed" : "review not confirmed"}` : ""}`; }
  function clock(seconds) {
    const rounded = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
  }
  function formatMetric(value, digits = 3) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—"; }
  function schemeLabel(scheme, id) { return scheme.labels.find((label) => label.id === id); }
  function labelColor(scheme, id) { return schemeLabel(scheme, id)?.color || fallbackColors[id] || "#75716a"; }
  function stageName(index, episode = state.episode) {
    const item = episode?.stages?.[index];
    return typeof item === "string" ? item : item?.name || item?.text || item?.description || `Subtask ${Number(index) + 1}`;
  }
  function timeRange(start, end) { return `${start.toFixed(2)}–${end.toFixed(2)} s`; }
  function coverageSegments() {
    const duration = state.episode?.duration_seconds || 0, points = state.points, segments = [];
    const gap = (start, end) => { if (end > start) segments.push({start, end, kind: "unlabelled"}); };
    if (defaultNormal()) return trackIntervals(ownTrack(), state.episode).map((interval) => ({start: interval.start, end: interval.end, point: interval, kind: interval.source === "default" ? "default" : "labelled"}));
    if (!isCurve() && state.annotationFormat !== "legacy-points") {
      let cursor = 0;
      state.intervals.forEach((interval) => { gap(cursor, interval.start); segments.push({start: interval.start, end: interval.end, point: interval, kind: "labelled"}); cursor = interval.end; });
      gap(cursor, duration); return segments;
    }
    if (!points.length) { gap(0, duration); return segments; }
    gap(0, points[0].time);
    points.forEach((point, index) => {
      const next = points[index + 1];
      if (isCurve() && !next) return;
      const end = next?.time ?? duration;
      if (end > point.time) segments.push({start: point.time, end, point, next,
        kind: isCurve() ? "curve" : "labelled", tail: !next});
    });
    if (isCurve()) gap(points[points.length - 1].time, duration);
    return segments;
  }
  function coverageLabel(segment) {
    if (segment.kind === "unlabelled") return "Unlabelled";
    if (segment.kind === "default") return defaultQualityName();
    if (segment.kind === "curve") return `${segment.point.value} → ${segment.next.value}${segment.next.step ? " · step jump" : " · interpolated"}`;
    const label = schemeLabel(state.scheme, segment.point.label)?.name || segment.point.label;
    return `${segment.point.stage != null ? `S${segment.point.stage + 1} · ` : ""}${label}${segment.tail ? " → end of clip" : ""}`;
  }
  function hatchPattern(svg) {
    const id = `unlabelled-${++plotSerial}`, defs = svgEl("defs");
    const pattern = svgEl("pattern", {id, width: 8, height: 8, patternUnits: "userSpaceOnUse"});
    pattern.append(svgEl("rect", {width: 8, height: 8, fill: "#1b1a17"}));
    pattern.append(svgEl("line", {x1: 0, y1: 8, x2: 8, y2: 0, stroke: "#3b3b4f", "stroke-width": 1}));
    defs.append(pattern); svg.append(defs);
    return `url(#${id})`;
  }
  function renderCoverage() {
    const summary = $("coverage-summary"), legend = $("coverage-legend"), list = $("coverage-interval-list"), mini = $("coverage-mini");
    summary.replaceChildren(); legend.replaceChildren(); list.replaceChildren(); mini.replaceChildren();
    if (!state.episode || !state.scheme) return;
    const segments = coverageSegments(), duration = state.episode.duration_seconds;
    const covered = segments.filter((segment) => segment.kind !== "unlabelled").reduce((total, segment) => total + segment.end - segment.start, 0);
    const missing = Math.max(0, duration - covered), percentage = duration ? 100 * covered / duration : 0;
    const defaultSeconds = segments.filter((segment) => segment.kind === "default").reduce((total, segment) => total + segment.end - segment.start, 0);
    const exceptionSeconds = defaultNormal() ? duration - defaultSeconds : 0;
    summary.dataset.coveredSeconds = String(covered); summary.dataset.unlabelledSeconds = String(missing);
    summary.dataset.defaultNormalSeconds = String(defaultSeconds); summary.dataset.exceptionSeconds = String(exceptionSeconds);
    if (defaultNormal()) summary.append(el("span", "coverage-chip coverage-default", `${machinePending() ? "Unreviewed" : "Nominal"} ${defaultSeconds.toFixed(1)} s`), el("span", "coverage-chip", `Suboptimal + Regressed ${exceptionSeconds.toFixed(1)} s`));
    else summary.append(el("span", "coverage-chip", `Labelled ${covered.toFixed(1)} / ${duration.toFixed(1)} s · ${percentage.toFixed(1)}%`), el("span", "coverage-chip coverage-unlabelled", `Unlabelled ${missing.toFixed(1)} s`));
    summary.append(el("span", "coverage-caption", `${countCaption()} · ${state.dirty ? "unsaved" : isSubmitted() ? "submitted" : state.record ? "saved" : "nothing saved"}`));
    const legendItem = (name, color, unlabelled = false) => {
      const item = el("span", "coverage-chip"), swatch = el("i", `coverage-swatch${unlabelled ? " coverage-unlabelled" : ""}`);
      if (color) swatch.style.background = color;
      item.append(swatch, I18N.node(name)); legend.append(item);
    };
    if (defaultNormal()) legendItem(defaultQualityName(), defaultQualityColor()); else legendItem("Unlabelled", null, true);
    if (isCurve()) legendItem("Interpolated between anchors", "#1d3a33");
    else state.scheme.labels.filter((label) => !defaultNormal() || label.id !== "normal").forEach((label) => legendItem(label.name, labelColor(state.scheme, label.id)));
    mini.append(el("p", "coverage-caption", defaultNormal() ? `${machinePending() ? "Unreviewed" : "Nominal"} ${defaultSeconds.toFixed(1)} s · Suboptimal + Regressed ${exceptionSeconds.toFixed(1)} s` : `Covered ${covered.toFixed(1)} s · unlabelled ${missing.toFixed(1)} s`));
    const strip = el("div", "coverage-strip");
    segments.forEach((segment) => {
      const button = el("button", `coverage-item${segment.kind === "unlabelled" ? " coverage-unlabelled" : segment.kind === "default" ? " coverage-default" : ""}`); button.type = "button";
      const color = segment.kind === "unlabelled" ? null : segment.kind === "default" ? defaultQualityColor() : segment.kind === "curve" ? "#30c5a5" : labelColor(state.scheme, segment.point.label);
      if (color) button.style.setProperty("--coverage-color", color);
      button.dataset.start = String(segment.start); button.dataset.end = String(segment.end); button.dataset.kind = segment.kind;
      button.append(el("span", "coverage-range mono", timeRange(segment.start, segment.end)), el("span", "coverage-label", coverageLabel(segment)));
      if (segment.tail) button.append(el("span", "coverage-caption", "Last label held to the end by protocol."));
      if (segment.point?.stage != null) button.title = stageName(segment.point.stage);
      button.addEventListener("click", () => { pauseAll(); seek(segment.start); }); list.append(button);
      const band = el("button", `coverage-band${segment.kind === "unlabelled" ? " coverage-unlabelled" : ""}`); band.type = "button";
      band.style.width = `${100 * (segment.end - segment.start) / duration}%`;
      if (color) band.style.backgroundColor = color;
      I18N.attr(band, "title", `${timeRange(segment.start, segment.end)} · ${coverageLabel(segment)}`);
      I18N.attr(band, "aria-label", band.title); band.addEventListener("click", () => { pauseAll(); seek(segment.start); }); strip.append(band);
    });
    mini.append(strip);
    if (isCurve() && state.points.length === 1) list.append(el("p", "coverage-caption", `${state.points[0].time.toFixed(2)} s has an anchor (Reward ${state.points[0].value}); add a second anchor to form a curve.`));
    const help = machinePending() ? "Grey regions have unreviewed quality, including machine-unknown time. Inspect them and mark exceptions before confirming the whole clip." : defaultNormal() ? `Dim green = default Nominal; only Suboptimal / Regressed spans are explicit. Deleting a span restores the default. ${independentStages() ? "Spans may cross subtask boundaries." : state.scheme.id === "stage_aq" ? "Default time has no subtask." : ""}` : isCurve() ? "Grey hatching = unlabelled; between anchors = interpolated. No extrapolation past the first or last anchor." : state.legacy ? (state.annotationFormat === "legacy-points" ? "Legacy record, read-only: each point holds until the next. Confirm the conversion to edit." : "Legacy record, read-only. Confirm the conversion to edit it under the current protocol.") : "Grey hatching = unlabelled; colour = closed intervals. Gaps stay unlabelled.";
    I18N.text($("timeline-help"), help);
    updateCurrentAnnotation();
  }
  function previewValueAt(time) {
    const curve = state.curve;
    if (!curve?.times?.length) return null;
    let index = -1;
    for (let i = 0; i < curve.times.length && curve.times[i] <= time; i += 1) index = i;
    if (index < 0 || typeof curve.values[index] !== "number") return null;
    const value = curve.values[index], next = index + 1;
    if (curve.times[index] === time || next === curve.times.length) return value;
    if (typeof curve.values[next] !== "number") return null;
    const span = curve.times[next] - curve.times[index];
    return span > 0 ? value + (curve.values[next] - value) * (time - curve.times[index]) / span : value;
  }
  function updateCurrentAnnotation(time = currentTime()) {
    const node = $("current-annotation"); node.replaceChildren();
    let label = "Unlabelled", detail = "No label at this time yet.", kind = "unlabelled", color = null;
    const points = state.points;
    if (state.scheme && isCurve()) {
      const anchor = points.find((point) => Math.abs(point.time - time) <= 1e-6);
      if (anchor) { label = `Reward = ${anchor.value}`; detail = `${anchor.time.toFixed(2)} s · explicit anchor${anchor.step ? " · Step" : ""}`; kind = "anchor"; }
      else if (points.length > 1 && time > points[0].time && time < points[points.length - 1].time) {
        const value = previewValueAt(time);
        label = value === null ? "Covered · awaiting fit" : `Reward ≈ ${value.toFixed(3)}`;
        detail = value === null ? "Waiting for the fit; click Fit curve." : "Interpolated preview, not an anchor."; kind = "curve";
      } else if (points.length) detail = "No anchor here; not extrapolated.";
      if (kind !== "unlabelled") color = "#30c5a5";
    } else if (state.scheme && state.annotationFormat !== "legacy-points") {
      const interval = state.intervals.find((item) => item.start <= time && (time < item.end || time === state.episode.duration_seconds && time === item.end));
      if (defaultNormal()) { label = defaultQualityName(); kind = "default"; color = machinePending() ? "#9599a3" : "#5fae8c"; detail = machinePending() ? "Machine quality is unreviewed; inspect this region before confirming Nominal." : `Default Nominal. ${independentStages() ? "" : state.scheme.id === "stage_aq" ? "Subtask unknown." : ""}`; }
      if (interval) { label = coverageLabel({kind: "labelled", point: interval}); color = labelColor(state.scheme, interval.label); kind = "labelled"; detail = `${timeRange(interval.start, interval.end)} · closed interval${interval.stage != null ? ` · ${stageName(interval.stage)}` : ""}`; }
      if (state.pendingInterval && time >= state.pendingInterval.start && !interval) { if (!defaultNormal()) { label = "Pending · not yet covered"; kind = "pending"; } detail += ` ${state.pendingInterval.start.toFixed(2)} s: started ${schemeLabel(state.scheme, state.pendingInterval.label)?.name || state.pendingInterval.label} but not yet ended; currently ${defaultNormal() ? "the default background is unchanged" : "not covered"}.`; }
    } else if (state.scheme) {
      let active = -1;
      points.forEach((point, index) => { if (point.time <= time) active = index; });
      if (active >= 0) {
        const point = points[active], end = points[active + 1]?.time ?? state.episode.duration_seconds;
        label = coverageLabel({kind: "labelled", point}); color = labelColor(state.scheme, point.label); kind = "labelled";
        detail = `${timeRange(point.time, end)}${active === points.length - 1 ? " · last label held to the end" : " · active interval"}${point.stage != null ? ` · ${stageName(point.stage)}` : ""}`;
      }
    }
    node.dataset.kind = kind;
    node.dataset.source = kind === "default" ? "default" : ["labelled", "anchor"].includes(kind) ? "explicit" : kind;
    node.style.setProperty("--coverage-color", color || "#75716a");
    node.append(el("span", "coverage-caption", `${time.toFixed(2)} s · current label`), el("strong", "current-annotation-label", label), el("span", "current-annotation-detail", detail));
  }
  function annotationFeedback(message) {
    I18N.text($("annotation-feedback"), message); $("annotation-feedback").hidden = !message;
  }
  function clearLoadedRecord() {
    pauseAll(); state.record = null; state.points = []; state.activeSeconds = 0;
    state.dirty = false; state.saveConflict = false; state.sessionConflict = false; state.history = []; state.selectedLabel = null; state.selectedStage = null;
    state.interpolation = "pchip"; state.curve = null; state.editingTime = null; state.curveTimePinned = false;
    state.intervals = []; state.pendingInterval = null; state.editingInterval = null; state.intervalTimePinned = false; state.legacy = false; state.migratedFromRevision = null;
    state.annotationFormat = preferredFormat(); state.reviewConfirmed = false;
    state.stageIntervals = []; state.pendingStageInterval = null; state.editingStageInterval = null; state.selectedSemanticStage = null; state.stageTimePinned = false;
    invalidateCurve();
    renderPoints(); renderSelectedLabel();
  }
  /** The playhead. A seek parks the cameras 1 µs past a frame's start (mediaTime); that reads back as the start itself,
   *  k / fps or its 6-decimal time, whichever is later, so S / E and the subtask under the playhead match the picture. A seek
   *  to the clip end, cut to the µs below it, reads back as the end. */
  function currentTime() {
    const time = state.videos[0]?.currentTime || 0, fps = state.episode?.fps || 30, frame = Math.round(time * fps), edge = Math.round(frame / fps * 1e6);
    const past = Math.round(time * 1e6) - edge, end = state.episode?.duration_seconds || 0;
    if (end && Math.abs(time - end) <= 2e-6) return end;
    return past >= 0 && past <= 2 ? Math.max(edge / 1e6, frame / fps) : time;
  }
  function markInteraction() { state.lastInteraction = Date.now(); }
  let holding = false;   // playback is requested but waits for a camera (holdPlayback / resumePlayback)
  function pauseAll() {
    state.playbackRequested = false; holding = false;
    state.videos.forEach((video) => video.pause());
    I18N.text($("play-button"), "Play");
  }
  function seek(time) {
    // the playhead lands on the start of the frame `time` falls in (the picture a seek there shows), so a click on a track,
    // ← / → and S / E all name that frame
    const duration = state.episode?.duration_seconds || 0, fps = state.episode?.fps || 30;
    const value = Math.min(frameAt(Math.max(0, Math.min(Number(time) || 0, duration))) / fps, duration), at = mediaTime(value);
    holdPlayback();   // while playing, the cameras restart together once each has landed
    state.videos.forEach((video) => {
      video.dataset.pendingSeek = String(at);
      if (Number.isFinite(video.duration)) video.currentTime = Math.min(at, video.duration);
    });
    markInteraction();
    state.curveTimePinned = false;
    state.intervalTimePinned = false;
    state.stageTimePinned = false;
    updatePlayback(value);
  }
  // Browsers keep a seek time in whole µs, cut rather than rounded (1106 / 30 s lands on 36.866666, a µs before frame 1106's
  // 36.866667), and Chrome starts some frames a µs late. A time within 3 µs of a frame's start (not 0 s or the clip end)
  // lands 1 µs past it, so the picture is that frame; currentTime reads it back as the frame's start.
  function mediaTime(time) {
    const fps = state.episode?.fps || 30, frame = Math.round(time * fps), edge = Math.round(frame / fps * 1e6);
    const inside = frame > 0 && frame < Math.round((state.episode?.duration_seconds || 0) * fps);
    return inside && Math.abs(time * 1e6 - edge) < 3 ? (edge + 1.25) / 1e6 : time;
  }
  /** The frame on screen at `time` (whole µs; frame k starts at k / fps rounded to the µs). */
  function frameAt(time) { return Math.floor((Math.round(time * 1e6) + 0.5) * (state.episode?.fps || 30) / 1e6); }
  function updatePlayback(time = currentTime()) {
    const duration = state.episode?.duration_seconds || 0;
    // the frame on screen (not the nearest one): ← → on the focused bar then step exactly one frame from a mid-frame pause
    $("video-seek").value = String(frameAt(time)); I18N.attr($("video-seek"), "aria-valuetext", `${time.toFixed(2)} s`);
    I18N.text($("video-time"), `${time.toFixed(2)} / ${duration.toFixed(2)} s`);
    I18N.text($("current-time"), `${time.toFixed(2)} s`);
    if (isCurve() && !state.curveTimePinned && document.activeElement !== $("curve-time")) $("curve-time").value = String(time);
    if (!isCurve() && !state.intervalTimePinned && document.activeElement !== $("interval-time")) $("interval-time").value = String(time);
    if (independentStages() && !state.stageTimePinned && document.activeElement !== $("stage-boundary-time")) $("stage-boundary-time").value = String(time);
    // own tracks: plot 784 × zoom wide (makeTimeline), read from the drawn track's viewBox
    const trackPlot = (line) => (line.ownerSVGElement?.viewBox?.baseVal?.width || 800) - 16;
    const marker = $("own-timeline").querySelector(".playhead");
    if (marker && duration) {
      const left = isCurve() ? 48 : 8, plotWidth = isCurve() ? 736 : trackPlot(marker);
      const x = left + Math.min(time / duration, 1) * plotWidth;
      marker.setAttribute("x1", x); marker.setAttribute("x2", x);
    }
    updateCurrentAnnotation(time);
    updateCurrentStage(time);
    $("own-stage-timeline").querySelectorAll(".playhead").forEach((line) => { const x = 8 + Math.min(time / (duration || 1), 1) * trackPlot(line); line.setAttribute("x1", x); line.setAttribute("x2", x); });
    updateControls();
  }
  function stageCoverage() {
    const duration = state.episode?.duration_seconds || 0;
    const covered = state.stageIntervals.reduce((total, interval) => total + interval.end - interval.start, 0);
    let cursor = 0, complete = state.stageIntervals.length > 0;
    state.stageIntervals.forEach((interval) => { if (Math.round(interval.start * 1e6) !== Math.round(cursor * 1e6)) complete = false; cursor = interval.end; });
    if (Math.round(cursor * 1e6) !== Math.round(duration * 1e6)) complete = false;
    return {covered, missing: Math.max(0, duration - covered), complete};
  }
  function makeStageTimeline(episode, tracks, own = false) {
    const scheme = {id: "stage_aq", name: "Subtask", kind: "stages", labels: episode.stages.map((_, index) => ({id: `stage_${index}`, name: `S${index + 1}. ${stageName(index, episode)}`, color: stageColor(index)}))};
    const converted = tracks.map((track) => ({annotator_id: track.annotator_id, annotation_format: "intervals-v1", intervals: (track.stage_intervals || trackIntervals(track, episode).filter((interval) => Number.isInteger(interval.stage))).map((interval) => ({start: interval.start, end: interval.end, stage: interval.stage, label: `stage_${interval.stage}`})), pending_interval: track.pending_stage_interval ? {...track.pending_stage_interval, label: `stage_${track.pending_stage_interval.stage}`} : null}));
    const svg = makeTimeline(episode, scheme, converted, [], own); svg.classList.add("semantic-stage-timeline"); return svg;
  }
  function updateCurrentStage(time = currentTime()) {
    const node = $("current-stage-annotation"); node.replaceChildren();
    if (!independentStages()) return;
    const interval = state.stageIntervals.find((item) => item.start <= time && (time < item.end || time === state.episode?.duration_seconds && time === item.end));
    node.dataset.stage = interval ? String(interval.stage) : "";
    node.append(el("strong", "", interval ? `S${interval.stage + 1}. ${stageName(interval.stage)}` : "No subtask here"), el("span", "current-annotation-detail", interval ? `${timeRange(interval.start, interval.end)} · subtask interval` : `${time.toFixed(2)} s · no subtask yet.`));
    if (state.pendingStageInterval) node.append(el("span", "current-annotation-detail", `Pending: ${state.pendingStageInterval.start.toFixed(2)} s · S${state.pendingStageInterval.stage + 1}, not yet counted as coverage.`));
  }
  function renderStageEditor() {
    const active = independentStages();
    $("stage-track-card").hidden = !active; $("independent-stage-editor").hidden = !active; $("independent-aq-heading").hidden = !active || isSubtaskOnly(); I18N.text($("independent-stage-editor").querySelector("h3"), isSubtaskOnly() ? "Subtask" : "1 · Subtask");
    $("own-stage-timeline").replaceChildren(); $("stage-interval-list").replaceChildren(); $("stage-coverage-summary").replaceChildren();
    if (!active || !state.episode) return;
    const {covered, missing, complete} = stageCoverage(), duration = state.episode.duration_seconds;
    $("stage-coverage-summary").dataset.coveredSeconds = String(covered); $("stage-coverage-summary").dataset.missingSeconds = String(missing); $("stage-coverage-summary").dataset.complete = String(complete);
    $("stage-coverage-summary").append(el("span", "coverage-chip", `Subtasks cover ${covered.toFixed(1)} / ${duration.toFixed(1)} s`), el("span", missing > 1e-6 ? "coverage-chip coverage-unlabelled" : "coverage-chip", missing > 1e-6 ? `Missing ${missing.toFixed(1)} s` : "Fully covered"), el("span", "coverage-caption", `${state.stageIntervals.length} subtask interval${state.stageIntervals.length === 1 ? "" : "s"}${state.pendingStageInterval ? " · 1 pending" : ""} · cover the whole clip`));
    $("own-stage-timeline").append(makeStageTimeline(state.episode, [{stage_intervals: state.stageIntervals, pending_stage_interval: state.pendingStageInterval}], true));
    I18N.text($("stage-interval-count"), state.stageIntervals.length);
    if (!state.stageIntervals.length) $("stage-interval-list").append(el("p", "small muted", "No subtasks yet. Pick one, then drag on the track."));
    state.stageIntervals.forEach((interval, index) => {
      const row = el("div", "point-row interval-row"), time = el("button", "point-time mono", timeRange(interval.start, interval.end)); time.type = "button"; time.addEventListener("click", () => { pauseAll(); seek(interval.start); });
      // One line: swatch + "S2 · prompt" without its [precondition] prefix, cut with an ellipsis; the full prompt is the title.
      const label = el("button", "point-label edit-stage-interval"), swatch = el("i"); label.type = "button"; label.title = `S${interval.stage + 1} · ${stageName(interval.stage)}`;
      swatch.style.background = stageColor(interval.stage); label.append(swatch, el("span", "", `S${interval.stage + 1} · ${stageName(interval.stage).replace(/^\[[^\]]*\]\s*/, "")}`));
      label.addEventListener("click", () => { if (!canEdit()) return; if (state.pendingStageInterval) { showNotice("End or cancel the pending subtask first.", "error"); return; } state.editingStageInterval = index; state.selectedSemanticStage = interval.stage; $("independent-stage-select").value = String(interval.stage); $("stage-edit-start").value = String(interval.start); $("stage-edit-end").value = String(interval.end); pauseAll(); seek(interval.start); updateControls(); annotationFeedback("Editing this subtask. Change its ends or subtask, then Update subtask."); });
      const remove = el("button", "remove-stage-interval", "×"); remove.type = "button"; I18N.attr(remove, "aria-label", `Remove subtask ${timeRange(interval.start, interval.end)}`);
      remove.addEventListener("click", () => { if (!canEdit()) return; remember(); state.stageIntervals.splice(index, 1); state.editingStageInterval = null; changed(); annotationFeedback("Subtask removed; its range has no subtask now. Undo restores it."); });
      row.append(time, label, remove); $("stage-interval-list").append(row);
    });
    updateCurrentStage();
  }
  async function playPause() {
    if (!state.videos.length) return;
    markInteraction();
    if (!state.videos[0].paused || holding) { pauseAll(); return; }
    state.playbackRequested = true;
    const live = state.videos.filter((video) => !video.error);   // a camera that failed to load does not hold the others back
    try {
      await Promise.all((live.length ? live : state.videos).map((video) => video.play()));
      if (!state.playbackRequested) { pauseAll(); return; }
      I18N.text($("play-button"), "Pause");
    } catch (error) {
      if (holding && error.name === "AbortError") { I18N.text($("play-button"), "Pause"); return; }   // a camera is loading: resumePlayback
      pauseAll(); showNotice(`Unable to play video: ${error.message}`, "error");
    }
  }
  // A camera that runs out of data (waiting / stalled), or a seek during playback, pauses every camera while playback stays
  // requested; once all can play they restart together from the first camera's time.
  const canPlay = (video) => Boolean(video.error) || !video.seeking && (video.readyState >= 3 || video.ended);
  function holdPlayback() {
    if (holding || !state.playbackRequested || state.videos.length < 2) return;
    holding = true; state.videos.forEach((video) => video.pause());
  }
  function resumePlayback() {
    if (!holding || !state.videos.every(canPlay)) return;
    if (state.videos[0].ended) { pauseAll(); return; }   // held at the clip end (a seek there): playback ends, as it does when played to the end
    const time = state.videos[0].currentTime, apart = state.videos.filter((video) => !video.error && Math.abs(video.currentTime - Math.min(time, video.duration)) > 0.02);
    if (apart.length) { apart.forEach((video) => { video.currentTime = Math.min(time, video.duration); }); return; }   // back here on their seeked
    holding = false; I18N.text($("play-button"), "Pause");
    // play() on an ended camera (a shorter file) would restart it from 0 s; one that failed to load cannot play
    Promise.all(state.videos.filter((video) => !video.ended && !video.error).map((video) => video.play())).catch((error) => { if (error.name !== "AbortError") pauseAll(); });
  }
  function syncViews() {
    const master = state.videos[0];
    if (!master) return;
    state.videos.slice(1).forEach((video) => {
      if (!video.error && Number.isFinite(video.duration) && Math.abs(video.currentTime - master.currentTime) > 0.12) video.currentTime = Math.min(master.currentTime, video.duration);
    });
    updatePlayback(currentTime());
  }
  function updateControls() {
    const editable = canEdit();
    $("episode-select").disabled = state.busy || !state.episodes.length;
    $("scheme-select").disabled = state.busy || !state.schemes.length;
    const locked = !isCurve() && Boolean(state.pendingInterval);
    $("stage-select").disabled = !editable || locked;
    $("quality-buttons").querySelectorAll("button").forEach((button) => { button.disabled = !editable || locked; });
    const continuous = isCurve();
    $("add-point").disabled = !editable || (!continuous && !locked && (!state.selectedLabel || (requiresExceptionStage() && state.selectedStage === null)));
    I18N.text($("add-point"), continuous ? (state.editingTime === null ? "Add reward anchor" : "Update selected anchor") : state.editingInterval !== null ? "Update interval" : state.pendingInterval ? "End interval" : "Start interval");
    $("interval-controls").hidden = continuous || state.legacy || isSubtaskOnly(); $("add-point").hidden = isSubtaskOnly();
    $("interval-controls").querySelectorAll("input,button").forEach((input) => { input.disabled = !editable; });
    I18N.text($("interval-time-label"), state.pendingInterval ? "End time (s)" : "Start time (s)");
    $("interval-time").closest("label").hidden = state.editingInterval !== null;
    $("interval-use-time").parentElement.hidden = state.editingInterval !== null;
    $("interval-edit-fields").hidden = state.editingInterval === null;
    $("interval-cancel-edit").hidden = state.editingInterval === null;
    $("interval-cancel-pending").hidden = !state.pendingInterval;
    $("interval-pending").hidden = !state.pendingInterval;
    I18N.text($("interval-pending"), state.pendingInterval ? `${state.pendingInterval.start.toFixed(2)} s → pending · ${schemeLabel(state.scheme, state.pendingInterval.label)?.name || state.pendingInterval.label}${state.pendingInterval.stage != null ? ` · ${stageName(state.pendingInterval.stage)}` : ""}. Label locked; ${defaultNormal() ? "the default stays until then" : "nothing counts as covered until the end"}.` : "");
    $("legacy-notice").hidden = !state.legacy;
    I18N.text($("legacy-title"), state.annotationFormat === "intervals-v1" ? "Earlier start/end annotation · read-only" : "Legacy hold-forward annotation · read-only");
    I18N.text($("legacy-description"), state.scheme?.id === "stage_aq" ? "Legacy Subtask + AQ record, read-only. Enable independent subtasks to keep its subtask spans and AQ exceptions; the rest becomes default Nominal." : isAQ() ? "Legacy AQ record, read-only. Use default Nominal to keep its spans and make all other time default Nominal." : "Legacy hold-forward record, read-only. Convert to start/end intervals; the last point needs an end time.");
    I18N.text($("legacy-convert"), state.scheme?.id === "stage_aq" ? "Enable independent subtasks" : isAQ() ? "Use default Nominal" : "Convert to start/end");
    $("legacy-convert").disabled = state.busy || !state.session.annotator_id;
    $("default-normal-note").hidden = !defaultNormal() || isSubtaskOnly();
    I18N.text($("default-normal-note"), skippedQuality() ? "AQ was not requested. Unmarked time defaults to Normal; this is not a model judgment." : machinePending() ? "Unpainted quality is unreviewed. It may include machine-unknown time; inspect it before confirming Nominal." : `Unmarked time = Nominal. Mark only the exceptions. ${independentStages() ? "" : state.scheme?.id === "stage_aq" ? "Spans need a subtask." : ""}`);
    renderMachineImport();
    I18N.text($("review-confirmed-text"), state.record?.machine_import ? "I reviewed every proposal and unknown region; unmarked time is Nominal." : isSubtaskOnly() ? "I reviewed the whole clip; every subtask is marked." : "I reviewed the whole clip; unmarked time is Nominal.");
    $("review-confirmed-field").hidden = !defaultNormal();
    $("review-confirmed").checked = state.reviewConfirmed;
    $("review-confirmed").disabled = !editable || Boolean(state.pendingInterval) || Boolean(state.pendingStageInterval);
    I18N.text($("point-list-title"), continuous ? "Control points" : state.legacy ? "Original annotation" : defaultNormal() ? "Explicit exceptions" : "Closed intervals");
    $("curve-controls").querySelectorAll("input,select,button").forEach((input) => { input.disabled = !editable; });
    $("curve-fit").disabled = state.busy || !continuous || !state.points.length || !state.session.annotator_id;
    $("curve-cancel-edit").hidden = state.editingTime === null;
    const pending = Boolean(state.pendingInterval || state.pendingStageInterval);
    $("save-button").disabled = !editable || pending || state.saveConflict;
    I18N.attr($("save-button"), "title", pending ? "End or cancel the pending span before saving." : "");
    $("save-conflict").hidden = !state.saveConflict && !state.sessionConflict;
    I18N.text($("recovery-message"), state.sessionConflict ? `Your editing name is ${state.session.annotator_id}. Restore it to continue without losing unsaved labels, or export them before leaving.` : "Saved labels changed while you were editing. Your edits are still here. Export them before reloading, then compare and reapply any changes.");
    $("restore-editing-name").hidden = !state.sessionConflict;
    $("restore-editing-name").disabled = state.busy;
    $("reload-saved-labels").hidden = state.sessionConflict;
    $("reload-saved-labels").disabled = state.busy;
    $("export-unsaved-labels").disabled = state.busy || pending;
    $("submit-button").disabled = !editable || state.saveConflict || state.sessionConflict || unresolvedOutcomes() || (continuous ? !state.points.length : defaultNormal() ? !state.reviewConfirmed || Boolean(state.pendingInterval) || Boolean(state.pendingStageInterval) || independentStages() && !stageCoverage().complete : !state.intervals.length || Boolean(state.pendingInterval));
    $("undo-button").disabled = !editable || !state.history.length;
    $("clear-button").disabled = !editable || !(continuous ? state.points.length : state.intervals.length || state.pendingInterval || state.stageIntervals.length || state.pendingStageInterval);
    $("reopen-button").disabled = state.busy || state.legacy;
    $("save-button").hidden = isSubmitted();
    $("submit-button").hidden = isSubmitted();
    $("reopen-button").hidden = !isSubmitted();
    $("point-list").querySelectorAll(".remove-point,.edit-point").forEach((button) => { button.disabled = !editable; });
    $("independent-stage-editor").querySelectorAll("input,select,button").forEach((control) => { control.disabled = !editable; });
    $("independent-stage-select").disabled = !editable || Boolean(state.pendingStageInterval);
    $("add-stage-interval").disabled = !editable || state.selectedSemanticStage === null && !state.pendingStageInterval;
    I18N.text($("add-stage-interval"), state.editingStageInterval !== null ? "Update subtask" : state.pendingStageInterval ? "End subtask" : "Start subtask");
    I18N.text($("stage-boundary-label"), state.pendingStageInterval ? "Subtask end (s)" : "Subtask start (s)");
    $("stage-boundary-field").hidden = state.editingStageInterval !== null; $("stage-time-tools").hidden = state.editingStageInterval !== null;
    $("stage-edit-fields").hidden = state.editingStageInterval === null; $("cancel-stage-edit").hidden = state.editingStageInterval === null;
    $("cancel-pending-stage").hidden = !state.pendingStageInterval; $("stage-pending-note").hidden = !state.pendingStageInterval;
    I18N.text($("stage-pending-note"), state.pendingStageInterval ? `${state.pendingStageInterval.start.toFixed(2)} s → pending · S${state.pendingStageInterval.stage + 1} ${stageName(state.pendingStageInterval.stage)}. Not yet counted as coverage.` : "");
    I18N.text($("record-status"), state.dirty ? "Unsaved draft" : isSubmitted() ? "Submitted" : state.record ? "Saved draft" : "No annotation");
    $("record-status").className = `status-chip ${state.dirty ? "unsaved" : isSubmitted() ? "submitted" : ""}`;
    I18N.text($("active-time"), `${clock(state.activeSeconds)} active`);
    const coverageCaption = $("coverage-summary").querySelector(".coverage-caption");
    if (coverageCaption) I18N.text(coverageCaption, `${countCaption()} · ${state.dirty ? "unsaved" : isSubmitted() ? "submitted" : state.record ? "saved" : "nothing saved"}`);
    const saved = state.record?.labeled_by ? `Last saved by ${state.record.labeled_by}${savedAt(state.record.updated_at)}. ` : "";
    I18N.text($("save-description"), !state.session.annotator_id ? "Enter a name to save your labels." : saved + (pending ? "End or cancel the pending span before saving or leaving this episode." : isSubmitted() ? "Submitted: grana build-cache will include this episode. Reopen to edit." : "Drafts are not used by grana build-cache. Save before leaving."));
    $("export-own").hidden = !state.record;
    if (state.record && state.episode && state.scheme) $("export-own").href = `${ownPath()}/export?annotator_id=${encodeURIComponent(state.session.annotator_id)}`;
    document.querySelectorAll(".workspace-nav button,#account-button,#reviewer-button,#logout-button").forEach((button) => { button.disabled = state.busy; });
  }
  function savedAt(iso) {
    const when = new Date(iso);
    if (Number.isNaN(when.getTime())) return "";
    const today = when.toDateString() === new Date().toDateString();
    return ` at ${when.toLocaleString([], today ? {hour: "2-digit", minute: "2-digit"} : {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"})}`;
  }
  async function withBusy(operation) {
    if (state.busy) return false;
    state.busy = true; updateControls();
    try { await operation(); return true; }
    catch (error) { showNotice(error.message, "error"); return false; }
    finally { state.busy = false; updateControls(); }
  }
  function renderSession() {
    const {annotator_id, reviewer} = state.session;
    const protectedMode = protectedStudy(), manages = canManage();
    if (annotator_id) $("session-badge").textContent = annotator_id;
    else I18N.text($("session-badge"), protectedMode && reviewer ? "Researcher" : "No name yet");
    I18N.text($("account-button"), annotator_id ? "Change name" : "Enter a name to start");
    $("reviewer-button").hidden = !protectedMode || Boolean(reviewer);
    if (!protectedMode && $("reviewer-dialog").open) $("reviewer-dialog").close();
    $("logout-button").hidden = !annotator_id && !reviewer;
    document.querySelectorAll(".management-only").forEach((node) => { node.hidden = !manages; });
    document.querySelectorAll(".management-eyebrow").forEach((node) => { I18N.text(node, protectedMode ? "RESEARCHER WORKSPACE" : "LOCAL WORKSPACE"); });
    $("access-mode-badge").replaceChildren(el("i"), I18N.node(protectedMode ? "Protected study workspace" : "Runs in your browser"));
    I18N.text($("agreement-access-note"), protectedMode ? "Protected study mode: submitted labels and alignment review are available only with researcher access. Keep this page separate from independent annotation." : "Open local mode: anyone with access to this app can view agreement results and export annotation records. For a formal independent-annotation study, restart with --require-reviewer-key and keep participants away from alignment review.");
    I18N.text($("study-access-instruction"), protectedMode ? "Counterbalance protocol order; keep agreement review researcher-only." : "For a formal study, restart with --require-reviewer-key, counterbalance protocol order and hide alignment results from participants.");
    $("empty-signin").hidden = Boolean(annotator_id);
    I18N.text($("empty-description"), manages ? "Enter a name to start; labels are saved on this machine in the recipe's labels directory." : !annotator_id ? "Enter a name to start; labels are saved on this machine in the recipe's labels directory." : state.episodes.length ? "Pick an episode above." : "No trajectories have been imported yet. Ask your researcher to add a video, task description and subtask vocabulary.");
    if (!manages && state.panel !== "annotate") setPanel("annotate");
    updateWorkspaceNote();
  }
  function renderSelectors() {
    const episodeId = state.episode?.id;
    [$("episode-select"), $("report-episode")].forEach((select) => {
      const previous = select.value;
      select.replaceChildren();
      if (!state.episodes.length) select.append(el("option", "", "No trajectories imported"));
      state.episodes.forEach((episode) => { const option = el("option", "", episode.title); option.value = episode.id; select.append(option); });
      if (state.episodes.some((episode) => episode.id === previous)) select.value = previous;
    });
    if (episodeId) $("episode-select").value = episodeId;
    const listed = state.episodes.find((episode) => episode.id === episodeId);   // titles carry the saved status: keep the card header in step
    if (listed) I18N.text($("video-title"), listed.observation_title || listed.title);
    $("scheme-select").replaceChildren();
    state.schemes.forEach((scheme) => { const option = el("option", "", ((window.AGREEMENT_STATIC.copy || {})[scheme.id] || {}).option || scheme.name); option.value = scheme.id; $("scheme-select").append(option); });
    if (state.scheme) $("scheme-select").value = state.scheme.id;
  }
  function renderProtocol() {
    const scheme = state.scheme;
    $("protocol-note").replaceChildren();
    $("quality-buttons").replaceChildren();
    $("stage-select").replaceChildren(el("option", "", "Choose a subtask"));
    $("stage-select").firstChild.value = "";
    if (!scheme) return;
    const continuous = isCurve(scheme);
    $("protocol-note").append(el("strong", "", `${scheme.name}.`), ...(() => { const c = (window.AGREEMENT_STATIC.copy || {})[scheme.id] || {}; const d = el("details", "protocol-rubric"); d.append(el("summary", "", "Rubric"), el("p", "", c.rubric || scheme.description || "")); return [el("span", "protocol-summary", c.line || ""), d]; })());
    $("stage-field").hidden = !requiresExceptionStage(); document.documentElement.dataset.scheme = scheme.id;
    $("quality-buttons").hidden = continuous || isSubtaskOnly(scheme);
    $("curve-controls").hidden = !continuous;
    $("curve-preview-status").hidden = !continuous;
    I18N.text($("judgment-title"), continuous ? "Place a reward anchor" : isSubtaskOnly(scheme) ? "Mark the subtasks" : "Label this interval");
    I18N.text($("own-timeline-title"), continuous ? "Your fitted reward curve" : "Annotation timeline");
    I18N.text($("timeline-interaction-hint"), continuous ? "Click to select · double-click to add" : "");
    I18N.text($("timeline-help"), continuous ? "Anchors define the curve; 0 s and the end are required." : "Each point holds until the next; gaps stay blank.");
    I18N.text($("keyboard-note"), continuous ? "P = time at playhead · Space play · ←/→ frame · ? shortcuts" : "Space play · ←/→ frame · S/E span · ? shortcuts");
    I18N.text($("annotation-help"), !state.session.annotator_id ? "Read-only until you enter a name." : continuous ? "Add time–reward anchors; values may span subtasks." : isSubtaskOnly() ? "Repeat until the clip is covered, then submit." : defaultNormal() ? (independentStages() ? "Subtasks first, then Suboptimal / Regressed spans." : "Pick Suboptimal or Regressed, then drag on the track.") : "Pick a subtask and High / Mid / Low, then drag on the track.");
    [$("interval-time"), $("interval-edit-start"), $("interval-edit-end")].forEach((input) => { input.max = state.episode?.duration_seconds || 0; });
    [$("stage-boundary-time"), $("stage-edit-start"), $("stage-edit-end")].forEach((input) => { input.max = state.episode?.duration_seconds || 0; });
    $("independent-stage-select").replaceChildren(el("option", "", "Choose a subtask")); $("independent-stage-select").firstChild.value = "";
    state.episode?.stages?.forEach((_, index) => { const option = el("option", "", `S${index + 1}. ${stageName(index)}`); option.value = String(index); $("independent-stage-select").append(option); });
    $("independent-stage-select").value = state.selectedSemanticStage === null ? "" : String(state.selectedSemanticStage);
    $("curve-time").max = state.episode?.duration_seconds || 0;
    $("curve-value").max = scoreMax();
    $("curve-interpolation").value = state.interpolation;
    I18N.text($("curve-range"), `Reward 0–${scoreMax()} · clip end ${(state.episode?.duration_seconds ?? 0).toFixed(2)} s.`);
    I18N.text($("stage-field").querySelector("label"), scheme.id === "stage_score" ? "Subtask" : "Subtask");
    state.episode?.stages?.forEach((_, index) => { const option = el("option", "", `S${index + 1}. ${stageName(index)}`); option.value = index; $("stage-select").append(option); });
    scheme.labels.filter((label) => !defaultNormal() || label.id !== "normal").forEach((label) => {
      const button = el("button", "quality-button"); button.type = "button";
      const dot = el("i", "quality-dot"); dot.style.background = labelColor(scheme, label.id);
      button.append(dot, el("span", "quality-name", label.name), el("span", "check", "✓"));
      button.setAttribute("aria-pressed", "false");
      button.dataset.label = label.id;
      if (label.description) I18N.attr(button, "title", label.description);
      button.addEventListener("click", () => {
        if (!canEdit() || state.pendingInterval) return;
        state.selectedLabel = label.id; markInteraction(); renderSelectedLabel();
      });
      $("quality-buttons").append(button);
    });
    renderSelectedLabel();
  }
  function renderSelectedLabel() {
    $("quality-buttons").querySelectorAll("button").forEach((button) => { const selected = button.dataset.label === state.selectedLabel; button.classList.toggle("selected", selected); button.setAttribute("aria-pressed", String(selected)); });
    $("stage-select").value = state.selectedStage === null ? "" : String(state.selectedStage);
    updateControls();
  }
  /** Why a camera failed: the server's JSON error for its URL, or the codec message when the server did send the video. */
  async function videoFailed(video, frame, index, name, url) {
    if (frame.classList.contains("failed") || !state.videos.includes(video)) return;   // a camera of an episode left behind
    let detail = null;   // stays null when the server sent the video: then the browser cannot decode it
    try {
      const response = await fetch(url, {headers: {Range: "bytes=0-0"}});
      if (response.status === 416) detail = "the video file is empty.";   // no byte 0 to send
      else if (!response.ok) detail = (await response.json().catch(() => ({}))).error || response.statusText || `HTTP ${response.status}`;
    } catch (_) { detail = "no reply from the grana label server; see its terminal."; }
    if (!state.videos.includes(video)) return;   // another episode opened meanwhile
    // the server sends the video now, so the first load may have failed on the way (a dropped connection): load it once more,
    // at the playhead and the chosen speed (defaultPlaybackRate); running playback waits for it and restarts all together
    if (detail === null && !video.dataset.reloaded) {
      video.dataset.reloaded = "1";
      if (video !== state.videos[0] || video.readyState) video.dataset.pendingSeek = String(mediaTime(currentTime()));   // else: the last seek
      holdPlayback(); video.load(); return;
    }
    const codec = `Cannot play ${name}. Check that this browser supports its video codec.`;
    frame.classList.remove("loading"); frame.classList.add("failed"); I18N.attr(frame, "title", detail === null ? codec : `${name}: ${detail}`);
    // the tile shows the start of the reason, with long paths cut to their last two parts (the notice and the title keep it all)
    const short = detail === null ? codec : detail.replace(/\/(?:[^\s/]+\/)+([^\s/]+\/[^\s/]+)/g, "…/$1");
    I18N.text(frame.querySelector(".view-status"), short.charAt(0).toUpperCase() + short.slice(1));
    videoProblems[index] = {name, detail};
    // the first camera drives the playhead (currentTime, syncViews): a failed one hands that to a camera that loads, and
    // playback waiting for this one goes on without it
    const next = state.videos[0] === video && state.videos.find((v) => !v.error);
    if (next) { state.videos.splice(state.videos.indexOf(next), 1); state.videos.unshift(next); updatePlayback(currentTime()); }
    resumePlayback(); showNotice("");
  }
  let enlargedView = null;   // the camera shown large; kept when the next episode opens
  function enlargeView(frame, on) {
    $("video-grid").querySelectorAll(".view-frame").forEach((node) => {
      const big = node === frame && on;
      node.classList.toggle("enlarged", big); node.querySelector(".view-expand").setAttribute("aria-pressed", String(big));
      I18N.attr(node.querySelector(".view-expand"), "title", big ? "Restore (Esc)" : "Enlarge (double-click)");
    });
    enlargedView = on ? frame.dataset.view : null;
  }
  function restoreEnlargedView() {
    const frame = $("video-grid").querySelector(".view-frame.enlarged");
    if (!frame) return false;
    enlargeView(frame, false);
    return true;
  }
  function renderVideos() {
    // stop the previous cameras' downloads too: paused, they would hold the browser's connections to the server, so saves
    // and the new cameras would wait for them
    pauseAll(); $("video-grid").querySelectorAll("video").forEach((video) => { video.removeAttribute("src"); video.load(); });
    state.videos = []; videoProblems = []; $("video-grid").replaceChildren();
    const episode = state.episode;
    if (!episode) return;
    I18N.text($("video-title"), episode.observation_title || episode.title);
    $("task-description").textContent = episode.task;
    $("stage-vocabulary").replaceChildren();
    if (episode.stages?.length) {
      $("stage-vocabulary").append(el("span", "stage-vocabulary-label", "Subtasks"));
      episode.stages.forEach((_, index) => $("stage-vocabulary").append(el("span", "stage-vocabulary-item", `${index + 1}. ${stageName(index)}`)));
    }
    I18N.text($("view-count"), `${episode.views.length} file${episode.views.length === 1 ? "" : "s"} · ${episode.views.map((v, i) => v.name || `View ${i + 1}`).join(", ")} · ${episode.duration_seconds.toFixed(1)} s`);
    $("video-grid").classList.toggle("single-view", episode.views.length === 1);
    episode.views.forEach((view, index) => {
      const name = view.name || `View ${index + 1}`;
      const frame = el("div", "view-frame loading"); frame.dataset.view = name;
      const video = el("video"); video.src = view.url; video.preload = "auto"; video.muted = true; video.playsInline = true;
      video.defaultPlaybackRate = video.playbackRate = Number($("playback-rate").value);   // load() resets the rate to the default
      video.setAttribute("aria-label", name);
      video.addEventListener("loadedmetadata", () => {
        if (video.dataset.pendingSeek !== undefined) video.currentTime = Math.min(Number(video.dataset.pendingSeek), video.duration);
      });
      video.addEventListener("loadeddata", () => frame.classList.remove("loading"));
      // A click plays / pauses; a double-click enlarges and undoes its first click (play state and time).
      let before = null;
      video.addEventListener("click", (event) => { if (event.detail > 1) return; before = {time: currentTime(), paused: !state.videos[0] || state.videos[0].paused}; playPause(); });
      video.addEventListener("dblclick", () => {
        if (before && before.paused) { pauseAll(); seek(before.time); } else if (before && state.videos[0]?.paused) playPause();
        before = null; enlargeView(frame, !frame.classList.contains("enlarged"));
      });
      video.addEventListener("error", () => videoFailed(video, frame, index, name, view.url));
      ["waiting", "stalled"].forEach((type) => video.addEventListener(type, () => { if (video.readyState < 3) holdPlayback(); }));
      ["canplay", "seeked"].forEach((type) => video.addEventListener(type, resumePlayback));
      // the first camera in state.videos drives the playhead (videoFailed hands that on when it fails)
      video.addEventListener("timeupdate", () => { if (video === state.videos[0]) syncViews(); });
      video.addEventListener("ended", () => { if (video === state.videos[0]) pauseAll(); });
      const expand = el("button", "view-expand"); expand.type = "button"; I18N.attr(expand, "aria-label", `Enlarge ${name}`); expand.setAttribute("aria-pressed", "false"); I18N.attr(expand, "title", "Enlarge (double-click)");
      expand.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path class="grow" d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/><path class="shrink" d="M6 2v4H2M10 2v4h4M14 10h-4v4M2 10h4v4"/></svg>';
      expand.addEventListener("click", () => enlargeView(frame, !frame.classList.contains("enlarged")));
      frame.append(video, el("span", "view-label", name), el("span", "view-status", "Loading…"), expand);
      $("video-grid").append(frame); state.videos.push(video);
      if (name === enlargedView) enlargeView(frame, true);
    });
    $("video-seek").max = Math.round(episode.duration_seconds * (episode.fps || 30));   // in frames, so ← → on the seek bar step one frame
    updatePlayback(0);
  }
  /** Keep narrow labels within the plot in two nonoverlapping lanes. If both are full, the segment's accessible name
   *  and tooltip still identify it; zooming in or the Subtasks list reveals the full label. Reserve room for a failed " F". */
  function layoutStageLabels(labels, left, right, size) {
    const ends = [left - 3, left - 3];
    return labels.map((label) => {
      const half = (label.text.length + 2) * size * 0.36;
      const ideal = Math.max(left + half, Math.min(right - half, label.mid));
      let chosen = null;
      ends.forEach((end, lane) => {
        const x = Math.max(ideal, end + 3 + half);
        if (x + half <= right && (!chosen || Math.abs(x - label.mid) < Math.abs(chosen.x - label.mid))) chosen = {x, lane};
      });
      if (!chosen) return {...label, half, elided: true};
      ends[chosen.lane] = chosen.x + half;
      return {...label, half, ...chosen, elided: false};
    });
  }
  function makeTimeline(episode, scheme, tracks, disagreement = [], own = false) {
    const rh = own ? (window.innerWidth < 640 ? 64 : 36) : 23, lf = own && window.innerWidth < 640 ? 19 : 9;
    // A zoomed own track is wider in viewBox units and on screen alike (lab-interact.js scrolls it), so text keeps its size.
    const zoom = own ? state.trackZoom : 1, width = own ? 16 + 784 * zoom : 800, left = own ? 8 : 106, right = 8, rowHeight = own ? rh + 12 : 34, top = 11;
    // phones: the own subtask track's narrow-segment labels (19 units) need room above the axis labels (20 units, lab-theme.css)
    const plotWidth = width - left - right;
    let height = top + tracks.length * rowHeight + 25 + (own && lf > 9 && scheme.kind === "stages" ? 14 : 0);
    const duration = episode.duration_seconds;
    const svg = svgEl("svg", {viewBox: `0 0 ${width} ${height}`, class: "timeline-svg", role: scheme.kind === "stages" ? "group" : "img", "aria-label": own ? "Annotation intervals; click to seek" : `${scheme.name} independent annotator timelines`});
    if (zoom !== 1) svg.style.width = `${width / 8}%`;
    const unlabelledFill = own ? hatchPattern(svg) : "#1d1c19";
    tracks.forEach((track, index) => {
      const y = top + index * rowHeight;
      if (!own) {
        const fullName = String(track.annotator_id || ""), letters = Array.from(fullName);
        const name = svgEl("text", {x: 0, y: y + 15, "font-size": 10, fill: "#d8d3c9"}, letters.length > 15 ? `${letters.slice(0, 13).join("")}…` : fullName);
        name.append(svgEl("title", {}, fullName)); svg.append(name);
      }
      svg.append(svgEl("rect", {x: left, y, width: plotWidth, height: rh, rx: 3, fill: unlabelledFill, class: own ? "unlabelled-background" : ""}));
      const intervals = trackIntervals(track, episode);
      let cursor = 0;
      const gapLabel = (start, end) => { if (own && (end - start) / duration * plotWidth > lf * 7) svg.append(svgEl("text", {x: left + start / duration * plotWidth + 7, y: y + rh / 2 + 4, "font-size": lf + 1, fill: "#aea99f", "pointer-events": "none"}, "Unlabelled")); };
      const narrow = [];   // subtask segments too narrow for their S#: labelled under the row once all are known (below)
      intervals.forEach((interval, intervalIndex) => {
        gapLabel(cursor, interval.start); cursor = interval.end;
        const startX = left + interval.start / duration * plotWidth, endX = left + interval.end / duration * plotWidth;
        const name = schemeLabel(scheme, interval.label)?.name || interval.label;
        const isDefault = interval.source === "default";
        const rect = svgEl("rect", {x: startX, y, width: Math.max(0, endX - startX), height: rh, fill: isDefault ? defaultQualityColor() : labelColor(scheme, interval.label), opacity: own ? 1 : .92, class: scheme.kind === "stages" ? "labelled-interval stage-timeline-segment" : isDefault ? "default-normal-interval" : "labelled-interval", "data-start": interval.start, "data-end": interval.end, "data-label": interval.label, "data-stage": interval.stage ?? "", "data-source": isDefault ? "default" : "explicit"});
        const description = `${isDefault ? `${defaultQualityName()} · ${track.annotation_format === "stage-aq-independent-v1" ? "subtask in its own track" : "subtask unknown"}` : name}${interval.stage != null ? ` · ${stageName(interval.stage, episode)}` : ""} · ${timeRange(interval.start, interval.end)}${interval.legacy ? " · legacy hold-forward" : isDefault ? " · default background" : " · explicit start/end"}`;
        rect.setAttribute("role", "img"); I18N.attr(rect, "aria-label", description);
        rect.append(svgEl("title", {}, description)); svg.append(rect);
        const shortLabel = scheme.kind === "stages" ? `S${interval.stage + 1}` : isDefault ? defaultQualityName() : `${own && interval.stage != null ? `S${interval.stage + 1} · ` : ""}${name}`;
        const fill = isDefault ? defaultQualityColor() : labelColor(scheme, interval.label);
        // own subtask track: the S# goes below the row under 34 units (no room for the C / F mark either, lab-interact.js), and
        // where it would run into that mark at the segment's end (a 15-unit box 7 units in; phone rows are tall enough to clear it)
        const room = endX - startX, labelWidth = shortLabel.length * (own ? lf * 0.66 : 6);
        const narrowStage = own && scheme.kind === "stages" && (room < 34 || rh / 2 + 6 > rh - 17 && room < labelWidth + 31);
        if (room > labelWidth + 14 && !narrowStage) svg.append(svgEl("text", {x: startX + 7, y: y + rh / 2 + 4, "font-size": own ? lf : 9, "paint-order": "stroke", stroke: fill, "stroke-width": own ? 4 : 0, "stroke-linejoin": "round", fill: isDefault ? "#c8ead8" : inkOn(fill), "pointer-events": "none"}, shortLabel));
        else if (scheme.kind === "stages" && tracks.length === 1) narrow.push({interval, rect, mid: (startX + endX) / 2, text: shortLabel, color: fill});
        [startX, endX].forEach((x) => svg.append(svgEl("line", {x1: x, x2: x, y1: y, y2: y + rh, stroke: "#0f0e0d", "stroke-width": 1.4, "pointer-events": "none"})));
        if (own && !isDefault) {
          [interval.start, interval.end].forEach((time) => {
            const anchor = svgEl("circle", {cx: left + time / duration * plotWidth, cy: y, r: 3.5, fill: labelColor(scheme, interval.label), stroke: "#0f0e0d", "stroke-width": 1.4, class: "annotation-anchor", tabindex: 0, role: "button", "aria-label": `${time.toFixed(2)} s · ${name}`});
            const jump = () => { pauseAll(); seek(time); };
            anchor.addEventListener("click", (event) => { event.stopPropagation(); jump(); });
            anchor.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); jump(); } }); svg.append(anchor);
          });
        }
      });
      gapLabel(cursor, duration);
      // Bound both ends of every label; a crowded track uses a second lane instead of pushing earlier labels off-screen.
      const size = own ? lf : 9, base = y + rh + size + 5;
      const placed = layoutStageLabels(narrow, left, width - right, size);
      if (placed.some((label) => label.lane === 1)) height += size + 7;
      placed.forEach(({interval, rect, mid, x, lane, elided, text, color}) => {
        rect.dataset.labelElided = String(elided);
        if (elided) return;
        const baseline = base + lane * (size + 7);
        svg.append(svgEl("line", {x1: mid, x2: x, y1: y + rh + 1, y2: baseline - size * 0.8, stroke: color, "stroke-width": 2, "pointer-events": "none", "aria-hidden": "true"}));
        svg.append(svgEl("text", {x, y: baseline, "font-size": size, fill: "#d8d3c9", style: "text-anchor: middle", class: "stage-narrow-label", "data-lane": lane, "data-start": interval.start, "data-end": interval.end, "pointer-events": "none", "aria-hidden": "true"}, text));
      });
      if (own && track.pending_interval) {
        const x = left + track.pending_interval.start / duration * plotWidth;
        svg.append(svgEl("line", {x1: x, x2: x, y1: y - 7, y2: y + rh + 4, stroke: "#e3a72f", "stroke-width": 2.5, "stroke-dasharray": "3 2", class: "pending-interval-marker", "data-start": track.pending_interval.start, "pointer-events": "none"}));
        svg.append(svgEl("text", {x: Math.min(x + 5, width - 82), y: y - 2, "font-size": 9, fill: "#fcda86", "pointer-events": "none"}, "pending"));
      }
    });
    disagreement.forEach((interval) => {
      const rect = svgEl("rect", {x: left + interval.start / duration * plotWidth, y: top - 4, width: (interval.end - interval.start) / duration * plotWidth, height: tracks.length * rowHeight - 2, fill: "#ec5b52", "fill-opacity": .08, stroke: "#ec5b52", "stroke-opacity": .4, "stroke-width": .7, "stroke-dasharray": "3 3", "pointer-events": "none"});
      svg.append(rect);
    });
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const ticks = Math.round(4 * zoom);
    for (let index = 0; index <= ticks; index += 1) {
      const x = left + index / ticks * plotWidth;
      svg.append(svgEl("text", {x, y: height - 3, "font-size": 9, fill: "#aea99f", "text-anchor": index === 0 ? "start" : index === ticks ? "end" : "middle", class: "timeline-tick"}, `${(duration * index / ticks).toFixed(1)} s`));
    }
    if (own) {
      const x = left + currentTime() / duration * plotWidth;
      svg.append(svgEl("line", {x1: x, x2: x, y1: top - 5, y2: top + rh + 5, stroke: "#30c5a5", "stroke-width": 2, class: "playhead"}));
    }
    svg.addEventListener("click", async (event) => {
      const bounds = svg.getBoundingClientRect();
      const x = (event.clientX - bounds.left) / bounds.width * width;
      const time = Math.max(0, Math.min(1, (x - left) / plotWidth)) * duration;
      if (own) { seek(time); return; }
      await withBusy(async () => { await saveIfDirty(); await loadContext(episode.id, scheme.id); setPanel("annotate"); seek(time); });
    });
    return svg;
  }
  const annotatorColors = ["#30c5a5", "#e3a72f", "#84d765", "#30c5a5", "#ec5b52", "#b9ce6b", "#7fe0b8", "#ec5b52"];
  function selectCurveAnchor(point) {
    if (!canEdit()) return;
    pauseAll(); seek(point.time);
    state.curveTimePinned = true; state.editingTime = point.time;
    $("curve-time").value = String(point.time); $("curve-value").value = String(point.value); $("curve-step").checked = Boolean(point.step);
    updateControls();
  }
  function makeCurvePlot(episode, scheme, tracks, own = false) {
    const width = 800, height = 248, left = 48, right = 16, top = 18, bottom = 32;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const duration = episode.duration_seconds, maximum = scoreMax(episode);
    const xAt = (time) => left + time / duration * plotWidth;
    const yAt = (value) => top + plotHeight * (1 - value / maximum);
    const svg = svgEl("svg", {viewBox: `0 0 ${width} ${height}`, class: "timeline-svg reward-curve-svg", role: "img", "aria-label": own ? "Reward control points and fitted curve" : "Independent annotators’ reward curves"});
    svg.append(svgEl("rect", {x: left, y: top, width: plotWidth, height: plotHeight, fill: "#1d1c19", rx: 3}));
    if (own) {
      const unlabelledFill = hatchPattern(svg);
      coverageSegments().forEach((segment) => {
        const x = xAt(segment.start), span = xAt(segment.end) - x;
        svg.append(svgEl("rect", {x, y: top, width: span, height: plotHeight, fill: segment.kind === "unlabelled" ? unlabelledFill : "#1d3a33", class: segment.kind === "unlabelled" ? "curve-unlabelled-region" : "curve-covered-region", "data-start": segment.start, "data-end": segment.end, "pointer-events": "none"}));
        if (segment.kind === "unlabelled" && span > 68) svg.append(svgEl("text", {x: x + span / 2, y: top + 14, "text-anchor": "middle", "font-size": 10, fill: "#aea99f", "pointer-events": "none"}, "Unlabelled"));
      });
    }
    for (let index = 0; index <= 4; index += 1) {
      const value = maximum * index / 4, y = yAt(value), x = left + index / 4 * plotWidth;
      svg.append(svgEl("line", {x1: left, x2: width - right, y1: y, y2: y, stroke: "#2b2925", "stroke-width": 1}));
      svg.append(svgEl("text", {x: left - 9, y: y + 3, "text-anchor": "end", "font-size": 10, fill: "#d8d3c9"}, value.toFixed(2)));
      svg.append(svgEl("text", {x, y: height - 9, "text-anchor": index === 0 ? "start" : index === 4 ? "end" : "middle", "font-size": 10, fill: "#d8d3c9"}, `${(duration * index / 4).toFixed(2)} s`));
    }
    tracks.forEach((track, index) => {
      const color = annotatorColors[index % annotatorColors.length];
      const curve = track.curve;
      if (curve?.times?.length) {
        let path = "", connected = false;
        curve.times.forEach((time, sample) => {
          const value = curve.values[sample];
          if (typeof time !== "number" || typeof value !== "number" || !Number.isFinite(value)) { connected = false; return; }
          path += `${connected ? "L" : "M"}${xAt(time).toFixed(3)},${yAt(value).toFixed(3)} `;
          connected = true;
        });
        if (path) svg.append(svgEl("path", {d: path, fill: "none", stroke: color, "stroke-width": own ? 2.6 : 2, "stroke-linejoin": "round", "pointer-events": "none"}));
      }
      (track.points || []).forEach((point) => {
        const x = xAt(point.time), y = yAt(point.value);
        const marker = point.step ? svgEl("rect", {x: x - 4, y: y - 4, width: 8, height: 8, rx: 1}) : svgEl("circle", {cx: x, cy: y, r: 4});
        marker.setAttribute("fill", color); marker.setAttribute("stroke", "#0f0e0d"); marker.setAttribute("stroke-width", "1.4");
        marker.append(svgEl("title", {}, `${track.annotator_id ? `${track.annotator_id}: ` : ""}${point.time.toFixed(3)} s → ${point.value}${point.step ? " · step" : ""}`));
        if (own) {
          marker.classList.add("curve-anchor");
          marker.addEventListener("click", (event) => { event.stopPropagation(); selectCurveAnchor(point); });
          marker.addEventListener("dblclick", (event) => event.stopPropagation());
        }
        svg.append(marker);
      });
    });
    if (own) svg.append(svgEl("line", {x1: xAt(currentTime()), x2: xAt(currentTime()), y1: top, y2: top + plotHeight, stroke: "#30c5a5", "stroke-width": 1.2, "stroke-dasharray": "4 3", class: "playhead", "pointer-events": "none"}));
    const coordinates = (event) => {
      const bounds = svg.getBoundingClientRect(), x = (event.clientX - bounds.left) / bounds.width * width, y = (event.clientY - bounds.top) / bounds.height * height;
      return {time: Math.max(0, Math.min(1, (x - left) / plotWidth)) * duration, value: Math.round(Math.max(0, Math.min(1, (top + plotHeight - y) / plotHeight)) * maximum * 1000) / 1000};
    };
    svg.addEventListener("click", async (event) => {
      const {time, value} = coordinates(event);
      if (own) {
        pauseAll(); seek(time);
        if (canEdit()) { state.curveTimePinned = true; state.editingTime = null; $("curve-time").value = String(time); $("curve-value").value = String(value); updateControls(); }
        return;
      }
      await withBusy(async () => { await saveIfDirty(); await loadContext(episode.id, scheme.id); setPanel("annotate"); seek(time); });
    });
    if (own) svg.addEventListener("dblclick", (event) => { if (!canEdit()) return; event.preventDefault(); const {time, value} = coordinates(event); $("curve-time").value = String(time); $("curve-value").value = String(value); state.editingTime = null; addPoint(); });
    return svg;
  }
  function invalidateCurve() {
    state.curveRequest += 1; clearTimeout(state.curveTimer); state.curveTimer = null; state.curve = null;
  }
  function curveSignature() { return JSON.stringify([state.session.annotator_id, state.episode?.id, state.scheme?.id, state.interpolation, state.points]); }
  function scheduleCurvePreview(force = false) {
    invalidateCurve();
    if (!isCurve() || !state.episode) return;
    $("curve-preview-status").hidden = false;
    if (!state.points.length) { I18N.text($("curve-preview-status"), "No anchors yet."); return; }
    if (!force && !$("curve-auto-fit").checked) { I18N.text($("curve-preview-status"), "Anchors changed; press Fit curve."); return; }
    const request = state.curveRequest, signature = curveSignature();
    I18N.text($("curve-preview-status"), "Fitting preview…");
    state.curveTimer = setTimeout(async () => {
      try {
        const data = await api("/api/curves/preview", {method: "POST", body: JSON.stringify({episode_id: state.episode.id, points: state.points.map((point) => ({...point})), interpolation: state.interpolation})});
        if (request !== state.curveRequest || signature !== curveSignature() || !isCurve()) return;
        state.curve = data.curve;
        I18N.text($("curve-preview-status"), `${state.interpolation === "pchip" ? "PCHIP" : "Linear"} preview · ${state.points.length} anchor${state.points.length === 1 ? "" : "s"}.`);
        renderPoints();
      } catch (error) {
        if (request !== state.curveRequest || signature !== curveSignature()) return;
        I18N.text($("curve-preview-status"), `Preview unavailable: ${error.message} Your control points are unchanged; you can still save a draft.`);
        if (isSubmitted()) I18N.text($("own-result-description"), `Your submitted anchors are saved, but fitting failed: ${error.message} Click Fit curve to retry. No curve has been fabricated.`);
      }
    }, force ? 0 : 220);
  }
  function makeCategoricalCurve(episode, scheme, record) {
    const width = 800, height = 230, left = 104, right = 16, top = 27, bottom = 36;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const xAt = (time) => left + time / episode.duration_seconds * plotWidth;
    const yAt = (label) => top + scheme.labels.findIndex((item) => item.id === label) / Math.max(1, scheme.labels.length - 1) * plotHeight;
    const svg = svgEl("svg", {viewBox: `0 0 ${width} ${height}`, class: "timeline-svg submitted-category-curve", role: "img", "aria-label": "Your submitted categorical annotation. Named levels are categories, not scalar rewards."});
    svg.append(svgEl("rect", {x: left, y: top - 12, width: plotWidth, height: plotHeight + 24, fill: hatchPattern(svg)}));
    const intervals = trackIntervals(record, episode);
    intervals.forEach((interval) => svg.append(svgEl("rect", {x: xAt(interval.start), y: top - 12, width: xAt(interval.end) - xAt(interval.start), height: plotHeight + 24, fill: "#1d1c19"})));
    scheme.labels.forEach((label) => {
      const y = yAt(label.id);
      svg.append(svgEl("line", {x1: left, x2: width - right, y1: y, y2: y, stroke: "#2b2925", "stroke-width": 1}));
      svg.append(svgEl("text", {x: left - 10, y: y + 4, "text-anchor": "end", "font-size": 12, fill: labelColor(scheme, label.id)}, (schemeLabel(scheme, label.id)?.name || label.id)));
    });
    intervals.forEach((interval, index) => {
      const isDefault = interval.source === "default";
      const y = yAt(interval.label), line = svgEl("line", {x1: xAt(interval.start), x2: xAt(interval.end), y1: y, y2: y, stroke: isDefault ? "#5fae8c" : labelColor(scheme, interval.label), "stroke-width": isDefault ? 2.5 : 3.5, "stroke-dasharray": isDefault ? "5 3" : "none", class: "submitted-category-segment", "data-start": interval.start, "data-end": interval.end, "data-label": interval.label, "data-source": isDefault ? "default" : "explicit"});
      line.append(svgEl("title", {}, `${timeRange(interval.start, interval.end)} · ${isDefault ? `Default Nominal · ${record.annotation_format === "stage-aq-independent-v1" ? "subtask in its own track" : "subtask unknown"}` : interval.label}${interval.stage != null ? ` · ${stageName(interval.stage, episode)}` : ""}`)); svg.append(line);
      if (!isDefault) [interval.start, interval.end].forEach((time) => svg.append(svgEl("circle", {cx: xAt(time), cy: y, r: 3.2, fill: labelColor(scheme, interval.label), stroke: "#0f0e0d", "stroke-width": 1})));
      const next = intervals[index + 1];
      if (next && interval.end === next.start && next.label !== interval.label) svg.append(svgEl("line", {x1: xAt(interval.end), x2: xAt(interval.end), y1: y, y2: yAt(next.label), stroke: "#75716a", "stroke-width": 1.5, class: "submitted-category-transition"}));
    });
    for (let index = 0; index <= 4; index += 1) svg.append(svgEl("text", {x: left + index / 4 * plotWidth, y: height - 9, "font-size": 10, fill: "#aea99f", "text-anchor": index === 0 ? "start" : index === 4 ? "end" : "middle"}, `${(episode.duration_seconds * index / 4).toFixed(2)} s`));
    svg.addEventListener("click", (event) => { const bounds = svg.getBoundingClientRect(), x = (event.clientX - bounds.left) / bounds.width * width; pauseAll(); seek(Math.max(0, Math.min(1, (x - left) / plotWidth)) * episode.duration_seconds); });
    return svg;
  }
  function renderOwnResult(focus = false) {
    const card = $("own-result"), plot = $("own-result-plot");
    card.hidden = !isSubmitted() || !state.episode || !state.scheme;
    if (card.hidden) { plot.replaceChildren(); return; }
    const snapshot = state.record; plot.replaceChildren();
    I18N.text($("own-result-status"), `Submitted · revision ${snapshot.revision}`);
    if (isCurve()) {
      const curveMatches = JSON.stringify(snapshot.points || []) === JSON.stringify(state.points) && (snapshot.interpolation || "pchip") === state.interpolation;
      const curve = curveMatches ? state.curve : null;
      I18N.text($("own-result-description"), curve ? `Your saved reward curve · ${(snapshot.interpolation || "pchip").toUpperCase()}.` : "Anchors saved; fitting the curve.");
      plot.append(makeCurvePlot(state.episode, state.scheme, [{annotator_id: snapshot.annotator_id, points: snapshot.points || [], curve}]));
    } else {
      I18N.text($("own-result-description"), ["aq-default-normal-v1", "stage-aq-independent-v1"].includes(snapshot.annotation_format) ? `Submitted revision: dim green = unmarked (Nominal), solid = marked spans. ${snapshot.annotation_format === "stage-aq-independent-v1" ? "Subtasks in their own track." : "Default time has no subtask."}` : `${state.legacy ? "Original annotation semantics" : "Explicit start/end"} · submitted revision. Grey gaps are unlabelled.`);
      if (snapshot.annotation_format === "stage-aq-independent-v1") { const stagePlot = el("div", "submitted-stage-track"); stagePlot.id = "own-result-stage"; stagePlot.append(el("h3", "", "Subtasks · saved snapshot"), makeStageTimeline(state.episode, [{...snapshot, annotator_id: ""}])); plot.append(stagePlot, el("h3", "", "Action quality · saved snapshot")); }
      plot.append(makeCategoricalCurve(state.episode, state.scheme, snapshot));
    }
    if (focus) { card.scrollIntoView({behavior: "smooth", block: "center"}); card.focus({preventScroll: true}); }
  }
  function renderPoints() {
    renderStageEditor();
    if (!isCurve() && state.annotationFormat !== "legacy-points") { renderIntervalList(); return; }
    $("point-list").replaceChildren(); I18N.text($("point-count"), state.points.length);
    if (!state.points.length) $("point-list").append(el("p", "small muted", "No points yet. Add your first judgment at 0.00 s."));
    state.points.forEach((point, index) => {
      const row = el("div", "point-row");
      const time = el("button", "point-time mono", point.time.toFixed(2)); time.type = "button"; I18N.attr(time, "title", "Seek to this point"); time.addEventListener("click", () => seek(point.time));
      const label = el("button", "point-label edit-point"); label.type = "button"; I18N.attr(label, "title", isCurve() ? "Edit anchor time, reward or step status" : "Select this point to relabel at the same time");
      const dot = el("i"); dot.style.background = isCurve() ? annotatorColors[0] : labelColor(state.scheme, point.label);
      label.append(dot, el("span", isCurve() ? "mono" : "", isCurve() ? `${point.value}${point.step ? " · step" : ""}` : schemeLabel(state.scheme, point.label)?.name || point.label));
      label.addEventListener("click", () => { if (!canEdit()) return; if (isCurve()) { selectCurveAnchor(point); return; } seek(point.time); state.selectedLabel = point.label; state.selectedStage = point.stage ?? null; renderSelectedLabel(); showNotice("Point selected; add a label at this time to replace it.", "info"); });
      const remove = el("button", "remove-point", "×"); remove.type = "button"; I18N.attr(remove, "aria-label", `Remove point at ${point.time.toFixed(2)} seconds`);
      remove.addEventListener("click", () => { if (!canEdit()) return; remember(); state.points.splice(index, 1); state.editingTime = null; changed(); annotationFeedback(`Removed the point at ${point.time.toFixed(2)} s; coverage updated · unsaved.`); });
      row.append(time, label, remove);
      if (point.stage != null) row.append(el("span", "point-stage", stageName(point.stage)));
      $("point-list").append(row);
    });
    $("own-timeline").replaceChildren();
    if (state.episode && state.scheme) $("own-timeline").append(isCurve() ? makeCurvePlot(state.episode, state.scheme, [{points: state.points, curve: state.curve}], true) : makeTimeline(state.episode, state.scheme, [ownTrack()], [], true));
    renderCoverage();
    renderOwnResult();
    updateControls();
  }
  /** Redraw only the two own tracks, e.g. at a new zoom (lab-interact.js); lists and coverage stay as they are. */
  function renderTracks() {
    if (!state.episode || !state.scheme) return;
    if (independentStages()) $("own-stage-timeline").replaceChildren(makeStageTimeline(state.episode, [{stage_intervals: state.stageIntervals, pending_stage_interval: state.pendingStageInterval}], true));
    if (!isCurve()) $("own-timeline").replaceChildren(makeTimeline(state.episode, state.scheme, [ownTrack()], [], true));
  }
  function renderIntervalList() {
    $("point-list").replaceChildren(); I18N.text($("point-count"), state.intervals.length);
    if (!state.intervals.length) $("point-list").append(el("p", "small muted", state.pendingInterval ? "Span started; press E or End interval." : defaultNormal() ? "No spans marked; the clip is Nominal." : "No intervals yet."));
    state.intervals.forEach((interval, index) => {
      const row = el("div", "point-row interval-row");
      const time = el("button", "point-time mono", timeRange(interval.start, interval.end)); time.type = "button"; time.addEventListener("click", () => { pauseAll(); seek(interval.start); });
      const label = el("button", "point-label edit-point", schemeLabel(state.scheme, interval.label)?.name || interval.label); label.type = "button"; I18N.attr(label, "title", "Edit interval label and endpoints");
      label.addEventListener("click", () => { if (!canEdit()) return; if (state.pendingInterval) { showNotice("End or cancel the pending interval first.", "error"); return; } state.editingInterval = index; state.selectedLabel = interval.label; state.selectedStage = interval.stage ?? null; $("interval-edit-start").value = String(interval.start); $("interval-edit-end").value = String(interval.end); pauseAll(); seek(interval.start); renderSelectedLabel(); annotationFeedback("Editing this closed interval. Change its start, end or label, then click Update interval."); });
      const remove = el("button", "remove-point", "×"); remove.type = "button"; I18N.attr(remove, "aria-label", `Remove interval ${timeRange(interval.start, interval.end)}`);
      remove.addEventListener("click", () => { if (!canEdit()) return; remember(); state.intervals.splice(index, 1); state.editingInterval = null; changed(); annotationFeedback(defaultNormal() ? "Exception removed. This region returns to default Nominal; whole-trajectory review must be confirmed again." : "Interval removed. Its time range is unlabelled unless covered by another explicit interval."); });
      row.append(time, label, remove); if (interval.stage != null) row.append(el("span", "point-stage", stageName(interval.stage))); $("point-list").append(row);
    });
    $("own-timeline").replaceChildren();
    if (state.episode && state.scheme) $("own-timeline").append(makeTimeline(state.episode, state.scheme, [ownTrack()], [], true));
    renderCoverage(); renderOwnResult(); updateControls();
  }
  function remember() { const outcomes = window.GRANA_LAB_INTERACT ? window.GRANA_LAB_INTERACT.outcomeList() : []; state.history.push({outcomes, points: state.points.map((point) => ({...point})), interpolation: state.interpolation, intervals: cloneIntervals(state.intervals), pendingInterval: state.pendingInterval ? {...state.pendingInterval} : null, stageIntervals: cloneIntervals(state.stageIntervals), pendingStageInterval: state.pendingStageInterval ? {...state.pendingStageInterval} : null, selectedSemanticStage: state.selectedSemanticStage, legacy: state.legacy, annotationFormat: state.annotationFormat, reviewConfirmed: state.reviewConfirmed, migratedFromRevision: state.migratedFromRevision, recordStatus: state.record?.status}); if (state.history.length > 100) state.history.shift(); }
  function changed() { state.dirty = true; if (defaultNormal()) state.reviewConfirmed = false; markInteraction(); if (isCurve()) scheduleCurvePreview(); renderPoints(); }
  function addPoint() {
    if (isCurve()) {
      if (!canEdit()) return;
      const timeInput = $("curve-time").value.trim(), valueInput = $("curve-value").value.trim();
      let time = Number(timeInput); const value = Number(valueInput), duration = state.episode.duration_seconds;
      if (!timeInput || !valueInput || !Number.isFinite(time) || !Number.isFinite(value)) { showNotice("Enter a time and a reward value.", "error"); return; }
      if (Math.abs(time - duration) < 1e-6) time = duration;
      if (time < 0 || time > duration || value < 0 || value > scoreMax()) { showNotice(`Time must be 0–${duration} s and reward must be 0–${scoreMax()}.`, "error"); return; }
      const existing = state.points.findIndex((candidate) => Math.abs(candidate.time - time) < 1e-6);
      if (existing >= 0 && state.points[existing].time !== state.editingTime && !window.confirm(I18N.t("An anchor already exists at this time. Replace it with this reward and step setting?"))) return;
      remember();
      if (state.editingTime !== null) state.points = state.points.filter((point) => point.time !== state.editingTime);
      state.points = state.points.filter((point) => Math.abs(point.time - time) >= 1e-6);
      state.points.push({time, value, step: $("curve-step").checked}); state.points.sort((a, b) => a.time - b.time);
      state.editingTime = null; changed(); pauseAll(); seek(time); showNotice("");
      annotationFeedback(`${existing >= 0 ? "Updated" : "Added"} anchor: ${time.toFixed(2)} s → Reward ${value}${$("curve-step").checked ? " · Step" : ""} · unsaved.`); return;
    }
    if (!canEdit()) return;
    try {
      if (state.editingInterval !== null) {
        const start = boundaryTime($("interval-edit-start").value), end = boundaryTime($("interval-edit-end").value);
        const interval = {start, end, label: state.selectedLabel, stage: requiresExceptionStage() ? state.selectedStage : null};
        validateClosedInterval(interval, state.editingInterval); remember(); state.intervals[state.editingInterval] = interval; state.intervals.sort((a, b) => a.start - b.start); state.editingInterval = null; changed(); showNotice(""); annotationFeedback(`Updated interval ${timeRange(start, end)} · unsaved.`); return;
      }
      const time = boundaryTime($("interval-time").value);
      if (state.pendingInterval) {
        const interval = {...state.pendingInterval, end: time}; validateClosedInterval(interval); remember(); state.intervals.push(interval); state.intervals.sort((a, b) => a.start - b.start); state.pendingInterval = null; pauseAll(); seek(time); changed(); showNotice(""); annotationFeedback(`Closed ${schemeLabel(state.scheme, interval.label)?.name || interval.label}: ${timeRange(interval.start, interval.end)}. ${defaultNormal() ? "Confirm the review again" : "Time outside intervals stays unlabelled"} · unsaved.`);
      } else {
        if (!state.selectedLabel || requiresExceptionStage() && state.selectedStage === null) throw new Error((requiresExceptionStage() ? "Pick a subtask and High / Mid / Low first." : independentStages() ? "Pick Suboptimal or Regressed first · for a subtask, pick it and press Shift+S." : "Pick Suboptimal or Regressed first."));
        if (defaultNormal() && state.selectedLabel === "normal") throw new Error("Nominal is the default background. Mark only Suboptimal or Regressed exceptions.");
        if (time >= state.episode.duration_seconds) throw new Error("Start before the video end so the interval can have positive duration.");
        if (state.intervals.some((interval) => interval.start <= time && time < interval.end)) throw new Error("This start time is already inside a closed interval. Edit or remove that interval first.");
        remember(); state.pendingInterval = {start: time, label: state.selectedLabel, stage: requiresExceptionStage() ? state.selectedStage : null}; pauseAll(); seek(time); changed(); showNotice(""); annotationFeedback(`Started at ${time.toFixed(2)} s; now mark its end (press E).${defaultNormal() ? "" : " Nothing counts as covered yet."}`);
      }
    } catch (error) { showNotice(error.message, "error"); }
  }
  function boundaryTime(raw) {
    if (String(raw).trim() === "") throw new Error("Enter an explicit boundary time.");
    let time = Number(raw); const duration = state.episode.duration_seconds;
    if (!Number.isFinite(time) || time < 0 || time > duration + 1e-6) throw new Error(`Boundary time must be 0–${duration} s.`);
    time = Math.round(time * 1e6) / 1e6;
    return Math.abs(time - duration) <= 1e-6 ? duration : time;
  }
  function validateClosedInterval(interval, skipIndex = null) {
    if (!(interval.end > interval.start)) throw new Error("End time must be later than start time. The pending interval is unchanged.");
    if (!interval.label || requiresExceptionStage() && interval.stage === null) throw new Error((requiresExceptionStage() ? "Choose a label and subtask." : "Choose a label."));
    if (defaultNormal() && interval.label === "normal") throw new Error("Only Suboptimal / Regressed are explicit exceptions; remove an exception to restore default Nominal.");
    if (state.intervals.some((existing, index) => index !== skipIndex && interval.start < existing.end && interval.end > existing.start)) throw new Error("This interval overlaps an existing closed interval. Choose a non-overlapping end or edit the existing interval.");
  }
  function validateStageInterval(interval, skipIndex = null) {
    if (!(interval.end > interval.start)) throw new Error("Subtask end must be after its start.");
    if (!Number.isInteger(interval.stage) || interval.stage < 0 || interval.stage >= state.episode.stages.length) throw new Error("Choose a subtask.");
    if (state.stageIntervals.some((existing, index) => index !== skipIndex && interval.start < existing.end && interval.end > existing.start)) throw new Error("Subtask intervals cannot overlap.");
  }
  function addStageInterval() {
    if (!canEdit() || !independentStages()) return;
    try {
      if (state.editingStageInterval !== null) {
        const interval = {start: boundaryTime($("stage-edit-start").value), end: boundaryTime($("stage-edit-end").value), stage: state.selectedSemanticStage};
        validateStageInterval(interval, state.editingStageInterval); remember(); state.stageIntervals[state.editingStageInterval] = interval; state.stageIntervals.sort((a, b) => a.start - b.start); state.editingStageInterval = null; changed(); annotationFeedback(`Updated subtask ${timeRange(interval.start, interval.end)}. Confirm the review again.`); return;
      }
      const time = boundaryTime($("stage-boundary-time").value);
      if (state.pendingStageInterval) {
        const interval = {...state.pendingStageInterval, end: time}; validateStageInterval(interval); remember(); state.stageIntervals.push(interval); state.stageIntervals.sort((a, b) => a.start - b.start); state.pendingStageInterval = null; pauseAll(); seek(time); changed(); annotationFeedback(`Closed subtask S${interval.stage + 1}: ${timeRange(interval.start, interval.end)}. Fill the remaining subtask gaps.`);
      } else {
        if (!Number.isInteger(state.selectedSemanticStage)) throw new Error("Choose a subtask before starting.");
        if (time >= state.episode.duration_seconds) throw new Error("Start the subtask before the clip end.");
        if (state.stageIntervals.some((interval) => interval.start <= time && time < interval.end)) throw new Error("This time already has a subtask. Edit or remove it first.");
        remember(); state.pendingStageInterval = {start: time, stage: state.selectedSemanticStage}; pauseAll(); seek(time); changed(); annotationFeedback(`Subtask S${state.selectedSemanticStage + 1} started at ${time.toFixed(2)} s; mark its end to count it.`);
      }
    } catch (error) { showNotice(error.message, "error"); }
  }
  async function saveRecord(status = "draft") {
    if (!state.session.annotator_id || !state.episode || !state.scheme) return;
    if (state.pendingInterval || state.pendingStageInterval) throw new Error("End or cancel the pending span before saving or leaving this episode. Your edits are still here.");
    if (state.saveConflict) throw new Error("Resolve the save conflict first: export your unsaved labels, then reload the saved labels to compare and reapply changes.");
    if (state.legacy) throw new Error(state.scheme?.id === "stage_aq" ? "This earlier Subtask + AQ record is read-only. Confirm Enable independent subtasks before saving." : isAQ() ? "This earlier AQ record is read-only. Confirm Use default Nominal explicitly before saving under the new protocol." : "Legacy records are read-only. Confirm the explicit start/end conversion before saving.");
    if (isSubmitted() && status !== "draft") return;
    if (defaultNormal() && isSubmitted() && status === "draft") state.reviewConfirmed = false;
    if (isCurve() && status === "submitted" && (state.points.length < 2 || state.points[0].time !== 0 || Math.abs(state.points[state.points.length - 1].time - state.episode.duration_seconds) > 1e-6)) throw new Error("Before submitting a reward curve, explicitly add anchors at 0 s and the exact trajectory endpoint. The Start and End buttons set those times; you must supply each reward.");
    if (!isCurve() && status === "submitted" && (state.pendingInterval || (defaultNormal() ? !state.reviewConfirmed : !state.intervals.length))) throw new Error(defaultNormal() ? "End or cancel any pending exception and confirm that you reviewed the full trajectory. Zero exceptions are allowed." : "Submit at least one closed interval and end or cancel the pending interval first. Unlabelled gaps are allowed.");
    if (independentStages() && status === "submitted" && (state.pendingStageInterval || !stageCoverage().complete)) throw new Error("Finish or cancel the pending subtask and cover the whole clip with subtasks first.");
    const body = {annotator_id: state.session.annotator_id, revision: state.record?.revision ?? 0, status, active_seconds: Math.round(state.activeSeconds * 1000) / 1000};
    if (isCurve()) { body.points = state.points.map((point) => ({...point})); body.interpolation = state.interpolation; }
    else { body.annotation_format = state.annotationFormat || "intervals-v1"; body.intervals = isSubtaskOnly() ? [] : cloneIntervals(state.intervals); body.pending_interval = !isSubtaskOnly() && state.pendingInterval ? {...state.pendingInterval} : null; if (defaultNormal()) { body.default_label = "normal"; body.review_confirmed = state.reviewConfirmed; } if (independentStages()) { body.stage_intervals = cloneIntervals(state.stageIntervals); body.pending_stage_interval = state.pendingStageInterval ? {...state.pendingStageInterval} : null; } if (state.migratedFromRevision !== null) body.migrated_from_revision = state.migratedFromRevision; }
    let data;
    try { data = await api(ownPath(), {method: "PUT", body: JSON.stringify(body)}); }
    catch (error) { if (error.conflict) { state.saveConflict = true; updateControls(); } throw error; }
    state.record = data.record;
    state.points = (data.record.points || []).map((point) => ({...point}));
    state.intervals = cloneIntervals(data.record.intervals); state.pendingInterval = data.record.pending_interval ? {...data.record.pending_interval} : null;
    state.stageIntervals = cloneIntervals(data.record.stage_intervals); state.pendingStageInterval = data.record.pending_stage_interval ? {...data.record.pending_stage_interval} : null; state.editingStageInterval = null;
    state.annotationFormat = isCurve() ? null : data.record.annotation_format || "legacy-points"; state.reviewConfirmed = data.record.review_confirmed === true;
    state.migratedFromRevision = null; state.editingInterval = null; state.history = [];
    state.activeSeconds = data.record.active_seconds || 0;
    if (isCurve()) { state.interpolation = data.record.interpolation || "pchip"; $("curve-interpolation").value = state.interpolation; scheduleCurvePreview(true); }
    state.dirty = false;
    renderPoints();
    if (status === "submitted") renderOwnResult(true);
    annotationFeedback(status === "submitted" ? "Submitted." : "Draft saved; grana build-cache uses submitted episodes only.");
  }
  async function saveIfDirty() {
    if (state.dirty && !isSubmitted() && !state.legacy) await saveRecord("draft");
  }
  async function loadContext(episodeId, schemeId) {
    pauseAll();
    annotationFeedback("");
    invalidateCurve();
    const episode = state.episodes.find((item) => item.id === episodeId) || null;
    const scheme = state.schemes.find((item) => item.id === schemeId) || state.schemes[0] || null;
    let record = null;
    // Resolve the new record before replacing the current editor: failed loads preserve its data.
    if (episode && scheme && state.session.annotator_id) record = (await api(`${ownPath(episode, scheme)}?annotator_id=${encodeURIComponent(state.session.annotator_id)}`)).record;
    if (record && record.annotator_id !== state.session.annotator_id) throw new Error("Your annotator session changed in another tab. Sign in again here before loading or saving annotations.");
    const mediaChanged = episode?.id !== state.episode?.id || !state.videos.length;
    state.contextSerial += 1; if (episode?.id !== state.episode?.id) state.trackZoom = 1;   // another episode opens fitted
    state.episode = episode; state.scheme = scheme; state.record = record; state.saveConflict = false; state.sessionConflict = false;
    state.points = (record?.points || []).map((point) => ({...point}));
    state.intervals = cloneIntervals(record?.intervals); state.pendingInterval = record?.pending_interval ? {...record.pending_interval} : null;
    state.annotationFormat = isCurve(scheme) ? null : record ? record.annotation_format || "legacy-points" : preferredFormat(scheme);
    state.legacy = Boolean(record && !isCurve(scheme) && record.annotation_format !== preferredFormat(scheme));
    state.stageIntervals = cloneIntervals(record?.stage_intervals); state.pendingStageInterval = record?.pending_stage_interval ? {...record.pending_stage_interval} : null; state.editingStageInterval = null; state.selectedSemanticStage = state.pendingStageInterval?.stage ?? null; state.stageTimePinned = false;
    $("stage-boundary-time").value = String(currentTime());
    state.reviewConfirmed = record?.review_confirmed === true;
    state.migratedFromRevision = null; state.editingInterval = null; state.intervalTimePinned = false;
    $("interval-time").value = String(currentTime());
    state.interpolation = record?.interpolation || "pchip"; state.editingTime = null; state.curveTimePinned = false;
    $("curve-value").value = ""; $("curve-step").checked = false; $("curve-time").value = String(currentTime());
    state.activeSeconds = record?.active_seconds || 0;
    state.selectedLabel = state.pendingInterval?.label || null; state.selectedStage = state.pendingInterval?.stage ?? null; state.history = []; state.dirty = false;
    state.lastInteraction = Date.now(); state.lastTick = Date.now();
    $("episode-select").value = episode?.id || ""; $("scheme-select").value = scheme?.id || "";
    $("annotate-empty").hidden = Boolean(episode && (state.session.annotator_id || canManage()));
    $("annotation-workspace").hidden = !episode || !(state.session.annotator_id || canManage());
    renderProtocol(); if (mediaChanged) renderVideos(); renderPoints(); if (isCurve()) scheduleCurvePreview(true);
  }
  async function changeContext(episodeId, schemeId) {
    $("episode-select").value = state.episode?.id || ""; $("scheme-select").value = state.scheme?.id || "";
    await withBusy(async () => { await saveIfDirty(); await loadContext(episodeId, schemeId); showNotice(""); });
  }
  function setPanel(name) {
    if (name !== "annotate" && !canManage()) return;
    state.panel = name;
    if (name !== "annotate") pauseAll();
    document.querySelectorAll(".workspace-panel").forEach((node) => { node.hidden = node.id !== `panel-${name}`; });
    document.querySelectorAll(".workspace-nav button").forEach((button) => { const active = button.dataset.panel === name; button.classList.toggle("active", active); if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current"); });
    updateWorkspaceNote();
    if (name === "agreement" && state.episode) $("report-episode").value = state.episode.id;
  }
  function updateWorkspaceNote() {
    I18N.text($("workspace-note"), state.panel === "annotate" ? "One shared label file per episode." : protectedStudy() ? "Researcher-only · Submitted records drive metrics." : "One label file per episode · submitted episodes go to grana build-cache.");
  }
  async function navigate(name) { await withBusy(async () => { await saveIfDirty(); setPanel(name); showNotice(""); }); }
  async function refreshBootstrap() {
    const data = await api("/api/bootstrap");
    state.schemes = data.schemes; state.episodes = data.episodes; state.session = data.session;
    state.access = data.access || {requires_reviewer_key: true, can_manage: Boolean(data.session.reviewer)};
    renderSession(); renderSelectors();
  }
  function coverageTable(report) {
    const card = el("section", "card coverage-card"); card.append(el("h2", "", "Submission coverage"));
    const coverage = report.coverage || [];
    const annotators = [...new Set(coverage.map((entry) => entry.annotator_id).concat(Object.values(report.schemes || {}).flatMap((scheme) => scheme.annotator_ids || [])))].sort();
    if (!annotators.length) { card.append(el("p", "small muted", "No annotations have been saved for this trajectory yet.")); return card; }
    const scroll = el("div", "table-scroll"), table = el("table", "coverage-table"), head = el("thead"), headRow = el("tr");
    headRow.append(el("th", "", "Annotator")); state.schemes.forEach((scheme) => headRow.append(el("th", "", scheme.name))); head.append(headRow); table.append(head);
    const body = el("tbody");
    annotators.forEach((id) => {
      const row = el("tr"); row.append(el("td", "mono", id));
      state.schemes.forEach((scheme) => {
        const entry = coverage.find((item) => item.annotator_id === id && item.scheme_id === scheme.id);
        const submitted = (report.schemes[scheme.id]?.annotator_ids || []).includes(id);
        const status = entry?.status || (submitted ? "submitted" : "missing");
        const cell = el("td"); cell.append(el("span", `status-chip ${status === "submitted" ? "submitted" : ""}`, status === "submitted" ? "Submitted" : status === "draft" ? "Draft" : "Missing")); row.append(cell);
      });
      body.append(row);
    });
    table.append(body); scroll.append(table); card.append(scroll, el("p", "small muted", "Missing and draft records are excluded. Direct AQ comparison requires the same set of at least two submitted annotators in both AQ protocols."));
    return card;
  }
  function metricItem(label, value, suffix = "") {
    const item = el("div", "metric-item"); item.append(el("dt", "", label)); const valueNode = el("dd", "", value); if (suffix) valueNode.append(el("small", "", suffix)); item.append(valueNode); return item;
  }
  function renderReport(report, episode) {
    const container = $("report-content"); container.replaceChildren();
    container.append(coverageTable(report));
    state.schemes.forEach((scheme) => {
      const metric = report.schemes?.[scheme.id];
      if (!metric) return;
      const card = el("section", "card report-card");
      const heading = el("div", "card-heading"), headingText = el("div");
      headingText.append(el("h2", "", scheme.name), el("p", "protocol-provenance", ((window.AGREEMENT_STATIC.copy || {})[scheme.id] || {}).line || scheme.provenance || scheme.description));
      heading.append(headingText, el("span", "status-chip", `${metric.n_annotators} submitted annotator${metric.n_annotators === 1 ? "" : "s"}`)); card.append(heading);
      const grid = el("dl", "metric-grid");
      if (isCurve(scheme)) {
        const continuous = metric.continuous || {};
        grid.append(metricItem("Absolute agreement · ICC(A,1)", formatMetric(continuous.icc_a1)), metricItem("Mean pairwise MAE", formatMetric(continuous.mean_pairwise_mae)), metricItem("Mean pairwise RMSE", formatMetric(continuous.mean_pairwise_rmse)), metricItem("Range-normalized MAE", continuous.normalized_mae == null ? "—" : `${(continuous.normalized_mae * 100).toFixed(2)}%`));
        card.append(grid);
        if (metric.tracks?.length) {
          const wrap = el("div", "report-timeline-wrap"); wrap.append(makeCurvePlot(episode, scheme, metric.tracks)); card.append(wrap);
          const legend = el("div", "report-legend");
          metric.tracks.forEach((track, index) => { const item = el("span"), dot = el("i"); dot.style.background = annotatorColors[index % annotatorColors.length]; item.append(dot, I18N.node(`${track.annotator_id} · ${(track.interpolation || "pchip").toUpperCase()}`)); legend.append(item); });
          card.append(legend, el("p", "small muted", "Each curve is one independent submitted record. Circles are ordinary anchors; squares are explicit step points. Click the chart to open your own record at that time."));
        } else card.append(el("p", "small muted", "No submitted reward curves yet. Explicit start and end anchors are required for submission."));
        if (continuous.pairwise?.length) {
          const details = el("details", "curve-pairwise-details"); details.append(el("summary", "", "Pairwise curve differences"));
          const table = el("table", "coverage-table"), head = el("thead"), row = el("tr"); ["Annotator pair", "MAE", "RMSE"].forEach((title) => row.append(el("th", "", title))); head.append(row); table.append(head);
          const body = el("tbody"); continuous.pairwise.forEach((pair) => { const item = el("tr"); item.append(el("td", "mono", `${pair.annotator_a} ↔ ${pair.annotator_b}`), el("td", "mono", formatMetric(pair.mae)), el("td", "mono", formatMetric(pair.rmse))); body.append(item); }); table.append(body); details.append(table); card.append(details);
        }
        card.append(el("p", "small muted", `Continuous reward uses its native 0–${continuous.score_max ?? scoreMax(episode)} range. Lower MAE/RMSE means closer curves. ICC and categorical α measure different forms of agreement and must not be ranked against each other. A reward decrease is not automatically classified as failure.`));
        container.append(card); return;
      }
      if (metric.abnormal_overlap) {
        const overlap = metric.abnormal_overlap, exceptionGrid = el("dl", "metric-grid abnormal-metric-grid");
        card.append(el("h3", "", "Abnormal-interval overlap"));
        exceptionGrid.append(metricItem("Any abnormal · mean IoU", formatMetric(overlap.any_abnormal?.mean_iou)), metricItem("Suboptimal · mean IoU", formatMetric(overlap.by_label?.suboptimal?.mean_iou)), metricItem("Failure · mean IoU", formatMetric(overlap.by_label?.failure?.mean_iou)), metricItem("Any abnormal · positive F1", formatMetric(overlap.any_abnormal?.mean_positive_f1)));
        card.append(exceptionGrid, el("p", "small muted", "Duration-based overlap on jointly labelled time. When both annotators mark no positive interval, IoU/F1 is undefined (—), not perfect agreement."));
      }
      grid.append(metricItem("Krippendorff α (categorical)", formatMetric(metric.nominal_alpha)), metricItem("Overall label agreement", metric.observed_agreement == null ? "—" : `${(metric.observed_agreement * 100).toFixed(1)}%`));
      if (metric.failure_onset) grid.append(metricItem("Failure-onset pairwise F1", formatMetric(metric.failure_onset.f1)), metricItem("Matched boundary error", formatMetric(metric.failure_onset.median_boundary_error_seconds, 2), "s"));
      else grid.append(metricItem("Subtask α (categorical)", formatMetric(metric.stage_agreement?.nominal_alpha)), metricItem("Failure-onset comparison", "N/A"));
      card.append(grid);
      if (metric.default_normal_annotators?.length) card.append(el("p", "abnormal-metric-note", `Note: these overall scores include the default-Nominal background that nobody labelled by hand; read them together with the interval-overlap scores above.`));
      if (metric.support) card.append(el("p", "small muted event-support", `Agreement support: ${metric.support.pairable_seconds?.toFixed(2) ?? "—"} s (${((metric.support.pairable_fraction || 0) * 100).toFixed(1)}%) with labels from at least two annotators. Unlabelled samples are missing, not an agreement category.`));
      if (metric.failure_onset) {
        const events = metric.failure_onset;
        card.append(el("p", "small muted event-support", `${events.matched_pairs ?? 0} matched event pairs · ${events.unmatched_events ?? 0} unmatched events across annotator pairs. Boundary error uses matched events only.`));
        if (events.onsets?.length) card.append(el("p", "small muted event-support", `Failure onsets per annotator: ${events.onsets.map((item) => `${item.annotator_id}: ${item.times.length}`).join(" · ")}. An initial failure at 0 s is a state, not an observed onset.`));
      }
      if (metric.tracks?.length && scheme.id !== "subtask_only") {
        const wrap = el("div", "report-timeline-wrap"); wrap.append(makeTimeline(episode, scheme, metric.tracks, metric.disagreement_intervals || [])); card.append(wrap);
        const legend = el("div", "report-legend"); scheme.labels.forEach((label) => { const item = el("span"); const dot = el("i"); dot.style.background = labelColor(scheme, label.id); item.append(dot, I18N.node(label.name)); legend.append(item); });
        if (metric.default_normal_annotators?.length) { const item = el("span"), dot = el("i"); dot.style.background = "#22503f"; item.append(dot, I18N.node("Default Nominal · not per-frame manual")); legend.append(item); }
        const mismatch = el("span"); const dot = el("i"); dot.style.background = "#3a1f2b"; dot.style.border = "1px dashed #ec5b52"; mismatch.append(dot, I18N.node("Annotator disagreement")); legend.append(mismatch); card.append(legend);
        card.append(el("p", "small muted", "Click an interval to open this trajectory and protocol in the annotation workspace. Each row is an independent submitted record."));
      } else card.append(el("p", "small muted", "No submitted records yet. Save and submit independent annotations to populate this panel."));
      if ((scheme.id === "stage_aq" || scheme.id === "subtask_only") && metric.stage_agreement) {
        const stageSection = el("section", "stage-report-section"); stageSection.append(el("h3", "", "Subtask agreement"));
        const metrics = el("dl", "metric-grid"); metrics.append(metricItem("Subtask α (categorical)", formatMetric(metric.stage_agreement.nominal_alpha)), metricItem("Observed subtask agreement", metric.stage_agreement.observed_agreement == null ? "—" : `${(metric.stage_agreement.observed_agreement * 100).toFixed(1)}%`), metricItem("Subtask pairable coverage", `${((metric.stage_agreement.support?.pairable_fraction || 0) * 100).toFixed(1)}%`), metricItem("Joint (quality, subtask) α", formatMetric(metric.joint_agreement?.nominal_alpha))); stageSection.append(metrics);
        if (metric.tracks?.length) { const wrap = el("div", "report-timeline-wrap"); wrap.append(makeStageTimeline(episode, metric.tracks)); stageSection.append(wrap); }
        stageSection.append(el("p", "small muted", "Full subtask tracks; earlier records show only their recorded spans.")); card.append(stageSection);
      }
      if (scheme.id === "stage_score") card.append(el("p", "small muted", "Subtask-outcome scores measure a different construct; High / Mid / Low are not mapped to action quality."));
      container.append(card);
    });
    const comparison = report.comparison || {};
    const note = el("section", "comparison-note"); note.append(el("h3", "", "Controlled comparison: AQ only ↔ Subtask + AQ"));
    note.append(el("p", "", comparison.reason || (comparison.ready ? "Both protocols use the same AQ vocabulary and the same submitted annotator set. Positive deltas indicate higher agreement for Subtask + AQ on this trajectory." : "Complete both AQ protocols with the same set of at least two independent annotators.")));
    if (comparison.ready) {
      const values = el("div", "comparison-values");
      [["Δ Krippendorff α", comparison.alpha_delta], ["Δ failure-onset F1", comparison.failure_f1_delta]].forEach(([label, value]) => { const item = el("span"); item.append(el("strong", "mono", value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(3)}`), I18N.node(label)); values.append(item); }); note.append(values);
      note.append(el("p", "", "Deltas are descriptive for this trajectory; they are not evidence of statistical significance across a study."));
    }
    container.append(note);
    const notes = el("div", "report-notes");
    notes.append(el("p", "", `Time grid: ${report.grid_hz || 10} Hz. Failure-onset matching tolerance: ±${report.tolerance_seconds ?? 1} s. “—” means the metric is undefined, including insufficient annotators or no label variation; it does not mean zero.`));
    (Array.isArray(report.notes) ? report.notes : report.notes ? [report.notes] : []).forEach((text) => notes.append(el("p", "", text)));
    container.append(notes);
    const encodedId = encodeURIComponent(episode.id);
    $("export-report").href = `/api/reports/${encodedId}/export`; $("export-report").hidden = false;
    $("export-records").href = `/api/records/${encodedId}/export`; $("export-records").hidden = false;
  }
  async function fetchReport() {
    const episode = state.episodes.find((item) => item.id === $("report-episode").value);
    if (!episode) { showNotice("Choose a trajectory before viewing agreement.", "info"); return; }
    await withBusy(async () => {
      $("refresh-report").disabled = true;
      try {
        const report = await api(`/api/reports/${encodeURIComponent(episode.id)}`);
        state.reportEpisode = episode; renderReport(report, episode); showNotice("");
      } finally { $("refresh-report").disabled = false; }
    });
  }

  $("episode-select").addEventListener("change", (event) => changeContext(event.target.value, state.scheme?.id));
  $("scheme-select").addEventListener("change", (event) => changeContext(state.episode?.id, event.target.value));
  $("stage-select").addEventListener("change", (event) => { if (state.pendingInterval) return; state.selectedStage = event.target.value === "" ? null : Number(event.target.value); markInteraction(); updateControls(); });
  $("independent-stage-select").addEventListener("change", (event) => { if (!canEdit() || state.pendingStageInterval) return; state.selectedSemanticStage = event.target.value === "" ? null : Number(event.target.value); markInteraction(); updateControls(); });
  $("stage-boundary-time").addEventListener("input", () => { state.stageTimePinned = true; markInteraction(); });
  $("stage-use-time").addEventListener("click", () => { if (!canEdit()) return; state.stageTimePinned = false; $("stage-boundary-time").value = String(currentTime()); markInteraction(); });
  $("stage-video-end").addEventListener("click", () => { if (!canEdit()) return; pauseAll(); seek(state.episode.duration_seconds); state.stageTimePinned = true; $("stage-boundary-time").value = String(state.episode.duration_seconds); markInteraction(); });
  const stageStartButton = el("button", "text-button", "Start · 0 s"); stageStartButton.id = "stage-video-start"; stageStartButton.type = "button"; $("stage-time-tools").prepend(stageStartButton);
  stageStartButton.addEventListener("click", () => { if (!canEdit()) return; pauseAll(); seek(0); state.stageTimePinned = true; $("stage-boundary-time").value = "0"; markInteraction(); });
  $("add-stage-interval").addEventListener("click", addStageInterval);
  $("cancel-pending-stage").addEventListener("click", () => { if (!canEdit() || !state.pendingStageInterval) return; remember(); state.pendingStageInterval = null; changed(); annotationFeedback("Pending subtask cancelled; Undo restores it."); });
  $("cancel-stage-edit").addEventListener("click", () => { state.editingStageInterval = null; updateControls(); annotationFeedback("Subtask edit cancelled."); });
  $("play-button").addEventListener("click", playPause);
  $("video-seek").addEventListener("input", (event) => seek(event.target.value / (state.episode?.fps || 30)));
  $("previous-frame").addEventListener("click", () => { pauseAll(); seek(currentTime() - 1 / (state.episode?.fps || 30)); });
  $("next-frame").addEventListener("click", () => { pauseAll(); seek(currentTime() + 1 / (state.episode?.fps || 30)); });
  $("playback-rate").addEventListener("change", (event) => { state.videos.forEach((video) => { video.defaultPlaybackRate = video.playbackRate = Number(event.target.value); }); markInteraction(); });
  $("add-point").addEventListener("click", addPoint);
  $("undo-button").addEventListener("click", () => { if (!canEdit() || !state.history.length) return; const previous = state.history.pop(); state.points = previous.points; state.interpolation = previous.interpolation; state.intervals = previous.intervals; state.pendingInterval = previous.pendingInterval; state.stageIntervals = previous.stageIntervals; state.pendingStageInterval = previous.pendingStageInterval; state.selectedSemanticStage = state.pendingStageInterval?.stage ?? previous.selectedSemanticStage; state.editingStageInterval = null; state.legacy = previous.legacy; state.annotationFormat = previous.annotationFormat; state.reviewConfirmed = previous.reviewConfirmed; if (previous.outcomes) window.GRANA_LAB_API.outcomes.set(state.episode.id, state.session.annotator_id, state.scheme.id, previous.outcomes); state.migratedFromRevision = previous.migratedFromRevision; if (state.record && previous.recordStatus) state.record = {...state.record, status: previous.recordStatus}; state.editingTime = null; state.editingInterval = null; if (state.pendingInterval) { state.selectedLabel = state.pendingInterval.label; state.selectedStage = state.pendingInterval.stage ?? null; } $("curve-interpolation").value = state.interpolation; if (state.legacy) { state.dirty = false; renderPoints(); } else changed(); renderProtocol(); annotationFeedback(state.legacy ? "Conversion undone; the legacy record is read-only again." : "Undone · unsaved."); });
  $("clear-button").addEventListener("click", () => { if (!canEdit() || !window.confirm(I18N.t(independentStages() ? "Reset both tracks? All subtask intervals and quality spans are cleared; Undo is available until saving." : defaultNormal() ? "Clear all explicit exceptions and pending annotation? The entire trajectory returns to default Nominal and review confirmation will reset. You can undo until saving." : "Clear all annotations and any pending interval in this draft? You can undo until saving."))) return; remember(); if (independentStages()) window.GRANA_LAB_API.outcomes.set(state.episode.id, state.session.annotator_id, state.scheme.id, []); state.points = []; state.intervals = []; state.pendingInterval = null; state.stageIntervals = []; state.pendingStageInterval = null; state.editingStageInterval = null; state.editingTime = null; state.editingInterval = null; changed(); annotationFeedback(independentStages() ? "Both tracks reset; Undo restores them. Unsaved." : defaultNormal() ? "Spans cleared; confirm the review again. Undo restores them; unsaved." : "Draft cleared; Undo restores it. Unsaved."); });
  $("interval-time").addEventListener("input", () => { state.intervalTimePinned = true; markInteraction(); });
  $("interval-use-time").addEventListener("click", () => { if (!canEdit()) return; state.intervalTimePinned = false; $("interval-time").value = String(currentTime()); markInteraction(); });
  $("interval-video-end").addEventListener("click", () => { if (!canEdit()) return; pauseAll(); seek(state.episode.duration_seconds); state.intervalTimePinned = true; $("interval-time").value = String(state.episode.duration_seconds); markInteraction(); });
  $("interval-cancel-pending").addEventListener("click", () => { if (!canEdit() || !state.pendingInterval) return; remember(); state.pendingInterval = null; changed(); annotationFeedback(defaultNormal() ? "Pending span cancelled; Undo restores it." : "Pending interval cancelled; Undo restores it."); });
  $("interval-cancel-edit").addEventListener("click", () => { state.editingInterval = null; updateControls(); annotationFeedback("Edit cancelled."); });
  $("legacy-convert").addEventListener("click", () => {
    if (state.busy || !state.legacy || !state.session.annotator_id) return;
    if (state.scheme.id === "stage_aq") {
      if (!window.confirm(I18N.t("Enable independent semantic stages? Preserve only genuinely annotated old stage spans (including old hold-forward spans), and keep Suboptimal/Regressed intervals as independent AQ exceptions. Existing default-Nominal background has NO stage and will not generate one. Previously unlabelled AQ gaps become default Nominal only after this confirmation. Old pending stage starts remain pending. You must complete the separate stage track from 0 s to the video end and confirm full review before submitting. Historical revisions are retained."))) return;
      remember(); const previousTrack = ownTrack(), spans = trackIntervals(previousTrack, state.episode), oldPending = state.pendingInterval;
      state.stageIntervals = spans.filter((interval) => Number.isInteger(interval.stage)).map(({start, end, stage}) => ({start, end, stage}));
      state.pendingStageInterval = oldPending && Number.isInteger(oldPending.stage) ? {start: oldPending.start, stage: oldPending.stage} : null;
      state.intervals = spans.filter((interval) => interval.label !== "normal").map(({start, end, label}) => ({start, end, label, stage: null}));
      state.pendingInterval = oldPending && oldPending.label !== "normal" ? {start: oldPending.start, label: oldPending.label, stage: null} : null;
      state.migratedFromRevision = state.record.revision; state.record = {...state.record, status: "draft"}; state.annotationFormat = "stage-aq-independent-v1"; state.legacy = false; state.points = []; state.editingInterval = null; state.editingStageInterval = null; state.reviewConfirmed = false;
      state.selectedLabel = state.pendingInterval?.label || null; state.selectedStage = null; state.selectedSemanticStage = state.pendingStageInterval?.stage ?? null;
      changed(); renderProtocol(); annotationFeedback("Independent-subtask draft: legacy subtask spans kept, unknown subtasks left empty, AQ exceptions kept, the rest default Nominal. Cover the clip with subtasks and confirm the review again; unsaved."); return;
    }
    if (isAQ()) {
      if (!window.confirm(I18N.t("Use default Nominal for this existing AQ record? Preserve its Suboptimal/Regressed spans, including the last span of old hold-forward records. All other time—including previously unlabelled gaps and explicit Nominal labels—becomes DEFAULT Normal (not per-frame manual labels). Nominal pending markers are removed; abnormal pending markers are retained. Default background stage is unknown. You must review the full trajectory again before submitting. Historical revisions are retained."))) return;
      remember();
      state.intervals = trackIntervals(ownTrack(), state.episode).filter((interval) => interval.label !== "normal").map(({start, end, label, stage}) => ({start, end, label, stage: stage ?? null}));
      state.pendingInterval = state.pendingInterval && state.pendingInterval.label !== "normal" ? {...state.pendingInterval} : null;
      state.migratedFromRevision = state.record.revision; state.record = {...state.record, status: "draft"}; state.annotationFormat = "aq-default-normal-v1"; state.legacy = false; state.points = []; state.editingInterval = null; state.reviewConfirmed = false;
      state.selectedLabel = state.pendingInterval?.label || null; state.selectedStage = state.pendingInterval?.stage ?? null;
      changed(); renderProtocol(); annotationFeedback("Default-Nominal draft: abnormal spans kept, everything else default. Confirm the review again; unsaved."); return;
    }
    if (!window.confirm(I18N.t("Convert this legacy annotation to explicit start/end? Every point except the last becomes an interval ending at the next point. The LAST point becomes pending: you must choose its end or cancel it. Nothing is automatically extended to the video end. This local conversion is saved only when you save or switch context; historical revisions are retained."))) return;
    remember(); const points = state.points;
    state.intervals = points.slice(0, -1).map((point, index) => ({start: point.time, end: points[index + 1].time, label: point.label, stage: point.stage ?? null}));
    const last = points[points.length - 1]; state.pendingInterval = last ? {start: last.time, label: last.label, stage: last.stage ?? null} : null;
    state.migratedFromRevision = state.record.revision; state.record = {...state.record, status: "draft"}; state.annotationFormat = "intervals-v1"; state.legacy = false; state.points = []; state.editingInterval = null;
    state.selectedLabel = last?.label || null; state.selectedStage = last?.stage ?? null;
    changed(); renderProtocol(); annotationFeedback("Converted to start/end intervals; the last legacy point is now a pending start. End or cancel it; unsaved.");
  });
  $("review-confirmed").addEventListener("change", (event) => { if (!canEdit() || !defaultNormal() || state.pendingInterval || state.pendingStageInterval) return; state.reviewConfirmed = event.target.checked; state.dirty = true; markInteraction(); renderPoints(); annotationFeedback(state.reviewConfirmed ? unresolvedOutcomes() ? "Choose Completed or Failed for every subtask marked Needs review before submitting." : independentStages() && !stageCoverage().complete ? "Review confirmed, but subtasks do not cover the clip yet." : "Review confirmed; you can submit." : "Review unconfirmed; confirm before submitting."); });
  $("curve-time").addEventListener("input", () => { state.curveTimePinned = true; markInteraction(); });
  $("curve-use-time").addEventListener("click", () => { if (!canEdit()) return; state.curveTimePinned = false; $("curve-time").value = String(currentTime()); markInteraction(); });
  $("curve-start-time").addEventListener("click", () => { if (!canEdit()) return; pauseAll(); seek(0); state.curveTimePinned = true; $("curve-time").value = "0"; markInteraction(); });
  $("curve-end-time").addEventListener("click", () => { if (!canEdit()) return; pauseAll(); seek(state.episode.duration_seconds); state.curveTimePinned = true; $("curve-time").value = String(state.episode.duration_seconds); markInteraction(); });
  $("curve-cancel-edit").addEventListener("click", () => { state.editingTime = null; $("curve-value").value = ""; $("curve-step").checked = false; updateControls(); });
  $("curve-interpolation").addEventListener("change", (event) => { if (!canEdit()) return; remember(); state.interpolation = event.target.value; changed(); });
  $("curve-auto-fit").addEventListener("change", () => { scheduleCurvePreview(); renderPoints(); markInteraction(); });
  $("curve-fit").addEventListener("click", () => { scheduleCurvePreview(true); renderPoints(); markInteraction(); });
  $("export-unsaved-labels").addEventListener("click", () => {
    try {
      const api = window.GRANA_LAB_API;
      const body = {stage_intervals: state.stageIntervals, intervals: state.intervals, pending_interval: state.pendingInterval,
        pending_stage_interval: state.pendingStageInterval, status: "draft"};
      api.download(api.draftDocument(state.episode.id, body, state.session.annotator_id), `episode_${state.episode.id}_unsaved.json`);
      showNotice("Exported your unsaved labels. The server's saved labels are unchanged.");
    } catch (error) { showNotice(error.message, "error"); }
  });
  $("reload-saved-labels").addEventListener("click", () => {
    if (!window.confirm(I18N.t("Reload the saved labels and discard your unsaved edits? Export your unsaved labels first if you want to keep them."))) return;
    withBusy(async () => { await loadContext(state.episode.id, state.scheme.id); showNotice("Loaded the saved labels. Compare your export and reapply any changes before saving."); });
  });
  $("save-button").addEventListener("click", () => withBusy(async () => { await saveRecord("draft"); showNotice("Draft saved; grana build-cache uses submitted episodes only."); }));
  $("submit-button").addEventListener("click", () => withBusy(async () => { pauseAll(); await saveRecord("submitted"); showNotice("Submitted. grana build-cache will include this episode; its converted progress curve is shown under the tracks."); }));
  $("reopen-button").addEventListener("click", () => withBusy(async () => { if (!window.confirm(I18N.t("Reopen this episode for editing? It becomes a draft and cannot be used by grana build-cache until it is submitted again."))) return; await saveRecord("draft"); state.lastInteraction = Date.now(); showNotice("Annotation reopened as a draft."); }));
  document.querySelectorAll(".workspace-nav button").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.panel)));
  $("empty-import").addEventListener("click", () => navigate("import"));
  function openAccount() { $("account-error").hidden = true; $("annotator-id").value = ""; $("access-code").value = ""; $("account-dialog").showModal(); }
  $("account-button").addEventListener("click", openAccount); $("empty-signin").addEventListener("click", openAccount);
  $("restore-editing-name").addEventListener("click", () => { openAccount(); $("annotator-id").value = state.session.annotator_id || ""; });
  $("reviewer-button").addEventListener("click", () => { if (!protectedStudy()) return; $("reviewer-error").hidden = true; $("reviewer-key").value = ""; $("reviewer-dialog").showModal(); });
  document.querySelectorAll(".close-dialog").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
  $("account-form").addEventListener("submit", async (event) => {
    event.preventDefault(); if (state.busy) return;
    const button = event.submitter; button.disabled = true;
    let named = false, offline = false;
    const success = await withBusy(async () => {
      const name = $("annotator-id").value.trim(), editingName = state.session.annotator_id;
      const changedSession = editingName && (state.sessionConflict || window.GRANA_LAB_API.session() !== editingName);
      if (changedSession && name === editingName) {
        await api("/api/session", {method: "POST", body: JSON.stringify({annotator_id: name})});
        state.sessionConflict = false; renderSession(); updateControls();
        $("account-dialog").close(); showNotice(`Editing name restored as ${name}. Your unsaved labels and Undo history are unchanged.`);
        return;
      }
      if (changedSession && state.dirty) throw new Error(`Restore the editing name ${editingName} first to keep these unsaved labels. You can change name after saving.`);
      await saveIfDirty().catch((error) => { offline = Boolean(error.offline); throw error; });
      const identity = await api("/api/session", {method: "POST", body: JSON.stringify({annotator_id: $("annotator-id").value.trim(), access_code: $("access-code").value})});
      state.session.annotator_id = identity.annotator_id; named = true;
      clearLoadedRecord(); renderSession();
      await refreshBootstrap(); await loadContext(state.episode?.id || state.episodes[0]?.id, state.scheme?.id || state.schemes[0]?.id);
      $("account-dialog").close(); $("access-code").value = ""; showNotice(`Signed in as ${state.session.annotator_id}.`);
    });
    button.disabled = false;
    // The name was accepted but the episode did not open: its error stays in the main notice, not under the name.
    if (!success && named) { $("account-dialog").close(); $("access-code").value = ""; }
    else if (!success) { I18N.text($("account-error"), offline ? "Cannot reach the grana label server. Try again once it is back." : I18N.source($("notice"))); $("account-error").hidden = false; }
  });
  $("reviewer-form").addEventListener("submit", async (event) => {
    event.preventDefault(); if (state.busy) return;
    const button = event.submitter; button.disabled = true;
    const success = await withBusy(async () => { await saveIfDirty(); await api("/api/reviewer", {method: "POST", body: JSON.stringify({key: $("reviewer-key").value})}); await refreshBootstrap(); await loadContext(state.episode?.id || state.episodes[0]?.id, state.scheme?.id || state.schemes[0]?.id); $("reviewer-dialog").close(); $("reviewer-key").value = ""; showNotice("Researcher tools unlocked. Keep alignment review separate from independent annotation."); });
    button.disabled = false;
    if (!success) { I18N.text($("reviewer-error"), I18N.source($("notice"))); $("reviewer-error").hidden = false; }
  });
  $("logout-button").addEventListener("click", () => withBusy(async () => { await saveIfDirty(); await api("/api/logout", {method: "POST"}); state.session = {}; clearLoadedRecord(); renderSession(); $("annotation-workspace").hidden = true; $("annotate-empty").hidden = false; $("report-content").replaceChildren(); $("export-report").hidden = true; $("export-records").hidden = true; await refreshBootstrap(); await loadContext(state.episode?.id, state.scheme?.id); showNotice("Signed out. Saved labels stay in the recipe's labels directory."); }));
  $("import-videos").addEventListener("change", (event) => { $("selected-videos").textContent = Array.from(event.target.files).map((file) => `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`).join(" · "); });
  $("import-manifest").addEventListener("change", async (event) => {
    try {
      const file = event.target.files[0]; if (!file) return;
      const manifest = JSON.parse(await file.text());
      if (typeof manifest.title !== "string" || typeof manifest.task !== "string" || !Array.isArray(manifest.stages) || !manifest.stages.every((stage) => typeof stage === "string")) throw new Error("Manifest must contain string title, string task, and an array of stage strings.");
      $("import-title").value = manifest.title; $("import-task").value = manifest.task; $("import-stages").value = manifest.stages.join("\n"); showNotice("Manifest fields loaded. Select the matching local video files before importing.");
    } catch (error) { showNotice(`Unable to read manifest: ${error.message}`, "error"); }
  });
  $("import-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await withBusy(async () => {
      $("import-submit").disabled = true; $("import-progress").hidden = false;
      try {
        const body = new FormData(); body.append("title", $("import-title").value.trim()); body.append("task", $("import-task").value.trim()); body.append("stages", JSON.stringify($("import-stages").value.split("\n").map((stage) => stage.trim()).filter(Boolean)));
        Array.from($("import-videos").files).forEach((file) => body.append("videos", file));
        const data = await api("/api/episodes", {method: "POST", body});
        await refreshBootstrap(); await loadContext(data.episode.id, state.scheme?.id || state.schemes[0]?.id);
        $("import-form").reset(); $("selected-videos").textContent = ""; setPanel("annotate"); showNotice("Trajectory ready. Each annotator can now create an independent record under each protocol.");
      } finally { $("import-submit").disabled = false; $("import-progress").hidden = true; }
    });
  });
  $("restore-form").addEventListener("submit", async (event) => {
    event.preventDefault(); const button = event.submitter;
    await withBusy(async () => { button.disabled = true; try { const body = new FormData(); body.append("file", $("restore-file").files[0]); await api("/api/annotations/import", {method: "POST", body}); await loadContext(state.episode?.id, state.scheme?.id); showNotice("Annotation import completed. Open Agreement to inspect submission coverage."); $("restore-form").reset(); } finally { button.disabled = false; } });
  });
  $("refresh-report").addEventListener("click", fetchReport);
  $("report-episode").addEventListener("change", () => { $("export-report").hidden = true; $("export-records").hidden = true; $("report-content").replaceChildren(el("p", "small muted", "Click View agreement to load the selected trajectory.")); });
  document.addEventListener("keydown", (event) => {
    markInteraction();
    if (event.defaultPrevented || state.panel !== "annotate" || !state.episode || document.querySelector("dialog[open]") || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable) return;
    const onButton = event.target.tagName === "BUTTON";   // a focused button keeps Space / Enter; ← → still step frames
    if (event.code === "Space" && !onButton) { event.preventDefault(); playPause(); }
    if (event.code === "KeyP" && !onButton && isCurve() && canEdit()) { event.preventDefault(); pauseAll(); state.curveTimePinned = false; $("curve-time").value = String(currentTime()); $("curve-value").focus(); }
    if (event.code === "ArrowLeft" || event.code === "ArrowRight") { event.preventDefault(); pauseAll(); seek(currentTime() + (event.code === "ArrowLeft" ? -1 : 1) / (state.episode.fps || 30)); }
  });
  // lab-interact.js owns Escape ordering: cancel a held gesture before restoring a camera.
  ["pointerdown", "input"].forEach((name) => document.addEventListener(name, markInteraction, {passive: true}));
  document.addEventListener("visibilitychange", () => { state.lastTick = Date.now(); if (document.hidden) pauseAll(); });
  window.addEventListener("storage", event => {
    if ((event.key === "grana-label.session" || event.key === null) && state.session.annotator_id) {
      state.sessionConflict = window.GRANA_LAB_API.session() !== state.session.annotator_id;
      updateControls();
      if (state.sessionConflict) showNotice("Your name changed in another tab. Restore your editing name to continue; your unsaved labels are still here.", "error");
    }
  });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
  setInterval(() => {
    const now = Date.now(), elapsed = Math.min((now - state.lastTick) / 1000, 2); state.lastTick = now;
    if (!document.hidden && !document.querySelector("dialog[open]") && state.panel === "annotate" && canEdit() && (state.videos[0] && !state.videos[0].paused || now - state.lastInteraction < 60000)) {
      state.activeSeconds += elapsed;   // grana label does not store active time, so counting it never marks the page unsaved
      I18N.text($("active-time"), `${clock(state.activeSeconds)} active`);
      updateControls();
    }
  }, 1000);
  withBusy(async () => { await refreshBootstrap(); await loadContext(state.episodes[0]?.id, (window.AGREEMENT_STATIC && window.AGREEMENT_STATIC.initialScheme) || state.schemes[0]?.id); });
})();
