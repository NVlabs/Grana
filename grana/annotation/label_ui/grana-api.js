/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* grana-api.js — connects the Annotation Lab editor (agreement.js, lab-interact.js) to `grana label`.
 *
 * The Lab sends every request through window.GRANA_LAB_API.handle(method, path, body). Here the requests become
 * calls to the grana label server, and the Lab's Subtask + AQ record is converted to and from one grana label file
 * per episode (native frames, subtasks with a binary outcome, suboptimal/regressed spans):
 *   GET  /api/bootstrap                         one protocol (Subtask + AQ) and the recipe's episodes
 *   POST /api/session · /api/logout             the annotator name, kept in this browser
 *   GET  /api/annotations/<ep>/stage_aq         the episode's label file as a Lab record (null before first save)
 *   PUT  /api/annotations/<ep>/stage_aq         save: draft or submitted (= marked done, used by grana build-cache)
 *   GET  /api/annotations/<ep>/stage_aq/export  the canonical saved label document, including frames, outcomes and provenance
 * Outcomes live in a side store keyed by subtask interval (lab-interact.js keeps it in step with the editor);
 * Completed → high (1), Failed → low (0).
 */
(() => {
  "use strict";
  const I18N = window.GRANA_I18N || {text: (n, s) => { n.textContent = s; }, attr: (n, k, s) => n.setAttribute(k, s), node: s => document.createTextNode(s), source: n => n.textContent, t: s => s};
  const PROTOCOL_VERSION = "agreement-lab-v1";
  const SCHEME = {
    id: "stage_aq", name: "Subtask + AQ", requires_stage: true, independent_stages: true,
    annotation_format: "stage-aq-independent-v1", default_label: "normal",
    description: "Subtask intervals covering the whole clip, an outcome for each subtask, and suboptimal / regressed spans; unmarked time is nominal.",
    provenance: "Grana labels: subtask boundaries, action-quality spans and a binary outcome per subtask.",
    labels: [{id: "normal", name: "Nominal", color: "#30c5a5"}, {id: "suboptimal", name: "Suboptimal", color: "#e3a72f"},
             {id: "failure", name: "Regressed", color: "#ec5b52"}],
  };
  const RUBRIC = [
    "Nominal: the task proceeds normally (a correct recovery counts).",
    "Suboptimal: degraded execution without physical failure, such as a misaligned screw that stays held.",
    "Regressed: physical failure that loses progress, such as dropping the screw.",
    "Unmarked time is Nominal; confirm you reviewed the whole clip before submitting.",
    "Subtasks must cover the whole clip, and each ends with an outcome: Completed or Failed.",
  ].join("\n");
  window.AGREEMENT_STATIC = {
    initialScheme: "stage_aq", schemeOrder: ["stage_aq"], bootstrap: {schemes: [SCHEME]},
    copy: {stage_aq: {rubric: RUBRIC, line: "Subtask intervals with outcomes plus suboptimal / regressed spans.",
                      step: "Subtasks + quality spans; the constructed target is exactly what grana build-cache will train on."}},
  };

  const KEY_SESSION = "grana-label.session", KEY_LAST = "grana-label.last";
  const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;
  const store = (() => { try { localStorage.setItem("__grana_probe", "1"); localStorage.removeItem("__grana_probe"); return localStorage; } catch (_) { const mem = {}; return {getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); }, removeItem: (k) => { delete mem[k]; }}; } })();
  const readJSON = (key, fallback) => { try { const raw = store.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch (_) { return fallback; } };
  const writeJSON = (key, value) => store.setItem(key, JSON.stringify(value));
  const clone = (value) => (value == null ? value : JSON.parse(JSON.stringify(value)));
  const session = () => { const s = readJSON(KEY_SESSION, null); return s && ID_PATTERN.test(s.annotator_id || "") ? s.annotator_id : null; };
  const identifier = (value) => {
    if (typeof value !== "string" || !ID_PATTERN.test(value)) throw new Error("Names may contain only 1–64 letters, digits, underscores or hyphens.");
    return value;
  };

  // ---------- server ----------
  async function server(path, body) {
    const init = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
    let response;
    try { response = await fetch(path, init); }
    catch (_) {   // only a save (POST) has edits waiting on this page
      throw Object.assign(new Error(`Cannot reach the grana label server. ${body === undefined ? "Try again once it is back." : "Your edits are still on this page; save again once it is back."}`), {offline: true});
    }
    const data = await response.json().catch(() => ({error: response.statusText}));
    if (!response.ok) throw Object.assign(new Error(data.error || response.statusText), {status: response.status, data});
    return data;
  }
  let project = null;
  const episodes = new Map();     // id -> Lab episode
  const loaded = new Map();       // id -> {record, companion, document, base}: base = saved revision
  let ordered = false, firstId = null;   // the list is reordered once (resume where this browser left off), then kept stable
  async function bootstrap() {
    project = await server("/api/project");
    const mark = {done: " · ✓ done", draft: " · draft", new: ""};
    const list = project.episodes.map((e) => {
      const episode = {
        id: String(e.index), index: e.index, title: `Episode ${e.index} · ${(e.length / project.fps).toFixed(1)} s${mark[e.status] || ""}`,
        observation_title: `Episode ${e.index} · ${(e.length / project.fps).toFixed(1)} s`,
        task: project.compressed_task_prompt, stages: project.stages.map((s) => s.prompt), stage_labels: project.stages.map((s) => s.label),
        duration_seconds: e.length / project.fps, fps: project.fps, total_frames: e.length, signature: `${project.task}#${e.index}`, status: e.status,
        project_revision: project.revision,
        created_at: null, views: project.views.map((v) => ({id: v, name: v, url: `/video?episode=${e.index}&view=${encodeURIComponent(v)}`, poster: null})),
      };
      episodes.set(episode.id, episode);
      return episode;
    });
    if (!ordered) {
      ordered = true;
      const params = new URLSearchParams(location.search), last = readJSON(KEY_LAST, null) || {};
      const wanted = params.get("episode") || last.episode || (list.find((e) => !/done/.test(e.title)) || {}).id;
      const i = list.findIndex((e) => e.id === wanted);
      if (i > 0) { const [first] = list.splice(i, 1); list.unshift(first); firstId = first.id; }
    }
    if (firstId !== null) {   // keep the opened-first episode in front so the selector does not jump around
      const i = list.findIndex((e) => e.id === firstId);
      if (i > 0) list.unshift(...list.splice(i, 1));
    }
    return list;
  }
  const episodeFor = (id) => { const e = episodes.get(String(id)); if (!e) throw new Error("This episode is not part of the recipe."); return e; };
  /** The server's JSON / schema errors for a label file do not say which file: name it (or the episode, for other errors). */
  function located(episode, error) {
    if (error.status !== 400 || /\.json\b/.test(error.message)) return error;
    const file = episode.status !== "new" && !/parquet|LeRobot|episode length/i.test(error.message);
    error.message = `${file ? `${project.labels}/episode_${String(episode.index).padStart(6, "0")}.json` : `Episode ${episode.index}`}: ${error.message}`;
    return error;
  }

  // ---------- conversion between Lab records (seconds) and grana label files (native frames) ----------
  const frame = (episode, t) => Math.min(episode.total_frames, Math.max(0, Math.round(t * episode.fps)));
  const seconds = (episode, f) => f / episode.fps;
  function recordFrom(episode, doc, revision, annotator, activeSeconds) {
    return {
      annotation_format: SCHEME.annotation_format, default_label: "normal", review_confirmed: doc.review_confirmed === true,
      intervals: doc.quality.map((q) => ({start: seconds(episode, q.start_frame), end: seconds(episode, q.end_frame), label: q.label === "regressed" ? "failure" : "suboptimal", stage: null})),
      pending_interval: null, status: doc.done ? "submitted" : "draft", revision, active_seconds: activeSeconds || 0,
      stage_intervals: doc.subtasks.map((s) => ({start: seconds(episode, s.start_frame), end: seconds(episode, s.end_frame), stage: episode.stage_labels.indexOf(s.stage)})),
      pending_stage_interval: null, schema_version: 4, protocol_version: PROTOCOL_VERSION,
      episode_id: episode.id, episode_signature: episode.signature, scheme_id: "stage_aq", annotator_id: annotator,
      duration_seconds: episode.duration_seconds, fps: episode.fps, updated_at: doc.updated_at, created_at: doc.updated_at,
      labeled_by: doc.annotator,
      machine_import: clone(doc.machine_import || null),
    };
  }
  function companionFrom(episode, doc, revision) {
    return {scheme_id: "stage_score", episode_id: episode.id, revision, import_source: "companion:stage_aq",
      intervals: doc.subtasks.map((s) => ({start: seconds(episode, s.start_frame), end: seconds(episode, s.end_frame),
        label: s.outcome || "unknown", stage: episode.stage_labels.indexOf(s.stage)}))};
  }
  /** Lab stage/AQ intervals (+ the outcome side store) → the body grana label expects. */
  function toLabels(episode, body, annotator) {
    const list = outcomes.get(episode.id, annotator, "stage_aq");
    const runs = [];
    for (const iv of [...(body.stage_intervals || [])].sort((a, b) => a.start - b.start)) {
      if (!Number.isInteger(iv.stage) || !episode.stage_labels[iv.stage]) continue;
      const start = frame(episode, iv.start), end = frame(episode, iv.end);
      if (end <= start) continue;
      const outcome = OUTCOME_LABEL[outcomes.lookup(list, iv)];
      const last = runs[runs.length - 1];
      if (last && last.stage === episode.stage_labels[iv.stage] && last.end_frame === start) { last.end_frame = end; last.outcome = last.outcome === null || outcome === null ? null : outcome; }
      else runs.push({start_frame: start, end_frame: end, stage: episode.stage_labels[iv.stage], outcome});
    }
    const quality = [...(body.intervals || [])].filter((iv) => iv.label === "suboptimal" || iv.label === "failure").sort((a, b) => a.start - b.start)
      .map((iv) => ({start_frame: frame(episode, iv.start), end_frame: frame(episode, iv.end), label: iv.label === "failure" ? "regressed" : "suboptimal"}))
      .filter((q) => q.end_frame > q.start_frame);
    return {subtasks: runs, quality, annotator: annotator || "", done: body.status === "submitted", review_confirmed: body.review_confirmed === true};
  }

  // ---------- light checks before the server's own validation ----------
  function check(episode, body) {
    if (!body || typeof body !== "object") throw new Error("The annotation must be a JSON object.");
    if (body.pending_interval || body.pending_stage_interval) throw new Error("End or cancel the pending span before saving or leaving this episode. Its start point has not been saved.");
    if (body.status === "submitted") {
      if (body.pending_interval || body.pending_stage_interval) throw new Error("There is still an unclosed start point. Mark its end point or cancel the interval before submitting.");
      if (body.review_confirmed !== true) throw new Error("Confirm that the whole video was reviewed (unmarked regions count as Nominal) before submitting.");
      let cursor = 0;
      for (const iv of [...(body.stage_intervals || [])].sort((a, b) => a.start - b.start)) {
        if (Math.abs(iv.start - cursor) > 1e-6) break;
        cursor = iv.end;
      }
      if (Math.abs(cursor - episode.duration_seconds) > 1e-6) throw new Error("Subtasks do not yet cover the whole video. Fill in the unlabelled subtasks or save a draft.");
      if (toLabels(episode, body, session()).subtasks.some(run => run.outcome === null)) throw new Error("Choose Completed or Failed for every subtask marked Needs review before submitting.");
    }
  }

  // ---------- per-subtask outcomes: {episodeId: {annotator: {scheme: [{start, end, stage, outcome}]}}} ----------
  const OUTCOME_LABEL = {complete: "high", failed: "low", unknown: null}, LABEL_OUTCOME = {high: "complete", low: "failed", unknown: "unknown"};
  const sameSpan = (a, b) => Math.abs(a.start - b.start) < 1e-6 && Math.abs(a.end - b.end) < 1e-6;
  const sideStore = {};
  const outcomes = {
    LABEL: OUTCOME_LABEL, OUTCOME: LABEL_OUTCOME, sameSpan, isCompanion: () => true,
    get(ep, annotator, scheme) { return clone((((sideStore[ep] || {})[annotator] || {})[scheme]) || []); },
    set(ep, annotator, scheme, list) {
      sideStore[ep] = sideStore[ep] || {}; sideStore[ep][annotator] = sideStore[ep][annotator] || {};
      sideStore[ep][annotator][scheme] = (Array.isArray(list) ? list : []).filter((o) => o && o.outcome in OUTCOME_LABEL)
        .map((o) => ({start: o.start, end: o.end, stage: o.stage, outcome: o.outcome}));
      emit({type: "outcomes", episode_id: ep});
    },
    lookup: (list, iv) => {
      const hit = (list || []).find((o) => o.stage === iv.stage && sameSpan(o, iv));
      if (hit) return hit.outcome;
      return (list || []).some(o => o.outcome === "unknown" && o.stage === iv.stage && o.start < iv.end && o.end > iv.start) ? "unknown" : "complete";
    },
    fromRecord: (record) => (record && Array.isArray(record.intervals) ? record.intervals : [])
      .filter((iv) => iv.label in LABEL_OUTCOME && Number.isInteger(iv.stage)).map((iv) => ({start: iv.start, end: iv.end, stage: iv.stage, outcome: LABEL_OUTCOME[iv.label]})),
    standalone: () => null,
    companion: (ep) => (loaded.get(String(ep)) || {}).companion || null,
  };
  function getRecord(episodeId, schemeId) {
    const hit = loaded.get(String(episodeId));
    if (!hit) return null;
    return clone(schemeId === "stage_score" ? hit.companion : hit.record);
  }

  // ---------- router ----------
  const listeners = new Set();
  const emit = (event) => { for (const fn of listeners) { try { fn(event); } catch (_) { /* listeners must not break the Lab */ } } };
  const parseBody = (body) => { if (body == null) return {}; if (typeof body === "string") { try { return JSON.parse(body); } catch (_) { throw new Error("The request body is not valid JSON."); } } return body; };
  const requireAnnotator = (q) => {
    const annotator = session();
    if (!annotator) throw Object.assign(new Error("Your editing name was signed out in another tab. Restore your editing name to continue; your unsaved labels are still here."), {sessionConflict: true});
    const expected = q.get("annotator_id");
    if (expected !== null && expected !== annotator) throw Object.assign(new Error("Your name changed in another tab. Restore your editing name to continue; your unsaved labels are still here."), {sessionConflict: true});
    return annotator;
  };
  const activeTime = {};
  async function handle(method, path, rawBody) {
    method = String(method || "GET").toUpperCase();
    const url = new URL(path, location.origin);
    const p = url.pathname.replace(/\/+$/, ""), q = url.searchParams;
    let m;
    if (p === "/api/bootstrap" && method === "GET") {
      return {schemes: clone([SCHEME]), episodes: clone(await bootstrap()), grid_hz: 10,
        access: {requires_reviewer_key: false, can_manage: false}, session: {annotator_id: session(), reviewer: false}};
    }
    if (p === "/api/session" && method === "POST") {
      const annotator = identifier(parseBody(rawBody).annotator_id);
      writeJSON(KEY_SESSION, {annotator_id: annotator, signed_in_at: new Date().toISOString()});
      emit({type: "session", annotator_id: annotator});
      return {annotator_id: annotator};
    }
    if (p === "/api/logout" && method === "POST") { store.removeItem(KEY_SESSION); emit({type: "session", annotator_id: null}); return {success: true}; }
    if (p === "/api/reviewer" && method === "POST") return {reviewer: false};
    if ((m = p.match(/^\/api\/annotations\/([^/]+)\/([^/]+)(\/export)?$/))) {
      const id = decodeURIComponent(m[1]);
      const episode = method === "PUT" || m[3] ? (loaded.get(id)?.episode || episodeFor(id)) : episodeFor(id);
      if (decodeURIComponent(m[2]) !== "stage_aq") throw new Error("grana label uses the Subtask + AQ protocol only.");
      const annotator = requireAnnotator(q);
      if (m[3]) {
        const hit = loaded.get(episode.id);
        if (!hit || !hit.record) throw new Error("Save this annotation first.");
        return clone(hit.document);
      }
      if (method === "GET") {
        const data = await server(`/api/episode?episode=${episode.index}`).catch((error) => { throw located(episode, error); });
        if (data.project_revision !== episode.project_revision) throw new Error("Sub-stage definitions changed. Reload the page to load the new version.");
        const doc = data.labels;
        writeJSON(KEY_LAST, {episode: episode.id});
        if (!doc.revision) {
          loaded.set(episode.id, {record: null, companion: null, document: clone(doc), base: null, episode: clone(episode)});
          outcomes.set(episode.id, annotator, "stage_aq", []);
          return {record: null};
        }
        const revision = 1;
        const record = recordFrom(episode, doc, revision, annotator, activeTime[episode.id]);
        const companion = companionFrom(episode, doc, revision);
        loaded.set(episode.id, {record, companion, document: clone(doc), base: doc.revision, episode: clone(episode)});
        // Saved outcomes seed the side store, so the curve and the C / F marks show them right after loading.
        outcomes.set(episode.id, annotator, "stage_aq", outcomes.fromRecord(companion));
        return {record: clone(record)};
      }
      if (method === "PUT") {
        const body = parseBody(rawBody);
        if (body.annotator_id !== annotator) throw Object.assign(new Error("Your name changed in another tab. Restore your editing name to continue; your unsaved labels are still here."), {sessionConflict: true});
        check(episode, body);
        const previous = (loaded.get(episode.id) || {}).record;
        const revision = previous ? previous.revision : 0;
        if (body.revision !== revision) throw new Error("This episode was saved elsewhere (for example in another tab). Reload it before saving so nothing is overwritten.");
        const base = (loaded.get(episode.id) || {}).base || null;
        let saved;
        try {
          saved = await server(`/api/labels?episode=${episode.index}`, {...toLabels(episode, body, annotator), base_revision: base, project_revision: episode.project_revision});
        } catch (error) {
          if (error.status !== 409) throw error;
          if (error.data.project_conflict) throw Object.assign(new Error(error.data.error), {conflict: true});
          // Keep the local editor intact; the annotator chooses how to recover.
          // A removed or 11-key file names no save made after this page loaded it.
          if (error.data.anonymous) throw Object.assign(new Error("The saved labels for this episode changed after this page loaded them. Your edits are still here. Export your unsaved labels, then reload the saved labels to compare and reapply changes."), {conflict: true});
          const when = new Date(error.data.updated_at);
          const time = Number.isNaN(when.getTime()) ? error.data.updated_at : when.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
          throw Object.assign(new Error(`${error.data.annotator} saved this episode at ${time}, after you opened it. Your edits are still here. Export your unsaved labels, then reload the saved labels to compare and reapply changes.`), {conflict: true});
        }
        const doc = saved.labels;
        activeTime[episode.id] = Math.max(activeTime[episode.id] || 0, body.active_seconds || 0);
        const record = {...recordFrom(episode, doc, revision + 1, annotator, activeTime[episode.id]), pending_interval: body.pending_interval || null,
                        pending_stage_interval: body.pending_stage_interval || null, created_at: previous ? previous.created_at : doc.updated_at};
        const companion = companionFrom(episode, doc, revision + 1);
        loaded.set(episode.id, {record, companion, document: clone(doc), base: doc.revision, episode: clone(episode)});
        outcomes.set(episode.id, annotator, "stage_aq", outcomes.fromRecord(companion));
        emit({type: "record", episode_id: episode.id, scheme_id: "stage_aq", annotator_id: annotator, status: record.status, revision: record.revision});
        return {record: clone(record)};
      }
    }
    if (p.startsWith("/api/reports/") || p.startsWith("/api/records/")) return {episode: null, records: [], notes: ["Agreement reports are not part of grana label."]};
    throw new Error(`grana label has no handler for ${method} ${p}.`);
  }
  function filenameFor(path) {
    const m = new URL(path, location.origin).pathname.match(/^\/api\/annotations\/([^/]+)\/([^/]+)\/export$/);
    return m ? `episode_${decodeURIComponent(m[1])}_${session() || "annotator"}.json` : "export.json";
  }
  function download(payload, filename) {
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
    const href = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = href; a.download = filename; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  }
  function draftDocument(episodeId, body, annotator) {
    const hit = loaded.get(String(episodeId)), episode = hit?.episode || episodeFor(episodeId);
    if (!hit || !hit.document) throw new Error("Load the episode before exporting labels.");
    check(episode, {...body, status: "draft"});
    return {...clone(hit.document), ...toLabels(episode, body, annotator), done: false, review_confirmed: false, revision: null, updated_at: null};
  }

  let saving = 0;   // saves in flight: a stroke made meanwhile says "Saving…" rather than "Busy…" (lab-interact.js)
  window.GRANA_LAB_API = {
    handle: async (method, path, body) => {
      const put = String(method || "").toUpperCase() === "PUT";
      if (put) saving += 1;
      try { return await handle(method, path, body); } finally { if (put) saving -= 1; }
    },
    saving: () => saving > 0, getRecord, session, identifier, outcomes, filenameFor, download, episodeFor, toLabels, draftDocument,
    project: () => project,
    updateStages: async (stages, base_revision) => {
      const result = await server("/api/stages", {stages, base_revision});
      // Keep loaded snapshots until the next GET succeeds, so an interrupted
      // refresh still permits exporting the old editor's labels.
      return result;
    },
    on: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
  };

  // ---------- page chrome the Lab does not own ----------
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) I18N.text(n, text); return n; };
  const STEPS = ["Pick a subtask brush above the Subtask timeline and drag to paint it until the clip is covered; click C / F at a subtask's end to set its outcome.",
                 "Pick Suboptimal or Regressed above the Annotation timeline and drag over the moments that went wrong.",
                 "Check the Converted progress curve, tick the review box, then Submit."];
  function guide() {
    // Open on the first visit only; afterwards the guide stays as this browser left it.
    const details = $("lab-guide"), KEY_GUIDE = "grana-label.guide";
    if (details) {
      let saved = null;
      try { saved = store.getItem(KEY_GUIDE); if (saved === null) store.setItem(KEY_GUIDE, "closed"); } catch (_) { /* storage blocked: open */ }
      details.open = saved !== "closed";
      let shown = details.open;
      details.addEventListener("toggle", () => { if (details.open === shown) return; shown = details.open; try { store.setItem(KEY_GUIDE, shown ? "open" : "closed"); } catch (_) { /* best effort */ } });
      document.querySelectorAll("a[href='#lab-guide']").forEach((a) => a.addEventListener("click", () => { details.open = true; }));
    }
    let lastKey = null;
    setInterval(() => {
      const signed = Boolean(session()), key = String(signed);
      if (key === lastKey) return; lastKey = key;
      const ol = $("lab-guide-steps"), count = $("lab-guide-count");
      const steps = [signed ? "Pick an episode." : "Enter your name, then pick an episode.", ...STEPS];
      if (ol) ol.replaceChildren(...steps.map((t) => el("li", null, t)));
      if (count) I18N.text(count, `${steps.length} steps`);
    }, 300);
  }
  function exportLinks() {
    document.addEventListener("click", async (event) => {
      const a = event.target && event.target.closest && event.target.closest("a[href^='/api/']");
      if (!a) return;
      event.preventDefault();
      try { download(await handle("GET", a.getAttribute("href")), filenameFor(a.getAttribute("href"))); }
      catch (error) { const notice = $("notice"); if (notice) { I18N.text(notice, error.message); notice.className = "notice error"; notice.hidden = false; } }
    });
  }
  function sessionBadge() {
    const badge = $("session-badge"); if (!badge) return;
    const apply = () => badge.classList.toggle("signed-in", Boolean(session()));
    apply(); listeners.add(apply);
  }
  function tooltips() {
    const sync = () => document.querySelectorAll(".lab-info[data-tip-from]").forEach((b) => { const src = $(b.dataset.tipFrom); const t = src ? I18N.source(src).trim() : ""; I18N.attr(b, "data-tip", t); if (b.hidden !== !t) b.hidden = !t; });
    sync(); setInterval(sync, 400);
    document.addEventListener("click", (event) => {
      const b = event.target && event.target.closest && event.target.closest(".lab-info");
      document.querySelectorAll(".lab-info.open").forEach((o) => { if (o !== b) o.classList.remove("open"); });
      if (b) { b.classList.toggle("open"); if (b.closest("summary")) event.preventDefault(); }
    });
  }
  function refreshTitlesAfterSave() {
    // Episode titles carry the saved done / draft state. A conflict leaves local edits untouched until explicit recovery.
    listeners.add((event) => {
      if (event.type !== "record" || !window.__agreementCtl) return;
      setTimeout(() => { Promise.resolve(window.__agreementCtl.refreshBootstrap()).catch(() => {}); }, 50);
    });
  }
  document.addEventListener("DOMContentLoaded", () => { guide(); exportLinks(); sessionBadge(); tooltips(); refreshTitlesAfterSave(); });
})();
