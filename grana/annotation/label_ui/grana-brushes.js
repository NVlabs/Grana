/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* grana-brushes.js — brush palettes right above the two tracks.
 *
 * Subtask track: one brush per subtask (S1, S2, ...). Quality track: Nominal (erases), Suboptimal, Regressed.
 * Pick a brush, then drag on its track: the stretch takes that label, replacing what was there (lab-interact.js
 * canPaint / commitPaint). Clicking the picked brush again puts it down. Keys: Shift+1–9 subtasks (Shift+0 = S10),
 * 0 / 1 / 2 quality.
 */
(() => {
  "use strict";
  const I18N = window.GRANA_I18N || {text: (n, s) => { n.textContent = s; }, attr: (n, k, s) => n.setAttribute(k, s), node: s => document.createTextNode(s), source: n => n.textContent, t: s => s};
  const $ = (id) => document.getElementById(id);
  const ctl = () => window.__agreementCtl, st = () => window.__agreementState, ux = () => window.GRANA_LAB_INTERACT;
  const QUALITY = [{id: "normal", name: "Nominal", note: "erase", key: "0"}, {id: "suboptimal", name: "Suboptimal", key: "1"}, {id: "failure", name: "Regressed", key: "2"}];
  const QUALITY_COLOR = {normal: "#30c5a5", suboptimal: "#e3a72f", failure: "#ec5b52"};
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) I18N.text(n, text); return n; };

  function setSubtask(index) {
    const select = $("independent-stage-select"); if (!select || select.disabled) return;
    select.value = index === null ? "" : select.options[index + 1].value;
    select.dispatchEvent(new Event("change", {bubbles: true}));
  }
  function setQuality(id) {
    const s = st(), c = ctl(); if (!s || !c || !c.canEdit()) return;
    s.selectedLabel = id;
    c.renderSelectedLabel();
  }
  function palette(kind) {
    const box = el("div", "grana-brushes");
    box.dataset.kind = kind;
    box.append(el("span", "grana-brushes-label", "Brush"));
    return box;
  }
  let stageSig = null;
  function renderStage(box) {
    const s = st(); if (!s || !s.episode) return;
    const sig = JSON.stringify(s.episode.stages);
    if (sig === stageSig) return;
    stageSig = sig;
    box.querySelectorAll("button").forEach((b) => b.remove());
    s.episode.stages.forEach((prompt, i) => {
      const b = el("button", "grana-brush");
      b.type = "button"; b.dataset.index = String(i); b.title = `S${i + 1} · ${prompt}${i < 10 ? ` · Shift+${(i + 1) % 10}` : ""}`;
      b.setAttribute("aria-label", `S${i + 1}: ${prompt}`);
      const sw = el("i"); sw.style.background = ux() ? ux().stageColor(i) : "#888";
      b.append(sw, el("span", null, `S${i + 1}`));
      b.addEventListener("click", () => setSubtask(st().selectedSemanticStage === i ? null : i));
      box.insertBefore(b, box.querySelector(".grana-brushes-hint"));
    });
  }
  function renderQuality(box) {
    QUALITY.forEach((q) => {
      const b = el("button", "grana-brush");
      b.type = "button"; b.dataset.label = q.id; I18N.attr(b, "title", `${q.name}${q.note ? ` (${q.note})` : ""} · key ${q.key}`);
      const sw = el("i"); sw.style.background = QUALITY_COLOR[q.id];
      b.append(sw, el("span", null, q.name), q.note ? el("small", null, q.note) : "");
      b.addEventListener("click", () => setQuality(st().selectedLabel === q.id ? null : q.id));
      box.append(b);
    });
  }
  const STAGE_HINT = ["Pick a subtask, then drag on the track", " · painting replaces what was there"];   // the second part only where it fits
  const stageHint = () => [I18N.node(STAGE_HINT[0]), el("span", "grana-hint-more", STAGE_HINT[1])];
  function hint(kind) {
    return el("span", "grana-brushes-hint", kind === "stage" ? null : "Pick a brush, then drag over the moments · Nominal erases");
  }
  /** The picked subtask's prompt, once, in the hint line: "S3 · Place the screwdriver…" without its "[From …]" precondition
   *  (one line; the full text on hover, as for the hint itself). */
  function showPicked(stage, k, editable) {
    const node = stage.querySelector(".grana-brushes-hint"); if (!node) return;
    const prompt = editable && Number.isInteger(k) ? ctl().stageName(k) : null, key = prompt === null ? "" : `${k}:${prompt}`;
    if (node.dataset.picked === key) return;
    node.dataset.picked = key; node.classList.toggle("picked", Boolean(key));
    if (!key) { node.replaceChildren(...stageHint()); I18N.attr(node, "title", STAGE_HINT.join("")); return; }
    node.replaceChildren(el("b", null, `S${k + 1}`), ` · ${prompt.replace(/^\[[^\]]*\]\s*/, "")}`); node.title = `S${k + 1} · ${prompt}`;
  }
  function mount() {
    const stageHost = $("own-stage-timeline"), aqHost = $("own-timeline");
    if (!stageHost || !aqHost || $("grana-brushes-stage")) return Boolean($("grana-brushes-stage"));
    const stage = palette("stage"); stage.id = "grana-brushes-stage";
    const quality = palette("aq"); quality.id = "grana-brushes-aq";
    renderQuality(quality);
    stage.append(hint("stage")); quality.append(hint("aq"));
    stageHost.before(stage); aqHost.before(quality);
    return true;
  }
  function sync() {
    const s = st(), c = ctl(); if (!s || !c) return;
    if (!mount()) return;
    const stage = $("grana-brushes-stage"), quality = $("grana-brushes-aq");
    for (const node of [stage, quality]) { const hint = node.querySelector(".grana-brushes-hint"); if (hint) { const text = " · " + I18N.t("Reopen the submitted episode to edit"); if (hint.dataset.lockedHint !== text) hint.dataset.lockedHint = text; } }
    renderStage(stage);
    const editable = c.canEdit();
    if (stage.hidden !== !c.independentStages()) stage.hidden = !c.independentStages();   // written only on change (idle polling)
    const mark = (b, on) => { if (b.disabled !== !editable) b.disabled = !editable; b.classList.toggle("on", on); if (b.getAttribute("aria-pressed") !== String(on)) b.setAttribute("aria-pressed", String(on)); };
    stage.querySelectorAll("button").forEach((b) => mark(b, Number(b.dataset.index) === s.selectedSemanticStage));
    quality.querySelectorAll("button").forEach((b) => mark(b, b.dataset.label === s.selectedLabel));
    const locked = !editable && !s.busy;   // a save in flight only dims the brushes for a moment, without the "reopen" hint
    stage.classList.toggle("locked", locked); quality.classList.toggle("locked", locked);
    showPicked(stage, s.selectedSemanticStage, editable || s.busy);   // not blanked by a save in flight
    // A pending start (S key) or an interval being edited lives in the folded exact-time panel: show it.
    const exactStage = $("grana-exact-stage"), exactAq = $("grana-exact-aq");
    if (exactStage && (s.pendingStageInterval || s.editingStageInterval !== null)) exactStage.open = true;
    if (exactAq && (s.pendingInterval || s.editingInterval !== null)) exactAq.open = true;
  }
  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || document.querySelector("dialog[open]")) return;
    // The physical number-row key works on non-US layouts; Shift+0 selects S10 in lab-interact.js.
    if (!(e.key === "0" || e.code === "Digit0") || e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest && e.target.closest("input, select, textarea, [contenteditable]")) return;
    e.preventDefault(); setQuality(st() && st().selectedLabel === "normal" ? null : "normal");
  });
  setInterval(sync, 200);
})();
