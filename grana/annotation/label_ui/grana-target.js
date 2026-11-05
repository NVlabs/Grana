/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* grana-target.js — the Lab's "Constructed target" card, computed by the grana label server.
 *
 * The server runs the same function as grana build-cache on the current (unsaved) labels, so the curve is the
 * configured progress target that training will use. Regressed and suboptimal spans are shaded,
 * subtask boundaries are dashed, and clicking the plot seeks the video.
 */
(() => {
  "use strict";
  const I18N = window.GRANA_I18N || {text: (n, s) => { n.textContent = s; }, attr: (n, k, s) => n.setAttribute(k, s), node: s => document.createTextNode(s), source: n => n.textContent, t: s => s};
  const SVGNS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const svg = (tag, attrs, parent) => {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) I18N.attr(node, k, v);
    if (parent) parent.appendChild(node);
    return node;
  };
  const st = () => window.__agreementState || null;
  const API = () => window.GRANA_LAB_API;
  const SHADE = {suboptimal: "var(--g-gold, #fbc959)", failure: "var(--q-bad-dark, #ec5b52)"};
  const PAD = {l: 38, r: 12, t: 18, b: 24}, HEIGHT = 200;
  let signature = null, updateTimer = null, drawTimer = null, serial = 0, controller = null;
  let result = null, head = null, lastWidth = 0, offline = false, retry = null;

  function active(s) {
    return Boolean(s && s.episode && s.scheme && s.scheme.id === "stage_aq" && s.annotationFormat === "stage-aq-independent-v1" && !s.legacy);
  }
  function snapshot() {
    const s = st(); if (!active(s)) return null;
    const preview = window.GRANA_LAB_INTERACT?.previewIntervals();
    const stages = s.stageIntervals.map(iv => ({...iv}));
    const intervals = (preview?.intervals || s.intervals).map(iv => ({...iv}));
    const episode = {...s.episode};
    const body = API().toLabels(episode, {stage_intervals: stages, intervals, status: "draft"}, s.session?.annotator_id);
    const request = {subtasks: body.subtasks, quality: body.quality, project_revision: episode.project_revision};
    return {episode, stages, intervals, request,
      signature: JSON.stringify([s.contextSerial, episode.id, request, stages, intervals])};
  }
  function invalidate() {
    ++serial;
    controller?.abort(); controller = null;
    clearTimeout(updateTimer); clearTimeout(retry);
  }
  async function update() {
    const input = snapshot(); if (!input) return;
    invalidate();
    const id = serial;
    signature = input.signature;
    controller = new AbortController();
    let data, unreachable = false;
    try {
      const response = await fetch(`/api/preview?episode=${input.episode.index}`, {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(input.request), signal: controller.signal});
      data = await response.json();
      if (!response.ok) data = {problems: [data.error || response.statusText]};
    } catch (error) {
      if (error.name === "AbortError") return;
      unreachable = error instanceof TypeError;
      data = {problems: [unreachable ? "the grana label server cannot be reached" : error.message]};
    }
    // Compare the current inputs as well as request order: an edit can happen
    // before the next polling tick schedules its replacement request.
    if (id !== serial || snapshot()?.signature !== input.signature) return;
    controller = null;
    offline = unreachable;
    if (offline) retry = setTimeout(update, 4000);
    // The curve and its shading must come from the same immutable input snapshot.
    result = {...data, episode: input.episode, stages: input.stages, intervals: input.intervals};
    draw();
  }
  function draw() {
    const host = $("target-card-body"), note = $("target-card-note");
    if (!host || !result) return;
    const episode = result.episode, duration = episode.duration_seconds, width = Math.max(320, host.clientWidth || 640);
    lastWidth = width;
    const x = (t) => PAD.l + (t / duration) * (width - PAD.l - PAD.r);
    const y = (p) => PAD.t + (1 - p) * (HEIGHT - PAD.t - PAD.b);
    const plot = svg("svg", {class: "grana-target", viewBox: `0 0 ${width} ${HEIGHT}`, width: "100%", height: HEIGHT, role: "img",
                             "aria-label": "Constructed progress target"});
    for (const iv of result.intervals) {
      if (!(iv.label in SHADE)) continue;
      svg("rect", {x: x(iv.start), y: PAD.t, width: Math.max(1, x(iv.end) - x(iv.start)), height: HEIGHT - PAD.t - PAD.b, fill: SHADE[iv.label], "fill-opacity": 0.16}, plot);
    }
    for (const p of [0, 0.5, 1]) {
      svg("line", {x1: PAD.l, x2: width - PAD.r, y1: y(p), y2: y(p), class: "grid"}, plot);
      I18N.text(svg("text", {x: PAD.l - 6, y: y(p) + 3, "text-anchor": "end", class: "tick"}, plot), p.toFixed(1));
    }
    const stages = result.stages.slice().sort((a, b) => a.start - b.start);
    const ctl = window.__agreementCtl;
    stages.forEach((iv, i) => {
      if (i) svg("line", {x1: x(iv.start), x2: x(iv.start), y1: PAD.t, y2: HEIGHT - PAD.b, class: "boundary"}, plot);
      const text = `S${iv.stage + 1}`, w = text.length * 6.2 + 8, mid = (x(iv.start) + x(iv.end)) / 2;
      if (x(iv.end) - x(iv.start) < w + 4) return;
      // a pill in the subtask's colour (the same palette as the tracks), with ink that reads on it
      const color = ctl && ctl.stageColor ? ctl.stageColor(iv.stage) : "#75716a";
      svg("rect", {x: mid - w / 2, y: PAD.t + 1, width: w, height: 13, rx: 3, fill: color}, plot);
      I18N.text(svg("text", {x: mid, y: PAD.t + 11, "text-anchor": "middle", class: "stage-label", fill: ctl && ctl.inkOn ? ctl.inkOn(color) : "#0f0e0d"}, plot), text);
    });
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((v) => duration / v <= 8) || 600;
    for (let t = 0; t <= duration + 1e-9; t += step) {
      I18N.text(svg("text", {x: x(t), y: HEIGHT - 6, "text-anchor": t === 0 ? "start" : "middle", class: "tick"}, plot), `${t} s`);
    }
    if (result.frames && result.frames.length) {
      const points = result.frames.map((f, i) => `${x((f + 0.5) / episode.fps).toFixed(1)},${y(result.progress[i]).toFixed(1)}`).join(" ");
      svg("polyline", {points, class: "curve"}, plot);
      if (note) I18N.text(note, "Computed by grana build-cache's target function on the current labels (saved or not).");
    } else if (note) {
      I18N.text(note, `No target yet: ${(result.problems && result.problems[0]) || "mark subtasks over the whole clip"}.`);
    }
    head = svg("line", {y1: PAD.t, y2: HEIGHT - PAD.b, class: "playhead"}, plot);
    plot.addEventListener("click", (event) => {
      const rect = plot.getBoundingClientRect();
      const t = ((event.clientX - rect.left) / rect.width * width - PAD.l) / (width - PAD.l - PAD.r) * duration;
      if (window.__agreementSeek) window.__agreementSeek(Math.min(duration, Math.max(0, t)));
    });
    host.replaceChildren(plot);
    movePlayhead(true);
  }
  let lastTime = -1;
  function movePlayhead(force) {
    const ctl = window.__agreementCtl;
    if (!head || !result || !ctl) return;
    const t = ctl.currentTime(), width = lastWidth, duration = result.episode.duration_seconds;
    if (!force && t === lastTime) return;
    lastTime = t;
    const xp = PAD.l + (Math.min(duration, Math.max(0, t)) / duration) * (width - PAD.l - PAD.r);
    head.setAttribute("x1", xp); head.setAttribute("x2", xp);
  }
  setInterval(() => {
    const s = st(), card = $("target-card");
    if (!card) return;
    const on = active(s);
    if (card.hidden === on) card.hidden = !on;   // written only when it changes: the idle page makes no DOM work
    if (!on) {
      if (signature !== null) {
        invalidate(); signature = null; result = null; head = null;
        $("target-card-body").replaceChildren();
      }
      return;
    }
    const input = snapshot();
    if (input.signature !== signature) {
      invalidate(); signature = input.signature;
      I18N.text($("target-card-note"), "Updating curve…");
      updateTimer = setTimeout(update, 150);
    }
  }, 300);
  // The playhead follows the video each frame only while it plays; a seek or a pause moves it once. No loop while idle.
  let looping = false;
  function frame() {
    movePlayhead(false);
    const s = st(), video = s && s.videos[0];
    looping = Boolean(video && !video.paused && !document.hidden);
    if (looping) requestAnimationFrame(frame);
  }
  const wake = () => { if (!looping) { looping = true; requestAnimationFrame(frame); } };
  ["play", "pause", "seeked"].forEach((type) => document.addEventListener(type, wake, true));   // media events reach document in capture
  document.addEventListener("visibilitychange", () => { if (!document.hidden) wake(); });
  window.addEventListener("resize", () => { clearTimeout(drawTimer); drawTimer = setTimeout(draw, 150); });
  API().on((event) => { if (event.type === "record" && offline) { clearTimeout(retry); retry = setTimeout(update); } });   // a save: the server is back
})();
