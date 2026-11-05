#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare an allowlisted video review site and three real HTTP inference examples."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grana.inference.selection import PREDICTION_FRAME_SLOTS, ROLES_HELP, resolve_prediction_slot

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Grana · deployment review</title>
<link rel="stylesheet" href="/review.css"><script defer src="/review.js"></script></head><body><main>
<header><span class="eyebrow">GRANA / REVIEW WORKSPACE</span><h1>Watch the comparisons.<br>Try the real inference server.</h1>
<p>Saved checkpoint comparisons and a live RGB → HTTP → checkpoint request, in one place.</p>
<nav><a href="#live">Live client/server</a><a href="#videos">Saved comparison videos</a><a href="/comparison/" target="_blank">Full comparison metrics ↗</a></nav></header>
<section id="live"><div class="heading"><h2>Live client → server → checkpoint</h2><span id="health" class="pill">Checking server…</span></div>
<p id="model" class="muted"></p><div class="flow"><span>Client: 8 RGB observations per camera</span><b>→</b><span>POST /api/predict</span><b>→</b><span>Loaded checkpoint</span><b>→</b><span id="selected-roles">C → D</span></div>
<p id="frame-note" class="note"></p>
<div class="controls"><label for="example">Replay window</label><select id="example"></select>
<label for="roles">Outputs</label><select id="roles"><option value="A">A · task progress only</option><option value="C">C · stage only</option><option value="C D">C + D · stage and focus progress</option></select>
<label for="prediction-frame">Prediction frame</label><select id="prediction-frame"><option value="last">LAST · slot 7 (latest observation)</option><option value="poi" selected>POI · slot 5 (point of interest)</option></select>
<button id="predict" disabled>Send RGB to server</button></div>
<p id="selection-label"></p><p id="window" class="muted"></p><img id="observations" alt="Eight observations per camera; selected prediction slot highlighted" />
<p id="status" role="status" aria-live="polite">Choose a window and send a real request. Saved scores are never substituted for a failed request.</p>
<div class="scores"><article id="card-a" hidden><small>A · task progress at selected frame</small><strong id="a">—</strong></article>
<article id="card-b" hidden><small>B · progress for requested focus at selected frame</small><strong id="b">—</strong></article>
<article id="card-c"><small>C · stage ID at selected frame (zero-based)</small><strong id="c">—</strong></article>
<article id="card-d"><small>D · progress for C's stage at selected frame</small><strong id="d">—</strong></article></div>
<figure id="plot-container" hidden><svg id="prediction-plot" role="img" aria-labelledby="plot-caption"></svg><figcaption id="plot-caption"></figcaption></figure>
<p id="parity" class="muted"></p><details><summary>HTTP response, all eight outputs and checkpoint identity</summary><pre id="response">No request yet.</pre></details></section>
<section id="videos"><div class="heading"><h2>Saved offline POI comparison videos</h2><a href="/comparison/">Open full metrics →</a></div>
<p>These saved comparisons use <strong>POI · slot 5</strong>, including D's focus selected from C at slot 5. Their measurements are unchanged by the live controls above. The selected checkpoints see four cameras at the same sampled positions. Playback shows sampled predictions, not original-speed footage; the full comparison lists its playback rate.</p><div id="video-grid"></div></section>
<footer>Private loopback review service. Media uses HTTP byte ranges for seeking. Browser requests use the actual server; source data and saved runs remain unchanged.</footer>
</main></body></html>"""

CSS = """*{box-sizing:border-box}[hidden]{display:none!important}html{scroll-behavior:smooth}body{margin:0;color:#142634;background:#f3f5f5;font:16px/1.6 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:36px}header{padding:28px 0 36px}.eyebrow{font-size:12px;letter-spacing:.18em;font-weight:750;color:#45716e}h1{font-size:clamp(32px,4vw,54px);line-height:1.12;letter-spacing:-.04em;margin:18px 0}h2{font-size:25px;letter-spacing:-.02em;margin:0}h3{font-size:19px;margin:10px 0}header p,.muted{color:#526671}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}a{color:#18695f}nav a{padding:10px 16px;background:white;border:1px solid #cbd6d4;border-radius:8px;text-decoration:none}section{background:white;border:1px solid #d7e0df;border-radius:16px;padding:28px;margin-bottom:28px}.heading{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}.pill{border-radius:30px;padding:5px 14px;background:#eef3f3;font-size:13px}.ready{background:#dff2e7;color:#17623d}.error{color:#a42f28;background:#fce6e3}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:16px;background:#f3f8f7;border-radius:9px;font-size:14px}.flow span{padding:5px}.note{font-size:14px;border-left:3px solid #d5ab52;padding:8px 14px;background:#fffaf0}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:24px 0 12px}button,select{font:inherit;border-radius:8px;padding:11px 16px;border:1px solid #b6c9c5}button{background:#166b5e;color:white;cursor:pointer}button:disabled{opacity:.45;cursor:wait}#observations{display:block;width:100%;border:1px solid #d7e0df;border-radius:8px;background:#152326}#selection-label{font-weight:700}.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.scores article{border:1px solid #d7e0df;border-radius:10px;padding:18px}.scores small{display:block;color:#526671}.scores strong{display:block;font-size:30px;letter-spacing:-.03em;margin-top:8px}figure{margin:24px 0}#prediction-plot{display:block;width:100%;background:#f6f9f9;border:1px solid #d7e0df;border-radius:10px}figcaption{font-size:13px;color:#526671;margin-top:8px}details{margin-top:20px}summary{cursor:pointer;color:#18695f}pre{font-size:12px;padding:16px;background:#f3f6f6;max-height:420px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}#video-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.video-card{border:1px solid #d7e0df;border-radius:12px;overflow:hidden;padding:16px}video{display:block;width:100%;background:#122026;border-radius:6px}.video-card p{font-size:13px;color:#526671}footer{color:#526671;font-size:13px;padding:0 6px 24px}@media(max-width:760px){main{padding:14px}section{padding:18px}#video-grid,.scores{grid-template-columns:1fr}h1{font-size:36px}}"""

JS = """'use strict';
const $ = id => document.getElementById(id);
let config, ready = false;
async function readJSON(url, options) {
  const r = await fetch(url, options); const body = await r.json();
  if (!r.ok) throw new Error(body.error?.message || body.error || `HTTP ${r.status}`);
  return body;
}
function binaryWithRequestId(template, requestId, roles = ['C','D'], predictionFrame = 'poi', frameIndices) {
  const bytes = new Uint8Array(template);
  if (bytes.length < 12 || String.fromCharCode(...bytes.subarray(0, 4)) !== 'GRNB')
    throw new Error('Invalid binary request template');
  const view = new DataView(template);
  const length = view.getUint32(8, true);
  if (view.getUint32(4, true) !== 1 || length < 2 || length > 1048576 || 12 + length > bytes.length)
    throw new Error('Invalid binary request header');
  if (!Array.isArray(roles) || !roles.length || new Set(roles).size !== roles.length || roles.some(r => !['A','B','C','D'].includes(r)))
    throw new Error('Choose a unique selection of roles A, B, C, D');
  if (!['poi','last'].includes(predictionFrame)) throw new Error('Choose POI or LAST prediction frame');
  if (frameIndices !== undefined && (!Array.isArray(frameIndices) || frameIndices.length !== 8 || frameIndices.some(i => !Number.isSafeInteger(i) || i < 0)))
    throw new Error('Expected eight nonnegative source frame indices');
  const header = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes.subarray(12, 12 + length)));
  header.request_id = requestId; header.roles = roles;
  header.prediction_frame = predictionFrame; header.poi_slot = predictionFrame === 'poi' ? 5 : 7;
  if (frameIndices !== undefined) header.frame_indices = frameIndices;
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  if (encoded.length > 1048576) throw new Error('Binary request header exceeds 1 MiB');
  const preamble = new ArrayBuffer(12);
  const prefix = new Uint8Array(preamble); prefix.set([71, 82, 78, 66]);
  const preambleView = new DataView(preamble);
  preambleView.setUint32(4, 1, true); preambleView.setUint32(8, encoded.length, true);
  const pixels = bytes.subarray(12 + length);
  const body = new Uint8Array(12 + encoded.length + pixels.length);
  body.set(prefix); body.set(encoded, 12); body.set(pixels, 12 + encoded.length);
  return body;
}
function selected() { return config.examples[Number($('example').value)]; }
function selection() {
  const roles = $('roles').value.split(' '), mode = $('prediction-frame').value;
  return {roles, mode, slot: mode === 'poi' ? 5 : 7};
}
function selectionLabel(e, chosen) {
  return `Roles ${chosen.roles.join(' + ')} · ${chosen.mode.toUpperCase()} · slot ${chosen.slot} · source frame ${e.frame_indices[chosen.slot]}`;
}
function clearPrediction() {
  window.lastGranaReply = null;
  for (const role of ['a','b','c','d']) $(role).textContent = '—';
  $('plot-container').hidden = true; $('prediction-plot').innerHTML = '';
  $('parity').textContent = '';
}
function showWindow() {
  clearPrediction();
  const e = selected(), chosen = selection();
  $('observations').src = e.previews[chosen.mode];
  $('observations').alt = `Eight observations per camera; ${chosen.mode.toUpperCase()} slot ${chosen.slot} highlighted`;
  $('selection-label').textContent = selectionLabel(e, chosen);
  $('selected-roles').textContent = chosen.roles.join(chosen.roles.join(' ') === 'C D' ? ' → ' : ' + ');
  $('window').textContent = `${e.label} · source frame indices in supplied slot order: ${e.frame_indices.join(', ')}`;
  $('frame-note').textContent = chosen.mode === 'last'
    ? 'LAST scores the latest supplied observation (slot 7), using this complete eight-observation window. Suitable for live monitoring when the supplied window contains only observations already received. Ground-truth labels are not sent to the model.'
    : 'POI scores slot 5, following the saved offline convention. Slots 6 and 7 are later observations, so this mode requires lookahead. Ground-truth labels are not sent to the model.';
  for (const role of ['A','B','C','D']) $('card-' + role.toLowerCase()).hidden = !chosen.roles.includes(role);
  $('status').textContent = 'Ready to send this window to the checkpoint server.';
  $('response').textContent = 'No request for this selection yet.';
}
function roleValues(p, role) {
  const values = role === 'C' ? p.stage?.ids : p.progress?.[role]?.values;
  if (!Array.isArray(values) || values.length !== 8 || values.some(v => !Number.isFinite(v)) ||
      (role === 'C' && values.some(v => !Number.isInteger(v) || v < 0)))
    throw new Error(`Expected eight valid ${role} outputs`);
  return values;
}
function drawPlot(p, e, chosen) {
  const height = 150 * chosen.roles.length + 52;
  const colors = {A:'#326cb5', B:'#9858a6', C:'#b07712', D:'#167768'};
  const x = i => 105 + i * 100;
  const parts = [];
  for (const [row, role] of chosen.roles.entries()) {
    const values = roleValues(p, role), top = 25 + row * 150, bottom = top + 100;
    const max = role === 'C' ? Math.max(1, (e.stage_count || 1) - 1, ...values) : 1;
    const y = value => bottom - value / max * 100;
    const focus = role === 'D' ? p.D_focus_stage : p.progress?.[role]?.focus_stage;
    parts.push(`<text x="12" y="${top+12}" fill="${colors[role]}" font-size="15">${role}${focus === undefined ? '' : ' · focus '+Number(focus)}</text>`);
    parts.push(`<text x="68" y="${top+5}" font-size="11">${max}</text><text x="68" y="${bottom+4}" font-size="11">0</text>`);
    parts.push(`<path d="M 98 ${top} V ${bottom} H 820" fill="none" stroke="#bdccca"/>`);
    parts.push(`<line data-selected-slot="${chosen.slot}" x1="${x(chosen.slot)}" x2="${x(chosen.slot)}" y1="${top-8}" y2="${bottom+6}" stroke="#c18c13" stroke-width="3" stroke-dasharray="5 4"/>`);
    parts.push(`<polyline points="${values.map((v,i)=>x(i)+','+y(v)).join(' ')}" fill="none" stroke="${colors[role]}" stroke-width="2"/>`);
    values.forEach((v,i) => parts.push(`<circle cx="${x(i)}" cy="${y(v)}" r="${i===chosen.slot?6:3}" fill="${colors[role]}"><title>${role} slot ${i}, source frame ${e.frame_indices[i]}: ${v}</title></circle>`));
  }
  for (let i=0;i<8;i++) {
    parts.push(`<text x="${x(i)}" y="${height-35}" text-anchor="middle" font-size="13" font-weight="${i===chosen.slot?700:400}">slot ${i}${i===chosen.slot?' ★':''}</text>`);
    parts.push(`<text x="${x(i)}" y="${height-15}" text-anchor="middle" font-size="12">f${e.frame_indices[i]}</text>`);
  }
  $('prediction-plot').setAttribute('viewBox', `0 0 860 ${height}`);
  $('prediction-plot').innerHTML = parts.join(''); $('plot-container').hidden = false;
  $('plot-caption').textContent = `${selectionLabel(e, chosen)}. All eight raw outputs are shown in supplied slot order; the selected frame is marked in gold.` +
    (chosen.roles.includes('D') ? ` Every D point uses the same focus stage ${p.D_focus_stage}, chosen by C at slot ${chosen.slot}.` : '');
}
function compareReference(p, e, chosen) {
  const ref = e.reference;
  const comparison = {comparable:false, maxDiff:null, stageMatch:null, focusMatch:null, messages:[]};
  if (!ref || ref.poi_slot !== chosen.slot) {
    comparison.messages.push(`Not comparable: saved offline ${String(ref?.prediction_frame || 'poi').toUpperCase()} uses slot ${ref?.poi_slot ?? 5}; this request selects ${chosen.mode.toUpperCase()} slot ${chosen.slot}.`);
    return comparison;
  }
  let comparable = true;
  for (const role of chosen.roles) {
    const saved = role === 'C' ? ref.stage_ids : ref.progress?.[role];
    if (!Array.isArray(saved) || saved.length !== 8 || saved.some(v => !Number.isFinite(v))) {
      comparable = false; comparison.messages.push(`${role} not comparable: no saved output for this role.`); continue;
    }
    if (role === 'D' || role === 'B') {
      const actualFocus = role === 'D' ? p.D_focus_stage : p.progress.B.focus_stage;
      const savedFocus = ref[role + '_focus_stage'];
      if (!Number.isInteger(savedFocus) || actualFocus !== savedFocus) {
        comparable = false; comparison.focusMatch = false;
        comparison.messages.push(`${role} not comparable: saved focus ${savedFocus ?? 'unknown'} differs from selected-frame focus ${actualFocus}.`);
        continue;
      }
      if (comparison.focusMatch !== false) comparison.focusMatch = true;
    }
    const values = roleValues(p, role);
    if (role === 'C') {
      comparison.stageMatch = values.every((v,i) => v === saved[i]);
      comparison.messages.push(`C stage IDs ${comparison.stageMatch ? 'match' : 'DIFFER'}.`);
    } else {
      const difference = Math.max(...values.map((v,i) => Math.abs(v-saved[i])));
      comparison.maxDiff = Math.max(comparison.maxDiff ?? 0, difference);
      comparison.messages.push(`${role} max progress difference ${difference}.`);
    }
  }
  comparison.comparable = comparable;
  return comparison;
}
async function initialize() {
  config = await readJSON('/review.json');
  $('model').textContent = `${config.model_label} · weights SHA256 ${config.weights_sha256.slice(0,16)}…`;
  config.examples.forEach((e,i) => $('example').add(new Option(e.label,String(i))));
  const defaultRoles = (config.default_roles || ['C','D']).join(' ');
  if (!['A','C','C D'].includes(defaultRoles)) $('roles').add(new Option(defaultRoles,defaultRoles));
  $('roles').value = defaultRoles; $('prediction-frame').value = config.default_prediction_frame || 'poi';
  $('example').value = config.examples.length > 1 ? '1' : '0'; showWindow();
  for (const key of ['example','roles','prediction-frame']) $(key).addEventListener('change',showWindow);
  for (const media of config.videos) {
    const card = document.createElement('article'); card.className = 'video-card';
    const title = document.createElement('h3'); title.textContent = media.label;
    const video = document.createElement('video'); video.controls = true; video.playsInline = true;
    video.preload = 'metadata'; video.poster = media.poster; video.src = media.video;
    const text = document.createElement('p'); text.textContent = 'Saved offline POI · slot 5 · sampled positions · H.264';
    const link = document.createElement('a'); link.href = media.video; link.textContent = 'Open / download video';
    card.append(title,video,text,link); $('video-grid').append(card);
  }
  try {
    const health = await readJSON('/healthz');
    const digest = health.weights_sha256 || health.checkpoint?.weights_sha256;
    if (digest !== config.weights_sha256) throw new Error('Server checkpoint differs from this review manifest');
    if (!health.wire_formats?.includes('grana-binary-v1')) throw new Error('Server does not support binary requests');
    ready = true; $('health').textContent = 'Checkpoint loaded · ready'; $('health').className = 'pill ready';
  } catch (error) { $('health').textContent = `Unavailable: ${error.message}`; $('health').className = 'pill error'; }
  $('predict').disabled = !ready;
}
$('predict').addEventListener('click', async () => {
  const e = selected(), chosen = selection();
  for (const key of ['predict','example','roles','prediction-frame']) $(key).disabled = true;
  clearPrediction(); $('response').textContent = 'Request in progress.';
  $('status').textContent = 'Loading the selected RGB observations…';
  try {
    const template = await fetch(e.payload);
    if (!template.ok) throw new Error(`Failed to load binary request: HTTP ${template.status}`);
    const bytes = await template.arrayBuffer();
    const requestId = `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const start = performance.now();
    const body = binaryWithRequestId(bytes, requestId, chosen.roles, chosen.mode, e.frame_indices);
    $('status').textContent = `Sending ${(body.byteLength/1048576).toFixed(2)} MiB of RGB to POST /api/predict…`;
    const reply = await readJSON('/api/predict', {method:'POST',headers:{'Content-Type':'application/vnd.grana.binary-v1'},body});
    const elapsed = (performance.now()-start)/1000;
    if (reply.request_id !== requestId) throw new Error('Request/response ID mismatch');
    const p = reply.result;
    if (p.checkpoint.weights_sha256 !== config.weights_sha256) throw new Error('Checkpoint identity mismatch');
    if (p.poi_slot !== chosen.slot || (p.prediction_frame && p.prediction_frame !== chosen.mode))
      throw new Error('Server returned a different prediction frame');
    if (!Array.isArray(p.roles) || p.roles.join(' ') !== chosen.roles.join(' ')) throw new Error('Server returned different roles');
    if (p.frame_indices && p.frame_indices.some((v,i) => v !== e.frame_indices[i])) throw new Error('Source frame indices differ');
    for (const role of chosen.roles) roleValues(p, role);
    if (chosen.roles.includes('D') && p.D_focus_stage !== roleValues(p, 'C')[chosen.slot])
      throw new Error('D focus does not match C at the selected frame');
    for (const role of chosen.roles) {
      const value = roleValues(p, role)[chosen.slot];
      const focus = role === 'D' ? p.D_focus_stage : p.progress?.[role]?.focus_stage;
      $(role.toLowerCase()).textContent = role === 'C' ? String(value) : value.toFixed(4) + (focus === undefined ? '' : ` (focus ${focus})`);
    }
    drawPlot(p, e, chosen);
    const comparison = compareReference(p, e, chosen);
    $('parity').textContent = 'Saved offline reference: ' + comparison.messages.join(' ');
    $('status').textContent = `Real HTTP prediction completed in ${elapsed.toFixed(2)} s (server ${Number(reply.elapsed_seconds).toFixed(2)} s). ${selectionLabel(e, chosen)}.`;
    $('response').textContent = JSON.stringify(reply,null,2);
    window.lastGranaReply = {request_id:reply.request_id, ...comparison, elapsed, roles:chosen.roles,
                            prediction_frame:chosen.mode, poi_slot:chosen.slot, frame_index:e.frame_indices[chosen.slot]};
  } catch (error) {
    clearPrediction(); $('response').textContent = 'Request failed; no prediction available.';
    $('status').textContent = `Request failed: ${error.message}`; $('parity').textContent = 'No saved prediction was used as a fallback.';
  }
  finally {
    $('predict').disabled = !ready;
    for (const key of ['example','roles','prediction-frame']) $(key).disabled = false;
  }
});
initialize().catch(error => { $('health').textContent = error.message; $('health').className='pill error'; });
"""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_binary_example(path, request, request_id, *, roles=("C", "D"),
                         prediction_frame="poi", frame_indices=None):
    """Write a lossless template with an explicit role/frame selection."""
    from grana.inference.client import build_binary

    request = replace(request, poi_slot=resolve_prediction_slot(prediction_frame),
                      prediction_frame=prediction_frame,
                      frame_indices=request.frame_indices if frame_indices is None else frame_indices)
    Path(path).write_bytes(build_binary(request, roles=roles, request_id=request_id))


def preview(images, indices, destination, *, selected_slot=5):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=15)
    height = 50 + len(images) * 120
    canvas = Image.new("RGB", (1360, height), "#15272d")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(indices):
        label = "POI" if slot == 5 else "LAST" if slot == 7 else ""
        draw.text((82 + slot * 160, 8), f"{slot} {label} | f{index}", font=font,
                  fill="#f3d07e" if slot == selected_slot else "white")
    for row, (view, frames) in enumerate(images.items()):
        draw.text((4, 52 + row * 120), view, font=font, fill="white")
        for slot, frame in enumerate(frames):
            image = Image.fromarray(frame)
            image.thumbnail((158, 118))
            canvas.paste(image, (80 + slot * 160 + (160-image.width)//2, 38 + row * 120 + (120-image.height)//2))
    x = 80 + selected_slot * 160
    draw.rectangle((x, 35, x + 159, height - 12), outline="#f3d07e", width=3)
    canvas.save(destination)


def reference_snapshot(predictions):
    """Retain the saved prediction's actual frame/focus for honest comparisons."""
    slot = predictions.get("poi_slot", 5)
    progress = predictions.get("progress", {})
    result = {"roles": predictions.get("roles", []), "poi_slot": slot,
              "prediction_frame": predictions.get("prediction_frame", "poi" if slot == 5 else "last" if slot == 7 else "custom"),
              "progress": {role: item["values"] for role, item in progress.items()}}
    if "stage" in predictions:
        result["stage_ids"] = predictions["stage"]["ids"]
    for role in ("B", "D"):
        focus = predictions.get(role + "_focus_stage", progress.get(role, {}).get("focus_stage"))
        if focus is not None:
            result[role + "_focus_stage"] = focus
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="Checkpoint arm ID from the comparison manifest")
    parser.add_argument("--episode", required=True, help="Episode ID from the comparison manifest")
    parser.add_argument("--roles", nargs="+", choices=("A", "B", "C", "D"), default=["C", "D"], help=ROLES_HELP)
    parser.add_argument("--prediction-frame", choices=tuple(PREDICTION_FRAME_SLOTS), default="poi",
                        help="Initial live output frame: poi=slot 5 (default), last=slot 7")
    parser.add_argument("--focus-stage", type=int, help="Explicit stage ID required when selecting role B")
    return parser


def main(argv=None):
    from compare_inference import arm_includes_episode, arrays_for, input_digest, select_images
    from grana.inference.predictor import InferenceRequest

    parser = argument_parser()
    args = parser.parse_args(argv)
    if len(set(args.roles)) != len(args.roles):
        parser.error("--roles must be unique")
    if "B" in args.roles and args.focus_stage is None:
        parser.error("--roles B requires --focus-stage")
    manifest_path = args.comparison_manifest.expanduser().resolve()
    comparison = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    arm = next(a for a in manifest["arms"] if a["id"] == args.arm)
    episode = next(e for e in manifest["episodes"] if e["id"] == args.episode)
    if args.focus_stage is not None and not 0 <= args.focus_stage < len(episode["stage_prompts"]):
        parser.error("--focus-stage must be a valid zero-based stage ID for this task")
    if not arm_includes_episode(arm, episode):
        raise ValueError("Selected checkpoint arm is not qualified for this episode's task")
    saved = json.loads((comparison / "results" / args.arm / f"{args.episode}.json").read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if saved["manifest_sha256"] != manifest_hash or saved["checkpoint"]["weights_sha256"] != arm["weights_sha256"]:
        raise ValueError("Saved comparison identity does not match the selected manifest/checkpoint")
    metadata = Path(arm["checkpoint"]) / "metadata.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != arm["checkpoint_metadata_sha256"]:
        raise ValueError("Selected checkpoint metadata has changed")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "examples").mkdir()
    assets = {}
    for name, content in (("index.html", HTML), ("review.css", CSS), ("review.js", JS)):
        (output / name).write_text(content)
        assets["/" if name == "index.html" else "/" + name] = str(output / name)
    assets["/comparison/"] = str(comparison / "media/index.html")
    assets["/comparison/render_receipt.json"] = str(comparison / "media/render_receipt.json")
    videos = []
    for item in manifest["episodes"]:
        key = item["id"]
        for suffix in ("mp4", "png"):
            file = comparison / "media" / f"{key}.{suffix}"
            if not file.is_file():
                raise FileNotFoundError(file)
            assets[f"/comparison/{key}.{suffix}"] = str(file)
        videos.append(dict(label=f"{item['task']} · {key}", video=f"/comparison/{key}.mp4", poster=f"/comparison/{key}.png"))
    arrays = arrays_for(episode, manifest["view_names"])
    selection = [0, len(episode["windows"]) // 2, len(episode["windows"]) - 1]
    examples, local_requests = [], []
    for position, (label, number) in enumerate(zip(("Start", "Middle", "End"), selection, strict=True)):
        window, reference = episode["windows"][number], saved["records"][number]
        images = select_images(arrays, window["frame_indices"])
        if input_digest(episode, window, images) != window["input_sha256"] or reference["input_sha256"] != window["input_sha256"]:
            raise ValueError("RGB input differs from the saved inference comparison")
        request = InferenceRequest(images=images, task=episode["task"], compressed_task_prompt=episode["compressed_task_prompt"],
                                   stage_prompts=episode["stage_prompts"], focus_stage=args.focus_stage,
                                   prediction_frame=args.prediction_frame, frame_indices=window["frame_indices"])
        payload_path = output / f"examples/window-{position}.bin"
        write_binary_example(payload_path, request, request_id=f"prepared-window-{position}",
                             roles=args.roles, prediction_frame=args.prediction_frame)
        payload_url = f"/examples/window-{position}.bin"
        assets[payload_url] = str(payload_path)
        previews = {}
        for mode, slot in PREDICTION_FRAME_SLOTS.items():
            image_path = output / f"examples/window-{position}-{mode}.png"
            preview(images, window["frame_indices"], image_path, selected_slot=slot)
            image_url = f"/examples/window-{position}-{mode}.png"
            assets[image_url], previews[mode] = str(image_path), image_url
        predictions = reference["output"]
        examples.append(dict(label=label, frame_indices=window["frame_indices"], payload=payload_url,
                             preview=previews[args.prediction_frame], previews=previews,
                             stage_count=len(episode["stage_prompts"]), input_sha256=window["input_sha256"],
                             reference=reference_snapshot(predictions)))
        local_request = dict(views=episode["frame_paths"], frame_indices=window["frame_indices"], task=episode["task"],
                             compressed_task_prompt=episode["compressed_task_prompt"], stage_prompts=episode["stage_prompts"],
                             poi_slot=PREDICTION_FRAME_SLOTS[args.prediction_frame], prediction_frame=args.prediction_frame)
        if args.focus_stage is not None:
            local_request["focus_stage"] = args.focus_stage
        local_requests.append(local_request)
        write_json(output / f"client_request_{position}.json", local_requests[-1])
    config = dict(model_label=f"{arm.get('label', arm['id'])} / {episode['task']} / step {saved['checkpoint']['source_step']}",
                  weights_sha256=arm["weights_sha256"], examples=examples, videos=videos,
                  default_roles=args.roles, default_prediction_frame=args.prediction_frame)
    write_json(output / "review.json", config)
    assets["/review.json"] = str(output / "review.json")
    write_json(output / "media_manifest.json", assets)
    receipt = dict(status="prepared", checkpoint=arm["checkpoint"], weights_sha256=arm["weights_sha256"], comparison_manifest=str(manifest_path), comparison_manifest_sha256=manifest_hash,
                   media_manifest=str(output / "media_manifest.json"), output=str(output), examples=len(examples), videos=len(videos), files={str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()})
    write_json(output / "preparation_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
