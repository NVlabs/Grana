// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Dependency-free checks of the browser adapter against the label-server contract.
const assert = require('node:assert/strict');
const {test} = require('node:test');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const asset = name => fs.readFileSync(path.join(__dirname, '../grana/annotation/label_ui', name), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));

async function fixture() {
  const project = {task: 'token-and-lid', fps: 10, compressed_task_prompt: 'Place token, close lid.',
    stages: [{label: '1', prompt: 'Place token'}, {label: '2', prompt: 'Close lid'}],
    views: ['top'], episodes: [{index: 0, length: 30, status: 'draft'}]};
  let document = {schema: 'grana.labels.v1', episode_index: 0, trajectory_id: 'raw__episode_000000',
    frame_count: 30, timestamps_sha256: 'fixture-clock', stages: ['1', '2'], stage_registry: project.stages,
    source_identity: {fixture: true}, revision: 'revision-a', updated_at: '2026-11-03T00:00:00Z',
    done: false, review_confirmed: false, annotator: 'Ada',
    subtasks: [{start_frame: 0, end_frame: 18, stage: '1', outcome: 'high'},
      {start_frame: 18, end_frame: 30, stage: '2', outcome: 'low'}],
    quality: [{start_frame: 8, end_frame: 11, label: 'regressed'}]};
  const storage = new Map(), requests = [], window = {};
  let conflict = false;
  const context = {window, URL, URLSearchParams, Blob, location: {origin: 'http://127.0.0.1:8789', search: ''},
    document: {addEventListener() {}}, setTimeout() {},
    localStorage: {getItem: k => storage.get(k) || null, setItem: (k, v) => storage.set(k, v), removeItem: k => storage.delete(k)},
    fetch: async (url, init) => {
      const body = init?.body ? JSON.parse(init.body) : null;
      requests.push({url, body});
      let data;
      if (url === '/api/project') data = project;
      else if (url.startsWith('/api/labels')) {
        if (conflict) return {ok: false, status: 409, json: async () => ({annotator: 'Bo', updated_at: document.updated_at})};
        assert.equal(body.base_revision, document.revision);
        const {base_revision, ...labels} = body;
        document = {...document, ...labels, revision: 'revision-b'};
        data = {labels: document, revision: document.revision};
      } else data = {labels: document};
      return {ok: true, json: async () => plain(data)};
    }};
  vm.runInNewContext(asset('grana-api.js'), context);
  const api = window.GRANA_LAB_API;
  await api.handle('POST', '/api/session', {annotator_id: 'Ada'});
  await api.handle('GET', '/api/bootstrap');
  const {record} = await api.handle('GET', '/api/annotations/0/stage_aq');
  return {api, record, requests, document: () => document, conflict: () => {conflict = true;}};
}

test('saved export preserves canonical frames, failed outcomes and source metadata', async () => {
  const f = await fixture();
  const exported = await f.api.handle('GET', '/api/annotations/0/stage_aq/export');
  assert.deepEqual(plain(exported), f.document());
  exported.subtasks[1].outcome = 'high';
  assert.equal((await f.api.handle('GET', '/api/annotations/0/stage_aq/export')).subtasks[1].outcome, 'low');
});

test('draft save retains outcomes and uses the returned revision snapshot without a second GET', async () => {
  const f = await fixture(), before = f.requests.length;
  const pending = f.api.handle('PUT', '/api/annotations/0/stage_aq', {...f.record, review_confirmed: true});
  assert.equal(f.api.saving(), true);
  const {record} = await pending;
  assert.equal(f.api.saving(), false);
  assert.equal(f.requests.length, before + 1);
  assert.equal(f.requests.at(-1).body.base_revision, 'revision-a');
  assert.equal(f.requests.at(-1).body.review_confirmed, true);
  assert.equal(f.requests.at(-1).body.subtasks[1].outcome, 'low');
  assert.equal(record.review_confirmed, true);
  assert.equal((await f.api.handle('GET', '/api/annotations/0/stage_aq/export')).revision, 'revision-b');
});

test('unfinished spans cannot be silently dropped by save or recovery export', async () => {
  for (const key of ['pending_interval', 'pending_stage_interval']) {
    const f = await fixture(), before = f.requests.length;
    const body = {...f.record, [key]: {start: 1, stage: 1, label: 'failure'}};
    await assert.rejects(f.api.handle('PUT', '/api/annotations/0/stage_aq', body), /End or cancel/);
    assert.throws(() => f.api.draftDocument('0', body, 'Ada'), /End or cancel/);
    assert.equal(f.requests.length, before);
  }
});

test('conflict preserves loaded revision and local outcomes for explicit recovery', async () => {
  const f = await fixture();
  f.api.outcomes.set('0', 'Ada', 'stage_aq', [{start: 0, end: 1.8, stage: 0, outcome: 'failed'}]);
  f.conflict();
  await assert.rejects(f.api.handle('PUT', '/api/annotations/0/stage_aq', f.record), e => e.conflict === true && !e.reload);
  assert.equal(f.api.saving(), false);
  assert.equal(f.api.getRecord('0', 'stage_aq').revision, 1);
  const draft = f.api.draftDocument('0', f.record, 'Ada');
  assert.equal(draft.subtasks[0].outcome, 'low');
  assert.equal(draft.revision, null);
  assert.equal(draft.updated_at, null);
  assert.equal(draft.done, false);
  assert.equal(draft.review_confirmed, false);
  assert.deepEqual(plain(draft.source_identity), f.document().source_identity);
});

test('brush shortcuts ignore open dialogs and events already consumed elsewhere', () => {
  let listener, open = false, changed = 0;
  const state = {selectedLabel: null};
  const document = {addEventListener: (_, fn) => {listener = fn;}, querySelector: () => open ? {} : null};
  const window = {__agreementState: state, __agreementCtl: {canEdit: () => true, renderSelectedLabel: () => changed++}};
  vm.runInNewContext(asset('grana-brushes.js'), {document, window, setInterval() {}});
  const event = {key: '0', target: {closest: () => null}, preventDefault() {}};
  open = true; listener(event); assert.equal(changed, 0);
  open = false; listener({...event, defaultPrevented: true}); assert.equal(changed, 0);
  listener({...event, shiftKey: true, code: 'Digit0'}); assert.equal(changed, 0);
  listener(event); assert.equal(state.selectedLabel, 'normal'); assert.equal(changed, 1);
  listener({...event, key: 'à', code: 'Digit0'}); assert.equal(state.selectedLabel, null); assert.equal(changed, 2);
});

// Exercise the production geometry and frame math without a browser or a duplicate implementation.
function agreementFunction(name, context = {}) {
  const source = asset('agreement.js'), start = source.indexOf(`  function ${name}(`);
  assert.notEqual(start, -1, `missing production helper ${name}`);
  const lineEnd = source.indexOf('\n', start);
  const end = source.slice(start, lineEnd).endsWith('}') ? lineEnd : source.indexOf('\n  }', start) + 4;
  return vm.runInNewContext(`(${source.slice(start, end)})`, context);
}

test('dense stage labels stay inside both track edges without overlapping either lane', () => {
  const layout = agreementFunction('layoutStageLabels');
  const labels = count => Array.from({length: count}, (_, i) => ({mid: 8 + (i + .5) * 784 / count, text: `S${i % 2 + 1}`}));
  const phone = layout(labels(22), 8, 792, 19);
  assert.equal(phone.filter(label => !label.elided).length, 22);
  assert.equal(new Set(phone.map(label => label.lane)).size, 2);
  for (const size of [9, 19, 30]) {
    for (const count of [1, 2, 22, 50, 200]) {
      const result = layout(labels(count), 8, 792, size);
      const ends = [8, 8];
      for (const label of result.filter(label => !label.elided)) {
        assert.ok(label.x - label.half >= 8 - 1e-9, 'left edge');
        assert.ok(label.x + label.half <= 792 + 1e-9, 'right edge');
        assert.ok(label.x - label.half >= ends[label.lane] - 1e-9, 'same-lane overlap');
        ends[label.lane] = label.x + label.half;
      }
    }
  }
  assert.ok(layout(labels(200), 8, 792, 19).some(label => label.elided), 'overflow is elided instead of shifted off-screen');
});

test('30 FPS frame boundaries and clip ends survive browser microsecond truncation', () => {
  const state = {episode: {fps: 30, duration_seconds: 100}, videos: [{currentTime: 0}]};
  const frameAt = agreementFunction('frameAt', {state});
  const mediaTime = agreementFunction('mediaTime', {state});
  const currentTime = agreementFunction('currentTime', {state});
  for (let frame = 1; frame < 3000; frame++) {
    state.videos[0].currentTime = Math.floor(mediaTime(frame / 30) * 1e6) / 1e6;
    assert.equal(frameAt(currentTime()), frame);
    assert.equal(frameAt(currentTime() - 1 / 30), frame - 1);
    assert.equal(frameAt(currentTime() + 1 / 30), frame + 1);
  }
  for (let count = 1; count <= 3000; count++) {
    state.episode.duration_seconds = count / 30;
    state.videos[0].currentTime = Math.floor(state.episode.duration_seconds * 1e6) / 1e6;
    assert.equal(currentTime(), state.episode.duration_seconds);
    assert.equal(frameAt(currentTime()), count);
  }
});
