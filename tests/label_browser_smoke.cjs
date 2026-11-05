// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Real Chrome regression via CDP. Node 22+ supplies WebSocket; no npm packages are needed.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const [base, chrome, output] = process.argv.slice(2);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const report = {checks: [], errors: [], screenshots: [], started: new Date().toISOString()};
let child, socket, send, evaluate, pageLoads = 0, scenario = '', cleanupPromise;
const dialogs = [], chromeLog = [], heldSaves = [];
let holdSaves = false;

async function until(check, label, timeout = 12000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const result = await check();
    if (result) return result;
    await sleep(60);
  }
  throw new Error(`Timed out: ${label}`);
}
const wait = (expression, label = expression) => until(() => evaluate(expression), label);
const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const idle = () => wait('window.__agreementState && !window.__agreementState.busy', 'idle');
const settle = () => evaluate('new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))');
const labels = () => evaluate('JSON.stringify([window.__agreementState.stageIntervals, window.__agreementState.intervals])');
const mouse = (type, x, y, extra = {}) => send('Input.dispatchMouseEvent', {type, x, y, ...extra});

async function screenshot(name, full = false) {
  const params = {format: 'png', captureBeyondViewport: full};
  if (full) {
    const {cssContentSize} = await send('Page.getLayoutMetrics');
    params.clip = {...cssContentSize, scale: 1};
  }
  const capture = await send('Page.captureScreenshot', params);
  const file = `${name}.png`;
  await fs.writeFile(path.join(output, file), Buffer.from(capture.data, 'base64'));
  report.screenshots.push(file);
  return file;
}
async function stateSnapshot() {
  return evaluate(`({episode:window.__agreementState?.episode?.id, busy:window.__agreementState?.busy,
    dirty:window.__agreementState?.dirty, status:window.__agreementState?.record?.status,
    brush:window.__agreementState?.selectedLabel, stageBrush:window.__agreementState?.selectedSemanticStage,
    stages:window.__agreementState?.stageIntervals, quality:window.__agreementState?.intervals,
    zoom:window.__agreementState?.trackZoom, notice:document.getElementById('notice')?.textContent,
    feedback:document.getElementById('annotation-feedback')?.textContent, time:window.__agreementCtl?.currentTime()})`);
}
async function check(name, action) {
  scenario = name;
  try {
    const evidence = await action();
    report.checks.push({name, passed: true, evidence});
    console.log(`PASS ${name} ${JSON.stringify(evidence ?? {})}`);
  } catch (error) {
    const state = await stateSnapshot().catch(() => null);
    const image = await screenshot(`failure-${report.checks.length}`).catch(() => null);
    report.checks.push({name, passed: false, error: error.stack, state, screenshot: image});
    console.error(`FAIL ${name}: ${error.stack}\n${JSON.stringify(state)}`);
  }
  await fs.writeFile(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2));
}
async function key(value, code = value, modifiers = 0, blur = true) {
  if (blur) await evaluate('document.activeElement.blur()');
  const virtual = {Enter: 13, ' ': 32, Escape: 27, ArrowRight: 39, ArrowLeft: 37, Home: 36, End: 35, Delete: 46, Backspace: 8}[value];
  const params = {key: value, code, modifiers, ...(virtual ? {windowsVirtualKeyCode: virtual} : {})};
  // CDP needs the character event for the browser's native Enter activation of a focused button.
  await send('Input.dispatchKeyEvent', {type: 'keyDown', ...params, ...(value === 'Enter' ? {text: '\r', unmodifiedText: '\r'} : {})});
  await send('Input.dispatchKeyEvent', {type: 'keyUp', ...params});
  await sleep(60);
}
async function geometry(id) {
  await evaluate(`document.getElementById(${JSON.stringify(id)}).scrollIntoView({block:'center', inline:'nearest', behavior:'instant'})`);
  await settle();
  return evaluate(`(() => {
    const host=document.getElementById(${JSON.stringify(id)}), svg=host.querySelector('svg.timeline-svg');
    const s=svg.getBoundingClientRect(), h=host.getBoundingClientRect(), v=svg.getAttribute('viewBox').split(' ').map(Number);
    return {left:s.left, top:s.top, width:s.width, height:s.height, vw:v[2], vh:v[3],
      hostLeft:h.left+host.clientLeft, hostRight:h.left+host.clientLeft+host.clientWidth,
      duration:window.__agreementState.episode.duration_seconds};
  })()`);
}
function point(g, time) {
  return {x: g.left + (8 + time / g.duration * (g.vw - 16)) / g.vw * g.width,
          y: g.top + 22 / g.vh * g.height};
}
async function stroke(id, start, end, {held = false, edge = null, whileBusy = false} = {}) {
  if (!whileBusy) await idle();
  const g = await geometry(id), a = point(g, start), b = point(g, end);
  if (edge === 'right') b.x = g.hostRight + 80;
  if (edge === 'left') b.x = g.hostLeft - 80;
  await mouse('mouseMoved', a.x, a.y);
  await mouse('mousePressed', a.x, a.y, {button: 'left', buttons: 1, clickCount: 1});
  for (let step = 1; step <= 6; step++) {
    await mouse('mouseMoved', a.x + (b.x - a.x) * step / 6, a.y, {button: 'left', buttons: 1});
  }
  const drag = {a, b};
  if (!held) await release(drag);
  return drag;
}
async function release(drag) {
  await mouse('mouseReleased', drag.b.x, drag.b.y, {button: 'left', buttons: 0, clickCount: 1});
  await settle();
}
async function selectSpan(time = 9) {
  const p = point(await geometry('own-timeline'), time);
  await mouse('mousePressed', p.x, p.y, {button: 'left', buttons: 1, clickCount: 1});
  await mouse('mouseReleased', p.x, p.y, {button: 'left', buttons: 0, clickCount: 1});
}
async function pickStage(index) {
  if (await evaluate(`window.__agreementState.selectedSemanticStage !== ${index}`)) {
    await wait(`!document.querySelector('#grana-brushes-stage button[data-index="${index}"]').disabled`);
    await click(`#grana-brushes-stage button[data-index="${index}"]`);
  }
  await wait(`window.__agreementState.selectedSemanticStage === ${index}`);
}
async function pickQuality(label) {
  if (await evaluate(`window.__agreementState.selectedLabel !== ${JSON.stringify(label)}`)) {
    await wait(`!document.querySelector('#grana-brushes-aq button[data-label="${label}"]').disabled`);
    await click(`#grana-brushes-aq button[data-label="${label}"]`);
  }
}
async function fit() {
  await click('[data-zoom="fit"]');
  await wait('(window.__agreementState.trackZoom || 1) === 1', 'Fit');
  await settle();
}
async function zoom4() {
  await fit();
  for (const level of [2, 4]) {
    await click('[data-zoom="in"]');
    await wait(`window.__agreementState.trackZoom === ${level}`);
  }
}
async function seedAnnotation() {
  await idle(); await fit();
  if (await evaluate('window.__agreementState.intervals.length + window.__agreementState.stageIntervals.length > 0')) {
    dialogs.push(true); await click('#clear-button'); await settle();
  }
  await pickStage(0); await stroke('own-stage-timeline', 0, 12);
  await pickStage(1); await stroke('own-stage-timeline', 12, 24);
  await click('.lab-oc.complete[data-index="0"]');
  await click('.lab-oc.failed[data-index="1"]');
  await pickQuality('failure'); await stroke('own-timeline', 8, 11);
}
async function reload() {
  const before = pageLoads;
  await send('Page.reload');
  await until(() => pageLoads > before, 'page load event');
  await wait('window.__agreementState?.record && !window.__agreementState.busy && window.GRANA_LAB_INTERACT', 'draft reload');
  await wait('window.__agreementState.videos.every(v => v.readyState >= 2)', 'decoded cameras after reload');
}
async function downloaded(filename) {
  const file = path.join(output, 'downloads', filename);
  await until(async () => {try {return (await fs.stat(file)).size > 0;} catch {return false;}}, filename);
  return JSON.parse(await fs.readFile(file, 'utf8'));
}
async function savedEpisode() {
  return (await (await fetch(`${base}/api/episode?episode=0`)).json()).labels;
}
async function submit() {
  if (!await evaluate('window.__agreementState.reviewConfirmed')) await click('#review-confirmed');
  await click('#submit-button');
  await wait('window.__agreementState.record?.status === "submitted" && !window.__agreementState.busy', 'submitted');
}
async function resizeHeld() {
  await key('Escape'); // Put both brushes down before taking an existing boundary.
  await geometry('own-timeline');
  const a = await evaluate(`(() => {
    const b=document.querySelector('#own-timeline .lab-handle[data-index="0"][data-edge="end"]').getBoundingClientRect();
    return {x:b.left+b.width/2, y:b.top+b.height/2};
  })()`);
  const b = {x: a.x + 80, y: a.y};
  await mouse('mousePressed', a.x, a.y, {button: 'left', buttons: 1, clickCount: 1});
  await mouse('mouseMoved', b.x, b.y, {button: 'left', buttons: 1});
  return {a, b};
}

async function expectedTarget(override = null) {
  const input = await evaluate(`(() => {const s=__agreementState; return {episode:s.episode,
    body:GRANA_LAB_API.toLabels(s.episode,{stage_intervals:s.stageIntervals,intervals:s.intervals,status:'draft'},s.session.annotator_id)};})()`);
  if (override) input.body.quality = override;
  const response = await fetch(`${base}/api/preview?episode=${input.episode.index}`, {method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({subtasks:input.body.subtasks,quality:input.body.quality,project_revision:input.episode.project_revision})});
  assert.equal(response.status,200);
  const result=await response.json(); assert.deepEqual(result.problems,[]);
  return {episode:input.episode,result};
}
async function targetMatches(expected) {
  const actual=await evaluate(`(() => {const svg=document.querySelector('#target-card-body svg');return {
    width:svg?.viewBox.baseVal.width,points:svg?.querySelector('polyline')?.getAttribute('points')};})()`);
  const {episode,result}=expected;
  const points=result.frames.map((frame,i)=>`${(38+(frame+0.5)/episode.fps/episode.duration_seconds*(actual.width-50)).toFixed(1)},${(18+(1-result.progress[i])*158).toFixed(1)}`).join(' ');
  return actual.points===points;
}
async function waitForTarget(expected) { await until(()=>targetMatches(expected),'curve matches current annotation boundaries'); }
async function dragQualityEdge(edge, time) {
  await key('Escape'); await idle();
  const g=await geometry('own-timeline');
  const a=await evaluate(`(() => {const r=document.querySelector('#own-timeline .lab-handle[data-index="0"][data-edge="${edge}"]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  const b={...point(g,time),y:a.y};
  await mouse('mousePressed',a.x,a.y,{button:'left',buttons:1,clickCount:1});
  await mouse('mouseMoved',b.x,b.y,{button:'left',buttons:1});
  return {a,b};
}

async function startChrome() {
  await fs.mkdir(path.join(output, 'downloads'), {recursive: true});
  await fs.copyFile(__filename, path.join(output, 'harness.cjs'));
  const profile = await fs.mkdtemp(path.join(output, 'chrome-profile-'));
  child = spawn(chrome, ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
    '--window-size=1440,1000', '--mute-audio', '--autoplay-policy=no-user-gesture-required', 'about:blank'],
    {stdio: ['ignore', 'ignore', 'pipe']});
  child.stderr.on('data', data => chromeLog.push(data.toString()));
  child.on('error', error => report.errors.push({scenario, error: error.message}));
  const port = await until(async () => {
    try {return Number((await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]);}
    catch {return false;}
  }, 'Chrome debugging port');
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'})).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, {once: true});
    socket.addEventListener('error', reject, {once: true});
  });
  const pending = new Map(); let serial = 0;
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const handler = pending.get(message.id);
      if (!handler) return;
      pending.delete(message.id); clearTimeout(handler.timer);
      if (message.error) handler.reject(new Error(JSON.stringify(message.error)));
      else handler.resolve(message.result);
    } else if (message.method === 'Page.loadEventFired') pageLoads++;
    else if (message.method === 'Runtime.exceptionThrown') report.errors.push({scenario, exception: message.params.exceptionDetails});
    else if (message.method === 'Fetch.requestPaused') {
      if (holdSaves && message.params.request.url.includes('/api/labels')) {heldSaves.push(message.params.requestId); return;}
      send('Fetch.fulfillRequest', {requestId: message.params.requestId, responseCode: 404,
        responseHeaders: [{name:'Content-Type',value:'application/json'}],
        body: Buffer.from(JSON.stringify({error:'Synthetic camera transport failure'})).toString('base64')}).catch(() => {});
    } else if (message.method === 'Page.javascriptDialogOpening') {
      send('Page.handleJavaScriptDialog', {accept: dialogs.length ? dialogs.shift() : true}).catch(() => {});
    }
  });
  send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => {pending.delete(id); reject(new Error(`CDP timeout: ${method}`));}, method === 'Page.navigate' ? 45000 : 15000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params}));
  });
  evaluate = async expression => {
    const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true, userGesture: true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await send('Runtime.enable'); await send('Page.enable'); await send('Network.enable');
  await send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: path.join(output, 'downloads')});
  await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  console.log('HTTP bootstrap', (await fetch(base, {signal:AbortSignal.timeout(5000)})).status);
  await send('Page.addScriptToEvaluateOnNewDocument', {source: "Object.defineProperty(navigator, 'language', {get: () => 'zh-CN'});"});
  await send('Page.navigate', {url: base});
  await wait('window.__agreementState?.episodes.length === 2 && !window.__agreementState.busy', 'bootstrap');
  await check('English is the default even with a Chinese browser locale', async () => {
    assert.equal(await evaluate('navigator.language'), 'zh-CN');
    assert.equal(await evaluate('document.documentElement.lang'), 'en');
    assert.equal(await evaluate('document.getElementById("account-button").textContent'), 'Enter a name to start');
    const mixed = await evaluate(`(() => { const copy=document.body.cloneNode(true); copy.querySelectorAll('.language-switch,script').forEach(n=>n.remove()); return /[\\u3400-\\u9fff]/.test(copy.textContent); })()`);
    assert.equal(mixed, false);
    await click('[data-language="zh"]');
    await click('#account-button');
    assert.equal(await evaluate('document.querySelector("#account-dialog h2").textContent'), '输入姓名');
    await evaluate('document.getElementById("annotator-id").value="TestDraft"');
    // Programmatic language event also covers an already open native modal.
    await click('[data-language="en"]');
    assert.equal(await evaluate('document.querySelector("#account-dialog h2").textContent'), 'Enter a name');
    assert.equal(await evaluate('document.getElementById("annotator-id").value'), 'TestDraft');
    await click('#account-dialog .close-dialog');
    return {defaultLanguage:'en', chineseOnlyOnExplicitSelection:true};
  });
  await click('#account-button');
  await evaluate(`document.getElementById('annotator-id').value='Ada';
    const form=document.getElementById('account-form'); form.requestSubmit(form.querySelector('[type=submit]'));`);
  await wait(`window.__agreementState.session.annotator_id === 'Ada' && !window.__agreementState.busy && !document.getElementById('account-dialog').open`);
  await wait('window.__agreementState.videos.length === 2 && window.__agreementState.videos.every(v => v.readyState >= 2)');
  await wait(`document.querySelectorAll('#grana-brushes-stage button').length === 2`);
}

async function main() {
  await startChrome();
  await check('Two decoded cameras and native 30 fps frame boundaries', async () => {
    assert.equal(await evaluate('window.__agreementState.episode.fps'), 30);
    assert.equal(await evaluate('window.__agreementState.episode.total_frames'), 720);
    await key('Home'); await key('ArrowRight');
    assert.ok(Math.abs(await evaluate('window.__agreementCtl.currentTime()') - 1 / 30) < 1e-6);
    await key('ArrowRight', 'ArrowRight', 8);
    assert.ok(Math.abs(await evaluate('window.__agreementCtl.currentTime()') - 31 / 30) < 1e-6);
    await key('End'); assert.equal(await evaluate('window.__agreementCtl.currentTime()'), 24);
    await key('ArrowLeft');
    const last = await evaluate('window.__agreementCtl.currentTime()');
    assert.ok(Math.abs(last - 719 / 30) < 1e-6); await key('Home');
    return {fps: 30, frames: 720, last};
  });
  await check('Two-camera playback, seek, and 2x speed stay synchronized', async () => {
    await click('#play-button'); await sleep(650);
    const first = await evaluate('window.__agreementState.videos.map(v => ({time:v.currentTime, paused:v.paused}))');
    assert.ok(first.every(v => !v.paused && v.time > 0.2));
    assert.ok(Math.abs(first[0].time - first[1].time) < 0.12);
    await evaluate(`document.getElementById('playback-rate').value='2';
      document.getElementById('playback-rate').dispatchEvent(new Event('change',{bubbles:true})); window.__agreementCtl.seek(10)`);
    await wait('window.__agreementState.videos.every(v => !v.paused && !v.seeking)', 'seek resumes together');
    await sleep(650);
    const second = await evaluate('window.__agreementState.videos.map(v => ({time:v.currentTime, rate:v.playbackRate}))');
    assert.ok(second.every(v => v.rate === 2 && v.time > 10.8));
    assert.ok(Math.abs(second[0].time - second[1].time) < 0.12);
    await evaluate('window.__agreementCtl.pauseAll()'); return {first, second};
  });
  await check('Keyboard focus, modal guards, and zoom button focus', async () => {
    await key('!', 'Digit1', 8); assert.equal(await evaluate('window.__agreementState.selectedSemanticStage'), 0);
    await pickQuality('failure'); await evaluate('document.getElementById("episode-select").focus()');
    await key('0', 'Digit0', 0, false); assert.equal(await evaluate('window.__agreementState.selectedLabel'), 'failure');
    await key('?', 'Slash', 8); assert.ok(await evaluate('document.getElementById("lab-shortcuts").open'));
    await key('0', 'Digit0'); assert.equal(await evaluate('window.__agreementState.selectedLabel'), 'failure');
    await key('Escape'); assert.equal(await evaluate('window.__agreementState.selectedLabel'), 'failure');
    await click('[data-zoom="in"]'); await wait('window.__agreementState.trackZoom === 2'); await settle();
    await evaluate('document.querySelector("[data-zoom=fit]").focus()'); await key('Enter', 'Enter', 0, false);
    await wait('window.__agreementState.trackZoom === 1');
    assert.equal(await evaluate('document.activeElement.dataset.zoom'), 'in');
  });
  await check('Pointer painting and outcome Undo preserve both tracks', async () => {
    await seedAnnotation(); await click('.lab-oc.complete[data-index="1"]');
    const before = await labels(); await click('.lab-oc.failed[data-index="1"]'); await key('z', 'KeyZ', 2);
    assert.equal(await labels(), before);
    assert.equal(await evaluate('window.GRANA_LAB_INTERACT.outcomeList().find(x => x.start === 12).outcome'), 'complete');
    await click('#save-button'); await wait('!window.__agreementState.dirty && !window.__agreementState.busy');
    assert.equal((await savedEpisode()).subtasks[1].outcome, 'high');
    const exported = await evaluate("window.GRANA_LAB_API.handle('GET','/api/annotations/0/stage_aq/export')");
    assert.equal(exported.subtasks[1].outcome, 'high');
    await click('.lab-oc.failed[data-index="1"]'); return await stateSnapshot();
  });
  await check('Language switching preserves unsaved labels, stage edits, video and canonical API values', async () => {
    await evaluate('window.__agreementCtl.pauseAll(); window.__agreementCtl.seek(5); document.getElementById("stage-settings").open=false');
    await wait('window.__agreementState.videos.every(v=>!v.seeking)');
    const before = await labels();
    const stateBefore = await evaluate('JSON.stringify([__agreementState.dirty,__agreementState.history,__agreementState.selectedSemanticStage,__agreementState.selectedLabel])');
    const savedBefore = await (await fetch(`${base}/api/episode?episode=0`)).json();
    await wait('!document.querySelector("[data-add-stage]").disabled');
    await click('[data-add-stage]');
    assert.equal(await evaluate('document.getElementById("stage-settings").closest("#stage-track-card") !== null'), true);
    assert.equal(await evaluate('document.querySelectorAll("#stage-settings-rows textarea").length'), 3);
    assert.equal(await evaluate('document.activeElement === document.querySelector("#stage-settings-rows").lastElementChild.querySelector("textarea")'), true);
    await evaluate('document.activeElement.value="Place the token carefully"; document.activeElement.dispatchEvent(new Event("input",{bubbles:true})); window.savedVideoNodes=[...__agreementState.videos];');
    await click('[data-language="zh"]');
    assert.equal(await evaluate('document.documentElement.lang'), 'zh-CN');
    assert.equal(await evaluate('document.getElementById("save-button").textContent'), '保存草稿');
    assert.equal(await evaluate('document.getElementById("stage-settings-apply").textContent'), '应用阶段定义');
    assert.equal(await evaluate('document.querySelector("#stage-settings-rows").lastElementChild.querySelector("textarea").value'), 'Place the token carefully');
    assert.equal(await labels(), before);
    assert.equal(await evaluate('JSON.stringify([__agreementState.dirty,__agreementState.history,__agreementState.selectedSemanticStage,__agreementState.selectedLabel])'), stateBefore);
    assert.equal(await evaluate('savedVideoNodes.every((v,i)=>v===__agreementState.videos[i])'), true);
    assert.ok(Math.abs(await evaluate('__agreementCtl.currentTime()') - 5) < 0.04);
    assert.deepEqual(await (await fetch(`${base}/api/episode?episode=0`)).json(), savedBefore);
    await evaluate('__agreementCtl.showNotice("End or cancel the pending span first.", "error")');
    assert.equal(await evaluate('document.getElementById("notice").textContent'), '请先结束或取消未完成的区间。');
    await screenshot('chinese-stage-editor');
    await click('[data-language="en"]');
    assert.equal(await evaluate('document.getElementById("notice").textContent'), 'End or cancel the pending span first.');
    assert.equal(await evaluate('document.getElementById("save-button").textContent'), 'Save draft');
    await click('#stage-settings-reset');
    await evaluate('document.getElementById("stage-settings").open=false; __agreementCtl.showNotice("")');
    assert.equal(await labels(), before);
    return {labelsPreserved:true,videoNodesPreserved:true,stageInputPreserved:true,noApiWrite:true};
  });
  await check('Painting during an in-flight save leaves labels unchanged', async () => {
    const before = await labels(); await pickQuality('failure'); holdSaves = true;
    await send('Fetch.enable', {patterns:[{urlPattern:'*api/labels?episode=0*',requestStage:'Request'}]});
    try {
      await click('#save-button');
      await until(async () => heldSaves.length && await evaluate('window.__agreementState.busy'), 'save in flight');
      await stroke('own-timeline', 14, 17, {whileBusy:true});
      assert.equal(await labels(), before);
      assert.match(await evaluate('document.getElementById("annotation-feedback").textContent'), /saving/i);
    } finally {
      holdSaves = false;
      for (const requestId of heldSaves.splice(0)) await send('Fetch.continueRequest', {requestId});
      await send('Fetch.disable');
    }
    await wait('!window.__agreementState.dirty && !window.__agreementState.busy');
    assert.deepEqual((await savedEpisode()).quality, [{start_frame:240,end_frame:330,label:'regressed'}]);
  });
  await check('Escape never deletes; Delete and Backspace are undoable; Nominal erases', async () => {
    const before = await labels(); await selectSpan(); await key('Escape');
    assert.equal(await labels(), before); assert.equal(await evaluate('window.__agreementState.selectedLabel'), null);
    for (const erase of ['Delete', 'Backspace']) {
      await selectSpan(); await key(erase); assert.equal(await evaluate('window.__agreementState.intervals.length'), 0);
      await key('z', 'KeyZ', 2); assert.equal(await labels(), before);
    }
    await key('0', 'Digit0'); await stroke('own-timeline', 9, 10);
    assert.deepEqual(await evaluate('window.__agreementState.intervals.map(x => [x.start,x.end])'), [[8, 9], [10, 11]]);
    await key('z', 'KeyZ', 2); assert.equal(await labels(), before);
  });
  await check('Zoom +, −, Fit and synchronized horizontal pan preserve labels', async () => {
    const before = await labels(); await zoom4();
    await evaluate('document.getElementById("own-stage-timeline").scrollLeft = 600'); await sleep(150);
    const pan = await evaluate(`['own-stage-timeline','own-timeline'].map(id => {
      const h=document.getElementById(id); return {scroll:h.scrollLeft,
        time:window.GRANA_LAB_INTERACT.timeAt(h.querySelector('svg'),h.getBoundingClientRect().left+h.clientLeft)};
    })`);
    assert.ok(pan[0].scroll > 0 && pan[1].scroll > 0);
    assert.ok(Math.abs(pan[0].time - pan[1].time) < 1 / 30, JSON.stringify(pan));
    await screenshot('zoomed-tracks');
    await click('[data-zoom="out"]'); await wait('window.__agreementState.trackZoom === 2');
    await fit(); assert.equal(await labels(), before); return pan;
  });
  await check('Ctrl-wheel anchors the timeline without zooming the page', async () => {
    await fit(); const p = point(await geometry('own-stage-timeline'), 12);
    await mouse('mouseWheel', p.x, p.y, {deltaY: -60, deltaX: 0, modifiers: 2});
    await wait('window.__agreementState.trackZoom > 1'); await settle();
    const evidence = await evaluate(`({time:window.GRANA_LAB_INTERACT.timeAt(document.querySelector('#own-stage-timeline svg'),${p.x}), zoom:window.__agreementState.trackZoom, scale:visualViewport.scale})`);
    assert.ok(Math.abs(evidence.time - 12) < 0.05); assert.equal(evidence.scale, 1);
    await fit(); return evidence;
  });
  for (const edge of ['right', 'left']) await check(`Zoomed paint stays inside the ${edge} visible frame boundary`, async () => {
    await pickQuality('suboptimal'); await key('Home'); await zoom4();
    await evaluate(`document.getElementById('own-stage-timeline').scrollLeft = ${edge === 'right' ? 0 : 900}`);
    await sleep(100); const g = await geometry('own-timeline');
    const raw = await evaluate(`window.GRANA_LAB_INTERACT.timeAt(document.querySelector('#own-timeline svg'),${edge === 'right' ? g.hostRight : g.hostLeft})`);
    const boundary = edge === 'right' ? Math.floor(raw * 30 + 1e-6) / 30 : Math.ceil(raw * 30 - 1e-6) / 30;
    const before = await labels();
    try {
      await stroke('own-timeline', edge === 'right' ? 2 : boundary + 2, boundary, {edge});
      const interval = await evaluate('window.__agreementState.intervals.find(x => x.label === "suboptimal")');
      assert.ok(Math.abs(interval[edge === 'right' ? 'end' : 'start'] - boundary) < 1e-6);
      assert.deepEqual(await evaluate('window.__agreementState.intervals.find(x => x.label === "failure")'),
        {start: 8, end: 11, label: 'failure', stage: null});
      return {boundary, interval};
    } finally {
      if (await labels() !== before) await key('z', 'KeyZ', 2);
      await fit();
    }
  });
  for (const enlarged of [false, true]) {
    await check(`Escape cancels held paint${enlarged ? ' with enlarged camera' : ''}`, async () => {
      await fit(); await pickQuality('suboptimal'); const before = await labels();
      if (enlarged) await click('.view-expand');
      const drag = await stroke('own-timeline', 14, 17, {held: true});
      assert.ok(await evaluate('Boolean(document.querySelector(".lab-ghost"))'));
      await key('Escape'); await release(drag);
      assert.equal(await labels(), before);
      assert.equal(await evaluate('document.querySelectorAll(".lab-ghost,.lab-dragging").length'), 0);
      assert.equal(await evaluate('window.__agreementState.selectedLabel'), 'suboptimal');
      if (await evaluate('Boolean(document.querySelector(".view-frame.enlarged"))')) await key('Escape');
    });
    await check(`Escape cancels held resize${enlarged ? ' with enlarged camera' : ''}`, async () => {
      await fit(); const before = await labels(); const drag = await resizeHeld();
      if (enlarged) await click('.view-expand');
      await key('Escape'); await release(drag); assert.equal(await labels(), before);
      assert.equal(await evaluate('document.querySelectorAll(".lab-dragging").length'), 0);
      if (await evaluate('Boolean(document.querySelector(".view-frame.enlarged"))')) await key('Escape');
    });
  }
  await check('Native touch cancellation discards a held paint stroke', async () => {
    await pickQuality('suboptimal'); const before = await labels();
    const g = await geometry('own-timeline'), a = point(g, 14), b = point(g, 17);
    await send('Emulation.setTouchEmulationEnabled', {enabled: true, maxTouchPoints: 1});
    try {
      await send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [{...a, id: 1}]});
      await send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: [{...b, id: 1}]});
      assert.ok(await evaluate('Boolean(document.querySelector(".lab-ghost"))'));
      await send('Input.dispatchTouchEvent', {type: 'touchCancel', touchPoints: []}); await settle();
      assert.equal(await labels(), before);
      assert.equal(await evaluate('document.querySelectorAll(".lab-ghost,.lab-dragging").length'), 0);
    } finally {await send('Emulation.setTouchEmulationEnabled', {enabled: false});}
  });
  await check('Pending spans block saving and navigation; Escape preserves closed spans', async () => {
    await fit(); await pickQuality('failure'); await key('Home'); const before = await labels();
    await key('s', 'KeyS'); assert.equal(await evaluate('document.getElementById("save-button").disabled'), true);
    await evaluate(`const select=document.getElementById('episode-select'); select.value='1'; select.dispatchEvent(new Event('change',{bubbles:true}));`);
    await idle(); assert.equal(await evaluate('window.__agreementState.episode.id'), '0');
    assert.ok(await evaluate('Boolean(window.__agreementState.pendingInterval)'));
    await key('Escape'); assert.equal(await evaluate('window.__agreementState.pendingInterval'), null);
    assert.equal(await labels(), before);
  });
  await check('Dragging a regressed boundary refreshes the curve without saving', async () => {
    await seedAnnotation(); await waitForTarget(await expectedTarget());
    const before=await labels(), old=(await expectedTarget()).result.progress;
    const drag=await dragQualityEdge('end',14); await release(drag);
    assert.equal(await evaluate('__agreementState.intervals[0].end'),14);
    assert.equal(await evaluate('__agreementState.dirty'),true);
    const expected=await expectedTarget();assert.notDeepEqual(expected.result.progress,old);await waitForTarget(expected);
    await click('#undo-button');assert.equal(await labels(),before);await waitForTarget(await expectedTarget());
    return {afterRelease:true,undo:true};
  });
  await check('Held boundary drag previews the curve and Escape restores it without editing the draft', async () => {
    await seedAnnotation();const before=await labels();const original=await expectedTarget();await waitForTarget(original);
    const drag=await dragQualityEdge('end',14);
    try {
      await waitForTarget(await expectedTarget([{start_frame:240,end_frame:420,label:'regressed'}]));
      assert.equal(await labels(),before);
      await screenshot('regressed-boundary-live-preview');
    } finally {await key('Escape');await release(drag);}
    assert.equal(await labels(),before);await waitForTarget(original);
    return {duringDrag:true,cancelRestored:true,draftUnchanged:true};
  });
  await check('Window redraw cannot cancel the pending target calculation after a boundary edit', async () => {
    await seedAnnotation();await waitForTarget(await expectedTarget());
    const drag=await dragQualityEdge('start',6);
    await evaluate('window.__targetResizeTest=setInterval(()=>window.dispatchEvent(new Event("resize")),40)');
    try {await release(drag);await sleep(800);} finally {await evaluate('clearInterval(window.__targetResizeTest)');}
    assert.equal(await evaluate('__agreementState.intervals[0].start'),6);
    await waitForTarget(await expectedTarget());
    return {redrawDidNotCancelCalculation:true};
  });
  await check('A delayed curve response cannot overwrite a cancelled boundary drag', async () => {
    await seedAnnotation();const before=await labels(), original=await expectedTarget();await waitForTarget(original);
    await evaluate(`window.__originalTargetFetch=window.fetch;window.__holdTargetResponse=true;window.__releaseTargetResponse=null;
      window.fetch=async (...args)=>{
        const hold=String(args[0]).includes('/api/preview') && window.__holdTargetResponse;
        if(hold) window.__holdTargetResponse=false;
        const response=await window.__originalTargetFetch(...args);
        if(!hold) return response;
        const data=await response.json();
        return {ok:response.ok,statusText:response.statusText,json:()=>new Promise(resolve=>{window.__releaseTargetResponse=()=>resolve(data);})};
      };`);
    const drag=await dragQualityEdge('end',14);
    try {
      await wait('typeof window.__releaseTargetResponse === "function"','delayed preview ready');
      // Cancel and deliver the stale reply in the same turn, before polling can
      // enqueue a replacement. The old curve must never flash onto the page.
      await evaluate('GRANA_LAB_INTERACT.cancelDrag();window.__releaseTargetResponse()');
      await settle();assert.equal(await targetMatches(original),true);
      assert.equal(await labels(),before);
      await release(drag);await waitForTarget(original);
    } finally {
      await evaluate('window.fetch=window.__originalTargetFetch;window.__releaseTargetResponse?.();GRANA_LAB_INTERACT.cancelDrag()');
      await release(drag);
    }
    return {staleReplyRejected:true,cancelledDraftIntact:true};
  });
  await check('Dragging a shared Regressed/Suboptimal boundary updates both regions and the curve', async () => {
    await seedAnnotation();await pickQuality('suboptimal');await stroke('own-timeline',11,15);
    await waitForTarget(await expectedTarget());const before=await labels();
    const drag=await dragQualityEdge('end',10);
    try {
      await waitForTarget(await expectedTarget([{start_frame:240,end_frame:300,label:'regressed'},
        {start_frame:300,end_frame:450,label:'suboptimal'}]));
      assert.equal(await labels(),before);await release(drag);
    } finally {await key('Escape');await release(drag);}
    assert.deepEqual(await evaluate('__agreementState.intervals.map(iv=>[iv.start,iv.end,iv.label])'),
      [[8,10,'failure'],[10,15,'suboptimal']]);
    await waitForTarget(await expectedTarget());
    await click('#undo-button');assert.equal(await labels(),before);await waitForTarget(await expectedTarget());
    return {sharedBoundaryPreview:true,committedCurveUpdated:true,undo:true};
  });
  await check('Draft saves native frames and UUID revision', async () => {
    await seedAnnotation(); await click('#save-button');
    await wait('window.__agreementState.record && !window.__agreementState.dirty && !window.__agreementState.busy');
    const saved = await savedEpisode();
    assert.deepEqual(saved.subtasks, [{start_frame: 0, end_frame: 360, stage: '1', outcome: 'high'},
      {start_frame: 360, end_frame: 720, stage: '2', outcome: 'low'}]);
    assert.deepEqual(saved.quality, [{start_frame: 240, end_frame: 330, label: 'regressed'}]);
    assert.match(saved.revision, /^[0-9a-f]{32}$/i); assert.equal(saved.done, false);
    await fs.writeFile(path.join(output, 'saved-draft.json'), JSON.stringify(saved, null, 2)); return saved;
  });
  await check('Reload and real Export JSON download preserve the canonical document', async () => {
    await reload(); assert.equal(await evaluate('window.GRANA_LAB_INTERACT.outcomeList()[1].outcome'), 'failed');
    await click('#export-own');
    const exported = await downloaded('episode_0_Ada.json');
    assert.deepEqual(exported, await savedEpisode()); return exported;
  });
  await check('Submit, preview, reopen and resubmit', async () => {
    await submit(); await wait('document.querySelector("#target-card-body polyline")', 'constructed target');
    const saved = await savedEpisode(); assert.equal(saved.done, true); assert.equal(saved.review_confirmed, true);
    dialogs.push(true); await click('#reopen-button');
    await wait('window.__agreementState.record?.status === "draft" && !window.__agreementState.busy');
    assert.equal((await savedEpisode()).done, false); await submit();
    return {submitted: saved, reopened: true, resubmitted: true};
  });
  for (const width of [1440, 700, 390]) await check(`Responsive layout at ${width}px`, async () => {
    await send('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: false});
    await evaluate('window.scrollTo(0,0)'); await sleep(300);
    const full = await screenshot(`label-ui-${width}-full`, true);
    const size = await evaluate('({viewport:innerWidth, body:document.body.scrollWidth, html:document.documentElement.scrollWidth})');
    assert.ok(size.body <= width + 1 && size.html <= width + 1, JSON.stringify(size));
    await geometry('own-stage-timeline');
    const editor = await screenshot(`label-ui-${width}-editor`); return {full, editor, ...size};
  });
  await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await check('UUID conflict keeps edits and exports them before explicit reload', async () => {
    dialogs.push(true); await click('#reopen-button');
    await wait('window.__agreementState.record?.status === "draft" && !window.__agreementState.busy');
    const saved = await savedEpisode();
    const projectRevision = (await (await fetch(`${base}/api/project`)).json()).revision;
    const response = await fetch(`${base}/api/labels?episode=0`, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({subtasks: saved.subtasks, quality: [], annotator: 'Bo', done: false,
        review_confirmed: false, base_revision: saved.revision, project_revision: projectRevision})});
    assert.equal(response.status, 200); const newer = await savedEpisode(); assert.notEqual(newer.revision, saved.revision);
    await click('.lab-oc.failed[data-index="0"]'); await click('#save-button');
    await wait('window.__agreementState.saveConflict && !window.__agreementState.busy', 'recoverable conflict');
    assert.equal(await evaluate('window.__agreementState.dirty'), true);
    assert.equal(await evaluate('window.__agreementState.intervals.length'), 1);
    assert.equal(await evaluate('window.GRANA_LAB_INTERACT.outcomeList()[0].outcome'), 'failed');
    await click('#export-unsaved-labels'); const unsaved = await downloaded('episode_0_unsaved.json');
    assert.equal(unsaved.subtasks[0].outcome, 'low'); assert.equal(unsaved.revision, null);
    assert.deepEqual(unsaved.quality, saved.quality); await screenshot('conflict-edits-preserved');
    dialogs.push(false); await click('#reload-saved-labels'); await sleep(100);
    assert.equal(await evaluate('window.__agreementState.saveConflict'), true);
    dialogs.push(true); await click('#reload-saved-labels');
    await wait('!window.__agreementState.saveConflict && !window.__agreementState.busy');
    assert.equal(await evaluate('window.__agreementState.intervals.length'), 0);
    assert.equal(await evaluate('window.__agreementState.record.labeled_by'), 'Bo');
    return {base: saved.revision, remote: newer.revision, unsavedExport: unsaved};
  });
  await check('Offline save keeps edits and succeeds after reconnection', async () => {
    await seedAnnotation(); const before = await labels();
    await send('Network.emulateNetworkConditions', {offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1});
    try {
      await click('#save-button'); await idle();
      assert.equal(await evaluate('window.__agreementState.dirty'), true); assert.equal(await labels(), before);
      assert.match(await evaluate('document.getElementById("notice").textContent'), /edits are still/i);
    } finally {
      await send('Network.emulateNetworkConditions', {offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1});
    }
    await click('#save-button'); await wait('!window.__agreementState.dirty && !window.__agreementState.busy');
    await submit(); const saved = await savedEpisode();
    assert.equal(saved.done, true); assert.equal(saved.subtasks[0].outcome, 'high'); assert.equal(saved.subtasks[1].outcome, 'low');
    await fs.writeFile(path.join(output, 'submitted-labels.json'), JSON.stringify(saved, null, 2));
  });
  await check('Failed primary camera transport shows its reason and promotes the playable view', async () => {
    await send('Fetch.enable', {patterns:[{urlPattern:'*video?episode=1*view=top*',requestStage:'Request'}]});
    await evaluate(`const select=document.getElementById('episode-select'); select.value='1'; select.dispatchEvent(new Event('change',{bubbles:true}));`);
    await wait('window.__agreementState.episode?.id === "1" && !window.__agreementState.busy');
    await wait('document.querySelector(".view-frame.failed")', 'camera failure');
    await wait('window.__agreementState.videos[0].readyState >= 2 && !window.__agreementState.videos[0].error');
    assert.equal(await evaluate('window.__agreementState.videos[0].getAttribute("aria-label")'), 'wrist');
    await click('#play-button'); await sleep(650);
    assert.ok(await evaluate('window.__agreementCtl.currentTime() > 0.2 && !window.__agreementState.videos[0].paused'));
    await evaluate('window.__agreementCtl.pauseAll(); window.scrollTo(0,0)');
    await screenshot('failed-primary-camera', true);
    await send('Fetch.disable');
    return await evaluate('document.getElementById("notice").textContent');
  });
  await check('Dense 390px labels remain bounded with accessible segment text', async () => {
    await send('Emulation.setDeviceMetricsOverride', {width: 390, height: 1000, deviceScaleFactor: 1, mobile: false});
    await fit();
    // Layout fixture only: no save, and the actual end-to-end annotation above was pointer-authored.
    await evaluate(`window.__agreementState.stageIntervals=Array.from({length:22},(_,i)=>({start:i,end:i+1,stage:i%2})); window.__agreementCtl.renderPoints()`);
    await geometry('own-stage-timeline'); await sleep(150);
    const evidence = await evaluate(`(() => {
      const host=document.getElementById('own-stage-timeline'), box=host.getBoundingClientRect();
      return {left:box.left, right:box.right,
        labels:[...host.querySelectorAll('.stage-narrow-label')].map(n=>{const b=n.getBoundingClientRect();return{text:n.textContent,left:b.left,right:b.right}}),
        segments:[...host.querySelectorAll('.stage-timeline-segment')].map(n=>n.getAttribute('aria-label') || n.querySelector('title')?.textContent || '')};
    })()`);
    await screenshot('dense-mobile-labels');
    await fs.writeFile(path.join(output, 'dense-mobile-labels.json'), JSON.stringify(evidence, null, 2));
    assert.ok(evidence.labels.length > 0);
    assert.ok(evidence.labels.every(n => n.left >= evidence.left - 1 && n.right <= evidence.right + 1), JSON.stringify(evidence));
    assert.equal(evidence.segments.length, 22); assert.ok(evidence.segments.every(text => text.length > 15));
    await evaluate('window.__agreementState.stageIntervals=[]; window.__agreementCtl.renderPoints()');
    return evidence;
  });
  await check('Apply stages preserves unsaved intervals, quality, outcomes and Undo across reordering', async () => {
    await send('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await seedAnnotation();
    assert.equal(await evaluate('window.__agreementState.dirty'), true);
    const prior = await (await fetch(`${base}/api/project`)).json();
    const previousSubmitted = await fs.readFile(path.join(prior.labels,'episode_000000.json'),'utf8');
    const beforeOpening = await labels();
    const before = await evaluate(`({stages:__agreementState.stageIntervals, quality:__agreementState.intervals,
      outcomes:GRANA_LAB_INTERACT.outcomeList(), history:__agreementState.history.length,
      stageBrush:__agreementState.selectedSemanticStage, qualityBrush:__agreementState.selectedLabel})`);
    await evaluate('document.querySelector("#stage-settings > summary").scrollIntoView({block:"center",behavior:"instant"})');
    await settle(); await screenshot('stage-settings-entry');
    await click('#stage-settings > summary');
    await wait('document.getElementById("stage-settings").open');
    assert.equal(await labels(), beforeOpening);
    assert.equal(await evaluate('window.__agreementState.dirty'), true);
    assert.equal(await evaluate('document.getElementById("stage-settings").closest("#stage-track-card") !== null'), true);
    await wait('document.querySelectorAll("#stage-settings-rows textarea").length === 2');
    await click('#stage-settings-add');
    await evaluate(`const input=document.querySelector('#stage-settings-rows').lastElementChild.querySelector('textarea');
      input.value='Inspect the closed lid';input.dispatchEvent(new Event('input',{bubbles:true}));
      const first=document.querySelector('#stage-settings-rows textarea');
      first.value='Place the token inside the tray';first.dispatchEvent(new Event('input',{bubbles:true}));`);
    await click('#stage-settings-rows .stage-settings-row:nth-child(2) [data-action="up"]');
    await click('#stage-settings-add');
    await click('#stage-settings-rows .stage-settings-row:last-child [data-action="remove"]');
    await click('#stage-settings-apply');
    await wait('window.__agreementState.episode.stages.length === 3 && !window.__agreementState.busy', 'live stage update');
    await wait('document.querySelectorAll("#grana-brushes-stage button").length === 3');
    assert.deepEqual(await evaluate('window.__agreementState.episode.stages'), ['Close the tray lid','Place the token inside the tray','Inspect the closed lid']);
    const remap = iv => ({...iv, stage:1-iv.stage});
    assert.deepEqual(await evaluate('__agreementState.stageIntervals'),before.stages.map(remap));
    assert.deepEqual(await evaluate('__agreementState.intervals'),before.quality);
    assert.deepEqual(await evaluate('GRANA_LAB_INTERACT.outcomeList()'),before.outcomes.map(remap));
    assert.equal(await evaluate('__agreementState.history.length'),before.history);
    assert.equal(await evaluate('__agreementState.selectedSemanticStage'),1-before.stageBrush);
    assert.equal(await evaluate('__agreementState.selectedLabel'),before.qualityBrush);
    assert.equal(await evaluate('__agreementState.dirty'),false);
    const archived=JSON.parse(await fs.readFile(path.join(prior.labels,'episode_000001.json'),'utf8'));
    assert.equal(archived.done,false);assert.equal(archived.subtasks.length,2);
    assert.equal(archived.subtasks[0].stage,'1');assert.equal(archived.subtasks[0].end_frame,360);
    assert.equal(archived.subtasks[1].outcome,'low');assert.equal(archived.quality.length,1);
    const current=await (await fetch(`${base}/api/project`)).json();
    assert.notEqual(current.revision,prior.revision);assert.notEqual(current.labels,prior.labels);
    assert.deepEqual(current.stages.map(stage=>stage.label),['2','1','3']);
    const migrated=(await (await fetch(`${base}/api/episode?episode=1`)).json()).labels;
    assert.deepEqual(migrated.subtasks,archived.subtasks);assert.deepEqual(migrated.quality,archived.quality);
    assert.deepEqual(migrated.stage_registry,current.stages);assert.equal(migrated.done,false);
    assert.equal(migrated.review_confirmed,false);assert.notEqual(migrated.revision,archived.revision);
    const other=await savedEpisode();
    assert.deepEqual(other.subtasks,JSON.parse(previousSubmitted).subtasks);assert.equal(other.done,false);
    assert.equal(await fs.readFile(path.join(prior.labels,'episode_000000.json'),'utf8'),previousSubmitted);
    const stale=await fetch(`${base}/api/labels?episode=1`,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({subtasks:archived.subtasks,quality:[],annotator:'StaleTab',done:false,review_confirmed:false,
        base_revision:null,project_revision:prior.revision})});
    assert.equal(stale.status,409);
    await evaluate('window.scrollTo(0,0)'); await sleep(200);
    await screenshot('stage-apply-preserves-annotations',true);
    await click('#undo-button');
    assert.deepEqual(await evaluate('__agreementState.intervals'),[]);
    assert.deepEqual(await evaluate('__agreementState.stageIntervals'),before.stages.map(remap));
    assert.deepEqual(await evaluate('GRANA_LAB_INTERACT.outcomeList()'),before.outcomes.map(remap));
    assert.equal(await evaluate('__agreementState.dirty'),true);
    await pickStage(2); await stroke('own-stage-timeline',0,8); await click('#save-button');
    await wait('window.__agreementState.record && !window.__agreementState.dirty && !window.__agreementState.busy');
    await evaluate('history.replaceState(null, "", "#stage-settings")');
    await reload();
    assert.equal(await evaluate('document.getElementById("stage-settings").open'),true);
    assert.equal(await evaluate('window.__agreementState.episode.stages.length'),3);
    assert.equal(await evaluate('window.__agreementState.stageIntervals[0].stage'),2);
    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await wait('document.querySelectorAll("#stage-settings-rows textarea").length === 3');
    await evaluate('document.getElementById("stage-settings").open=false');
    await click('#independent-stage-editor [data-open-stage-settings]');
    assert.equal(await evaluate('document.getElementById("stage-settings").open'),true);
    await evaluate('window.scrollTo(0,0)');await sleep(200);
    const size=await evaluate('({body:document.body.scrollWidth,html:document.documentElement.scrollWidth})');
    assert.ok(size.body<=391 && size.html<=391,JSON.stringify(size));
    await screenshot('inline-stage-editor-mobile',true);
    return {stages:current.stages,annotationsPreserved:true,undoPreserved:true,allEpisodesMigrated:true,staleTabRejected:true};
  });
  await check('Removing a used stage preserves unsaved edits and blocks Apply', async () => {
    await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await idle(); await fit();
    await pickQuality('failure'); await stroke('own-timeline',9,11);
    const before=await labels(), history=await evaluate('__agreementState.history');
    const prior=await (await fetch(`${base}/api/project`)).json();
    const saved=(await (await fetch(`${base}/api/episode?episode=1`)).json()).labels;
    await click('#stage-settings-rows .stage-settings-row:last-child [data-action="remove"]');
    await click('#stage-settings-apply'); await idle();
    await wait('document.getElementById("stage-settings-message").classList.contains("form-error")');
    assert.match(await evaluate('document.getElementById("stage-settings-message").textContent'),/Cannot remove stages S3/);
    assert.equal(await labels(),before);assert.equal(await evaluate('__agreementState.dirty'),true);
    assert.deepEqual(await evaluate('__agreementState.history'),history);
    assert.equal((await (await fetch(`${base}/api/project`)).json()).revision,prior.revision);
    assert.deepEqual((await (await fetch(`${base}/api/episode?episode=1`)).json()).labels,saved);
    await click('[data-language="zh"]');
    assert.match(await evaluate('document.getElementById("stage-settings-message").textContent'),/无法删除阶段 S3/);
    await click('[data-language="en"]'); await click('#stage-settings-reset');
    return {unsavedEditsIntact:true,recipeUnchanged:true,historyIntact:true};
  });
  await check('Apply with a pending span leaves the draft and stage definitions intact', async () => {
    await pickQuality('failure'); await evaluate('__agreementCtl.seek(16)'); await key('s','KeyS');
    await wait('__agreementState.pendingInterval !== null');
    const pending=await evaluate('__agreementState.pendingInterval'), before=await labels();
    const history=await evaluate('__agreementState.history');
    const prior=await (await fetch(`${base}/api/project`)).json();
    await click('#stage-settings-add');
    await evaluate(`const input=document.querySelector('#stage-settings-rows').lastElementChild.querySelector('textarea');
      input.value='Release the tray';input.dispatchEvent(new Event('input',{bubbles:true}));`);
    await click('#stage-settings-apply');await idle();
    await wait('document.getElementById("stage-settings-message").classList.contains("form-error")');
    assert.match(await evaluate('document.getElementById("stage-settings-message").textContent'),/pending span/);
    assert.deepEqual(await evaluate('__agreementState.pendingInterval'),pending);
    assert.deepEqual(await evaluate('__agreementState.history'),history);assert.equal(await labels(),before);
    assert.equal((await (await fetch(`${base}/api/project`)).json()).revision,prior.revision);
    await key('Escape');await click('#stage-settings-reset');
    return {pendingSpanPreserved:true,annotationsPreserved:true,recipeUnchanged:true};
  });
  await check('Reset clears failed outcomes; Undo restores them; new stages start completed', async () => {
    await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await idle(); await fit();
    dialogs.push(true); await click('#clear-button'); await settle();
    await wait('!document.querySelectorAll("#grana-brushes-stage button")[0].disabled');
    await pickStage(0); await stroke('own-stage-timeline',0,6);
    await click('.lab-oc.failed[data-index="0"]');
    assert.equal(await evaluate('GRANA_LAB_INTERACT.outcomeList()[0].outcome'),'failed');
    dialogs.push(true); await click('#clear-button'); await settle();
    assert.deepEqual(await evaluate('GRANA_LAB_INTERACT.outcomeList()'),[]);
    await key('z','KeyZ',2);
    assert.equal(await evaluate('GRANA_LAB_INTERACT.outcomeList()[0].outcome'),'failed');
    dialogs.push(true); await click('#clear-button'); await settle();
    await pickStage(1); await stroke('own-stage-timeline',0,6);
    assert.equal(await evaluate('GRANA_LAB_INTERACT.outcomeList()[0].outcome'),'complete');
    await click('#save-button'); await wait('!__agreementState.busy && !__agreementState.dirty');
    const doc=(await (await fetch(`${base}/api/episode?episode=1`)).json()).labels;
    assert.equal(doc.subtasks[0].outcome,'high');
    return {resetCleared:true,undoRestored:true,newStageCompleted:true};
  });
  await check('Language preference persists and the control fits a narrow screen', async () => {
    await click('[data-language="zh"]');
    await reload();
    assert.equal(await evaluate('document.documentElement.lang'), 'zh-CN');
    assert.equal(await evaluate('document.getElementById("save-button").textContent'), '保存草稿');
    await send('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:false});
    await evaluate('window.scrollTo(0,0)'); await settle();
    const bounds=await evaluate(`(() => {const r=document.querySelector('.language-switch').getBoundingClientRect(); return {left:r.left,right:r.right,top:r.top,width:innerWidth};})()`);
    assert.ok(bounds.left>=0 && bounds.right<=bounds.width && bounds.top>=0, JSON.stringify(bounds));
    await screenshot('mobile-language-switch');
    await click('[data-language="en"]'); await reload();
    assert.equal(await evaluate('document.documentElement.lang'), 'en');
    assert.equal(await evaluate('document.getElementById("save-button").textContent'), 'Save draft');
    await send('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('document.getElementById("stage-track-card").scrollIntoView({block:"center"})');
    await screenshot('english-inline-stages');
    return {persisted:true,mobileControl:bounds};
  });
  await check('No uncaught browser JavaScript exceptions', async () => {assert.deepEqual(report.errors, []);});
  report.finished = new Date().toISOString(); report.passed = report.checks.every(item => item.passed);
  console.log(`RESULT ${report.checks.filter(item => item.passed).length}/${report.checks.length} checks passed. Artifacts: ${output}`);
  process.exitCode = report.passed ? 0 : 1;
}

async function cleanup() {
  if (cleanupPromise) return cleanupPromise;
  cleanupPromise = (async () => {
    if (socket) socket.close();
    if (child && child.exitCode === null && child.signalCode === null) {
      const exited = once(child, 'exit'); child.kill('SIGTERM');
      await Promise.race([exited, sleep(3000)]);
      if (child.exitCode === null && child.signalCode === null) {child.kill('SIGKILL'); await exited;}
    }
    await fs.writeFile(path.join(output, 'chrome.log'), chromeLog.join(''));
  })();
  return cleanupPromise;
}
process.once('SIGTERM', () => {cleanup().finally(() => process.exit(143));});
main().catch(async error => {
  report.fatal = error.stack; console.error(error.stack);
  if (send) {
    report.fatalScreenshot = await screenshot('fatal').catch(() => null);
    report.fatalState = await stateSnapshot().catch(() => null);
  }
  process.exitCode = 1;
}).finally(async () => {
  await cleanup();
  report.chromeStopped = !child || child.exitCode !== null || child.signalCode !== null;
  await fs.writeFile(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2));
});
