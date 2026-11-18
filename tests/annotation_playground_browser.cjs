// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const [base, chrome, output] = process.argv.slice(2);
let child, socket;
const errors = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) { const result = await check(); if (result) return result; await sleep(80); }
  throw Error('Browser condition timed out');
}
async function cleanup() {
  socket?.close();
  if (child && child.exitCode === null) { child.kill('SIGTERM'); await once(child, 'exit'); }
}
process.on('SIGTERM', () => cleanup().finally(() => process.exit(143)));
(async () => {
  await fs.mkdir(output, {recursive:true});
  const profile = await fs.mkdtemp(path.join(output, 'chrome-profile-'));
  child = spawn(chrome, ['--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
    '--disable-background-networking','--disable-extensions','--disable-component-update',
    '--no-first-run','--no-default-browser-check','--remote-debugging-port=0',
    `--user-data-dir=${profile}`, 'about:blank'], {stdio:'ignore'});
  const port = await until(async () => {try {return Number((await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);}catch{return null;}});
  const target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(x=>x.type==='page');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await once(socket, 'open');
  let id = 0;
  const waiting = new Map();
  socket.addEventListener('message', event => {
    const data = JSON.parse(event.data);
    if (data.id) { const item = waiting.get(data.id); waiting.delete(data.id); data.error ? item.reject(Error(JSON.stringify(data.error))) : item.resolve(data.result); }
    if (data.method==='Runtime.exceptionThrown') errors.push(data.params.exceptionDetails);
  });
  const send = (method, params={}) => new Promise((resolve,reject)=>{const n=++id;waiting.set(n,{resolve,reject});socket.send(JSON.stringify({id:n,method,params}));});
  const evaluate = async expression => { const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true}); if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value; };
  const wait = expression => until(()=>evaluate(expression));
  const clickText = text => evaluate(`(()=>{const button=[...document.querySelectorAll('#api-rows button')].find(b=>b.textContent===${JSON.stringify(text)}&&!b.disabled);if(!button)throw Error('Button missing');button.click();})()`);
  const status = () => evaluate("fetch('/api/status').then(r=>r.json())");
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:base});
  await wait("document.querySelector('#task-name')?.textContent==='demo'");
  if (!(await status()).models.length) {
    await wait("document.getElementById('models-line').textContent.includes('No annotation provider configured')");
    await evaluate("document.querySelector('[data-mode=api] a').click()");
    await wait("!document.getElementById('view-api').hidden && document.getElementById('api-model').textContent.includes('No annotation provider configured')");
    assert.equal(await evaluate("[...document.querySelectorAll('#api-rows button')].every(button=>button.disabled)"),true);
    await evaluate("document.querySelector('[data-language=zh]').click()");
    await wait("document.getElementById('api-model').textContent.includes('尚未配置标注服务')");
    await evaluate("document.querySelector('[data-language=en]').click()");
    await evaluate("document.querySelector('[data-view=human]').click()");
    await wait("!document.getElementById('view-human').hidden && document.getElementById('human-open').href.includes('?lang=en')");
    assert.deepEqual(errors,[]);
    await fs.writeFile(path.join(output,'report.json'),JSON.stringify({passed:true,checks:['no provider defaults','generation disabled','human labeling available','bilingual provider notice'],errors},null,2));
    return;
  }
  const single = (await status()).routes.api.episodes.length === 1;
  if (single) {
    assert.equal(await evaluate("document.getElementById('home-seed').textContent.includes('at least two episodes')"),true);
    await evaluate("document.querySelector('[data-mode=seed] a').click()");
    await wait("!document.getElementById('view-seed').hidden && document.getElementById('seed-unavailable')");
    assert.equal(await evaluate("document.querySelectorAll('#stepper button').length"),0);
    assert.equal(await evaluate("/[\\u4e00-\\u9fff]/.test(document.getElementById('view-seed').innerText)"),false);
    assert.equal(await evaluate("!!document.querySelector('#seed-unavailable a[href=\"#/api\"]')"),true);
    await evaluate("document.querySelector('[data-language=zh]').click()");
    await wait("document.getElementById('seed-unavailable')?.innerText.includes('仍可独立使用人工标注和纯 API 标注')");
    await evaluate("document.querySelector('[data-language=en]').click()");
    await wait("document.getElementById('seed-unavailable')?.innerText.includes('at least two episodes')");
    const notice = await send('Page.captureScreenshot',{format:'png'});
    await fs.writeFile(path.join(output,'single-episode-seed.png'),Buffer.from(notice.data,'base64'));
    await evaluate("document.querySelector('#seed-unavailable a[href=\"#/human\"]').click()");
    await wait("!document.getElementById('view-human').hidden && document.getElementById('human-open').href.includes('?lang=en')");
    await evaluate("document.querySelector('[data-view=home]').click()");
    await wait("!document.getElementById('view-home').hidden");
  }
  if (!single) {
    await evaluate("document.querySelector('[data-mode=seed] a').click()");
    await wait("document.querySelector('#step-example[data-state=stale]')?.innerText.includes('Seed labels changed (0); build the example again.')");
    assert.equal(await evaluate("/[\\u4e00-\\u9fff]/.test(document.getElementById('view-seed').innerText)"),false);
    await evaluate("document.querySelector('[data-language=zh]').click()");
    await wait("document.getElementById('step-example')?.innerText.includes('Seed 标注已更改（0）；请重新生成示例。')");
    await evaluate("document.querySelector('[data-language=en]').click()");
    await wait("document.getElementById('step-example')?.innerText.includes('Seed labels changed')");
  }
  await evaluate("document.querySelector('[data-mode=api] a').click()");
  await wait("!document.getElementById('view-api').hidden && document.querySelector('#api-model input')");
  assert.equal(await evaluate("document.querySelector('#api-model input').checked"),false);
  assert.equal(await evaluate("/[\\u4e00-\\u9fff]/.test(document.getElementById('view-api').innerText)"),false);
  await clickText('Generate stages');
  await until(async()=> (await status()).routes.api.episodes[0].models.default.job?.action_quality_status==='default_normal_as_not_requested');
  await wait("[...document.querySelectorAll('#api-rows button')].some(b=>b.textContent==='Add AQ'&&!b.disabled)");
  await clickText('Add AQ');
  await until(async()=> (await status()).routes.api.episodes[0].models.default.job?.action_quality_status==='annotated');
  await wait("![...document.querySelectorAll('#api-rows button')].some(b=>b.textContent==='Add AQ')");
  await evaluate("document.querySelector('#api-model input').click()");
  assert.equal(await evaluate("document.querySelector('#api-model input').checked"),true);
  if (!single) {
    await clickText('Generate stages');
    await until(async()=> (await status()).routes.api.episodes[1].models.default.job?.action_quality_status==='annotated');
  }
  const english = await send('Page.captureScreenshot',{format:'png'});
  await fs.writeFile(path.join(output,'playground-english.png'),Buffer.from(english.data,'base64'));
  await evaluate("document.querySelector('[data-language=zh]').click()");
  await wait("document.documentElement.lang==='zh-CN' && document.getElementById('view-api').innerText.includes('完成 stage 后再标 AQ')");
  const chinese = await send('Page.captureScreenshot',{format:'png'});
  await fs.writeFile(path.join(output,'playground-chinese.png'),Buffer.from(chinese.data,'base64'));
  await evaluate("document.querySelector('[data-view=home]').click()");
  await wait("!document.getElementById('view-home').hidden");
  assert.deepEqual(errors,[]);
  const checks = ['stage default','add AQ','AQ checkbox','English first','Chinese switch','home navigation'];
  if (single) checks.push('single episode seed isolation','bilingual seed notice','independent human navigation');
  else checks.push('stale seed example notice in English and Chinese');
  await fs.writeFile(path.join(output,'report.json'),JSON.stringify({passed:true,checks,errors},null,2));
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(cleanup);
