/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* UI-only workflow context. Canonical annotations stay in the shipped editor. */
document.addEventListener('DOMContentLoaded', () => {
  'use strict';
  const mode=document.body.dataset.playgroundMode, given=JSON.parse(document.body.dataset.playgroundPorts||'{}');
  const ports={home:given.overview,human:given.human,api:given.api,seeded:given.seeded,seed:given.seed};
  // Each label page belongs to one view of the home page: the seed page and the seeded drafts to Seed examples.
  const view={human:'human',api:'api',seeded:'seed',seed:'seed'}[mode];
  const params=new URLSearchParams(location.search), embedded=window.top!==window;
  const $=id=>document.getElementById(id), state=()=>window.__agreementState;
  let snapshot=null, offline=false, destination=null, lastRender='', saving=false;
  const copy={
    en:{view_human:'Human labeling',view_api:'API drafts',view_seed:'Seed examples',human:'Human labeling',api:'API drafts',seeded:'Seeded drafts',seed:'Seeds & held-out',episode:'Episode',
      humanTitle:'Label this episode yourself',apiTitle:'Review the API draft',seededTitle:'Review the seeded draft',seedTitle:'Label seeds and held-out episodes',
      humanDescription:'Mark subtasks and action quality, review the full video, then submit.',apiDescription:'The model provides a first draft. Check it against the video and correct it before submitting.',
      seededDescription:'Drafted with your seeds as examples. Check it against the video before submitting.',seedDescription:'Submitted episodes here become seeds, or held-out episodes that score the API.',
      watch:'Watch & label',submit:'Submit',generate:'Generate',importStep:'Import',review:'Review & submit',example:'Build example',choose:'Choose in Seed examples',
      next:'Next: review the whole video, resolve outcomes, and submit.',seedNext:'Next: finish this episode, confirm the full review, and submit. Drafts cannot be seeds.',
      submitted:'Submitted. Go back for the next episode.',seedDone:'Submitted. Choose it as a seed or held-out episode in Seed examples.',
      generateTitle:'No draft for this episode yet',empty:'Generate this episode in the playground, then import it to review it here.',generating:'The model is generating this draft. Follow it in the playground.',
      failed:'Generation did not finish. Inspect it in the playground before continuing.',imported:'A result is ready. Import it in the playground to review it here.',
      generateAction:'Go to generation →',checking:'Checking workflow status…',unavailable:'Workflow status is unavailable. Your editor remains open; go back to the playground to reconnect.',
      leaveTitle:'Keep your changes before leaving',leaveBody:'Save this draft before switching workflows. Saving does not submit your review.',unfinished:'Finish or cancel the pending span and apply or reset stage definitions before leaving. Your edits are still here.',
      busy:'An operation is still running. Stay here until it finishes.',stay:'Keep editing',saveLeave:'Save draft & continue',saveFailed:'The draft was not saved. Check the editor message and resolve the issue before leaving.'},
    zh:{view_human:'人工标注',view_api:'API 草稿',view_seed:'Seed 示例',human:'人工标注',api:'API 草稿',seeded:'Seed 流程草稿',seed:'Seed 与留出片段',episode:'片段',
      humanTitle:'亲自标注这个片段',apiTitle:'审核 API 草稿',seededTitle:'审核 Seed 流程草稿',seedTitle:'标注 Seed 与留出片段',
      humanDescription:'标记子阶段和动作质量，复核完整视频后提交。',apiDescription:'模型已提供初稿。请对照视频检查、修正，再提交。',
      seededDescription:'这份草稿以你的 Seed 为示例生成。提交前请对照视频检查。',seedDescription:'这里提交的片段可以作为 Seed，或作为给 API 打分的留出片段。',
      watch:'观看并标注',submit:'提交',generate:'生成',importStep:'导入',review:'检查并提交',example:'生成示例',choose:'在 Seed 示例中选择',
      next:'下一步：检查完整视频、确认阶段结果，然后提交。',seedNext:'下一步：完成这个片段，确认已复核完整视频，再提交。草稿不能作为 Seed。',
      submitted:'已提交。返回处理下一个片段。',seedDone:'已提交。可在 Seed 示例中把它选为 Seed 或留出片段。',
      generateTitle:'这个片段还没有草稿',empty:'请先在试验场中生成这个片段，再导入后在这里审核。',generating:'模型正在生成草稿。可在试验场中查看进度。',
      failed:'本次生成未完成。请先在试验场中检查，再继续。',imported:'结果已生成。请在试验场中导入后在这里审核。',
      generateAction:'前往生成 →',checking:'正在检查流程状态…',unavailable:'暂时无法读取流程状态。编辑器仍可使用；可以返回试验场重新连接。',
      leaveTitle:'离开前保留你的修改',leaveBody:'切换流程前先保存草稿。保存草稿不会提交审核。',unfinished:'请先结束或取消未闭合区间，并应用或重置阶段定义。当前修改仍保留。',
      busy:'当前操作尚未结束，请稍候再离开。',stay:'继续编辑',saveLeave:'保存草稿并继续',saveFailed:'草稿未保存。请检查编辑器提示，解决问题后再离开。'}
  };
  const lang=()=>window.GRANA_I18N?.language==='zh'?'zh':'en', t=key=>copy[lang()][key];
  if(['en','zh'].includes(params.get('lang')))window.GRANA_I18N?.setLanguage(params.get('lang'));
  const current=()=>{const id=Number(state()?.episode?.id);return Number.isInteger(id)?id:Number.isInteger(Number(params.get('episode')))&&params.get('episode')!==null?Number(params.get('episode')):null;};
  /** Episodes of a page's route, from the home page's status; null until known. */
  function routeEpisodes(flow){
    const routes=snapshot?.routes;if(!routes)return null;
    if(flow==='api')return routes.api.episodes.map(row=>row.episode);
    if(flow==='seeded'||flow==='seed')return routes.seeded.steps.choose.episodes.map(row=>row.episode);
    return null;
  }
  function row(){
    const routes=snapshot?.routes,index=current();if(!routes||index===null)return null;
    const rows=mode==='api'?routes.api.episodes:mode==='seeded'?[...routes.seeded.steps.apply.episodes,...routes.seeded.steps.validate.episodes]:[];
    return rows.find(item=>item.episode===index)||null;
  }
  function url(flow){
    const u=new URL(location.href),index=current(),episodes=routeEpisodes(flow);u.port=ports[flow];u.pathname='/';u.search='';u.hash='';
    if(flow!=='home'&&index!==null&&(episodes===null||episodes.includes(index)))u.searchParams.set('episode',String(index));
    u.searchParams.set('lang',lang());if(flow==='home')u.hash='/'+view;return u;
  }
  const element=(tag,id)=>{const n=document.createElement(tag);if(id)n.id=id;return n;};
  const home=element('a','playground-home');home.dataset.pgNavigate='';home.href=url('home');
  const top=document.querySelector('.topnav'), inner=top.querySelector('.topnav-inner');inner.insertBefore(home,inner.querySelector('.lab-title').nextSibling);
  const brand=inner.querySelector('.wordmark');brand.dataset.pgNavigate='';brand.href=url('home');
  const nav=element('nav','playground-nav');nav.setAttribute('aria-label','Annotation workflows');
  for(const flow of ['human','api','seeded','seed']){const a=element('a');a.dataset.pgFlow=flow;a.dataset.pgNavigate='';if(flow===mode)a.setAttribute('aria-current','page');nav.append(a);}
  const context=element('span','playground-context');nav.append(context);top.append(nav);
  const heading=element('section','playground-heading'), title=element('h1'),description=element('p'),steps=element('ol','playground-steps');heading.append(title,description,steps);
  const next=element('section','playground-next'),nextText=element('p','playground-next-text'),nextLink=element('a','playground-next-link');nextLink.dataset.pgNavigate='';next.append(nextText,nextLink);document.querySelector('.app-shell').prepend(heading,next);
  const dialog=element('dialog','playground-leave');dialog.setAttribute('aria-labelledby','playground-leave-title');dialog.innerHTML='<h2 id="playground-leave-title"></h2><p id="playground-leave-message"></p><p id="playground-leave-error" role="alert" hidden></p><div class="pg-actions"><button id="playground-stay" type="button" autofocus></button><button id="playground-save-leave" type="button"></button></div>';document.body.append(dialog);
  function stageDirty(){const reset=$('stage-settings-reset');return !!document.querySelector('#stage-settings-rows textarea')&&!!reset&&!reset.disabled;}
  function blocked(){return state()?.busy||state()?.pendingInterval||state()?.pendingStageInterval||stageDirty()||window.GRANA_LAB_INTERACT?.hasActiveDrag();}
  const STEPS={human:['watch','submit'],api:['generate','importStep','review'],seeded:['example','generate','importStep','review'],seed:['watch','submit','choose']};
  function render(){
    const s=state(),index=current(),item=row(),done=s?.record?.status==='submitted'&&!s?.dirty;
    const token=JSON.stringify([lang(),index,s?.record?.status,s?.dirty,s?.pendingInterval,s?.pendingStageInterval,snapshot?.updated_at,offline]);if(token===lastRender)return;lastRender=token;
    const u=new URL(location.href);u.searchParams.set('lang',lang());if(index!==null)u.searchParams.set('episode',String(index));if(u.href!==location.href)history.replaceState(null,'',u);
    home.textContent='← '+t('view_'+view);home.href=url('home');brand.href=url('home');context.textContent=index===null?'':t('episode')+' '+index;
    nav.querySelectorAll('[data-pg-flow]').forEach(a=>{a.textContent=t(a.dataset.pgFlow);a.href=url(a.dataset.pgFlow);});
    title.textContent=t(mode+'Title');description.textContent=t(mode+'Description');document.title='Grana · '+t(mode);
    const keys=STEPS[mode],generated=mode==='api'||mode==='seeded',at=done||generated&&item?.label&&item.label!=='none'?keys.length-1:generated?keys.length-(item?.status==='generated'?2:3):0;
    steps.replaceChildren(...keys.map((key,i)=>{const li=element('li');li.textContent=(i+1)+'. '+t(key);if(i===Math.max(0,at))li.setAttribute('aria-current','step');return li;}));
    let text=done?(mode==='seed'?'seedDone':'submitted'):mode==='seed'?'seedNext':'next',action=null,link=url('home');
    // An episode without a draft in a generated route is a generation step, not an unlabeled human editor.
    const awaiting=generated&&!!snapshot&&(!item||item.label==='none')&&!s?.dirty&&!s?.pendingInterval&&!s?.pendingStageInterval;
    document.body.classList.toggle('playground-awaiting-draft',awaiting);
    if(generated&&!snapshot)text=offline?'unavailable':'checking';
    if(awaiting){
      title.textContent=t('generateTitle');action='generateAction';
      text=['running','queued'].includes(item?.status)?'generating':item?.status==='failed'?'failed':item?.status==='generated'?'imported':'empty';
    }
    nextText.textContent=t(text);nextLink.textContent=action?t(action):t('view_'+view)+' →';nextLink.href=link;
    if(embedded)document.querySelectorAll('[data-pg-navigate]').forEach(a=>{a.target='_blank';a.rel='noopener';});
  }
  function openDestination(){const href=destination;destination=null;dialog.close();location.assign(href);}
  document.addEventListener('click',event=>{
    const a=event.target.closest('a[data-pg-navigate]');if(!a)return;
    // Resolve from the live editor, including a just-completed episode switch.
    lastRender='';render();
    if(embedded||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||event.button!==0)return;
    if(a.getAttribute('aria-current')==='page'){event.preventDefault();return;}
    if(!state()?.dirty&&!blocked())return;
    event.preventDefault();destination=a.href;$('playground-leave-title').textContent=t('leaveTitle');$('playground-leave-message').textContent=t(state()?.busy?'busy':blocked()?'unfinished':'leaveBody');$('playground-stay').textContent=t('stay');$('playground-save-leave').textContent=t('saveLeave');$('playground-save-leave').disabled=!!blocked()||!state()?.session?.annotator_id||!!state()?.saveConflict||!!state()?.sessionConflict;$('playground-leave-error').hidden=true;dialog.showModal();
  });
  $('playground-stay').onclick=()=>{destination=null;dialog.close();};
  dialog.addEventListener('cancel',()=>{destination=null;});
  $('playground-save-leave').onclick=async()=>{if(saving||blocked())return;saving=true;$('playground-save-leave').disabled=true;const ctl=window.__agreementCtl;const ok=await ctl.withBusy(()=>ctl.saveRecord('draft'));saving=false;if(ok&&!state().dirty&&!blocked())openDestination();else{$('playground-leave-error').hidden=false;$('playground-leave-error').textContent=t('saveFailed');$('playground-save-leave').disabled=!!blocked();}};
  window.addEventListener('grana:languagechange',()=>{lastRender='';render();});
  async function poll(){try{const r=await fetch('/playground-status',{cache:'no-store',signal:AbortSignal.timeout(4500)});if(!r.ok)throw new Error(String(r.status));snapshot=await r.json();offline=false;}catch{offline=true;}lastRender='';render();}
  render();poll();setInterval(render,350);setInterval(poll,4000);
});
