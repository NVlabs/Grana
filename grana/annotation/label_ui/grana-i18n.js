/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* Presentation-only bindings. A language change updates text/attributes in place:
 * no editor re-render, API request, form reset, or change to canonical label values.
 * Dynamic UI writers opt in via text()/attr(); user-authored content stays raw.
 */
(() => {
  'use strict';
  const KEY = 'grana-label.language';
  let language = 'en';
  // English is the primary experience, irrespective of the browser's locale.
  try { const saved = localStorage.getItem(KEY); if (['en', 'zh'].includes(saved)) language = saved; } catch (_) { /* English remains the default */ }
  const pairs = window.GRANA_MESSAGES;
  const exact = new Map(pairs.filter(([en]) => !/\{\d+\}/.test(en)));
  const escape = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const templates = pairs.filter(([en]) => /\{\d+\}/.test(en)).sort((a, b) => b[0].replace(/\{\d+\}/g, '').length - a[0].replace(/\{\d+\}/g, '').length).map(([en, zh]) => {
    const indices = [...en.matchAll(/\{(\d+)\}/g)].map(m => Number(m[1]));
    return {regex: new RegExp('^' + en.split(/\{\d+\}/).map(escape).join('([\\s\\S]*?)') + '$'), zh, indices};
  });
  function translate(source, depth = 0) {
    const raw = String(source ?? ''), value = raw.trim(), before = raw.slice(0, raw.indexOf(value)), after = raw.slice(before.length + value.length);
    if (language === 'en' || !value || depth > 3) return raw;
    if (exact.has(value)) return before + exact.get(value) + after;
    for (const {regex, zh, indices} of templates) {
      const hit = value.match(regex);
      if (hit) return before + zh.replace(/\{(\d+)\}/g, (_, n) => hit[indices.indexOf(Number(n)) + 1] ?? '') + after;
    }
    // UI sentences are often composed from independent status fragments.
    // Never run this on task descriptions, stage prompts, names or input values.
    if (value.includes('\n')) return before + value.split('\n').map(s => translate(s, depth + 1)).join('\n') + after;
    if (value.includes(' · ')) return before + value.split(' · ').map(s => translate(s, depth + 1)).join(' · ') + after;
    return raw;
  }
  const sources = new WeakMap(), attributes = new WeakMap();
  function text(node, value) {
    const raw = String(value ?? '');
    const translated = translate(raw);
    if (node.textContent !== translated) node.textContent = translated;
    if (node.firstChild?.nodeType === 3 && node.childNodes.length === 1) sources.set(node.firstChild, raw);
    return raw;
  }
  function textNode(value) { const raw = String(value); const node = document.createTextNode(translate(raw)); sources.set(node, raw); return node; }
  function attr(node, name, value) {
    if (!['title', 'aria-label', 'aria-valuetext', 'placeholder', 'data-tip', 'data-outcome-label'].includes(name)) { node.setAttribute(name, value); return value; }
    let map = attributes.get(node); if (!map) { map = new Map(); attributes.set(node, map); }
    const raw = String(value ?? ''); map.set(name, raw);
    const translated = translate(raw); if (node.getAttribute(name) !== translated) node.setAttribute(name, translated);
    return raw;
  }
  function source(node) {
    if (node.nodeType === 3) return sources.get(node) ?? node.textContent;
    return [...node.childNodes].map(child => source(child)).join('');
  }
  function refresh() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('.grana-brushes-hint').forEach(node => node.dataset.lockedHint = ' · ' + translate('Reopen the submitted episode to edit'));
    const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.nodeType === 3 && sources.has(node)) { const next = translate(sources.get(node)); if (node.data !== next) node.data = next; }
      const map = attributes.get(node);
      if (map) for (const [name, raw] of map) { const next = translate(raw); if (node.getAttribute(name) !== next) node.setAttribute(name, next); }
    }
    document.querySelectorAll('[data-language]').forEach(b => { b.setAttribute('aria-pressed', String(b.dataset.language === language)); const name = window.GRANA_LANGUAGE_NAMES?.[b.dataset.language]; if (name && b.textContent !== name) b.textContent = name; });
  }
  function setLanguage(next, persist = true) {
    if (!['en', 'zh'].includes(next)) return;
    language = next;
    if (persist) try { localStorage.setItem(KEY, next); } catch (_) { /* preference remains usable in memory */ }
    refresh();
    window.dispatchEvent(new CustomEvent('grana:languagechange', {detail: {language}}));
  }
  window.GRANA_I18N = {text, node: textNode, attr, source, t: translate, setLanguage, get language() { return language; }};
  // Bind only the original markup before any editor scripts create user content.
  // Script is loaded with defer before the editor scripts.
  const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    if (node.parentElement?.closest('script,style,textarea,code,kbd,[translate="no"]')) continue;
    const raw = node.data;
    sources.set(node, raw);
  }
  document.querySelectorAll('[title],[aria-label],[aria-valuetext],[placeholder],[data-tip]').forEach(node => {
    if (node.closest('[translate="no"]')) return;
    for (const name of ['title','aria-label','aria-valuetext','placeholder','data-tip']) if (node.hasAttribute(name)) attr(node, name, node.getAttribute(name));
  });
  document.querySelectorAll('[data-language]').forEach(b => b.addEventListener('click', () => setLanguage(b.dataset.language)));
  window.addEventListener('storage', e => { if (e.key === KEY && ['en','zh'].includes(e.newValue)) setLanguage(e.newValue, false); });
  refresh();
})();
