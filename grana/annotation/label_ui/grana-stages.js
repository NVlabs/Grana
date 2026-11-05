/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* Edit the shared stage vocabulary, preserving the previous recipe and labels. */
(() => {
  "use strict";
  const I18N = window.GRANA_I18N || {text: (n, s) => { n.textContent = s; }, attr: (n, k, s) => n.setAttribute(k, s), node: s => document.createTextNode(s), source: n => n.textContent, t: s => s};
  const $ = id => document.getElementById(id);
  const api = () => window.GRANA_LAB_API, ctl = () => window.__agreementCtl;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text != null) I18N.text(node, text);
    if (className) node.className = className;
    return node;
  };
  let baseline = null, rows = [], dirty = false, applying = false;
  const disable = (node, value) => { if (node.disabled !== value) node.disabled = value; };
  const message = (text, error = false) => {
    I18N.text($("stage-settings-message"), text);
    $("stage-settings-message").classList.toggle("form-error", error);
  };
  function controls() {
    const busy = applying || Boolean(window.__agreementState?.busy);
    const count = `${rows.length} stages`;
    I18N.text($("stage-settings-count"), count);
    disable($("stage-settings-apply"), busy || !dirty || !baseline?.stages_editable);
    disable($("stage-settings-reset"), busy || !dirty);
    [$("stage-settings-add"), ...document.querySelectorAll("[data-add-stage]")].forEach(button => disable(button, busy || rows.length >= 26 || !baseline?.stages_editable));
    $("stage-settings-rows").querySelectorAll("textarea,button").forEach(node => {
      disable(node, busy || node.dataset.unavailable === "true" || !baseline?.stages_editable);
    });
  }
  function changed() {
    dirty = JSON.stringify(rows) !== JSON.stringify(baseline.stages);
    message(dirty ? "Stage definitions have not been applied." : "");
    controls();
  }
  function render() {
    const list = $("stage-settings-rows"); list.replaceChildren();
    rows.forEach((stage, index) => {
      const row = element("div", null, "stage-settings-row");
      const field = element("label", null, "field stage-settings-description");
      field.append(element("span", `S${index + 1} · Stage description`));
      const input = element("textarea"); input.rows = 2; input.required = true; input.maxLength = 4000;
      input.value = stage.prompt; I18N.attr(input, "placeholder", "e.g. Grasp the bottom edge of the towel with both hands");
      I18N.attr(input, "aria-label", `S${index + 1} stage description`);
      input.addEventListener("input", () => { stage.prompt = input.value; changed(); });
      field.append(input); row.append(field);
      const actions = element("div", null, "stage-settings-order");
      for (const [action, text, title, unavailable] of [
        ["up", "↑", "Move stage up", index === 0], ["down", "↓", "Move stage down", index === rows.length - 1],
        ["remove", "Remove", "Remove stage", rows.length === 1],
      ]) {
        const button = element("button", text, "button compact"); button.type = "button";
        button.dataset.action = action; button.dataset.unavailable = String(unavailable);
        I18N.attr(button, "aria-label", `${title} S${index + 1}`);
        button.addEventListener("click", () => {
          if (action === "remove") rows.splice(index, 1);
          else { const to = index + (action === "up" ? -1 : 1); [rows[index], rows[to]] = [rows[to], rows[index]]; }
          render(); changed();
        });
        actions.append(button);
      }
      row.append(actions); list.append(row);
    });
    controls();
  }
  function load(project) {
    baseline = project; rows = project.stages.map(stage => ({...stage})); dirty = false;
    render();
    message(project.stages_editable ? "" : "Launch with grana label --config to save stage definitions.", !project.stages_editable);
  }
  function openSettings() {
    ctl()?.pauseAll();
    const settings = $("stage-settings");
    settings.open = true;
    settings.querySelector("summary").focus({preventScroll: true});
    settings.scrollIntoView({block: "start", behavior: "instant"});
  }
  function remapHistory(history, oldLabels, newLabels, reopened) {
    const mapping = oldLabels.map(label => newLabels.indexOf(label));
    // Removed stages can still occur in old Undo snapshots. Keep only snapshots
    // that can be restored without resurrecting a deleted stage.
    return history.flatMap(entry => {
      let valid = true;
      const copy = JSON.parse(JSON.stringify(entry), (key, value) => {
        if ((key === "stage" || key === "selectedSemanticStage") && Number.isInteger(value)) {
          if (mapping[value] < 0 || mapping[value] === undefined) { valid = false; return null; }
          return mapping[value];
        }
        return value;
      });
      if (reopened) { copy.reviewConfirmed = false; copy.recordStatus = "draft"; }
      return valid ? [copy] : [];
    });
  }
  async function apply(event) {
    event.preventDefault();
    if (!dirty || applying || window.__agreementState?.busy || !baseline?.stages_editable) return;
    const stages = rows.map(stage => ({...stage, prompt: stage.prompt.trim()}));
    if (stages.some(stage => !stage.prompt) || new Set(stages.map(stage => stage.prompt)).size !== stages.length) {
      message("Give each stage a different description.", true); return;
    }
    applying = true; controls();
    const c = ctl(), state = window.__agreementState, episode = state.episode?.id, time = c.currentTime();
    const previous = {history: JSON.parse(JSON.stringify(state.history)), labels: [...(state.episode?.stage_labels || [])],
      stage: state.selectedSemanticStage, quality: state.selectedLabel};
    let applied = false, reopened = false;
    const success = await c.withBusy(async () => {
      // A pending span or save conflict aborts before any definitions change.
      if (state.pendingInterval || state.pendingStageInterval) throw new Error("End or cancel the pending span first.");
      const retained = new Set(stages.map(stage => stage.label));
      const removed = [...new Set(state.stageIntervals.filter(iv => !retained.has(previous.labels[iv.stage])).map(iv => `S${iv.stage + 1}`))];
      if (removed.length) throw new Error(`Cannot remove stages ${removed.join(", ")}: episode ${episode} still has intervals assigned to them. Reassign or remove those intervals first.`);
      await c.saveIfDirty();
      const result = await api().updateStages(stages, baseline.revision);
      applied = result.changed; reopened = result.reopened === true;
      await c.refreshBootstrap();
      await c.loadContext(episode || state.episodes[0]?.id, state.scheme?.id || "stage_aq");
      c.seek(time);
      load(api().project());
      message(!applied ? "Stage definitions are unchanged." : "Stages applied. Existing intervals, quality spans and outcomes are preserved as drafts. Review labels and progress parameters before submitting.");
      c.showNotice(`Applied ${stages.length} stages. Your annotations are preserved.`);
    });
    const currentLabels = state.episode?.stage_labels || previous.labels;
    state.history = remapHistory(previous.history, previous.labels, currentLabels, reopened);
    state.selectedSemanticStage = Number.isInteger(previous.stage) ? currentLabels.indexOf(previous.labels[previous.stage]) : null;
    if (state.selectedSemanticStage === -1) state.selectedSemanticStage = null;
    state.selectedLabel = previous.quality;
    $("independent-stage-select").value = state.selectedSemanticStage === null ? "" : String(state.selectedSemanticStage);
    c.renderSelectedLabel(); c.updateControls();
    applying = false;
    if (!success) message(applied ? "The definitions were saved, but the page could not reload. Refresh to continue." : I18N.source($("notice")) || "Could not apply stages. Existing labels are unchanged.", true);
    controls();
  }
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-open-stage-settings]").forEach(link => {
      link.addEventListener("click", event => {
        event.preventDefault();
        history.replaceState(null, "", "#stage-settings");
        openSettings();
      });
    });
    const followHash = () => { if (location.hash === "#stage-settings") openSettings(); };
    window.addEventListener("hashchange", followHash);
    followHash();
    $("stage-settings-form").addEventListener("submit", apply);
    const addStage = () => {
      if (!baseline || rows.length >= 26) return;
      let label = 1;
      while ([...baseline.stages, ...rows].some(stage => stage.label === String(label))) label++;
      openSettings();
      rows.push({label: String(label), prompt: ""}); render(); changed();
      $("stage-settings-rows").lastElementChild.querySelector("textarea").focus({preventScroll: true});
      $("stage-settings-rows").lastElementChild.scrollIntoView({block: "nearest", behavior: "instant"});
    };
    $("stage-settings-add").addEventListener("click", addStage);
    document.querySelectorAll("[data-add-stage]").forEach(button => button.addEventListener("click", addStage));
    $("stage-settings-reset").addEventListener("click", () => load(api().project()));
    setInterval(() => {
      const project = api()?.project();
      if (!project || applying) return;
      if (!baseline || (!dirty && project.revision !== baseline.revision)) load(project);
      controls();
    }, 300);
  });
  window.addEventListener("beforeunload", event => {if (dirty) {event.preventDefault(); event.returnValue = "";}});
})();
