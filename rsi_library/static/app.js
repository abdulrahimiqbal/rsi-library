"use strict";

const el = (id) => document.getElementById(id);
const kinds = ["target", "proposer", "judge", "search", "harness"];
const roles = {
  target: "What can change?",
  proposer: "Who proposes the next change?",
  judge: "How is a candidate scored?",
  search: "Which parent gets another chance?",
  harness: "Where are candidates tested?",
};
const state = {catalog: null, source: null, base: {}, control: null, latest: null, busy: false};
const pretty = (value) => JSON.stringify(value, null, 2);
const clone = (value) => JSON.parse(JSON.stringify(value));
const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const has = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

function element(tag, className, label) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (label !== undefined) node.textContent = label;
  return node;
}

function svg(tag, attrs, label) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  return node;
}

function report(message = "", error = false) {
  const output = el("form-message");
  output.textContent = message;
  output.classList.toggle("error", error);
}

function readJSON(field, label) {
  const raw = el(field).value.trim();
  if (!raw) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    el(field).closest("details").open = true;
    el(field).focus();
    throw new Error(`${label} has invalid JSON. Check commas, quotes, and brackets.`);
  }
}

function validateConfig(config) {
  if (!isRecord(config)) throw new Error("Configuration must be a JSON object.");
  if (typeof config.name !== "string" || !config.name.trim() || config.name.length > 100) {
    throw new Error("Enter a run name between 1 and 100 characters.");
  }
  for (const kind of kinds) {
    if (!state.catalog.components[kind].some((item) => item.id === config[kind])) {
      throw new Error(`Unknown ${kind}: ${String(config[kind])}.`);
    }
  }
  if (!Number.isInteger(config.budget) || config.budget < 1 || config.budget > 100) {
    throw new Error("Choose a budget between 1 and 100 candidate evaluations.");
  }
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295) {
    throw new Error("Choose a whole-number seed from 0 to 4294967295.");
  }
  if (has(config, "task") && !isRecord(config.task)) {
    throw new Error("Task data must be a JSON object containing train and holdout examples.");
  }
  if (has(config, "settings") && !isRecord(config.settings)) {
    throw new Error("Component settings must be a JSON object.");
  }
}

function getDraft() {
  const config = {
    ...state.base,
    name: el("run-name").value.trim(),
    budget: el("budget").valueAsNumber,
    seed: el("seed").valueAsNumber,
  };
  for (const kind of kinds) config[kind] = el(kind).value;
  for (const [key, id, label] of [
    ["task", "task-data", "Task data"],
    ["initial_candidate", "initial-candidate", "The starting candidate"],
    ["settings", "component-settings", "Component settings"],
  ]) {
    const data = readJSON(id, label);
    if (data === undefined) delete config[key];
    else config[key] = data;
  }
  validateConfig(config);
  return config;
}

function syncTaskCount() {
  const raw = el("task-data").value.trim();
  try {
    const task = raw ? JSON.parse(raw) : state.catalog.example_task;
    const train = Array.isArray(task.train) ? task.train.length : 0;
    const holdout = Array.isArray(task.holdout) ? task.holdout.length : 0;
    el("task-counts").textContent = `${train} train · ${holdout} holdout`;
  } catch {
    el("task-counts").textContent = "Check JSON";
  }
}

function describeNode(kind) {
  const item = state.catalog.components[kind].find((entry) => entry.id === el(kind).value);
  el("node-hint").textContent = item ? `${roles[kind]} ${item.description}` : roles[kind];
}

function safeSourceURL(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function setDraft(config, source) {
  if (isRecord(config)) config = {name: "Untitled loop", ...config};
  validateConfig(config);
  state.base = clone(config);
  state.source = source;
  el("run-name").value = config.name;
  el("budget").value = config.budget;
  el("seed").value = config.seed;
  for (const kind of kinds) el(kind).value = config[kind];
  el("task-data").value = pretty(config.task ?? state.catalog.example_task);
  el("initial-candidate").value = has(config, "initial_candidate") ? pretty(config.initial_candidate) : "";
  el("component-settings").value = has(config, "settings") ? pretty(config.settings) : "";
  el("node-hint").textContent = "Select any component to see its role.";
  syncTaskCount();

  const label = el("template-origin");
  label.textContent = source ? "Template fork" : "Imported draft";
  const url = safeSourceURL(source?.source_url);
  if (url) {
    const sourceLink = element("a", "", "Method ↗");
    sourceLink.href = url;
    sourceLink.target = "_blank";
    sourceLink.rel = "noopener noreferrer";
    label.append(" · ", sourceLink);
  }
  for (const button of el("template-list").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.template === source?.id));
  }
}

function populateCatalog() {
  const {components, templates} = state.catalog;
  for (const kind of kinds) {
    for (const item of components[kind]) {
      const option = element("option", "", item.name);
      option.value = item.id;
      el(kind).append(option);
    }
    el(kind).addEventListener("focus", () => describeNode(kind));
    el(kind).addEventListener("change", () => describeNode(kind));
  }
  el("template-count").textContent = `${String(templates.length).padStart(2, "0")} offline starters`;
  for (const [index, template] of templates.entries()) {
    const card = element("button", "template-card");
    card.type = "button";
    card.dataset.template = template.id;
    card.setAttribute("aria-pressed", "false");
    card.append(
      element("span", "template-kicker", `Template ${String(index + 1).padStart(2, "0")}`),
      element("span", "template-title", template.name),
      element("span", "template-description", template.description),
      element("span", "template-cta", "Fork template ↗"),
    );
    card.addEventListener("click", () => {
      if (state.busy) return;
      setDraft(template.config, template);
      report(`Forked ${template.name}. This draft is yours to edit.`);
    });
    el("template-list").append(card);
  }
  if (!templates.length) throw new Error("No templates are available.");
  setDraft(templates[0].config, templates[0]);
  el("experiment-controls").disabled = false;
}

function setPin(control) {
  state.control = control;
  el("pin-summary").hidden = !control;
  el("compare-button").hidden = !control;
  el("pin-button").textContent = control ? "Replace control" : "Pin as control";
  el("pin-name").textContent = control
    ? `Control: ${control.name} · ${control.budget} evaluations · seed ${control.seed}`
    : "";
}

async function fetchJSON(url, config) {
  const response = await fetch(url, config === undefined ? undefined : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(config),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`The workbench returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) throw new Error(data.error || `The request failed (${response.status}).`);
  return data;
}

function setBusy(busy, comparison) {
  state.busy = busy;
  el("experiment-controls").disabled = busy;
  el("result-panel").setAttribute("aria-busy", String(busy));
  for (const button of el("template-list").querySelectorAll("button")) button.disabled = busy;
  el("compare-button").textContent = busy && comparison ? "Comparing…" : "Compare runs";
}

async function execute(comparison) {
  if (state.busy || !el("experiment").reportValidity()) return;
  let draft;
  try {
    draft = getDraft();
  } catch (error) {
    report(error.message, true);
    return;
  }
  if (comparison && !state.control) return;
  setBusy(true, comparison);
  report(comparison ? "Running both configurations…" : "Evaluating candidates…");
  try {
    const response = comparison
      ? await fetchJSON("/api/compare", {configs: [state.control, draft]})
      : await fetchJSON("/api/run", draft);
    const runs = comparison ? response.runs : [response];
    renderResults(runs);
    state.latest = response;
    el("export-results").hidden = false;
    report(comparison ? "Comparison complete. Both runs are in the results panel." : "Run complete. See the results panel.");
    if (matchMedia("(max-width: 820px)").matches) el("results").scrollIntoView({block: "start"});
  } catch (error) {
    report(error.message, true);
  } finally {
    setBusy(false, comparison);
  }
}

function percent(score) {
  return typeof score === "number" ? new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 1}).format(score) : "—";
}

function delta(score) {
  const points = Math.round(score * 1000) / 10;
  return `${points > 0 ? "+" : ""}${points} pp`;
}

function metricGrid(run) {
  const metrics = element("dl", "metric-grid");
  for (const [name, value, sub, style] of [
    ["Baseline · train", run.baseline_score, "Starting candidate", ""],
    ["Best · train", run.best_score, delta(run.best_score - run.baseline_score) + " vs baseline", "best"],
    ["Winner · holdout", run.holdout_score, run.holdout_score == null ? "No holdout cases" : "Final score on unseen cases", ""],
  ]) {
    const card = element("div", `metric ${style}`);
    card.append(element("dt", "", name), element("dd", "", percent(value)), element("small", "", sub));
    metrics.append(card);
  }
  return metrics;
}

function progressChart(runs) {
  const box = element("div", "chart-box");
  const heading = element("div", "chart-header");
  heading.append(element("strong", "", "Best train score"), element("span", "", "BY EVALUATION"));
  box.append(heading);
  const width = 505;
  const height = 128;
  const left = 42;
  const right = 490;
  const top = 13;
  const bottom = 104;
  const count = Math.max(...runs.map((run) => run.trials.length));
  const x = (index) => count === 1 ? left : left + index * (right - left) / (count - 1);
  const y = (score) => bottom - score * (bottom - top);
  const graph = svg("svg", {viewBox: `0 0 ${width} ${height}`, class: "chart", role: "img", "aria-label": runs.map((run) => `${run.config.name}: ${percent(run.baseline_score)} to ${percent(run.best_score)} in ${run.evaluations_used} evaluations.`).join(" ")});
  for (const score of [0, .5, 1]) {
    const row = y(score);
    graph.append(
      svg("line", {x1: left, x2: right, y1: row, y2: row, class: "gridline"}),
      svg("text", {x: left - 6, y: row + 3, "text-anchor": "end"}, percent(score)),
    );
  }
  const labels = new Set([0, Math.floor((count - 1) / 2), count - 1]);
  for (const index of labels) {
    graph.append(svg("text", {x: x(index), y: 121, "text-anchor": "middle"}, String(index + 1)));
  }
  // Draw the draft first so a dashed control remains visible when both traces coincide.
  for (const run of [...runs].reverse()) {
    if (!run.trials.length) continue;
    const isControl = runs.length > 1 && run === runs[0];
    const trace = run.trials.map((trial, index) => index === 0
      ? `M ${x(0)} ${y(trial.best_score)}`
      : `H ${x(index)} V ${y(trial.best_score)}`).join(" ");
    graph.append(
      svg("path", {d: trace, class: `trace${isControl ? " control" : ""}`}),
      svg("circle", {cx: x(run.trials.length - 1), cy: y(run.best_score), r: 3, class: `dot${isControl ? " control" : ""}`}),
    );
  }
  box.append(graph);
  if (runs.length > 1) {
    const legend = element("div", "legend");
    legend.append(element("span", "control", "Control"), element("span", "", "Draft"));
    box.append(legend);
  }
  return box;
}

function comparisonTable(runs) {
  const table = element("table", "comparison-table");
  const head = element("thead");
  const body = element("tbody");
  const labels = element("tr");
  for (const label of ["Run", "Baseline", "Best train", "Holdout"]) {
    const cell = element("th", "", label);
    cell.scope = "col";
    labels.append(cell);
  }
  head.append(labels);
  for (const [index, run] of runs.entries()) {
    const row = element("tr");
    const label = element("th", "", `${index ? "Draft" : "Control"} · ${run.config.name}`);
    label.scope = "row";
    row.append(label, ...[run.baseline_score, run.best_score, run.holdout_score].map((value) => element("td", "", percent(value))));
    body.append(row);
  }
  table.append(element("caption", "", "Scores with a shared evaluation budget"), head, body);
  return table;
}

function candidateDrawer(run, prefix) {
  const details = element("details", "drawer result-drawer");
  const summary = element("summary");
  summary.append(element("span", "", `${prefix}Winning candidate`));
  if (run.best_trial_id != null) summary.append(element("span", "", String(run.best_trial_id)));
  details.append(summary, element("pre", "", pretty(run.best_candidate)));
  return details;
}

function lineageDrawer(run, prefix) {
  const details = element("details", "drawer result-drawer");
  const summary = element("summary");
  summary.append(element("span", "", `${prefix}Candidate lineage`), element("span", "", `${run.trials.length} trials`));
  const list = element("div", "lineage-list");
  for (const trial of run.trials) {
    const branch = element("details", "trial");
    const summary = element("summary");
    const winner = trial.id === run.best_trial_id;
    summary.append(
      element("span", `trial-id${winner ? " winner" : ""}`, String(trial.id)),
      element("span", "", trial.parent_id == null ? "Starting point" : `← ${trial.parent_id}`),
      element("span", "", percent(trial.score)),
    );
    branch.append(summary);
    if (trial.note) branch.append(element("p", "", trial.note));
    branch.append(element("pre", "", pretty(trial.candidate)));
    list.append(branch);
  }
  details.append(summary, list);
  return details;
}

function renderResults(runs) {
  const comparison = runs.length > 1;
  const run = runs.at(-1);
  const heading = element("div", "result-heading");
  heading.append(
    element("h3", "", comparison ? "Run comparison" : run.config.name),
    element("span", "", `${run.evaluations_used} evaluations${comparison ? " each" : ""}`),
  );
  const content = [
    heading,
    comparison ? comparisonTable(runs) : metricGrid(run),
    progressChart(runs),
  ];
  if (run.train_cases != null && run.holdout_cases != null) {
    content.push(element("p", "run-data", `${run.train_cases} train cases · ${run.holdout_cases} holdout cases · seed ${run.config.seed}`));
  }
  for (const [index, entry] of runs.entries()) {
    const prefix = comparison ? (index === 0 ? "Control · " : "Draft · ") : "";
    content.push(candidateDrawer(entry, prefix), lineageDrawer(entry, prefix));
  }
  el("results-content").replaceChildren(...content);
}

function download(name, value) {
  const body = new Blob([pretty(value), "\n"], {type: "application/json"});
  const url = URL.createObjectURL(body);
  const link = element("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fileImport(inputId, handler) {
  el(inputId).addEventListener("change", async (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw new Error("Choose a JSON file no larger than 1 MiB.");
      let value;
      try {
        value = JSON.parse(await file.text());
      } catch {
        throw new Error("The selected file is not valid JSON.");
      }
      if (!state.busy) handler(value);
    } catch (error) {
      report(error.message, true);
    }
  });
}

el("experiment").addEventListener("submit", (event) => {
  event.preventDefault();
  execute(false);
});
el("compare-button").addEventListener("click", () => execute(true));
el("pin-button").addEventListener("click", () => {
  if (!el("experiment").reportValidity()) return;
  try {
    setPin(clone(getDraft()));
    report("Control pinned. Change the proposer or search, then compare the two configurations.");
  } catch (error) {
    report(error.message, true);
  }
});
el("clear-pin").addEventListener("click", () => {
  setPin(null);
  report("Control cleared.");
});
el("task-data").addEventListener("input", syncTaskCount);
el("import-task").addEventListener("click", () => el("task-file").click());
el("reset-task").addEventListener("click", () => {
  el("task-data").value = pretty(state.catalog.example_task);
  syncTaskCount();
  report("Example task restored.");
});
el("import-config").addEventListener("click", () => el("config-file").click());
fileImport("task-file", (value) => {
  if (!isRecord(value) || !Array.isArray(value.train)) {
    throw new Error("Import a task object with a train array and an optional holdout array.");
  }
  el("task-data").value = pretty(value);
  el("task-drawer").open = true;
  syncTaskCount();
  report("Task imported.");
});
fileImport("config-file", (value) => {
  setDraft(value, null);
  report("Configuration imported.");
});
el("export-config").addEventListener("click", () => {
  if (!el("experiment").reportValidity()) return;
  try {
    download("rsi-configuration.json", getDraft());
  } catch (error) {
    report(error.message, true);
  }
});
el("export-results").addEventListener("click", () => {
  if (state.latest) download("rsi-results.json", state.latest);
});

fetchJSON("/api/catalog").then((data) => {
  state.catalog = data;
  populateCatalog();
}).catch((error) => {
  el("template-count").textContent = "Unavailable";
  el("template-list").replaceChildren(element("p", "template-error", `Could not load the template library. ${error.message}`));
});
