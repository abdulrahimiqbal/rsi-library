"use strict";

const el = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const KINDS = ["target", "proposer", "harness", "judge", "search"];
const KIND_LABELS = {target: "Changes", proposer: "Proposes", harness: "Tests on", judge: "Scores by", search: "Builds on"};
const KIND_STEPS = {
  target: "the thing being improved",
  proposer: "writes the next version",
  harness: "runs it on the cases",
  judge: "marks each answer",
  search: "picks what to build on",
};
const VERDICTS = {improved: "Significant gain", inconclusive: "Inconclusive", not_improved: "No gain"};
const STAGES = {
  baseline_train: ["harness", "Scoring the original on the learning examples"],
  baseline_train_judge: ["judge", "Grading the original's answers"],
  propose: ["proposer", "Writing a revision"],
  candidate_train: ["harness", "Scoring a revision on the learning examples"],
  candidate_train_judge: ["judge", "Grading a revision's answers"],
  baseline_holdout: ["judge", "Final check: the original on the held-back examples"],
  baseline_holdout_judge: ["judge", "Final check: grading the original"],
  candidate_holdout: ["judge", "Final check: the best revision on the held-back examples"],
  candidate_holdout_judge: ["judge", "Final check: grading the best revision"],
};
const OPENROUTER = "https://openrouter.ai/api/v1/chat/completions";
const DIVIDER = "--- held back for the final check ---";
const RUN_PATH = /^\/runs\/(run_[A-Za-z0-9_-]{16,40})$/;
const RUN_SECONDS = 3;
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 0});
const calm = window.matchMedia("(prefers-reduced-motion: reduce)");

const state = {
  catalog: null, mode: "loop", loop: null, method: null, swaps: {}, node: "search", token: 0,
  cases: {offline: "", driven: "", code: ""}, target: "prompt", prompt: "",
  program: {language: "javascript", entry: "", source: ""},
  objective: "normalized", scoring: {tolerance: "", fields: "", rubric: ""},
  who: "assistant", key: "", model: "openai/gpt-4.1-mini",
  view: null, shown: 0, selected: 0, series: 0, timer: null, running: false, stop: false, run: null,
};

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function svg(tag, attrs = {}, text) {
  const value = document.createElementNS(SVG, tag);
  for (const [key, entry] of Object.entries(attrs)) value.setAttribute(key, entry);
  if (text !== undefined) value.textContent = text;
  return value;
}

function notice(id, message = "", error = false) {
  const target = el(id);
  target.textContent = message;
  target.classList.toggle("error", error);
}

const share = (value) => (Number.isFinite(value) ? percent.format(value) : "–");
const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
const pValue = (value) => (value < 0.001 ? "under 0.001" : (Math.floor(value * 1000 + 0.5) / 1000).toFixed(3));
const points = (value) => `${value < 0 ? "−" : "+"}${Math.abs(value * 100).toFixed(1)} pts`;
const listed = (items) => (items.length < 3 ? items.join(" and ") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);
const loopById = (id) => state.catalog.loops.find((loop) => loop.id === id);
const loopName = (id) => loopById(id)?.name || id;
const objectiveById = (id) => state.catalog.objectives.find((entry) => entry.id === id) || state.catalog.objectives[1];

async function call(path, body, method) {
  let response;
  const options = {cache: "no-store"};
  if (method) options.method = method;
  if (body !== undefined) Object.assign(options, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Could not reach RSI Library. Check your connection and try again.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("RSI Library returned an unreadable reply.");
  }
  if (!response.ok) {
    const error = new Error(typeof data?.error === "string" ? data.error : "The request failed.");
    error.status = response.status;
    throw error;
  }
  return data;
}

async function copy(text, noteId, done) {
  try {
    await navigator.clipboard.writeText(text);
    notice(noteId, done);
  } catch {
    notice(noteId, "Copying is blocked here. Select the text and copy it.", true);
  }
}

function artifactText(artifact) {
  if (artifact == null) return "";
  if (typeof artifact === "string") return artifact;
  if (artifact.type === "program") return artifact.source;
  if (artifact.type === "prompt" && !artifact.improver) return artifact.text;
  if (artifact.type === "prompt") return `${artifact.text}\n\nEditing instructions passed to later versions:\n${artifact.improver}`;
  if (artifact.type === "prompt-chain") return artifact.steps.map((step) => `${step.name}:\n${step.text}`).join("\n\n");
  if (typeof artifact.prompt === "string") {
    const policy = artifact.policy && typeof artifact.policy === "object"
      ? `\n\nEdit policy carried by this version: ${Object.entries(artifact.policy).map(([key, value]) => `${key} ${value}`).join(", ")}`
      : "";
    return artifact.prompt + policy;
  }
  return JSON.stringify(artifact, null, 2);
}

const artifactNoun = (artifact) => (artifact?.type === "program" ? "function" : artifact?.type === "prompt-chain" ? "prompt chain" : "prompt");

// Examples are typed one per line: "input → expected". A divider line separates the held-back cases.

function casesToText(task) {
  const lines = (rows) => rows.map((row) => `${row.input} → ${row.expected}`);
  return [...lines(task.train), DIVIDER, ...lines(task.holdout || [])].join("\n");
}

function parseCases(text) {
  const groups = [[]];
  for (const [index, raw] of text.split("\n").entries()) {
    const line = raw.trim();
    if (!line) continue;
    if (/^-{3,}/.test(line)) {
      groups.push([]);
      continue;
    }
    const match = /^(.*\S)\s*(?:→|=>|->|\t)\s*(\S.*)$/.exec(line);
    if (!match) throw new Error(`Line ${index + 1} needs an arrow between the input and the expected answer, like “Where is my invoice? → billing”.`);
    groups[groups.length - 1].push({input: match[1], expected: match[2]});
  }
  if (groups.length > 2) throw new Error("Use one divider line: learning examples above it, held-back examples below.");
  if (groups.length === 2) return {train: groups[0], holdout: groups[1], split: "yours"};
  // No divider: hold back every second example of each answer, so both halves see every answer.
  const train = [], holdout = [], seen = {};
  for (const row of groups[0]) {
    seen[row.expected] = (seen[row.expected] || 0) + 1;
    (seen[row.expected] % 2 ? train : holdout).push(row);
  }
  return {train, holdout, split: "automatic"};
}

function casesSummary(mode) {
  try {
    const {train, holdout, split} = parseCases(state.cases[mode]);
    const minimum = mode === "offline" ? 0 : state.catalog.limits.driven.min_holdout_cases;
    const short = holdout.length < minimum ? ` Add ${minimum - holdout.length} more held-back examples so the final check can show a real gain.` : "";
    const how = split === "automatic" ? " Split automatically; add a line of dashes to split them yourself." : "";
    return {ok: !short && train.length > 0, text: `${plural(train.length, "example")} to learn from, ${holdout.length} held back for the final check.${how}${short}`};
  } catch (error) {
    return {ok: false, text: error.message};
  }
}

const casesKey = () => (state.loop.mode === "offline" ? "offline" : state.target === "program" ? "code" : "driven");

// Templates, the measured results, and the gallery

function templateButton(id, name, note, choose) {
  const item = node("li");
  const button = node("button", "template");
  button.type = "button";
  button.dataset.template = id;
  button.append(node("span", "template-name", name), node("span", "template-note", note));
  button.addEventListener("click", choose);
  item.append(button);
  return item;
}

function renderTemplates() {
  for (const mode of ["offline", "driven"]) {
    el(`templates-${mode}`).replaceChildren(...state.catalog.loops.filter((loop) => loop.mode === mode).map((loop) => (
      templateButton(loop.id, loop.name, loop.parts.search.name, () => selectLoop(loop.id, true))
    )));
  }
  el("templates-code").replaceChildren(
    ...state.catalog.loops.filter((loop) => loop.artifact_types?.includes("program")).map((loop) => (
      templateButton(`code:${loop.id}`, loop.id === "reflect-v1" ? "Fix a function" : "Evolve a function", loop.name, () => selectCode(loop.id))
    )),
  );
}

function currentTemplate() {
  if (state.mode !== "loop" || !state.loop) return null;
  return state.loop.mode === "driven" && state.target === "program" ? `code:${state.loop.id}` : state.loop.id;
}

function markTemplates() {
  const current = currentTemplate();
  for (const button of document.querySelectorAll(".template")) button.setAttribute("aria-current", String(button.dataset.template === current));
}

function drivenMeasured(id) {
  const driven = state.catalog.benchmarks.driven;
  return driven.status === "measured" ? driven.loops?.[id] : null;
}

function measuredLine(entry) {
  const bench = state.catalog.benchmarks;
  if (entry.mode === "driven") {
    const row = drivenMeasured(entry.id);
    return row
      ? `Measured on ${bench.driven.model}: ${points(row.holdout_gain)} on held-back examples, significant in ${row.improved} of ${row.runs} runs, about ${Math.round(row.model_calls)} calls a run.`
      : "Not measured on a live model yet: its results depend on the model that answers.";
  }
  const cell = bench?.loops?.[entry.id]?.["24"];
  return cell ? `Measured: ${points(cell.holdout_gain)} on held-back examples at 24 versions, better than the start in ${percent.format(cell.better_than_start)} of runs.` : "";
}

function renderMeasured() {
  const bench = state.catalog.benchmarks;
  const loops = state.catalog.loops.filter((loop) => loop.mode === "offline" && bench.loops[loop.id]);
  el("measured-lede").textContent = (
    `Every loop that runs here was run on the same ${plural(bench.tasks.length, "labelled task")} with ${bench.seeds} seeds at each budget. ` +
    "Each cell is how far the winning prompt beat the starting prompt on held-back examples, on average, in percentage points."
  );
  const best = Object.fromEntries(bench.budgets.map((budget) => [budget, Math.max(...loops.map((loop) => bench.loops[loop.id][budget].holdout_gain))]));
  const head = node("tr");
  head.append(node("th", "", "Loop"), ...bench.budgets.map((budget) => node("th", "num", `${budget} versions`)), node("th", "num", "Per version, at 24"));
  const rows = loops.map((loop) => {
    const row = node("tr");
    const name = node("th");
    const open = node("button", "link", loop.name);
    open.type = "button";
    open.addEventListener("click", () => openOnCanvas(loop.id));
    name.append(open);
    row.append(name);
    for (const budget of bench.budgets) {
      const cell = bench.loops[loop.id][budget];
      const td = node("td", cell.holdout_gain === best[budget] ? "num lead" : "num");
      td.append(node("strong", "", points(cell.holdout_gain)), node("span", "sub", `better in ${percent.format(cell.better_than_start)} of runs`));
      row.append(td);
    }
    row.append(node("td", "num", points(bench.loops[loop.id]["24"].gain_per_version)));
    return row;
  });
  const thead = node("thead");
  thead.append(head);
  const tbody = node("tbody");
  tbody.append(...rows);
  el("bench-table").replaceChildren(thead, tbody);

  const taskHead = node("tr");
  taskHead.append(node("th", "", "Task"), node("th", "num", "Start"), ...loops.map((loop) => node("th", "num", `${loop.name}, 24`)));
  const taskRows = bench.tasks.map((task) => {
    const row = node("tr");
    const name = node("th");
    name.append(node("span", "", task.name), node("span", "sub", `${task.description} ${task.train_cases} learning, ${task.holdout_cases} held back.`));
    row.append(name, node("td", "num", share(task.baseline_holdout)));
    for (const loop of loops) row.append(node("td", "num", points(bench.loops[loop.id]["24"].per_task[task.id])));
    return row;
  });
  const taskThead = node("thead");
  taskThead.append(taskHead);
  const taskBody = node("tbody");
  taskBody.append(...taskRows);
  el("bench-task-table").replaceChildren(taskThead, taskBody);
  const driven = bench.driven;
  const live = driven.status === "measured"
    ? `The loops that improve your own prompt were measured on ${driven.model} (${driven.date}), starting from “${driven.start_prompt}”: ` +
      `${listed(Object.entries(driven.loops).map(([id, row]) => `${loopName(id)} ${points(row.holdout_gain)} with about ${Math.round(row.model_calls)} calls`))}. ${driven.note}`
    : `${driven.note} To measure them, run ${driven.how}.`;
  el("bench-note").textContent = `${live} The offline numbers are exact: run python -m rsi_library.benchmarks to reproduce them.`;
}

const filters = {family: null, style: null, runnable: false, improver: false};
let compared = [];

function matches(entry) {
  return (!filters.family || entry.family === filters.family)
    && (!filters.style || entry.search_style === filters.style)
    && (!filters.runnable || entry.runnable)
    && (!filters.improver || /^(Yes|Partly)/.test(entry.rewrites_improver));
}

function renderFilters() {
  const gallery = state.catalog.gallery;
  const chip = (label, pressed, toggle) => {
    const button = node("button", "chip", label);
    button.type = "button";
    button.setAttribute("aria-pressed", String(pressed));
    button.addEventListener("click", () => {
      toggle();
      renderFilters();
      renderGallery();
    });
    return button;
  };
  const group = (title, chips) => {
    const wrap = node("div", "filter-group");
    wrap.append(node("span", "filter-title", title), ...chips);
    return wrap;
  };
  el("filters").replaceChildren(
    group("What it changes", gallery.families.map((family) => chip(family, filters.family === family, () => {
      filters.family = filters.family === family ? null : family;
    }))),
    group("How it searches", gallery.search_styles.map((style) => chip(style, filters.style === style, () => {
      filters.style = filters.style === style ? null : style;
    }))),
    group("Show only", [
      chip("Runs here", filters.runnable, () => {
        filters.runnable = !filters.runnable;
      }),
      chip("Rewrites its own improver", filters.improver, () => {
        filters.improver = !filters.improver;
      }),
    ]),
  );
}

function renderGalleryMeta() {
  const gallery = state.catalog.gallery;
  const day = new Date(`${gallery.checked}T00:00:00`).toLocaleDateString(undefined, {day: "numeric", month: "long", year: "numeric"});
  const link = node("a", "", "Suggest a method");
  link.href = gallery.suggest;
  link.rel = "noopener";
  el("gallery-meta").replaceChildren(document.createTextNode(`Fact sheets checked against each paper on ${day}. Missing one? `), link, document.createTextNode("."));
}

function openOnCanvas(id) {
  const entry = state.catalog.gallery.methods.find((item) => item.id === id);
  if (entry && !entry.runnable) showMethod(id, true);
  else selectLoop(id, true);
  el("canvas").scrollIntoView({block: "center"});
}

function renderGallery() {
  const all = state.catalog.gallery.methods;
  const shown = all.filter(matches);
  el("gallery-count").textContent = shown.length === all.length
    ? `${all.length} loops: ${all.filter((entry) => entry.runnable).length} you can run here and ${all.filter((entry) => !entry.runnable).length} published methods.`
    : `${plural(shown.length, "loop")} of ${all.length} match.`;
  const cards = shown.map((entry) => {
    const card = node("article", entry.runnable ? "card runnable" : "card");
    const tags = node("p", "card-tags");
    for (const label of [entry.family, entry.search_style, entry.runnable ? "Runs here" : "", /^(Yes|Partly)/.test(entry.rewrites_improver) ? "Rewrites its improver" : ""]) {
      if (label) tags.append(node("span", label === "Runs here" ? "tag tag-improved" : "tag", label));
    }
    const facts = node("dl", "card-facts");
    for (const kind of ["target", "proposer", "search"]) facts.append(node("dt", "", KIND_LABELS[kind]), node("dd", "", entry.parts[kind]));
    const more = node("details", "card-more");
    const rest = node("dl", "card-facts");
    for (const kind of ["harness", "judge"]) rest.append(node("dt", "", KIND_LABELS[kind]), node("dd", "", entry.parts[kind]));
    rest.append(node("dt", "", "Rewrites its own improver?"), node("dd", "", entry.rewrites_improver));
    if (entry.best_for) rest.append(node("dt", "", "Good for"), node("dd", "", entry.best_for));
    more.append(node("summary", "", "Full fact sheet"), rest);
    const actions = node("div", "card-actions");
    const open = node("button", entry.runnable ? "primary" : "", "Open on the canvas");
    open.type = "button";
    open.addEventListener("click", () => openOnCanvas(entry.id));
    actions.append(open);
    if (!entry.runnable && entry.closest) {
      const closest = node("button", "", `Run the closest: ${loopName(entry.closest)}`);
      closest.type = "button";
      closest.addEventListener("click", () => openOnCanvas(entry.closest));
      actions.append(closest);
    }
    const pick = node("button", "", compared.includes(entry.id) ? "Remove from comparison" : "Compare");
    pick.type = "button";
    pick.setAttribute("aria-pressed", String(compared.includes(entry.id)));
    pick.addEventListener("click", () => {
      compared = compared.includes(entry.id) ? compared.filter((id) => id !== entry.id) : [...compared.slice(-1), entry.id];
      renderCompare();
      renderGallery();
    });
    actions.append(pick);
    if (entry.source) {
      const link = node("a", "", entry.runnable ? "Method it is based on" : "Paper");
      link.href = entry.source;
      link.rel = "noopener";
      actions.append(link);
    }
    card.append(node("h3", "", entry.name), node("p", "card-origin", `${entry.year}, ${entry.origin}`), tags, facts);
    if (entry.runnable) card.append(node("p", "card-measured", measuredLine(entry)));
    card.append(more, actions);
    return card;
  });
  el("gallery-grid").replaceChildren(...cards);
}

function renderCompare() {
  const picked = compared.map((id) => state.catalog.gallery.methods.find((entry) => entry.id === id)).filter(Boolean);
  el("compare-panel").hidden = picked.length === 0;
  if (!picked.length) return;
  el("compare-title").textContent = picked.length === 2 ? `${picked[0].name} and ${picked[1].name}` : `${picked[0].name}: pick one more to compare`;
  const rows = [
    ["From", (entry) => `${entry.year}, ${entry.origin}`],
    ["What it changes", (entry) => entry.family],
    ["How it searches", (entry) => entry.search_style],
    ...KINDS.map((kind) => [KIND_LABELS[kind], (entry) => entry.parts[kind]]),
    ["Rewrites its own improver?", (entry) => entry.rewrites_improver],
    ["Runs here", (entry) => (entry.runnable ? "Yes" : entry.closest ? `No. Closest: ${loopName(entry.closest)}` : "No")],
  ];
  const head = node("tr");
  head.append(node("th", "", ""), ...picked.map((entry) => node("th", "", entry.name)));
  const body = rows.map(([label, read]) => {
    const row = node("tr");
    const values = picked.map(read);
    if (picked.length === 2 && values[0] !== values[1]) row.className = "differs";
    row.append(node("th", "", label), ...values.map((value) => node("td", "", value)));
    return row;
  });
  const thead = node("thead");
  thead.append(head);
  const tbody = node("tbody");
  tbody.append(...body);
  el("compare-table").replaceChildren(thead, tbody);
  if (picked.length === 2) el("compare-panel").scrollIntoView({block: "nearest"});
}

// Canvas: the ring, its five nodes, and the hub

const RING = {cx: 400, cy: 250, rx: 292, ry: 186};
const ringPoint = (degrees) => {
  const angle = (degrees * Math.PI) / 180;
  return [RING.cx + RING.rx * Math.cos(angle), RING.cy + RING.ry * Math.sin(angle)];
};

function renderRing() {
  const parts = [svg("ellipse", {class: "ring-track", cx: RING.cx, cy: RING.cy, rx: RING.rx, ry: RING.ry})];
  parts.push(svg("ellipse", {class: "ring-flow", cx: RING.cx, cy: RING.cy, rx: RING.rx, ry: RING.ry, pathLength: 100}));
  for (const degrees of [-54, 18, 90, 162, 234]) {
    const [x, y] = ringPoint(degrees);
    const [nx, ny] = ringPoint(degrees + 1);
    const turn = (Math.atan2(ny - y, nx - x) * 180) / Math.PI;
    parts.push(svg("path", {class: "ring-arrow", d: "M-7,-6 L3,0 L-7,6", transform: `translate(${x.toFixed(1)},${y.toFixed(1)}) rotate(${turn.toFixed(1)})`}));
  }
  el("ring").replaceChildren(...parts);
}

function scoringText(objective, scoring) {
  const settings = Object.entries(scoring || {});
  return settings.length ? ` Settings: ${settings.map(([key, value]) => `${key} ${Array.isArray(value) ? value.join(", ") : value}`).join("; ")}.` : "";
}

function partFor(kind) {
  if (state.mode === "method") {
    return {name: state.method.labels[kind], description: state.method.parts[kind]};
  }
  const loop = state.loop;
  const request = state.mode === "run" ? state.run?.view?.request : null;
  if (loop.mode === "offline" && state.swaps[kind]) {
    return state.catalog.components[kind].find((entry) => entry.id === state.swaps[kind]) || loop.parts[kind];
  }
  if (loop.mode === "driven" && kind === "judge") {
    const objective = objectiveById(request ? request.objective : state.objective);
    return {id: objective.id, name: objective.name, description: objective.description};
  }
  if (loop.mode === "driven" && kind === "target") {
    const noun = artifactNoun(request ? request.artifact : {type: state.target === "program" ? "program" : "prompt"});
    return {id: "artifact", name: `Your ${noun}`, description: `The ${noun} being improved.`};
  }
  if (loop.mode === "driven" && kind === "proposer" && state.mode === "run" && state.run?.view?.loops.length > 1) {
    return {...loop.parts.proposer, name: `${state.run.view.loops.length} loops`};
  }
  if (loop.mode === "driven" && kind === "proposer" && state.who === "key") {
    return {...loop.parts.proposer, description: `Your model ${state.model} writes the revisions.`};
  }
  return loop.parts[kind];
}

function renderNodes() {
  const buttons = KINDS.map((kind) => {
    const part = partFor(kind);
    const button = node("button", `node node-${kind}`);
    button.type = "button";
    button.dataset.kind = kind;
    button.setAttribute("aria-pressed", String(kind === state.node));
    if (state.mode === "loop" && state.swaps[kind]) button.classList.add("swapped");
    button.append(node("span", "node-kind", KIND_LABELS[kind]), node("strong", "node-name", part.name), node("span", "node-step", KIND_STEPS[kind]));
    button.addEventListener("click", () => {
      state.node = kind;
      renderNodes();
      renderInspector();
    });
    return button;
  });
  el("nodes").replaceChildren(...buttons);
}

function setBusy(kind) {
  for (const button of document.querySelectorAll(".node")) button.classList.toggle("busy", button.dataset.kind === kind);
}

function setHub(score, label, step) {
  el("hub-score").textContent = score;
  el("hub-label").textContent = label;
  el("hub-step").textContent = step || "";
}

// Inspector: adjust the selected part

function casesEditor(mode) {
  const wrap = node("div", "inspector-block");
  const label = node("label", "", mode === "offline" ? "Examples, one per line" : "Your examples, one per line: input → expected answer");
  const area = node("textarea");
  area.rows = 8;
  area.spellcheck = false;
  area.value = state.cases[mode];
  const summary = node("p", "hint");
  const refresh = () => {
    const result = casesSummary(mode);
    summary.textContent = result.text;
    summary.classList.toggle("warning", !result.ok);
  };
  let wait;
  area.addEventListener("input", () => {
    state.cases[mode] = area.value;
    refresh();
    if (mode === "offline") {
      clearTimeout(wait);
      wait = setTimeout(() => void runOffline(), 500);
    }
  });
  label.append(area);
  refresh();
  wrap.append(label, summary);
  if (mode === "offline") {
    const reset = node("button", "quiet", "Restore the built-in examples");
    reset.type = "button";
    reset.addEventListener("click", () => {
      state.cases.offline = casesToText(state.catalog.example_task);
      renderInspector();
      void runOffline();
    });
    wrap.append(reset);
  }
  return wrap;
}

function optionList(kind, options, current, choose) {
  const list = node("div", "options");
  list.setAttribute("role", "radiogroup");
  list.setAttribute("aria-label", `${KIND_LABELS[kind]}: choose a part`);
  for (const entry of options) {
    const button = node("button", "option");
    button.type = "button";
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(entry.id === current));
    button.append(node("strong", "", entry.name), node("span", "", entry.description));
    button.addEventListener("click", () => choose(entry.id));
    list.append(button);
  }
  return list;
}

function field(labelText, input) {
  const label = node("label", "", labelText);
  label.append(input);
  return label;
}

function targetEditor() {
  const blocks = [];
  if (state.loop.artifact_types.includes("program")) {
    blocks.push(optionList("target", [
      {id: "prompt", name: "A prompt", description: "A system prompt that a model follows for each input."},
      {id: "program", name: "A function", description: "JavaScript or Python that takes the input text and returns the answer."},
    ], state.target, (id) => {
      state.target = id;
      afterTargetChange();
    }));
  } else {
    blocks.push(node("p", "hint", `${state.loop.name} improves prompts only. Fix a function with Reflective search or Archive evolution.`));
  }
  if (state.target === "program") {
    const pair = node("div", "pair");
    const language = node("select");
    for (const [value, text] of [["javascript", "JavaScript: runs in this browser"], ["python", "Python: your assistant runs it"]]) {
      const option = node("option", "", text);
      option.value = value;
      language.append(option);
    }
    language.value = state.program.language;
    language.addEventListener("change", () => {
      state.program.language = language.value;
    });
    const entry = node("input");
    entry.type = "text";
    entry.spellcheck = false;
    entry.value = state.program.entry;
    entry.addEventListener("input", () => {
      state.program.entry = entry.value.trim();
    });
    pair.append(field("Language", language), field("Function to call", entry));
    const source = node("textarea");
    source.rows = 9;
    source.spellcheck = false;
    source.value = state.program.source;
    source.addEventListener("input", () => {
      state.program.source = source.value;
    });
    blocks.push(pair, field("The function to improve", source), node("p", "hint", (
      "It is called with each example's input as one text argument; its return value is compared with the expected answer. " +
      "RSI Library never runs it: this page runs JavaScript in a sandboxed worker, and your assistant runs Python."
    )));
  } else {
    const area = node("textarea");
    area.rows = 4;
    area.maxLength = 4000;
    area.spellcheck = false;
    area.value = state.prompt;
    area.addEventListener("input", () => {
      state.prompt = area.value;
    });
    blocks.push(field("The prompt to improve", area));
  }
  return blocks;
}

function judgeEditor() {
  const blocks = [optionList("judge", state.catalog.objectives, state.objective, (id) => {
    state.objective = id;
    renderNodes();
    renderInspector();
  })];
  if (state.objective === "number") {
    const tolerance = node("input");
    tolerance.type = "number";
    tolerance.min = "0";
    tolerance.step = "any";
    tolerance.placeholder = "0";
    tolerance.value = state.scoring.tolerance;
    tolerance.addEventListener("input", () => {
      state.scoring.tolerance = tolerance.value;
    });
    blocks.push(field("Allowed difference (optional)", tolerance));
  } else if (state.objective === "json") {
    const fields = node("input");
    fields.type = "text";
    fields.spellcheck = false;
    fields.placeholder = "every field of the expected object";
    fields.value = state.scoring.fields;
    fields.addEventListener("input", () => {
      state.scoring.fields = fields.value;
    });
    blocks.push(field("Fields to check, separated by commas (optional)", fields));
  } else if (state.objective === "judge") {
    const rubric = node("textarea");
    rubric.rows = 3;
    rubric.maxLength = 2000;
    rubric.placeholder = "Correct, complete, and consistent with the expected answer.";
    rubric.value = state.scoring.rubric;
    rubric.addEventListener("input", () => {
      state.scoring.rubric = rubric.value;
    });
    blocks.push(field("What a good answer does (optional)", rubric), node("p", "hint", "Each case costs one more model call, for the grade."));
  }
  return blocks;
}

function keyFields() {
  const pair = node("div", "pair");
  const key = node("input");
  key.type = "password";
  key.autocomplete = "off";
  key.placeholder = "sk-or-…";
  key.value = state.key;
  key.id = "or-key";
  key.addEventListener("input", () => {
    state.key = key.value.trim();
  });
  const model = node("input");
  model.type = "text";
  model.spellcheck = false;
  model.value = state.model;
  model.addEventListener("input", () => {
    state.model = model.value.trim();
  });
  model.addEventListener("change", renderNodes);
  pair.append(field("OpenRouter API key", key), field("Model", model));
  return [pair, node("p", "hint", "The key stays in this tab and is never sent to RSI Library. Model calls are billed to your OpenRouter account.")];
}

function proposerEditor() {
  const blocks = [optionList("proposer", [
    {id: "assistant", name: "My AI assistant", description: "Copy a message that hands the run to your assistant. It answers the model calls itself."},
    {id: "key", name: "My OpenRouter key, here", description: "This page calls your model directly and runs JavaScript functions in a sandbox."},
  ], state.who, (id) => {
    state.who = id;
    renderNodes();
    renderInspector();
    syncToolbar();
  })];
  if (state.who === "key") blocks.push(...keyFields());
  return blocks;
}

function runInspector(kind) {
  const view = state.run?.view;
  const request = view?.request;
  if (!request) return [node("p", "inspector-text", "This run's settings are not available.")];
  if (kind === "target") {
    const text = node("pre", "artifact");
    text.textContent = artifactText(request.artifact);
    text.tabIndex = 0;
    return [text];
  }
  if (kind === "harness") {
    return [node("p", "inspector-text", (
      `${plural(request.train_cases, "example")} to learn from and ${request.holdout_cases} held back for the final check` +
      `${request.holdout_repeats > 1 ? `, each tried ${request.holdout_repeats} times` : ""}. Up to ${plural(request.max_trials, "revision")} per loop. ` +
      "The cases stay on the server and are not shown here."
    ))];
  }
  if (kind === "judge") {
    const objective = objectiveById(request.objective);
    return [node("p", "inspector-text", `${objective.description}${scoringText(request.objective, request.scoring)}`)];
  }
  if (kind === "proposer") {
    const names = listed(view.loops.map(loopName));
    const code = request.artifact.type === "program" ? " and runs its functions in a sandbox in this tab" : "";
    return [
      node("p", "inspector-text", `${names} ${view.loops.length > 1 ? "write" : "writes"} the revisions. To answer this run's remaining work from this page, enter your OpenRouter key and continue.`),
      ...keyFields(),
      node("p", "hint", `Continuing sends this run's prompts to your model${code}. Only continue runs you started or trust.`),
    ];
  }
  return [node("p", "inspector-text", partFor(kind).description)];
}

function renderInspector() {
  const kind = state.node;
  const heading = node("h2", "", `${KIND_LABELS[kind]}: ${partFor(kind).name}`);
  const blocks = [];
  if (state.mode === "method") {
    blocks.push(node("p", "inspector-text", state.method.parts[kind]));
    if (kind === "proposer") blocks.push(node("p", "hint", `Rewrites its own improver? ${state.method.rewrites_improver}`));
    if (kind === "search") blocks.push(node("p", "hint", `Search style: ${state.method.search_style}.`));
  } else if (state.mode === "run") {
    blocks.push(...runInspector(kind));
  } else if (state.loop.mode === "offline") {
    const loop = state.loop;
    blocks.push(optionList(kind, state.catalog.components[kind], state.swaps[kind] || loop.parts[kind].id, (id) => {
      if (id === loop.parts[kind].id) delete state.swaps[kind];
      else state.swaps[kind] = id;
      renderNodes();
      renderInspector();
      void runOffline();
    }));
    if (kind === "harness") blocks.push(casesEditor("offline"));
  } else if (kind === "target") {
    blocks.push(...targetEditor());
  } else if (kind === "harness") {
    blocks.push(casesEditor(casesKey()));
  } else if (kind === "judge") {
    blocks.push(...judgeEditor());
  } else if (kind === "proposer") {
    blocks.push(...proposerEditor());
  } else {
    blocks.push(node("p", "inspector-text", `${partFor(kind).description} This part is fixed for ${state.loop.name}. Load another template to change how the loop picks what to build on.`));
  }
  el("inspector").replaceChildren(heading, ...blocks);
}

// Modes: a template on the canvas, a published method, or a run

function stopWatching() {
  if (state.run) clearTimeout(state.run.poll);
}

function setMode(mode) {
  if (mode !== "run") {
    stopWatching();
    state.run = null;
  }
  state.mode = mode;
  el("studio").dataset.mode = mode;
  el("insights-title").textContent = {loop: "Live run", run: "This run", method: "Fact sheet"}[mode];
  el("toolbar").hidden = mode !== "loop";
  el("runbar").hidden = mode !== "run";
  el("methodbar").hidden = mode !== "method";
  markTemplates();
}

function syncToolbar() {
  const offline = state.loop.mode === "offline";
  el("budget-field").hidden = !offline;
  el("seed-field").hidden = !offline;
  el("compare").hidden = !offline;
  el("trials-field").hidden = offline;
  el("all-field").hidden = offline;
  el("handoff").hidden = offline;
  el("run").hidden = !offline && state.who !== "key";
  el("run").textContent = offline ? "Run" : "Run with my key";
  el("handoff").classList.toggle("primary", !offline && state.who !== "key");
  el("all-field").lastChild.textContent = state.target === "program" ? " Test both loops that fix functions" : " Test all three prompt loops";
}

function describeLoop() {
  const loop = state.loop;
  el("loop-name").textContent = loop.mode === "driven" && state.target === "program" ? `${loop.name}, on a function` : loop.name;
  el("loop-summary").textContent = loop.mode === "offline"
    ? `${loop.summary} Select any part of the loop to change it; the run updates as you go.`
    : state.target === "program"
      ? `${loop.summary} Put your function and its examples into the loop, then choose who answers. RSI Library never runs your code.`
      : `${loop.summary} Put your prompt and examples into the loop, then choose who answers.`;
}

function selectLoop(id, userAction = false) {
  const loop = loopById(id) || state.catalog.loops[0];
  state.loop = loop;
  state.swaps = {};
  state.token += 1;
  state.stop = true;
  state.node = loop.mode === "offline" ? "search" : "target";
  if (state.target === "program" && !loop.artifact_types?.includes("program")) state.target = "prompt";
  setMode("loop");
  describeLoop();
  notice("stage-message");
  setBusy(null);
  syncToolbar();
  renderNodes();
  renderInspector();
  clearView(loop.mode === "offline" ? "Running…" : `Add your ${state.target === "program" ? "function" : "prompt"} and examples, then run the loop.`);
  if (userAction) history.replaceState(null, "", `/?${new URLSearchParams(state.target === "program" && loop.mode === "driven" ? {loop: loop.id, target: "program"} : {loop: loop.id})}`);
  if (loop.mode === "offline") {
    el("budget").value = loop.defaults.budget;
    el("seed").value = loop.defaults.seed;
    void runOffline();
  }
}

function useProgram() {
  state.target = "program";
  const example = state.catalog.example_program_run;
  if (!state.program.source) state.program = {...example.artifact};
  state.objective = example.objective;
}

function afterTargetChange() {
  if (state.target === "program") useProgram();
  else if (state.objective === "number") state.objective = "normalized";
  describeLoop();
  syncToolbar();
  renderNodes();
  renderInspector();
  markTemplates();
  history.replaceState(null, "", `/?${new URLSearchParams(state.target === "program" ? {loop: state.loop.id, target: "program"} : {loop: state.loop.id})}`);
}

function selectCode(id) {
  state.target = "program";
  selectLoop(id, false);
  useProgram();
  afterTargetChange();
}

function showMethod(id, userAction = false) {
  const entry = state.catalog.gallery.methods.find((item) => item.id === id);
  if (!entry) return;
  state.method = entry;
  state.token += 1;
  state.stop = true;
  state.node = "proposer";
  setMode("method");
  el("loop-name").textContent = entry.name;
  el("loop-summary").textContent = (
    `${entry.year}, ${entry.origin}. A published method, drawn in the same five parts as the loops here. ` +
    `It doesn't run on this site${entry.closest ? `; the closest loop that does is ${loopName(entry.closest)}` : ""}.`
  );
  el("method-run").hidden = !entry.closest;
  el("method-run").textContent = entry.closest ? `Run the closest loop: ${loopName(entry.closest)}` : "";
  el("method-paper").href = entry.source;
  notice("stage-message");
  setBusy(null);
  renderNodes();
  renderInspector();
  const changes = {Prompt: "a prompt", Pipeline: "the prompts of a pipeline", "Memory and skills": "memory and skills", Code: "code", "Agent itself": "the agent itself", Weights: "the model's weights"};
  const searches = {"Single line": "a single line of versions", Population: "a population of versions", Archive: "an archive of versions"};
  clearView(`Changes ${changes[entry.family] || entry.family.toLowerCase()} and searches with ${searches[entry.search_style] || entry.search_style.toLowerCase()}. Select a part of the loop to read how it works.`);
  setHub(String(entry.year), entry.family, entry.search_style);
  const sheet = node("dl", "card-facts sheet");
  for (const kind of KINDS) sheet.append(node("dt", "", KIND_LABELS[kind]), node("dd", "", entry.parts[kind]));
  sheet.append(node("dt", "", "Rewrites its own improver?"), node("dd", "", entry.rewrites_improver));
  el("verdicts").replaceChildren(sheet);
  el("verdicts").hidden = false;
  if (userAction) history.replaceState(null, "", `/?method=${encodeURIComponent(entry.id)}`);
}

// Running offline loops

function offlineRequest() {
  const body = {loop: state.loop.id, budget: el("budget").valueAsNumber, seed: el("seed").valueAsNumber, ...state.swaps};
  if (!Number.isInteger(body.budget) || !Number.isInteger(body.seed)) throw new Error("Enter whole numbers for versions and seed.");
  if (state.cases.offline.trim() !== casesToText(state.catalog.example_task)) {
    const {train, holdout} = parseCases(state.cases.offline);
    if (!train.length) throw new Error("Add at least one example to learn from.");
    body.task = {train, holdout};
  }
  return body;
}

function apiSnippet(path, body) {
  return `curl -s ${location.origin}${path} \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(body).replaceAll("'", "'\\''")}'`;
}

const offlineTrials = (run) => run.trials.map((trial) => ({
  id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score, text: artifactText(trial.candidate), note: trial.note,
}));

async function runOffline(event) {
  event?.preventDefault();
  if (state.mode !== "loop") return;
  if (state.loop.mode !== "offline") return runDriven();
  const token = ++state.token;
  let body;
  try {
    body = offlineRequest();
  } catch (error) {
    notice("stage-message", error.message, true);
    return;
  }
  notice("stage-message");
  try {
    const result = await call("/api/run", body);
    if (token !== state.token) return;
    const trials = offlineTrials(result);
    const holdout = result.holdout_score == null ? "" : ` On ${plural(result.holdout_cases, "held-back example")}, the winner scores ${share(result.holdout_score)}.`;
    showView({
      headline: result.best_score > result.baseline_score
        ? `Training score rose from ${share(result.baseline_score)} to ${share(result.best_score)}.${holdout}`
        : `No version beat the starting score of ${share(result.baseline_score)}.${holdout}`,
      series: [{name: state.loop.name, trials, best: result.best_trial_id}],
      holdout: result.holdout_score,
      snippet: apiSnippet("/api/run", body),
    }, true);
  } catch (error) {
    if (token === state.token) notice("stage-message", error.message, true);
  }
}

async function compareOffline() {
  const token = ++state.token;
  let body;
  try {
    const {budget, seed, task} = offlineRequest();
    body = {loops: "all", budget, seed, ...(task ? {task} : {})};
  } catch (error) {
    notice("stage-message", error.message, true);
    return;
  }
  notice("stage-message");
  try {
    const result = await call("/api/compare", body);
    if (token !== state.token) return;
    const series = result.runs.map((run) => ({name: run.config.name, best: run.best_trial_id, trials: offlineTrials(run), train: run.best_score, holdout: run.holdout_score}));
    const best = result.summary.best;
    const basis = result.summary.ranked_by === "holdout_score" ? "the held-back examples" : "the learning examples";
    showView({
      headline: Array.isArray(best) ? `${best.join(" and ")} tie on ${basis} at this budget and seed.` : `${best} scores highest on ${basis} at this budget and seed.`,
      series,
      table: series.map((item) => ({name: item.name, cells: [`Training ${share(item.train)}`, `Held back ${share(item.holdout)}`]})),
      snippet: apiSnippet("/api/compare", body),
    }, false);
  } catch (error) {
    if (token === state.token) notice("stage-message", error.message, true);
  }
}

// Live run panel

function clearView(message) {
  clearInterval(state.timer);
  state.view = null;
  el("canvas").classList.remove("playing");
  el("headline").textContent = message;
  el("stats").replaceChildren();
  el("verdicts").hidden = true;
  el("plot").replaceChildren();
  el("legend").replaceChildren();
  el("feed").replaceChildren();
  el("version-title").hidden = true;
  el("version-meta").textContent = "";
  el("version-text").hidden = true;
  el("script-fold").hidden = true;
  setHub("–", "best training score", "");
}

function showView(view, animate, keepSeries = false) {
  clearInterval(state.timer);
  state.view = view;
  if (!keepSeries || state.series >= view.series.length) state.series = 0;
  const total = view.series[state.series]?.trials.length || 0;
  el("script-fold").hidden = !view.snippet;
  el("api-call").textContent = view.snippet || "";
  renderVerdicts();
  if (!animate || calm.matches || total < 3 || view.series.length > 1) {
    finishPlayback(keepSeries);
    return;
  }
  state.shown = 1;
  state.selected = 0;
  el("headline").textContent = "Running…";
  el("canvas").classList.add("playing");
  renderLive();
  state.timer = setInterval(() => {
    state.shown += 1;
    if (state.shown >= total) finishPlayback();
    else renderLive();
  }, Math.max(45, Math.min(130, 2600 / total)));
}

function finishPlayback(keepSelection = false) {
  clearInterval(state.timer);
  const view = state.view;
  const series = view.series[state.series];
  state.shown = series ? series.trials.length : 0;
  const ids = series ? series.trials.map((trial) => trial.id) : [];
  if (!keepSelection || !ids.includes(state.selected)) state.selected = series?.best ?? ids[ids.length - 1] ?? 0;
  if (!view.live) el("canvas").classList.remove("playing");
  el("headline").textContent = view.headline;
  renderLive();
}

function progress(series, shown) {
  let best = null, improvements = 0, lastGain = 0;
  const seen = series.trials.slice(0, shown);
  for (const trial of seen) {
    if (!Number.isFinite(trial.score)) continue;
    if (best === null) best = trial.score;
    else if (trial.score > best) {
      best = trial.score;
      improvements += 1;
      lastGain = trial.id;
    }
  }
  return {seen, best, improvements, lastGain, start: seen[0]?.score};
}

function renderLive() {
  const view = state.view;
  const series = view.series[state.series];
  if (!series) {
    renderPlot();
    return;
  }
  const total = series.trials.length;
  const done = state.shown >= total;
  const now = progress(series, state.shown);
  const latest = now.seen[now.seen.length - 1];
  if (view.live) {
    setHub(now.best === null ? "–" : share(now.best), "best training score so far", series.doing || "");
  } else if (view.series.length > 1 && !view.driven) {
    setHub(String(view.series.length), "loops compared", "same examples, budget and seed");
  } else if (total === 0) {
    setHub("–", "no versions scored", "");
  } else {
    setHub(share(now.best), "best training score", done
      ? (view.holdout != null ? `${share(view.holdout)} on held-back examples` : `${plural(total - 1, "version")} tried`)
      : `version ${latest.id} of ${total - 1}`);
  }
  const stat = (term, value) => [node("dt", "", term), node("dd", "", value)];
  const held = view.live ? "after the run" : done && view.holdout != null ? share(view.holdout) : done ? "see below" : "after the run";
  const stats = view.series.length > 1 && !view.driven ? [] : [
    ...stat("Started at", share(now.start)),
    ...stat("Best so far", share(now.best)),
    ...stat("Versions tried", String(Math.max(0, now.seen.length - 1))),
    ...stat("New bests", String(now.improvements)),
    ...stat("Last gain at", now.improvements ? `version ${now.lastGain}` : "none yet"),
    ...stat("Held back", held),
  ];
  el("stats").replaceChildren(...stats);
  renderPlot();
  renderFeed(now.seen);
  renderVersion(series);
}

function renderVerdicts() {
  const view = state.view;
  const rows = [];
  for (const [index, row] of (view.table || []).entries()) {
    const button = node("button", "verdict");
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === state.series));
    button.append(node("strong", "", row.name));
    if (row.tag) button.append(node("span", `tag tag-${row.tone || "plain"}`, row.tag));
    for (const cell of row.cells) button.append(node("span", "verdict-cell", cell));
    button.addEventListener("click", () => {
      state.series = index;
      state.shown = view.series[index].trials.length;
      state.selected = view.series[index].best ?? 0;
      renderVerdicts();
      renderLive();
    });
    rows.push(button);
  }
  if (view.adopt) {
    const box = node("div", view.adopt.confirmed ? "adopt confirmed" : "adopt");
    box.append(node("h3", "", view.adopt.title), node("pre", "", view.adopt.text));
    const button = node("button", "", `Copy ${view.adopt.noun}`);
    button.type = "button";
    const status = node("span", "hint");
    status.id = "adopt-note";
    status.setAttribute("role", "status");
    button.addEventListener("click", () => copy(view.adopt.text, "adopt-note", "Copied."));
    const actions = node("div", "control-actions");
    actions.append(button, status);
    box.append(actions);
    rows.push(box);
  }
  el("verdicts").replaceChildren(...rows);
  el("verdicts").hidden = rows.length === 0;
}

function renderPlot() {
  const view = state.view;
  const width = 420, height = 220, left = 38, right = 12, top = 12, bottom = 26;
  const count = Math.max(2, ...view.series.map((item) => item.planned ?? item.trials.length));
  const x = (id) => left + (id / (count - 1)) * (width - left - right);
  const y = (value) => top + (1 - value) * (height - top - bottom);
  const parts = [];
  for (const tick of [0, 0.5, 1]) {
    parts.push(svg("line", {class: "grid", x1: left, x2: width - right, y1: y(tick), y2: y(tick)}));
    parts.push(svg("text", {class: "axis", x: left - 6, y: y(tick) + 4, "text-anchor": "end"}, percent.format(tick)));
  }
  parts.push(svg("text", {class: "axis", x: left, y: height - 8}, "start"));
  parts.push(svg("text", {class: "axis", x: width - right, y: height - 8, "text-anchor": "end"}, `version ${count - 1}`));
  const many = view.series.length > 1;
  view.series.forEach((series, index) => {
    const active = index === state.series;
    const visible = active ? series.trials.slice(0, state.shown) : series.trials;
    const scored = visible.filter((trial) => Number.isFinite(trial.score));
    const byId = new Map(scored.map((trial) => [trial.id, trial]));
    const group = svg("g", {class: `series series-${index}${active ? " active" : ""}`});
    let best = -1;
    const path = scored.map((trial, position) => {
      best = Math.max(best, trial.score);
      return `${position ? "L" : "M"}${x(trial.id).toFixed(1)},${y(best).toFixed(1)}`;
    });
    if (scored.length) {
      path.push(`L${x(visible[visible.length - 1].id).toFixed(1)},${y(best).toFixed(1)}`);
      group.append(svg("path", {class: "best-line", d: path.join(" ")}));
    }
    if (active) {
      for (const trial of scored) {
        for (const parentId of trial.parents) {
          const parent = byId.get(parentId);
          if (parent) group.append(svg("line", {class: "edge", x1: x(parent.id), y1: y(parent.score), x2: x(trial.id), y2: y(trial.score)}));
        }
      }
      const finished = !view.live && state.shown >= series.trials.length;
      for (const trial of visible) {
        const isScored = Number.isFinite(trial.score);
        const classes = ["dot", isScored ? "" : "skipped", finished && trial.id === series.best ? "winner" : "", state.selected === trial.id ? "chosen" : ""].filter(Boolean).join(" ");
        const dot = svg("circle", {
          class: classes, cx: x(trial.id), cy: isScored ? y(trial.score) : height - bottom, r: finished && trial.id === series.best ? 5.5 : 3.6,
          tabindex: "0", role: "button", "aria-label": `Version ${trial.id}, ${isScored ? share(trial.score) : "not scored"}`,
        });
        const pick = () => {
          state.selected = trial.id;
          renderLive();
        };
        dot.addEventListener("click", pick);
        dot.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            pick();
          }
        });
        group.append(dot);
      }
    }
    parts.push(group);
  });
  el("plot").replaceChildren(...parts);
  const keys = many
    ? view.series.map((series, index) => [`key key-${index}`, series.name])
    : [["key key-line", "Best so far"], ["key key-dot", "A version"], ...(view.live ? [] : [["key key-winner", "Winner"]])];
  el("legend").replaceChildren(...keys.map(([className, label]) => {
    const item = node("li", className);
    item.append(node("span", "swatch"), document.createTextNode(label));
    return item;
  }));
}

function renderFeed(seen) {
  let best = null;
  const lines = [];
  for (const trial of seen) {
    let text;
    if (trial.id === 0) text = `Start: ${share(trial.score)}`;
    else if (!Number.isFinite(trial.score)) text = `Version ${trial.id}: ${trial.status === "duplicate" ? "a repeat, skipped" : "unusable, skipped"}`;
    else {
      const from = trial.parents.map((id) => (id === 0 ? "start" : id)).join(" and ");
      text = `Version ${trial.id}, from ${from}: ${share(trial.score)}${trial.score > best ? ", new best" : ""}`;
    }
    const gain = trial.id !== 0 && Number.isFinite(trial.score) && trial.score > best;
    if (Number.isFinite(trial.score)) best = best === null ? trial.score : Math.max(best, trial.score);
    const item = node("li", gain ? "gain" : "");
    const button = node("button", "feed-line", text);
    button.type = "button";
    button.setAttribute("aria-current", String(trial.id === state.selected));
    button.addEventListener("click", () => {
      state.selected = trial.id;
      renderLive();
    });
    item.append(button);
    lines.push(item);
  }
  el("feed").replaceChildren(...lines.reverse());
}

function renderVersion(series) {
  const trial = series.trials.slice(0, state.shown).find((entry) => entry.id === state.selected);
  el("version-title").hidden = !trial;
  el("version-text").hidden = !trial;
  if (!trial) {
    el("version-meta").textContent = "";
    return;
  }
  const finished = !state.view.live && state.shown >= series.trials.length;
  el("version-title").textContent = trial.id === 0 ? "Starting version" : `Version ${trial.id}${trial.id === series.best && finished ? ", the winner" : ""}`;
  const facts = [Number.isFinite(trial.score) ? `${share(trial.score)} on the learning examples` : "Not scored"];
  if (trial.note && trial.note !== "Starting candidate") facts.push(trial.note);
  el("version-meta").textContent = facts.join(". ").replace(/\.+$/, "") + ".";
  el("version-text").textContent = trial.text || (state.view.live
    ? "The text of each version appears here when the run finishes."
    : "This attempt did not produce a usable version.");
}

// Loops that improve the visitor's prompt or function

function scoringSettings() {
  if (state.objective === "number") {
    const tolerance = Number.parseFloat(state.scoring.tolerance);
    return Number.isFinite(tolerance) && tolerance > 0 ? {tolerance} : null;
  }
  if (state.objective === "json") {
    const fields = state.scoring.fields.split(",").map((item) => item.trim()).filter(Boolean);
    return fields.length ? {fields} : null;
  }
  if (state.objective === "judge") {
    const rubric = state.scoring.rubric.trim();
    return rubric ? {rubric} : null;
  }
  return null;
}

function drivenRequest() {
  const program = state.target === "program";
  if (program && !state.program.source.trim()) throw new Error("Select “Changes” on the canvas and enter the function you want to improve.");
  if (program && !/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(state.program.entry)) throw new Error("Select “Changes” and enter the name of the function to call.");
  if (!program && !state.prompt.trim()) throw new Error("Select “Changes” on the canvas and enter the prompt you want to improve.");
  const {train, holdout} = parseCases(state.cases[casesKey()]);
  const minimum = state.catalog.limits.driven.min_holdout_cases;
  if (!train.length) throw new Error("Select “Tests on” and add examples to learn from.");
  if (holdout.length < minimum) throw new Error(`Select “Tests on” and add examples until at least ${minimum} are held back. Fewer can't show a real gain.`);
  const trials = el("trials").valueAsNumber;
  if (!Number.isInteger(trials)) throw new Error("Enter a whole number of revisions.");
  const body = {task: {train, holdout}, max_trials: trials, objective: state.objective};
  if (program) body.artifact = {type: "program", language: state.program.language, entry: state.program.entry, source: state.program.source};
  else body.prompt = state.prompt;
  const scoring = scoringSettings();
  if (scoring) body.scoring = scoring;
  if (el("all-loops").checked) body.loops = "all";
  else body.loop = state.loop.id;
  return body;
}

function copyHandoff() {
  let body;
  try {
    body = drivenRequest();
  } catch (error) {
    notice("stage-message", error.message, true);
    return;
  }
  const program = Boolean(body.artifact);
  const which = body.loops ? "every suitable loop (\"loops\": \"all\")" : `the ${state.loop.id} loop`;
  const settings = body.scoring ? ` and scoring ${JSON.stringify(body.scoring)}` : "";
  const text = [
    `Go to ${location.origin}/llms.txt and follow it to improve my ${program ? "function" : "prompt"} with ${which}.`,
    `Use max_trials ${body.max_trials} and objective "${body.objective}"${settings}.`,
    program
      ? "Run each 'run' work item yourself, in a sandbox, and reply with the function's result. Answer the other work items as the model."
      : "Answer the work items yourself, as the prompt being tested would.",
    "Give me the run's page link as soon as it starts. When the run is done, tell me the verdict, the holdout scores before and after, the p-value, and the version you recommend, in full.",
    "", program ? "Function (artifact):" : "Prompt:", program ? JSON.stringify(body.artifact) : body.prompt, "", "Task:", JSON.stringify(body.task),
  ].join("\n");
  void copy(text, "stage-message", "Message copied. Paste it into your assistant and it will run this loop for you.");
}

async function askModel(item) {
  let response;
  try {
    response = await fetch(OPENROUTER, {
      method: "POST",
      headers: {Authorization: `Bearer ${state.key}`, "Content-Type": "application/json"},
      body: JSON.stringify({
        model: state.model, temperature: 0, max_tokens: item.kind === "propose" ? 4000 : item.kind === "judge" ? 16 : 600,
        messages: [{role: "system", content: item.system}, {role: "user", content: item.input}],
      }),
      credentials: "omit", referrerPolicy: "no-referrer",
    });
  } catch {
    throw new Error("Could not reach OpenRouter from this browser.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("OpenRouter returned an unreadable reply.");
  }
  if (!response.ok) {
    const reason = data?.error?.message || `status ${response.status}`;
    throw new Error(response.status === 401 ? "OpenRouter did not accept that key." : `OpenRouter refused the call: ${reason}`);
  }
  const output = data?.choices?.[0]?.message?.content;
  if (typeof output !== "string") throw new Error("OpenRouter returned no answer for a call.");
  return output;
}

// Each call runs in a fresh worker whose own security policy blocks the network.
function runInWorker(item, seconds = RUN_SECONDS) {
  return new Promise((resolve) => {
    let worker;
    try {
      worker = new Worker("/runner.js");
    } catch {
      resolve("Error: this browser could not start a sandbox.");
      return;
    }
    const finish = (output) => {
      clearTimeout(timer);
      worker.terminate();
      resolve(output);
    };
    const timer = setTimeout(() => finish(`Error: timed out after ${seconds} seconds`), seconds * 1000);
    worker.addEventListener("message", (event) => finish(String(event.data)));
    worker.addEventListener("error", (event) => {
      event.preventDefault();
      finish(`Error: ${event.message || "the function could not run"}`);
    });
    worker.postMessage({source: item.source, entry: item.entry, input: item.input});
  });
}

async function sandboxReady() {
  const output = await runInWorker({source: "function ping() { return \"ok\"; }", entry: "ping", input: ""});
  if (output !== "ok") throw new Error("This browser blocks the code sandbox, so it can't run your function. Hand the run to your assistant instead.");
}

function answerItem(item) {
  if (item.kind !== "run") return askModel(item);
  if (item.language !== "javascript") {
    return Promise.reject(new Error("This run needs Python functions run, which this page can't do. Hand it to your assistant instead."));
  }
  return runInWorker(item);
}

async function answerAll(work, onProgress) {
  const outputs = [];
  let next = 0;
  const worker = async () => {
    while (next < work.length && !state.stop) {
      const item = work[next++];
      outputs.push({id: item.id, output: await answerItem(item)});
      onProgress(outputs.length);
    }
  };
  await Promise.all(Array.from({length: Math.min(4, work.length)}, worker));
  return outputs;
}

function stageText(item) {
  const [kind, text] = STAGES[item.stage] || ["harness", "Working"];
  return item.kind === "run" ? [kind, text.replace("Scoring", "Running").replace("the original", "the original function")] : [kind, text];
}

function drivenView(result) {
  const entries = Object.entries(result.results || {});
  const series = entries.map(([id, item]) => ({
    name: loopName(id), best: item.candidate_trial_id ?? 0,
    trials: item.trials.map((trial) => ({
      id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score, text: artifactText(trial.artifact), note: trial.reason || "",
    })),
  }));
  const table = entries.map(([id, item]) => ({
    name: loopName(id), tag: VERDICTS[item.verdict], tone: item.verdict,
    cells: [
      `Held back ${share(item.baseline_holdout)} to ${share(item.candidate_holdout)}`,
      `${item.evidence.improved_cases} fixed, ${item.evidence.regressed_cases} broken`,
      `p = ${pValue(item.evidence.p_value)}`,
    ],
  }));
  for (const [id, reason] of Object.entries(result.failed || {})) {
    table.push({name: loopName(id), tag: "Stopped", tone: "not_improved", cells: [reason]});
    series.push({name: loopName(id), best: 0, trials: []});
  }
  const advice = result.recommendation;
  const first = entries[0]?.[1];
  const noun = artifactNoun(advice?.artifact);
  return {
    driven: true,
    headline: advice?.reason || "The run stopped before it produced a result.",
    series, table,
    holdout: first ? first.candidate_holdout : null,
    adopt: advice && {
      confirmed: advice.action === "adopt",
      noun,
      title: advice.action === "adopt" ? `${noun[0].toUpperCase()}${noun.slice(1)} to adopt` : advice.candidate_to_retest ? "Candidate worth retesting" : `Keep your ${noun}`,
      text: artifactText(advice.action === "adopt" ? advice.artifact : advice.candidate_to_retest || advice.artifact),
    },
  };
}

function progressView(view) {
  const entries = view.loops.map((id) => [id, view.progress?.[id] || {state: "working", doing: "Starting"}]);
  const series = entries.map(([id, entry]) => ({
    name: loopName(id), best: null, doing: entry.doing, planned: entry.planned_trials != null ? entry.planned_trials + 1 : undefined,
    trials: (entry.trace || []).map((trial) => ({id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score, text: "", note: ""})),
  }));
  const table = entries.map(([id, entry]) => ({
    name: loopName(id),
    tag: entry.state === "done" ? VERDICTS[entry.verdict] : entry.state === "failed" ? "Stopped" : "Working",
    tone: entry.state === "done" ? entry.verdict : entry.state === "failed" ? "not_improved" : "working",
    cells: entry.state === "working"
      ? [entry.doing, entry.best_train == null ? "Not scored yet" : `Best training score ${share(entry.best_train)}`]
      : entry.state === "failed" ? [entry.reason] : ["Finished; waiting for the other loops"],
  }));
  const waiting = view.pending_work ?? view.work?.length ?? 0;
  return {
    driven: true, live: true, series, table, holdout: null,
    headline: `Waiting for ${plural(waiting, "answer")}. ${view.answered} received so far; this page updates by itself.`,
  };
}

function runSummary(view) {
  const request = view.request;
  if (!request) return "Anyone with this link can see the run.";
  const objective = objectiveById(request.objective).name.toLowerCase();
  return (
    `Improving a ${artifactNoun(request.artifact)} with ${request.train_cases} learning and ${request.holdout_cases} held-back examples, ` +
    `scored by ${objective}. Anyone with this link can see the run.`
  );
}

function renderRun(view, keep = true) {
  state.run.view = view;
  const working = view.state === "needs_work";
  el("loop-name").textContent = `Run of ${listed(view.loops.map(loopName))}`;
  el("loop-summary").textContent = runSummary(view);
  el("run-continue").hidden = !working || state.running;
  el("run-stop").hidden = !state.running;
  el("run-delete").disabled = state.running;
  if (working) showView(progressView(view), false, keep);
  else showView(drivenView(view), false, keep);
  if (working) el("canvas").classList.add("playing");
}

function enterRun(view) {
  stopWatching();
  state.run = {id: view.id, view, poll: null, delay: 3000};
  state.loop = loopById(view.loops[0]) || state.loop;
  state.swaps = {};
  state.node = "target";
  const path = `/runs/${view.id}`;
  if (location.pathname !== path) history.replaceState(null, "", path);
  setMode("run");
  el("run-copy").hidden = false;
  el("run-delete").hidden = false;
  el("run-delete").textContent = "Delete this run";
  renderNodes();
  renderInspector();
  renderRun(view, false);
  watchRun();
}

function watchRun() {
  const run = state.run;
  if (!run) return;
  clearTimeout(run.poll);
  if (run.view?.state !== "needs_work" || state.running || run.deleted || document.hidden) return;
  run.poll = setTimeout(async () => {
    if (state.run !== run || state.running) return;
    try {
      const view = await call(`/api/runs/${run.id}?work=0`);
      if (state.run !== run) return;
      const changed = view.answered !== run.view.answered || view.state !== run.view.state;
      run.delay = changed ? 3000 : Math.min(15000, Math.round(run.delay * 1.5));
      if (changed) {
        renderRun(view);
        if (view.state !== "needs_work") notice("stage-message", "The run finished.");
      }
    } catch (error) {
      run.delay = 15000;
      if (error.status === 404) {
        run.deleted = true;
        notice("stage-message", "This run no longer exists.", true);
        return;
      }
    }
    watchRun();
  }, run.delay);
}

async function openRun(id) {
  setMode("run");
  state.loop = state.loop || state.catalog.loops.find((loop) => loop.mode === "driven");
  el("loop-name").textContent = "Loading the run…";
  el("loop-summary").textContent = "";
  try {
    enterRun(await call(`/api/runs/${id}?work=0`));
  } catch (error) {
    el("loop-name").textContent = error.status === 404 ? "This run doesn't exist" : "The run could not be loaded";
    el("loop-summary").textContent = error.status === 404
      ? "It may have been deleted or expired. Unfinished runs are kept for 14 days after their last step, finished runs for 90 days."
      : `${error.message} Reload the page to try again.`;
    for (const id of ["run-continue", "run-stop", "run-copy", "run-delete"]) el(id).hidden = true;
    renderNodes();
    renderInspector();
    clearView("Nothing to show.");
  }
}

async function driveRun(view, token) {
  let answered = 0;
  while (view.state === "needs_work" && !state.stop && token === state.token) {
    renderRun(view);
    const [kind, text] = stageText(view.work[0]);
    setBusy(kind);
    const base = answered;
    setHub(String(base), "answers so far", text);
    const outputs = await answerAll(view.work, (done) => setHub(String(base + done), "answers so far", text));
    answered += outputs.length;
    if (!outputs.length) break;
    view = await call(view.submit.path, {outputs});
  }
  return {view, answered};
}

function needsKey() {
  if (!state.key) {
    state.node = "proposer";
    if (state.mode === "loop") {
      state.who = "key";
      syncToolbar();
    }
    renderNodes();
    renderInspector();
    el("or-key")?.focus();
    throw new Error("Paste your OpenRouter key below, then try again.");
  }
  if (!state.model) throw new Error("Enter an OpenRouter model, for example openai/gpt-4.1-mini.");
}

async function drive(start) {
  const token = ++state.token;
  state.running = true;
  state.stop = false;
  el("canvas").classList.add("playing");
  notice("stage-message");
  let answered = 0;
  try {
    const view = await start();
    if (token !== state.token) return;
    if (state.mode !== "run" || state.run?.id !== view.id) enterRun(view);
    el("run-continue").hidden = true;
    el("run-stop").hidden = false;
    const result = await driveRun(view, token);
    answered = result.answered;
    if (token !== state.token) return;
    renderRun(result.view);
    notice("stage-message", result.view.state === "needs_work"
      ? `Stopped after ${plural(answered, "answer")}. Continue whenever you like; the run keeps its answers.`
      : `Finished after ${plural(answered, "answer")}.`);
  } catch (error) {
    if (token === state.token) notice("stage-message", error.message, true);
  } finally {
    state.running = false;
    setBusy(null);
    el("run").disabled = false;
    if (state.mode === "run" && state.run) {
      el("run-stop").hidden = true;
      el("run-delete").disabled = false;
      if (state.run.view) renderRun(state.run.view);
      watchRun();
    } else {
      el("canvas").classList.remove("playing");
    }
  }
}

async function runDriven() {
  if (state.running) return;
  let body;
  try {
    body = drivenRequest();
    needsKey();
    if (body.artifact?.language === "python") throw new Error("This page runs JavaScript only. Python functions run with your assistant: choose “My AI assistant” and copy the message.");
    if (body.artifact) await sandboxReady();
  } catch (error) {
    notice("stage-message", error.message, true);
    return;
  }
  el("run").disabled = true;
  await drive(() => call("/api/runs", body));
}

async function continueRun() {
  if (state.running || !state.run) return;
  try {
    needsKey();
    if (state.run.view?.request?.artifact?.type === "program") {
      if (state.run.view.request.artifact.language !== "javascript") throw new Error("This run needs Python functions run, which this page can't do. Hand it to your assistant instead.");
      await sandboxReady();
    }
  } catch (error) {
    notice("stage-message", error.message, true);
    return;
  }
  const id = state.run.id;
  await drive(() => call(`/api/runs/${id}`));
}

let deleteArmed = null;

async function deleteRun() {
  const button = el("run-delete");
  if (!state.run || state.running) return;
  if (!deleteArmed) {
    button.textContent = "Select again to delete";
    deleteArmed = setTimeout(() => {
      deleteArmed = null;
      button.textContent = "Delete this run";
    }, 4000);
    return;
  }
  clearTimeout(deleteArmed);
  deleteArmed = null;
  try {
    await call(`/api/runs/${state.run.id}`, undefined, "DELETE");
    stopWatching();
    state.run.deleted = true;
    clearView("This run was deleted.");
    el("loop-name").textContent = "Run deleted";
    el("loop-summary").textContent = "Its prompt or function, cases and answers are gone from RSI Library.";
    for (const id of ["run-continue", "run-stop", "run-copy", "run-delete"]) el(id).hidden = true;
    notice("stage-message");
  } catch (error) {
    button.textContent = "Delete this run";
    notice("stage-message", error.message, true);
  }
}

// Page wiring

async function start() {
  el("mcp-url").textContent = `${location.origin}/mcp`;
  el("guide-url").textContent = `${location.origin}/llms.txt`;
  renderRing();
  try {
    state.catalog = await call("/api/catalog");
  } catch (error) {
    el("loop-name").textContent = "The loops could not be loaded";
    el("loop-summary").textContent = `${error.message} Reload the page to try again.`;
    return;
  }
  const catalog = state.catalog;
  state.cases.offline = casesToText(catalog.example_task);
  state.cases.driven = casesToText(catalog.example_prompt_run.task);
  state.cases.code = casesToText(catalog.example_program_run.task);
  state.prompt = catalog.example_prompt_run.prompt;
  state.program = {...catalog.example_program_run.artifact};
  renderTemplates();
  renderMeasured();
  renderFilters();
  renderGalleryMeta();
  renderGallery();
  const run = RUN_PATH.exec(location.pathname);
  if (run) {
    await openRun(run[1]);
    return;
  }
  const params = new URLSearchParams(location.search);
  const method = catalog.gallery.methods.find((entry) => entry.id === params.get("method") && !entry.runnable);
  if (method) {
    state.loop = catalog.loops[0];
    showMethod(method.id);
    return;
  }
  const loop = loopById(params.get("loop")) || catalog.loops[0];
  if (params.get("target") === "program" && loop.artifact_types?.includes("program")) {
    selectCode(loop.id);
    return;
  }
  selectLoop(loop.id);
  // A shared link can carry the budget, seed and swapped parts of an offline loop.
  if (state.loop.mode === "offline") {
    let changed = false;
    for (const key of ["budget", "seed"]) {
      if (/^\d{1,10}$/.test(params.get(key) || "")) {
        el(key).value = params.get(key);
        changed = true;
      }
    }
    for (const kind of KINDS) {
      const wanted = params.get(kind);
      if (wanted && wanted !== state.loop.parts[kind].id && catalog.components[kind].some((entry) => entry.id === wanted)) {
        state.swaps[kind] = wanted;
        changed = true;
      }
    }
    if (changed) {
      renderNodes();
      renderInspector();
      void runOffline();
    }
  }
}

function shareLink() {
  const params = new URLSearchParams({loop: state.loop.id});
  if (state.loop.mode === "offline") {
    params.set("budget", el("budget").value);
    params.set("seed", el("seed").value);
    for (const [kind, id] of Object.entries(state.swaps)) params.set(kind, id);
  } else if (state.target === "program") {
    params.set("target", "program");
  }
  const url = `${location.origin}/?${params}`;
  history.replaceState(null, "", url);
  void copy(url, "stage-message", state.loop.mode === "offline"
    ? "Link copied. It opens this loop with these parts, budget and seed."
    : `Link copied. It opens this template; your ${state.target === "program" ? "function" : "prompt"} and examples are not included.`);
}

el("toolbar").addEventListener("submit", runOffline);
el("compare").addEventListener("click", compareOffline);
el("handoff").addEventListener("click", copyHandoff);
el("share").addEventListener("click", shareLink);
el("compare-clear").addEventListener("click", () => {
  compared = [];
  renderCompare();
  renderGallery();
});
el("stop").addEventListener("click", () => {
  state.stop = true;
});
el("run-stop").addEventListener("click", () => {
  state.stop = true;
});
el("run-continue").addEventListener("click", () => void continueRun());
el("run-copy").addEventListener("click", () => copy(`${location.origin}/runs/${state.run.id}`, "stage-message", "Link copied. Anyone with it can see this run."));
el("run-delete").addEventListener("click", () => void deleteRun());
el("method-run").addEventListener("click", () => {
  if (state.method?.closest) selectLoop(state.method.closest, true);
});
el("method-back").addEventListener("click", () => selectLoop(state.loop?.id || state.catalog.loops[0].id, true));
for (const id of ["budget", "seed"]) {
  el(id).addEventListener("change", () => {
    if (state.mode === "loop" && state.loop?.mode === "offline") void runOffline();
  });
}
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.mode === "run") watchRun();
});
el("copy-mcp").addEventListener("click", () => copy(el("mcp-url").textContent, "ask-note", "Connector address copied."));
for (const button of document.querySelectorAll(".ask")) {
  button.addEventListener("click", () => copy(button.dataset.ask.replace("{origin}", location.origin), "ask-note", "Message copied. Paste it into your assistant."));
}

void start();
