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
  baseline_train: ["harness", "Scoring your prompt on the learning examples"],
  propose: ["proposer", "Writing a revision"],
  candidate_train: ["harness", "Scoring a revision on the learning examples"],
  baseline_holdout: ["judge", "Final check: your prompt on the held-back examples"],
  candidate_holdout: ["judge", "Final check: the best revision on the held-back examples"],
};
const OPENROUTER = "https://openrouter.ai/api/v1/chat/completions";
const DIVIDER = "--- held back for the final check ---";
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 0});
const calm = window.matchMedia("(prefers-reduced-motion: reduce)");

const state = {
  catalog: null, loop: null, swaps: {}, node: "search", token: 0,
  cases: {offline: "", driven: ""}, prompt: "", objective: "normalized", who: "assistant", key: "", model: "openai/gpt-4.1-mini",
  view: null, shown: 0, selected: 0, series: 0, timer: null, running: false, stop: false,
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

async function call(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? {cache: "no-store"} : {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), cache: "no-store",
    });
  } catch {
    throw new Error("Could not reach RSI Library. Check your connection and try again.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("RSI Library returned an unreadable reply.");
  }
  if (!response.ok) throw new Error(typeof data?.error === "string" ? data.error : "The request failed.");
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

// Examples are typed one per line: "message → label". A divider line separates the held-back cases.

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
    if (!match) throw new Error(`Line ${index + 1} needs an arrow between the message and its label, like “Where is my invoice? → billing”.`);
    groups[groups.length - 1].push({input: match[1], expected: match[2]});
  }
  if (groups.length > 2) throw new Error("Use one divider line: learning examples above it, held-back examples below.");
  if (groups.length === 2) return {train: groups[0], holdout: groups[1], split: "yours"};
  // No divider: hold back every second example of each label, so both halves see every label.
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
    const minimum = mode === "driven" ? state.catalog.limits.driven.min_holdout_cases : 0;
    const short = holdout.length < minimum ? ` Add ${minimum - holdout.length} more held-back examples so the final check can show a real gain.` : "";
    const how = split === "automatic" ? " Split automatically; add a line of dashes to split them yourself." : "";
    return {ok: !short && train.length > 0, text: `${plural(train.length, "example")} to learn from, ${holdout.length} held back for the final check.${how}${short}`};
  } catch (error) {
    return {ok: false, text: error.message};
  }
}

// Templates and gallery

function renderTemplates() {
  for (const mode of ["offline", "driven"]) {
    const items = state.catalog.loops.filter((loop) => loop.mode === mode).map((loop) => {
      const item = node("li");
      const button = node("button", "template");
      button.type = "button";
      button.dataset.loop = loop.id;
      button.append(node("span", "template-name", loop.name), node("span", "template-note", loop.parts.search.name));
      button.addEventListener("click", () => selectLoop(loop.id, true));
      item.append(button);
      return item;
    });
    el(`templates-${mode}`).replaceChildren(...items);
  }
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

function openOnCanvas(id) {
  selectLoop(id, true);
  el("canvas").scrollIntoView({block: "center"});
}

function renderGallery() {
  const all = state.catalog.gallery.methods;
  const shown = all.filter(matches);
  el("gallery-count").textContent = shown.length === all.length
    ? `${all.length} loops: ${all.filter((entry) => entry.runnable).length} you can run here and ${all.filter((entry) => !entry.runnable).length} published methods.`
    : `${plural(shown.length, "loop")} of ${all.length} match.`;
  const loopName = (id) => state.catalog.loops.find((loop) => loop.id === id)?.name;
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
    if (entry.closest) {
      const open = node("button", entry.runnable ? "primary" : "", entry.runnable ? "Open on the canvas" : `Run the closest loop: ${loopName(entry.closest)}`);
      open.type = "button";
      open.addEventListener("click", () => openOnCanvas(entry.closest));
      actions.append(open);
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
    card.append(node("h3", "", entry.name), node("p", "card-origin", `${entry.year}, ${entry.origin}`), tags, facts, more, actions);
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
    ["Runs here", (entry) => (entry.runnable ? "Yes" : entry.closest ? `No. Closest: ${state.catalog.loops.find((loop) => loop.id === entry.closest)?.name}` : "No")],
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

function partFor(kind) {
  const loop = state.loop;
  if (loop.mode === "offline" && state.swaps[kind]) {
    return state.catalog.components[kind].find((entry) => entry.id === state.swaps[kind]) || loop.parts[kind];
  }
  if (loop.mode === "driven" && kind === "judge") {
    return state.objective === "exact"
      ? {id: "exact", name: "Exact match", description: "The answer must match the label exactly."}
      : {id: "normalized", name: "Text match", description: "Match the label, ignoring case and spacing."};
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
    if (state.swaps[kind]) button.classList.add("swapped");
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
  const label = node("label", "", mode === "driven" ? "Your examples, one per line" : "Examples, one per line");
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

function renderInspector() {
  const kind = state.node;
  const loop = state.loop;
  const blocks = [];
  const heading = node("h2", "", `${KIND_LABELS[kind]}: ${partFor(kind).name}`);
  if (loop.mode === "offline") {
    const options = state.catalog.components[kind];
    blocks.push(optionList(kind, options, state.swaps[kind] || loop.parts[kind].id, (id) => {
      if (id === loop.parts[kind].id) delete state.swaps[kind];
      else state.swaps[kind] = id;
      renderNodes();
      renderInspector();
      void runOffline();
    }));
    if (kind === "harness") blocks.push(casesEditor("offline"));
  } else if (kind === "target") {
    const label = node("label", "", "The prompt to improve");
    const area = node("textarea");
    area.rows = 4;
    area.maxLength = 4000;
    area.spellcheck = false;
    area.value = state.prompt;
    area.addEventListener("input", () => {
      state.prompt = area.value;
    });
    label.append(area);
    blocks.push(label);
  } else if (kind === "harness") {
    blocks.push(casesEditor("driven"));
  } else if (kind === "judge") {
    blocks.push(optionList(kind, [
      {id: "normalized", name: "Text match", description: "Match the label, ignoring case and spacing."},
      {id: "exact", name: "Exact match", description: "The answer must match the label character for character."},
    ], state.objective, (id) => {
      state.objective = id;
      renderNodes();
      renderInspector();
    }));
  } else if (kind === "proposer") {
    blocks.push(optionList(kind, [
      {id: "assistant", name: "My AI assistant", description: "Copy a message that hands the run to your assistant. It answers the model calls itself."},
      {id: "key", name: "My OpenRouter key, here", description: "This page calls your model directly. The key stays in this tab and is never sent to RSI Library."},
    ], state.who, (id) => {
      state.who = id;
      renderNodes();
      renderInspector();
      syncToolbar();
    }));
    if (state.who === "key") {
      const pair = node("div", "pair");
      const keyLabel = node("label", "", "OpenRouter API key");
      const key = node("input");
      key.type = "password";
      key.autocomplete = "off";
      key.placeholder = "sk-or-…";
      key.value = state.key;
      key.id = "or-key";
      key.addEventListener("input", () => {
        state.key = key.value.trim();
      });
      keyLabel.append(key);
      const modelLabel = node("label", "", "Model");
      const model = node("input");
      model.type = "text";
      model.spellcheck = false;
      model.value = state.model;
      model.addEventListener("input", () => {
        state.model = model.value.trim();
      });
      model.addEventListener("change", renderNodes);
      modelLabel.append(model);
      pair.append(keyLabel, modelLabel);
      blocks.push(pair, node("p", "hint", "Model calls are billed to your OpenRouter account."));
    }
  } else {
    blocks.push(node("p", "inspector-text", `${partFor(kind).description} This part is fixed for ${loop.name}. Load another template to change how the loop picks what to build on.`));
  }
  el("inspector").replaceChildren(heading, ...blocks);
}

// Loading a template

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
}

function selectLoop(id, userAction = false) {
  const loop = state.catalog.loops.find((entry) => entry.id === id) || state.catalog.loops[0];
  state.loop = loop;
  state.swaps = {};
  state.token += 1;
  state.stop = true;
  state.node = loop.mode === "offline" ? "search" : "target";
  for (const button of document.querySelectorAll(".template")) {
    button.setAttribute("aria-current", String(button.dataset.loop === loop.id));
  }
  el("loop-name").textContent = loop.name;
  el("loop-summary").textContent = loop.mode === "offline"
    ? `${loop.summary} Select any part of the loop to change it; the run updates as you go.`
    : `${loop.summary} Put your prompt and examples into the loop, then choose who answers.`;
  notice("stage-message");
  setBusy(null);
  syncToolbar();
  renderNodes();
  renderInspector();
  clearView(loop.mode === "offline" ? "Running…" : "Add your prompt and examples, then run the loop.");
  if (userAction) history.replaceState(null, "", `?loop=${encodeURIComponent(loop.id)}`);
  if (loop.mode === "offline") {
    el("budget").value = loop.defaults.budget;
    el("seed").value = loop.defaults.seed;
    void runOffline();
  }
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

function showView(view, animate) {
  clearInterval(state.timer);
  state.view = view;
  state.series = 0;
  const total = view.series[0].trials.length;
  el("script-fold").hidden = !view.snippet;
  el("api-call").textContent = view.snippet || "";
  renderVerdicts();
  if (!animate || calm.matches || total < 3 || view.series.length > 1) {
    finishPlayback();
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

function finishPlayback() {
  clearInterval(state.timer);
  const view = state.view;
  const series = view.series[state.series];
  state.shown = series.trials.length;
  state.selected = series.best ?? 0;
  el("canvas").classList.remove("playing");
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
  const total = series.trials.length;
  const done = state.shown >= total;
  const now = progress(series, state.shown);
  const latest = now.seen[now.seen.length - 1];
  if (view.series.length > 1 && !view.driven) {
    setHub(String(view.series.length), "loops compared", "same examples, budget and seed");
  } else if (total === 0) {
    setHub("–", "no versions scored", "");
  } else {
    setHub(share(now.best), "best training score", done
      ? (view.holdout != null ? `${share(view.holdout)} on held-back examples` : `${plural(total - 1, "version")} tried`)
      : `version ${latest.id} of ${total - 1}`);
  }
  const stat = (term, value) => [node("dt", "", term), node("dd", "", value)];
  const stats = view.series.length > 1 && !view.driven ? [] : [
    ...stat("Started at", share(now.start)),
    ...stat("Best so far", share(now.best)),
    ...stat("Versions tried", String(Math.max(0, now.seen.length - 1))),
    ...stat("New bests", String(now.improvements)),
    ...stat("Last gain at", now.improvements ? `version ${now.lastGain}` : "none yet"),
    ...stat("Held back", done && view.holdout != null ? share(view.holdout) : done ? "see below" : "after the run"),
  ];
  el("stats").replaceChildren(...stats);
  renderPlot();
  renderFeed(now.seen, series);
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
    const button = node("button", "", "Copy prompt");
    button.type = "button";
    const status = node("span", "hint");
    status.id = "adopt-note";
    status.setAttribute("role", "status");
    button.addEventListener("click", () => copy(view.adopt.text, "adopt-note", "Prompt copied."));
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
  const count = Math.max(2, ...view.series.map((item) => item.trials.length));
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
      const finished = state.shown >= series.trials.length;
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
    : [["key key-line", "Best so far"], ["key key-dot", "A version"], ["key key-winner", "Winner"]];
  el("legend").replaceChildren(...keys.map(([className, label]) => {
    const item = node("li", className);
    item.append(node("span", "swatch"), document.createTextNode(label));
    return item;
  }));
}

function renderFeed(seen, series) {
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
  el("version-title").textContent = trial.id === 0 ? "Starting version" : `Version ${trial.id}${trial.id === series.best && state.shown >= series.trials.length ? ", the winner" : ""}`;
  const facts = [Number.isFinite(trial.score) ? `${share(trial.score)} on the learning examples` : "Not scored"];
  if (trial.note && trial.note !== "Starting candidate") facts.push(trial.note);
  el("version-meta").textContent = facts.join(". ").replace(/\.+$/, "") + ".";
  el("version-text").textContent = trial.text || "This attempt did not produce a usable prompt.";
}

// Loops that improve the visitor's prompt

function drivenRequest() {
  if (!state.prompt.trim()) throw new Error("Select “Changes” on the canvas and enter the prompt you want to improve.");
  const {train, holdout} = parseCases(state.cases.driven);
  const minimum = state.catalog.limits.driven.min_holdout_cases;
  if (!train.length) throw new Error("Select “Tests on” and add examples to learn from.");
  if (holdout.length < minimum) throw new Error(`Select “Tests on” and add examples until at least ${minimum} are held back. Fewer can't show a real gain.`);
  const trials = el("trials").valueAsNumber;
  if (!Number.isInteger(trials)) throw new Error("Enter a whole number of revisions.");
  const body = {prompt: state.prompt, task: {train, holdout}, max_trials: trials, objective: state.objective};
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
  const which = body.loops ? "every loop (\"loops\": \"all\")" : `the ${state.loop.id} loop`;
  const text = [
    `Go to ${location.origin}/llms.txt and follow it to improve my prompt with ${which}.`,
    `Use max_trials ${body.max_trials} and objective "${body.objective}". Answer the work items yourself, as the prompt being tested would.`,
    "When the run is done, tell me the verdict, the holdout scores before and after, the p-value, and the prompt you recommend, in full.",
    "", "Prompt:", body.prompt, "", "Task:", JSON.stringify(body.task),
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
        model: state.model, temperature: 0, max_tokens: item.kind === "propose" ? 2000 : 300,
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

async function answerAll(work, onProgress) {
  const outputs = [];
  let next = 0;
  const worker = async () => {
    while (next < work.length && !state.stop) {
      const item = work[next++];
      outputs.push({id: item.id, output: await askModel(item)});
      onProgress(outputs.length);
    }
  };
  await Promise.all(Array.from({length: Math.min(4, work.length)}, worker));
  return outputs;
}

function drivenView(result) {
  const name = (id) => state.catalog.loops.find((loop) => loop.id === id)?.name || id;
  const entries = Object.entries(result.results || {});
  const series = entries.map(([id, item]) => ({
    name: name(id), best: item.candidate_trial_id ?? 0,
    trials: item.trials.map((trial) => ({
      id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score, text: artifactText(trial.artifact), note: trial.reason || "",
    })),
  }));
  const table = entries.map(([id, item]) => ({
    name: name(id), tag: VERDICTS[item.verdict], tone: item.verdict,
    cells: [
      `Held back ${share(item.baseline_holdout)} to ${share(item.candidate_holdout)}`,
      `${item.evidence.improved_cases} fixed, ${item.evidence.regressed_cases} broken`,
      `p = ${pValue(item.evidence.p_value)}`,
    ],
  }));
  for (const [id, reason] of Object.entries(result.failed || {})) {
    table.push({name: name(id), tag: "Stopped", tone: "not_improved", cells: [reason]});
    series.push({name: name(id), best: 0, trials: []});
  }
  const advice = result.recommendation;
  const first = entries[0]?.[1];
  return {
    driven: true,
    headline: advice?.reason || "The run stopped before it produced a result.",
    series, table,
    holdout: first ? first.candidate_holdout : null,
    adopt: advice && {
      confirmed: advice.action === "adopt",
      title: advice.action === "adopt" ? "Prompt to adopt" : advice.candidate_to_retest ? "Candidate worth retesting" : "Keep your prompt",
      text: artifactText(advice.action === "adopt" ? advice.artifact : advice.candidate_to_retest || advice.artifact),
    },
  };
}

async function runDriven() {
  if (state.running) return;
  let body;
  try {
    body = drivenRequest();
    if (!state.key) {
      state.node = "proposer";
      state.who = "key";
      renderNodes();
      renderInspector();
      syncToolbar();
      el("or-key")?.focus();
      throw new Error("Paste your OpenRouter key below, then run again.");
    }
    if (!state.model) throw new Error("Enter an OpenRouter model, for example openai/gpt-4.1-mini.");
  } catch (error) {
    notice("stage-message", error.message, true);
    return;
  }
  const token = ++state.token;
  state.running = true;
  state.stop = false;
  el("run").disabled = true;
  el("stop").hidden = false;
  clearView("Running on your model…");
  el("canvas").classList.add("playing");
  notice("stage-message");
  let answered = 0;
  try {
    let view = await call("/api/runs", body);
    while (view.state === "needs_work" && !state.stop && token === state.token) {
      const [kind, text] = STAGES[view.work[0].stage] || ["harness", "Working"];
      setBusy(kind);
      const base = answered;
      setHub(String(base), "model calls so far", text);
      const outputs = await answerAll(view.work, (done) => setHub(String(base + done), "model calls so far", text));
      answered += outputs.length;
      if (!outputs.length) break;
      view = await call(view.submit.path, {outputs});
    }
    if (token !== state.token) return;
    if (view.state === "needs_work") {
      clearView(`Stopped after ${plural(answered, "model call")}.`);
    } else {
      showView(drivenView(view), true);
      notice("stage-message", `Finished after ${plural(answered, "model call")}.`);
    }
  } catch (error) {
    if (token === state.token) {
      clearView("The run stopped.");
      notice("stage-message", error.message, true);
    }
  } finally {
    state.running = false;
    setBusy(null);
    el("canvas").classList.remove("playing");
    el("run").disabled = false;
    el("stop").hidden = true;
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
  state.cases.offline = casesToText(state.catalog.example_task);
  state.cases.driven = casesToText(state.catalog.example_prompt_run.task);
  state.prompt = state.catalog.example_prompt_run.prompt;
  renderTemplates();
  renderFilters();
  renderGallery();
  const params = new URLSearchParams(location.search);
  selectLoop(params.get("loop") || state.catalog.loops[0].id);
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
      if (wanted && wanted !== state.loop.parts[kind].id && state.catalog.components[kind].some((entry) => entry.id === wanted)) {
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
  }
  const url = `${location.origin}/?${params}`;
  history.replaceState(null, "", url);
  void copy(url, "stage-message", state.loop.mode === "offline"
    ? "Link copied. It opens this loop with these parts, budget and seed."
    : "Link copied. It opens this template; your prompt and examples are not included.");
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
for (const id of ["budget", "seed"]) {
  el(id).addEventListener("change", () => {
    if (state.loop?.mode === "offline") void runOffline();
  });
}
el("copy-mcp").addEventListener("click", () => copy(el("mcp-url").textContent, "ask-note", "Connector address copied."));
for (const button of document.querySelectorAll(".ask")) {
  button.addEventListener("click", () => copy(button.dataset.ask.replace("{origin}", location.origin), "ask-note", "Message copied. Paste it into your assistant."));
}

void start();
