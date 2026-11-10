"use strict";

const el = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const KIND_LABELS = {target: "Changes", proposer: "Proposes", harness: "Tests on", judge: "Scores by", search: "Builds on"};
const PART_ORDER = ["target", "proposer", "harness", "judge", "search"];
const VERDICTS = {improved: "Significant gain", inconclusive: "Inconclusive", not_improved: "No gain"};
const STAGES = {
  baseline_train: "Scoring your prompt on the training cases",
  propose: "Writing a revision",
  candidate_train: "Scoring a revision on the training cases",
  baseline_holdout: "Scoring your prompt on the holdout cases",
  candidate_holdout: "Scoring the best revision on the holdout cases",
};
const OPENROUTER = "https://openrouter.ai/api/v1/chat/completions";
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 0});

const state = {catalog: null, loop: null, swaps: {}, shown: null, selected: null, running: false, stop: false, token: 0};

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

const pretty = (value) => JSON.stringify(value, null, 2);
const share = (value) => (Number.isFinite(value) ? percent.format(value) : "not scored");
const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

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
  return pretty(artifact);
}

// Shelf and loop sheet

function renderShelf() {
  for (const mode of ["offline", "driven"]) {
    const items = state.catalog.loops.filter((loop) => loop.mode === mode).map((loop) => {
      const item = node("li");
      const button = node("button", "shelf-item");
      button.type = "button";
      button.dataset.loop = loop.id;
      button.append(node("span", "shelf-name", loop.name), node("span", "shelf-id", loop.id));
      button.addEventListener("click", () => selectLoop(loop.id, true));
      item.append(button);
      return item;
    });
    el(`shelf-${mode}`).replaceChildren(...items);
  }
}

function renderParts() {
  const loop = state.loop;
  const items = PART_ORDER.map((kind) => {
    const item = node("li", "part");
    item.append(node("span", "part-kind", KIND_LABELS[kind]));
    const options = loop.mode === "offline" ? state.catalog.components[kind] : [];
    const current = state.swaps[kind] || loop.parts[kind].id;
    const chosen = options.find((entry) => entry.id === current) || loop.parts[kind];
    if (options.length > 1) {
      const select = node("select", "part-select");
      select.setAttribute("aria-label", `${KIND_LABELS[kind]}: choose a part`);
      for (const entry of options) {
        const option = node("option", "", entry.name);
        option.value = entry.id;
        select.append(option);
      }
      select.value = chosen.id;
      select.addEventListener("change", () => {
        if (select.value === loop.parts[kind].id) delete state.swaps[kind];
        else state.swaps[kind] = select.value;
        renderParts();
        void runOffline();
      });
      item.append(select);
    } else {
      item.append(node("strong", "part-name", chosen.name));
    }
    item.append(node("span", "part-text", chosen.description));
    if (state.swaps[kind]) item.classList.add("swapped");
    return item;
  });
  el("parts").replaceChildren(...items);
}

function selectLoop(id, userAction = false) {
  const loop = state.catalog.loops.find((entry) => entry.id === id) || state.catalog.loops[0];
  state.loop = loop;
  state.swaps = {};
  state.token += 1;
  state.stop = true;
  for (const button of document.querySelectorAll(".shelf-item")) {
    button.setAttribute("aria-current", button.dataset.loop === loop.id ? "true" : "false");
  }
  el("loop-name").textContent = loop.name;
  el("loop-summary").textContent = `${loop.summary} ${loop.improves}`;
  el("loop-source").hidden = !loop.inspired_by;
  if (loop.inspired_by) el("loop-source").href = loop.inspired_by;
  el("offline-form").hidden = loop.mode !== "offline";
  el("driven-form").hidden = loop.mode !== "driven";
  el("outcome").hidden = true;
  notice("offline-message");
  notice("driven-message");
  renderParts();
  if (userAction) history.replaceState(null, "", `?loop=${encodeURIComponent(loop.id)}#loops`);
  if (loop.mode === "offline") {
    el("budget").value = loop.defaults.budget;
    el("seed").value = loop.defaults.seed;
    void runOffline();
  } else {
    updateHandoff();
  }
}

// Offline loops

function offlineRequest() {
  const body = {loop: state.loop.id, budget: el("budget").valueAsNumber, seed: el("seed").valueAsNumber, ...state.swaps};
  const raw = el("offline-task").value.trim();
  if (raw && raw !== pretty(state.catalog.example_task)) {
    try {
      body.task = JSON.parse(raw);
    } catch {
      throw new Error("The task is not valid JSON. Check for a missing comma or quote.");
    }
  }
  return body;
}

function apiSnippet(path, body) {
  return `curl -s ${location.origin}${path} \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(body).replaceAll("'", "'\\''")}'`;
}

async function runOffline(event) {
  event?.preventDefault();
  const token = ++state.token;
  let body;
  try {
    body = offlineRequest();
    if (!Number.isInteger(body.budget) || !Number.isInteger(body.seed)) throw new Error("Enter whole numbers for versions and seed.");
  } catch (error) {
    notice("offline-message", error.message, true);
    return;
  }
  notice("offline-message");
  try {
    const result = await call("/api/run", body);
    if (token !== state.token) return;
    const trials = result.trials.map((trial) => ({
      id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score, text: artifactText(trial.candidate), note: trial.note,
    }));
    const holdout = result.holdout_score == null ? "" :
      ` On ${plural(result.holdout_cases, "holdout case")} it never saw, the winner scores ${share(result.holdout_score)}.`;
    const gained = result.best_score > result.baseline_score;
    showOutcome({
      headline: gained
        ? `Training score rose from ${share(result.baseline_score)} to ${share(result.best_score)} over ${plural(trials.length, "version")}.${holdout}`
        : `No version beat the starting score of ${share(result.baseline_score)} in ${plural(trials.length, "version")}.${holdout}`,
      series: [{name: state.loop.name, trials, best: result.best_trial_id}],
      snippet: apiSnippet("/api/run", body),
    });
  } catch (error) {
    if (token === state.token) notice("offline-message", error.message, true);
  }
}

async function compareOffline() {
  const token = ++state.token;
  let body;
  try {
    const {budget, seed, task} = offlineRequest();
    body = {loops: "all", budget, seed, ...(task ? {task} : {})};
  } catch (error) {
    notice("offline-message", error.message, true);
    return;
  }
  notice("offline-message");
  try {
    const result = await call("/api/compare", body);
    if (token !== state.token) return;
    const series = result.runs.map((run) => ({
      name: run.config.name,
      best: run.best_trial_id,
      trials: run.trials.map((trial) => ({
        id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score, text: artifactText(trial.candidate), note: trial.note,
      })),
      train: run.best_score,
      holdout: run.holdout_score,
    }));
    const best = result.summary.best;
    showOutcome({
      headline: Array.isArray(best)
        ? `${best.join(" and ")} tie on ${result.summary.ranked_by === "holdout_score" ? "holdout" : "training"} score at this budget and seed.`
        : `${best} scores highest on ${result.summary.ranked_by === "holdout_score" ? "holdout" : "training"} cases at this budget and seed.`,
      series,
      table: series.map((item) => ({name: item.name, cells: [`Training ${share(item.train)}`, `Holdout ${share(item.holdout)}`]})),
      snippet: apiSnippet("/api/compare", body),
    });
  } catch (error) {
    if (token === state.token) notice("offline-message", error.message, true);
  }
}

// Outcome: plot, verdicts, version inspector

function showOutcome(view) {
  state.shown = view;
  state.selected = {series: 0, id: view.series[0]?.best ?? 0};
  el("outcome").hidden = false;
  el("headline").textContent = view.headline;
  el("api-call").textContent = view.snippet || "";
  el("api-call").closest("details").hidden = !view.snippet;
  renderVerdicts();
  renderPlot();
  renderVersions();
}

function renderVerdicts() {
  const view = state.shown;
  const rows = [];
  for (const [index, row] of (view.table || []).entries()) {
    const button = node("button", "verdict");
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === state.selected.series));
    button.append(node("strong", "", row.name));
    if (row.tag) button.append(node("span", `tag tag-${row.tone || "plain"}`, row.tag));
    for (const cell of row.cells) button.append(node("span", "verdict-cell", cell));
    button.addEventListener("click", () => {
      state.selected = {series: index, id: view.series[index].best ?? 0};
      renderVerdicts();
      renderPlot();
      renderVersions();
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
  const view = state.shown;
  const plot = el("plot");
  const width = 720, height = 300, left = 46, right = 18, top = 16, bottom = 36;
  const count = Math.max(2, ...view.series.map((item) => item.trials.length));
  const x = (id) => left + (id / (count - 1)) * (width - left - right);
  const y = (value) => top + (1 - value) * (height - top - bottom);
  const parts = [svg("title", {id: "plot-title"}, "Score of each version tried, in the order it was tried")];
  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(svg("line", {class: "grid", x1: left, x2: width - right, y1: y(tick), y2: y(tick)}));
    parts.push(svg("text", {class: "axis", x: left - 8, y: y(tick) + 4, "text-anchor": "end"}, percent.format(tick)));
  }
  const step = Math.max(1, Math.ceil((count - 1) / 8));
  for (let id = 0; id < count; id += step) {
    parts.push(svg("text", {class: "axis", x: x(id), y: height - 14, "text-anchor": "middle"}, id === 0 ? "start" : String(id)));
  }

  const many = view.series.length > 1;
  view.series.forEach((series, index) => {
    const active = index === state.selected.series;
    const scored = series.trials.filter((trial) => Number.isFinite(trial.score));
    const byId = new Map(scored.map((trial) => [trial.id, trial]));
    const group = svg("g", {class: `series series-${index}${active ? " active" : ""}`});
    let best = -1;
    const path = scored.map((trial, position) => {
      best = Math.max(best, trial.score);
      return `${position ? "L" : "M"}${x(trial.id).toFixed(1)},${y(best).toFixed(1)}`;
    });
    if (scored.length) {
      const last = Math.max(count - 1, scored[scored.length - 1].id);
      path.push(`L${x(last).toFixed(1)},${y(best).toFixed(1)}`);
      group.append(svg("path", {class: "best-line", d: path.join(" ")}));
    }
    if (active || !many) {
      for (const trial of scored) {
        for (const parentId of trial.parents) {
          const parent = byId.get(parentId);
          if (parent) group.append(svg("line", {class: "edge", x1: x(parent.id), y1: y(parent.score), x2: x(trial.id), y2: y(trial.score)}));
        }
      }
      for (const trial of series.trials) {
        const isScored = Number.isFinite(trial.score);
        const chosen = state.selected.id === trial.id;
        const classes = ["dot", isScored ? "" : "skipped", trial.id === series.best ? "winner" : "", chosen ? "chosen" : ""].filter(Boolean).join(" ");
        const dot = svg("circle", {
          class: classes, cx: x(trial.id), cy: isScored ? y(trial.score) : height - bottom, r: trial.id === series.best ? 6.5 : 4.5,
          tabindex: "0", role: "button",
          "aria-label": `Version ${trial.id}, ${isScored ? share(trial.score) : "not scored"}`,
        });
        const pick = () => {
          state.selected = {series: index, id: trial.id};
          renderPlot();
          renderVersions();
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
  plot.replaceChildren(...parts);

  const legend = many
    ? view.series.map((series, index) => {
        const item = node("li", `key key-${index}`);
        item.append(node("span", "swatch"), document.createTextNode(series.name));
        return item;
      })
    : [["key key-line", "Best training score so far"], ["key key-dot", "A version, in the order tried"], ["key key-winner", "The winner"], ["key key-edge", "Built from"]]
        .map(([className, label]) => {
          const item = node("li", className);
          item.append(node("span", "swatch"), document.createTextNode(label));
          return item;
        });
  if (many) legend.push(node("li", "key", "Lines show the best training score so far"));
  el("legend").replaceChildren(...legend);
}

function renderVersions() {
  const series = state.shown.series[state.selected.series];
  if (!series) return;
  const items = series.trials.map((trial) => {
    const item = node("li");
    const button = node("button", "version");
    button.type = "button";
    button.setAttribute("aria-current", String(trial.id === state.selected.id));
    const origin = trial.id === 0 ? "starting version" : `from ${trial.parents.map((id) => (id === 0 ? "start" : id)).join(" and ")}`;
    button.append(
      node("span", "version-id", trial.id === 0 ? "Start" : `Version ${trial.id}`),
      node("span", "version-score", Number.isFinite(trial.score) ? share(trial.score) : trial.status === "duplicate" ? "repeat" : "unusable"),
      node("span", "version-origin", trial.id === series.best ? `${origin}, winner` : origin),
    );
    button.addEventListener("click", () => {
      state.selected = {series: state.selected.series, id: trial.id};
      renderPlot();
      renderVersions();
    });
    item.append(button);
    return item;
  });
  el("versions").replaceChildren(...items);
  const trial = series.trials.find((entry) => entry.id === state.selected.id) || series.trials[0];
  el("version-title").textContent = trial.id === 0 ? "Starting version" : `Version ${trial.id}`;
  const facts = [Number.isFinite(trial.score) ? `${share(trial.score)} on training cases` : "Not scored"];
  if (trial.note && trial.note !== "Starting candidate") facts.push(trial.note);
  el("version-meta").textContent = facts.join(". ") + ".";
  el("version-text").textContent = trial.text || "This attempt did not produce a usable prompt.";
}

// Loops that improve the visitor's prompt

function drivenRequest() {
  const parse = (id, label) => {
    let value;
    try {
      value = JSON.parse(el(id).value);
    } catch {
      throw new Error(`${label} are not valid JSON. Check for a missing comma or quote.`);
    }
    if (!Array.isArray(value) || value.some((row) => !row || typeof row.input !== "string" || typeof row.expected !== "string")) {
      throw new Error(`${label} must be a list of {"input": "…", "expected": "…"} objects.`);
    }
    return value.map(({input, expected}) => ({input, expected}));
  };
  const prompt = el("driven-prompt").value;
  if (!prompt.trim()) throw new Error("Enter the prompt you want to improve.");
  const holdout = parse("driven-holdout", "Holdout cases");
  const minimum = state.catalog.limits.driven.min_holdout_cases;
  if (holdout.length < minimum) throw new Error(`Add at least ${minimum} holdout cases. Fewer can't show a real gain.`);
  const body = {prompt, task: {train: parse("driven-train", "Training cases"), holdout}, max_trials: el("driven-trials").valueAsNumber};
  if (el("driven-all").checked) body.loops = "all";
  else body.loop = state.loop.id;
  return body;
}

function updateHandoff() {
  if (!state.loop || state.loop.mode !== "driven") return;
  let text;
  try {
    const body = drivenRequest();
    const which = body.loops ? "every loop (\"loops\": \"all\")" : `the ${state.loop.id} loop`;
    text = [
      `Go to ${location.origin}/llms.txt and follow it to improve my prompt with ${which}.`,
      `Use max_trials ${body.max_trials}. Answer the work items yourself, as the prompt being tested would.`,
      "When the run is done, tell me the verdict, the holdout scores before and after, the p-value, and the prompt you recommend, in full.",
      "", "Prompt:", body.prompt, "", "Task:", JSON.stringify(body.task),
    ].join("\n");
    el("copy-handoff").disabled = false;
  } catch (error) {
    text = error.message;
    el("copy-handoff").disabled = true;
  }
  el("handoff-text").textContent = text;
  notice("handoff-note");
}

function syncWho() {
  const keyed = document.querySelector("input[name=who]:checked").value === "key";
  el("who-assistant").hidden = keyed;
  el("who-key").hidden = !keyed;
}

async function askModel(key, model, item) {
  let response;
  try {
    response = await fetch(OPENROUTER, {
      method: "POST",
      headers: {Authorization: `Bearer ${key}`, "Content-Type": "application/json"},
      body: JSON.stringify({
        model, temperature: 0, max_tokens: item.kind === "propose" ? 2000 : 300,
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

async function answerAll(key, model, work, onProgress) {
  const outputs = [];
  let next = 0;
  const worker = async () => {
    while (next < work.length && !state.stop) {
      const item = work[next++];
      outputs.push({id: item.id, output: await askModel(key, model, item)});
      onProgress(outputs.length);
    }
  };
  await Promise.all(Array.from({length: Math.min(4, work.length)}, worker));
  return outputs;
}

function drivenOutcome(view) {
  const loops = state.catalog.loops;
  const name = (id) => loops.find((loop) => loop.id === id)?.name || id;
  const entries = Object.entries(view.results || {});
  const series = entries.map(([id, result]) => ({
    name: name(id),
    best: result.candidate_trial_id ?? 0,
    trials: result.trials.map((trial) => ({
      id: trial.id, parents: trial.parent_ids, status: trial.status, score: trial.score,
      text: artifactText(trial.artifact), note: trial.reason || "",
    })),
  }));
  const table = entries.map(([id, result]) => {
    const evidence = result.evidence;
    return {
      name: name(id), tag: VERDICTS[result.verdict], tone: result.verdict,
      cells: [
        `Holdout ${share(result.baseline_holdout)} to ${share(result.candidate_holdout)}`,
        `${evidence.improved_cases} fixed, ${evidence.regressed_cases} broken`,
        `p = ${evidence.p_value < 0.001 ? "under 0.001" : (Math.floor(evidence.p_value * 1000 + 0.5) / 1000).toFixed(3)}`,
        plural(result.model_calls, "model call"),
      ],
    };
  });
  for (const [id, reason] of Object.entries(view.failed || {})) {
    table.push({name: name(id), tag: "Stopped", tone: "not_improved", cells: [reason]});
    series.push({name: name(id), best: 0, trials: []});
  }
  const advice = view.recommendation;
  const adopt = advice && {
    confirmed: advice.action === "adopt",
    title: advice.action === "adopt" ? "Prompt to adopt" : advice.candidate_to_retest ? "Candidate worth retesting" : "Keep your prompt",
    text: artifactText(advice.action === "adopt" ? advice.artifact : advice.candidate_to_retest || advice.artifact),
  };
  showOutcome({headline: advice?.reason || "The run stopped before it produced a result.", series, table, adopt});
}

async function runDriven(event) {
  event.preventDefault();
  if (state.running) return;
  const key = el("or-key").value.trim();
  const model = el("or-model").value.trim();
  let body;
  try {
    body = drivenRequest();
    if (!key) throw new Error("Paste your OpenRouter key, or copy the message for your assistant instead.");
    if (!model) throw new Error("Enter an OpenRouter model, for example openai/gpt-4.1-mini.");
  } catch (error) {
    notice("driven-message", error.message, true);
    return;
  }
  const token = ++state.token;
  state.running = true;
  state.stop = false;
  el("run-driven").disabled = true;
  el("stop-driven").hidden = false;
  el("outcome").hidden = true;
  notice("driven-message", "Starting the run…");
  try {
    let view = await call("/api/runs", body);
    let answered = 0;
    while (view.state === "needs_work" && !state.stop && token === state.token) {
      const stage = STAGES[view.work[0].stage] || "Working";
      const base = answered;
      const outputs = await answerAll(key, model, view.work, (done) => {
        notice("driven-message", `${stage}. ${plural(base + done, "model call")} so far.`);
      });
      answered += outputs.length;
      if (!outputs.length) break;
      view = await call(view.submit.path, {outputs});
    }
    if (token !== state.token) return;
    if (view.state === "needs_work") {
      notice("driven-message", `Stopped after ${plural(answered, "model call")}. Run again to start over.`);
    } else {
      notice("driven-message", `Finished after ${plural(answered, "model call")}.`);
      drivenOutcome(view);
    }
  } catch (error) {
    if (token === state.token) notice("driven-message", error.message, true);
  } finally {
    state.running = false;
    el("run-driven").disabled = false;
    el("stop-driven").hidden = true;
  }
}

// Page wiring

function fillExamples() {
  const task = state.catalog.example_task;
  el("offline-task").value = pretty(task);
  const example = state.catalog.example_prompt_run;
  el("driven-prompt").value = example.prompt;
  el("driven-train").value = pretty(example.task.train);
  el("driven-holdout").value = pretty(example.task.holdout);
}

async function start() {
  el("mcp-url").textContent = `${location.origin}/mcp`;
  el("guide-url").textContent = `${location.origin}/llms.txt`;
  try {
    state.catalog = await call("/api/catalog");
  } catch (error) {
    el("loop-name").textContent = "The loops could not be loaded";
    el("loop-summary").textContent = `${error.message} Reload the page to try again.`;
    return;
  }
  fillExamples();
  renderShelf();
  selectLoop(new URLSearchParams(location.search).get("loop") || state.catalog.loops[0].id);
}

el("offline-form").addEventListener("submit", runOffline);
el("compare-offline").addEventListener("click", compareOffline);
el("reset-task").addEventListener("click", () => {
  el("offline-task").value = pretty(state.catalog.example_task);
  void runOffline();
});
el("driven-form").addEventListener("submit", runDriven);
el("stop-driven").addEventListener("click", () => {
  state.stop = true;
});
for (const id of ["driven-prompt", "driven-train", "driven-holdout", "driven-trials"]) el(id).addEventListener("input", updateHandoff);
el("driven-all").addEventListener("change", updateHandoff);
for (const radio of document.querySelectorAll("input[name=who]")) radio.addEventListener("change", syncWho);
el("copy-handoff").addEventListener("click", () => copy(el("handoff-text").textContent, "handoff-note", "Copied. Paste it into your assistant."));
el("copy-mcp").addEventListener("click", () => copy(el("mcp-url").textContent, "ask-note", "Connector address copied."));
for (const button of document.querySelectorAll(".ask")) {
  button.addEventListener("click", () => copy(button.dataset.ask.replace("{origin}", location.origin), "ask-note", "Message copied. Paste it into your assistant."));
}

void start();
