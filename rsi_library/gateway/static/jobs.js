"use strict";

const el = (id) => document.getElementById(id);
const json = (value) => JSON.stringify(value, null, 2);
const activeStates = new Set(["queued", "running", "cancel_requested"]);
const stateLabels = {
  queued: "Queued", running: "Running", cancel_requested: "Stopping",
  cancelled: "Cancelled", succeeded: "Succeeded", failed: "Failed",
};
const stageLabels = {
  baseline_train: "Scoring the original on training cases",
  propose: "Proposing a candidate",
  candidate_train: "Scoring a candidate on training cases",
  baseline_holdout: "Checking the original on holdout",
  candidate_holdout: "Checking the candidate on holdout",
};
const verdictLabels = {improved: "significant gain", inconclusive: "inconclusive", not_improved: "no gain"};
const sample = {
  prompt: "Classify the incoming support message. Return exactly one label.",
  steps: [
    {name: "summarize", text: "Summarize the customer’s request in one sentence."},
    {name: "route", text: "Classify the request. Return exactly one label."},
  ],
  train: [
    {input: "I was charged twice for my subscription.", expected: "billing"},
    {input: "Where can I download last month’s invoice?", expected: "billing"},
    {input: "My refund still hasn’t arrived.", expected: "billing"},
    {input: "The link to reset my password expired.", expected: "account"},
    {input: "I can’t edit the email address in my profile.", expected: "account"},
    {input: "Two-factor codes never reach my phone.", expected: "account"},
  ],
  holdout: [
    {input: "How can I check the status of my refund?", expected: "billing"},
    {input: "Please send a corrected invoice with our tax ID.", expected: "billing"},
    {input: "Why is there an extra charge on my card?", expected: "billing"},
    {input: "Can billing switch us to annual payments?", expected: "billing"},
    {input: "The invoice total doesn’t match the quote.", expected: "billing"},
    {input: "The dashboard keeps looping back to login.", expected: "account"},
    {input: "How do I add a teammate to our workspace?", expected: "account"},
    {input: "My new password isn’t accepted.", expected: "account"},
  ],
};
let policy = {min_holdout_cases: 8, max_holdout_repeats: 5, alpha: 0.05};
const state = {
  session: null, connecting: false, jobs: [], selectedId: null, selected: null, selectedFull: false,
  listSignature: "", selectRevision: 0, submitting: false, cancelling: false,
  pending: new Map(), requests: [], operatorBusy: false, requesting: false,
};
const currency = new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 6,
});
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 1});
const time = new Intl.DateTimeFormat(undefined, {dateStyle: "medium", timeStyle: "short"});

class GatewayError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function notice(id, message = "", error = false) {
  const target = el(id);
  target.textContent = message;
  target.classList.toggle("error", error);
}

function dollars(microdollars) {
  return Number.isFinite(microdollars) ? currency.format(microdollars / 1_000_000) : "—";
}

function score(value) {
  return Number.isFinite(value) ? percent.format(value) : "—";
}

function pValue(value) {
  if (!Number.isFinite(value)) return "—";
  return value < 0.001 ? "< 0.001" : value.toFixed(3);
}

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function timestamp(value) {
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  return Number.isNaN(date.valueOf()) ? "" : time.format(date);
}

async function api(key, path, {method = "GET", body, idempotencyKey} = {}) {
  const headers = {Authorization: `Bearer ${key}`, Accept: "application/json"};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(`/v1${path}`, {
      method, headers, body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal,
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new GatewayError(response.status, "The gateway returned an unreadable response.");
    }
    if (!response.ok) {
      throw new GatewayError(response.status, typeof data?.error === "string" ? data.error : "The request failed.");
    }
    return data;
  } catch (error) {
    if (error instanceof GatewayError) throw error;
    if (error.name === "AbortError") throw new GatewayError(0, "The gateway took too long to respond.");
    throw new GatewayError(0, "Could not reach the gateway.");
  } finally {
    clearTimeout(timer);
  }
}

function setOptions(select, options, previous) {
  select.replaceChildren(...options.map(({id, name}) => {
    const option = node("option", "", name);
    option.value = id;
    return option;
  }));
  if (options.some(({id}) => id === previous)) select.value = previous;
}

function applyPolicy(value) {
  if (!value || typeof value !== "object") return;
  policy = {...policy, ...value};
  for (const target of document.querySelectorAll("[data-policy]")) {
    const entry = policy[target.dataset.policy];
    if (entry !== undefined) target.textContent = String(entry);
  }
  updateFixtureSummary();
}

function syncArtifact() {
  const chain = el("artifact-type").value === "prompt-chain";
  el("prompt-field").hidden = chain;
  el("chain-field").hidden = !chain;
  el("prompt-text").required = !chain;
  el("chain-steps").required = chain;
  const methods = (state.session?.catalog.methods || []).filter(
    (method) => method.artifact_types.includes(el("artifact-type").value),
  );
  setOptions(el("method"), [{id: "auto", name: "Auto route"}, ...methods], el("method").value);
  syncMethod();
}

function syncMethod() {
  const method = el("method").value;
  const entry = state.session?.catalog.methods.find(({id}) => id === method);
  el("improver-field").hidden = method !== "self-revise-v1";
  el("method-description").textContent = entry?.description || (
    state.session ? "The gateway selects a compatible optimizer and records the choice on the job." : ""
  );
  updateFixtureSummary();
}

function parseList(id, label) {
  let value;
  try {
    value = JSON.parse(el(id).value);
  } catch {
    throw new Error(`${label} must be valid JSON.`);
  }
  if (!Array.isArray(value) || !value.length) throw new Error(`${label} must be a nonempty JSON array.`);
  return value;
}

function fixtures(id, label) {
  return parseList(id, label).map((item) => {
    if (!item || typeof item.input !== "string" || typeof item.expected !== "string") {
      throw new Error(`${label} must contain objects with input and expected text.`);
    }
    return {input: item.input, expected: item.expected};
  });
}

function updateFixtureSummary() {
  const model = state.session?.catalog.models.find(({id}) => id === el("model").value);
  let minimum = "";
  try {
    const train = parseList("train", "Training cases").length;
    const holdout = parseList("holdout", "Holdout cases").length;
    const repeats = Number(el("holdout-repeats").value) || 1;
    const short = holdout < policy.min_holdout_cases;
    el("fixture-counts").textContent = `${train} training, ${holdout} holdout${short ? `, needs ${policy.min_holdout_cases} or more` : ""}`;
    el("fixture-counts").classList.toggle("warning", short);
    const steps = el("artifact-type").value === "prompt-chain" ? parseList("chain-steps", "Starting steps").length : 1;
    const minimumCalls = 2 * train * steps + 1 + 2 * holdout * steps * repeats;
    minimum = `Allow at least ${minimumCalls} model calls for one trial and the holdout check.`;
  } catch {
    el("fixture-counts").textContent = "Check fixture JSON";
    el("fixture-counts").classList.remove("warning");
  }
  if (Number.isFinite(model?.max_call_cost_microusd)) {
    minimum += ` Up to ${dollars(model.max_call_cost_microusd)} can be reserved per call.`;
  }
  el("budget-hint").textContent = minimum.trim() || "The model call limit covers proposals and scoring.";
}

function restoreExample() {
  el("prompt-text").value = sample.prompt;
  el("chain-steps").value = json(sample.steps);
  el("improver-text").value = "";
  el("train").value = json(sample.train);
  el("holdout").value = json(sample.holdout);
  updateFixtureSummary();
}

function syncAccess() {
  const current = state.session;
  const canWrite = current?.catalog.role === "write";
  const hasModels = Boolean(current?.catalog.models.length);
  const supported = current?.catalog.methods.some((method) => method.artifact_types.includes(el("artifact-type").value));
  el("connect-form").hidden = Boolean(current);
  el("workspace-tag").hidden = !current;
  el("workspace-id").textContent = current?.catalog.workspace_id || "";
  el("workspace-role").textContent = current ? (canWrite ? "Can submit and cancel jobs" : "Read-only access") : "";
  el("disconnect").hidden = !current;
  el("refresh-jobs").disabled = !current;
  el("connection-status").textContent = current ? "Connected" : "Disconnected";
  el("connection-status").classList.toggle("connected", Boolean(current));
  el("operator").hidden = !(current?.catalog.operator && canWrite);
  el("request-key").hidden = Boolean(current);
  el("job-fields").disabled = !canWrite || !hasModels || state.submitting;
  el("submit-job").disabled = !canWrite || !hasModels || !supported || state.submitting;
  if (state.submitting) el("submit-hint").textContent = "Submitting your job…";
  else if (!current) el("submit-hint").textContent = "Connect with a workspace key to start a job.";
  else if (!canWrite) el("submit-hint").textContent = "This key can view jobs. Use a write key to start one.";
  else if (!hasModels) el("submit-hint").textContent = "No models are configured for this gateway.";
  else if (!supported) el("submit-hint").textContent = "No optimizer supports this artifact type.";
  else el("submit-hint").textContent = "You can follow the run in your job history.";
}

function disconnect(message = "", error = false) {
  state.session = null;
  state.jobs = [];
  state.selectedId = null;
  state.selected = null;
  state.selectedFull = false;
  state.selectRevision += 1;
  state.submitting = false;
  state.cancelling = false;
  state.pending.clear();
  state.requests = [];
  el("api-key").value = "";
  el("job-form").reset();
  el("model").replaceChildren(node("option", "", "Connect to load models"));
  el("issued").hidden = true;
  el("issued-token").value = "";
  el("requests").replaceChildren();
  el("site-usage").textContent = "";
  restoreExample();
  syncArtifact();
  syncAccess();
  renderList();
  renderDetail();
  notice("job-message");
  notice("detail-message");
  notice("operator-message");
  notice("session-message", message, error);
}

function handleFailure(current, location, error) {
  if (state.session !== current) return;
  if (error.status === 401) {
    disconnect("Your workspace key is no longer accepted. Connect with an active key.", true);
    el("api-key").focus();
    return;
  }
  notice(location, error.message, true);
}

async function connect(event) {
  event.preventDefault();
  if (state.connecting || state.session) return;
  const key = el("api-key").value.trim();
  if (!key) {
    notice("session-message", "Enter a workspace API key.", true);
    return;
  }
  state.connecting = true;
  el("connect").disabled = true;
  el("api-key").value = "";
  el("api-key").disabled = true;
  notice("session-message", "Connecting…");
  try {
    const catalog = await api(key, "/catalog");
    if (!Array.isArray(catalog.methods) || !Array.isArray(catalog.models)) {
      throw new Error("The gateway catalog is incomplete.");
    }
    const current = {key, catalog, listing: false, reading: false, lastList: 0};
    state.session = current;
    applyPolicy(catalog.evidence_policy);
    setOptions(el("model"), [{id: "", name: catalog.models.length ? "Auto select model" : "No models configured"}, ...catalog.models]);
    syncArtifact();
    syncAccess();
    notice("session-message", "Connected to your workspace.");
    el(catalog.role === "write" && catalog.models.length ? "artifact-type" : "refresh-jobs").focus();
    await Promise.all([loadList(current), loadOperator(current)]);
  } catch (error) {
    notice("session-message", error.message, true);
  } finally {
    state.connecting = false;
    el("connect").disabled = false;
    el("api-key").disabled = false;
    if (!state.session) el("api-key").focus();
  }
}

function chip(status) {
  const value = node("span", "state-chip", stateLabels[status] || "Unknown");
  if (Object.hasOwn(stateLabels, status)) value.classList.add(`state-${status}`);
  return value;
}

function verdictOf(job) {
  // Job lists carry a summary verdict; a fully loaded job carries its evidence.
  return job.result?.verdict || job.result?.evidence?.verdict;
}

function renderList() {
  const signature = JSON.stringify([
    state.session?.catalog.workspace_id, state.selectedId,
    state.jobs.map((job) => [job.id, job.state, job.created_at, job.result?.method, verdictOf(job), job.request?._routing?.method, job.request?.method]),
  ]);
  if (signature === state.listSignature) return;
  state.listSignature = signature;
  const focused = document.activeElement?.closest(".job-row")?.dataset.id;
  const rows = state.jobs.map((job) => {
    const row = node("li");
    const button = node("button", "job-row");
    button.type = "button";
    button.dataset.id = job.id;
    button.setAttribute("aria-current", job.id === state.selectedId ? "true" : "false");
    const method = job.result?.method || job.request?._routing?.method || job.request?.method;
    const label = method === "auto" ? "Auto route" : method || "Improvement job";
    const verdict = verdictLabels[verdictOf(job)];
    button.append(
      node("span", "job-title", label),
      chip(job.state),
      node("span", "time", [timestamp(job.created_at), verdict].filter(Boolean).join(" · ")),
    );
    button.addEventListener("click", () => selectJob(job));
    row.append(button);
    return row;
  });
  el("jobs").replaceChildren(...rows);
  if (focused) rows.find((row) => row.firstChild.dataset.id === focused)?.firstChild.focus();
  el("history-empty").hidden = rows.length > 0;
  el("history-empty").textContent = state.session ? "No jobs in this workspace yet." : "Connect to view jobs saved in your workspace.";
}

function metric(label, value, theme = "") {
  const item = node("div", theme ? `metric ${theme}` : "metric");
  item.append(node("dt", "", label), node("dd", "", value));
  return item;
}

function decisionText(result) {
  const evidence = result.evidence;
  if (!evidence) {
    return result.accepted
      ? "The candidate scored higher on holdout and is ready to review."
      : "The original artifact was kept after the holdout check.";
  }
  const moved = `fixed ${plural(evidence.improved_cases, "holdout case")} and broke ${evidence.regressed_cases}`;
  const test = `p = ${pValue(evidence.p_value)}`;
  if (evidence.verdict === "improved") {
    return `The candidate ${moved} (${test}). The gain is statistically significant, so the candidate is ready to review.`;
  }
  if (evidence.verdict === "inconclusive") {
    return `The candidate scored higher but only ${moved} (${test}), which could be chance. The original was kept; the candidate is below if you want to test it on more cases.`;
  }
  if (evidence.note) return `${evidence.note} The original was kept.`;
  return `The candidate did not beat the original on holdout: it ${moved}. The original was kept.`;
}

function renderDetail() {
  const job = state.selected;
  el("detail-content").hidden = !job;
  el("detail-empty").hidden = Boolean(job);
  if (!job) {
    for (const id of ["job-id", "job-method", "run-error", "run-progress", "decision", "holdout-size", "artifact", "candidate", "trial-data", "routing-data"]) {
      el(id).textContent = "";
    }
    el("usage").replaceChildren();
    el("scores").replaceChildren();
    return;
  }
  el("job-id").textContent = job.id;
  const status = chip(job.state);
  el("job-state").className = status.className;
  el("job-state").textContent = status.textContent;
  const result = job.result;
  const route = result?.routing || job.request?._routing;
  const method = result?.method || route?.method || job.request?.method || "Pending";
  const model = result?.model || route?.model || job.request?.model;
  el("job-method").textContent = [method === "auto" ? "Auto route" : method, model].filter(Boolean).join(" · ");
  const usage = job.usage || {};
  el("usage").replaceChildren(
    metric("Model calls", String(usage.calls ?? 0)),
    metric("Recorded spend", dollars(usage.spent_microusd ?? 0)),
    metric("In progress", dollars(usage.reserved_microusd ?? 0)),
    metric("Uncertain spend", dollars(usage.uncertain_microusd ?? 0), usage.uncertain_microusd ? "uncertain" : ""),
  );
  el("uncertain-note").hidden = !usage.uncertain_microusd;
  el("run-error").hidden = !job.error;
  el("run-error").textContent = typeof job.error === "string" ? job.error : job.error?.message || "";
  const progress = typeof job.progress === "string" ? job.progress
    : job.progress?.message || stageLabels[job.progress?.stage];
  el("run-progress").hidden = !progress || !activeStates.has(job.state);
  el("run-progress").textContent = progress || "";
  el("cancel-job").hidden = state.session?.catalog.role !== "write" || !["queued", "running"].includes(job.state);
  el("cancel-job").disabled = state.cancelling;
  el("result").hidden = !result;
  el("download-artifact").hidden = result?.artifact == null;
  el("download-run").hidden = !state.selectedFull;
  if (!result) {
    el("artifact").textContent = "";
    el("candidate").textContent = "";
    el("trial-data").textContent = "";
    el("routing-data").textContent = "";
    return;
  }
  const evidence = result.evidence;
  el("decision").textContent = decisionText(result);
  el("decision").classList.toggle("accepted", result.accepted === true);
  el("decision").classList.toggle("inconclusive", evidence?.verdict === "inconclusive");
  const scores = [
    metric("Original holdout", score(result.baseline_holdout)),
    metric("Candidate holdout", score(result.candidate_holdout), result.accepted ? "positive" : ""),
  ];
  if (evidence) {
    scores.push(
      metric("Cases fixed / broken", `${evidence.improved_cases} / ${evidence.regressed_cases}`),
      metric("Paired test", `p = ${pValue(evidence.p_value)}`, result.accepted ? "positive" : ""),
    );
  }
  el("scores").replaceChildren(...scores);
  const count = result.holdout_cases;
  el("holdout-size").hidden = !Number.isInteger(count);
  const repeats = evidence?.repeats || result.holdout_repeats || 1;
  el("holdout-size").textContent = !Number.isInteger(count) ? "" : [
    `Scored on ${plural(count, "holdout case")}${repeats > 1 ? `, ${repeats} tries each` : ""}.`,
    evidence ? `A significant gain needs p ≤ ${evidence.alpha} on a one-sided sign test of the paired cases.` : "",
  ].filter(Boolean).join(" ");
  el("artifact-title").textContent = result.accepted ? "Improved artifact" : "Original artifact";
  el("artifact-output").hidden = result.artifact == null;
  const display = (artifact) => artifact?.type === "prompt" && !artifact?.improver ? artifact.text : json(artifact);
  el("artifact").textContent = result.artifact == null ? "" : display(result.artifact);
  const unadopted = !result.accepted && result.candidate != null && json(result.candidate) !== json(result.artifact);
  el("candidate-output").hidden = !unadopted;
  el("candidate").textContent = unadopted ? display(result.candidate) : "";
  el("trials").hidden = !result.trials?.length;
  el("trial-data").textContent = result.trials?.length ? json(result.trials) : "";
  el("routing").hidden = result.routing == null;
  el("routing-data").textContent = result.routing == null ? "" : json(result.routing);
}

function selectJob(job, fetchDetail = true) {
  state.selectedId = job.id;
  state.selectRevision += 1;
  state.selected = job;
  state.selectedFull = !fetchDetail;
  notice("detail-message");
  renderList();
  renderDetail();
  if (fetchDetail) void loadDetail(state.session, state.selectRevision);
}

function updateJob(job) {
  if (!job || typeof job.id !== "string") return;
  const index = state.jobs.findIndex((item) => item.id === job.id);
  if (index === -1) state.jobs.unshift(job);
  else state.jobs[index] = job;
  renderList();
  if (job.id === state.selectedId && (!state.selectedFull || json(job) !== json(state.selected))) {
    state.selected = job;
    state.selectedFull = true;
    renderDetail();
  }
}

async function loadList(current, notify = false) {
  if (!current || current.listing) return;
  current.listing = true;
  current.lastList = Date.now();
  try {
    const data = await api(current.key, "/jobs");
    if (state.session !== current) return;
    if (!Array.isArray(data.jobs)) throw new Error("The gateway job history is incomplete.");
    state.jobs = data.jobs;
    if (!state.selectedId && data.jobs.length) selectJob(data.jobs[0]);
    else renderList();
    if (notify) notice("session-message", "Job history refreshed.");
  } catch (error) {
    handleFailure(current, "session-message", error);
  } finally {
    current.listing = false;
  }
}

async function loadDetail(current, revision) {
  if (!current || !state.selectedId || current.reading === revision) return;
  current.reading = revision;
  const id = state.selectedId;
  try {
    const job = await api(current.key, `/jobs/${encodeURIComponent(id)}`);
    if (state.session === current && state.selectRevision === revision) updateJob(job);
  } catch (error) {
    if (state.selectRevision === revision) handleFailure(current, "detail-message", error);
  } finally {
    if (current.reading === revision) current.reading = null;
  }
}

function microusd(raw) {
  const match = /^(\d{1,9})(?:\.(\d{1,6}))?$/.exec(raw.trim());
  if (!match) throw new Error("Enter a dollar amount with up to six decimal places, such as 0.25.");
  const amount = Number(match[1]) * 1_000_000 + Number((match[2] || "").padEnd(6, "0"));
  if (amount > 1_000_000_000) throw new Error("Set the amount to $1,000 or less.");
  return amount;
}

function jobRequest() {
  const type = el("artifact-type").value;
  const method = el("method").value;
  let artifact;
  if (type === "prompt-chain") {
    const steps = parseList("chain-steps", "Starting steps");
    if (steps.length < 2 || steps.length > 4 || steps.some(
      (step) => !step || typeof step.name !== "string" || typeof step.text !== "string",
    )) {
      throw new Error("Starting steps must contain two to four objects with name and text.");
    }
    artifact = {type, steps: steps.map(({name, text}) => ({name, text}))};
  } else {
    const text = el("prompt-text").value;
    if (!text.trim()) throw new Error("Enter a starting prompt.");
    artifact = {type, text};
    const improver = el("improver-text").value;
    if (method === "self-revise-v1" && improver.trim()) artifact.improver = improver;
  }
  const holdout = fixtures("holdout", "Holdout cases");
  if (holdout.length < policy.min_holdout_cases) {
    throw new Error(`Add at least ${policy.min_holdout_cases} holdout cases. Fewer can't show a statistically significant gain.`);
  }
  const model = el("model").value;
  return {
    method, artifact, ...(model ? {model} : {}),
    objective: el("objective").value,
    data_policy: el("data-policy").value,
    holdout_repeats: Number(el("holdout-repeats").value) || 1,
    task: {train: fixtures("train", "Training cases"), holdout},
    budget: {
      max_calls: el("max-calls").valueAsNumber,
      max_cost_microusd: microusd(el("max-cost").value),
      max_seconds: el("max-seconds").valueAsNumber,
    },
    max_trials: el("max-trials").valueAsNumber,
  };
}

function newKey() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function submit(event) {
  event.preventDefault();
  const current = state.session;
  if (!current || state.submitting || el("submit-job").disabled) return;
  let body;
  try {
    body = jobRequest();
  } catch (error) {
    notice("job-message", error.message, true);
    return;
  }
  const fingerprint = JSON.stringify(body);
  const idempotencyKey = state.pending.get(fingerprint) || newKey();
  state.pending.set(fingerprint, idempotencyKey);
  state.submitting = true;
  syncAccess();
  notice("job-message");
  try {
    const job = await api(current.key, "/jobs", {method: "POST", body, idempotencyKey});
    if (state.session !== current) return;
    state.pending.delete(fingerprint);
    updateJob(job);
    selectJob(job, false);
    notice("job-message", "Job submitted. You can follow its progress in Job details.");
  } catch (error) {
    if (state.session !== current) return;
    if (error.status >= 400 && error.status < 500) state.pending.delete(fingerprint);
    handleFailure(current, "job-message", error);
    if (state.session === current && (error.status === 0 || error.status >= 500)) {
      notice("job-message", `${error.message} Submit the same configuration to retry without creating a duplicate job.`, true);
    }
  } finally {
    if (state.session === current) {
      state.submitting = false;
      syncAccess();
    }
  }
}

async function cancel() {
  const current = state.session;
  const id = state.selectedId;
  if (!current || !id || state.cancelling || el("cancel-job").hidden) return;
  state.cancelling = true;
  renderDetail();
  notice("detail-message", "Requesting cancellation…");
  try {
    const job = await api(current.key, `/jobs/${encodeURIComponent(id)}/cancel`, {method: "POST"});
    if (state.session !== current) return;
    updateJob(job);
    if (state.selectedId === id) notice("detail-message", "Cancellation request recorded.");
  } catch (error) {
    if (state.selectedId === id) handleFailure(current, "detail-message", error);
  } finally {
    if (state.session === current) {
      state.cancelling = false;
      renderDetail();
    }
  }
}

function download(suffix, value) {
  if (value == null || !state.selected) return;
  const id = state.selected.id.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 60);
  const url = URL.createObjectURL(new Blob([json(value)], {type: "application/json"}));
  const link = node("a");
  link.href = url;
  link.download = `rsi-${id}-${suffix}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Operator: review access requests and watch site-wide spend.

function renderSiteUsage(usage) {
  const parts = [`Site model spend: ${dollars(usage.spent_microusd)} settled`];
  if (usage.reserved_microusd) parts.push(`${dollars(usage.reserved_microusd)} in progress`);
  if (usage.uncertain_microusd) parts.push(`${dollars(usage.uncertain_microusd)} uncertain`);
  const limit = usage.site_spend_limit_microusd;
  const cap = Number.isFinite(limit) ? ` of a ${dollars(limit)} site-wide cap` : ", with no site-wide cap set";
  el("site-usage").textContent = `${parts.join(", ")}${cap}, across ${plural(usage.calls, "model call")}.`;
}

function renderRequests() {
  const rows = state.requests.map((request) => {
    const item = node("li", "request-row");
    const head = node("div", "request-head");
    head.append(node("strong", "", request.email), node("span", "hint", `${request.name} · ${timestamp(request.created_at)}`));
    const actions = node("div", "request-actions");
    const label = node("label", "", "Allowance (USD)");
    const input = node("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.value = "1.00";
    label.append(input);
    const approve = node("button", "primary", "Approve");
    approve.type = "button";
    approve.addEventListener("click", () => decide(request, "approve", input.value));
    const decline = node("button", "quiet danger", "Decline");
    decline.type = "button";
    decline.addEventListener("click", () => decide(request, "decline"));
    actions.append(label, approve, decline);
    item.append(head, node("p", "request-use", request.use_case), actions);
    return item;
  });
  el("requests").replaceChildren(...rows);
  el("requests-empty").hidden = rows.length > 0;
}

async function loadOperator(current, notify = false) {
  if (!current?.catalog.operator || current.catalog.role !== "write") return;
  try {
    const [requests, usage] = await Promise.all([
      api(current.key, "/operator/access-requests"),
      api(current.key, "/operator/usage"),
    ]);
    if (state.session !== current) return;
    state.requests = Array.isArray(requests.requests) ? requests.requests : [];
    renderRequests();
    renderSiteUsage(usage.usage);
    if (notify) notice("operator-message", "Requests refreshed.");
  } catch (error) {
    handleFailure(current, "operator-message", error);
  }
}

async function decide(request, action, allowance) {
  const current = state.session;
  if (!current || state.operatorBusy) return;
  let body = {};
  if (action === "approve") {
    try {
      body = {spend_limit_microusd: microusd(allowance)};
    } catch (error) {
      notice("operator-message", error.message, true);
      return;
    }
  }
  state.operatorBusy = true;
  notice("operator-message", `${action === "approve" ? "Approving" : "Declining"} ${request.email}…`);
  try {
    const result = await api(current.key, `/operator/access-requests/${encodeURIComponent(request.id)}/${action}`, {method: "POST", body});
    if (state.session !== current) return;
    if (action === "approve") {
      el("issued").hidden = false;
      el("issued-for").textContent = `For ${request.email}, with a ${dollars(result.spend_limit_microusd)} allowance in workspace ${result.workspace_id}.`;
      el("issued-token").value = result.key.token;
      el("issued-token").focus();
      el("issued-token").select();
    }
    notice("operator-message", `${action === "approve" ? "Approved" : "Declined"} ${request.email}.`);
    await loadOperator(current);
  } catch (error) {
    handleFailure(current, "operator-message", error);
  } finally {
    state.operatorBusy = false;
  }
}

async function copyToken() {
  const value = el("issued-token").value;
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    notice("operator-message", "Key copied.");
  } catch {
    el("issued-token").select();
    notice("operator-message", "Select the key and copy it.");
  }
}

// Public: describe the gateway and accept requests for keys.

async function loadInfo() {
  try {
    const response = await fetch("/public/info", {cache: "no-store", credentials: "omit", redirect: "error"});
    if (!response.ok) return;
    const info = await response.json();
    el("mode-banner").hidden = info.mode !== "demo";
    const open = info.access_requests === true;
    el("access-form").hidden = !open;
    el("requests-closed").hidden = open;
    el("hero-request").hidden = !open;
    applyPolicy(info.evidence_policy);
  } catch {
    // The workbench works without this description.
  }
}

async function requestAccess(event) {
  event.preventDefault();
  if (state.requesting) return;
  const email = el("access-email").value.trim();
  const body = {email, use_case: el("access-use").value.trim()};
  const name = el("access-name").value.trim();
  if (name) body.name = name;
  if (el("access-website").value) body.website = el("access-website").value;
  if (body.use_case.length < 10) {
    notice("access-message", "Tell us a little more about what you want to improve.", true);
    return;
  }
  state.requesting = true;
  el("access-submit").disabled = true;
  notice("access-message", "Sending your request…");
  try {
    const response = await fetch("/access-requests", {
      method: "POST", headers: {"Content-Type": "application/json", Accept: "application/json"},
      body: JSON.stringify(body), cache: "no-store", credentials: "omit", redirect: "error",
    });
    let data = {};
    try {
      data = await response.json();
    } catch {
      // Status alone is enough below.
    }
    if (response.status === 202) {
      el("access-form").reset();
      notice("access-message", `Thanks. We’ll reply to ${email} about access.`);
    } else if (response.status === 429) {
      notice("access-message", "Too many requests from this network. Please try again later.", true);
    } else if (response.status === 404) {
      notice("access-message", "Access requests are closed right now.", true);
    } else {
      notice("access-message", typeof data.error === "string" ? data.error : "The request could not be sent.", true);
    }
  } catch {
    notice("access-message", "Could not reach the gateway. Please try again.", true);
  } finally {
    state.requesting = false;
    el("access-submit").disabled = false;
  }
}

el("connect-form").addEventListener("submit", connect);
el("disconnect").addEventListener("click", () => {
  disconnect("Disconnected from your workspace.");
  el("api-key").focus();
});
el("artifact-type").addEventListener("change", () => {
  syncArtifact();
  syncAccess();
});
el("method").addEventListener("change", syncMethod);
el("model").addEventListener("change", updateFixtureSummary);
el("holdout-repeats").addEventListener("change", updateFixtureSummary);
for (const id of ["train", "holdout", "chain-steps"]) el(id).addEventListener("input", updateFixtureSummary);
el("load-example").addEventListener("click", restoreExample);
el("job-form").addEventListener("submit", submit);
el("refresh-jobs").addEventListener("click", () => {
  void loadList(state.session, true);
  void loadDetail(state.session, state.selectRevision);
});
el("cancel-job").addEventListener("click", cancel);
el("download-artifact").addEventListener("click", () => download("artifact", state.selected?.result?.artifact));
el("download-run").addEventListener("click", () => download("run", state.selected));
el("refresh-requests").addEventListener("click", () => void loadOperator(state.session, true));
el("copy-token").addEventListener("click", copyToken);
el("access-form").addEventListener("submit", requestAccess);
setInterval(() => {
  const current = state.session;
  if (!current || document.hidden) return;
  if (Date.now() - current.lastList >= 6000) void loadList(current);
  if (activeStates.has(state.selected?.state)) void loadDetail(current, state.selectRevision);
}, 2000);

restoreExample();
syncArtifact();
syncAccess();
void loadInfo();
