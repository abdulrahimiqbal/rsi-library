"""Public loops that need no workspace key.

Two kinds of loop are served here:

* Offline loops run entirely on this server against a small built-in interpreter.
  They show how a loop behaves and cost nothing.
* Driven loops improve a real prompt, with the caller acting as the model. The server
  runs the loop, hands out work items ("answer this prompt for this input"), scores
  the answers, keeps the holdout labels to itself, and decides with the same paired
  test as a hosted job.

A driven run stores only the caller's request and the answers supplied so far. Each
step replays the loop from the start with those answers, which is cheap and exact
because the loop is deterministic for a given seed and set of answers.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from .. import engine
from ..builtins import TEMPLATES, catalog as offline_catalog, make_registry
from ..engine import ConfigError
from . import methods
from .methods import MethodError
from .methods_gallery import FAMILIES, SEARCH_STYLES, gallery

DRIVEN_IDS = tuple(methods._METHODS)
MAX_DRIVEN_TRIALS = 10
MAX_DRIVEN_REPEATS = 3
MAX_DRIVEN_CALLS = 1000
MAX_OUTPUT_CHARS = 16_000
BATCH = 40
MAX_WORK = 64
RETENTION_SECONDS = 14 * 86_400
MAX_STORED_RUNS = 5000
_PENDING = "[[rsi-library-pending-output]]"

_BEST_FOR = {
    "greedy-prompt": "The simplest loop. Start here to see propose, score, keep.",
    "archive-prompt": "Seeing how branching from older versions escapes a dead end.",
    "adaptive-improver": "Watching a loop change how it edits, not only what it edits.",
    "reflect-v1": "A prompt that makes the same kind of mistake repeatedly.",
    "evolve-v1": "Tasks where different fixes help different cases.",
    "self-revise-v1": "Longer runs, where better editing instructions pay off.",
}

_SEARCH_TEXT = {
    "reflect-v1": ("Best first", "Build on the highest scoring version, the earliest on ties."),
    "evolve-v1": ("Complementary parents", "Pair the best version with the one that fixes most of its training mistakes."),
    "self-revise-v1": ("Newest best", "Build on the best version, preferring the newest so fresh editing instructions get used."),
}

WORK_INSTRUCTIONS = (
    "You are the model for this run. For each work item, produce the reply a chat model would give "
    "when `system` is its system prompt and `input` is the user message. Reply with the answer only, "
    "with no commentary. For kind 'score', follow the prompt in `system` exactly; it is the version "
    "being tested, so do not improve on it or correct it. For kind 'propose', return the JSON object "
    "that `system` asks for. Answer every item, then submit them together."
)


EXAMPLE_PROMPT_RUN = {
    "prompt": "Classify the support message as billing or account. Return only the label.",
    "task": {
        "train": [
            {"input": "Could you resend my latest invoice?", "expected": "billing"},
            {"input": "I asked for a refund last Thursday.", "expected": "billing"},
            {"input": "My plan renewed but I had cancelled it.", "expected": "billing"},
            {"input": "How do I reset my password?", "expected": "account"},
            {"input": "Change the email on my profile.", "expected": "account"},
            {"input": "I'm locked out after paying; the invoice is fine.", "expected": "account"},
        ],
        "holdout": [
            {"input": "I need a copy of the April invoice.", "expected": "billing"},
            {"input": "Is the refund already on its way?", "expected": "billing"},
            {"input": "There is a duplicate charge on this statement.", "expected": "billing"},
            {"input": "My bank shows two debits for one subscription.", "expected": "billing"},
            {"input": "The trial ended early and money left my account.", "expected": "billing"},
            {"input": "Can you add our tax number to last month's receipt?", "expected": "billing"},
            {"input": "Where can I update my profile picture?", "expected": "account"},
            {"input": "The password reset message went to spam.", "expected": "account"},
            {"input": "My login code never arrives on my new phone.", "expected": "account"},
            {"input": "How do I give a teammate access to the dashboard?", "expected": "account"},
            {"input": "The billing page logs me out every time I open it.", "expected": "account"},
            {"input": "Our admin left; how do I take over the workspace?", "expected": "account"},
        ],
    },
}


def _p(value: float) -> str:
    return "under 0.001" if value < 0.001 else f"{int(value * 1000 + 0.5) / 1000:.3f}"


class LabError(ValueError):
    """A public loop request could not be used."""


class _Pending(BaseException):
    """Raised inside a replay when the loop needs answers that the caller has not supplied yet."""


def catalog() -> dict:
    """Every loop on this server, its parts, and how to run it."""
    offline = offline_catalog()
    parts = {kind: {entry["id"]: entry for entry in entries} for kind, entries in offline["components"].items()}
    loops = []
    for template in TEMPLATES:
        config = template["config"]
        loops.append({
            "id": template["id"],
            "name": template["name"],
            "mode": "offline",
            "summary": template["description"],
            "best_for": _BEST_FOR[template["id"]],
            "improves": "A small routing prompt for the built-in interpreter. No model is called.",
            "inspired_by": template["source_url"],
            "parts": {kind: deepcopy(parts[kind][config[kind]]) for kind in engine.KINDS},
            "defaults": {"budget": config["budget"], "seed": config["seed"]},
            "run": {"method": "POST", "path": "/api/run", "body": {"loop": template["id"]}},
        })
    for entry in methods.catalog():
        search_name, search_text = _SEARCH_TEXT[entry["id"]]
        loops.append({
            "id": entry["id"],
            "name": entry["name"],
            "mode": "driven",
            "summary": entry["description"],
            "best_for": _BEST_FOR[entry["id"]],
            "improves": "Your own prompt" + (" or prompt chain" if "prompt-chain" in entry["artifact_types"] else "")
                        + ". Your model answers the work items; the server runs the loop and the holdout check.",
            "artifact_types": entry["artifact_types"],
            "parts": {
                "target": {"id": "artifact", "name": "Your prompt", "description": "The prompt or prompt chain you submit."},
                "proposer": {"id": entry["id"], "name": entry["name"], "description": entry["description"]},
                "judge": {"id": "normalized", "name": "Text match", "description": "Exact match, or a match that ignores case and spacing."},
                "search": {"id": _search_id(entry["id"]), "name": search_name, "description": search_text},
                "harness": {"id": "training-fixtures", "name": "Training cases", "description": "Loops see training cases only. Holdout labels stay on the server."},
            },
            "defaults": {"max_trials": 3, "objective": "normalized", "holdout_repeats": 1},
            "run": {"method": "POST", "path": "/api/runs", "body": {"loop": entry["id"], "prompt": "…", "task": {"train": "[…]", "holdout": "[…]"}}},
        })
    return {
        "loops": loops,
        "gallery": {"families": list(FAMILIES), "search_styles": list(SEARCH_STYLES), "methods": gallery(loops)},
        "components": offline["components"],
        "example_task": offline["example_task"],
        "example_prompt_run": deepcopy(EXAMPLE_PROMPT_RUN),
        "limits": {
            "offline": {"max_budget": engine.MAX_EVALUATIONS, "max_cases_per_split": 200},
            "driven": {
                "max_train_cases": methods.MAX_TRAIN_CASES, "min_holdout_cases": methods.MIN_HOLDOUT_CASES,
                "max_holdout_cases": methods.MAX_HOLDOUT_CASES, "max_trials": MAX_DRIVEN_TRIALS,
                "max_holdout_repeats": MAX_DRIVEN_REPEATS, "max_model_calls": MAX_DRIVEN_CALLS,
            },
        },
        "evidence_policy": methods.evidence_policy(),
    }


def _search_id(method: str) -> str:
    return methods._SEARCHES[method][0]


# Offline loops


_SHORTHAND = {"loop", "name", "budget", "seed", "task", "settings", "initial_candidate", *engine.KINDS}


def _offline_config(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise LabError("Send the run as a JSON object.")
    if "loop" not in payload:
        return payload
    unknown = payload.keys() - _SHORTHAND
    if unknown:
        raise LabError(f"Unknown field: {min(unknown)!r}.")
    template = next((item for item in TEMPLATES if item["id"] == payload["loop"]), None)
    if template is None:
        if payload["loop"] in DRIVEN_IDS:
            raise LabError(f"{payload['loop']} improves your own prompt, so it needs your model's answers. Start it with POST /api/runs.")
        raise LabError("Unknown loop. GET /api/catalog lists the loop IDs.")
    return {**deepcopy(template["config"]), **{key: value for key, value in payload.items() if key != "loop"}}


def offline_run(payload: object) -> dict:
    """Run one offline loop. Accepts a full configuration or {"loop": id, ...overrides}."""
    try:
        return engine.run(_offline_config(payload), make_registry())
    except ConfigError as exc:
        raise LabError(str(exc)) from None


def offline_compare(payload: object) -> dict:
    """Run 2 to 5 offline loops on the same task, seed and budget."""
    if not isinstance(payload, dict):
        raise LabError("Send the comparison as a JSON object.")
    if "configs" in payload:
        if set(payload) != {"configs"} or not isinstance(payload["configs"], list):
            raise LabError("Provide the experiments in a 'configs' list.")
        configs = [_offline_config(item) for item in payload["configs"]]
    else:
        loops = payload.get("loops")
        if loops == "all":
            loops = [template["id"] for template in TEMPLATES]
        if not isinstance(loops, list) or not all(isinstance(item, str) for item in loops):
            raise LabError("List the loop IDs to compare in 'loops', or use \"all\".")
        shared = {key: value for key, value in payload.items() if key != "loops"}
        if shared.keys() - {"budget", "seed", "task", "settings"}:
            raise LabError("A comparison shares budget, seed, task and settings.")
        configs = [_offline_config({"loop": loop, **deepcopy(shared)}) for loop in loops]
    try:
        result = engine.compare(configs, make_registry())
    except ConfigError as exc:
        raise LabError(str(exc)) from None
    runs = result["runs"]
    key = "holdout_score" if all(run["holdout_score"] is not None for run in runs) else "best_score"
    best = max(runs, key=lambda run: run[key])
    tied = [run["config"]["name"] for run in runs if run[key] == best[key]]
    result["summary"] = {
        "ranked_by": key,
        "best": tied if len(tied) > 1 else tied[0],
        "scores": [{"name": run["config"]["name"], "train": run["best_score"], "holdout": run["holdout_score"]} for run in runs],
    }
    return result


# Driven loops


def _work_id(stage: str, messages: list[dict[str, str]], seen: dict[str, int]) -> str:
    base = hashlib.sha256(json.dumps([stage, messages], ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    seen[base] = seen.get(base, 0) + 1
    return "w_" + hashlib.sha256(f"{base}:{seen[base]}".encode()).hexdigest()[:20]


def _advance(request: dict, answers: dict[str, str]) -> tuple[str, object]:
    """Replay one loop with the answers so far. Returns ("work", items), ("done", result) or ("failed", reason)."""
    pending: dict[str, dict] = {}
    pending_stage: list[str] = []
    seen: dict[str, int] = {}

    def complete(stage: str, messages: list[dict[str, str]]) -> str:
        if pending and stage != pending_stage[0]:
            # Later stages depend on these answers, so stop and ask for them.
            raise _Pending
        if any(_PENDING in message["content"] for message in messages):
            return _PENDING
        work_id = _work_id(stage, messages, seen)
        if work_id in answers:
            return answers[work_id]
        pending[work_id] = {
            "id": work_id,
            "kind": "propose" if stage == "propose" else "score",
            "stage": stage,
            "system": messages[0]["content"],
            "input": messages[-1]["content"],
        }
        pending_stage[:] = [stage]
        if stage == "propose" or len(pending) >= BATCH:
            raise _Pending
        return _PENDING

    try:
        result = methods.execute(request, request["method"], complete, lambda: None)
    except _Pending:
        return "work", list(pending.values())
    except (MethodError, ConfigError) as exc:
        return "failed", str(exc)
    if pending:
        return "work", list(pending.values())
    return "done", result


def _summary(result: dict) -> dict:
    keep = (
        "verdict", "accepted", "baseline_train", "best_train", "baseline_holdout", "candidate_holdout",
        "evidence", "artifact", "candidate", "candidate_trial_id", "trials", "model_calls", "holdout_cases", "holdout_repeats",
    )
    return {"verdict": result["evidence"]["verdict"], **{key: result[key] for key in keep if key in result}}


def _recommend(results: dict[str, dict], original: dict) -> dict:
    improved = {loop: item for loop, item in results.items() if item.get("verdict") == "improved"}
    if improved:
        loop, best = max(improved.items(), key=lambda pair: (pair[1]["evidence"]["gain"], -pair[1]["model_calls"]))
        return {
            "action": "adopt",
            "loop": loop,
            "artifact": best["artifact"],
            "reason": (
                f"{loop} raised holdout accuracy from {best['baseline_holdout']:.0%} to {best['candidate_holdout']:.0%} "
                f"(p = {_p(best['evidence']['p_value'])}), a statistically significant gain."
            ),
        }
    inconclusive = {loop: item for loop, item in results.items() if item.get("verdict") == "inconclusive"}
    if inconclusive:
        loop, best = max(inconclusive.items(), key=lambda pair: pair[1]["evidence"]["gain"])
        return {
            "action": "keep_original",
            "loop": loop,
            "artifact": original,
            "candidate_to_retest": best["candidate"],
            "reason": (
                f"{loop} scored higher on holdout ({best['baseline_holdout']:.0%} to {best['candidate_holdout']:.0%}), "
                f"but p = {_p(best['evidence']['p_value'])} could be chance. Keep the original, or retest the candidate "
                "with more holdout cases."
            ),
        }
    if results and all(item.get("baseline_train") == 1 for item in results.values()):
        reason = (
            "The original already answers every training case correctly, so the loops had nothing to fix. "
            "Keep it, or add training cases that it gets wrong."
        )
    else:
        reason = "No loop found a version that beat the original on holdout. Keep the original, or add training cases that show its mistakes."
    return {"action": "keep_original", "artifact": original, "reason": reason}


class Lab:
    """Storage and stepping for driven runs."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time):
        self.path = str(path)
        self._now = clock
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    requests TEXT NOT NULL,
                    answers TEXT NOT NULL,
                    final TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _requests(payload: object) -> list[dict]:
        if not isinstance(payload, dict):
            raise LabError("Send the run as a JSON object.")
        allowed = {"loop", "loops", "prompt", "artifact", "task", "max_trials", "objective", "holdout_repeats", "seed"}
        unknown = payload.keys() - allowed
        if unknown:
            raise LabError(f"Unknown field: {min(unknown)!r}. Use {', '.join(sorted(allowed))}.")
        if ("prompt" in payload) == ("artifact" in payload):
            raise LabError("Send either 'prompt' (text) or 'artifact' ({\"type\": \"prompt\"|\"prompt-chain\", …}).")
        artifact = payload["artifact"] if "artifact" in payload else {"type": "prompt", "text": payload["prompt"]}
        if "task" not in payload:
            raise LabError("Send 'task' with 'train' and 'holdout' case lists. Each case has 'input' and 'expected' text.")
        if _PENDING in json.dumps([artifact, payload["task"]], ensure_ascii=False, default=str):
            raise LabError("The task contains text reserved by the server.")
        if ("loop" in payload) == ("loops" in payload):
            raise LabError("Choose one loop with 'loop', or several with 'loops' (a list of IDs, or \"all\").")
        chosen = payload["loops"] if "loops" in payload else [payload["loop"]]
        trials = payload.get("max_trials", 3)
        repeats = payload.get("holdout_repeats", 1)
        if type(trials) is not int or not 1 <= trials <= MAX_DRIVEN_TRIALS:
            raise LabError(f"max_trials must be a whole number from 1 to {MAX_DRIVEN_TRIALS}.")
        if type(repeats) is not int or not 1 <= repeats <= MAX_DRIVEN_REPEATS:
            raise LabError(f"holdout_repeats must be a whole number from 1 to {MAX_DRIVEN_REPEATS}.")

        def build(method: str) -> dict:
            body = {
                "method": method, "artifact": artifact, "task": payload["task"],
                "budget": {"max_calls": MAX_DRIVEN_CALLS, "max_cost_microusd": 0, "max_seconds": 3600},
                "max_trials": trials, "holdout_repeats": repeats,
                "objective": payload.get("objective", "normalized"),
            }
            if "seed" in payload:
                body["seed"] = payload["seed"]
            try:
                return methods.validate_request(body)
            except MethodError as exc:
                raise LabError(str(exc)) from None

        if chosen == "all":
            probe = build("auto")
            chosen = list(methods.compatible_methods(probe))
        if not isinstance(chosen, list) or not chosen or len(set(map(str, chosen))) != len(chosen):
            raise LabError("List each loop ID once.")
        for loop in chosen:
            if loop not in DRIVEN_IDS:
                hint = " It runs without a model: use POST /api/run." if any(item["id"] == loop for item in TEMPLATES) else ""
                raise LabError(f"{loop!r} is not a loop for your own prompt.{hint} Choose from {', '.join(DRIVEN_IDS)}.")
        return [build(loop) for loop in chosen]

    def start(self, payload: object) -> dict:
        requests = self._requests(payload)
        run_id = "run_" + secrets.token_urlsafe(18)
        now = self._now()
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM runs WHERE updated_at < ?", (now - RETENTION_SECONDS,))
            count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            if count >= MAX_STORED_RUNS:
                connection.execute(
                    "DELETE FROM runs WHERE id IN (SELECT id FROM runs ORDER BY updated_at LIMIT ?)", (count - MAX_STORED_RUNS + 100,),
                )
            connection.execute(
                "INSERT INTO runs (id, created_at, updated_at, requests, answers) VALUES (?, ?, ?, ?, '{}')",
                (run_id, now, now, json.dumps(requests)),
            )
        return self._step(run_id, {})

    def get(self, run_id: str) -> dict | None:
        return self._step(run_id, {})

    def submit(self, run_id: str, outputs: object) -> dict | None:
        if isinstance(outputs, list):
            if not all(isinstance(item, dict) and set(item) == {"id", "output"} for item in outputs):
                raise LabError("Each output needs an 'id' and an 'output'.")
            outputs = {item["id"]: item["output"] for item in outputs}
        if not isinstance(outputs, dict) or not outputs:
            raise LabError("Send 'outputs' as a list of {\"id\", \"output\"} or an object mapping work IDs to outputs.")
        for work_id, output in outputs.items():
            if not isinstance(work_id, str) or not isinstance(output, str):
                raise LabError("Work IDs and outputs must be text.")
            if len(output) > MAX_OUTPUT_CHARS:
                raise LabError(f"Keep each output under {MAX_OUTPUT_CHARS} characters.")
            if _PENDING in output:
                raise LabError("An output contains text reserved by the server.")
        return self._step(run_id, outputs)

    def _step(self, run_id: str, outputs: dict[str, str]) -> dict | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            if row["final"] is not None:
                view = json.loads(row["final"])
                if outputs:
                    view["note"] = "This run is finished, so the outputs were not needed."
                return view
            requests = json.loads(row["requests"])
            answers = json.loads(row["answers"])
            states = {request["method"]: _advance(request, answers) for request in requests}
            if outputs:
                wanted = {item["id"] for state, items in states.values() if state == "work" for item in items}
                accepted = {work_id: output for work_id, output in outputs.items() if work_id in wanted}
                ignored = sorted(outputs.keys() - accepted.keys())
                if accepted:
                    answers.update(accepted)
                    states = {request["method"]: _advance(request, answers) for request in requests}
            else:
                ignored = []

            work: dict[str, dict] = {}
            results, failures = {}, {}
            for loop, (state, value) in states.items():
                if state == "work":
                    for item in value:
                        work.setdefault(item["id"], {**item, "loops": []})["loops"].append(loop)
                elif state == "done":
                    results[loop] = _summary(value)
                else:
                    failures[loop] = value
            view = {
                "id": run_id,
                "state": "needs_work" if work else "done" if results else "failed",
                "loops": list(states),
                "answered": len(answers),
            }
            if work:
                view["work"] = list(work.values())[:MAX_WORK]
                view["instructions"] = WORK_INSTRUCTIONS
                view["submit"] = {
                    "method": "POST", "path": f"/api/runs/{run_id}/work",
                    "body": {"outputs": [{"id": "<work id>", "output": "<your model's reply>"}]},
                }
            if ignored:
                view["ignored"] = ignored[:20]
                view["note"] = "Some outputs did not match a pending work item and were ignored."
            if results:
                view["results"] = results
            if failures:
                view["failed"] = failures
            if not work:
                if results:
                    view["recommendation"] = _recommend(results, requests[0]["artifact"])
                view["holdout_note"] = (
                    "Verdicts come from a one-sided exact sign test on paired holdout cases at the 5% level. "
                    "Scores reflect the answers this run was given."
                )
            now = self._now()
            connection.execute(
                "UPDATE runs SET answers = ?, updated_at = ?, final = ? WHERE id = ?",
                (json.dumps(answers), now, None if work else json.dumps(view), run_id),
            )
            return view


def agent_guide(base: str) -> str:
    """Plain-text instructions for an LLM agent that arrives with only a URL."""
    loops = catalog()["loops"]
    offline = "\n".join(f"- `{loop['id']}`: {loop['summary']}" for loop in loops if loop["mode"] == "offline")
    driven = "\n".join(f"- `{loop['id']}`: {loop['summary']}" for loop in loops if loop["mode"] == "driven")
    return f"""# RSI Library

RSI Library runs self-improvement loops. A loop repeatedly proposes a change to something (here, a prompt),
scores the change on training cases, keeps what works, and finally checks the winner on held-out cases.

You can use everything on this site without an account or a key. If you have MCP tools, connect to
{base}/mcp (Streamable HTTP, no sign-in). Otherwise use the JSON API below. Requests and replies are JSON.

## Loops that need no model

They improve a small routing prompt for a built-in interpreter, so they run instantly on the server.
Use them to see how each loop searches, or to compare loops on a labelled classification task.

{offline}

Run one:      POST {base}/api/run      {{"loop": "greedy-prompt", "budget": 24, "seed": 13}}
              GET  {base}/api/run?loop=greedy-prompt&budget=24&seed=13
Compare:      POST {base}/api/compare  {{"loops": "all", "budget": 24, "seed": 13}}
              GET  {base}/api/compare?loops=all&budget=24&seed=13

Optional fields: `task` ({{"train": [{{"input", "expected"}}…], "holdout": […]}}), `budget` (1-100 scored versions,
including the starting one), `seed`, and a replacement part: `target`, `proposer`, `judge`, `search` or `harness`.
GET {base}/api/catalog lists every loop, the parts you can swap in, and an example task.
Its `gallery` field has fact sheets for published self-improvement methods (OPRO, GEPA, Darwin Gödel Machine and
others) described in the same five parts, each with the closest loop you can run here.

## Loops that improve the user's own prompt

Here you are the model. The server runs the loop and sends you work items; you answer them; the server
scores your answers, keeps the holdout labels, and decides whether the new prompt is really better.

{driven}

1. Start:   POST {base}/api/runs
            {{"loop": "reflect-v1", "prompt": "<the user's prompt>",
              "task": {{"train": [{{"input": "…", "expected": "…"}}], "holdout": [ …at least 8 cases… ]}},
              "max_trials": 3}}
   Use "loops": "all" instead of "loop" to test every loop on the same task in one run.
   A prompt chain goes in "artifact": {{"type": "prompt-chain", "steps": [{{"name", "text"}}, …]}} instead of "prompt".
2. The reply has "state": "needs_work" and a "work" list. Each item has `id`, `kind`, `system` and `input`.
   {WORK_INSTRUCTIONS}
3. Submit:  POST {base}/api/runs/<run id>/work   {{"outputs": [{{"id": "w_…", "output": "…"}}]}}
   The reply is the next batch of work, or the finished run. Repeat until "state" is "done".
4. A finished run has "results" for each loop (verdict, holdout scores, p-value, the candidate prompt)
   and a "recommendation": adopt a specific prompt, or keep the original and why.
   GET {base}/api/runs/<run id> returns the current state at any time.

The user supplies the prompt and the labelled cases. If they gave no cases, ask for them or draft cases and
have the user confirm the expected answers; do not invent labels silently. Keep holdout cases different from
training cases. Answer 'score' items honestly as the given prompt would, even when the prompt is poor:
the comparison only means something if the original is scored fairly.

Verdicts: "improved" means the candidate scored higher on holdout and a one-sided exact sign test on the
paired cases gave p <= 0.05. "inconclusive" means it scored higher but the gain could be chance.
"not_improved" means it did not score higher. Only "improved" is a reason to replace the prompt.

## Reporting back

Tell the user which loop you ran, the holdout scores before and after, the p-value and verdict, and the
recommended prompt in full. If you compared loops, say which did best and what you would do next.
"""
