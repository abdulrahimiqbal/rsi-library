"""Public loops that need no workspace key.

Two kinds of loop are served here:

* Offline loops run entirely on this server against a small built-in interpreter.
  They show how a loop behaves and cost nothing.
* Driven loops improve a real prompt, prompt chain or program, with the caller acting
  as the model (and running any program). The server runs the loop, hands out work
  items ("answer this prompt for this input", "run this function on this input"),
  scores the answers, keeps the holdout labels to itself, and decides with the same
  paired test as a hosted job. The server never executes a program.

A driven run stores the caller's request, the answers supplied so far, the latest view
and the IDs of the work it is waiting for. A step replays the loop from the start with
the stored answers, which is exact because the loop is deterministic for a given seed
and set of answers. Reading a run serves the stored view without replaying it, and a
finished run keeps only its final view.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path

from .. import engine
from ..benchmarks import results as benchmark_results
from ..builtins import TEMPLATES, catalog as offline_catalog, make_registry
from ..engine import ConfigError
from . import methods
from .methods import MethodError
from .methods_gallery import CHECKED, FAMILIES, SEARCH_STYLES, gallery

DRIVEN_IDS = tuple(methods._METHODS)
MAX_DRIVEN_TRIALS = 10
MAX_DRIVEN_REPEATS = 3
MAX_DRIVEN_CALLS = 1000
MAX_OUTPUT_CHARS = 16_000
BATCH = 40
MAX_WORK = 64
UNFINISHED_SECONDS = 14 * 86_400
FINISHED_SECONDS = 90 * 86_400
MAX_STORED_RUNS = 5000
# Answers and the cached view of one unfinished run, and of every stored run together.
MAX_RUN_BYTES = 8_000_000
MAX_TOTAL_BYTES = 1_500_000_000
_LOCKS = 64
_PENDING = "[[rsi-library-pending-output]]"
RETENTION_TEXT = "Unfinished runs are kept for 14 days after their last step, finished runs for 90 days."

_BEST_FOR = {
    "greedy-prompt": "The simplest loop. Start here to see propose, score, keep.",
    "archive-prompt": "Seeing how branching from older versions escapes a dead end.",
    "adaptive-improver": "Watching a loop change how it edits, not only what it edits.",
    "reflect-v1": "A prompt or function that makes the same kind of mistake repeatedly.",
    "evolve-v1": "Tasks where different fixes help different cases.",
    "self-revise-v1": "Longer runs, where better editing instructions pay off.",
}

_SEARCH_TEXT = {
    "reflect-v1": ("Best first", "Build on the highest scoring version, the earliest on ties."),
    "evolve-v1": ("Complementary parents", "Pair the best version with the one that fixes most of its training mistakes."),
    "self-revise-v1": ("Newest best", "Build on the best version, preferring the newest so fresh editing instructions get used."),
}

_KIND_INSTRUCTIONS = {
    "score": (
        "For kind 'score', reply as a chat model would when `system` is its system prompt and `input` is the user "
        "message. Follow the prompt exactly: it is the version being tested, so do not improve on it or correct it."
    ),
    "judge": "For kind 'judge', grade the reply as `system` asks and answer with the number only.",
    "propose": "For kind 'propose', return the JSON object that `system` asks for.",
    "run": (
        "For kind 'run', call the function `entry` defined in `source` (written in `language`) with `input` as its "
        "only argument, in a sandbox with no network or file access and a time limit of a few seconds. Reply with "
        "the return value: text as it is, anything else as JSON, or 'Error: ' and the message if it raises or times out."
    ),
}


def _instructions(kinds: set[str]) -> str:
    parts = ["You are the model for this run. Reply with the answer only, with no commentary."]
    parts += [_KIND_INSTRUCTIONS[kind] for kind in ("score", "judge", "propose", "run") if kind in kinds]
    parts.append("Answer every item, then submit them together.")
    return " ".join(parts)


WORK_INSTRUCTIONS = _instructions(set(_KIND_INSTRUCTIONS))


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

EXAMPLE_PROGRAM_RUN = {
    "artifact": {
        "type": "program",
        "language": "javascript",
        "entry": "toSeconds",
        "source": (
            "// Convert a duration such as \"90s\" or \"2m\" to a number of seconds.\n"
            "function toSeconds(text) {\n"
            "  const match = text.match(/(\\d+)\\s*([sm])/);\n"
            "  if (!match) return 0;\n"
            "  return Number(match[1]) * (match[2] === \"m\" ? 60 : 1);\n"
            "}\n"
        ),
    },
    "objective": "number",
    "task": {
        "train": [
            {"input": "45s", "expected": "45"},
            {"input": "2m", "expected": "120"},
            {"input": "1h", "expected": "3600"},
            {"input": "1h 30m", "expected": "5400"},
            {"input": "90 seconds", "expected": "90"},
            {"input": "1.5h", "expected": "5400"},
            {"input": "2 minutes 10 seconds", "expected": "130"},
            {"input": "3m 5s", "expected": "185"},
        ],
        "holdout": [
            {"input": "10s", "expected": "10"},
            {"input": "4m", "expected": "240"},
            {"input": "2h", "expected": "7200"},
            {"input": "1h 15m", "expected": "4500"},
            {"input": "30 seconds", "expected": "30"},
            {"input": "2.5m", "expected": "150"},
            {"input": "1 hour 1 minute", "expected": "3660"},
            {"input": "5m 30s", "expected": "330"},
            {"input": "0.5h", "expected": "1800"},
            {"input": "3 hours", "expected": "10800"},
            {"input": "1m 1s", "expected": "61"},
            {"input": "12 minutes", "expected": "720"},
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
    for entry in methods.catalog(programs=True):
        search_name, search_text = _SEARCH_TEXT[entry["id"]]
        types = entry["artifact_types"]
        loops.append({
            "id": entry["id"],
            "name": entry["name"],
            "mode": "driven",
            "summary": entry["description"],
            "best_for": _BEST_FOR[entry["id"]],
            "improves": ("Your own prompt, prompt chain or function" if "program" in types else "Your own prompt")
                        + ". Your model answers the work items and your sandbox runs any code; the server runs the loop "
                        "and the holdout check.",
            "artifact_types": types,
            "parts": {
                "target": {"id": "artifact", "name": "Your prompt", "description": "The prompt, prompt chain or function you submit."
                           if "program" in types else "The prompt you submit."},
                "proposer": {"id": entry["id"], "name": entry["name"], "description": entry["description"]},
                "judge": {"id": "normalized", "name": "Text match", "description": "Match the expected output, ignoring case and spacing. Other objectives can be chosen."},
                "search": {"id": _search_id(entry["id"]), "name": search_name, "description": search_text},
                "harness": {"id": "training-fixtures", "name": "Training cases", "description": "Loops see training cases only. Holdout labels stay on the server."},
            },
            "defaults": {"max_trials": 3, "objective": "normalized", "holdout_repeats": 1},
            "run": {"method": "POST", "path": "/api/runs", "body": {"loop": entry["id"], "prompt": "…", "task": {"train": "[…]", "holdout": "[…]"}}},
        })
    return {
        "loops": loops,
        "objectives": methods.objectives(),
        "gallery": {
            "families": list(FAMILIES), "search_styles": list(SEARCH_STYLES), "checked": CHECKED,
            "suggest": "https://github.com/abdulrahimiqbal/rsi-library/issues/new?template=suggest-method.yml",
            "methods": gallery(loops),
        },
        "benchmarks": benchmark_results(),
        "components": offline["components"],
        "example_task": offline["example_task"],
        "example_prompt_run": deepcopy(EXAMPLE_PROMPT_RUN),
        "example_program_run": deepcopy(EXAMPLE_PROGRAM_RUN),
        "limits": {
            "offline": {"max_budget": engine.MAX_EVALUATIONS, "max_cases_per_split": 200},
            "driven": {
                "max_train_cases": methods.MAX_TRAIN_CASES, "min_holdout_cases": methods.MIN_HOLDOUT_CASES,
                "max_holdout_cases": methods.MAX_HOLDOUT_CASES, "max_trials": MAX_DRIVEN_TRIALS,
                "max_holdout_repeats": MAX_DRIVEN_REPEATS, "max_model_calls": MAX_DRIVEN_CALLS,
                "max_output_chars": MAX_OUTPUT_CHARS, "max_source_bytes": methods._MAX_SOURCE_BYTES,
                "max_stored_bytes_per_run": MAX_RUN_BYTES,
            },
            "retention": RETENTION_TEXT,
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


def _work_item(work_id: str, stage: str, messages: list[dict[str, str]]) -> dict:
    if messages[0]["role"] == "program":
        spec = json.loads(messages[0]["content"])
        return {
            "id": work_id, "kind": "run", "stage": stage, "language": spec["language"], "entry": spec["entry"],
            "source": spec["source"], "input": messages[-1]["content"],
        }
    kind = "propose" if stage == "propose" else "judge" if stage.endswith("_judge") else "score"
    return {"id": work_id, "kind": kind, "stage": stage, "system": messages[0]["content"], "input": messages[-1]["content"]}


def _advance(request: dict, answers: dict[str, str]) -> tuple[str, object, dict | None]:
    """Replay one loop with the answers so far.

    Returns ("work", items, progress), ("done", result, None) or ("failed", reason, None).
    Progress is the loop's last report before it needed an answer it doesn't have yet, so
    it never reflects a placeholder.
    """
    pending: dict[str, dict] = {}
    pending_stage: list[str] = []
    seen: dict[str, int] = {}
    latest: list[dict] = []

    def complete(stage: str, messages: list[dict[str, str]]) -> str:
        if pending and stage != pending_stage[0]:
            # Later stages depend on these answers, so stop and ask for them.
            raise _Pending
        if any(_PENDING in message["content"] for message in messages):
            return _PENDING
        work_id = _work_id(stage, messages, seen)
        if work_id in answers:
            return answers[work_id]
        pending[work_id] = _work_item(work_id, stage, messages)
        pending_stage[:] = [stage]
        if stage == "propose" or len(pending) >= BATCH:
            raise _Pending
        return _PENDING

    def progress(event: dict) -> None:
        if not pending:
            latest[:] = [event]

    try:
        result = methods.execute(request, request["method"], complete, lambda: None, progress=progress, runs_programs=True)
    except _Pending:
        return "work", list(pending.values()), latest[0] if latest else None
    except (MethodError, ConfigError) as exc:
        return "failed", str(exc), None
    if pending:
        return "work", list(pending.values()), latest[0] if latest else None
    return "done", result, None


_DOING = {
    "baseline_train": "Scoring the original on the training cases",
    "propose": "Writing version {next} of {planned}",
    "candidate_train": "Scoring version {next} of {planned} on the training cases",
    "baseline_holdout": "Final check: scoring the original on the holdout cases",
    "candidate_holdout": "Final check: scoring the best version on the holdout cases",
}


def _progress(event: dict | None) -> dict:
    if event is None:
        return {"state": "working", "doing": "Starting"}
    doing = _DOING.get(event["stage"], "Working").format(next=event["trials"] + 1, planned=event["planned_trials"])
    return {
        "state": "working", "stage": event["stage"], "doing": doing, "trials": event["trials"],
        "planned_trials": event["planned_trials"], "best_train": event["best_train"], "calls": event["calls"],
        "trace": event["trace"],
    }


def _summary(result: dict) -> dict:
    keep = (
        "verdict", "accepted", "baseline_train", "best_train", "baseline_holdout", "candidate_holdout",
        "evidence", "artifact", "candidate", "candidate_trial_id", "trials", "model_calls", "holdout_cases", "holdout_repeats",
    )
    return {"verdict": result["evidence"]["verdict"], **{key: result[key] for key in keep if key in result}}


def _recommend(results: dict[str, dict], original: dict) -> dict:
    improved = {loop: item for loop, item in results.items() if item.get("verdict") == "improved"}
    what = "function" if original["type"] == "program" else "prompt"
    if improved:
        loop, best = max(improved.items(), key=lambda pair: (pair[1]["evidence"]["gain"], -pair[1]["model_calls"]))
        return {
            "action": "adopt",
            "loop": loop,
            "artifact": best["artifact"],
            "reason": (
                f"{loop} raised the holdout score from {best['baseline_holdout']:.0%} to {best['candidate_holdout']:.0%} "
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
            f"The original {what} already scores full marks on every training case, so the loops had nothing to fix. "
            "Keep it, or add training cases that it gets wrong."
        )
    else:
        reason = (
            f"No loop found a version that beat the original {what} on holdout. Keep the original, or add training "
            "cases that show its mistakes."
        )
    return {"action": "keep_original", "artifact": original, "reason": reason}


def _request_summary(requests: list[dict]) -> dict:
    first = requests[0]
    return {
        "artifact": first["artifact"], "objective": first["objective"], "scoring": first.get("scoring", {}),
        "train_cases": len(first["task"]["train"]), "holdout_cases": len(first["task"]["holdout"]),
        "max_trials": first["max_trials"], "holdout_repeats": first["holdout_repeats"], "seed": first["seed"],
    }


def _view(run_id: str, requests: list[dict], answers: dict[str, str], states: dict) -> tuple[dict, list[str]]:
    """The caller's view of a run, and the IDs of every work item it is waiting for."""
    work: dict[str, dict] = {}
    results, failures, progress = {}, {}, {}
    for loop, (state, value, event) in states.items():
        if state == "work":
            for item in value:
                work.setdefault(item["id"], {**item, "loops": []})["loops"].append(loop)
            progress[loop] = _progress(event)
        elif state == "done":
            results[loop] = _summary(value)
            progress[loop] = {"state": "done", "verdict": results[loop]["verdict"]}
        else:
            failures[loop] = value
            progress[loop] = {"state": "failed", "reason": value}
    view = {
        "id": run_id,
        "state": "needs_work" if work else "done" if results else "failed",
        "page": f"/runs/{run_id}",
        "loops": list(states),
        "request": _request_summary(requests),
        "answered": len(answers),
        "progress": progress,
    }
    if work:
        items = list(work.values())
        view["pending_work"] = len(items)
        view["work"] = items[:MAX_WORK]
        view["instructions"] = _instructions({item["kind"] for item in view["work"]})
        view["submit"] = {
            "method": "POST", "path": f"/api/runs/{run_id}/work",
            "body": {"outputs": [{"id": "<work id>", "output": "<your reply, or the function's result>"}]},
        }
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
    return view, list(work)


def without_work(view: dict) -> dict:
    """A run's view without its work items, for watching a run cheaply."""
    return {key: value for key, value in view.items() if key not in ("work", "instructions", "submit")}


def with_base(view: dict, base: str) -> dict:
    """Make the run page link absolute for callers outside the browser."""
    return {**view, "page": base + view["page"]} if view.get("page", "").startswith("/") else view


class Lab:
    """Storage and stepping for driven runs."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time):
        self.path = str(path)
        self._now = clock
        # One lock per run, striped: a slow replay holds up only runs that share its stripe.
        self._locks = tuple(threading.Lock() for _ in range(_LOCKS))
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
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
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
            for name, kind in (("view", "TEXT"), ("pending", "TEXT"), ("size", "INTEGER NOT NULL DEFAULT 0")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {kind}")
            connection.execute("CREATE INDEX IF NOT EXISTS runs_updated ON runs (updated_at)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _row(self, run_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as connection:
            return connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def _lock(self, run_id: str) -> threading.Lock:
        return self._locks[int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16) % _LOCKS]

    @staticmethod
    def _requests(payload: object) -> list[dict]:
        if not isinstance(payload, dict):
            raise LabError("Send the run as a JSON object.")
        allowed = {"loop", "loops", "prompt", "artifact", "task", "max_trials", "objective", "scoring", "holdout_repeats", "seed"}
        unknown = payload.keys() - allowed
        if unknown:
            raise LabError(f"Unknown field: {min(unknown)!r}. Use {', '.join(sorted(allowed))}.")
        if ("prompt" in payload) == ("artifact" in payload):
            raise LabError(
                "Send either 'prompt' (text) or 'artifact': {\"type\": \"prompt-chain\", \"steps\": […]} or "
                "{\"type\": \"program\", \"language\", \"entry\", \"source\"}."
            )
        artifact = payload["artifact"] if "artifact" in payload else {"type": "prompt", "text": payload["prompt"]}
        if "task" not in payload:
            raise LabError("Send 'task' with 'train' and 'holdout' case lists. Each case has 'input' and 'expected' text.")
        if _PENDING in json.dumps([artifact, payload["task"], payload.get("scoring")], ensure_ascii=False, default=str):
            raise LabError("The request contains text reserved by the server.")
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
            for key in ("seed", "scoring"):
                if key in payload:
                    body[key] = payload[key]
            try:
                return methods.validate_request(body, programs=True)
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
                raise LabError(f"{loop!r} is not a loop for your own prompt or program.{hint} Choose from {', '.join(DRIVEN_IDS)}.")
        return [build(loop) for loop in chosen]

    def start(self, payload: object) -> dict:
        requests = self._requests(payload)
        run_id = "run_" + secrets.token_urlsafe(18)
        now = self._now()
        with self._transaction() as connection:
            self._purge(connection, now)
            connection.execute(
                "INSERT INTO runs (id, created_at, updated_at, requests, answers, size) VALUES (?, ?, ?, ?, '{}', 0)",
                (run_id, now, now, json.dumps(requests)),
            )
        return self._step(run_id, {})

    @staticmethod
    def _purge(connection: sqlite3.Connection, now: float) -> None:
        connection.execute("DELETE FROM runs WHERE final IS NULL AND updated_at < ?", (now - UNFINISHED_SECONDS,))
        connection.execute("DELETE FROM runs WHERE final IS NOT NULL AND updated_at < ?", (now - FINISHED_SECONDS,))
        count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        if count >= MAX_STORED_RUNS:
            connection.execute(
                "DELETE FROM runs WHERE id IN (SELECT id FROM runs ORDER BY updated_at LIMIT ?)", (count - MAX_STORED_RUNS + 100,),
            )
        total = connection.execute("SELECT COALESCE(SUM(size), 0) FROM runs").fetchone()[0]
        if total > MAX_TOTAL_BYTES:
            # Free a tenth of the allowance at once, starting with the runs untouched longest.
            excess, doomed = total - MAX_TOTAL_BYTES * 9 // 10, []
            for row in connection.execute("SELECT id, size FROM runs ORDER BY updated_at"):
                doomed.append((row["id"],))
                excess -= row["size"]
                if excess <= 0:
                    break
            connection.executemany("DELETE FROM runs WHERE id = ?", doomed)

    def get(self, run_id: str, *, work: bool = True) -> dict | None:
        row = self._row(run_id)
        if row is None:
            return None
        if row["final"] is not None:
            view = self._final(run_id, row)
        elif row["view"] is not None:
            view = json.loads(row["view"])
        else:
            # Stored before views were cached: replay once and store the view.
            view = self._step(run_id, {})
            if view is None:
                return None
        return view if work else without_work(view)

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

    def delete(self, run_id: str) -> bool:
        with self._lock(run_id), self._transaction() as connection:
            return connection.execute("DELETE FROM runs WHERE id = ?", (run_id,)).rowcount > 0

    def backup(self, path: str | os.PathLike[str]) -> None:
        """Write a consistent online backup of the runs into a new owner-readable file."""
        destination = Path(path)
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with closing(self._connect()) as source, closing(sqlite3.connect(destination)) as target:
                source.backup(target)
        except BaseException:
            destination.unlink()
            raise

    @staticmethod
    def _final(run_id: str, row: sqlite3.Row) -> dict:
        view = json.loads(row["final"])
        view.setdefault("page", f"/runs/{run_id}")
        return view

    @staticmethod
    def _states(requests: list[dict], answers: dict[str, str]) -> dict:
        return {request["method"]: _advance(request, answers) for request in requests}

    def _step(self, run_id: str, outputs: dict[str, str]) -> dict | None:
        with self._lock(run_id):
            row = self._row(run_id)
            if row is None:
                return None
            if row["final"] is not None:
                view = self._final(run_id, row)
                if outputs:
                    view["note"] = "This run is finished, so the outputs were not needed."
                return view
            requests = json.loads(row["requests"])
            answers = json.loads(row["answers"])
            ignored: list[str] = []
            if outputs:
                if row["pending"] is not None:
                    wanted = set(json.loads(row["pending"]))
                else:
                    wanted = {item["id"] for state, items, _ in self._states(requests, answers).values() if state == "work" for item in items}
                accepted = {work_id: output for work_id, output in outputs.items() if work_id in wanted}
                ignored = sorted(outputs.keys() - accepted.keys())
                if not accepted and row["view"] is not None:
                    return self._ignored(json.loads(row["view"]), ignored)
                answers.update(accepted)
            elif row["view"] is not None:
                return json.loads(row["view"])
            stored_answers = json.dumps(answers)
            if len(stored_answers) > MAX_RUN_BYTES:
                raise LabError(
                    f"This run has reached its storage limit of {MAX_RUN_BYTES // 1_000_000} MB of answers. "
                    "Start a smaller run, with fewer loops, trials or cases, or shorter outputs."
                )
            view, pending = _view(run_id, requests, answers, self._states(requests, answers))
            stored_view = json.dumps(view)
            now = self._now()
            with self._transaction() as connection:
                if pending:
                    connection.execute(
                        "UPDATE runs SET answers = ?, updated_at = ?, view = ?, pending = ?, size = ? WHERE id = ?",
                        (stored_answers, now, stored_view, json.dumps(pending), len(stored_answers) + len(stored_view), run_id),
                    )
                else:
                    # A finished run keeps only its final view.
                    connection.execute(
                        "UPDATE runs SET answers = '{}', updated_at = ?, final = ?, view = NULL, pending = NULL, size = ? WHERE id = ?",
                        (now, stored_view, len(stored_view), run_id),
                    )
            return self._ignored(view, ignored) if ignored else view

    @staticmethod
    def _ignored(view: dict, ignored: list[str]) -> dict:
        return {**view, "ignored": ignored[:20], "note": "Some outputs did not match a pending work item and were ignored."}


def agent_guide(base: str) -> str:
    """Plain-text instructions for an LLM agent that arrives with only a URL."""
    loops = catalog()["loops"]
    offline = "\n".join(f"- `{loop['id']}`: {loop['summary']}" for loop in loops if loop["mode"] == "offline")
    driven = "\n".join(
        f"- `{loop['id']}`: {loop['summary']} Improves: {', '.join(loop['artifact_types'])}."
        for loop in loops if loop["mode"] == "driven"
    )
    objectives = "\n".join(
        f"- `{entry['id']}`: {entry['description']}"
        + (" Settings in `scoring`: " + "; ".join(f"`{name}`: {text}" for name, text in entry["settings"].items()) if entry["settings"] else "")
        for entry in methods.objectives()
    )
    return f"""# RSI Library

RSI Library runs self-improvement loops. A loop repeatedly proposes a change to something (a prompt, a
prompt chain, or a function), scores the change on training cases, keeps what works, and finally checks
the winner on held-out cases with a significance test.

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
GET {base}/api/catalog lists every loop, the parts you can swap in, example tasks, and measured results
(`benchmarks`: each offline loop on four tasks, 20 seeds and three budgets).
Its `gallery` field has fact sheets for published self-improvement methods (OPRO, GEPA, Darwin Gödel Machine and
others) described in the same five parts, each with the closest loop you can run here.

## Loops that improve the user's own prompt or function

Here you are the model, and your sandbox runs any code. The server runs the loop and sends you work items;
you answer them; the server scores your answers, keeps the holdout labels, and decides whether the new
version is really better.

{driven}

1. Start:   POST {base}/api/runs
            {{"loop": "reflect-v1", "prompt": "<the user's prompt>",
              "task": {{"train": [{{"input": "…", "expected": "…"}}], "holdout": [ …at least 8 cases… ]}},
              "max_trials": 3}}
   Use "loops": "all" instead of "loop" to test every suitable loop on the same task in one run.
   A prompt chain goes in "artifact": {{"type": "prompt-chain", "steps": [{{"name", "text"}}, …]}} instead of "prompt".
   A function goes in "artifact": {{"type": "program", "language": "python" or "javascript", "entry": "<function name>",
   "source": "<code>"}}. The function takes the case input as one text argument; its result is compared with `expected`.
   Choose how replies are scored with "objective" (default "normalized") and its optional "scoring" settings:
{objectives}
2. The reply has "state": "needs_work", a "work" list, and a "page" link. Give the user the page link: it shows
   the run live, and the final verdict, to anyone who has it. Each work item has `id`, `kind` and `input`.
   {WORK_INSTRUCTIONS}
3. Submit:  POST {base}/api/runs/<run id>/work   {{"outputs": [{{"id": "w_…", "output": "…"}}]}}
   The reply is the next batch of work, or the finished run. Repeat until "state" is "done".
   "progress" says what each loop is doing ("Writing version 2 of 3") and its best training score so far.
4. A finished run has "results" for each loop (verdict, holdout scores, p-value, the candidate)
   and a "recommendation": adopt a specific version, or keep the original and why.
   GET {base}/api/runs/<run id> returns the current state at any time; add ?work=0 to leave out the work items.
   DELETE {base}/api/runs/<run id> deletes the run. {RETENTION_TEXT}

The user supplies the prompt or function and the labelled cases. If they gave no cases, ask for them or draft
cases and have the user confirm the expected answers; do not invent labels silently. Keep holdout cases
different from training cases. Answer 'score' items honestly as the given prompt would, even when the prompt is
poor, and run 'run' items exactly as written: the comparison only means something if the original is scored fairly.

Verdicts: "improved" means the candidate scored higher on holdout and a one-sided exact sign test on the
paired cases gave p <= 0.05. "inconclusive" means it scored higher but the gain could be chance.
"not_improved" means it did not score higher. Only "improved" is a reason to replace the original.

## Reporting back

Tell the user which loop you ran, the holdout scores before and after, the p-value and verdict, the
recommended version in full, and the run page link. If you compared loops, say which did best and what
you would do next.
"""
