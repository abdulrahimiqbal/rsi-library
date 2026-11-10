"""Measure the loops that improve your own prompt, on a live model, through the public runs API.

The offline loops are measured exactly by `python -m rsi_library.benchmarks`. The loops
that improve your own prompt (reflect-v1, evolve-v1, self-revise-v1) need a model, so
their results depend on which model answers. This script runs every one of them on the
same benchmark tasks with an OpenRouter model you choose, answering each run's work
items itself, and reports how much each loop improved the holdout score and how many
model calls that took.

    export OPENROUTER_API_KEY=sk-or-...
    python examples/lab_benchmark.py --model openai/gpt-4.1-mini --out driven.json
    python -m rsi_library.benchmarks --write --driven driven.json   # publish on the site

Each task starts from a deliberately vague prompt ("Label this message with one word.")
so the loops have to discover the labels and the rules from the training examples. Calls
are billed to your OpenRouter account; one full run of the default settings makes a few
thousand short calls. The key never leaves this machine except to OpenRouter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi_library.benchmarks import TASKS  # noqa: E402

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
LOOPS = ("reflect-v1", "evolve-v1", "self-revise-v1")
START = "Label this message with one word."


def post(url: str, payload: dict, headers: dict | None = None, timeout: float = 120) -> dict:
    request = Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", **(headers or {})})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise SystemExit(f"{url} answered {error.code}: {error.read()[:300]!r}") from None
        except URLError as error:
            if attempt == 3:
                raise SystemExit(f"Could not reach {url}: {error.reason}") from None
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def ask(model: str, key: str, item: dict) -> str:
    if item["kind"] == "run":
        raise SystemExit("This benchmark only improves prompts; a run asked for a function to be run.")
    reply = post(OPENROUTER, {
        "model": model, "temperature": 0,
        "max_tokens": 4000 if item["kind"] == "propose" else 16 if item["kind"] == "judge" else 300,
        "messages": [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["input"]}],
    }, {"Authorization": f"Bearer {key}"})
    content = reply.get("choices", [{}])[0].get("message", {}).get("content")
    return content if isinstance(content, str) else ""


def drive(base: str, payload: dict, answer, workers: int = 8) -> tuple[dict, int]:
    """Start a run and answer its work until it finishes. Returns the final view and the answers given."""
    view = post(f"{base}/api/runs", payload)
    print(f"  run page: {view['page']}", file=sys.stderr)
    answered = 0
    with ThreadPoolExecutor(workers) as pool:
        while view["state"] == "needs_work":
            items = view["work"]
            outputs = list(pool.map(lambda item: {"id": item["id"], "output": answer(item)}, items))
            answered += len(outputs)
            view = post(f"{base}{view['submit']['path']}", {"outputs": outputs})
    return view, answered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="an OpenRouter model ID, for example openai/gpt-4.1-mini")
    parser.add_argument("--base", default="https://rsilibrary.xyz", help="the RSI Library site to run on")
    parser.add_argument("--tasks", default=",".join(TASKS), help="benchmark task IDs, comma separated")
    parser.add_argument("--trials", type=int, default=3, help="revisions each loop may try (1-10)")
    parser.add_argument("--seeds", type=int, default=1, help="runs per task, with seeds 0, 1, …")
    parser.add_argument("--out", help="write the results as JSON to this file")
    args = parser.parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    base = args.base.rstrip("/")
    runs = []
    for task_id in args.tasks.split(","):
        task = TASKS[task_id]
        for seed in range(args.seeds):
            print(f"{task_id}, seed {seed}", file=sys.stderr)
            view, answered = drive(base, {
                "loops": list(LOOPS), "prompt": START, "task": task["task"], "max_trials": args.trials, "seed": seed,
            }, lambda item: ask(args.model, key, item))
            for loop, result in view.get("results", {}).items():
                runs.append({
                    "task": task_id, "seed": seed, "loop": loop, "verdict": result["verdict"],
                    "baseline_holdout": result["baseline_holdout"], "candidate_holdout": result["candidate_holdout"],
                    "gain": result["candidate_holdout"] - result["baseline_holdout"], "p_value": result["evidence"]["p_value"],
                    "model_calls": result["model_calls"],
                })
            for loop, reason in view.get("failed", {}).items():
                runs.append({"task": task_id, "seed": seed, "loop": loop, "verdict": "failed", "reason": reason})
            print(f"  {answered} answers", file=sys.stderr)
    loops = {}
    for loop in LOOPS:
        rows = [row for row in runs if row["loop"] == loop and row["verdict"] != "failed"]
        if rows:
            gain, calls = fmean(row["gain"] for row in rows), fmean(row["model_calls"] for row in rows)
            loops[loop] = {
                "runs": len(rows), "improved": sum(row["verdict"] == "improved" for row in rows),
                "holdout_gain": round(gain, 4), "model_calls": round(calls, 1),
                "gain_per_100_calls": round(100 * gain / calls, 4) if calls else None,
            }
    result = {
        "status": "measured", "model": args.model, "date": time.strftime("%Y-%m-%d"), "start_prompt": START,
        "tasks": args.tasks.split(","), "max_trials": args.trials, "seeds": args.seeds, "loops": loops, "runs": runs,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    for loop, row in loops.items():
        print(f"{loop:16} {row['holdout_gain'] * 100:+6.1f} pts  {row['improved']}/{row['runs']} improved  "
              f"{row['model_calls']:.0f} calls per run", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
