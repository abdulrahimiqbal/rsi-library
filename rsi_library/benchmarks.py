"""Measured results for the offline loops: every loop on the same tasks, seeds and budgets.

Each cell runs one loop on one task with one seed and budget, then compares the winner's
holdout score with the starting prompt's. Nothing here calls a model, so the numbers
are exact and anyone can reproduce them:

    python -m rsi_library.benchmarks --write

rewrites rsi_library/benchmark_results.json. The tests recompute a slice of the grid,
so the published numbers stay tied to the code.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from statistics import fmean

from . import engine
from .builtins import EXAMPLE_TASK, TEMPLATES, make_registry


def _cases(*rows: tuple[str, str]) -> list[dict[str, str]]:
    return [{"input": text, "expected": label} for text, label in rows]


TASKS = {
    "tickets": {
        "name": "Support tickets",
        "description": "Route a support message to billing or account.",
        "task": EXAMPLE_TASK,
    },
    "intents": {
        "name": "Product requests",
        "description": "Tell cancellations, upgrades, bug reports and questions apart.",
        "task": {
            "train": _cases(
                ("Please cancel my subscription today.", "cancel"),
                ("I want to cancel before the next renewal.", "cancel"),
                ("Stop my plan, I no longer need it.", "cancel"),
                ("How do I close my account for good?", "cancel"),
                ("I'd like to upgrade to the team plan.", "upgrade"),
                ("Can we upgrade and add five more seats?", "upgrade"),
                ("Move us to the annual premium tier.", "upgrade"),
                ("We need more seats for new hires.", "upgrade"),
                ("The export button crashes the app.", "bug"),
                ("Saving a file shows an error every time.", "bug"),
                ("The app crashes when I open settings.", "bug"),
                ("Charts are blank after the update, looks broken.", "bug"),
                ("What file formats can I import?", "question"),
                ("Does the API support webhooks?", "question"),
                ("Is there a mobile version?", "question"),
                ("Which browsers do you support?", "question"),
            ),
            "holdout": _cases(
                ("Cancel the renewal for my workspace.", "cancel"),
                ("Please stop billing me and close everything.", "cancel"),
                ("I want to cancel my trial.", "cancel"),
                ("Upgrade our workspace to premium.", "upgrade"),
                ("Add more seats to our plan.", "upgrade"),
                ("Can I upgrade in the middle of the month?", "upgrade"),
                ("The editor crashes on large files.", "bug"),
                ("I get an error when I sign in.", "bug"),
                ("Images look broken in the preview.", "bug"),
                ("Do you support single sign-on?", "question"),
                ("What formats can I export?", "question"),
                ("Is there a desktop version?", "question"),
            ),
        },
    },
    "reviews": {
        "name": "Product reviews",
        "description": "Label a review positive, negative or neutral.",
        "task": {
            "train": _cases(
                ("I love how fast it is.", "positive"),
                ("Great value and friendly support.", "positive"),
                ("Excellent build quality, would buy again.", "positive"),
                ("Works perfectly, love it.", "positive"),
                ("Absolutely great, five stars.", "positive"),
                ("It broke after two days.", "negative"),
                ("Terrible battery life, very disappointed.", "negative"),
                ("Refund please, it stopped working.", "negative"),
                ("The worst purchase I have made.", "negative"),
                ("Disappointed, the screen arrived broken.", "negative"),
                ("It arrived on Tuesday.", "neutral"),
                ("The box contains a cable and a charger.", "neutral"),
                ("It comes in blue and grey.", "neutral"),
                ("I bought it for my office.", "neutral"),
                ("The manual is twenty pages long.", "neutral"),
            ),
            "holdout": _cases(
                ("I love the colour.", "positive"),
                ("Great sound for the price.", "positive"),
                ("Excellent, works as described.", "positive"),
                ("Would buy again, great seller.", "positive"),
                ("Stopped working after a week.", "negative"),
                ("Terrible customer service.", "negative"),
                ("Broke on the first day, I want a refund.", "negative"),
                ("Very disappointed with the quality.", "negative"),
                ("It arrived in a small box.", "neutral"),
                ("The charger is sold separately.", "neutral"),
                ("It comes with a manual.", "neutral"),
                ("I use it at my office.", "neutral"),
            ),
        },
    },
    "languages": {
        "name": "Languages",
        "description": "Tell English, French and Spanish sentences apart.",
        "task": {
            "train": _cases(
                ("Where is the train station?", "english"),
                ("Thank you for the lovely dinner.", "english"),
                ("I would like a cup of tea.", "english"),
                ("The meeting starts at noon.", "english"),
                ("We walked along the river.", "english"),
                ("Où est la gare ?", "french"),
                ("Merci pour le dîner.", "french"),
                ("Je voudrais un café.", "french"),
                ("Le musée est fermé le lundi.", "french"),
                ("Nous avons marché le long de la rivière.", "french"),
                ("¿Dónde está la estación?", "spanish"),
                ("Gracias por la cena.", "spanish"),
                ("Quiero un café con leche.", "spanish"),
                ("El museo está cerrado el lunes.", "spanish"),
                ("Caminamos por el río.", "spanish"),
            ),
            "holdout": _cases(
                ("The library opens at nine.", "english"),
                ("Thank you for waiting.", "english"),
                ("I would like the soup.", "english"),
                ("Where is the museum?", "english"),
                ("Le train est en retard.", "french"),
                ("Je cherche la bibliothèque.", "french"),
                ("Merci beaucoup.", "french"),
                ("Où est le musée ?", "french"),
                ("El tren llega tarde.", "spanish"),
                ("Gracias por todo.", "spanish"),
                ("¿Dónde está el museo?", "spanish"),
                ("Quiero la sopa.", "spanish"),
            ),
        },
    },
}
LOOPS = tuple(template["id"] for template in TEMPLATES)
BUDGETS = (12, 24, 48)
SEEDS = tuple(range(20))
RESULTS_FILE = "benchmark_results.json"
DRIVEN = {
    "status": "not_measured",
    "note": (
        "The loops that improve your own prompt need a live model, so their results depend on the model. "
        "They have not been measured on a live model yet."
    ),
    "how": "python examples/lab_benchmark.py --model <openrouter model> (needs OPENROUTER_API_KEY)",
}


def run_cell(loop: str, task_id: str, budget: int, seed: int) -> dict:
    """One loop on one task: training and holdout scores of the start and the winner."""
    template = next(item for item in TEMPLATES if item["id"] == loop)
    config = {**template["config"], "budget": budget, "seed": seed, "task": TASKS[task_id]["task"]}
    registry = make_registry()
    result = engine.run(config, registry)
    start = engine.run({**config, "budget": 1}, registry)
    return {
        "train_gain": result["best_score"] - result["baseline_score"],
        "holdout": result["holdout_score"],
        "baseline_holdout": start["holdout_score"],
    }


def _round(value: float) -> float:
    return round(value, 4)


def measure(loops=LOOPS, task_ids=tuple(TASKS), budgets=BUDGETS, seeds=SEEDS) -> dict:
    """Run the grid and summarise each loop at each budget across every task and seed."""
    summary: dict[str, dict] = {}
    baselines: dict[str, float] = {}
    for loop in loops:
        summary[loop] = {}
        for budget in budgets:
            per_task, holdout_gains, train_gains, better, worse = {}, [], [], 0, 0
            for task_id in task_ids:
                gains = []
                for seed in seeds:
                    cell = run_cell(loop, task_id, budget, seed)
                    baselines[task_id] = cell["baseline_holdout"]
                    gain = cell["holdout"] - cell["baseline_holdout"]
                    gains.append(gain)
                    train_gains.append(cell["train_gain"])
                    better += gain > 0
                    worse += gain < 0
                per_task[task_id] = _round(fmean(gains))
                holdout_gains.extend(gains)
            runs = len(holdout_gains)
            summary[loop][str(budget)] = {
                "holdout_gain": _round(fmean(holdout_gains)),
                "train_gain": _round(fmean(train_gains)),
                "better_than_start": _round(better / runs),
                "worse_than_start": _round(worse / runs),
                "gain_per_version": _round(fmean(holdout_gains) / (budget - 1)),
                "per_task": per_task,
            }
    return {
        "description": (
            "Each offline loop ran on every task with every seed at each budget (versions scored, including the "
            "start). holdout_gain is the winner's holdout accuracy minus the starting prompt's, averaged over tasks "
            "and seeds; gain_per_version divides it by the versions tried after the start."
        ),
        "seeds": len(seeds),
        "budgets": list(budgets),
        "tasks": [
            {
                "id": task_id, "name": TASKS[task_id]["name"], "description": TASKS[task_id]["description"],
                "train_cases": len(TASKS[task_id]["task"]["train"]), "holdout_cases": len(TASKS[task_id]["task"]["holdout"]),
                "labels": len({row["expected"] for row in TASKS[task_id]["task"]["train"]}),
                "baseline_holdout": _round(baselines[task_id]),
            }
            for task_id in task_ids
        ],
        "loops": summary,
        "driven": dict(DRIVEN),
    }


@lru_cache(maxsize=1)
def _stored() -> str:
    return files("rsi_library").joinpath(RESULTS_FILE).read_text(encoding="utf-8")


def results() -> dict:
    """The published benchmark results."""
    return json.loads(_stored())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the offline loops on the benchmark tasks.")
    parser.add_argument("--write", action="store_true", help=f"rewrite rsi_library/{RESULTS_FILE}")
    args = parser.parse_args(argv)
    data = measure()
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.write:
        Path(__file__).with_name(RESULTS_FILE).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
