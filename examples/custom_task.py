"""Extend RSI Library with a custom target and proposer.

From the repository root: python3 -m examples.custom_task
"""

import itertools
import json
import re
import unicodedata

from rsi_library import ConfigError, make_registry, run
from rsi_library.engine import Proposal


class SlugTarget:
    def start(self, context):
        return {"separator": "_", "fold_accents": False, "strip_articles": False}

    def validate(self, candidate, context):
        if (
            not isinstance(candidate, dict)
            or set(candidate) != {"separator", "fold_accents", "strip_articles"}
            or candidate["separator"] not in ("-", "_")
            or type(candidate["fold_accents"]) is not bool
            or type(candidate["strip_articles"]) is not bool
        ):
            raise ConfigError("A slug candidate needs a separator and two boolean policy options.")

    def predict(self, candidate, input, context):
        if not isinstance(input, str):
            raise ConfigError("Each slug input must be text.")
        text = input.casefold()
        if candidate["fold_accents"]:
            text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
        words = re.findall(r"[^\W_]+", text)
        if candidate["strip_articles"] and words and words[0] in {"a", "an", "the"}:
            words.pop(0)
        return candidate["separator"].join(words)


def edit_policy(context):
    """Try a new policy, preferring changes close to the selected parent."""
    parent = context.parent.candidate
    candidates = [
        {"separator": sep, "fold_accents": fold, "strip_articles": strip}
        for sep, fold, strip in itertools.product(("-", "_"), (True, False), (True, False))
    ]
    fresh = [candidate for candidate in candidates if all(t.candidate != candidate for t in context.history)]
    if not fresh:
        return Proposal(parent, note="All eight policies have been evaluated.")
    context.rng.shuffle(fresh)
    candidate = min(fresh, key=lambda option: sum(option[key] != parent[key] for key in parent))
    return Proposal(candidate, note="Change the slug formatting policy.")


TASK = {
    "train": [
        {"input": "Déjà vu", "expected": "deja-vu"},
        {"input": "The park", "expected": "park"},
        {"input": "2026 Field Guide", "expected": "2026-field-guide"},
        {"input": "FAQ", "expected": "faq"},
    ],
    "holdout": [
        {"input": "An Étude in Blue", "expected": "etude-in-blue"},
        {"input": "Maps & Routes", "expected": "maps-routes"},
    ],
}


def experiment():
    registry = make_registry()
    registry.register("target", "slug", SlugTarget(), name="Title slug", description="Convert a title into a URL slug.")
    registry.register("proposer", "edit-slug", edit_policy, name="Edit policy", description="Change slug formatting rules.")
    config = {
        "name": "Title slug policies",
        "target": "slug",
        "proposer": "edit-slug",
        "judge": "exact",
        "search": "greedy",
        "harness": "fixtures",
        "budget": 8,
        "seed": 23,
        "task": TASK,
    }
    return run(config, registry)


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2, ensure_ascii=False))
