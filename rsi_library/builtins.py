"""Offline templates and components for the RSI Library workbench."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import re
from statistics import fmean

from .engine import (
    Case, ConfigError, Evaluation, EvaluationContext, Feedback, JSONValue, Judge, Proposal,
    ProposalContext, Registry, SearchContext, Settings, Target, TargetContext, Task,
    canonical_json, json_value, unit_score,
)


def _cases(*rows: tuple[str, str]) -> list[dict[str, str]]:
    return [{"input": message, "expected": label} for message, label in rows]


EXAMPLE_TASK = {
    "train": _cases(
        ("Could you resend my latest invoice?", "billing"),
        ("The invoice shows the wrong company name.", "billing"),
        ("I asked for a refund last Thursday.", "billing"),
        ("When will the refund reach my bank?", "billing"),
        ("A duplicate charge appeared yesterday.", "billing"),
        ("Please investigate this duplicate charge.", "billing"),
        ("How do I reset my password?", "account"),
        ("The password link expired before I could use it.", "account"),
        ("Change the email on my profile.", "account"),
        ("My profile still shows my old name.", "account"),
        ("My login code never arrived.", "account"),
        ("The login page keeps sending me back to the start.", "account"),
    ),
    "holdout": _cases(
        ("I need a copy of the April invoice.", "billing"),
        ("Is the refund already on its way?", "billing"),
        ("There is a duplicate charge on this statement.", "billing"),
        ("Could you explain this extra charge?", "billing"),
        ("Where can I update my profile picture?", "account"),
        ("The password reset message went to spam.", "account"),
        ("I forgot my login details while travelling.", "account"),
        ("My password suddenly stopped working.", "account"),
    ),
}


def _config(name: str, proposer: str, search: str) -> dict:
    return {
        "name": name,
        "target": "cue-prompt",
        "proposer": proposer,
        "judge": "exact",
        "search": search,
        "harness": "fixtures",
        "budget": 24,
        "seed": 13,
    }


TEMPLATES = (
    {
        "id": "greedy-prompt",
        "name": "Greedy prompt edits",
        "description": "Revise a small routing prompt and keep proposing from the highest scoring version.",
        "source_url": "https://arxiv.org/abs/2309.03409",
        "config": _config("Greedy prompt edits", "guided-edit", "greedy"),
    },
    {
        "id": "archive-prompt",
        "name": "Archive prompt search",
        "description": "Explore earlier prompt variants alongside the current best to try different routes to improvement.",
        "source_url": "https://arxiv.org/abs/2309.08532",
        "config": _config("Archive prompt search", "guided-edit", "archive"),
    },
    {
        "id": "adaptive-improver",
        "name": "Adaptive improver",
        "description": "Let each candidate carry and revise the edit policy its descendants will use.",
        "source_url": "https://arxiv.org/abs/2505.22954",
        "config": _config("Adaptive improver", "adaptive-edit", "archive"),
    },
)


@dataclass(frozen=True)
class FixtureHarness:
    balanced: bool = False

    def prepare(self, task: Mapping[str, JSONValue], settings: Settings) -> Task:
        if task.keys() - {"train", "holdout"}:
            raise ConfigError("Fixture task keys are train and holdout.")
        if len(canonical_json(dict(task))) > 128 * 1024:
            raise ConfigError("Keep the combined train and holdout fixtures under 128 KiB as JSON.")

        def split(name: str, min_cases: int) -> tuple[Case, ...]:
            rows = task.get(name, [])
            if not isinstance(rows, list) or not min_cases <= len(rows) <= 200:
                raise ConfigError(f"Task {name} must be an array of {min_cases} to 200 fixtures.")
            cases = []
            for i, row in enumerate(rows, 1):
                if not isinstance(row, dict) or row.keys() != {"input", "expected"}:
                    raise ConfigError(f"{name.title()} fixture {i} needs input and expected fields.")
                cases.append(Case(json_value(row["input"], "Fixture input"), json_value(row["expected"], "Expected output")))
            return tuple(cases)

        train, holdout = split("train", 1), split("holdout", 0)
        if {canonical_json(case.input) for case in train} & {canonical_json(case.input) for case in holdout}:
            raise ConfigError("Use different inputs for training and holdout fixtures.")
        return Task(train, holdout)

    def evaluate(
        self,
        candidate: JSONValue,
        cases: tuple[Case, ...],
        target: Target,
        judge: Judge,
        context: EvaluationContext,
    ) -> Evaluation:
        feedback = []
        by_label: dict[str, list[float]] = {}
        for case in cases:
            prediction = json_value(
                target.predict(deepcopy(candidate), deepcopy(case.input), deepcopy(context.target)),
                "Prediction",
            )
            score = unit_score(judge(deepcopy(prediction), deepcopy(case.expected), deepcopy(context.judge_settings)), "Judge score")
            feedback.append(Feedback(case.input, prediction, score))
            if self.balanced:
                by_label.setdefault(canonical_json(case.expected), []).append(score)
        score = fmean(fmean(scores) for scores in by_label.values()) if self.balanced else fmean(row.score for row in feedback)
        return Evaluation(score, tuple(feedback))


def exact(prediction: JSONValue, expected: JSONValue, settings: Settings) -> float:
    return float(canonical_json(prediction) == canonical_json(expected))


def normalized(prediction: JSONValue, expected: JSONValue, settings: Settings) -> float:
    if isinstance(prediction, str) and isinstance(expected, str):
        return float(" ".join(prediction.split()).casefold() == " ".join(expected.split()).casefold())
    return exact(prediction, expected, settings)


def greedy(context: SearchContext) -> int:
    return max(context.history, key=lambda trial: (trial.score, -trial.id)).id


def archive(context: SearchContext) -> int:
    exploration = unit_score(context.settings.get("exploration", 0.35), "Archive exploration")
    best = greedy(context)
    if len(context.history) == 1 or context.rng.random() >= exploration:
        return best
    seen = {canonical_json(context.history[best].candidate)}
    alternatives = []
    for trial in context.history:
        key = canonical_json(trial.candidate)
        if key not in seen:
            seen.add(key)
            alternatives.append(trial)
    if not alternatives:
        return best
    return context.rng.choices(alternatives, weights=[0.2 + trial.score for trial in alternatives])[0].id


_LITERAL = r'"(?:[^"\\]|\\.)*"'
_DEFAULT_LINE = re.compile(rf"Default to ({_LITERAL})\.")
_RULE_LINE = re.compile(rf"If text contains ({_LITERAL}), predict ({_LITERAL})\.")
_MAX_RULES = 32
_STOP_WORDS = set("a am an and are as at be by can could did do for from had has have how i in is it its me my of on or our please so than that the their this to was we were what when where which who will with would you your".split())


@lru_cache(maxsize=1024)
def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


@dataclass(frozen=True)
class CueState:
    fallback: str
    rules: tuple[tuple[str, str], ...]
    span: int = 1
    focus: str = "misses"


def _encode(state: CueState, as_prompt: bool) -> JSONValue:
    policy: JSONValue = {"span": state.span, "focus": state.focus}
    if as_prompt:
        def literal(value: str) -> str:
            return json.dumps(value, ensure_ascii=False)

        lines = [f"Default to {literal(state.fallback)}."]
        lines.extend(f"If text contains {literal(phrase)}, predict {literal(label)}." for phrase, label in state.rules)
        return {"prompt": "\n".join(lines), "policy": policy}
    return {
        "default": state.fallback,
        "rules": [{"phrase": phrase, "label": label} for phrase, label in state.rules],
        "policy": policy,
    }


def _decode(candidate: JSONValue, labels: tuple[JSONValue, ...], as_prompt: bool) -> CueState:
    if not isinstance(candidate, dict):
        raise ConfigError("A cue candidate must be an object.")
    if as_prompt:
        prompt = candidate.get("prompt")
        if candidate.keys() - {"prompt", "policy"} or not isinstance(prompt, str) or len(prompt) > 12_000:
            raise ConfigError("A cue prompt candidate needs prompt text and an optional policy.")
        lines = prompt.strip().split("\n")
        default = _DEFAULT_LINE.fullmatch(lines[0].strip()) if lines else None
        if not default:
            raise ConfigError('The first prompt line must say: Default to "label".')
        try:
            fallback = json.loads(default[1])
            rules = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                match = _RULE_LINE.fullmatch(line.strip())
                if not match:
                    raise ConfigError('Each rule must say: If text contains "phrase", predict "label".')
                rules.append((json.loads(match[1]), json.loads(match[2])))
        except json.JSONDecodeError as exc:
            raise ConfigError("Prompt phrases and labels must be valid quoted JSON strings.") from exc
    else:
        fallback, rows = candidate.get("default"), candidate.get("rules")
        if candidate.keys() - {"default", "rules", "policy"} or not isinstance(rows, list):
            raise ConfigError("A cue table candidate needs default and rules fields and an optional policy.")
        rules = []
        for row in rows:
            if not isinstance(row, dict) or row.keys() != {"phrase", "label"}:
                raise ConfigError("Every cue table rule needs phrase and label fields.")
            rules.append((row["phrase"], row["label"]))

    if not isinstance(fallback, str) or fallback not in labels:
        raise ConfigError("The default cue label must match a label in the training fixtures.")
    if len(rules) > _MAX_RULES:
        raise ConfigError(f"Cue candidates can have at most {_MAX_RULES} rules.")
    for phrase, label in rules:
        if not isinstance(phrase, str) or not 1 <= len(phrase) <= 100 or not _tokens(phrase):
            raise ConfigError("Every cue phrase needs text of 1 to 100 characters with at least one word.")
        if not isinstance(label, str) or label not in labels:
            raise ConfigError("Every cue rule label must match a label in the training fixtures.")
    policy = candidate.get("policy", {})
    if not isinstance(policy, dict) or policy.keys() - {"span", "focus"}:
        raise ConfigError("A cue policy can have span and focus settings.")
    span, focus = policy.get("span", 1), policy.get("focus", "misses")
    if type(span) is not int or not 1 <= span <= 3 or focus not in ("misses", "mixed"):
        raise ConfigError('Cue policy span must be 1, 2 or 3, and focus must be "misses" or "mixed".')
    return CueState(fallback, tuple(rules), span, focus)


@dataclass(frozen=True)
class CueTarget:
    as_prompt: bool
    uses_train_label_vocabulary = True

    def start(self, context: TargetContext) -> JSONValue:
        self._check_labels(context)
        fallback = context.settings.get("fallback", context.labels[0])
        if not isinstance(fallback, str) or fallback not in context.labels:
            raise ConfigError("The target fallback must match a training label.")
        return _encode(CueState(fallback, ()), self.as_prompt)

    def validate(self, candidate: JSONValue, context: TargetContext) -> None:
        self._check_labels(context)
        _decode(candidate, context.labels, self.as_prompt)

    def predict(self, candidate: JSONValue, input: JSONValue, context: TargetContext) -> JSONValue:
        if not isinstance(input, str) or not input.strip() or len(input) > 2000:
            raise ConfigError("Cue targets need text inputs of 1 to 2000 characters.")
        state = _decode(candidate, context.labels, self.as_prompt)
        text = " " + " ".join(_tokens(input)) + " "
        for phrase, label in state.rules:
            if " " + " ".join(_tokens(phrase)) + " " in text:
                return label
        return state.fallback

    @staticmethod
    def _check_labels(context: TargetContext) -> None:
        if not context.labels or not all(isinstance(label, str) and label.strip() and len(label) <= 100 for label in context.labels):
            raise ConfigError("Cue targets need text labels of 1 to 100 characters.")


@lru_cache(maxsize=512)
def _phrases(text: str, max_span: int) -> frozenset[str]:
    words = _tokens(text)
    return frozenset(
        " ".join(phrase)
        for span in range(1, max_span + 1)
        for start in range(len(words) - span + 1)
        if (phrase := words[start:start + span])
        and any(word not in _STOP_WORDS for word in phrase)
        and sum(map(len, phrase)) + span - 1 <= 100
    )


def _edit(context: ProposalContext, adaptive: bool) -> Proposal:
    candidate = context.parent.candidate
    if not isinstance(candidate, dict) or not ({"prompt", "rules"} & candidate.keys()):
        raise ConfigError("Cue edit proposers need a cue prompt or cue table target.")
    as_prompt = "prompt" in candidate
    parent = _decode(candidate, context.target.labels, as_prompt)
    rng = context.rng
    next_span, next_focus = parent.span, parent.focus
    changes: list[str] = []

    if adaptive and rng.random() < unit_score(context.settings.get("policy_rate", 0.32), "Policy mutation rate"):
        if rng.random() < 0.6:
            next_span = rng.choice([span for span in (1, 2, 3) if span != parent.span])
            changes.append(f"Set future cue length to up to {next_span} {'word' if next_span == 1 else 'words'}.")
        else:
            next_focus = "mixed" if parent.focus == "misses" else "misses"
            changes.append("Let descendants explore all training cases." if next_focus == "mixed" else "Let descendants focus on missed cases.")

    span, focus = (parent.span, parent.focus) if adaptive else (1, "misses")
    feedback = tuple(row for row in context.parent.feedback if isinstance(row.input, str))
    missed = tuple(row for row in feedback if row.score < 1)
    pool = feedback if focus == "mixed" and rng.random() < 0.3 else missed
    rules = list(parent.rules)

    if rules and (not pool or rng.random() < 0.08):
        phrase, _ = rules.pop(rng.randrange(len(rules)))
        changes.append(f"Remove cue {phrase!r}.")
    elif pool:
        row = rng.choice(pool)
        phrases = sorted(_phrases(row.input, span))
        if phrases:
            counts = Counter(phrase for failure in missed for phrase in _phrases(failure.input, span))
            good_counts = Counter(
                phrase for success in feedback if success.score == 1 for phrase in _phrases(success.input, span)
            )
            weights = [(1 + counts[phrase]) ** 3 / (1 + good_counts[phrase]) ** 2 for phrase in phrases]
            phrase = rng.choices(phrases, weights=weights)[0]
            alternatives = [label for label in context.target.labels if label != row.prediction]
            label = rng.choice(alternatives) if row.score < 1 and alternatives else row.prediction
            if isinstance(label, str) and label in context.target.labels:
                rules = [(cue, destination) for cue, destination in rules if cue != phrase]
                rules.insert(0, (phrase, label))
                changes.append(f"Route {phrase!r} to {label!r}.")

    child = CueState(parent.fallback, tuple(rules[:_MAX_RULES]), next_span, next_focus)
    return Proposal(_encode(child, as_prompt), " ".join(changes) or "Retest the parent; no usable new cue.")


def guided_edit(context: ProposalContext) -> Proposal:
    return _edit(context, adaptive=False)


def adaptive_edit(context: ProposalContext) -> Proposal:
    return _edit(context, adaptive=True)


def make_registry() -> Registry:
    """Create a fresh registry with the offline starter components."""
    registry = Registry(EXAMPLE_TASK)
    registry.register("target", "cue-prompt", CueTarget(as_prompt=True), name="Cue prompt", description="A local interpreter routes text using ordered instructions in a small prompt.")
    registry.register("target", "cue-table", CueTarget(as_prompt=False), name="Cue table", description="Route text with an editable table of phrases and destinations.")
    registry.register("proposer", "guided-edit", guided_edit, name="Guided cue edits", description="Use missed training cases to add or remove routing cues.")
    registry.register("proposer", "adaptive-edit", adaptive_edit, name="Adaptive cue edits", description="Revise cues and the inherited policy that controls how descendants pick edits.")
    registry.register("judge", "exact", exact, name="Exact match", description="Give credit when the predicted and expected JSON values match exactly.")
    registry.register("judge", "normalized", normalized, name="Normalized text match", description="Ignore capitalization and spacing differences in text answers.")
    registry.register("search", "greedy", greedy, name="Keep the best", description="Use the highest scoring trial as the next parent, keeping the earliest on ties.")
    registry.register("search", "archive", archive, name="Explore the archive", description="Usually use the best trial, but sometimes branch from a distinct earlier candidate.")
    registry.register("harness", "fixtures", FixtureHarness(), name="Fixture accuracy", description="Run every case and average the scores, giving each case equal weight.")
    registry.register("harness", "balanced-fixtures", FixtureHarness(balanced=True), name="Balanced fixture accuracy", description="Run every case and give each expected label equal weight in the aggregate score.")
    return registry


def catalog(registry: Registry | None = None) -> dict:
    """Describe the offline presets and all components available in the selected registry."""
    registry = registry if registry is not None else make_registry()
    components = registry.describe()
    ids = {kind: {entry["id"] for entry in entries} for kind, entries in components.items()}
    return {
        "templates": [deepcopy(preset) for preset in TEMPLATES if all(preset["config"][kind] in options for kind, options in ids.items())],
        "components": components,
        "example_task": deepcopy(registry.default_task),
    }
