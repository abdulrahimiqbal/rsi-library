"""Run a bounded propose → evaluate → select loop with swappable components."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
import random
import re
from typing import Literal, Protocol, cast


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
Settings = Mapping[str, JSONValue]
Kind = Literal["target", "proposer", "judge", "search", "harness"]
KINDS: tuple[Kind, ...] = ("target", "proposer", "judge", "search", "harness")
MAX_EVALUATIONS = 100
MAX_SEED = 2**32 - 1


class ConfigError(ValueError):
    """A configuration, candidate, or component result cannot be used."""


class ProposalRejected(ConfigError):
    """A proposer produced something unusable, such as malformed model output.

    By default this ends the run like any other configuration error. A caller that
    meters proposals can record it as a rejected round instead (see ``Controls``).
    """


@dataclass(frozen=True)
class Case:
    input: JSONValue
    expected: JSONValue


@dataclass(frozen=True)
class Task:
    train: tuple[Case, ...]
    holdout: tuple[Case, ...] = ()


@dataclass(frozen=True)
class TargetContext:
    """Target options and seed; labels are present only for opt-in classifiers."""

    settings: Settings
    labels: tuple[JSONValue, ...]
    seed: int


class Target(Protocol):
    """Define what changes, its starting state, and how to run it on one input."""

    def start(self, context: TargetContext) -> JSONValue: ...

    def validate(self, candidate: JSONValue, context: TargetContext) -> None: ...

    def predict(
        self, candidate: JSONValue, input: JSONValue, context: TargetContext
    ) -> JSONValue: ...


Judge = Callable[[JSONValue, JSONValue, Settings], float]


@dataclass(frozen=True)
class Feedback:
    """Training observation made available to the proposer.

    Harnesses leave ``expected`` unset by default, so proposers learn only from scores.
    A harness may reveal training labels, as model-based prompt optimizers need them.
    Holdout labels never reach the proposer either way.
    """

    input: JSONValue
    prediction: JSONValue
    score: float
    expected: JSONValue = None


@dataclass(frozen=True)
class Evaluation:
    score: float
    feedback: tuple[Feedback, ...] = ()


@dataclass(frozen=True)
class EvaluationContext:
    target: TargetContext
    harness_settings: Settings
    judge_settings: Settings


class Harness(Protocol):
    """Prepare fixtures and evaluate a candidate on one split at a time."""

    def prepare(self, task: Mapping[str, JSONValue], settings: Settings) -> Task: ...

    def evaluate(
        self,
        candidate: JSONValue,
        cases: tuple[Case, ...],
        target: Target,
        judge: Judge,
        context: EvaluationContext,
    ) -> Evaluation: ...


@dataclass(frozen=True)
class Trial:
    id: int
    parent_id: int | None
    score: float
    best_score: float
    candidate: JSONValue
    note: str
    feedback: tuple[Feedback, ...]
    parent_ids: tuple[int, ...] = ()
    case_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class Proposal:
    candidate: JSONValue
    note: str = ""


@dataclass(frozen=True)
class ProposalContext:
    """The selected parents' feedback and the candidate/score history (without feedback).

    ``parent`` is the primary parent. ``parents`` lists every selected parent, primary
    first, for proposers that combine several candidates.
    """

    parent: Trial
    history: tuple[Trial, ...]
    rng: random.Random
    settings: Settings
    target: TargetContext
    parents: tuple[Trial, ...] = ()


@dataclass(frozen=True)
class SearchContext:
    """Candidate/score history with per-case scores, but no inputs or predictions."""

    history: tuple[Trial, ...]
    rng: random.Random
    settings: Settings


@dataclass(frozen=True)
class Controls:
    """Optional loop behaviour for callers that pay for each proposal and evaluation.

    With every option off (the default), a run behaves as documented in ``run``.
    """

    # Record a repeated candidate as a duplicate round instead of evaluating it again.
    skip_duplicates: bool = False
    # Record a ProposalRejected or a failed candidate validation as a rejected round.
    record_rejections: bool = False
    # Candidates with the same key share one training evaluation.
    evaluation_key: Callable[[JSONValue], str] | None = None
    # Called before each proposal and each new evaluation with {"stage", "round",
    # "best_score", "trace"}: the best training score so far (None before the baseline
    # is scored) and the rounds recorded so far, without candidates.
    observe: Callable[[dict], None] | None = None


Proposer = Callable[[ProposalContext], Proposal]
# A search returns the next parent's trial ID, or a sequence of up to four distinct IDs
# with the primary parent first.
Search = Callable[[SearchContext], "int | Sequence[int]"]
Component = Target | Proposer | Judge | Search | Harness
MAX_PARENTS = 4


@dataclass(frozen=True)
class _Entry:
    component: Component
    name: str
    description: str


class Registry:
    """Opt in to trusted Python components; run configurations only contain IDs and data."""

    def __init__(self, default_task: Mapping[str, JSONValue] | None = None) -> None:
        self.default_task = deepcopy(dict(default_task)) if default_task is not None else None
        self._entries: dict[Kind, dict[str, _Entry]] = {kind: {} for kind in KINDS}

    def register(
        self,
        kind: Kind,
        id: str,
        component: Component,
        *,
        name: str,
        description: str,
    ) -> None:
        if kind not in self._entries:
            raise ConfigError(f"Unknown component kind: {kind!r}.")
        if not isinstance(id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", id):
            raise ConfigError("Component IDs must start with a lowercase letter and use letters, digits, dashes, or underscores.")
        if id in self._entries[kind]:
            raise ConfigError(f"{kind.title()} {id!r} is already registered.")
        methods = {"target": ("start", "validate", "predict"), "harness": ("prepare", "evaluate")}.get(kind)
        valid = callable(component) if methods is None else all(
            callable(getattr(component, method, None)) for method in methods
        )
        if not valid:
            raise ConfigError(f"{id!r} does not implement the {kind} interface.")
        self._entries[kind][id] = _Entry(component, name, description)

    def resolve(self, kind: Kind, id: str) -> Component:
        if id not in self._entries[kind]:
            raise ConfigError(f"Unknown {kind}: {id!r}.")
        return self._entries[kind][id].component

    def describe(self) -> dict[str, list[dict[str, str]]]:
        return {
            kind: [
                {"id": id, "name": entry.name, "description": entry.description}
                for id, entry in entries.items()
            ]
            for kind, entries in self._entries.items()
        }


def canonical_json(value: JSONValue) -> str:
    """Canonical comparison and identity for JSON values, including candidate artifacts."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def json_value(value: object, label: str, depth: int = 0) -> JSONValue:
    if depth > 24:
        raise ConfigError(f"{label} is nested too deeply.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [json_value(item, label, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: json_value(item, label, depth + 1) for key, item in value.items()}
    raise ConfigError(f"{label} must contain only finite JSON values.")


def unit_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ConfigError(f"{label} must be a number between 0 and 1.")
    return float(value)


@dataclass(frozen=True)
class _Prepared:
    config: dict[str, JSONValue]
    target: Target
    proposer: Proposer
    judge: Judge
    search: Search
    harness: Harness
    task: Task
    initial: JSONValue
    budget: int
    seed: int
    settings: dict[str, dict[str, JSONValue]]
    context: EvaluationContext


def _prepare(raw: Mapping[str, object], registry: Registry) -> _Prepared:
    if not isinstance(raw, Mapping):
        raise ConfigError("A run configuration must be a JSON object.")
    config = cast(dict[str, JSONValue], json_value(dict(raw), "Run configuration"))
    unknown = config.keys() - {*KINDS, "name", "budget", "seed", "task", "settings", "initial_candidate"}
    if unknown:
        raise ConfigError(f"Unknown configuration field: {min(unknown)!r}.")

    name = config.setdefault("name", "Untitled loop")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ConfigError("Give the run a name of 1 to 100 characters.")
    config["name"] = name.strip()

    budget = config.get("budget")
    if type(budget) is not int or not 1 <= budget <= MAX_EVALUATIONS:
        raise ConfigError(f"Budget must be a whole number from 1 to {MAX_EVALUATIONS} evaluations.")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ConfigError(f"Seed must be a whole number from 0 to {MAX_SEED}.")

    components: dict[Kind, Component] = {}
    for kind in KINDS:
        id = config.get(kind)
        if not isinstance(id, str):
            raise ConfigError(f"Choose a {kind} component ID.")
        components[kind] = registry.resolve(kind, id)

    all_settings = config.get("settings", {})
    if not isinstance(all_settings, dict) or all_settings.keys() - set(KINDS):
        raise ConfigError("Settings must be an object with component kinds as keys.")
    settings: dict[str, dict[str, JSONValue]] = {}
    for kind in KINDS:
        kind_settings = all_settings.get(kind, {})
        if not isinstance(kind_settings, dict):
            raise ConfigError(f"Settings for {kind} must be an object.")
        settings[kind] = kind_settings

    task_data = config.get("task", registry.default_task)
    if not isinstance(task_data, dict):
        raise ConfigError("Task must be an object with train and optional holdout fixtures.")
    config["task"] = deepcopy(task_data)
    harness = cast(Harness, components["harness"])
    task = harness.prepare(deepcopy(task_data), deepcopy(settings["harness"]))
    if not isinstance(task, Task) or not task.train:
        raise ConfigError("The harness must return a task with at least one training case.")
    if not all(isinstance(case, Case) for case in (*task.train, *task.holdout)):
        raise ConfigError("The harness must return Case objects for training and holdout fixtures.")
    target = cast(Target, components["target"])
    labels = (
        {canonical_json(case.expected): case.expected for case in task.train}
        if getattr(target, "uses_train_label_vocabulary", False) is True else {}
    )
    target_context = TargetContext(settings["target"], tuple(labels[key] for key in sorted(labels)), seed)
    initial = config["initial_candidate"] if "initial_candidate" in config else target.start(deepcopy(target_context))
    initial = json_value(initial, "Starting candidate")
    target.validate(deepcopy(initial), deepcopy(target_context))

    return _Prepared(
        config, target, cast(Proposer, components["proposer"]), cast(Judge, components["judge"]),
        cast(Search, components["search"]), harness, task, initial, budget, seed, settings,
        EvaluationContext(target_context, settings["harness"], settings["judge"]),
    )


def _evaluate(prepared: _Prepared, candidate: JSONValue, cases: tuple[Case, ...]) -> Evaluation:
    result = prepared.harness.evaluate(
        deepcopy(candidate), deepcopy(cases), prepared.target, prepared.judge, deepcopy(prepared.context)
    )
    if not isinstance(result, Evaluation):
        raise ConfigError("The harness must return an Evaluation.")
    score = unit_score(result.score, "Evaluation score")
    if not isinstance(result.feedback, tuple) or not all(isinstance(row, Feedback) for row in result.feedback):
        raise ConfigError("Evaluation feedback must be a tuple of Feedback objects.")
    feedback = tuple(
        Feedback(json_value(row.input, "Feedback input"), json_value(row.prediction, "Feedback prediction"),
                 unit_score(row.score, "Feedback score"), json_value(row.expected, "Feedback expected output"))
        for row in result.feedback
    )
    return Evaluation(score, feedback)


def _parent_ids(choice: object, trials: Mapping[int, Trial]) -> tuple[int, ...]:
    ids = (choice,) if type(choice) is int else tuple(choice) if isinstance(choice, (list, tuple)) else ()
    if (
        not 1 <= len(ids) <= MAX_PARENTS or len(set(ids)) != len(ids)
        or not all(type(id) is int and id in trials for id in ids)
    ):
        raise ConfigError(f"Search must select 1 to {MAX_PARENTS} distinct earlier evaluated trial IDs as parents.")
    return ids


def _record(trial: Trial, status: str = "evaluated") -> dict:
    return {
        "id": trial.id,
        "parent_id": trial.parent_id,
        "parent_ids": list(trial.parent_ids),
        "status": status,
        "score": trial.score,
        "best_score": trial.best_score,
        "candidate": trial.candidate,
        "note": trial.note,
    }


def _run(prepared: _Prepared, controls: Controls = Controls()) -> dict:
    rng = random.Random(prepared.seed)
    cache: dict[str, Evaluation] = {}
    evaluations = 0
    records: list[dict] = []

    def observe(stage: str, round: int, best_score: float | None) -> None:
        if controls.observe is not None:
            trace = [
                {"id": record["id"], "parent_ids": list(record["parent_ids"]), "status": record["status"], "score": record["score"]}
                for record in records
            ]
            controls.observe({"stage": stage, "round": round, "best_score": best_score, "trace": trace})

    def evaluate(candidate: JSONValue, stage: str, round: int, best_score: float | None) -> Evaluation:
        nonlocal evaluations
        key = controls.evaluation_key(deepcopy(candidate)) if controls.evaluation_key else None
        if key is not None and key in cache:
            return cache[key]
        observe(stage, round, best_score)
        result = _evaluate(prepared, candidate, prepared.task.train)
        evaluations += 1
        if key is not None:
            cache[key] = result
        return result

    def scores(result: Evaluation) -> tuple[float, ...]:
        return tuple(row.score for row in result.feedback)

    baseline = evaluate(prepared.initial, "baseline_train", 0, None)
    best = Trial(0, None, baseline.score, baseline.score, prepared.initial, "Starting candidate", baseline.feedback,
                 (), scores(baseline))
    trials = {0: best}
    records.append(_record(best))
    seen = {canonical_json(prepared.initial)}
    for id in range(1, prepared.budget):
        # Search sees scores only; proposers also see the selected parents' training observations.
        history = tuple(replace(trial, candidate=deepcopy(trial.candidate), feedback=()) for trial in trials.values())
        parent_ids = _parent_ids(prepared.search(SearchContext(history, rng, deepcopy(prepared.settings["search"]))), trials)
        parents = tuple(deepcopy(trials[parent_id]) for parent_id in parent_ids)
        proposal_history = tuple(replace(trial, candidate=deepcopy(trial.candidate), feedback=()) for trial in trials.values())
        observe("propose", id, best.score)
        try:
            proposal = prepared.proposer(ProposalContext(
                parents[0], proposal_history, rng,
                deepcopy(prepared.settings["proposer"]), deepcopy(prepared.context.target), parents,
            ))
            if not isinstance(proposal, Proposal) or not isinstance(proposal.note, str):
                raise ConfigError("The proposer must return a Proposal with a text note.")
            candidate = json_value(proposal.candidate, "Proposed candidate")
            try:
                prepared.target.validate(deepcopy(candidate), deepcopy(prepared.context.target))
            except ConfigError as exc:
                raise ProposalRejected(str(exc)) from exc
        except ProposalRejected as exc:
            if not controls.record_rejections:
                raise
            records.append({
                "id": id, "parent_id": parent_ids[0], "parent_ids": list(parent_ids), "status": "rejected",
                "score": None, "best_score": best.score, "candidate": None, "note": str(exc),
            })
            continue
        key = canonical_json(candidate)
        if controls.skip_duplicates and key in seen:
            records.append({
                "id": id, "parent_id": parent_ids[0], "parent_ids": list(parent_ids), "status": "duplicate",
                "score": None, "best_score": best.score, "candidate": candidate, "note": proposal.note,
            })
            continue
        seen.add(key)
        result = evaluate(candidate, "candidate_train", id, best.score)
        trial = Trial(id, parent_ids[0], result.score, max(result.score, best.score), candidate, proposal.note,
                      result.feedback, parent_ids, scores(result))
        trials[id] = trial
        records.append(_record(trial))
        if trial.score > best.score:
            best = trial

    holdout = _evaluate(prepared, best.candidate, prepared.task.holdout) if prepared.task.holdout else None
    return {
        "config": prepared.config,
        "baseline_score": baseline.score,
        "best_score": best.score,
        "best_trial_id": best.id,
        "holdout_score": holdout.score if holdout else None,
        "best_candidate": best.candidate,
        "evaluations_used": evaluations,
        "holdout_evaluations": int(holdout is not None),
        "train_cases": len(prepared.task.train),
        "holdout_cases": len(prepared.task.holdout),
        "trials": records,
    }


def run(config: Mapping[str, object], registry: Registry, controls: Controls | None = None) -> dict:
    """Run ``budget - 1`` proposal rounds after evaluating the starting candidate.

    By default every round is evaluated, so the run spends exactly ``budget`` training
    evaluations, including the baseline, and a duplicate proposal still costs an
    evaluation. ``controls`` can make duplicate, rejected, or equivalent candidates
    skip evaluation instead. Ties keep the earliest best trial. Only that winner is
    evaluated on holdout, outside the search budget.
    """
    return _run(_prepare(config, registry), controls or Controls())


def compare(configs: Sequence[Mapping[str, object]], registry: Registry) -> dict:
    """Run variants on the same task, starting candidate, seed and evaluation budget."""
    if not isinstance(configs, (list, tuple)) or not 2 <= len(configs) <= 5:
        raise ConfigError("Compare 2 to 5 run configurations at a time.")
    prepared = [_prepare(config, registry) for config in configs]
    first = prepared[0]
    for variant in prepared[1:]:
        for field in ("target", "judge", "harness", "budget", "seed", "task"):
            if canonical_json(variant.config[field]) != canonical_json(first.config[field]):
                raise ConfigError(f"Comparison runs must use the same {field}.")
        for kind in ("target", "judge", "harness"):
            if canonical_json(variant.settings[kind]) != canonical_json(first.settings[kind]):
                raise ConfigError(f"Comparison runs must use the same {kind} settings.")
        if canonical_json(variant.initial) != canonical_json(first.initial):
            raise ConfigError("Comparison runs must use the same starting candidate.")
    return {"runs": [_run(item) for item in prepared]}
