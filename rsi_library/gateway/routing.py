"""Route normalized requests to a versioned method and an available model."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from . import methods
from .providers import Model


@dataclass(frozen=True)
class Adapter:
    metadata: dict
    execute: Callable
    minimum_calls: Callable
    external: bool = False

    def required_calls(self, request: dict) -> int:
        if not self.external:
            return self.minimum_calls(request)
        required = self.minimum_calls(training_request(request, request["budget"]["max_calls"]))
        if type(required) is not int or required < 0:
            raise ValueError("An external optimizer declared an invalid call requirement.")
        return required + methods.verification_calls(request)


def training_request(request: dict, capacity: int) -> dict:
    result = deepcopy({key: value for key, value in request.items() if key not in {"task", "_routing"}})
    result["task"] = {"train": deepcopy(request["task"]["train"])}
    result["budget"]["max_calls"] = max(0, capacity - methods.verification_calls(request))
    return result


def load_adapters(modules: str = "") -> dict[str, Adapter]:
    adapters = {
        entry["id"]: Adapter(entry, methods.execute, methods.minimum_calls)
        for entry in methods.catalog()
    }
    for name in filter(None, (module.strip() for module in modules.split(","))):
        plugin = importlib.import_module(name)
        entry = plugin.METHOD
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", entry["id"])
            or entry["id"] in {*adapters, "auto"}
            or not all(isinstance(entry.get(field), str) for field in ("name", "description"))
            or not isinstance(entry.get("artifact_types"), list)
            or not entry["artifact_types"]
            or not all(kind in ("prompt", "prompt-chain") for kind in entry["artifact_types"])
            or not callable(plugin.execute) or not callable(plugin.minimum_calls)
        ):
            raise ValueError(f"Invalid adapter manifest in {name!r}.")
        adapters[entry["id"]] = Adapter(entry, plugin.execute, plugin.minimum_calls, external=True)
    return adapters


def load_evidence(path: str | None, *, config_json: str | None = None) -> list[dict]:
    """Load benchmark rows for the auto route from a file or, for hosted deployments, JSON text."""
    if path and config_json:
        raise ValueError("Configure routing evidence using a file or environment JSON, not both.")
    if not path and not config_json:
        return []
    entries = json.loads(config_json if config_json else Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Routing evidence must be an array.")
    for row in entries:
        if (
            not isinstance(row, dict)
            or not all(isinstance(row.get(key), str) and row[key] for key in
                       ("benchmark", "method", "model_fingerprint", "artifact_type", "objective"))
            or type(row.get("runs")) is not int or row["runs"] < 3
            or type(row.get("mean_gain")) not in {int, float} or not -1 <= row["mean_gain"] <= 1
            or type(row.get("cost_microusd")) is not int or row["cost_microusd"] < 0
            or type(row.get("calls")) is not int or row["calls"] < 1
        ):
            raise ValueError("Each routing evidence row needs a benchmark, at least three runs, and measured gain, calls, and cost.")
    return entries


def call_capacity(request: dict, model: Model) -> int:
    cap = request["budget"]["max_calls"]
    if model.reservation:
        cap = min(cap, request["budget"]["max_cost_microusd"] // model.reservation)
    return cap


def choose(request: dict, adapters: dict[str, Adapter], models: dict[str, Model], evidence: list[dict]) -> dict:
    wanted, artifact = request["method"], request["artifact"]
    allowed_builtin = set(methods.compatible_methods(request))
    candidates = []
    for method_id, adapter in adapters.items():
        if wanted != "auto" and wanted != method_id:
            continue
        if artifact["type"] not in adapter.metadata["artifact_types"]:
            continue
        if not adapter.external and method_id not in allowed_builtin:
            continue
        for model in models.values():
            if request.get("model", model.id) != model.id:
                continue
            if request.get("data_policy") == "zero-retention" and model.retention != "zero-retention":
                continue
            if call_capacity(request, model) < adapter.required_calls(request):
                continue
            candidates.append((method_id, model))
    if not candidates:
        raise ValueError("No configured optimizer and model satisfy this job's artifact, budget, and provider policy.")
    rows = []
    for method_id, model in candidates:
        for row in evidence:
            if (
                row["method"] == method_id and row["model_fingerprint"] == model.fingerprint
                and row["artifact_type"] == artifact["type"] and row["objective"] == request["objective"]
                and row["calls"] <= call_capacity(request, model)
                and row["cost_microusd"] <= request["budget"]["max_cost_microusd"]
            ):
                rows.append((row, method_id, model))
    if wanted == "auto" and rows:
        row, method_id, model = max(rows, key=lambda item: (
            item[0]["mean_gain"], -item[0]["cost_microusd"], item[0]["runs"], item[1], item[2].id,
        ))
        reason, benchmark = "Best measured gain in configured benchmarks within this job's limits.", row["benchmark"]
    else:
        method_id, model = min(candidates, key=lambda item: (item[1].demo, item[0] != "reflect-v1", item[1].reservation, item[0], item[1].id))
        reason = "Caller selected this optimizer." if wanted != "auto" else "Compatible default. No matching benchmark evidence is configured."
        benchmark = None
    return {
        "method": method_id, "model": model.id, "model_fingerprint": model.fingerprint,
        "reason": reason, "benchmark": benchmark, "reserved_per_call_microusd": model.reservation,
    }
