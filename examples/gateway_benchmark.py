"""Run a serial, paired benchmark through the RSI Gateway API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4


METHODS = ("reflect-v1", "evolve-v1")
TERMINAL = {"succeeded", "failed", "cancelled"}
MIN_HOLDOUT, MAX_HOLDOUT = 8, 64
DEMO = {
    "artifact": {"type": "prompt", "text": "Classify every ticket as account. Return only the label."},
    "task": {
        "train": [
            {"input": "Where is my invoice?", "expected": "billing"},
            {"input": "My password stopped working.", "expected": "account"},
        ],
        "holdout": [
            {"input": "I need to check a refund.", "expected": "billing"},
            {"input": "Please resend last month's invoice.", "expected": "billing"},
            {"input": "Why is there a second charge on my card?", "expected": "billing"},
            {"input": "Can billing add our tax number?", "expected": "billing"},
            {"input": "The refund still has not arrived.", "expected": "billing"},
            {"input": "How can I change my profile photo?", "expected": "account"},
            {"input": "I can't sign in after resetting my password.", "expected": "account"},
            {"input": "Please update the email on my profile.", "expected": "account"},
        ],
    },
}
# The simulator always answers "account" for the original and labels every case correctly
# for its fixed proposal, so 5 of 8 holdout cases improve: p = 1/32.
SMOKE_RESULT = (0.375, 1.0)


class BenchmarkError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def origin(value: str) -> str:
    url = urlsplit(value)
    local = url.hostname in {"127.0.0.1", "::1", "localhost"}
    if (
        url.scheme != "https" and not (url.scheme == "http" and local)
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query or url.fragment or url.path not in {"", "/"}
    ):
        raise BenchmarkError("Use an HTTPS gateway origin, or HTTP for a loopback gateway. Omit credentials and paths.")
    return f"{url.scheme}://{url.netloc}"


def workspace_key(path: Path | None) -> str:
    if path is None:
        value = os.environ.get("RSI_WORKSPACE_KEY", "")
    else:
        with path.open(encoding="utf-8") as source:
            permissions = os.fstat(source.fileno()).st_mode
            if not stat.S_ISREG(permissions) or permissions & 0o077:
                raise BenchmarkError("Use a regular key file that is accessible only to its owner, for example mode 600.")
            value = source.read(257)
    value = value.strip()
    if not re.fullmatch(r"[!-~]{13,249}", value):
        raise BenchmarkError("Set RSI_WORKSPACE_KEY or provide a private --key-file containing a valid workspace write key.")
    return value


class Gateway:
    def __init__(self, base: str, key: str):
        self.base = base
        self.key = key
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, path: str, payload: dict | None = None, idem: str | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json"}
        body = None
        if payload is not None:
            headers.update({"Content-Type": "application/json", "Idempotency-Key": idem})
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(self.base + path, data=body, headers=headers)
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=15) as response:
                    payload = json.loads(response.read(4_000_001))
                if not isinstance(payload, dict):
                    raise BenchmarkError("The gateway returned an unexpected response.")
                return payload
            except HTTPError as error:
                status = error.code
                error.close()
                if status not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    hint = {
                        401: "Check the workspace key.",
                        403: "Use a workspace write key.",
                        404: "Check the gateway origin and model deployment.",
                        409: "This run ID conflicts with a different request. Use the original options or choose a new run ID.",
                    }.get(status, "Check readiness, model selection, and budget.")
                    raise BenchmarkError(f"Gateway request returned HTTP {status}. {hint}") from None
            except (TimeoutError, URLError, ConnectionError):
                if attempt == 2:
                    raise BenchmarkError("The gateway could not be reached. Retry with the same run ID and options.") from None
            except (UnicodeError, ValueError):
                raise BenchmarkError("The gateway returned invalid JSON.") from None
            time.sleep(2 ** attempt)
        raise BenchmarkError("The gateway did not complete the request.")


def microdollars(value: str) -> int:
    if not re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]{1,6})?", value):
        raise argparse.ArgumentTypeError("Enter a US dollar amount with up to six decimal places.")
    dollars, _, fraction = value.partition(".")
    return int(dollars) * 1_000_000 + int(fraction.ljust(6, "0"))


def dollars(value: int) -> str:
    return f"${value // 1_000_000}.{value % 1_000_000:06d}"


def run_job(gateway: Gateway, request: dict, idem: str) -> dict:
    job = gateway.request("/v1/jobs", request, idem)
    job_id = job.get("id", "")
    if not isinstance(job_id, str) or not re.fullmatch(r"job_[a-f0-9]{32}", job_id):
        raise BenchmarkError("The gateway returned an invalid job ID.")
    deadline = time.monotonic() + request["budget"]["max_seconds"] + 60
    while job.get("state") not in TERMINAL:
        if time.monotonic() >= deadline:
            raise BenchmarkError(f"Job {job_id} is still pending. Inspect it in the workbench or rerun with the same run ID.")
        time.sleep(1)
        job = gateway.request(f"/v1/jobs/{job_id}")
    return job


def outcome(job: dict, seed: int, method: str) -> dict:
    result, usage = job.get("result") or {}, job["usage"]
    baseline = result.get("baseline_holdout")
    candidate = result.get("candidate_holdout")
    evidence = result.get("evidence") or {}
    return {
        "id": job["id"], "seed": seed, "method": method, "state": job["state"],
        "baseline": baseline, "candidate": candidate, "accepted": result.get("accepted"),
        "gain_pp": 100 * (candidate - baseline) if candidate is not None and baseline is not None else None,
        "verdict": evidence.get("verdict"), "p_value": evidence.get("p_value"),
        "calls": usage["calls"], "spent_microusd": usage["spent_microusd"],
        "uncertain_microusd": usage["uncertain_microusd"],
        "model_fingerprint": job.get("request", {}).get("_routing", {}).get("model_fingerprint"),
    }


def print_row(row: dict) -> None:
    score = lambda value: f"{value:7.1%}" if value is not None else "      —"
    gain = f'{row["gain_pp"]:+7.1f}' if row["gain_pp"] is not None else "      —"
    p_value = f'{row["p_value"]:7.4f}' if row["p_value"] is not None else "      —"
    verdict = (row["verdict"] or "—").replace("_", " ")
    print(
        f'{row["seed"]:>10}  {row["method"]:<11}  {score(row["baseline"])}  {score(row["candidate"])}  {gain}  '
        f'{p_value}  {verdict:<13}  {row["calls"]:>5}  {dollars(row["spent_microusd"]):>12}  '
        f'{dollars(row["uncertain_microusd"]):>12}  {row["state"]}  {row["id"]}',
        flush=True,
    )


def evidence_rows(rows: list[dict], seeds: list[int], *, benchmark: str, artifact_type: str, objective: str) -> list[dict]:
    """Summarize each method as routing evidence: the gain users actually keep, per seed."""
    entries = []
    for method in METHODS:
        runs = [row for row in rows if row["method"] == method]
        if len(runs) != len(seeds) or len(runs) < 3:
            raise BenchmarkError("Routing evidence needs one finished job per seed and at least three seeds.")
        kept = [(row["candidate"] - row["baseline"]) if row["accepted"] else 0.0 for row in runs]
        entries.append({
            "benchmark": benchmark, "method": method, "model_fingerprint": runs[0]["model_fingerprint"],
            "artifact_type": artifact_type, "objective": objective, "runs": len(runs),
            "mean_gain": round(sum(kept) / len(kept), 6),
            "cost_microusd": max(row["spent_microusd"] for row in runs),
            "calls": max(row["calls"] for row in runs),
        })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766", help="gateway origin")
    parser.add_argument("--key-file", type=Path, help="private workspace key file, or set RSI_WORKSPACE_KEY")
    parser.add_argument("--smoke", action="store_true", help="test both methods with the zero-spend simulator")
    parser.add_argument("--model", help="explicit catalog model ID for a capability benchmark")
    parser.add_argument("--job-cap", type=microdollars, help="maximum US dollars per capability job")
    parser.add_argument("--seeds", nargs="+", type=int, help="paired seeds, default 11 22 33")
    parser.add_argument("--trials", type=int, help="proposal limit per job, default 4 for capability or 2 for smoke")
    parser.add_argument("--seconds", type=int, default=300, help="gateway runtime cap per job")
    parser.add_argument("--holdout-repeats", type=int, default=1, help="tries per holdout case, 1-5")
    parser.add_argument("--dataset", type=Path, help="JSON file with artifact and task fields")
    parser.add_argument("--run-id", default=uuid4().hex, help="reuse with the same options to resume without duplicate jobs")
    parser.add_argument("--output", type=Path, help="save report JSON, without credentials, fixtures, or artifacts")
    parser.add_argument("--evidence-out", type=Path, help="write routing evidence for the gateway's auto route (3+ seeds)")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9._-]{1,72}", args.run_id):
        parser.error("--run-id must contain 1–72 letters, digits, dots, underscores, or dashes")
    if not 1 <= args.seconds <= 3600:
        parser.error("--seconds must be between 1 and 3600")
    if not 1 <= args.holdout_repeats <= 5:
        parser.error("--holdout-repeats must be between 1 and 5")
    if args.smoke and any(option is not None for option in (args.model, args.job_cap, args.seeds, args.dataset, args.evidence_out)):
        parser.error("--smoke supplies its own demo model, zero budget, seed, and dataset, and produces no routing evidence")
    if not args.smoke and (args.model is None or args.job_cap is None or args.job_cap > 1_000_000_000):
        parser.error("a capability benchmark needs --model and an explicit --job-cap of at most $1000")
    seeds = [11] if args.smoke else args.seeds or [11, 22, 33]
    trials = args.trials if args.trials is not None else (2 if args.smoke else 4)
    if not 1 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error("choose up to 10 distinct seeds between 0 and 4294967295")
    if not 1 <= trials <= 50:
        parser.error("--trials must be between 1 and 50")

    try:
        return benchmark(args, seeds, trials)
    except BenchmarkError as error:
        print(f"Benchmark stopped. {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        print("Benchmark stopped. Check the dataset, key file, output path, and gateway response.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"Interrupted. Any submitted job remains subject to its budget. Resume with --run-id {args.run_id}.", file=sys.stderr)
        return 130


def benchmark(args, seeds: list[int], trials: int) -> int:
    if args.output and args.output.exists():
        raise BenchmarkError("The output report already exists. Choose a new path or omit --output.")
    if args.evidence_out and args.evidence_out.exists():
        raise BenchmarkError("The evidence file already exists. Choose a new path or omit --evidence-out.")
    if args.evidence_out and len(seeds) < 3:
        raise BenchmarkError("Routing evidence needs at least three seeds, for example --seeds 11 22 33.")
    model_id, cap = ("demo", 0) if args.smoke else (args.model, args.job_cap)
    dataset = args.dataset or Path(__file__).with_name("gateway_benchmark_tickets.json")
    fixture = DEMO if args.smoke else json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict) or set(fixture) != {"artifact", "task"}:
        raise BenchmarkError("Use a dataset JSON object with artifact and task fields.")
    train, holdout = (len(fixture["task"][split]) for split in ("train", "holdout"))
    if fixture["artifact"]["type"] == "prompt":
        steps = 1
    elif fixture["artifact"]["type"] == "prompt-chain":
        steps = len(fixture["artifact"]["steps"])
    else:
        raise BenchmarkError("The dataset needs a prompt or prompt-chain artifact.")
    if not 1 <= train <= 32 or not MIN_HOLDOUT <= holdout <= MAX_HOLDOUT or not 1 <= steps <= 4:
        raise BenchmarkError(f"Use 1–32 training cases, {MIN_HOLDOUT}–{MAX_HOLDOUT} holdout cases, and at most four prompt steps.")
    calls = steps * ((trials + 1) * train + 2 * holdout * args.holdout_repeats) + trials
    if calls > 1000:
        raise BenchmarkError("This dataset and trial count exceed the 1000-call gateway limit. Lower --trials.")

    gateway = Gateway(origin(args.url), workspace_key(args.key_file))
    if gateway.request("/health/ready").get("status") != "ready":
        raise BenchmarkError("The gateway worker is not ready.")
    catalog = gateway.request("/v1/catalog")
    if catalog.get("role") != "write":
        raise BenchmarkError("Use a workspace write key.")
    models = {item["id"]: item for item in catalog["models"]}
    if model_id not in models:
        raise BenchmarkError("This model ID is not in the gateway catalog.")
    model = models[model_id]
    if bool(model.get("demo")) != args.smoke:
        raise BenchmarkError("Use --smoke for the simulator or select a live catalog model for a capability benchmark.")
    available_methods = {item["id"] for item in catalog["methods"]}
    if not set(METHODS) <= available_methods:
        raise BenchmarkError("Both benchmark methods must be enabled on the gateway.")
    required_cap = calls * model["max_call_cost_microusd"]
    if required_cap > cap:
        raise BenchmarkError(f"This model reserves up to {dollars(required_cap)} for {calls} calls. Raise --job-cap or lower --trials.")

    fixture_hash = hashlib.sha256(json.dumps(fixture, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    budget = {"max_calls": calls, "max_cost_microusd": cap, "max_seconds": args.seconds}
    total = 2 * len(seeds) * cap
    print(f"Run ID {args.run_id}  |  {'Simulator smoke' if args.smoke else 'Capability benchmark'}  |  model {model_id}")
    print(f"Fixture {fixture_hash[:12]}  |  {train} training + {holdout} holdout × {args.holdout_repeats}  |  paired seeds: {len(seeds)}, serial")
    print(f"Caps {calls} calls and {dollars(cap)} per job, {dollars(total)} across this run.")
    print(f"{'Seed':>10}  {'Method':<11}  {'Before':>7}  {'After':>7}  {'Gain pp':>7}  {'p':>7}  {'Verdict':<13}  {'Calls':>5}  {'Spent':>12}  {'Uncertain':>12}  State and job ID", flush=True)
    rows = []
    fingerprint = None
    for index, seed in enumerate(seeds):
        for method in (METHODS if index % 2 == 0 else tuple(reversed(METHODS))):
            request = {**fixture, "method": method, "model": model_id, "seed": seed, "budget": budget,
                       "max_trials": trials, "objective": "normalized", "holdout_repeats": args.holdout_repeats}
            row = outcome(run_job(gateway, request, f"bench-{args.run_id}-{seed}-{method}"), seed, method)
            rows.append(row)
            print_row(row)
            if fingerprint is not None and row["model_fingerprint"] != fingerprint:
                raise BenchmarkError("The model configuration changed mid-run. Keep these results separate.")
            fingerprint = row["model_fingerprint"]
            if row["state"] != "succeeded" or row["uncertain_microusd"]:
                raise BenchmarkError("A job failed or has uncertain charges. Inspect it in the workbench before continuing.")

    paired = {(row["seed"], row["method"]): row for row in rows}
    differences = [paired[seed, METHODS[1]]["gain_pp"] - paired[seed, METHODS[0]]["gain_pp"] for seed in seeds]
    wins = sum(delta > 1e-9 for delta in differences)
    losses = sum(delta < -1e-9 for delta in differences)
    spent = sum(row["spent_microusd"] for row in rows)
    print(f"Evolution vs reflection on paired gains: {wins} wins, {losses} losses, {len(seeds) - wins - losses} ties. Total settled spend {dollars(spent)}.")
    for method in METHODS:
        improved = sum(row["verdict"] == "improved" for row in rows if row["method"] == method)
        print(f"{method}: significant improvement in {improved} of {len(seeds)} jobs.")
    if args.output:
        report = {"run_id": args.run_id, "model": model_id, "smoke": args.smoke, "fixture_sha256": fixture_hash,
                  "budget_per_job": budget, "total_cap_microusd": total, "rows": rows}
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(report, target, indent=2, allow_nan=False)
            target.write("\n")
    if args.evidence_out:
        name = (args.dataset.stem if args.dataset else "gateway-benchmark-tickets")[:48]
        entries = evidence_rows(
            rows, seeds, benchmark=f"{name}-{fixture_hash[:12]}",
            artifact_type=fixture["artifact"]["type"], objective="normalized",
        )
        with args.evidence_out.open("x", encoding="utf-8") as target:
            json.dump(entries, target, indent=2, allow_nan=False)
            target.write("\n")
        print(f"Wrote routing evidence for {len(entries)} methods to {args.evidence_out}.")
    if args.smoke and not all(
        (row["baseline"], row["candidate"]) == SMOKE_RESULT and row["accepted"] and row["verdict"] == "improved"
        and row["spent_microusd"] == 0 for row in rows
    ):
        raise BenchmarkError("The simulator did not reproduce its expected 37.5% → 100% significant result at zero spend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
