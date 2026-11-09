import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rsi_library.gateway.api import Gateway
from rsi_library.gateway.methods import validate_request
from rsi_library.gateway.providers import load_models
from rsi_library.gateway.routing import Adapter, choose, load_adapters
from rsi_library.gateway.store import Store
from rsi_library.gateway.worker import Worker


def job():
    return {
        "method": "reflect-v1",
        "artifact": {"type": "prompt", "text": "Always return account."},
        "task": {
            "train": [{"input": "Please resend the invoice", "expected": "billing"}],
            "holdout": [{"input": f"Unique holdout {index}: why was I charged?", "expected": "billing"} for index in range(8)],
        },
        # One training pass for each version, one proposal, and both versions on eight holdout cases.
        "budget": {"max_calls": 19, "max_cost_microusd": 100_000, "max_seconds": 20},
        "max_trials": 1,
        "model": "stub",
        "objective": "exact",
    }


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions" or self.headers.get("Authorization") != "Bearer local-test-token":
            self.send_error(403)
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append(payload)
        system, user = [message["content"] for message in payload["messages"]]
        if "training evidence" in system:
            output = json.dumps({"text": "Return billing for invoices or charges."})
        else:
            output = "billing" if "Return billing" in system else "account"
        body = {"choices": [{"message": {"content": output}}]}
        if self.server.include_usage:
            body["usage"] = {"prompt_tokens": 20, "completion_tokens": 5}
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class GatewayRouting(unittest.TestCase):
    def test_auto_records_benchmark_and_pin_overrides_it(self):
        models = load_models(None, demo=True)
        payload = {**job(), "method": "auto", "model": "demo"}
        evidence = [{
            "benchmark": "prompt-baseline-v1", "method": "evolve-v1",
            "model_fingerprint": models["demo"].fingerprint, "artifact_type": "prompt",
            "objective": "exact", "runs": 3, "mean_gain": 0.25,
            "cost_microusd": 0, "calls": 5,
        }]
        adapters = load_adapters()
        selected = choose(validate_request(payload), adapters, models, evidence)
        self.assertEqual(selected["method"], "evolve-v1")
        self.assertEqual(selected["benchmark"], "prompt-baseline-v1")
        pinned = choose(validate_request({**payload, "method": "reflect-v1"}), adapters, models, evidence)
        self.assertEqual(pinned["method"], "reflect-v1")
        self.assertIsNone(pinned["benchmark"])
        chain = {**payload, "artifact": {
            "type": "prompt-chain", "steps": [{"name": "summarize", "text": "Summarize."}, {"name": "classify", "text": "Label."}],
        }, "budget": {**payload["budget"], "max_calls": 37}}
        self.assertNotEqual(choose(validate_request(chain), adapters, models, evidence)["method"], "self-revise-v1")
        private_job = validate_request({**payload, "data_policy": "zero-retention"})
        self.assertEqual(choose(private_job, adapters, models, evidence)["model"], "demo")
        with self.assertRaisesRegex(ValueError, "provider policy"):
            choose(private_job, adapters, {"demo": replace(models["demo"], retention="unspecified")}, evidence)

    def test_external_optimizer_only_gets_train_and_cannot_claim_a_holdout_score(self):
        models = load_models(None, demo=True)
        adapters = load_adapters()
        name = "test-optimizer-v1"

        def optimize(request, method, complete, check, *, call_capacity, progress):
            self.assertEqual(set(request["task"]), {"train"})
            self.assertEqual(call_capacity, 1)
            complete("external-train", [{"role": "user", "content": "test"}])
            return {
                "candidate": {"type": "prompt", "text": "Return account for any request."},
                "accepted": True,
                "candidate_holdout": 1.0,
            }

        adapters[name] = Adapter({"id": name, "artifact_types": ["prompt"]}, optimize, lambda request: 1, external=True)
        payload = {**job(), "method": name, "model": "demo", "budget": {**job()["budget"], "max_calls": 17}}
        request = validate_request(payload, adapters.keys())
        request["_routing"] = choose(request, adapters, models, [])
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "gateway.db")
            workspace = store.create_workspace("external")
            queued = store.submit(workspace, request, "external-1")
            self.assertTrue(Worker(store, adapters, models).once())
            result = store.get(workspace, queued["id"])
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["usage"]["calls"], 17)
        self.assertEqual(result["result"]["candidate_holdout"], 0)
        self.assertFalse(result["result"]["accepted"])
        self.assertEqual(result["result"]["evidence"]["verdict"], "not_improved")
        self.assertEqual(result["result"]["artifact"], request["artifact"])


class GatewayHTTP(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "gateway.db"
        self.store = Store(self.path)
        self.workspace = self.store.create_workspace("alpha", 500_000)
        self.key = self.store.create_key(self.workspace)["token"]
        self.reader = self.store.create_key(self.workspace, "read")["token"]
        other = self.store.create_workspace("beta")
        self.other = self.store.create_key(other)["token"]
        patcher = patch.dict(os.environ, {"RSI_TEST_MODEL_SECRET": "local-test-token"})
        patcher.start()
        self.addCleanup(patcher.stop)
        try:
            provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
        except PermissionError as exc:
            self.skipTest(f"Loopback listeners are restricted: {exc}")
        self.provider = provider
        provider.received = []
        provider.include_usage = True
        self._start(provider)
        model_config = self.path.with_suffix(".json")
        model_config.write_text(json.dumps({"models": [{
            "id": "stub", "name": "Local contract test", "base_url": f"http://127.0.0.1:{provider.server_port}/v1",
            "model": "fixture-model", "api_key_env": "RSI_TEST_MODEL_SECRET", "max_input_tokens": 4096,
            "max_output_tokens": 256, "input_microusd_per_million": 1_000_000,
            "output_microusd_per_million": 2_000_000,
        }]}), encoding="utf-8")
        self.models = load_models(str(model_config), allow_loopback=True)
        self.adapters = load_adapters()
        self.gateway = Gateway("127.0.0.1", 0, self.store, self.adapters, self.models, [])
        self._start(self.gateway)

    def _start(self, server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

    def _request(self, path, *, token=None, payload=None, idem=None):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if idem:
            headers["Idempotency-Key"] = idem
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(f"http://127.0.0.1:{self.gateway.server_port}{path}", data=data, headers=headers)
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_isolated_idempotent_jobs_verified_through_provider_and_api(self):
        with self.assertRaises(HTTPError) as no_key:
            self._request("/v1/catalog")
        with no_key.exception as response:
            self.assertEqual(response.code, 401)
        with self.assertRaises(HTTPError) as cannot_write:
            self._request("/v1/jobs", token=self.reader, payload=job(), idem="job-1")
        with cannot_write.exception as response:
            self.assertEqual(response.code, 403)
        code, queued = self._request("/v1/jobs", token=self.key, payload=job(), idem="job-1")
        self.assertEqual(code, 202)
        job_id = queued["id"]
        code, duplicate = self._request("/v1/jobs", token=self.key, payload=job(), idem="job-1")
        self.assertEqual((code, duplicate["id"]), (200, job_id))
        with self.assertRaises(HTTPError) as collision:
            self._request("/v1/jobs", token=self.key, payload={**job(), "seed": 999}, idem="job-1")
        with collision.exception as response:
            self.assertEqual(response.code, 409)
        with self.assertRaises(HTTPError) as hidden:
            self._request(f"/v1/jobs/{job_id}", token=self.other)
        with hidden.exception as response:
            self.assertEqual(response.code, 404)

        worker = Worker(Store(self.path), self.adapters, self.models)
        self.assertTrue(worker.once())
        self.assertFalse(worker.once())
        code, outcome = self._request(f"/v1/jobs/{job_id}", token=self.reader)
        self.assertEqual((code, outcome["state"]), (200, "succeeded"))
        self.assertEqual(outcome["result"]["baseline_holdout"], 0)
        self.assertEqual(outcome["result"]["candidate_holdout"], 1)
        self.assertTrue(outcome["result"]["accepted"])
        self.assertEqual(outcome["result"]["evidence"]["verdict"], "improved")
        self.assertEqual(outcome["result"]["evidence"]["p_value"], 1 / 256)
        self.assertEqual(outcome["usage"]["calls"], 19)
        self.assertEqual(outcome["usage"]["spent_microusd"], 19 * 30)
        self.assertEqual(outcome["usage"]["uncertain_microusd"], 0)
        proposals = [call for call in self.provider.received if "training evidence" in call["messages"][0]["content"]]
        self.assertEqual(len(proposals), 1)
        self.assertNotIn("Unique holdout", json.dumps(proposals[0]))
        _, listing = self._request("/v1/jobs", token=self.key)
        self.assertNotIn("task", listing["jobs"][0]["request"])
        self.assertNotIn("trials", listing["jobs"][0]["result"])
        self.assertEqual(listing["jobs"][0]["result"]["verdict"], "improved")
        _, catalog = self._request("/v1/catalog", token=self.reader)
        self.assertEqual(catalog["evidence_policy"]["min_holdout_cases"], 8)
        self.assertEqual(self._request("/health/ready")[0], 200)

    def test_unknown_charge_is_preserved_and_never_replayed(self):
        self.provider.include_usage = False
        _, queued = self._request("/v1/jobs", token=self.key, payload=job(), idem="unknown")
        worker = Worker(self.store, self.adapters, self.models)
        self.assertTrue(worker.once())
        self.assertFalse(Worker(Store(self.path), self.adapters, self.models).once())
        _, outcome = self._request(f"/v1/jobs/{queued['id']}", token=self.key)
        self.assertEqual(outcome["state"], "failed")
        self.assertEqual(outcome["usage"]["calls"], 1)
        self.assertEqual(outcome["usage"]["uncertain_microusd"], self.models["stub"].reservation)
        self.assertEqual(len(self.provider.received), 1)
        dispatch = self.store.ledger(self.workspace)[0]
        self.store.reconcile(dispatch["id"], 30)
        _, usage = self._request("/v1/usage", token=self.reader)
        self.assertEqual(usage["usage"]["spent_microusd"], 30)
        self.assertEqual(usage["usage"]["uncertain_microusd"], 0)

    def test_preflight_rejects_overlong_messages_without_spend(self):
        self.models["stub"] = replace(self.models["stub"], max_input_tokens=512)
        payload = job()
        payload["artifact"]["text"] = "a" * 600
        _, queued = self._request("/v1/jobs", token=self.key, payload=payload, idem="long-prompt")
        self.assertTrue(Worker(self.store, self.adapters, self.models).once())
        _, outcome = self._request(f"/v1/jobs/{queued['id']}", token=self.key)
        self.assertEqual(outcome["state"], "failed")
        self.assertIn("input ceiling", outcome["error"])
        self.assertEqual(outcome["usage"]["calls"], 0)
        self.assertEqual(self.provider.received, [])
