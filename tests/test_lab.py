import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rsi_library.gateway import lab as lab_module, mcp
from rsi_library.gateway.api import Gateway
from rsi_library.gateway.lab import Lab, LabError
from rsi_library.gateway.methods import execute, validate_request
from rsi_library.gateway.providers import load_models
from rsi_library.gateway.routing import load_adapters
from rsi_library.gateway.store import Store

LABELS = {"refund": "billing", "invoice": "billing", "charge": "billing", "password": "account", "login": "account", "profile": "account"}
TRAIN = [{"input": f"train {word} question {index}", "expected": label} for index, (word, label) in enumerate(LABELS.items())]
HOLDOUT = [
    {"input": f"holdout {word} case {index}", "expected": label}
    for index, (word, label) in enumerate(list(LABELS.items()) + [("refund", "billing"), ("login", "account")])
]


def model(system: str, text: str) -> str:
    """A stand-in model: version N of the prompt knows the first N keywords."""
    if system.startswith("Improve a model artifact"):
        parents = json.loads(text)["parents"]
        best = parents[0]["artifact"]
        if best["type"] == "prompt-chain":
            steps = [dict(step) for step in best["steps"]]
            steps[-1]["text"] = "route v6"
            return json.dumps({"steps": steps})
        level = int(best["text"].rsplit("v", 1)[-1])
        if level >= 6:
            return "this is not JSON"
        reply = {"text": f"route v{min(6, level + 3)}"}
        if "improver" in system:
            reply["improver"] = f"policy after v{level}"
        return json.dumps(reply)
    if system == "extract":
        return text
    if text.startswith("{"):
        text = json.loads(text)["previous"]
    level = int(system.rsplit("v", 1)[-1])
    for word, label in list(LABELS.items())[:level]:
        if word in text:
            return label.upper()
    return "unsure"


def drive(lab: Lab, view: dict) -> tuple[dict, int]:
    rounds = 0
    while view["state"] == "needs_work":
        rounds += 1
        outputs = [{"id": item["id"], "output": model(item["system"], item["input"])} for item in view["work"]]
        view = lab.submit(view["id"], outputs)
    return view, rounds


class PublicLoops(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.lab = Lab(Path(self.directory.name) / "lab.db")

    def tearDown(self):
        self.directory.cleanup()

    def test_catalog_lists_both_kinds_of_loop_with_their_parts(self):
        loops = {loop["id"]: loop for loop in lab_module.catalog()["loops"]}
        self.assertEqual([loop["mode"] for loop in loops.values()], ["offline"] * 3 + ["driven"] * 3)
        for loop in loops.values():
            self.assertEqual(set(loop["parts"]), {"target", "proposer", "judge", "search", "harness"})
        self.assertEqual(loops["evolve-v1"]["parts"]["search"]["id"], "complementary")
        self.assertIn("/api/runs", lab_module.agent_guide("https://example.test"))

    def test_offline_run_and_compare_accept_loop_ids(self):
        result = lab_module.offline_run({"loop": "greedy-prompt", "budget": 6, "seed": 2})
        self.assertEqual(len(result["trials"]), 6)
        self.assertEqual(result["config"]["search"], "greedy")
        swapped = lab_module.offline_run({"loop": "greedy-prompt", "budget": 6, "search": "archive"})
        self.assertEqual(swapped["config"]["search"], "archive")
        compared = lab_module.offline_compare({"loops": "all", "budget": 8})
        self.assertEqual(len(compared["runs"]), 3)
        self.assertEqual(len(compared["summary"]["scores"]), 3)
        with self.assertRaisesRegex(LabError, "POST /api/runs"):
            lab_module.offline_run({"loop": "reflect-v1"})
        with self.assertRaisesRegex(LabError, "Budget"):
            lab_module.offline_run({"loop": "greedy-prompt", "budget": 1000})

    def test_driven_run_matches_a_direct_run_with_the_same_model(self):
        chain = {"type": "prompt-chain", "steps": [{"name": "extract", "text": "extract"}, {"name": "route", "text": "route v1"}]}
        cases = [
            ("reflect-v1", {"prompt": "route v1"}, 1),
            ("evolve-v1", {"prompt": "route v1"}, 2),
            ("self-revise-v1", {"prompt": "route v1"}, 1),
            ("reflect-v1", {"artifact": chain}, 1),
        ]
        for loop, target, repeats in cases:
            with self.subTest(loop=loop, target=list(target)):
                payload = {"loop": loop, **target, "task": {"train": TRAIN, "holdout": HOLDOUT}, "max_trials": 3, "holdout_repeats": repeats, "seed": 5}
                first = self.lab.start(payload)
                self.assertEqual(first["state"], "needs_work")
                self.assertTrue(all(item["kind"] == "score" and item["stage"] == "baseline_train" for item in first["work"]))
                self.assertNotIn("billing", json.dumps([item["input"] for item in first["work"]]))
                view, rounds = drive(self.lab, first)
                request = validate_request({
                    "method": loop, "artifact": target.get("artifact", {"type": "prompt", "text": target.get("prompt")}),
                    "task": {"train": TRAIN, "holdout": HOLDOUT}, "max_trials": 3, "holdout_repeats": repeats, "seed": 5,
                    "objective": "normalized", "budget": {"max_calls": 1000, "max_cost_microusd": 0, "max_seconds": 3600},
                })
                direct = execute(request, loop, lambda stage, messages: model(messages[0]["content"], messages[-1]["content"]), lambda: None)
                result = view["results"][loop]
                for key in ("accepted", "artifact", "candidate", "trials", "model_calls", "evidence", "baseline_train", "best_train"):
                    self.assertEqual(result[key], direct[key], key)
                self.assertEqual(result["verdict"], "improved")
                self.assertEqual(view["recommendation"]["action"], "adopt")
                # Whole stages arrive together, so a run takes a handful of round trips, not one per call.
                self.assertLess(rounds, 14 if "artifact" in target else 10)
                self.assertEqual(self.lab.get(view["id"]), view)

    def test_all_loops_share_baseline_answers_and_report_a_recommendation(self):
        first = self.lab.start({"loops": "all", "prompt": "route v1", "task": {"train": TRAIN, "holdout": HOLDOUT}})
        self.assertEqual(first["loops"], ["reflect-v1", "evolve-v1", "self-revise-v1"])
        self.assertEqual(len(first["work"]), len(TRAIN))
        self.assertEqual(first["work"][0]["loops"], first["loops"])
        view, _ = drive(self.lab, first)
        self.assertEqual(set(view["results"]), set(first["loops"]))
        self.assertIn(view["recommendation"]["loop"], first["loops"])
        self.assertEqual(view["recommendation"]["artifact"]["type"], "prompt")

    def test_bad_answers_and_requests_are_handled(self):
        task = {"train": TRAIN, "holdout": HOLDOUT}
        with self.assertRaisesRegex(LabError, "between 8 and 64"):
            self.lab.start({"loop": "reflect-v1", "prompt": "p", "task": {"train": TRAIN, "holdout": HOLDOUT[:3]}})
        with self.assertRaisesRegex(LabError, "POST /api/run"):
            self.lab.start({"loop": "greedy-prompt", "prompt": "p", "task": task})
        with self.assertRaisesRegex(LabError, "either 'prompt'"):
            self.lab.start({"loop": "reflect-v1", "task": task})
        view = self.lab.start({"loop": "reflect-v1", "prompt": "route v0", "task": task, "max_trials": 1})
        again = self.lab.submit(view["id"], [{"id": "w_unknown", "output": "x"}])
        self.assertEqual(again["ignored"], ["w_unknown"])
        self.assertEqual(len(again["work"]), len(view["work"]))
        partial = self.lab.submit(view["id"], {view["work"][0]["id"]: "billing"})
        self.assertEqual(len(partial["work"]), len(view["work"]) - 1)
        with self.assertRaises(LabError):
            self.lab.submit(view["id"], [{"id": view["work"][1]["id"], "output": 5}])
        self.assertIsNone(self.lab.get("run_missing"))
        # A model that never improves the prompt ends with the original kept.
        view = self.lab.submit(view["id"], [{"id": item["id"], "output": "account"} for item in partial["work"]])
        while view["state"] == "needs_work":
            view = self.lab.submit(view["id"], [
                {"id": item["id"], "output": "nonsense" if item["kind"] == "propose" else "account"} for item in view["work"]
            ])
        self.assertEqual(view["results"]["reflect-v1"]["verdict"], "not_improved")
        self.assertEqual(view["recommendation"]["action"], "keep_original")

    def test_mcp_tools_cover_the_same_flow(self):
        def call(method, params=None, request_id=1):
            return mcp.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}, self.lab)

        def tool(name, arguments):
            reply = call("tools/call", {"name": name, "arguments": arguments})["result"]
            return reply["isError"], reply["content"][0]["text"]

        started = call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
        self.assertEqual(started["result"]["protocolVersion"], "2025-06-18")
        self.assertIsNone(mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, self.lab))
        names = [entry["name"] for entry in call("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["list_loops", "run_offline_loop", "compare_offline_loops", "start_prompt_run", "submit_work", "get_run"])
        self.assertEqual(len(json.loads(tool("list_loops", {})[1])["loops"]), 6)
        self.assertEqual(len(json.loads(tool("run_offline_loop", {"loop": "archive-prompt", "budget": 5})[1])["trials"]), 5)
        failed, text = tool("run_offline_loop", {"loop": "nope"})
        self.assertTrue(failed)
        self.assertIn("Unknown loop", text)
        view = json.loads(tool("start_prompt_run", {"loop": "reflect-v1", "prompt": "route v1", "task": {"train": TRAIN, "holdout": HOLDOUT}})[1])
        while view["state"] == "needs_work":
            outputs = [{"id": item["id"], "output": model(item["system"], item["input"])} for item in view["work"]]
            view = json.loads(tool("submit_work", {"run_id": view["id"], "outputs": outputs})[1])
        self.assertEqual(view["recommendation"]["action"], "adopt")
        self.assertEqual(json.loads(tool("get_run", {"run_id": view["id"]})[1])["state"], "done")
        self.assertEqual(call("nope")["error"]["code"], -32601)
        self.assertEqual(len(mcp.handle([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "id": 2, "method": "ping"}], self.lab)), 2)


class PublicHttp(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        store = Store(Path(self.directory.name) / "gateway.db")
        try:
            self.gateway = Gateway("127.0.0.1", 0, store, load_adapters(), load_models(None, demo=True), [])
        except PermissionError as exc:
            self.directory.cleanup()
            raise unittest.SkipTest("This environment does not allow binding a local HTTP port.") from exc
        self.thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.gateway.server_port}"

    def tearDown(self):
        self.gateway.shutdown()
        self.gateway.server_close()
        self.thread.join(timeout=3)
        self.directory.cleanup()

    def request(self, path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data, {"Content-Type": "application/json"} if data else {}, method=method)
        try:
            with urlopen(request, timeout=10) as response:
                body = response.read()
                return response.status, response.headers, body
        except HTTPError as error:
            with error:
                return error.code, error.headers, error.read()

    def test_pages_guide_and_keyless_api(self):
        for path, media in (("/", "text/html"), ("/jobs", "text/html"), ("/site.css", "text/css"), ("/lab.js", "text/javascript"), ("/jobs.js", "text/javascript")):
            status, headers, body = self.request(path)
            self.assertEqual((status, headers.get_content_type()), (200, media), path)
            self.assertTrue(body)
        status, headers, body = self.request("/llms.txt")
        self.assertEqual(status, 200)
        self.assertIn(f"{self.base}/api/runs".encode(), body)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
        status, _, body = self.request("/api/catalog")
        self.assertEqual(json.loads(body)["mcp"], self.base + "/mcp")
        status, _, body = self.request("/api/run?loop=greedy-prompt&budget=5&seed=1")
        self.assertEqual((status, len(json.loads(body)["trials"])), (200, 5))
        status, _, body = self.request("/api/compare?loops=greedy-prompt,archive-prompt&budget=5")
        self.assertEqual((status, len(json.loads(body)["runs"])), (200, 2))
        self.assertEqual(self.request("/api/run", {"loop": "missing"})[0], 400)
        self.assertEqual(self.request("/api/run?task=x")[0], 400)
        self.assertEqual(self.request("/api/nothing")[0], 404)
        self.assertEqual(self.request("/v1/jobs")[0], 401)
        self.assertIsNone(self.request("/v1/jobs")[1]["Access-Control-Allow-Origin"])
        status, headers, _ = self.request("/api/runs", method="OPTIONS")
        self.assertEqual((status, headers["Access-Control-Allow-Origin"]), (204, "*"))

    def test_driven_run_and_mcp_over_http(self):
        status, _, body = self.request("/api/runs", {"loop": "reflect-v1", "prompt": "route v1", "task": {"train": TRAIN, "holdout": HOLDOUT}})
        view = json.loads(body)
        self.assertEqual(status, 201)
        while view["state"] == "needs_work":
            outputs = [{"id": item["id"], "output": model(item["system"], item["input"])} for item in view["work"]]
            status, _, body = self.request(view["submit"]["path"], {"outputs": outputs})
            view = json.loads(body)
        self.assertEqual(view["results"]["reflect-v1"]["verdict"], "improved")
        self.assertEqual(json.loads(self.request(f"/api/runs/{view['id']}")[2])["state"], "done")
        self.assertEqual(self.request("/api/runs/run_aaaaaaaaaaaaaaaaaaaa")[0], 404)

        status, _, body = self.request("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual((status, len(json.loads(body)["result"]["tools"])), (200, 6))
        self.assertEqual(self.request("/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"})[0], 202)
        self.assertEqual(self.request("/mcp")[0], 405)
