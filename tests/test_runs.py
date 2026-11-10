import json
import re
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rsi_library import benchmarks
from rsi_library.gateway import lab as lab_module, mcp, methods
from rsi_library.gateway.api import Gateway
from rsi_library.gateway.lab import Lab, LabError, with_base
from rsi_library.gateway.methods import MethodError, execute, minimum_calls, validate_request
from rsi_library.gateway.providers import load_models
from rsi_library.gateway.routing import load_adapters
from rsi_library.gateway.store import Store

from test_lab import HOLDOUT, TRAIN, drive, model

TASK = {"train": TRAIN, "holdout": HOLDOUT}
BUDGET = {"max_calls": 1000, "max_cost_microusd": 0, "max_seconds": 3600}

ORIGINAL_SOURCE = '''import re

def to_seconds(text):
    match = re.search(r"(\\d+)\\s*([sm])", text)
    if not match:
        return 0
    return int(match[1]) * (60 if match[2] == "m" else 1)
'''
FIXED_SOURCE = '''import re

UNITS = {"h": 3600, "m": 60, "s": 1}

def to_seconds(text):
    total = 0.0
    for amount, unit in re.findall(r"(\\d+(?:\\.\\d+)?)\\s*([hms])", text.lower()):
        total += float(amount) * UNITS[unit]
    return int(total) if total == int(total) else total
'''


def run_program(item: dict) -> str:
    """Stand in for the caller's sandbox: run entry(input) and report the result as text."""
    namespace: dict = {}
    try:
        exec(item["source"], namespace)
        value = namespace[item["entry"]](item["input"])
    except Exception as exc:  # noqa: BLE001 - errors are part of the reply
        return f"Error: {exc}"
    return value if isinstance(value, str) else json.dumps(value)


def coder(item: dict) -> str:
    """A caller that runs programs and proposes the fixed function."""
    if item["kind"] == "run":
        return run_program(item)
    if item["kind"] == "propose":
        assert "Keep the function to_seconds" in item["system"]
        return json.dumps({"source": "```python\n" + FIXED_SOURCE + "```"})
    raise AssertionError(item["kind"])


def program_payload(loop="reflect-v1", **extra):
    task = lab_module.EXAMPLE_PROGRAM_RUN["task"]
    artifact = {"type": "program", "language": "python", "entry": "to_seconds", "source": ORIGINAL_SOURCE}
    return {"loop": loop, "artifact": artifact, "task": task, "objective": "number", "max_trials": 2, **extra}


class Scoring(unittest.TestCase):
    def scorer(self, objective, settings=None):
        return methods._Scorer(objective, settings or {})

    def test_number_contains_and_json_objectives(self):
        number = self.scorer("number")
        self.assertEqual(number.one("The total is 1,234.5 seconds.", "1234.5"), 1.0)
        self.assertEqual(number.one("5400.0", "5400"), 1.0)
        self.assertEqual(number.one("no number", "5"), 0.0)
        self.assertEqual(number.one("101", "100"), 0.0)
        self.assertEqual(self.scorer("number", {"tolerance": 1.0}).one("101", "100"), 1.0)
        self.assertEqual(self.scorer("number", {"relative_tolerance": 0.05}).one("104", "100"), 1.0)
        self.assertEqual(self.scorer("contains").one("Route it to  BILLING please", "billing"), 1.0)
        json_fields = self.scorer("json")
        expected = '{"label": "Billing", "priority": 2}'
        self.assertEqual(json_fields.one('```json\n{"label": "billing ", "priority": 2.0}\n```', expected), 1.0)
        self.assertEqual(json_fields.one('{"label": "billing", "priority": 3}', expected), 0.5)
        self.assertEqual(json_fields.one("not json", expected), 0.0)
        self.assertEqual(json_fields.one('{"label": true, "priority": 1}', '{"label": 1, "priority": true}'), 0.0)
        self.assertEqual(self.scorer("json", {"fields": ["label"]}).one('{"label": "Billing"}', expected), 1.0)

    def test_settings_are_checked_against_every_expected_output(self):
        def request(objective, scoring=None, expected="7"):
            task = {"train": [{"input": "a", "expected": expected}], "holdout": [{"input": f"h{i}", "expected": expected} for i in range(8)]}
            body = {"method": "reflect-v1", "artifact": {"type": "prompt", "text": "p"}, "task": task, "budget": BUDGET, "objective": objective}
            if scoring is not None:
                body["scoring"] = scoring
            return validate_request(body)

        self.assertEqual(request("number", {"tolerance": 0.5})["scoring"], {"tolerance": 0.5})
        self.assertEqual(request("exact")["scoring"], {})
        for objective, scoring, expected, message in (
            ("number", None, "seven", "must contain a number"),
            ("number", {"tolerance": -1}, "7", "at least 0"),
            ("number", {"rubric": "x"}, "7", "accepts relative_tolerance, tolerance"),
            ("json", None, "[1]", "must be a JSON object"),
            ("json", {"fields": ["b"]}, '{"a": 1}', "no 'b' field"),
            ("contains", None, " ", "nonempty"),
            ("judge", {"rubric": "x" * 3000}, "7", "at most 2000"),
            ("fuzzy", None, "7", "objective must be one of"),
        ):
            with self.subTest(objective=objective, scoring=scoring):
                with self.assertRaisesRegex(MethodError, message):
                    request(objective, scoring, expected)

    def test_the_model_judge_is_budgeted_and_runs_after_the_replies(self):
        prompt = {"method": "reflect-v1", "artifact": {"type": "prompt", "text": "route v1"}, "task": TASK, "budget": BUDGET}
        plain = minimum_calls(validate_request({**prompt, "objective": "normalized"}))
        judged = minimum_calls(validate_request({**prompt, "objective": "judge"}))
        self.assertEqual(judged - plain, 2 * len(TRAIN) + 2 * len(HOLDOUT))

        stages = []

        def complete(stage, messages):
            stages.append(stage)
            if stage.endswith("_judge"):
                case = json.loads(messages[-1]["content"])
                return "Score: 9" if case["expected"].casefold() in case["reply"].casefold() else "1"
            return model(messages[0]["content"], messages[-1]["content"])

        result = execute(validate_request({**prompt, "objective": "judge"}), "reflect-v1", complete, lambda: None)
        self.assertEqual(result["evidence"]["verdict"], "improved")
        self.assertAlmostEqual(result["baseline_holdout"], (0.9 * 2 + 0.1 * 6) / 8)
        # Each evaluation finishes its replies before any judging starts.
        first_judge = stages.index("baseline_train_judge")
        self.assertEqual(stages[:first_judge], ["baseline_train"] * len(TRAIN))
        self.assertEqual(stages[first_judge:first_judge + len(TRAIN)], ["baseline_train_judge"] * len(TRAIN))


class Programs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.lab = Lab(Path(self.directory.name) / "lab.db")

    def tearDown(self):
        self.directory.cleanup()

    def test_hosted_jobs_refuse_programs(self):
        payload = {**program_payload(), "method": "reflect-v1", "budget": BUDGET}
        for key in ("loop", "max_trials"):
            payload.pop(key)
        with self.assertRaisesRegex(MethodError, "never executes code"):
            validate_request(payload)
        request = validate_request(payload, programs=True)
        with self.assertRaisesRegex(MethodError, "caller executes them"):
            execute(request, "reflect-v1", lambda stage, messages: "", lambda: None)
        self.assertNotIn("program", json.dumps(methods.catalog()))
        self.assertIn("program", json.dumps(methods.catalog(programs=True)))
        with self.assertRaisesRegex(MethodError, "must define the function"):
            validate_request({**payload, "artifact": {**payload["artifact"], "entry": "missing"}}, programs=True)

    def test_a_code_loop_fixes_a_function_the_caller_runs(self):
        view = self.lab.start(program_payload())
        self.assertEqual(view["request"]["artifact"]["type"], "program")
        first = view["work"][0]
        self.assertEqual((first["kind"], first["language"], first["entry"]), ("run", "python", "to_seconds"))
        self.assertNotIn("system", first)
        self.assertIn("kind 'run'", view["instructions"])
        rounds = 0
        while view["state"] == "needs_work":
            rounds += 1
            view = self.lab.submit(view["id"], [{"id": item["id"], "output": coder(item)} for item in view["work"]])
        result = view["results"]["reflect-v1"]
        self.assertEqual(result["baseline_holdout"], 4 / 12)
        self.assertEqual(result["candidate_holdout"], 1.0)
        self.assertEqual(result["verdict"], "improved")
        # The fence the proposer wrapped around its source was removed.
        self.assertEqual(view["recommendation"]["artifact"]["source"], FIXED_SOURCE.rstrip("\n"))
        self.assertLess(rounds, 8)

    def test_all_loops_for_a_program_and_the_shipped_example(self):
        payload = program_payload()
        payload.pop("loop")
        view = self.lab.start({**payload, "loops": "all"})
        self.assertEqual(view["loops"], ["reflect-v1", "evolve-v1"])
        example = lab_module.EXAMPLE_PROGRAM_RUN
        shipped = self.lab.start({"loop": "evolve-v1", **example})
        self.assertEqual(shipped["work"][0]["entry"], "toSeconds")
        with self.assertRaisesRegex(LabError, "self-revise-v1 does not support this program artifact. It improves: prompt."):
            self.lab.start({**program_payload(), "loop": "self-revise-v1"})


class RunStorage(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "lab.db"
        self.now = [1_000_000.0]
        self.lab = Lab(self.path, clock=lambda: self.now[0])

    def tearDown(self):
        self.directory.cleanup()

    def row(self, run_id):
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            return connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def start(self, **extra):
        return self.lab.start({"loop": "reflect-v1", "prompt": "route v1", "task": TASK, **extra})

    def test_reading_a_run_serves_the_stored_view_without_replaying(self):
        view = self.start()
        self.assertEqual(view["page"], f"/runs/{view['id']}")
        self.assertEqual(with_base(view, "https://rsi.test")["page"], f"https://rsi.test/runs/{view['id']}")
        with mock.patch.object(lab_module, "_advance", side_effect=AssertionError("replayed")):
            self.assertEqual(self.lab.get(view["id"]), view)
            again = self.lab.submit(view["id"], {"w_unknown": "x"})
        self.assertEqual(again["ignored"], ["w_unknown"])
        self.assertEqual(again["work"], view["work"])
        watch = self.lab.get(view["id"], work=False)
        self.assertFalse(watch.keys() & {"work", "submit", "instructions"})
        self.assertEqual(watch["pending_work"], len(TRAIN))
        self.assertEqual(json.loads(self.row(view["id"])["pending"]), [item["id"] for item in view["work"]])

    def test_progress_says_what_each_loop_is_doing(self):
        view = self.lab.start({"loops": "all", "prompt": "route v1", "task": TASK})
        for loop in view["loops"]:
            progress = view["progress"][loop]
            self.assertEqual((progress["stage"], progress["best_train"], progress["trace"]), ("baseline_train", None, []))
            self.assertEqual(progress["doing"], "Scoring the original on the training cases")
        view = self.lab.submit(view["id"], [{"id": item["id"], "output": model(item["system"], item["input"])} for item in view["work"]])
        progress = view["progress"]["reflect-v1"]
        self.assertEqual(progress["doing"], "Writing version 1 of 3")
        self.assertEqual(progress["best_train"], 1 / 6)
        self.assertEqual(progress["trace"], [{"id": 0, "parent_ids": [], "status": "evaluated", "score": 1 / 6}])
        self.assertEqual(progress["calls"], len(TRAIN))
        view, _ = drive(self.lab, view)
        self.assertEqual({entry["state"] for entry in view["progress"].values()}, {"done"})
        self.assertEqual(view["request"]["holdout_cases"], len(HOLDOUT))

    def test_a_finished_run_keeps_only_its_final_view(self):
        view, _ = drive(self.lab, self.start())
        row = self.row(view["id"])
        self.assertEqual((row["answers"], row["view"], row["pending"]), ("{}", None, None))
        self.assertEqual(row["size"], len(row["final"]))
        self.assertEqual(self.lab.get(view["id"]), view)
        self.assertIn("finished", self.lab.submit(view["id"], {"w_x": "y"})["note"])

    def test_runs_can_be_deleted(self):
        view = self.start()
        self.assertTrue(self.lab.delete(view["id"]))
        self.assertIsNone(self.lab.get(view["id"]))
        self.assertIsNone(self.lab.submit(view["id"], {"w_x": "y"}))
        self.assertFalse(self.lab.delete(view["id"]))

    def test_old_runs_expire_and_storage_is_capped(self):
        unfinished = self.start()
        finished, _ = drive(self.lab, self.start())
        self.now[0] += 15 * 86_400
        self.start()
        self.assertIsNone(self.lab.get(unfinished["id"]))
        self.assertIsNotNone(self.lab.get(finished["id"]))
        self.now[0] += 91 * 86_400
        latest = self.start()
        self.assertIsNone(self.lab.get(finished["id"]))
        with mock.patch.object(lab_module, "MAX_TOTAL_BYTES", 1):
            self.now[0] += 1
            newest = self.start()
        self.assertIsNone(self.lab.get(latest["id"]))
        self.assertIsNotNone(self.lab.get(newest["id"]))
        with mock.patch.object(lab_module, "MAX_RUN_BYTES", 100):
            with self.assertRaisesRegex(LabError, "storage limit"):
                self.lab.submit(newest["id"], [{"id": item["id"], "output": "x" * 20} for item in newest["work"]])

    def test_runs_stored_before_views_were_cached_still_work(self):
        legacy = Path(self.directory.name) / "legacy.db"
        requests = Lab._requests({"loop": "reflect-v1", "prompt": "route v1", "task": TASK})
        with sqlite3.connect(legacy) as connection:
            connection.execute(
                "CREATE TABLE runs (id TEXT PRIMARY KEY, created_at REAL NOT NULL, updated_at REAL NOT NULL, "
                "requests TEXT NOT NULL, answers TEXT NOT NULL, final TEXT)"
            )
            connection.execute("INSERT INTO runs VALUES ('run_legacylegacylegacy', 1, 1, ?, '{}', NULL)", (json.dumps(requests),))
        lab = Lab(legacy)
        view = lab.get("run_legacylegacylegacy")
        self.assertEqual((view["state"], len(view["work"])), ("needs_work", len(TRAIN)))
        view, _ = drive(lab, view)
        self.assertEqual(view["results"]["reflect-v1"]["verdict"], "improved")


class JudgedRuns(unittest.TestCase):
    def test_judge_items_follow_the_replies_they_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = Lab(Path(directory) / "lab.db")
            view = lab.start({"loop": "reflect-v1", "prompt": "route v1", "task": TASK, "objective": "judge",
                              "scoring": {"rubric": "Names the right team."}})
            kinds = []
            while view["state"] == "needs_work":
                kinds.append({item["kind"] for item in view["work"]})
                outputs = []
                for item in view["work"]:
                    if item["kind"] == "judge":
                        self.assertIn("Names the right team.", item["system"])
                        case = json.loads(item["input"])
                        outputs.append({"id": item["id"], "output": "10" if case["expected"].casefold() == case["reply"].casefold() else "0"})
                    else:
                        outputs.append({"id": item["id"], "output": model(item["system"], item["input"])})
                view = lab.submit(view["id"], outputs)
            self.assertEqual(kinds[:3], [{"score"}, {"judge"}, {"propose"}])
            self.assertIn("kind 'judge'", lab_module.WORK_INSTRUCTIONS)
            self.assertEqual(view["results"]["reflect-v1"]["verdict"], "improved")
            self.assertEqual(view["request"]["scoring"], {"rubric": "Names the right team."})


class Benchmarks(unittest.TestCase):
    def test_published_results_match_the_code(self):
        stored = benchmarks.results()
        self.assertEqual(stored["seeds"], len(benchmarks.SEEDS))
        self.assertEqual(stored["budgets"], list(benchmarks.BUDGETS))
        self.assertEqual(set(stored["loops"]), set(benchmarks.LOOPS))
        fresh = benchmarks.measure(task_ids=("tickets",), budgets=(24,))
        for loop in benchmarks.LOOPS:
            self.assertEqual(fresh["loops"][loop]["24"]["per_task"]["tickets"], stored["loops"][loop]["24"]["per_task"]["tickets"], loop)
        self.assertEqual(fresh["tasks"][0], stored["tasks"][0])
        self.assertEqual(lab_module.catalog()["benchmarks"]["driven"]["status"], "not_measured")


class RunPagesOverHttp(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        store = Store(Path(self.directory.name) / "gateway.db")
        try:
            self.gateway = Gateway("127.0.0.1", 0, store, load_adapters(), load_models(None, demo=True), [], trust_proxy=True)
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

    def request(self, path, payload=None, method=None, headers=None):
        data = None if payload is None else json.dumps(payload).encode()
        merged = {"Content-Type": "application/json"} if data else {}
        merged.update(headers or {})
        request = Request(self.base + path, data, merged, method=method)
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            with error:
                return error.code, error.headers, error.read()

    def test_run_pages_reads_and_deletion(self):
        secure = {"X-Forwarded-Proto": "https", "Host": "rsi.test"}
        status, _, body = self.request("/api/runs", {"loop": "reflect-v1", "prompt": "route v1", "task": TASK}, headers=secure)
        view = json.loads(body)
        self.assertEqual((status, view["page"]), (201, f"https://rsi.test/runs/{view['id']}"))
        status, headers, page = self.request(f"/runs/{view['id']}", headers=secure)
        self.assertEqual((status, headers.get_content_type(), headers["X-Robots-Tag"]), (200, "text/html", "noindex"))
        self.assertIn(b"lab.js", page)
        self.assertEqual(headers["Strict-Transport-Security"], "max-age=31536000")
        status, headers, body = self.request(f"/api/runs/{view['id']}?work=0", headers={"X-Forwarded-Proto": "http"})
        watched = json.loads(body)
        self.assertNotIn("work", watched)
        self.assertEqual((watched["pending_work"], headers["X-Robots-Tag"]), (len(TRAIN), "noindex"))
        self.assertIsNone(headers["Strict-Transport-Security"])
        status, headers, _ = self.request(f"/api/runs/{view['id']}", method="OPTIONS")
        self.assertIn("DELETE", headers["Access-Control-Allow-Methods"])
        self.assertEqual(self.request(f"/api/runs/{view['id']}", method="DELETE")[0], 200)
        self.assertEqual(self.request(f"/api/runs/{view['id']}", method="DELETE")[0], 404)
        self.assertEqual(self.request(f"/api/runs/{view['id']}")[0], 404)
        self.assertEqual(self.request("/api/run", method="DELETE")[0], 405)

    def test_static_files_headers_and_missing_pages(self):
        status, headers, _ = self.request("/")
        etag = headers["ETag"]
        self.assertEqual((status, headers["Cache-Control"], headers["Cross-Origin-Opener-Policy"]), (200, "no-cache", "same-origin"))
        self.assertIn("worker-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(self.request("/", headers={"If-None-Match": etag})[0], 304)
        status, headers, body = self.request("/runner.js")
        self.assertEqual(headers["Content-Security-Policy"], "default-src 'none'; script-src 'self' 'unsafe-eval'")
        self.assertIn(b"postMessage", body)
        status, headers, body = self.request("/og.png")
        self.assertEqual((status, headers.get_content_type(), body[:4]), (200, "image/png", b"\x89PNG"))
        status, headers, body = self.request("/nowhere", headers={"Accept": "text/html"})
        self.assertEqual((status, headers.get_content_type()), (404, "text/html"))
        self.assertIn(b"nothing at this address", body)
        status, headers, body = self.request("/nowhere")
        self.assertEqual((status, headers.get_content_type()), (404, "application/json"))
        status, _, body = self.request("/robots.txt")
        self.assertIn(b"Sitemap: ", body)
        status, headers, body = self.request("/sitemap.xml")
        self.assertEqual(len(re.findall(rb"<loc>", body)), 3)

    def test_mcp_starts_code_runs_and_links_the_page(self):
        def tool(name, arguments):
            reply = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
                               self.gateway.lab, base="https://rsi.test")["result"]
            return reply["isError"], reply["content"][0]["text"]

        payload = program_payload()
        artifact = payload.pop("artifact")
        failed, text = tool("start_code_run", {**payload, "language": "python", "entry": "to_seconds", "source": artifact["source"]})
        self.assertFalse(failed, text)
        view = json.loads(text)
        self.assertEqual(view["page"], f"https://rsi.test/runs/{view['id']}")
        self.assertEqual(view["work"][0]["kind"], "run")
        failed, text = tool("get_run", {"run_id": view["id"], "work": False})
        self.assertNotIn("work", json.loads(text))
        self.assertEqual(json.loads(tool("delete_run", {"run_id": view["id"]})[1]), {"id": view["id"], "deleted": True})
        failed, text = tool("delete_run", {"run_id": view["id"]})
        self.assertTrue(failed)
        self.assertIn("Unknown run_id", text)
        failed, text = tool("start_code_run", {**payload, "language": "ruby", "entry": "f", "source": "def f; end"})
        self.assertTrue(failed)


if __name__ == "__main__":
    unittest.main()
