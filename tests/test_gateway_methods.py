import json
import unittest

from rsi_library.gateway.methods import MethodError, compatible_methods, execute, minimum_calls, validate_request


def request(method="auto", artifact=None, calls=10):
    return {
        "method": method,
        "artifact": artifact or {"type": "prompt", "text": "original"},
        "task": {
            "train": [{"input": "first training input", "expected": "first"}, {"input": "second training input", "expected": "second"}],
            "holdout": [{"input": "private holdout input", "expected": "hidden-answer"}],
        },
        "budget": {"max_calls": calls, "max_cost_microusd": 1_000_000, "max_seconds": 30},
        "max_trials": 2,
    }


class GatewayMethods(unittest.TestCase):
    def test_archive_uses_multiple_parents_and_checks_untouched_holdout(self):
        config = validate_request(request("evolve-v1"))
        proposals = iter(("first improvement", "second improvement"))
        calls = []

        def complete(stage, messages):
            system, user = (message["content"] for message in messages)
            calls.append(stage)
            if stage == "propose":
                self.assertNotIn("private holdout input", user)
                self.assertNotIn("hidden-answer", user)
                parents = json.loads(user)["parents"]
                self.assertEqual(len(parents), calls.count("propose"))
                return json.dumps({"text": next(proposals)})
            if stage.endswith("holdout"):
                self.assertEqual(user, "private holdout input")
                self.assertNotIn("hidden-answer", system)
                return "hidden-answer" if system == "second improvement" else "miss"
            if system == "second improvement" or system == "first improvement" and user.startswith("first"):
                return user.split()[0]
            return "miss"

        result = execute(config, "evolve-v1", complete, lambda: None)
        self.assertEqual(result["trials"][2]["parent_ids"], [1, 0])
        self.assertEqual((result["baseline_holdout"], result["candidate_holdout"]), (0, 1))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["artifact"]["text"], "second improvement")
        self.assertEqual(result["model_calls"], config["budget"]["max_calls"])

    def test_reflection_preserves_original_when_holdout_does_not_improve(self):
        config = validate_request(request("reflect-v1", calls=7))
        phases = []

        def complete(stage, messages):
            if stage == "propose":
                return '{"text":"overfits train"}'
            if stage == "candidate_train":
                return messages[-1]["content"].split()[0]
            return "miss"

        result = execute(config, "reflect-v1", complete, lambda: None, progress=phases.append)
        self.assertEqual(result["best_train"], 1)
        self.assertEqual(result["candidate"]["text"], "overfits train")
        self.assertEqual(result["artifact"], config["artifact"])
        self.assertFalse(result["accepted"])
        self.assertEqual(phases[-1], {"stage": "candidate_holdout", "calls": 6, "trials": 1, "planned_trials": 1})

    def test_self_revision_uses_inherited_improver_to_create_next_prompt(self):
        config = validate_request(request("self-revise-v1"))
        proposal_count = 0

        def complete(stage, messages):
            nonlocal proposal_count
            if stage == "propose":
                proposal_count += 1
                if proposal_count == 1:
                    return '{"text":"original","improver":"Inherited policy goes here"}'
                self.assertIn("Inherited policy goes here", messages[0]["content"])
                return '{"text":"better","improver":"A further revision"}'
            if stage == "candidate_train":
                return messages[-1]["content"].split()[0]
            return "hidden-answer" if stage == "candidate_holdout" else "miss"

        result = execute(config, "self-revise-v1", complete, lambda: None)
        self.assertEqual(result["trials"][2]["parent_ids"], [1])
        self.assertEqual(result["artifact"]["improver"], "A further revision")
        self.assertEqual(result["model_calls"], 8)
        self.assertEqual(compatible_methods(validate_request({**config, "method": "auto", "artifact": result["artifact"]})), ("self-revise-v1",))

    def test_chain_passes_previous_output_and_reserves_both_holdout_passes(self):
        artifact = {"type": "prompt-chain", "steps": [{"name": "parse", "text": "parse"}, {"name": "answer", "text": "original"}]}
        config = validate_request(request("reflect-v1", artifact, calls=13))
        stages = []

        def complete(stage, messages):
            system, user = (message["content"] for message in messages)
            stages.append(stage)
            if stage == "propose":
                return '{"steps":[{"name":"parse","text":"parse"},{"name":"answer","text":"better"}]}'
            if system == "parse":
                return user.split()[0]
            data = json.loads(user)
            self.assertEqual(data["previous"], data["input"].split()[0])
            if system == "better":
                return "hidden-answer" if stage == "candidate_holdout" else data["previous"]
            return "miss"

        self.assertEqual(minimum_calls(config), 13)
        with self.assertRaisesRegex(MethodError, "needs at least 13"):
            execute(config, "reflect-v1", complete, lambda: None, call_capacity=12)
        self.assertEqual(stages, [])

        result = execute(config, "reflect-v1", complete, lambda: None, call_capacity=13)
        self.assertTrue(result["accepted"])
        self.assertEqual(len(stages), 13)
        self.assertEqual(stages[-4:], ["baseline_holdout"] * 2 + ["candidate_holdout"] * 2)

    def test_request_rejects_test_leakage_and_incompatible_method(self):
        payload = request()
        payload["task"]["holdout"][0]["input"] = " FIRST   training input "
        with self.assertRaisesRegex(MethodError, "unique"):
            validate_request(payload)

        artifact = {"type": "prompt", "text": "task", "improver": "custom editor"}
        with self.assertRaisesRegex(MethodError, "does not support"):
            validate_request(request("reflect-v1", artifact))
