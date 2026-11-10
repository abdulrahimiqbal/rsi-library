import json
import unittest

from rsi_library.gateway.methods import MethodError, compatible_methods, execute, minimum_calls, validate_request
from rsi_library.stats import minimum_wins, paired_comparison, sign_test


HOLDOUT = [{"input": f"private holdout input {index}", "expected": "hidden-answer"} for index in range(8)]


def request(method="auto", artifact=None, calls=24, **extra):
    return {
        "method": method,
        "artifact": artifact or {"type": "prompt", "text": "original"},
        "task": {
            "train": [{"input": "first training input", "expected": "first"}, {"input": "second training input", "expected": "second"}],
            "holdout": [dict(case) for case in HOLDOUT],
        },
        "budget": {"max_calls": calls, "max_cost_microusd": 1_000_000, "max_seconds": 30},
        "max_trials": 2,
        **extra,
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
                self.assertTrue(user.startswith("private holdout input"))
                self.assertNotIn("hidden-answer", system)
                return "hidden-answer" if system == "second improvement" else "miss"
            if system == "second improvement" or system == "first improvement" and user.startswith("first"):
                return user.split()[0]
            return "miss"

        result = execute(config, "evolve-v1", complete, lambda: None)
        self.assertEqual(result["trials"][2]["parent_ids"], [1, 0])
        self.assertEqual((result["baseline_holdout"], result["candidate_holdout"]), (0, 1))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["evidence"]["verdict"], "improved")
        self.assertEqual(result["evidence"]["improved_cases"], 8)
        self.assertEqual(result["evidence"]["p_value"], 1 / 256)
        self.assertEqual(result["artifact"]["text"], "second improvement")
        self.assertEqual(result["model_calls"], config["budget"]["max_calls"])

    def test_reflection_preserves_original_when_holdout_does_not_improve(self):
        config = validate_request(request("reflect-v1", calls=21))
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
        self.assertEqual(result["evidence"]["verdict"], "not_improved")
        self.assertEqual(phases[-1], {
            "stage": "candidate_holdout", "calls": 13, "trials": 1, "planned_trials": 1, "best_train": 1.0,
            "trace": [
                {"id": 0, "parent_ids": [], "status": "evaluated", "score": 0.0},
                {"id": 1, "parent_ids": [0], "status": "evaluated", "score": 1.0},
            ],
        })
        self.assertEqual(phases[0]["best_train"], None)
        self.assertEqual(phases[0]["trace"], [])

    def test_a_gain_that_could_be_chance_keeps_the_original(self):
        config = validate_request(request("reflect-v1", calls=21))

        def complete(stage, messages):
            system, user = (message["content"] for message in messages)
            if stage == "propose":
                return '{"text":"slightly better"}'
            if stage == "candidate_train":
                return user.split()[0]
            if stage == "candidate_holdout":
                return "hidden-answer" if user[-1] in "012" else "miss"
            return "miss"

        result = execute(config, "reflect-v1", complete, lambda: None)
        evidence = result["evidence"]
        self.assertEqual((evidence["improved_cases"], evidence["regressed_cases"]), (3, 0))
        self.assertEqual(evidence["p_value"], 0.125)
        self.assertEqual(evidence["verdict"], "inconclusive")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["artifact"], config["artifact"])
        self.assertEqual(result["candidate"]["text"], "slightly better")

    def test_repeated_holdout_tries_are_averaged_per_case(self):
        config = validate_request(request("reflect-v1", calls=53, holdout_repeats=3))
        self.assertEqual(minimum_calls(config), 2 * 2 + 1 + 2 * 8 * 3)
        tries = {}

        def complete(stage, messages):
            system, user = (message["content"] for message in messages)
            if stage == "propose":
                return '{"text":"sometimes right"}'
            if stage == "candidate_train":
                return user.split()[0]
            if stage == "candidate_holdout":
                tries[user] = tries.get(user, 0) + 1
                return "hidden-answer" if tries[user] != 2 else "miss"
            return "miss"

        result = execute(config, "reflect-v1", complete, lambda: None)
        self.assertEqual(set(tries.values()), {3})
        self.assertAlmostEqual(result["candidate_holdout"], 2 / 3)
        self.assertEqual(result["holdout_repeats"], 3)
        self.assertEqual(result["evidence"]["repeats"], 3)
        self.assertEqual(result["evidence"]["verdict"], "improved")
        self.assertEqual(result["model_calls"], 53)

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
        self.assertEqual(result["model_calls"], 22)
        self.assertEqual(compatible_methods(validate_request({**config, "method": "auto", "artifact": result["artifact"]})), ("self-revise-v1",))

    def test_chain_passes_previous_output_and_reserves_both_holdout_passes(self):
        artifact = {"type": "prompt-chain", "steps": [{"name": "parse", "text": "parse"}, {"name": "answer", "text": "original"}]}
        config = validate_request(request("reflect-v1", artifact, calls=41))
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

        self.assertEqual(minimum_calls(config), 41)
        with self.assertRaisesRegex(MethodError, "needs at least 41"):
            execute(config, "reflect-v1", complete, lambda: None, call_capacity=40)
        self.assertEqual(stages, [])

        result = execute(config, "reflect-v1", complete, lambda: None, call_capacity=41)
        self.assertTrue(result["accepted"])
        self.assertEqual(len(stages), 41)
        self.assertEqual(stages[-32:], ["baseline_holdout"] * 16 + ["candidate_holdout"] * 16)

    def test_request_rejects_test_leakage_small_holdout_and_incompatible_method(self):
        payload = request()
        payload["task"]["holdout"][0]["input"] = " FIRST   training input "
        with self.assertRaisesRegex(MethodError, "unique"):
            validate_request(payload)

        small = request()
        small["task"]["holdout"] = small["task"]["holdout"][:7]
        with self.assertRaisesRegex(MethodError, "between 8 and 64"):
            validate_request(small)

        with self.assertRaisesRegex(MethodError, "holdout_repeats"):
            validate_request(request(holdout_repeats=6))

        artifact = {"type": "prompt", "text": "task", "improver": "custom editor"}
        with self.assertRaisesRegex(MethodError, "does not support"):
            validate_request(request("reflect-v1", artifact))


class PairedEvidence(unittest.TestCase):
    def test_sign_test_matches_exact_binomial_tails(self):
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertEqual(sign_test(5, 0), 1 / 32)
        self.assertEqual(sign_test(6, 1), 8 / 128)
        self.assertEqual(sign_test(1, 1), 3 / 4)
        self.assertEqual(minimum_wins(8), 5)
        self.assertIsNone(minimum_wins(4))

    def test_verdicts(self):
        self.assertEqual(paired_comparison([0] * 8, [1] * 5 + [0] * 3)["verdict"], "improved")
        self.assertEqual(paired_comparison([0] * 8, [1] * 4 + [0] * 4)["verdict"], "inconclusive")
        self.assertEqual(paired_comparison([1] * 8, [1] * 8)["verdict"], "not_improved")
        mixed = paired_comparison([1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1])
        self.assertEqual((mixed["improved_cases"], mixed["regressed_cases"]), (6, 2))
        self.assertEqual(mixed["verdict"], "inconclusive")
