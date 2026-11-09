import copy
import unittest

from examples.custom_task import experiment
from rsi_library import ConfigError, Controls, ProposalRejected, catalog, compare, make_registry, run
from rsi_library.prompt import PromptEditor, PromptTarget


class Experiments(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog()
        self.configs = {entry["id"]: entry["config"] for entry in self.catalog["templates"]}
        self.registry = make_registry()

    def test_templates_repeat_and_record_parentage(self):
        improvements = []
        for template, config in self.configs.items():
            with self.subTest(template=template):
                result = run(config, self.registry)
                repeated = run(config, make_registry())
                self.assertEqual(result["trials"], repeated["trials"])
                trials = result["trials"]
                self.assertEqual(config["budget"], result["evaluations_used"])
                self.assertEqual(config["budget"], len(trials))
                self.assertIsNone(trials[0]["parent_id"])
                self.assertEqual(trials[0]["score"], result["baseline_score"])
                self.assertEqual(max(trial["score"] for trial in trials), result["best_score"])
                self.assertIn(result["holdout_score"], [i / result["holdout_cases"] for i in range(result["holdout_cases"] + 1)])
                for i, trial in enumerate(trials[1:], 1):
                    self.assertIn(trial["parent_id"], {parent["id"] for parent in trials[:i]})
                    self.assertGreaterEqual(trial["best_score"], trials[i - 1]["best_score"])
                improvements.append(result["best_score"] - result["baseline_score"])
        self.assertGreater(max(improvements), 0)

    def test_holdout_does_not_affect_search(self):
        config = copy.deepcopy(self.configs["adaptive-improver"])
        config["task"] = copy.deepcopy(self.catalog["example_task"])
        result = run(config, self.registry)
        labels = list(dict.fromkeys(case["expected"] for case in config["task"]["train"]))
        for case in config["task"]["holdout"]:
            case["expected"] = next(label for label in labels if label != case["expected"])
        changed = run(config, self.registry)
        self.assertEqual(result["trials"], changed["trials"])
        self.assertEqual(result["best_candidate"], changed["best_candidate"])
        config["task"]["holdout"] = []
        without_holdout = run(config, self.registry)
        self.assertEqual(result["trials"], without_holdout["trials"])
        self.assertIsNone(without_holdout["holdout_score"])

    def test_baseline_budget_includes_no_proposals(self):
        config = {**self.configs["greedy-prompt"], "budget": 1}
        result = run(config, self.registry)
        self.assertEqual(len(result["trials"]), 1)
        self.assertEqual(result["best_candidate"], result["trials"][0]["candidate"])

    def test_comparison_rejects_changes_to_controls(self):
        greedy = self.configs["greedy-prompt"]
        archive = self.configs["archive-prompt"]
        pair = compare([greedy, archive], self.registry)["runs"]
        self.assertEqual(pair[0]["evaluations_used"], pair[1]["evaluations_used"])
        self.assertEqual(pair[0]["trials"][0]["candidate"], pair[1]["trials"][0]["candidate"])
        task = copy.deepcopy(self.catalog["example_task"])
        task["train"][0]["input"] += " with a different input"
        for change in (
            {"seed": greedy["seed"] + 1},
            {"budget": 1},
            {"judge": "normalized"},
            {"task": task},
        ):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                compare([greedy, {**archive, **change}], self.registry)

    def test_custom_python_target_and_proposer(self):
        result = experiment()
        self.assertLess(result["baseline_score"], result["best_score"])
        self.assertEqual(result["best_score"], 1)
        self.assertEqual(result["holdout_score"], 1)
        self.assertEqual(result["evaluations_used"], 8)

    def test_controls_skip_repeats_record_rejections_and_share_evaluations(self):
        evaluated, events = [], []
        proposals = iter(["Lowercase.", "Lowercase.", None, "Lowercase. ", "Upper."])

        def predict(prompt, input):
            evaluated.append(prompt)
            return input.lower() if prompt.strip() == "Lowercase." else input

        def revise(prompt, context):
            proposal = next(proposals)
            if proposal is None:
                raise ProposalRejected("The editor returned nothing usable.")
            self.assertEqual(context.parents[0].id, context.parent.id)
            return proposal

        def pair(context):
            ranked = sorted(context.history, key=lambda trial: (-trial.score, trial.id))
            self.assertTrue(all(len(trial.case_scores) == 1 for trial in context.history))
            return [ranked[0].id, ranked[1].id] if len(ranked) > 1 else ranked[0].id

        self.registry.register("target", "custom-prompt", PromptTarget(predict, "Echo."), name="Agent", description="A test agent.")
        self.registry.register("proposer", "custom-editor", PromptEditor(revise), name="Editor", description="A test editor.")
        self.registry.register("search", "pair", pair, name="Pair", description="Best two parents.")
        result = run({
            **self.configs["greedy-prompt"],
            "target": "custom-prompt", "proposer": "custom-editor", "search": "pair", "budget": 6,
            "task": {"train": [{"input": "ONE", "expected": "one"}], "holdout": [{"input": "TWO", "expected": "two"}]},
        }, self.registry, Controls(
            skip_duplicates=True, record_rejections=True, observe=events.append,
            evaluation_key=lambda candidate: candidate.strip(),
        ))
        self.assertEqual([trial["status"] for trial in result["trials"]],
                         ["evaluated", "evaluated", "duplicate", "rejected", "evaluated", "evaluated"])
        self.assertEqual(result["trials"][3]["note"], "The editor returned nothing usable.")
        self.assertEqual(result["trials"][2]["parent_ids"], [1, 0])
        # "Lowercase. " shares the evaluation of "Lowercase."; the holdout check runs the winner once.
        self.assertEqual(evaluated, ["Echo.", "Lowercase.", "Upper.", "Lowercase."])
        self.assertEqual(result["evaluations_used"], 3)
        self.assertEqual(result["best_trial_id"], 1)
        self.assertEqual([event["stage"] for event in events if event["stage"] != "propose"],
                         ["baseline_train", "candidate_train", "candidate_train"])
        # Without controls, a rejected proposal still ends the run.
        proposals = iter([None])
        with self.assertRaises(ProposalRejected):
            run({**self.configs["greedy-prompt"], "target": "custom-prompt", "proposer": "custom-editor", "budget": 2,
                 "task": {"train": [{"input": "ONE", "expected": "one"}]}}, self.registry)

    def test_prompt_adapter_uses_training_feedback_and_scores_only_winner_on_holdout(self):
        calls = []

        def predict(prompt, input):
            calls.append((prompt, input))
            return input.lower() if prompt == "Lowercase." else input

        def revise(prompt, context):
            self.assertEqual([(row.input, row.score) for row in context.parent.feedback], [("ONE", 0)])
            return "Lowercase."

        self.registry.register("target", "custom-prompt", PromptTarget(predict, "Echo."), name="Agent", description="A test agent.")
        self.registry.register("proposer", "custom-editor", PromptEditor(revise), name="Editor", description="A test editor.")
        result = run({
            **self.configs["greedy-prompt"],
            "target": "custom-prompt", "proposer": "custom-editor", "budget": 2,
            "task": {"train": [{"input": "ONE", "expected": "one"}], "holdout": [{"input": "TWO", "expected": "two"}]},
        }, self.registry)
        self.assertEqual(result["holdout_score"], 1)
        self.assertEqual(calls, [("Echo.", "ONE"), ("Lowercase.", "ONE"), ("Lowercase.", "TWO")])
