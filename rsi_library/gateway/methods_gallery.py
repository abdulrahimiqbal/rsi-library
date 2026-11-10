"""Fact sheets for published self-improvement methods, in the same five parts as the loops here.

Each entry describes one method in this library's vocabulary so methods can be compared
part by part. Entries are summaries of the cited papers, not reimplementations. `closest`
names the runnable loop here that works most like it, when there is one. `labels` are
short names for the five parts, used where the full text doesn't fit.

The titles, years, affiliations, links and mechanisms were checked against each paper's
abstract page on the date in CHECKED.
"""

from __future__ import annotations

FAMILIES = ("Prompt", "Pipeline", "Memory and skills", "Code", "Agent itself", "Weights")
SEARCH_STYLES = ("Single line", "Population", "Archive")
CHECKED = "2026-11-10"
_KINDS = ("target", "proposer", "judge", "search", "harness")


def _entry(id, name, year, origin, family, search_style, target, proposer, judge, search, harness, improver, source,
           closest=None, labels=()):
    assert len(labels) == 5, id
    return {
        "id": id, "name": name, "year": year, "origin": origin, "family": family, "search_style": search_style,
        "parts": {"target": target, "proposer": proposer, "judge": judge, "search": search, "harness": harness},
        "labels": dict(zip(_KINDS, labels)),
        "rewrites_improver": improver, "source": source, "closest": closest, "runnable": False,
    }


METHODS = [
    _entry(
        "ape", "APE: Automatic Prompt Engineer", 2022, "University of Toronto, Vector Institute, University of Waterloo", "Prompt", "Population",
        "One instruction for a task.",
        "A model writes many candidate instructions from input and output examples, then paraphrases the best.",
        "Accuracy or likelihood of the expected outputs on held examples.",
        "Generate widely, keep the top scorers, resample around them.",
        "Labelled input and output pairs.",
        "No. The way candidates are written is fixed.",
        "https://arxiv.org/abs/2211.01910", "greedy-prompt",
        ("One instruction", "Generate, then resample", "Execution accuracy", "Keep the top scorers", "Input-output pairs"),
    ),
    _entry(
        "reflexion", "Reflexion", 2023, "Northeastern, MIT, Princeton", "Memory and skills", "Single line",
        "The agent's memory of written lessons from past attempts.",
        "After a failed attempt the model writes a short reflection on what went wrong.",
        "Tests, environment signals, or a model acting as evaluator.",
        "Retry the same task with the reflections added to context.",
        "Repeated attempts at a task with feedback after each.",
        "No. Only the memory changes.",
        "https://arxiv.org/abs/2303.11366",
        labels=("Memory of lessons", "Self-reflection", "Task feedback", "Retry with memory", "Repeated attempts"),
    ),
    _entry(
        "protegi", "ProTeGi: prompt optimization with textual gradients", 2023, "Microsoft", "Prompt", "Population",
        "One prompt.",
        "A model reads a batch of errors, writes a critique of the prompt, then edits the prompt to answer the critique.",
        "A metric on labelled examples.",
        "Beam search over edited prompts, with bandit selection to spend evaluations on promising ones.",
        "Minibatches of labelled examples.",
        "No. The critique and edit instructions are fixed.",
        "https://arxiv.org/abs/2305.03495", "reflect-v1",
        ("One prompt", "Textual gradients", "Labelled metric", "Beam search, bandits", "Minibatches"),
    ),
    _entry(
        "voyager", "Voyager", 2023, "NVIDIA, Caltech, UT Austin, Stanford, UW Madison", "Memory and skills", "Archive",
        "A growing library of code skills.",
        "A model writes a new skill as code for the next task in a curriculum it sets for itself.",
        "The environment's result plus a model that checks whether the task was completed.",
        "Keep every verified skill and retrieve relevant ones for later tasks.",
        "An open-ended game world (Minecraft).",
        "No. The agent's own prompts and code are fixed; its skills grow.",
        "https://arxiv.org/abs/2305.16291",
        labels=("Skill library", "Code writer", "Self-verification", "Keep verified skills", "Minecraft"),
    ),
    _entry(
        "opro", "OPRO: Optimization by Prompting", 2023, "Google DeepMind", "Prompt", "Single line",
        "One prompt.",
        "A model sees earlier prompts with their scores, sorted, and writes new prompts that should score higher.",
        "Accuracy on training examples.",
        "Keep a running list of the best prompts so far inside the proposer's context.",
        "Labelled training examples, with a test set kept for the final report.",
        "No. The meta-prompt is fixed.",
        "https://arxiv.org/abs/2309.03409", "reflect-v1",
        ("One prompt", "Scored history", "Training accuracy", "Best so far, in context", "Train and test split"),
    ),
    _entry(
        "evoprompt", "EvoPrompt", 2023, "Tsinghua University, Microsoft Research", "Prompt", "Population",
        "A population of prompts.",
        "A model performs crossover and mutation on parent prompts, following a genetic algorithm or differential evolution.",
        "A metric on a development set.",
        "Evolve the population generation by generation and keep the fittest.",
        "A labelled development set.",
        "No. The evolutionary operators are fixed.",
        "https://arxiv.org/abs/2309.08532", "evolve-v1",
        ("Prompt population", "Crossover, mutation", "Dev-set metric", "GA or DE", "Development set"),
    ),
    _entry(
        "promptbreeder", "Promptbreeder", 2023, "Google DeepMind", "Prompt", "Population",
        "Task prompts, and the mutation prompts used to change them.",
        "A model mutates a task prompt by following a mutation prompt, and also mutates the mutation prompts.",
        "Fitness on a batch of training examples.",
        "Evolve a population in which each member carries its own mutation prompt.",
        "Labelled training examples.",
        "Yes. The instructions that guide edits are themselves evolved.",
        "https://arxiv.org/abs/2309.16797", "self-revise-v1",
        ("Task and mutation prompts", "Mutation prompts", "Batch fitness", "Binary tournament", "Training batches"),
    ),
    _entry(
        "stop", "STOP: Self-Taught Optimizer", 2023, "Microsoft Research, Stanford, OpenAI", "Agent itself", "Single line",
        "The improver program itself: code that calls a model to improve programs.",
        "The current improver, applied to its own source code.",
        "How well the rewritten improver improves a set of downstream programs.",
        "Whatever the improver writes for itself; it arrived at strategies such as beam search and genetic search.",
        "Downstream programming tasks, with the model's weights fixed.",
        "Yes. The improver's code is the thing being improved.",
        "https://arxiv.org/abs/2310.02304",
        labels=("The improver", "Improver on itself", "Meta-utility", "Chosen by the improver", "Downstream tasks"),
    ),
    _entry(
        "funsearch", "FunSearch", 2023, "Google DeepMind", "Code", "Archive",
        "One function inside a program skeleton.",
        "A model writes new versions of the function after seeing a few high-scoring programs.",
        "An automatic evaluator that runs each program and scores it.",
        "A database of programs split into islands that evolve separately and are periodically reset.",
        "A problem with a programmatic scoring function.",
        "No. The search procedure is fixed.",
        "https://www.nature.com/articles/s41586-023-06924-6",
        labels=("One function", "Best-shot prompting", "Program evaluator", "Island database", "Scored problem"),
    ),
    _entry(
        "mipro", "DSPy MIPROv2", 2024, "Stanford, UC Berkeley, KTH, Basis", "Pipeline", "Population",
        "The instructions and few-shot examples of every step in a multi-step program.",
        "A model drafts instructions from summaries of the data and program; examples are collected from successful runs.",
        "A metric the developer defines, computed on minibatches.",
        "Bayesian optimization over combinations of instructions and examples.",
        "A training set run through the whole program.",
        "Partly. The paper also refines how instructions are proposed over time; the search itself is fixed.",
        "https://arxiv.org/abs/2406.11695",
        labels=("Instructions and examples", "Grounded proposals", "Developer metric", "Bayesian optimization", "Minibatches"),
    ),
    _entry(
        "textgrad", "TextGrad", 2024, "Stanford", "Pipeline", "Single line",
        "Any text in a system: prompts, answers, code.",
        "A model writes criticism of each piece of text, passed backwards through the system, and a second call rewrites the text to address it.",
        "A loss described in words or computed as a metric.",
        "Step by step from the current version, like gradient descent.",
        "The system's own runs on examples.",
        "No. The critic and editor prompts are fixed.",
        "https://arxiv.org/abs/2406.07496", "reflect-v1",
        ("Any text", "Textual backpropagation", "Loss in words", "Gradient steps", "The system's runs"),
    ),
    _entry(
        "adas", "ADAS: Automated Design of Agentic Systems", 2024, "University of British Columbia, Vector Institute", "Code", "Archive",
        "Whole agent designs written as code.",
        "A meta agent writes a new agent after reading the archive of earlier ones.",
        "Accuracy of each new agent on benchmark tasks.",
        "Keep an archive of discovered agents and build on it.",
        "Benchmark tasks; discovered agents are then tested on other domains and models.",
        "No. The meta agent that writes agents stays fixed.",
        "https://arxiv.org/abs/2408.08435",
        labels=("Agent code", "Meta agent", "Benchmark accuracy", "Agent archive", "Benchmarks"),
    ),
    _entry(
        "godel-agent", "Gödel Agent", 2024, "Peking University, UC Santa Barbara, University of Arizona", "Agent itself", "Single line",
        "The agent's own running logic, including the code that performs the self-modification.",
        "The agent, which reads and rewrites its own code at run time.",
        "Performance on the task after each change.",
        "Continue from the current self; the agent decides what to change next.",
        "Task benchmarks.",
        "Yes. The self-improvement routine can rewrite itself.",
        "https://arxiv.org/abs/2410.04444",
        labels=("Its own logic", "Self-rewrite", "Task score", "Current self", "Benchmarks"),
    ),
    _entry(
        "sica", "SICA: Self-Improving Coding Agent", 2025, "University of Bristol, iGent AI", "Agent itself", "Archive",
        "The coding agent's own codebase.",
        "The best agent so far, asked to improve its own code.",
        "Benchmark score weighed against cost and time.",
        "Keep an archive of agent versions and let the best one make the next change.",
        "Coding benchmarks run in a sandbox.",
        "Yes. The agent that edits is the agent being edited.",
        "https://arxiv.org/abs/2504.15228",
        labels=("Its own codebase", "Best agent edits itself", "Score, cost and time", "Agent archive", "Coding benchmarks"),
    ),
    _entry(
        "dgm", "Darwin Gödel Machine", 2025, "University of British Columbia, Vector Institute, Sakana AI", "Agent itself", "Archive",
        "A coding agent's own code.",
        "An agent chosen from the archive rewrites itself to produce a child.",
        "Measured results on coding benchmarks rather than proof.",
        "An archive of every working agent; parents are chosen by score and by how few children they have.",
        "Coding benchmarks, run in sandboxes with human oversight.",
        "Yes. Each child is a changed copy of the agent that made it.",
        "https://arxiv.org/abs/2505.22954", "archive-prompt",
        ("Its own code", "Self-modification", "Coding benchmarks", "Open-ended archive", "Sandboxed runs"),
    ),
    _entry(
        "alphaevolve", "AlphaEvolve", 2025, "Google DeepMind", "Code", "Archive",
        "Marked regions of a codebase, up to whole programs.",
        "A fast model and a stronger model propose code changes as diffs.",
        "Automatic evaluators that run the code, cheapest checks first.",
        "An evolutionary database that keeps both strong and diverse programs.",
        "Problems with machine-checkable scoring.",
        "Partly. Its prompts can evolve alongside the programs, and it has been used to speed up training of the model it runs on.",
        "https://arxiv.org/abs/2506.13131",
        labels=("Marked code", "Model ensemble diffs", "Evaluation cascade", "Evolutionary database", "Machine-checkable"),
    ),
    _entry(
        "seal", "SEAL: Self-Adapting Language Models", 2025, "MIT", "Weights", "Single line",
        "The model's own weights.",
        "The model writes its own training data and update instructions.",
        "How the model performs after being updated on what it wrote.",
        "Reinforcement learning rewards the kinds of self-written updates that helped.",
        "Tasks with a measurable result after the update.",
        "Partly. The model learns to write better updates for itself.",
        "https://arxiv.org/abs/2506.10943",
        labels=("Model weights", "Self-edits", "Score after update", "RL on self-edits", "Adaptation tasks"),
    ),
    _entry(
        "gepa", "GEPA: reflective prompt evolution", 2025, "UC Berkeley, Stanford, Databricks, MIT", "Pipeline", "Population",
        "The prompts of each step in a multi-step system.",
        "A model reads run traces and feedback in words, diagnoses the failure, and rewrites one step's prompt; candidates can also be merged.",
        "A metric plus written feedback for each example.",
        "Keep every candidate that is best on at least one example, and sample parents from that set.",
        "A training set run through the whole system, with a separate validation set.",
        "No. The reflection instructions are fixed.",
        "https://arxiv.org/abs/2507.19457", "evolve-v1",
        ("Step prompts", "Reflective mutation", "Metric and feedback", "Pareto front", "Train and validation"),
    ),
]

_RUNNABLE = {
    "greedy-prompt": ("Single line", "No. The edit rule is fixed."),
    "archive-prompt": ("Archive", "No. The edit rule is fixed."),
    "adaptive-improver": ("Archive", "Yes, in miniature. Each version carries the settings that steer its descendants' edits."),
    "reflect-v1": ("Single line", "No. The editing instructions are fixed."),
    "evolve-v1": ("Population", "No. The editing instructions are fixed."),
    "self-revise-v1": ("Single line", "Yes. Each version carries editing instructions that it can rewrite."),
}


def gallery(loops: list[dict]) -> list[dict]:
    """Runnable loops first, then published methods in order of year."""
    entries = []
    for loop in loops:
        style, improver = _RUNNABLE[loop["id"]]
        entries.append({
            "id": loop["id"], "name": loop["name"], "year": 2026, "origin": "RSI Library",
            "family": "Prompt", "search_style": style,
            "parts": {kind: f"{part['name']}. {part['description']}" for kind, part in loop["parts"].items()},
            "labels": {kind: part["name"] for kind, part in loop["parts"].items()},
            "rewrites_improver": improver, "source": loop.get("inspired_by"), "closest": loop["id"],
            "runnable": True, "mode": loop["mode"], "best_for": loop["best_for"],
        })
    return entries + [dict(entry) for entry in METHODS]
