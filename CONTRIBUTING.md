# Contributing to RSI Library

Thank you for helping. There are three common contributions, from smallest to largest: a method for the gallery, a scoring objective, and a runnable loop. Each one is described below, followed by how to test and send your change.

If you're not sure where something belongs, open an issue first. To suggest a method without writing code, use the [suggest a method](https://github.com/abdulrahimiqbal/rsi-library/issues/new?template=suggest-method.yml) form.

## Add a method to the gallery

The gallery describes published self-improvement methods in the same five parts as the loops on the site. Each entry is one call to `_entry` in [`rsi_library/gateway/methods_gallery.py`](rsi_library/gateway/methods_gallery.py):

| Field | What to write |
| --- | --- |
| `id`, `name`, `year`, `origin` | A short slug, the method's name, the year of its first paper, and the authors' institutions as the paper lists them. |
| `family` | What it changes, from `FAMILIES`: Prompt, Pipeline, Memory and skills, Code, Agent itself, or Weights. |
| `search_style` | Single line (always builds on one current version), Population (keeps a generation of candidates), or Archive (keeps everything and can branch from anything). |
| The five parts | One sentence each for the target (what changes), proposer (who writes the next version, and from what), judge (how a version is scored), search (which version is built on next), and harness (what each version is tested on). |
| `improver` | Whether the method rewrites the thing that does the improving: start with Yes, Partly, or No, then say how. |
| `source` | A link to the paper's abstract page, not a PDF. |
| `closest` | The runnable loop here that works most like it, or nothing. |
| `labels` | Two to four words for each part, shown on the canvas. |

Write from the paper, not from summaries of it, and describe what the method does rather than how well it did. Plain sentences are best: someone comparing two methods should see exactly where they differ. When you check or change entries, update `CHECKED` to the date you read the papers.

## Add a scoring objective

Objectives live in `OBJECTIVES` in [`rsi_library/gateway/methods.py`](rsi_library/gateway/methods.py). An objective needs:

1. An entry in `OBJECTIVES` with a name, a one-sentence description, and the settings it accepts.
2. A branch in `_Scorer.one` that returns a score from 0 to 1 for one reply. Keep it deterministic: the same reply and expected output must always score the same, or runs stop being reproducible.
3. A check in `_scoring` that rejects unusable settings and expected outputs when a run starts, with a message that says what to fix.
4. Tests in [`tests/test_runs.py`](tests/test_runs.py) for scoring and validation.

An objective that needs a model, like `judge`, makes its calls in `_Scorer.scores` after every reply in the split has been produced, so callers who answer work items receive them as one batch.

## Add a runnable loop

A loop is a configuration of five registered components; see [`examples/custom_task.py`](examples/custom_task.py) and the engine's docstrings in [`rsi_library/engine.py`](rsi_library/engine.py).

- **An offline loop** runs on the server with no model. Register its components in `make_registry` in [`rsi_library/builtins.py`](rsi_library/builtins.py) and add a template to `TEMPLATES`. Then run `python -m rsi_library.benchmarks --write` so the measured results include it, and add a line for it to `_BEST_FOR` and the gallery's `_RUNNABLE` table.
- **A model-driven loop** improves a caller's prompt or function. Add it to `_METHODS` and `_SEARCHES` in [`rsi_library/gateway/methods.py`](rsi_library/gateway/methods.py) with its proposer instructions in `_proposal_messages`. Hosted jobs and public runs share this code, so a new loop must keep the acceptance rule: holdout cases never enter the loop, and only a significant paired gain replaces the original.

Two rules apply to every loop. The server never executes generated code: a function is always run by the caller. And a loop must be deterministic for a given seed and set of answers, because public runs are replayed from their stored answers.

## Test and send your change

```sh
python3 -m unittest discover -s tests
```

The package has no third-party dependencies and supports Python 3.10 and later. Tests run on 3.10, 3.12 and 3.13 for every pull request. For changes to the site, also start the local demo and try them in a browser:

```sh
python3 -m rsi_library.gateway --db rsi-data/demo.db demo --open
```

Keep pull requests focused on one change, and say in the description what you checked. Don't commit keys, tokens, or `.db` files.

## Report a security problem

Please don't open a public issue for a vulnerability. Use GitHub's [private vulnerability reporting](https://github.com/abdulrahimiqbal/rsi-library/security/advisories/new) instead.
