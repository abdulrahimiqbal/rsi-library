# RSI Library

RSI Library improves prompts and small functions with a simple rule: **an optimizer may only change your prompt or function if the change holds up on cases it never saw.**

It has two parts:

- **The gateway** ([rsilibrary.xyz](https://rsilibrary.xyz)) runs metered improvement jobs on real models. You bring a prompt or a short prompt chain, training cases, and holdout cases. An optimizer revises the prompt using the training cases only. The gateway then scores the original and the best revision on the same holdout cases and runs a paired significance test. The revision replaces your prompt only when the gain is significant.
- **The loop library and workbench** describe every improvement method as the same loop with swappable parts: the **target** being changed, the **proposer** that edits it, the **judge** that scores it, the **search** that picks which trial to build on, and the **harness** that runs the tests. The gateway's optimizers are loops on this engine.

## Use it from your assistant

Everything on [rsilibrary.xyz](https://rsilibrary.xyz) works without an account, for people and for AI agents.

- **Tell your assistant in plain words.** "Go to rsilibrary.xyz/llms.txt and follow it. Test every loop on my prompt with these cases and tell me exactly what to change." The page at `/llms.txt` explains every loop and call.
- **Or connect it once.** Add `https://rsilibrary.xyz/mcp` as a custom connector (remote MCP server, no sign-in). It offers `list_loops`, `run_offline_loop`, `compare_offline_loops`, `start_prompt_run`, `start_code_run`, `submit_work`, `get_run` and `delete_run`.
- **Watch it work.** Every run has a page, `https://rsilibrary.xyz/runs/<run id>`, linked from the run's `page` field. It follows the run live (what each loop is doing, its best training score so far, every version it scored) and then shows the verdicts and the version to adopt. From the page you can continue the run with your own OpenRouter key, or delete it.

The homepage is a canvas: the loop is drawn as a ring of five parts, templates load from a sidebar, selecting a part changes it, and a live panel shows each version as the loop runs. Below it, measured results compare the loops on the same tasks, and a gallery holds fact sheets for published methods (OPRO, EvoPrompt, Promptbreeder, DSPy MIPROv2, TextGrad, GEPA, ADAS, AlphaEvolve, the Darwin Gödel Machine and others) in the same five parts, with filters, a side-by-side comparison, and the closest loop you can run here. Any of them opens on the canvas as a read-only loop. The fact sheets live in `rsi_library/gateway/methods_gallery.py`, were checked against each paper on 10 November 2026, and are served in the `gallery` field of `/api/catalog`.

There are two kinds of runnable loop:

| Kind | Loops | Who does the thinking |
| --- | --- | --- |
| No model | `greedy-prompt`, `archive-prompt`, `adaptive-improver` | Nobody: they improve a small routing prompt for a built-in interpreter, on the server, instantly. |
| Your prompt | `reflect-v1`, `evolve-v1`, `self-revise-v1` | The caller. The server runs the loop and hands out work items ("answer this prompt for this input", "propose a revision"). Your assistant, or your own OpenRouter key in the browser, answers them. The server scores the answers, keeps the holdout labels, and applies the acceptance rule below. |
| Your function | `reflect-v1`, `evolve-v1` | The caller again, who also runs the code. Work items of kind `run` ask for `entry(input)` to be called in the caller's sandbox. **The server never executes code.** The homepage runs JavaScript in a fresh web worker per call, under a policy that blocks every network request, with a 3-second limit; Python goes to your assistant. |

```sh
curl -s https://rsilibrary.xyz/api/catalog                      # every loop and its parts
curl -s "https://rsilibrary.xyz/api/run?loop=archive-prompt&budget=30"
curl -s "https://rsilibrary.xyz/api/compare?loops=all"
curl -s https://rsilibrary.xyz/api/runs -H 'Content-Type: application/json' \
  -d '{"loops": "all", "prompt": "…", "task": {"train": […], "holdout": […]}}'
```

A run started with `POST /api/runs` replies with work items. Answer them and post the answers to `/api/runs/{id}/work` until its state is `done`; the result has a verdict for each loop and a recommendation. `GET /api/runs/{id}?work=0` watches a run without its work items, and `DELETE /api/runs/{id}` deletes it. Unfinished runs are kept for 14 days after their last step and finished runs for 90 days. A run is reachable only by its ID, so treat its link like a password.

A function goes in as `"artifact": {"type": "program", "language": "python" | "javascript", "entry": "toSeconds", "source": "…"}`. It is called with each case's input as one text argument, and its result (text as it is, anything else as JSON) is scored against `expected`.

### Scoring

Choose how each reply is scored with `objective`, and tune it with `scoring`:

| Objective | Scores a reply by | `scoring` |
| --- | --- | --- |
| `normalized` (default here) | Equality, ignoring case and spacing | |
| `exact` | Exact equality | |
| `number` | The last number in the reply, compared with the expected number | `tolerance`, `relative_tolerance` |
| `contains` | Whether it contains the expected text | |
| `json` | Each checked field of a JSON object, with partial credit | `fields` |
| `judge` | A model's 0-10 grade against a rubric, with the expected output as a reference. One more model call per case. | `rubric` |

Expected outputs are checked when the run starts: a `number` run needs a number in every expected output, a `json` run a JSON object. Partial credit and repeated tries average per case, and the significance test compares those per-case scores.

## Measured results

`python -m rsi_library.benchmarks` runs every offline loop on four labelled tasks with 20 seeds at budgets of 12, 24 and 48 versions, and reports how far the winner beat the starting prompt on held-back examples. The numbers are exact and shipped in `rsi_library/benchmark_results.json`; a test recomputes part of the grid so they can't drift from the code. At 24 versions, greedy edits gain 34.7 points on average, archive search 31.5, and the adaptive improver 29.2: on these keyword tasks, exploring older versions costs more than it finds.

The loops that improve your own prompt depend on the model that answers, so they are measured with a live model: `OPENROUTER_API_KEY=… python examples/lab_benchmark.py --model <model> --out driven.json`, then `python -m rsi_library.benchmarks --write --driven driven.json` to publish the results on the site. No live-model results are published yet.

## The acceptance rule

Every gateway job ends with evidence, not just a score:

1. Optimizers see training cases only. Holdout cases never enter the loop.
2. The gateway scores the original and the candidate on every holdout case itself; optimizer-reported scores are ignored.
3. Each case where exactly one version is right is a vote for that version. A one-sided exact sign test on those votes (McNemar's test for single tries) gives a p-value.
4. The candidate is **improved** when it scores higher and p ≤ 0.05. A higher score with p > 0.05 is **inconclusive**, and the original is kept. Anything else is **not improved**.

Jobs need at least 8 holdout cases, because with fewer even a perfect sweep can't reach significance. More cases detect smaller gains. With `holdout_repeats`, each case is scored 2–5 times and the tries are averaged, which helps with models that answer inconsistently.

## Use hosted jobs

Hosted jobs run on the gateway's own models and need a workspace key. Request one at [rsilibrary.xyz/jobs](https://rsilibrary.xyz/jobs), connect with it, and queue a job from the workbench. Scripts and agents can use the same API:

```sh
curl https://rsilibrary.xyz/v1/jobs \
  -H "Authorization: Bearer $RSI_WORKSPACE_KEY" \
  -H "Idempotency-Key: first-job" \
  -H "Content-Type: application/json" \
  -d @job.json
```

See the [gateway guide](deployment/README.md) for the request format, the operator commands, and hosting your own gateway. To try everything locally with a simulated model and no spend:

```sh
python3 -m rsi_library.gateway --db rsi-data/demo.db demo --open
```

The command prints a workspace key for the local demo. That key can also review access requests in the operator panel.

## Use the loop library

The workbench runs offline loops on a small routing task. On macOS, double-click **Start RSI Library.command**, or run it with Python 3.10 or later:

```sh
python3 -m rsi_library serve
```

Open [the local workbench](http://127.0.0.1:8765). No account, key, or install is needed. Pick a template and run it, or fork it and swap a component. Pin a configuration as your control to compare changes to its proposer or search. Save the experiment as JSON to run it again or share it.

| Template | What it changes | How it searches |
| --- | --- | --- |
| Greedy prompt | A text instruction for an offline classifier | Proposes edits and builds on the best candidate |
| Archive prompt | The same text instruction | Builds on a selection of past candidates |
| Adaptive improver | The instruction and settings that guide its future edits | Retains candidates whose editing settings can steer their descendants |

The templates are runnable, simplified experiments inspired by [OPRO](https://arxiv.org/abs/2309.03409), [EvoPrompt](https://arxiv.org/abs/2309.08532), and the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954). They run offline and do not reproduce those papers. The adaptive template changes an explicit proposal policy carried in each candidate; it does not execute generated code.

The same loops run from a script or another agent:

```sh
python3 -m rsi_library catalog
python3 -m rsi_library fork greedy-prompt -o experiment.json
python3 -m rsi_library run experiment.json -o run.json
python3 -m rsi_library compare greedy-prompt archive-prompt -o comparison.json
```

To add a component, create a registry with `make_registry()`, register your implementation with `registry.register(kind, id, component, name=..., description=...)`, and pass the registry to `run` or `compare`. The [custom task example](examples/custom_task.py) registers a title formatter and a policy editor. For an existing agent, `rsi_library.prompt.PromptTarget(predict, initial_prompt)` and `PromptEditor(revise)` adapt two functions to the loop.

`run(config, registry, controls)` accepts optional `Controls` for callers that pay for each step: skip repeated proposals, record unusable ones instead of stopping, share evaluations between equivalent candidates, and observe progress. A search may return up to four parent IDs, and a harness may reveal training labels to the proposer through `Feedback.expected`. The gateway uses all of these.

## Develop

```sh
python3 -m unittest discover -s tests
python3 -m rsi_library.benchmarks               # the offline results; add --write to publish them
python3 examples/gateway_benchmark.py --smoke   # against a local demo gateway
```

The package has no third-party dependencies. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to add a method to the gallery, a runnable loop, or a scoring objective.
