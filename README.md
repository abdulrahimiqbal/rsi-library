# RSI Library

RSI Library improves prompts with a simple rule: **an optimizer may only change your prompt if the change holds up on cases it never saw.**

It has two parts:

- **The gateway** ([rsilibrary.xyz](https://rsilibrary.xyz)) runs metered improvement jobs on real models. You bring a prompt or a short prompt chain, training cases, and holdout cases. An optimizer revises the prompt using the training cases only. The gateway then scores the original and the best revision on the same holdout cases and runs a paired significance test. The revision replaces your prompt only when the gain is significant.
- **The loop library and workbench** describe every improvement method as the same loop with swappable parts: the **target** being changed, the **proposer** that edits it, the **judge** that scores it, the **search** that picks which trial to build on, and the **harness** that runs the tests. The gateway's optimizers are loops on this engine.

## Use it from your assistant

Everything on [rsilibrary.xyz](https://rsilibrary.xyz) works without an account, for people and for AI agents.

- **Tell your assistant in plain words.** "Go to rsilibrary.xyz/llms.txt and follow it. Test every loop on my prompt with these cases and tell me exactly what to change." The page at `/llms.txt` explains every loop and call.
- **Or connect it once.** Add `https://rsilibrary.xyz/mcp` as a custom connector (remote MCP server, no sign-in). It offers `list_loops`, `run_offline_loop`, `compare_offline_loops`, `start_prompt_run`, `submit_work` and `get_run`.

There are two kinds of loop:

| Kind | Loops | Who does the thinking |
| --- | --- | --- |
| No model | `greedy-prompt`, `archive-prompt`, `adaptive-improver` | Nobody: they improve a small routing prompt for a built-in interpreter, on the server, instantly. |
| Your prompt | `reflect-v1`, `evolve-v1`, `self-revise-v1` | The caller. The server runs the loop and hands out work items ("answer this prompt for this input", "propose a revision"). Your assistant, or your own OpenRouter key in the browser, answers them. The server scores the answers, keeps the holdout labels, and applies the acceptance rule below. |

```sh
curl -s https://rsilibrary.xyz/api/catalog                      # every loop and its parts
curl -s "https://rsilibrary.xyz/api/run?loop=archive-prompt&budget=30"
curl -s "https://rsilibrary.xyz/api/compare?loops=all"
curl -s https://rsilibrary.xyz/api/runs -H 'Content-Type: application/json' \
  -d '{"loops": "all", "prompt": "…", "task": {"train": […], "holdout": […]}}'
```

A run started with `POST /api/runs` replies with work items. Answer them and post the answers to `/api/runs/{id}/work` until its state is `done`; the result has a verdict for each loop and a recommendation. Runs are kept for 14 days and are reachable only by their ID.

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
python3 examples/gateway_benchmark.py --smoke   # against a local demo gateway
```

The package has no third-party dependencies.
