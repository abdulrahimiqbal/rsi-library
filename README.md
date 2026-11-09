# RSI Library

Build, fork, and compare improvement loops. A loop has five parts: the **target** being changed, the **proposer** that edits it, the **judge** that scores predictions, the **search** that picks which trial to build on, and the **harness** that runs the tests. A run records each candidate, its score, and its parent so you can inspect how the winner was found.

## Start the workbench

On macOS, double-click **Start RSI Library.command** in Finder to open the workbench. Keep its Terminal window open while you use it. You can also start it from this folder with Python 3.10 or later:

```sh
python3 -m rsi_library serve
```

Open [the local workbench](http://127.0.0.1:8765). No account, API key, or install is needed. Pick a template and run it, or fork it and change a component. Pin a configuration as your control to compare changes to its proposer or search. Save the experiment as JSON to run it again or share it with someone else.

The starter task routes support tickets into categories. To try your own classification task, replace its examples with a JSON object like this:

```json
{
  "train": [
    {"input": "My card was charged twice", "expected": "billing"},
    {"input": "I forgot my password", "expected": "account"},
    {"input": "Where is my invoice?", "expected": "billing"},
    {"input": "I cannot sign in", "expected": "account"}
  ],
  "holdout": [
    {"input": "Please resend my invoice", "expected": "billing"},
    {"input": "Can I reset my password?", "expected": "account"}
  ]
}
```

Use different examples for `train` and `holdout`. The search gets feedback from training examples; only the selected winner runs on the holdout examples. A result with no holdout set has no holdout score. A score shows performance on the supplied cases, so small fixture sets are best for inspecting how the loop works.

## Templates

| Template | What it changes | How it searches |
| --- | --- | --- |
| Greedy prompt | A text instruction for an offline classifier | Proposes edits and builds on the best candidate |
| Archive prompt | The same text instruction | Builds on a selection of past candidates |
| Adaptive improver | The instruction and settings that guide its future edits | Retains candidates whose editing settings can steer their descendants |

The templates are runnable, simplified experiments inspired by [OPRO](https://arxiv.org/abs/2309.03409), [EvoPrompt](https://arxiv.org/abs/2309.08532), and the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954). They run offline. They do not reproduce those papers or use a hosted language model. The adaptive template changes an explicit proposal policy carried in each candidate; it does not execute generated code.

Each budget counts candidate evaluations on the training set, including the baseline. A paired comparison holds fixed the seed, budget, task, initial candidate, target, judge, and harness. This controls the number of evaluations, which is different from matching elapsed time, tokens, or FLOPs. Report those costs separately when extending the runner with a model.

## Run from a script or another agent

```sh
python3 -m rsi_library catalog
python3 -m rsi_library fork greedy-prompt -o experiment.json
python3 -m rsi_library run experiment.json -o run.json
python3 -m rsi_library compare greedy-prompt archive-prompt -o comparison.json
```

The workbench uses the same local JSON API: `GET /api/catalog`, `POST /api/run` with a configuration, and `POST /api/compare` with `{"configs":[configA,configB]}`. Export a configuration from the workbench or use `fork` to see the full shape. An experiment can include `task`, `initial_candidate`, and per-component `settings` alongside its choice of components, seed, and budget.

To add a component, create a registry with `make_registry()`, register your implementation with `registry.register(kind, id, component, name=..., description=...)`, and pass the registry to `run` or `compare`. A target can call an agent or model client you provide. Use a proposer suited to that target's candidate format. The [custom task example](examples/custom_task.py) registers a title formatter and a policy editor while keeping the supplied judge, search, and harness:

```sh
python3 -m examples.custom_task
python3 -m unittest discover -s tests
```

For an existing agent, `rsi_library.prompt.PromptTarget(predict, initial_prompt)` and `PromptEditor(revise)` adapt two functions to the same loop. Implement `predict(prompt, input)` to run your agent and return its output. Implement `revise(prompt, context)` to produce the next prompt from `context.parent.feedback`; it contains training inputs, predictions, and scores. Register the adapters as a target and proposer, then select their IDs in your run configuration. Your client controls model selection, timeouts, and per-call costs.

To use those components in the local workbench too, call `rsi_library.server.serve(8765, registry_factory=your_registry_factory)`, where the factory returns a fresh registry with the built-in and custom components.

## Durable gateway

Run `python3 -m rsi_library.gateway demo` to try workspace keys, saved optimization jobs, routing, and the spend ledger with a simulated model. See the [gateway deployment guide](deployment/README.md) for the live model configuration, API, operator commands, and staging checks.
