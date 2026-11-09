# Test RSI Gateway

Use the [benchmark runner](../examples/gateway_benchmark.py) from the project folder with Python 3.10 or later. Supply a workspace write key through `RSI_WORKSPACE_KEY` or `--key-file /path/to/private-key`. A key file must be readable only by its owner. The runner sends the key in the authorization header and excludes it from output. Hosted origins require HTTPS.

## Verify a gateway without spending

Against a gateway in demo mode, such as `python3 -m rsi_library.gateway demo` on your machine with its printed key, run:

```sh
python3 examples/gateway_benchmark.py --smoke
```

Add `--url https://your-gateway` for a hosted gateway that is in demo mode. Both optimizers should complete and move from 37.5% to 100% holdout accuracy, with 5 of 8 holdout cases fixed, p = 0.031, an `improved` verdict, and $0 spent. The simulator uses a fixed proposal and label rules. This test checks access, queueing, worker execution, evaluation, the paired test, job retrieval, and usage reporting. Use a live model to measure optimizer capability.

## Compare two optimizers on a live model

You need a workspace with a spend allowance and the exact catalog model ID. Run a first pair with one seed and an explicit cap, for example:

```sh
python3 examples/gateway_benchmark.py --url https://rsilibrary.xyz \
  --model gpt-4-1-mini --job-cap 1.00 --seeds 11 --output first-pair.json
```

That command submits two serial jobs and authorizes at most $1.00 per job, for a combined cap of $2.00. The gateway also enforces the workspace allowance and any site-wide limit. The runner checks the catalog model's conservative per-call reservation before submitting. Adjust `--job-cap` or `--trials` if the chosen model requires more headroom. The same model, original artifact, train/holdout cases, seed, and call, time, and spending caps apply to each pair. The optimizers can spend different amounts under those caps.

The [included ticket dataset](../examples/gateway_benchmark_tickets.json) has six training examples and sixteen balanced holdout examples. It tests whether revisions handle misleading keywords and billing requests that omit the obvious cues. Each holdout case is worth 6.25 percentage points. The table reports each job's original and candidate scores on the same holdout, the gain, the paired test's p-value, the verdict, calls, settled cost, and uncertain charges. Examine failed jobs and uncertain charges separately; the runner stops submitting after either. Use the job IDs to review the decision and retrieve the accepted artifact in the workbench.

Add `--holdout-repeats 3` to score every holdout case three times per version, which steadies results from models that answer inconsistently. It triples the holdout calls.

## Give the auto route measured evidence

The `auto` method picks the optimizer with the best measured gain for the job's model and artifact type, if evidence is configured. Produce that evidence with at least three seeds on data that is separate from any customer's holdout:

```sh
python3 examples/gateway_benchmark.py --url https://rsilibrary.xyz \
  --model gpt-4-1-mini --job-cap 1.00 --seeds 11 22 33 \
  --output tickets-report.json --evidence-out tickets-evidence.json
```

The evidence file has one row per optimizer. `mean_gain` is the gain users would have kept: the holdout gain for jobs with an `improved` verdict and zero for the rest. `cost_microusd` and `calls` are the largest per job, so routing only picks a method whose measured cost fits the job's limits. Paste the file's contents into the `RSI_ROUTING_EVIDENCE_JSON` variable on Railway, or pass the file with `--evidence` or `RSI_ROUTING_EVIDENCE` elsewhere. Rows are tied to the model configuration's fingerprint, so rerun the benchmark after changing a model's settings.

Save the printed run ID. If the client disconnects, use `--run-id RUN_ID` with the same options and input file to resume and retrieve already submitted jobs. The same request keeps its idempotency key. Choose a new run ID to intentionally rerun the jobs. Reports created with `--output` contain job IDs and metrics but exclude the prompts and fixture text.

Pass `--dataset /path/to/task.json` to use your own artifact and cases in the same JSON format, with at least eight holdout cases. Keep a separate test set for the final check and keep its labels out of prompts. If you select a method using these benchmark results, test the selected method on fresh cases. Check your model provider's data policy before submitting private inputs.
