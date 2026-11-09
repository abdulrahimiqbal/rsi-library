# Test RSI Gateway

Use the [benchmark runner](../examples/gateway_benchmark.py) from the project folder with Python 3.10 or later. Supply a workspace write key through `RSI_WORKSPACE_KEY` or `--key-file /path/to/private-key`. A key file must be readable only by its owner. The runner sends the key in the authorization header and excludes it from output. Hosted origins require HTTPS.

## Verify the deployed service without spending

When the Railway deployment is in demo mode, run:

```sh
python3 examples/gateway_benchmark.py --url https://rsilibrary.xyz --smoke
```

For a local check, start `python3 -m rsi_library.gateway demo` and use its workspace key. Then run `python3 examples/gateway_benchmark.py --smoke` in another terminal.

Both optimizers should complete and move from 50% to 100% holdout accuracy at $0. The simulator uses a fixed proposal and label rules. This test checks access, queueing, worker execution, evaluation, job retrieval, and usage reporting. Use a live model to measure optimizer capability.

## Compare two optimizers on a live model

Ask the gateway operator to configure a model and grant your workspace a spend allowance. Use its exact catalog model ID. Run a first pair with one seed and an explicit cap, for example:

```sh
python3 examples/gateway_benchmark.py --url https://rsilibrary.xyz \
  --model YOUR_MODEL_ID --job-cap 0.50 --seeds 11 --output first-pair.json
```

That command submits two serial jobs and authorizes at most $0.50 per job, for a combined cap of $1.00. The gateway also enforces the workspace allowance. The runner checks the catalog model's conservative reservation before submitting. Adjust `--job-cap` or `--trials` if the chosen model requires more headroom. The same model, original artifact, train/holdout cases, seed, and call, time, and spending caps apply to each pair. The optimizers can spend different amounts under those caps.

The [included ticket dataset](../examples/gateway_benchmark_tickets.json) has six training examples and eight balanced holdout examples. It tests whether revisions handle misleading keywords and billing requests that omit the obvious cues. Each holdout case is worth 12.5 percentage points, so use this run to check the process before drawing quality conclusions. A second run with `--seeds 11 22 33` gives three paired observations and six jobs. The seed controls the gateway search order. Live model responses may vary.

Pass `--dataset /path/to/task.json` to use your own artifact and cases in the same JSON format. Keep a separate test set for the final check and keep its labels out of prompts. If you select a method using these benchmark results, test the selected method on fresh cases. Check your model provider's data policy before submitting private inputs.

The table reports the original and candidate scores on the same holdout, gain in percentage points, whether the gateway accepted the candidate, calls, settled cost, and uncertain charges. Examine failed jobs and uncertain charges separately. The runner stops submitting after either. Use the job IDs to review the decision and retrieve the accepted artifact in the workbench.

Save the printed run ID. If the client disconnects, use `--run-id RUN_ID` with the same options and input file to resume and retrieve already submitted jobs. The same request keeps its idempotency key. Choose a new run ID to intentionally rerun the jobs. Reports created with `--output` contain job IDs and metrics but exclude the prompts and fixture text.
