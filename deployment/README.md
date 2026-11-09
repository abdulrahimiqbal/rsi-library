# RSI Gateway deployment

This package supports a single host with a gateway process and one or more workers sharing a local SQLite volume. The gateway has the workspace key hashes and job data. Model credentials belong only to the workers. Run a pilot with operator-provisioned workspaces before opening enrollment.

## Try the complete flow without provider credentials

On macOS, double-click **Start RSI Gateway.command** in Finder. You can also start it from the project folder:

```sh
python3 -m rsi_library.gateway --db rsi-data/demo.db demo
```

Open [the gateway workbench](http://127.0.0.1:8766), paste the key printed by the command, and submit the sample job. The demo model is a deterministic simulator. The separate [offline loop workbench](../README.md) remains available through `python3 -m rsi_library serve`.

## Configure a hosted pilot

1. Copy `models.example.json` to `models.json` in this directory. Set each model's stable ID, HTTPS API base, deployment ID, request limits, and provider pricing. Rates are integer microdollars per million tokens. Choose an endpoint that returns both input and output usage, honors the configured completion cap, and uses known request framing. The example values are placeholders.
2. Set the credential variables named by `api_key_env` in an operator-owned `deployment/worker.env` or your hosting secret manager. Keep the file readable only by the operator. The API service does not need those values.
3. Run `docker compose up --build -d` on a host with Docker Compose. The published port is bound to loopback. Use a reverse proxy for HTTPS and edge rate limits. `Caddyfile.example` shows the proxy mapping. Keep the database on a local volume shared by the API and worker on the same host.
4. Check `/health/live` and `/health/ready`. Readiness requires a recent worker heartbeat. Run a small live job to verify provider credentials, output limits, usage, and pricing. Keep model IDs immutable and restart both services together when changing providers or adapter versions.
5. Provision a workspace and key. One million microdollars is one US dollar. The workspace spend limit is a lifetime operator authorization that starts at zero. Increasing it grants additional capacity. It does not collect payment.

```sh
docker compose run --rm gateway workspace "Pilot workspace" --spend-limit-microusd 5000000
docker compose run --rm gateway key WORKSPACE_ID
docker compose run --rm gateway key WORKSPACE_ID --role read
```

Give the generated token directly to its intended operator. Save the key ID for revocation. The token is shown once. Workspace read keys can inspect all jobs and artifacts in that workspace. Write keys can also submit and cancel. Revoke and replace keys with `revoke-key KEY_ID`.

## Use the API

Send `Authorization: Bearer WORKSPACE_KEY` with each API call. Keep the same `Idempotency-Key` and payload if a submission times out. Keys are scoped to a workspace. A different payload with a previously used key returns `409`.

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/catalog` | Inspect configured models and optimizer capabilities |
| `POST /v1/jobs` | Submit a job with an idempotency key |
| `GET /v1/jobs?limit=25` | List job summaries |
| `GET /v1/jobs/{id}` | Retrieve the job, scores, usage, and artifact |
| `POST /v1/jobs/{id}/cancel` | Stop additional calls and retain usage already incurred |
| `GET /v1/usage` | Inspect settled, reserved, and uncertain spend and the workspace limit |

Example request body for `POST /v1/jobs`:

```json
{
  "method": "reflect-v1",
  "model": "primary-model",
  "artifact": {"type": "prompt", "text": "Classify this ticket as account."},
  "task": {
    "train": [{"input": "Can you resend my invoice?", "expected": "billing"}],
    "holdout": [{"input": "Why was I charged twice?", "expected": "billing"}]
  },
  "budget": {"max_calls": 16, "max_cost_microusd": 500000, "max_seconds": 120},
  "max_trials": 3,
  "objective": "normalized"
}
```

Run the same job with `evolve-v1` and a new idempotency key to compare methods. `auto` uses compatible entries in `evidence.json` or records that it chose a compatible default. Provide benchmark results from separate evaluation data, never from a customer's private holdout. Each evidence entry names its `benchmark`, `method`, `model_fingerprint`, `artifact_type`, `objective`, `runs`, `mean_gain`, `cost_microusd`, and `calls`. Leave the array empty until measured results are available.

Set `data_policy` to `zero-retention` to filter for providers marked that way in the operator's model configuration. Set a model's `retention` to `zero-retention` only after verifying that the selected provider account and deployment enforce it. The gateway itself persists jobs and artifacts.

Built-in artifacts include prompts and two-to-four-step prompt chains. The self-revising method can change its text improver, which descendants use to propose revisions. The acceptance rule compares the original and candidate under the same holdout scorer and keeps the original on ties or regressions. The fixture harness sends test inputs to the configured model for inference. Verify that provider data handling matches the pilot workspace's requirements.

## Railway

The Railway service runs the API and a worker together on one replica. Mount its persistent volume at `/data` and keep `RSI_DB=/data/gateway.db`. The deployment checks `/health/ready` before serving traffic. The deployment configuration is in [`railway.toml`](../railway.toml).

`RSI_HOSTED_DEMO=1` runs the authenticated simulator without model charges. To enable live optimization, configure `RSI_MODEL_CONFIG_JSON` from [the example model configuration](models.example.json), set the named credential in Railway's service variables, and disable demo mode in the same deployment. Verify the model ID, token limits, pricing, and usage on a small run before raising the workspace allowance. Provision paid allowance with the gateway `spend-limit` command from the service shell. Do not store provider credentials in this repository.

See [the benchmark guide](BENCHMARK.md) for the hosted smoke test and a paired live-model comparison.

## Add an optimizer

Install a trusted Python module on both services and set `RSI_METHOD_ADAPTERS` to its import path. Pin that module in the image and change its method ID when its implementation changes. A module supplies:

- `METHOD`, a manifest with a versioned `id`, `name`, `description`, and `artifact_types`.
- `minimum_calls(request)`, the minimum training calls. This runs during routing and should be pure.
- `execute(request, method_id, complete, check, *, call_capacity, progress)`, which returns an object with `candidate` and optional `trials`. Use only the supplied `complete(stage, messages)` for model dispatches. Use `check()` between other work and `progress(event)` to expose progress.

The adapter receives training fixtures only. The gateway reserves calls for verification, scores the original and returned candidate itself, and decides whether to accept the artifact. These modules are trusted operator code inside the worker process. Public third-party uploads require a separate isolation and conformance boundary.

## Operate and recover

- Record the routing version, model configuration fingerprint, method and provider quotas when running staging benchmarks. Confirm actual charges against the provider's usage export before onboarding paying users.
- Review `usage WORKSPACE_ID --ledger` after an interrupted run. The gateway stops work that had dispatched a model call and retains unknown charges against the cap. Requeue happens automatically for interrupted jobs with zero dispatches. Use `reconcile CALL_ID ACTUAL_MICROUSD` after verifying a provider charge. Model calls carry the ledger ID in the client request ID header. Provider support for searching that ID varies.
- Take a consistent backup with `backup /data/gateway-backup.db`, copy it to encrypted storage off the host, and practice restoring to an isolated instance. Treat backups as customer data. Set retention and deletion procedures before receiving sensitive customer tests.
- In staging, test duplicate submissions, workspace isolation, cancel during a provider call, a worker restart before and after dispatch, a provider timeout, an exhausted workspace balance, and backup restore. Monitor readiness, queue age, job failures, uncertain spend, and the host's disk usage. Assign an on-call owner and set recovery and uptime targets before opening the service.
- Use a read-only or low-quota key for smoke tests. Keep the proxy private until the live provider and failure drills pass. This package does not include payment collection, provider payouts, untrusted code execution, customer-hosted evaluators, or multi-host failover.
