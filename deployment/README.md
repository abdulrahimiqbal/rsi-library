# RSI Gateway deployment

This package supports a single host with a gateway process and one or more workers sharing a local SQLite volume. The gateway has the workspace key hashes and job data. Model credentials belong only to the workers. Run a pilot with operator-provisioned workspaces before opening enrollment.

## Try the complete flow without provider credentials

On macOS, double-click **Start RSI Gateway.command** in Finder. You can also start it from the project folder:

```sh
python3 -m rsi_library.gateway --db rsi-data/demo.db demo
```

Open [the gateway workbench](http://127.0.0.1:8766), paste the key printed by the command, and submit the sample job. The demo model is a deterministic simulator. The demo key is also the operator, so you can try the access request form and review it in the same window. The separate [offline loop workbench](../README.md) remains available through `python3 -m rsi_library serve`.

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
| `GET /v1/catalog` | Inspect configured models, optimizer capabilities, and the evidence policy |
| `POST /v1/jobs` | Submit a job with an idempotency key |
| `GET /v1/jobs?limit=25` | List job summaries, each with its verdict |
| `GET /v1/jobs/{id}` | Retrieve the job, scores, evidence, usage, and artifact |
| `POST /v1/jobs/{id}/cancel` | Stop additional calls and retain usage already incurred |
| `GET /v1/usage` | Inspect settled, reserved, and uncertain spend and the workspace limit |
| `GET /public/info` | Without a key: the gateway mode, optimizers, evidence policy, and whether key requests are open |
| `POST /access-requests` | Without a key: request a workspace key with `email`, `use_case`, and an optional `name` |

Example request body for `POST /v1/jobs`, abbreviated to show one of the eight or more holdout cases a job needs:

```json
{
  "method": "reflect-v1",
  "model": "gpt-4-1-mini",
  "artifact": {"type": "prompt", "text": "Classify this ticket as account."},
  "task": {
    "train": [{"input": "Can you resend my invoice?", "expected": "billing"}],
    "holdout": [{"input": "Why was I charged twice?", "expected": "billing"}, "… at least 8 cases"]
  },
  "budget": {"max_calls": 40, "max_cost_microusd": 1000000, "max_seconds": 240},
  "max_trials": 3,
  "objective": "normalized",
  "holdout_repeats": 1
}
```

A job needs at least `2 × train × steps + 1 + 2 × holdout × steps × holdout_repeats` model calls: one training pass for each version, one proposal, and the paired holdout check. Each extra trial adds one proposal and one training pass. Task limits are 1–32 training cases and 8–64 holdout cases.

A finished job's `result.evidence` records the decision:

```json
{
  "test": "one-sided exact sign test on paired holdout cases",
  "alpha": 0.05, "cases": 8, "repeats": 1,
  "improved_cases": 5, "regressed_cases": 0, "tied_cases": 3,
  "baseline_score": 0.375, "candidate_score": 1.0, "gain": 0.625,
  "p_value": 0.03125, "minimum_wins": 5, "verdict": "improved"
}
```

`verdict` is `improved` when the candidate scores higher and `p_value ≤ alpha`, `inconclusive` when it scores higher but the gain could be chance, and `not_improved` otherwise. Only `improved` replaces the artifact; `result.candidate` always holds the best proposal for review. `minimum_wins` is the fewest improved cases, with none regressed, that could reach significance with this many holdout cases.

Run the same job with `evolve-v1` and a new idempotency key to compare methods. `auto` uses compatible entries in `evidence.json` or records that it chose a compatible default. Provide benchmark results from separate evaluation data, never from a customer's private holdout. Each evidence entry names its `benchmark`, `method`, `model_fingerprint`, `artifact_type`, `objective`, `runs`, `mean_gain`, `cost_microusd`, and `calls`. Leave the array empty until measured results are available.

Set `data_policy` to `zero-retention` to filter for providers marked that way in the operator's model configuration. Set a model's `retention` to `zero-retention` only after verifying that the selected provider account and deployment enforce it. The gateway itself persists jobs and artifacts.

Built-in artifacts include prompts and two-to-four-step prompt chains. The self-revising method can change its text improver, which descendants use to propose revisions. Every built-in optimizer is a loop on the shared engine in `rsi_library/engine.py`, with the same artifact target, training harness, and judge, and its own search and proposer. The fixture harness sends test inputs to the configured model for inference. Verify that provider data handling matches the pilot workspace's requirements.

## Public loops

The homepage, `/llms.txt`, `/api/*` and `/mcp` need no key. They serve the offline loops and runs driven by the caller's own model, described in the [project README](../README.md). They never call the gateway's models and never spend its budget. Driven runs are stored in `lab.db` beside the gateway database, kept for 14 days, and capped at 5,000 runs. Each network may start 40 runs and make 1,500 requests an hour. The homepage lets a visitor use their own OpenRouter key; the browser sends it straight to OpenRouter, so the content security policy allows connections to `openrouter.ai` and the key never reaches this server. Hosted jobs moved to `/jobs`.

## Access requests and the operator

Visitors can request a key from the homepage. Requests are rate limited per network, screened with a hidden form field, and answered the same way for every address. The gateway stores the email, optional name, and description until you decide.

The operator workspace reviews them. On Railway it is the owner workspace created from `RSI_BOOTSTRAP_KEY_SHA256`. Connect to the workbench with that workspace's write key to see the **Access requests** panel: approve a request with an allowance to create the requester's workspace and one write key, which is shown once for you to send, or decline it. The panel also shows model spend across all workspaces. From a shell on the host, the same actions are available as commands:

```sh
python -m rsi_library.gateway access-requests
python -m rsi_library.gateway approve-request REQUEST_ID --spend-limit-microusd 1000000
python -m rsi_library.gateway decline-request REQUEST_ID
python -m rsi_library.gateway site-usage
```

## Railway

The Railway service runs the API and a worker together on one replica. Mount its persistent volume at `/data` and keep `RSI_DB=/data/gateway.db`. The deployment checks `/health/ready` before serving traffic. The deployment configuration is in [`railway.toml`](../railway.toml).

| Variable | Purpose |
| --- | --- |
| `RSI_BOOTSTRAP_KEY_SHA256` | SHA-256 of the owner's workspace key. Creates the owner workspace once; it is also the operator. |
| `RSI_HOSTED_DEMO` | `1` runs the simulator with no model charges. Remove it, or set another value, for live models. |
| `RSI_MODEL_CONFIG_JSON` | The model configuration as JSON, for example [`models.openrouter.json`](models.openrouter.json). Set it only with demo mode off. |
| `OPENROUTER_API_KEY` | The credential named by `api_key_env` in that configuration. Add it in Railway, never in this repository. |
| `RSI_SITE_SPEND_LIMIT_MICROUSD` | Total committed model spend allowed across every workspace, for example `5000000` for $5. |
| `RSI_HOSTED_SPEND_LIMIT_MICROUSD` | The owner workspace's allowance, applied on every start. |
| `RSI_DEFAULT_WORKSPACE_SPEND_MICROUSD` | The default allowance when approving a request. Defaults to $1. |
| `RSI_ROUTING_EVIDENCE_JSON` | Benchmark rows for the auto route, as written by the benchmark's `--evidence-out`. |
| `RSI_ACCESS_REQUESTS` | `0` closes the request form. Requests are open by default. |

To switch from demo mode to live models, add the provider key first, then set `RSI_MODEL_CONFIG_JSON` and remove `RSI_HOSTED_DEMO` in the same deployment. The service refuses to start with both demo mode and a model configuration, or with a configured model whose key is missing. The OpenRouter configuration prices each model 25% above its listed rate, so recorded spend runs ahead of the provider's bill. Verify pricing and usage against the provider's activity log on a small run before raising allowances.

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
