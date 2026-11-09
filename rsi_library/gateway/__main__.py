"""Operate the RSI gateway, workspace keys, and usage reconciliation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path

from .api import Gateway
from .providers import load_models
from .routing import load_adapters, load_evidence
from .store import HOSTED_OWNER_ID, MAX_COST_MICROUSD, Store, StoreError
from .worker import Worker


def _print(value: object) -> None:
    print(json.dumps(value, indent=2), flush=True)


def _microusd_env(name: str) -> int | None:
    """Read a whole-microdollar amount from the environment; one dollar is 1000000."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) > MAX_COST_MICROUSD:
        raise ValueError(f"{name} must be a whole number of microdollars, for example 5000000 for $5.")
    return int(raw)


def _serve(args, store, adapters, models, evidence, worker: Worker | None = None, *, drain: int = 3, **options) -> None:
    with Gateway(args.host, args.port, store, adapters, models, evidence, **options) as gateway:
        worker_thread = None
        if worker is not None:
            worker_thread = threading.Thread(target=worker.run, daemon=True)
            worker_thread.start()
        print(f"RSI Gateway listening on {gateway.server_address[0]}:{gateway.server_port}", flush=True)
        if getattr(args, "open", False):
            webbrowser.open_new_tab(f"http://127.0.0.1:{gateway.server_port}")

        def terminate(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        try:
            gateway.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            if worker_thread is not None:
                worker.stop.set()
                worker_thread.join(timeout=drain)


def _hosted_storage(path: Path) -> None:
    """Prepare the Railway volume before dropping root permissions."""
    if not os.environ.get("RAILWAY_ENVIRONMENT_ID"):
        return
    volume = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if not volume or path.parent.resolve() != Path(volume).resolve() or not Path(volume).is_dir():
        raise ValueError("Attach a persistent Railway volume at the gateway database directory.")
    if os.getuid() != 0:
        return
    for item in (path.parent, path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        if item.is_symlink():
            raise ValueError("The gateway volume cannot contain a symlink at its database path.")
        if item.exists():
            os.chown(item, 10001, 10001)
    os.chmod(path.parent, 0o700)
    os.setgroups([])
    os.setgid(10001)
    os.setuid(10001)


def _hosted(args, store: Store) -> None:
    args.host = "0.0.0.0"
    args.port = int(os.environ.get("PORT", "8766"))
    if not 1 <= args.port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    demo = os.environ.get("RSI_HOSTED_DEMO") == "1"
    config_json = os.environ.get("RSI_MODEL_CONFIG_JSON")
    if demo and (args.models or config_json):
        raise ValueError("Disable demo mode before configuring paid models.")
    models = load_models(args.models, config_json=config_json, demo=demo)
    if not models or not all(model.available() for model in models.values()):
        raise ValueError("Configure hosted model credentials or explicitly enable demo mode.")
    token_hash = os.environ.get("RSI_BOOTSTRAP_KEY_SHA256", "").strip()
    owner_limit = _microusd_env("RSI_HOSTED_SPEND_LIMIT_MICROUSD")
    if token_hash:
        store.bootstrap_workspace(token_hash)
        if owner_limit is not None:
            # The environment is the source of truth for the owner's allowance on every start.
            store.set_spend_limit(HOSTED_OWNER_ID, owner_limit)
    elif owner_limit is not None:
        raise ValueError("RSI_HOSTED_SPEND_LIMIT_MICROUSD needs RSI_BOOTSTRAP_KEY_SHA256 to identify the owner workspace.")
    adapters = load_adapters(os.environ.get("RSI_METHOD_ADAPTERS", ""))
    evidence = load_evidence(args.evidence, config_json=os.environ.get("RSI_ROUTING_EVIDENCE_JSON"))
    default_spend = _microusd_env("RSI_DEFAULT_WORKSPACE_SPEND_MICROUSD")
    _serve(
        args, store, adapters, models, evidence, Worker(store, adapters, models), drain=40,
        operator_workspace=HOSTED_OWNER_ID if token_hash else None,
        access_requests=os.environ.get("RSI_ACCESS_REQUESTS", "1") == "1",
        trust_proxy=bool(os.environ.get("RAILWAY_ENVIRONMENT_ID")) or os.environ.get("RSI_TRUST_PROXY") == "1",
        default_spend_microusd=1_000_000 if default_spend is None else default_spend,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsi-gateway", description="Operate durable RSI optimization jobs.")
    parser.add_argument("--db", default=os.environ.get("RSI_DB", "rsi-data/gateway.db"))
    parser.add_argument("--models", default=os.environ.get("RSI_MODEL_CONFIG"))
    parser.add_argument("--evidence", default=os.environ.get("RSI_ROUTING_EVIDENCE"))
    commands = parser.add_subparsers(dest="command", required=True)
    workspace = commands.add_parser("workspace", help="provision a workspace")
    workspace.add_argument("name")
    workspace.add_argument("--spend-limit-microusd", type=int, default=0)
    key = commands.add_parser("key", help="issue a key to a workspace")
    key.add_argument("workspace")
    key.add_argument("--role", choices=("read", "write"), default="write")
    revoke = commands.add_parser("revoke-key", help="revoke a workspace key by ID")
    revoke.add_argument("key_id")
    backup = commands.add_parser("backup", help="take a consistent database backup")
    backup.add_argument("destination")
    usage = commands.add_parser("usage", help="read recorded usage for a workspace")
    usage.add_argument("workspace")
    usage.add_argument("--ledger", action="store_true")
    limit = commands.add_parser("spend-limit", help="set the total authorized model spend for a workspace")
    limit.add_argument("workspace")
    limit.add_argument("cost_microusd", type=int)
    reconcile = commands.add_parser("reconcile", help="record a verified provider charge for an uncertain call")
    reconcile.add_argument("reservation")
    reconcile.add_argument("cost_microusd", type=int)
    requests = commands.add_parser("access-requests", help="list requests for workspace keys")
    requests.add_argument("--state", choices=("pending", "approved", "declined", "all"), default="pending")
    approve = commands.add_parser("approve-request", help="create a workspace and write key for an access request")
    approve.add_argument("request_id")
    approve.add_argument("--spend-limit-microusd", type=int, default=1_000_000)
    decline = commands.add_parser("decline-request", help="decline an access request")
    decline.add_argument("request_id")
    commands.add_parser("site-usage", help="read model usage across every workspace")
    commands.add_parser("worker", help="claim and execute queued jobs")
    commands.add_parser("hosted", help="serve the API and worker together on a single persistent volume")
    server = commands.add_parser("serve", help="serve the gateway API and browser workbench")
    server.add_argument("--host", default=os.environ.get("RSI_HOST", "127.0.0.1"))
    server.add_argument("--port", type=int, default=8766)
    demo = commands.add_parser("demo", help="run the local gateway with a simulated model and no spend")
    demo.add_argument("--port", type=int, default=8766)
    demo.add_argument("--open", action="store_true", help="open the local workbench in your browser")
    demo.set_defaults(host="127.0.0.1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    try:
        path = Path(args.db)
        if args.command == "hosted" or os.environ.get("RAILWAY_ENVIRONMENT_ID"):
            # Operator commands from a Railway shell run as root too; drop to the service user
            # so the database and its journal files stay writable by the gateway.
            _hosted_storage(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        store = Store(path, site_spend_limit_microusd=_microusd_env("RSI_SITE_SPEND_LIMIT_MICROUSD"))
        if args.command == "workspace":
            _print({"id": store.create_workspace(args.name, args.spend_limit_microusd)})
        elif args.command == "key":
            _print(store.create_key(args.workspace, args.role))
        elif args.command == "revoke-key":
            _print({"revoked": store.revoke_key(args.key_id)})
        elif args.command == "backup":
            store.backup(args.destination)
            _print({"backup": args.destination})
        elif args.command == "usage":
            result = {"usage": store.usage(args.workspace)}
            if args.ledger:
                result["ledger"] = store.ledger(args.workspace)
            _print(result)
        elif args.command == "spend-limit":
            _print({"spend_limit_microusd": store.set_spend_limit(args.workspace, args.cost_microusd)})
        elif args.command == "reconcile":
            _print(store.reconcile(args.reservation, args.cost_microusd))
        elif args.command == "access-requests":
            _print({"requests": store.access_requests(None if args.state == "all" else args.state)})
        elif args.command == "approve-request":
            _print(store.approve_access(args.request_id, args.spend_limit_microusd))
        elif args.command == "decline-request":
            _print({"request": store.decline_access(args.request_id)})
        elif args.command == "site-usage":
            _print({"usage": store.site_usage()})
        elif args.command == "demo":
            adapters = load_adapters()
            models = load_models(None, demo=True)
            demo_workspace = store.create_workspace("Local gateway demo")
            _print({"workspace_id": demo_workspace, **store.create_key(demo_workspace)})
            # The demo key can also try the operator view and the access request form.
            _serve(args, store, adapters, models, [], Worker(store, adapters, models),
                   operator_workspace=demo_workspace, access_requests=True)
        elif args.command == "hosted":
            _hosted(args, store)
        else:
            adapters = load_adapters(os.environ.get("RSI_METHOD_ADAPTERS", ""))
            models = load_models(args.models)
            if not models:
                raise ValueError("Configure at least one real model with --models or RSI_MODEL_CONFIG. For offline use, run 'demo'.")
            if args.command == "serve":
                default_spend = _microusd_env("RSI_DEFAULT_WORKSPACE_SPEND_MICROUSD")
                _serve(
                    args, store, adapters, models,
                    load_evidence(args.evidence, config_json=os.environ.get("RSI_ROUTING_EVIDENCE_JSON")),
                    operator_workspace=os.environ.get("RSI_OPERATOR_WORKSPACE") or None,
                    access_requests=os.environ.get("RSI_ACCESS_REQUESTS") == "1",
                    trust_proxy=os.environ.get("RSI_TRUST_PROXY") == "1",
                    default_spend_microusd=1_000_000 if default_spend is None else default_spend,
                )
            else:
                missing = [model.api_key_env for model in models.values() if not model.available()]
                if missing:
                    raise ValueError("Missing worker credentials: " + ", ".join(sorted(set(missing))))
                worker = Worker(store, adapters, models)
                signal.signal(signal.SIGINT, lambda *_: worker.stop.set())
                signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
                worker.run()
        return 0
    except (OSError, ValueError, ImportError, AttributeError, sqlite3.Error, StoreError) as exc:
        print(f"RSI Gateway: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
