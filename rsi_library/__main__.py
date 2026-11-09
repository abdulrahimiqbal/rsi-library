"""Run the workbench or drive experiments from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .builtins import catalog, make_registry
from .engine import ConfigError, compare, run
from .server import serve


def _config(source: str) -> dict:
    for template in catalog()["templates"]:
        if template["id"] == source:
            return template["config"]
    try:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"No template or experiment file named {source!r}. Try 'catalog' for template IDs.") from exc
    if not isinstance(data, dict):
        raise ConfigError("An experiment must be a JSON object.")
    return data


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 0 to 65535")
    return port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m rsi_library", description="Fork, run, and compare improvement loops.")
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("serve", help="start the local browser workbench")
    local.add_argument("--port", type=_port, default=8765)
    local.add_argument("--open", action="store_true", help="open the workbench in your browser")
    listing = commands.add_parser("catalog", help="list templates and components as JSON")
    fork = commands.add_parser("fork", help="copy a template into an editable experiment")
    fork.add_argument("template", choices=[item["id"] for item in catalog()["templates"]])
    single = commands.add_parser("run", help="run a template ID or an experiment JSON file")
    single.add_argument("source", metavar="TEMPLATE_OR_FILE")
    pair = commands.add_parser("compare", help="run two experiments under the same evaluation budget")
    pair.add_argument("sources", nargs=2, metavar="TEMPLATE_OR_FILE")
    for command in (listing, fork, single, pair):
        command.add_argument("-o", "--output", type=Path, help="write the JSON result to this file")
    args = parser.parse_args(argv)

    try:
        if args.command == "serve":
            serve(args.port, open_browser=args.open)
            return 0
        if args.command == "catalog":
            result = catalog()
        elif args.command == "fork":
            result = {**_config(args.template), "task": catalog()["example_task"]}
        elif args.command == "run":
            result = run(_config(args.source), make_registry())
        else:
            result = compare([_config(source) for source in args.sources], make_registry())
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except (OSError, ValueError) as exc:
        print(f"RSI Library: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
