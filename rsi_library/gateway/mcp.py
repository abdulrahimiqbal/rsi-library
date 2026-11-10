"""A small Model Context Protocol server for the public loops.

It speaks JSON-RPC over HTTP POST (the Streamable HTTP transport, replying with plain
JSON) and needs no session or sign-in, so a chat assistant can connect to it as a
remote connector and run loops on a person's behalf.
"""

from __future__ import annotations

import json

from . import lab as lab_module
from .lab import Lab, LabError

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "rsi-library", "title": "RSI Library", "version": "0.2.0"}

_CASES = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"input": {"type": "string"}, "expected": {"type": "string"}},
        "required": ["input", "expected"],
    },
}
_TASK = {
    "type": "object",
    "description": "Labelled cases. Loops learn from 'train'; 'holdout' is only used for the final check.",
    "properties": {"train": _CASES, "holdout": _CASES},
    "required": ["train"],
}
_LOOPS = {
    "description": "Loop IDs to run on the same task, or \"all\".",
    "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string", "enum": ["all"]}],
}

TOOLS = [
    {
        "name": "list_loops",
        "title": "List loops",
        "description": (
            "List every self-improvement loop RSI Library can run, what each one changes, its five parts "
            "(target, proposer, judge, search, harness), limits, and an example task. Call this first."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "run_offline_loop",
        "title": "Run a loop without a model",
        "description": (
            "Run one of the offline loops (greedy-prompt, archive-prompt, adaptive-improver) on the server. "
            "No model is called: the loop improves a small routing prompt for a built-in interpreter. "
            "Returns every version tried with its score and parent, and the winner's holdout score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "loop": {"type": "string", "description": "An offline loop ID from list_loops."},
                "budget": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Versions to score, including the starting one."},
                "seed": {"type": "integer", "minimum": 0},
                "task": _TASK,
                "search": {"type": "string", "description": "Replace the search part, for example 'greedy' or 'archive'."},
                "proposer": {"type": "string", "description": "Replace the proposer part."},
                "judge": {"type": "string", "description": "Replace the judge part."},
                "harness": {"type": "string", "description": "Replace the harness part."},
                "target": {"type": "string", "description": "Replace the target part."},
            },
            "required": ["loop"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "compare_offline_loops",
        "title": "Compare loops without a model",
        "description": "Run 2 to 5 offline loops on the same task, seed and budget, and report which scored best.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "loops": _LOOPS,
                "budget": {"type": "integer", "minimum": 1, "maximum": 100},
                "seed": {"type": "integer", "minimum": 0},
                "task": _TASK,
            },
            "required": ["loops"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "start_prompt_run",
        "title": "Improve a prompt",
        "description": (
            "Start improving the user's own prompt with one loop, several, or \"all\" (reflect-v1, evolve-v1, "
            "self-revise-v1). You act as the model: the reply contains work items to answer, then pass the "
            "answers to submit_work. Needs the prompt, training cases, and at least 8 holdout cases with "
            "expected answers. The server keeps the holdout labels and decides with a paired significance test."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "loop": {"type": "string", "description": "One loop ID. Use this or 'loops'."},
                "loops": _LOOPS,
                "prompt": {"type": "string", "description": "The prompt to improve. Use this or 'artifact'."},
                "artifact": {
                    "type": "object",
                    "description": "A prompt chain: {\"type\": \"prompt-chain\", \"steps\": [{\"name\", \"text\"}, …]} with 2 to 4 steps.",
                },
                "task": {**_TASK, "required": ["train", "holdout"]},
                "max_trials": {"type": "integer", "minimum": 1, "maximum": lab_module.MAX_DRIVEN_TRIALS, "description": "Revisions each loop may try. Default 3."},
                "objective": {"type": "string", "enum": ["normalized", "exact"], "description": "'normalized' ignores case and spacing. Default."},
                "holdout_repeats": {"type": "integer", "minimum": 1, "maximum": lab_module.MAX_DRIVEN_REPEATS},
                "seed": {"type": "integer", "minimum": 0},
            },
            "required": ["task"],
        },
    },
    {
        "name": "submit_work",
        "title": "Submit answers",
        "description": (
            "Submit your answers to a run's work items. Returns the next work items, or the finished run with "
            "each loop's verdict, holdout scores, p-value, candidate prompt, and a recommendation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, "output": {"type": "string"}},
                        "required": ["id", "output"],
                    },
                },
            },
            "required": ["run_id", "outputs"],
        },
    },
    {
        "name": "get_run",
        "title": "Check a run",
        "description": "Return a run's current state: its pending work items, or its results and recommendation.",
        "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
        "annotations": {"readOnlyHint": True},
    },
]

INSTRUCTIONS = (
    "RSI Library runs self-improvement loops for prompts. Call list_loops to see them. Offline loops run on "
    "the server with no model. To improve the user's own prompt, call start_prompt_run, answer each work "
    "item as the model would (system prompt = `system`, user message = `input`), and call submit_work until "
    "the run is done. Then report the verdict, the holdout scores, the p-value and the recommended prompt."
)


def _call(name: str, arguments: dict, lab: Lab) -> dict:
    if name == "list_loops":
        return lab_module.catalog()
    if name == "run_offline_loop":
        return lab_module.offline_run(arguments)
    if name == "compare_offline_loops":
        return lab_module.offline_compare(arguments)
    if name == "start_prompt_run":
        return lab.start(arguments)
    if name in ("submit_work", "get_run"):
        run_id = arguments.get("run_id")
        if not isinstance(run_id, str):
            raise LabError("Provide the run_id returned by start_prompt_run.")
        if name == "get_run":
            if arguments.keys() - {"run_id"}:
                raise LabError("get_run takes only run_id.")
            view = lab.get(run_id)
        else:
            if arguments.keys() - {"run_id", "outputs"}:
                raise LabError("submit_work takes run_id and outputs.")
            view = lab.submit(run_id, arguments.get("outputs"))
        if view is None:
            raise LabError("Unknown run_id. Runs are kept for 14 days.")
        return view
    raise KeyError(name)


def _error(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _one(message: object, lab: Lab, allow: object) -> dict | None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Send a JSON-RPC 2.0 message.")
    method, request_id = message.get("method"), message.get("id")
    if not isinstance(method, str):
        return None  # A response from the client; nothing to do.
    if "id" not in message:
        return None  # Notifications get no reply.
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return _error(request_id, -32602, "params must be an object.")
    if method == "initialize":
        wanted = params.get("protocolVersion")
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": wanted if wanted in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name, arguments = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict) or not any(tool["name"] == name for tool in TOOLS):
            return _error(request_id, -32602, "Unknown tool. Call tools/list for the tool names.")
        if not allow(name):
            text, failed = "Too many requests from this network. Wait a while and try again.", True
        else:
            try:
                text, failed = json.dumps(_call(name, arguments, lab), ensure_ascii=False, allow_nan=False), False
            except (LabError, ValueError, RecursionError) as exc:
                text, failed = str(exc) if not isinstance(exc, RecursionError) else "The request is nested too deeply.", True
        return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": text}], "isError": failed}}
    return _error(request_id, -32601, f"Method not found: {method[:80]}")


def handle(payload: object, lab: Lab, allow=lambda name: True) -> object | None:
    """Answer one JSON-RPC message or a batch. Returns None when no reply is due."""
    if isinstance(payload, list):
        if not payload or len(payload) > 20:
            return _error(None, -32600, "Send 1 to 20 messages in a batch.")
        replies = [reply for reply in (_one(item, lab, allow) for item in payload) if reply is not None]
        return replies or None
    return _one(payload, lab, allow)
