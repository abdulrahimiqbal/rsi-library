"""A small Model Context Protocol server for the public loops.

It speaks JSON-RPC over HTTP POST (the Streamable HTTP transport, replying with plain
JSON) and needs no session or sign-in, so a chat assistant can connect to it as a
remote connector and run loops on a person's behalf.
"""

from __future__ import annotations

import json

from . import lab as lab_module
from .lab import RETENTION_TEXT, Lab, LabError, with_base
from .methods import OBJECTIVES

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "rsi-library", "title": "RSI Library", "version": "0.3.0"}
STARTING_TOOLS = frozenset({"start_prompt_run", "start_code_run"})

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
_SCORING = {
    "objective": {
        "type": "string", "enum": list(OBJECTIVES),
        "description": (
            "How each reply is scored against `expected`: normalized (text match ignoring case and spacing, the "
            "default), exact, number (last number in the reply, optional tolerance), contains, json (fields of a JSON "
            "object), or judge (a model grades 0-10 against a rubric; one more work item per case)."
        ),
    },
    "scoring": {
        "type": "object",
        "description": "Settings for the objective: {tolerance, relative_tolerance} for number, {fields} for json, {rubric} for judge.",
    },
}
_RUN_OPTIONS = {
    "max_trials": {"type": "integer", "minimum": 1, "maximum": lab_module.MAX_DRIVEN_TRIALS, "description": "Revisions each loop may try. Default 3."},
    "holdout_repeats": {"type": "integer", "minimum": 1, "maximum": lab_module.MAX_DRIVEN_REPEATS},
    "seed": {"type": "integer", "minimum": 0},
    **_SCORING,
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
            "expected answers. The server keeps the holdout labels and decides with a paired significance test. "
            "The reply's `page` is a link where the user can watch the run."
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
                **_RUN_OPTIONS,
            },
            "required": ["task"],
        },
    },
    {
        "name": "start_code_run",
        "title": "Improve a function",
        "description": (
            "Start improving the user's own Python or JavaScript function with reflect-v1, evolve-v1, or \"all\". "
            "The server never runs code: work items of kind 'run' ask you to call the function on an input in your "
            "own sandbox (no network or files, a few seconds' limit) and reply with its result, as text or JSON, or "
            "'Error: <message>'. Items of kind 'propose' ask you to write a revised function. Needs the source, the "
            "function name, training cases, and at least 8 holdout cases with expected results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "loop": {"type": "string", "description": "reflect-v1 or evolve-v1. Use this or 'loops'."},
                "loops": _LOOPS,
                "language": {"type": "string", "enum": ["python", "javascript"]},
                "entry": {"type": "string", "description": "The function to call. It takes the case input as one text argument."},
                "source": {"type": "string", "description": "The complete source that defines the function."},
                "task": {**_TASK, "required": ["train", "holdout"]},
                **_RUN_OPTIONS,
            },
            "required": ["language", "entry", "source", "task"],
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
        "description": (
            "Return a run's current state: what each loop is doing, its pending work items, or its results and "
            "recommendation. Set work to false to leave out the work items."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}, "work": {"type": "boolean", "description": "Include pending work items. Default true."}},
            "required": ["run_id"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "delete_run",
        "title": "Delete a run",
        "description": "Delete a run and everything stored with it: the prompt or function, the cases, and the answers.",
        "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
        "annotations": {"destructiveHint": True, "idempotentHint": True},
    },
]

INSTRUCTIONS = (
    "RSI Library runs self-improvement loops for prompts and functions. Call list_loops to see them. Offline "
    "loops run on the server with no model. To improve the user's own prompt, call start_prompt_run; for a "
    "Python or JavaScript function, call start_code_run. Answer each work item as its `instructions` say (you "
    "are the model, and your sandbox runs any function), and call submit_work until the run is done. Give the "
    "user the run's `page` link so they can watch it. Then report the verdict, the holdout scores, the p-value "
    "and the recommended version."
)


def _call(name: str, arguments: dict, lab: Lab, base: str) -> dict:
    if name == "list_loops":
        return lab_module.catalog()
    if name == "run_offline_loop":
        return lab_module.offline_run(arguments)
    if name == "compare_offline_loops":
        return lab_module.offline_compare(arguments)
    if name == "start_prompt_run":
        return with_base(lab.start(arguments), base)
    if name == "start_code_run":
        program = {key: arguments.get(key) for key in ("language", "entry", "source")}
        payload = {key: value for key, value in arguments.items() if key not in program}
        if "artifact" in payload or "prompt" in payload:
            raise LabError("start_code_run takes language, entry and source instead of a prompt or artifact.")
        return with_base(lab.start({**payload, "artifact": {"type": "program", **program}}), base)
    run_id = arguments.get("run_id")
    if not isinstance(run_id, str):
        raise LabError("Provide the run_id returned when the run started.")
    if name == "delete_run":
        if arguments.keys() - {"run_id"}:
            raise LabError("delete_run takes only run_id.")
        if not lab.delete(run_id):
            raise LabError("Unknown run_id. " + RETENTION_TEXT)
        return {"id": run_id, "deleted": True}
    if name == "get_run":
        if arguments.keys() - {"run_id", "work"} or not isinstance(arguments.get("work", True), bool):
            raise LabError("get_run takes run_id and an optional work flag.")
        view = lab.get(run_id, work=arguments.get("work", True))
    elif name == "submit_work":
        if arguments.keys() - {"run_id", "outputs"}:
            raise LabError("submit_work takes run_id and outputs.")
        view = lab.submit(run_id, arguments.get("outputs"))
    else:
        raise KeyError(name)
    if view is None:
        raise LabError("Unknown run_id. " + RETENTION_TEXT)
    return with_base(view, base)


def _error(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _one(message: object, lab: Lab, allow: object, base: str) -> dict | None:
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
                text, failed = json.dumps(_call(name, arguments, lab, base), ensure_ascii=False, allow_nan=False), False
            except (LabError, ValueError, RecursionError) as exc:
                text, failed = str(exc) if not isinstance(exc, RecursionError) else "The request is nested too deeply.", True
        return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": text}], "isError": failed}}
    return _error(request_id, -32601, f"Method not found: {method[:80]}")


def handle(payload: object, lab: Lab, allow=lambda name: True, base: str = "") -> object | None:
    """Answer one JSON-RPC message or a batch. Returns None when no reply is due.

    ``base`` is the site's address, used to make run page links absolute.
    """
    if isinstance(payload, list):
        if not payload or len(payload) > 20:
            return _error(None, -32600, "Send 1 to 20 messages in a batch.")
        replies = [reply for reply in (_one(item, lab, allow, base) for item in payload) if reply is not None]
        return replies or None
    return _one(payload, lab, allow, base)
