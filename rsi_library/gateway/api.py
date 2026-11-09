"""Authenticated HTTP interface to the RSI job queue."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

from .methods import validate_request
from .providers import Model
from .routing import Adapter, choose
from .store import Conflict, Store, StoreError


_ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/styles.css": ("styles.css", "text/css"),
}
_JOB_PATH = re.compile(r"/v1/jobs/(job_[a-f0-9]{32})(/cancel)?")
_IDEMPOTENCY = re.compile(r"[A-Za-z0-9._-]{1,128}")
_MAX_BODY = 160 * 1024
_LIMIT = 32


def _summary(job: dict) -> dict:
    request, result = job["request"], job["result"]
    request_summary = {field: request[field] for field in ("method", "model", "budget", "data_policy", "_routing") if field in request}
    request_summary["artifact"] = {"type": request["artifact"]["type"]}
    result_summary = None if result is None else {
        field: result[field] for field in ("method", "baseline_holdout", "candidate_holdout", "accepted", "routing") if field in result
    }
    return {**job, "request": request_summary, "result": result_summary}


class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(
        self, host: str, port: int, store: Store, adapters: dict[str, Adapter],
        models: dict[str, Model], evidence: list[dict],
    ):
        self.store = store
        self.adapters = adapters
        self.models = models
        self.evidence = evidence
        self.slots = threading.BoundedSemaphore(_LIMIT)
        super().__init__((host, port), Handler)

    def process_request(self, request, client_address) -> None:
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server: Gateway
    server_version = "RSI Gateway"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args) -> None:
        # Avoid logging request URLs, credentials, or submitted artifacts.
        pass

    def _write(self, status: int, data: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", (
            "default-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "frame-ancestors 'none'; object-src 'none'; form-action 'self'"
        ))
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        self._write(status, json.dumps(payload, allow_nan=False).encode(), "application/json")

    def _error(self, status: int, detail: str) -> None:
        self._json(status, {"error": detail})

    def _identity(self, write: bool) -> dict | None:
        auth = self.headers.get_all("Authorization", [])
        if len(auth) != 1 or not auth[0].startswith("Bearer ") or not 20 <= len(auth[0]) <= 256:
            self._error(401, "Use a workspace API key in the Authorization header.")
            return None
        identity = self.server.store.authenticate(auth[0][7:])
        if identity is None:
            self._error(401, "The workspace API key is invalid or revoked.")
            return None
        if write and identity["role"] != "write":
            self._error(403, "A write key is required.")
            return None
        return identity

    def _body(self) -> dict:
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit() or self.headers.get("Transfer-Encoding"):
            raise ValueError("Provide a valid JSON content length.")
        length = int(lengths[0])
        if not 0 < length <= _MAX_BODY:
            raise ValueError("The JSON request must be smaller than 160 KiB.")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Send the job as application/json.")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("Send the job as a JSON object.")
        return payload

    def do_GET(self) -> None:
        self._handle(write=False)

    def do_POST(self) -> None:
        self._handle(write=True)

    def _handle(self, *, write: bool) -> None:
        try:
            self._dispatch(write=write)
        except Conflict as exc:
            self._error(409, str(exc))
        except (ValueError, RecursionError) as exc:
            self._error(400, str(exc) if not isinstance(exc, RecursionError) else "The request is nested too deeply.")
        except TimeoutError:
            self._error(408, "The request timed out.")
        except (StoreError, sqlite3.Error):
            self._error(503, "The job store is temporarily unavailable.")
        except (BrokenPipeError, ConnectionResetError):
            return

    def _dispatch(self, *, write: bool) -> None:
        url = urlsplit(self.path)
        path = url.path
        store = self.server.store
        if not write and path in _ASSETS:
            filename, media_type = _ASSETS[path]
            data = files("rsi_library.gateway").joinpath("static").joinpath(filename).read_bytes()
            self._write(200, data, media_type)
            return
        if not write and path == "/health/live":
            self._json(200, {"status": "alive"})
            return
        if not write and path == "/health/ready":
            ready = bool(self.server.models) and store.worker_ready()
            self._json(200 if ready else 503, {"status": "ready" if ready else "unavailable"})
            return
        if not path.startswith("/v1/"):
            self._error(404, "Route not found.")
            return
        identity = self._identity(write)
        if identity is None:
            return
        workspace = identity["workspace_id"]
        if not write and path == "/v1/catalog":
            self._json(200, {
                **identity,
                "methods": [adapter.metadata for adapter in self.server.adapters.values()],
                "models": [model.public() for model in self.server.models.values()],
            })
            return
        if not write and path == "/v1/usage":
            self._json(200, {"usage": store.usage(workspace)})
            return
        if not write and path == "/v1/jobs":
            query = parse_qs(url.query, max_num_fields=5)
            limit = int(query.get("limit", ["25"])[0])
            if not 1 <= limit <= 50:
                raise ValueError("Use a job list limit from 1 to 50.")
            self._json(200, {"jobs": [_summary(job) for job in store.list(workspace, limit=limit)]})
            return
        if write and path == "/v1/jobs":
            keys = self.headers.get_all("Idempotency-Key", [])
            if len(keys) != 1 or not _IDEMPOTENCY.fullmatch(keys[0]):
                raise ValueError("Provide an Idempotency-Key of 1 to 128 letters, digits, dots, dashes, or underscores.")
            request = validate_request(self._body(), extra_method_ids=self.server.adapters.keys())
            previous = store.find_submission(workspace, keys[0], request)
            if previous is not None:
                self._json(200, previous)
                return
            request["_routing"] = choose(request, self.server.adapters, self.server.models, self.server.evidence)
            self._json(202, store.submit(workspace, request, keys[0]))
            return
        match = _JOB_PATH.fullmatch(path)
        if match and bool(match[2]) == write:
            job = store.cancel(workspace, match[1]) if write else store.get(workspace, match[1])
            if job is not None:
                self._json(200, job)
                return
        self._error(404, "Job or route not found.")
