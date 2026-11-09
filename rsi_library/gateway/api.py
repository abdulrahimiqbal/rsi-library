"""Authenticated HTTP interface to the RSI job queue."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

from .methods import catalog as method_catalog, evidence_policy, validate_request
from .providers import Model
from .routing import Adapter, choose
from .store import MAX_COST_MICROUSD, Conflict, Store, StoreError


_ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/styles.css": ("styles.css", "text/css"),
}
_JOB_PATH = re.compile(r"/v1/jobs/(job_[a-f0-9]{32})(/cancel)?")
_REQUEST_PATH = re.compile(r"/v1/operator/access-requests/(req_[a-f0-9]{32})/(approve|decline)")
_IDEMPOTENCY = re.compile(r"[A-Za-z0-9._-]{1,128}")
_MAX_BODY = 160 * 1024
_MAX_ACCESS_BODY = 8 * 1024
_LIMIT = 32


class _RateLimit:
    """Allow a few events per client and per window, plus a total, in this process."""

    def __init__(self, per_client: int, total: int, window_seconds: float):
        self.per_client, self.total, self.window = per_client, total, window_seconds
        self._clients: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._all and now - self._all[0] > self.window:
                self._all.popleft()
            for key in [key for key, events in self._clients.items() if not events or now - events[-1] > self.window]:
                del self._clients[key]
            events = self._clients.setdefault(client, deque())
            while events and now - events[0] > self.window:
                events.popleft()
            if len(events) >= self.per_client or len(self._all) >= self.total:
                return False
            events.append(now)
            self._all.append(now)
            return True


def _summary(job: dict) -> dict:
    request, result = job["request"], job["result"]
    request_summary = {field: request[field] for field in ("method", "model", "budget", "data_policy", "_routing") if field in request}
    request_summary["artifact"] = {"type": request["artifact"]["type"]}
    result_summary = None if result is None else {
        field: result[field] for field in ("method", "baseline_holdout", "candidate_holdout", "accepted", "routing") if field in result
    }
    if result_summary is not None and isinstance(result.get("evidence"), dict):
        result_summary["verdict"] = result["evidence"].get("verdict")
    return {**job, "request": request_summary, "result": result_summary}


class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(
        self, host: str, port: int, store: Store, adapters: dict[str, Adapter],
        models: dict[str, Model], evidence: list[dict], *,
        operator_workspace: str | None = None, access_requests: bool = False,
        trust_proxy: bool = False, default_spend_microusd: int = 1_000_000,
    ):
        self.store = store
        self.adapters = adapters
        self.models = models
        self.evidence = evidence
        # The workspace whose write keys may review access requests and see site-wide usage.
        self.operator_workspace = operator_workspace
        self.access_requests = access_requests
        self.trust_proxy = trust_proxy
        self.default_spend_microusd = default_spend_microusd
        self.request_limit = _RateLimit(per_client=3, total=60, window_seconds=3600)
        self.slots = threading.BoundedSemaphore(_LIMIT)
        super().__init__((host, port), Handler)

    @property
    def mode(self) -> str:
        return "demo" if self.models and all(model.demo for model in self.models.values()) else "live"

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

    def _client(self) -> str:
        if self.server.trust_proxy:
            # Behind the hosting edge, the client's address arrives in a forwarding header.
            forwarded = self.headers.get("X-Real-IP") or (self.headers.get("X-Forwarded-For") or "").split(",")[0]
            if forwarded.strip():
                return forwarded.strip()[:64]
        return self.client_address[0]

    def _operator(self, identity: dict) -> bool:
        return self.server.operator_workspace is not None and identity["workspace_id"] == self.server.operator_workspace

    def _body(self, limit: int = _MAX_BODY) -> dict:
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit() or self.headers.get("Transfer-Encoding"):
            raise ValueError("Provide a valid JSON content length.")
        length = int(lengths[0])
        if not 0 < length <= limit:
            raise ValueError(f"The JSON request must be smaller than {limit // 1024} KiB.")
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
        if not write and path == "/public/info":
            self._json(200, {
                "mode": self.server.mode,
                "access_requests": self.server.access_requests,
                "methods": [{key: entry[key] for key in ("id", "name", "description")} for entry in method_catalog()],
                "evidence_policy": evidence_policy(),
            })
            return
        if write and path == "/access-requests":
            self._access_request()
            return
        if not path.startswith("/v1/"):
            self._error(404, "Route not found.")
            return
        identity = self._identity(write)
        if identity is None:
            return
        workspace = identity["workspace_id"]
        if path.startswith("/v1/operator/"):
            self._operator_route(identity, path, url.query, write=write)
            return
        if not write and path == "/v1/catalog":
            self._json(200, {
                **identity,
                "operator": self._operator(identity),
                "methods": [adapter.metadata for adapter in self.server.adapters.values()],
                "models": [model.public() for model in self.server.models.values()],
                "evidence_policy": evidence_policy(),
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

    def _access_request(self) -> None:
        if not self.server.access_requests:
            self._error(404, "Access requests are not open on this gateway.")
            return
        if not self.server.request_limit.allow(self._client()):
            self._error(429, "Too many requests. Please try again later.")
            return
        payload = self._body(_MAX_ACCESS_BODY)
        if payload.keys() - {"email", "name", "use_case", "website"}:
            raise ValueError("Send email, name, and use_case.")
        # Browsers fill hidden fields that people don't see; treat a filled one as automated.
        if not payload.get("website"):
            self.server.store.request_access(payload.get("email"), payload.get("name"), payload.get("use_case"))
        # The same reply for new, repeated, and screened requests reveals nothing about other addresses.
        self._json(202, {"status": "received"})

    def _operator_route(self, identity: dict, path: str, query: str, *, write: bool) -> None:
        if not self._operator(identity) or identity["role"] != "write":
            self._error(403, "Operator access requires a write key for the operator workspace.")
            return
        store = self.server.store
        if not write and path == "/v1/operator/access-requests":
            state = parse_qs(query, max_num_fields=2).get("state", ["pending"])[0]
            self._json(200, {"requests": store.access_requests(None if state == "all" else state)})
            return
        if not write and path == "/v1/operator/usage":
            self._json(200, {"usage": store.site_usage()})
            return
        match = _REQUEST_PATH.fullmatch(path)
        if write and match:
            if match[2] == "decline":
                self._json(200, {"request": store.decline_access(match[1])})
                return
            payload = self._body(_MAX_ACCESS_BODY)
            limit = payload.get("spend_limit_microusd", self.server.default_spend_microusd)
            if payload.keys() - {"spend_limit_microusd"} or type(limit) is not int or not 0 <= limit <= MAX_COST_MICROUSD:
                raise ValueError("Set spend_limit_microusd to a whole number of microdollars.")
            self._json(200, store.approve_access(match[1], limit))
            return
        self._error(404, "Operator route not found.")
