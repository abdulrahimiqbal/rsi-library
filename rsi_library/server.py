"""Serve the local workbench and its JSON API."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from .builtins import catalog, make_registry
from .engine import ConfigError, Registry, compare, run

_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_MAX_BODY = 1024 * 1024


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, *, registry_factory: Callable[[], Registry] = make_registry):
        self.registry_factory = registry_factory
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    server: LocalServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def _write(self, status: int, body: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "base-uri 'none'; frame-ancestors 'none'; object-src 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: object) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self._write(status, body, "application/json; charset=utf-8")

    def _check_local(self) -> bool:
        # Avoid accepting API requests via another website's DNS alias.
        hosts = [f"{host}:{self.server.server_port}" for host in ("127.0.0.1", "localhost")]
        if self.server.server_port == 80:
            hosts.extend(("127.0.0.1", "localhost"))
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (origin and origin not in {f"http://{host}" for host in hosts}):
            self._json(403, {"error": "Open the workbench from its local address."})
            return False
        return True

    def do_GET(self) -> None:
        if not self._check_local():
            return
        if self.path == "/api/catalog":
            self._json(200, catalog(self.server.registry_factory()))
        elif self.path == "/api/health":
            self._json(200, {"status": "ready"})
        elif self.path in _ASSETS:
            filename, media_type = _ASSETS[self.path]
            self._write(200, files("rsi_library").joinpath("static").joinpath(filename).read_bytes(), media_type)
        else:
            self._json(404, {"error": "Page or API route not found."})

    def do_POST(self) -> None:
        if not self._check_local():
            return
        if self.path not in {"/api/run", "/api/compare"}:
            self._json(404, {"error": "API route not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Send the experiment as JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "Invalid request length."})
            return
        if not 0 < length <= _MAX_BODY:
            self._json(413, {"error": "Send a JSON body smaller than 1 MB."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, RecursionError):
            self._json(400, {"error": "Invalid JSON. Check the task or experiment format."})
            return
        except TimeoutError:
            self._json(408, {"error": "The request took too long to upload."})
            return
        try:
            registry = self.server.registry_factory()
            if self.path == "/api/run":
                result = run(payload, registry)
            else:
                if not isinstance(payload, dict) or set(payload) != {"configs"}:
                    raise ConfigError("Provide the experiments in a 'configs' list.")
                result = compare(payload["configs"], registry)
            self._json(200, result)
        except ConfigError as exc:
            self._json(400, {"error": str(exc)})


def serve(port: int, *, registry_factory: Callable[[], Registry] = make_registry, open_browser: bool = False) -> None:
    with LocalServer(port, registry_factory=registry_factory) as server:
        print(f"RSI Library is ready at {server.url}", flush=True)
        if open_browser:
            webbrowser.open_new_tab(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
