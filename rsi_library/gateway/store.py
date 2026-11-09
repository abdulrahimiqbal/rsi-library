"""Durable jobs, workspace credentials, and a conservative dispatch ledger."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path

MAX_CALLS = 10_000
MAX_COST_MICROUSD = 10**12
MAX_SECONDS = 86_400
MAX_ACTIVE_JOBS = 10
MAX_PENDING_REQUESTS = 500
# The hosted deployment's owner workspace, provisioned from RSI_BOOTSTRAP_KEY_SHA256.
HOSTED_OWNER_ID = "ws_" + hashlib.sha256(b"rsi-gateway-hosted-owner-v1").hexdigest()[:32]
_EMAIL = re.compile(r"[^@\s]{1,64}@[^@\s]{1,189}\.[A-Za-z]{2,63}")


class StoreError(RuntimeError):
    """A store operation could not be completed."""


class Conflict(StoreError):
    """A request conflicts with already recorded state."""


class InvalidInput(StoreError, ValueError):
    """Caller-supplied data was rejected; the API reports it as a bad request."""


class BudgetExceeded(StoreError):
    """A job hit its call, spend, or time limit."""


class Cancelled(StoreError):
    """Cancellation was requested for a running job."""


class LeaseLost(StoreError):
    """The worker no longer owns the job."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS workspace_limits (
    workspace_id TEXT PRIMARY KEY REFERENCES workspaces (id),
    spend_limit_microusd INTEGER NOT NULL CHECK (spend_limit_microusd >= 0)
);
CREATE TABLE IF NOT EXISTS workspace_keys (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces (id),
    role TEXT NOT NULL CHECK (role IN ('read', 'write')),
    token_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    revoked_at REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces (id),
    idempotency_key TEXT,
    request TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'running', 'cancel_requested', 'cancelled', 'succeeded', 'failed'
    )),
    result TEXT,
    error TEXT,
    progress TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    deadline_at REAL,
    lease_token TEXT,
    lease_until REAL,
    max_calls INTEGER NOT NULL,
    max_cost_microusd INTEGER NOT NULL,
    max_seconds INTEGER NOT NULL,
    UNIQUE (workspace_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (state, created_at);
CREATE INDEX IF NOT EXISTS jobs_workspace ON jobs (workspace_id, created_at);
CREATE INDEX IF NOT EXISTS jobs_active_workspace ON jobs (workspace_id)
    WHERE state IN ('queued', 'running', 'cancel_requested');
CREATE INDEX IF NOT EXISTS jobs_lease ON jobs (state, lease_until);
CREATE INDEX IF NOT EXISTS jobs_deadline ON jobs (state, deadline_at);
CREATE TABLE IF NOT EXISTS dispatches (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs (id),
    stage TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('reserved', 'settled', 'uncertain')),
    max_cost_microusd INTEGER NOT NULL,
    cost_microusd INTEGER,
    created_at REAL NOT NULL,
    settled_at REAL
);
CREATE INDEX IF NOT EXISTS dispatches_job ON dispatches (job_id, created_at);
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS access_requests (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    name TEXT NOT NULL,
    use_case TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'approved', 'declined')),
    workspace_id TEXT REFERENCES workspaces (id),
    created_at REAL NOT NULL,
    decided_at REAL
);
CREATE INDEX IF NOT EXISTS access_requests_state ON access_requests (state, created_at);
"""

_USAGE = """
COUNT(d.id) AS calls,
COALESCE(SUM(d.cost_microusd), 0) AS spent_microusd,
COALESCE(SUM(CASE WHEN d.state = 'reserved' THEN d.max_cost_microusd ELSE 0 END), 0)
    AS reserved_microusd,
COALESCE(SUM(CASE WHEN d.state = 'uncertain' THEN d.max_cost_microusd ELSE 0 END), 0)
    AS uncertain_microusd
"""

_TERMINAL = {"succeeded", "failed", "cancelled"}


def _id(prefix: str) -> str:
    return prefix + "_" + secrets.token_hex(16)


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise StoreError("Expected a JSON document.") from exc


def _submission_payload(request: dict) -> str:
    # Routing is selected by the gateway, rather than supplied by the caller.
    if not isinstance(request, dict):
        raise StoreError("The job request must be a JSON object.")
    return _json({key: value for key, value in request.items() if key != "_routing"})


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise StoreError(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _label(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise StoreError(f"{name} must contain 1 to {maximum} characters.")
    return value


def _lease_seconds(value: int) -> int:
    return _integer(value, "Lease duration", 1, MAX_SECONDS)


def _committed(usage: dict) -> int:
    return sum(usage[key] for key in ("spent_microusd", "reserved_microusd", "uncertain_microusd"))


class Store:
    """Keep each job transition and associated usage change atomic in SQLite."""

    def __init__(
        self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.time,
        site_spend_limit_microusd: int | None = None,
    ):
        self.path = Path(path).absolute()
        self._now = clock
        if site_spend_limit_microusd is not None:
            _integer(site_spend_limit_microusd, "Site spend limit", 0, MAX_COST_MICROUSD)
        # A ceiling on committed model spend across every workspace, on top of each workspace's own limit.
        self.site_spend_limit_microusd = site_spend_limit_microusd
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with closing(self._connect()) as connection:
            # WAL permits API reads while a worker commits progress and usage.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA + "\nCOMMIT;")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def create_workspace(self, name: str, spend_limit_microusd: int = 0) -> str:
        workspace_id = _id("ws")
        name = _label(name, "Workspace name", 128).strip()
        _integer(spend_limit_microusd, "Workspace spend limit", 0, MAX_COST_MICROUSD)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO workspaces (id, name, created_at) VALUES (?, ?, ?)",
                (workspace_id, name, self._now()),
            )
            connection.execute(
                "INSERT INTO workspace_limits (workspace_id, spend_limit_microusd) VALUES (?, ?)",
                (workspace_id, spend_limit_microusd),
            )
        return workspace_id

    def set_spend_limit(self, workspace_id: str, limit: int) -> int:
        """Set the total workspace dispatch allowance, including already incurred usage."""
        _integer(limit, "Workspace spend limit", 0, MAX_COST_MICROUSD)
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)).fetchone() is None:
                raise StoreError("Workspace does not exist.")
            connection.execute(
                """INSERT INTO workspace_limits (workspace_id, spend_limit_microusd) VALUES (?, ?)
                   ON CONFLICT (workspace_id) DO UPDATE SET spend_limit_microusd = excluded.spend_limit_microusd""",
                (workspace_id, limit),
            )
        return limit

    def create_key(self, workspace_id: str, role: str = "write") -> dict:
        if role not in ("read", "write"):
            raise StoreError("Key role must be read or write.")
        key_id = _id("key")
        token = "rsi_" + secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        with self._transaction() as connection:
            workspace = connection.execute("SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
            if workspace is None:
                raise StoreError("Workspace does not exist.")
            connection.execute(
                "INSERT INTO workspace_keys (id, workspace_id, role, token_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (key_id, workspace_id, role, digest, self._now()),
            )
        return {"id": key_id, "token": token}

    def bootstrap_workspace(self, token_hash: str) -> None:
        """Provision the hosted workspace once. Never replace or reissue its key."""
        if len(token_hash) != 64 or any(char not in "0123456789abcdef" for char in token_hash):
            raise StoreError("The hosted workspace key hash must be 64 lowercase hexadecimal characters.")
        workspace_id = HOSTED_OWNER_ID
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)).fetchone():
                return
            now = self._now()
            connection.execute("INSERT INTO workspaces (id, name, created_at) VALUES (?, ?, ?)", (workspace_id, "RSI Library", now))
            connection.execute("INSERT INTO workspace_limits (workspace_id, spend_limit_microusd) VALUES (?, 0)", (workspace_id,))
            connection.execute(
                "INSERT INTO workspace_keys (id, workspace_id, role, token_hash, created_at) VALUES (?, ?, 'write', ?, ?)",
                (_id("key"), workspace_id, token_hash, now),
            )

    def revoke_key(self, key_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE workspace_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (self._now(), key_id),
            )
            return cursor.rowcount == 1

    def authenticate(self, token: str) -> dict | None:
        if not isinstance(token, str) or not token.startswith("rsi_") or len(token) != 47 or not token.isascii():
            return None
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT workspace_id, role FROM workspace_keys WHERE token_hash = ? AND revoked_at IS NULL",
                (digest,),
            ).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _job_usage(connection: sqlite3.Connection, job_id: str) -> dict:
        row = connection.execute(
            "SELECT " + _USAGE + " FROM dispatches AS d WHERE job_id = ?", (job_id,),
        ).fetchone()
        return dict(row)

    @staticmethod
    def _workspace_usage(connection: sqlite3.Connection, workspace_id: str) -> dict:
        row = connection.execute(
            "SELECT " + _USAGE + " FROM dispatches AS d JOIN jobs AS j ON d.job_id = j.id WHERE j.workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        limit = connection.execute(
            "SELECT spend_limit_microusd FROM workspace_limits WHERE workspace_id = ?", (workspace_id,),
        ).fetchone()
        return {**dict(row), "spend_limit_microusd": limit[0] if limit is not None else 0}

    @staticmethod
    def _site_usage(connection: sqlite3.Connection) -> dict:
        return dict(connection.execute("SELECT " + _USAGE + " FROM dispatches AS d").fetchone())

    def site_usage(self) -> dict:
        """Model usage across every workspace, with the site-wide ceiling if one is set."""
        with self._transaction(write=False) as connection:
            usage = self._site_usage(connection)
        return {**usage, "site_spend_limit_microusd": self.site_spend_limit_microusd}

    @classmethod
    def _view(cls, connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "state": row["state"],
            "request": json.loads(row["request"]),
            "result": json.loads(row["result"]) if row["result"] is not None else None,
            "error": row["error"],
            "progress": json.loads(row["progress"]) if row["progress"] is not None else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "deadline_at": row["deadline_at"],
            "usage": cls._job_usage(connection, row["id"]),
        }

    @staticmethod
    def _find(connection: sqlite3.Connection, workspace_id: str, job_id: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM jobs WHERE workspace_id = ? AND id = ?", (workspace_id, job_id),
        ).fetchone()

    @classmethod
    def _find_submission(
        cls, connection: sqlite3.Connection, workspace_id: str, idempotency_key: str, request: dict,
    ) -> dict | None:
        existing = connection.execute(
            "SELECT * FROM jobs WHERE workspace_id = ? AND idempotency_key = ?",
            (workspace_id, idempotency_key),
        ).fetchone()
        if existing is None:
            return None
        if _submission_payload(json.loads(existing["request"])) != _submission_payload(request):
            raise Conflict("This idempotency key was already used for a different job.")
        return cls._view(connection, existing)

    def find_submission(self, workspace_id: str, idempotency_key: str, request: dict) -> dict | None:
        _label(idempotency_key, "Idempotency key", 128)
        with self._transaction(write=False) as connection:
            return self._find_submission(connection, workspace_id, idempotency_key, request)

    def submit(self, workspace_id: str, request: dict, idempotency_key: str | None = None) -> dict:
        if not isinstance(request, dict) or not isinstance(request.get("budget"), dict):
            raise StoreError("The job must declare a budget.")
        budget = request["budget"]
        max_calls = _integer(budget.get("max_calls"), "Call limit", 1, MAX_CALLS)
        max_cost = _integer(budget.get("max_cost_microusd"), "Spend limit", 0, MAX_COST_MICROUSD)
        max_seconds = _integer(budget.get("max_seconds"), "Time limit", 1, MAX_SECONDS)
        if idempotency_key is not None:
            _label(idempotency_key, "Idempotency key", 128)
        payload = _json(request)

        with self._transaction() as connection:
            if idempotency_key is not None:
                existing = self._find_submission(connection, workspace_id, idempotency_key, request)
                if existing is not None:
                    return existing
            workspace = connection.execute("SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
            if workspace is None:
                raise StoreError("Workspace does not exist.")
            active = connection.execute(
                """SELECT COUNT(*) FROM jobs
                   WHERE workspace_id = ? AND state IN ('queued', 'running', 'cancel_requested')""",
                (workspace_id,),
            ).fetchone()[0]
            if active >= MAX_ACTIVE_JOBS:
                raise Conflict(f"A workspace can have up to {MAX_ACTIVE_JOBS} active jobs.")
            job_id = _id("job")
            now = self._now()
            connection.execute(
                """INSERT INTO jobs
                    (id, workspace_id, idempotency_key, request, state, created_at, updated_at,
                     max_calls, max_cost_microusd, max_seconds)
                   VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)""",
                (job_id, workspace_id, idempotency_key, payload, now, now, max_calls, max_cost, max_seconds),
            )
            return self._view(connection, self._find(connection, workspace_id, job_id))

    def get(self, workspace_id: str, job_id: str) -> dict | None:
        with self._transaction(write=False) as connection:
            row = self._find(connection, workspace_id, job_id)
            return self._view(connection, row) if row is not None else None

    def list(self, workspace_id: str, limit: int = 50) -> list[dict]:
        _integer(limit, "Job list limit", 1, 100)
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (workspace_id, limit),
            )
            return [self._view(connection, row) for row in rows]

    def cancel(self, workspace_id: str, job_id: str) -> dict | None:
        with self._transaction() as connection:
            row = self._find(connection, workspace_id, job_id)
            if row is None:
                return None
            if row["state"] in ("queued", "running"):
                state = "cancelled" if row["state"] == "queued" else "cancel_requested"
                connection.execute(
                    "UPDATE jobs SET state = ?, updated_at = ? WHERE id = ?",
                    (state, self._now(), job_id),
                )
                row = self._find(connection, workspace_id, job_id)
            return self._view(connection, row)

    def claim(self, lease_seconds: int = 120) -> dict | None:
        duration = _lease_seconds(lease_seconds)
        with self._transaction() as connection:
            now = self._now()
            row = connection.execute(
                """SELECT id, workspace_id FROM jobs WHERE state = 'queued'
                   AND (deadline_at IS NULL OR deadline_at > ?)
                   ORDER BY created_at, rowid LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            lease_token = secrets.token_urlsafe(32)
            connection.execute(
                """UPDATE jobs SET state = 'running', lease_token = ?, lease_until = ?, updated_at = ?,
                          started_at = COALESCE(started_at, ?),
                          deadline_at = COALESCE(deadline_at, ? + max_seconds)
                   WHERE id = ?""",
                (lease_token, now + duration, now, now, now, row["id"]),
            )
            claimed = self._find(connection, row["workspace_id"], row["id"])
            return {**self._view(connection, claimed), "lease_token": lease_token}

    @staticmethod
    def _owned(connection: sqlite3.Connection, job_id: str, lease_token: str, now: float) -> sqlite3.Row:
        row = connection.execute(
            """SELECT workspace_id, state, deadline_at, max_calls, max_cost_microusd
               FROM jobs WHERE id = ? AND lease_token = ? AND lease_until > ?
               AND state IN ('running', 'cancel_requested')""",
            (job_id, lease_token, now),
        ).fetchone()
        if row is None:
            raise LeaseLost("The job lease has expired or been reassigned.")
        return row

    @staticmethod
    def _can_continue(row: sqlite3.Row, now: float) -> None:
        if row["state"] == "cancel_requested":
            raise Cancelled("Cancellation was requested.")
        if now >= row["deadline_at"]:
            raise BudgetExceeded("The job reached its time limit.")

    def heartbeat(self, job_id: str, lease_token: str, lease_seconds: int = 120) -> None:
        duration = _lease_seconds(lease_seconds)
        with self._transaction() as connection:
            now = self._now()
            self._owned(connection, job_id, lease_token, now)
            connection.execute("UPDATE jobs SET lease_until = ? WHERE id = ?", (now + duration, job_id))

    def worker_seen(self, worker_id: str) -> None:
        _label(worker_id, "Worker ID", 128)
        with self._transaction() as connection:
            now = self._now()
            connection.execute(
                """INSERT INTO workers (id, last_seen) VALUES (?, ?)
                   ON CONFLICT (id) DO UPDATE SET last_seen = excluded.last_seen""",
                (worker_id, now),
            )
            connection.execute("DELETE FROM workers WHERE last_seen < ?", (now - MAX_SECONDS,))

    def worker_ready(self, max_age_seconds: int = 30) -> bool:
        _integer(max_age_seconds, "Worker readiness window", 1, MAX_SECONDS)
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM workers WHERE last_seen >= ? LIMIT 1",
                (self._now() - max_age_seconds,),
            ).fetchone() is not None

    def check(self, job_id: str, lease_token: str) -> None:
        with self._transaction(write=False) as connection:
            now = self._now()
            row = self._owned(connection, job_id, lease_token, now)
            self._can_continue(row, now)

    def record_progress(self, job_id: str, lease_token: str, progress: dict) -> None:
        if not isinstance(progress, dict):
            raise StoreError("Progress must be a JSON object.")
        payload = _json(progress)
        if len(payload) > 16_384:
            raise StoreError("Progress updates must be 16 KiB or smaller.")
        with self._transaction() as connection:
            now = self._now()
            row = self._owned(connection, job_id, lease_token, now)
            self._can_continue(row, now)
            connection.execute(
                "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?", (payload, now, job_id),
            )

    def reserve(self, job_id: str, lease_token: str, max_cost_microusd: int, stage: str) -> str:
        _integer(max_cost_microusd, "Dispatch reservation", 0, MAX_COST_MICROUSD)
        _label(stage, "Dispatch stage", 96)
        reservation_id = _id("call")
        with self._transaction() as connection:
            now = self._now()
            row = self._owned(connection, job_id, lease_token, now)
            self._can_continue(row, now)
            usage = self._job_usage(connection, job_id)
            if usage["calls"] >= row["max_calls"]:
                raise BudgetExceeded("The job reached its dispatch limit.")
            if _committed(usage) + max_cost_microusd > row["max_cost_microusd"]:
                raise BudgetExceeded("There is insufficient budget for the next dispatch.")
            if max_cost_microusd:
                workspace_usage = self._workspace_usage(connection, row["workspace_id"])
                if _committed(workspace_usage) + max_cost_microusd > workspace_usage["spend_limit_microusd"]:
                    raise BudgetExceeded("There is insufficient workspace spend allowance for the next dispatch.")
                if self.site_spend_limit_microusd is not None:
                    site = self._site_usage(connection)
                    if _committed(site) + max_cost_microusd > self.site_spend_limit_microusd:
                        raise BudgetExceeded("The gateway has reached its total model spend limit for now.")
            connection.execute(
                """INSERT INTO dispatches (id, job_id, stage, state, max_cost_microusd, created_at)
                   VALUES (?, ?, ?, 'reserved', ?, ?)""",
                (reservation_id, job_id, stage, max_cost_microusd, now),
            )
        return reservation_id

    @staticmethod
    def _end(
        connection: sqlite3.Connection, job_id: str, state: str, now: float,
        *, result: str | None = None, error: str | None = None,
    ) -> None:
        connection.execute(
            """UPDATE dispatches SET state = 'uncertain', settled_at = ?
               WHERE job_id = ? AND state = 'reserved'""",
            (now, job_id),
        )
        connection.execute(
            """UPDATE jobs SET state = ?, result = ?, error = ?, updated_at = ?,
                       lease_token = NULL, lease_until = NULL WHERE id = ?""",
            (state, result, error, now, job_id),
        )

    def settle(self, job_id: str, lease_token: str, reservation_id: str, cost_microusd: int | None) -> None:
        if cost_microusd is not None:
            _integer(cost_microusd, "Actual dispatch cost", 0, MAX_COST_MICROUSD)
        state = "uncertain" if cost_microusd is None else "settled"
        with self._transaction() as connection:
            now = self._now()
            self._owned(connection, job_id, lease_token, now)
            reservation = connection.execute(
                "SELECT * FROM dispatches WHERE job_id = ? AND id = ?", (job_id, reservation_id),
            ).fetchone()
            if reservation is None:
                raise StoreError("Reservation does not belong to this job.")
            if reservation["state"] != "reserved":
                if reservation["state"] == state and reservation["cost_microusd"] == cost_microusd:
                    return
                raise Conflict("This dispatch has already been settled.")
            connection.execute(
                "UPDATE dispatches SET state = ?, cost_microusd = ?, settled_at = ? WHERE id = ?",
                (state, cost_microusd, now, reservation_id),
            )
            breach = cost_microusd is not None and cost_microusd > reservation["max_cost_microusd"]
            if breach:
                self._end(connection, job_id, "failed", now, error="Provider charge exceeded its dispatch reservation.")
        if breach:
            # Commit the observed charge and terminal state before signaling the error.
            raise BudgetExceeded("Provider charge exceeded its dispatch reservation.")

    def finish(
        self, job_id: str, lease_token: str, result: dict | None = None, error: str | None = None,
    ) -> None:
        if result is not None and not isinstance(result, dict):
            raise StoreError("Job results must be JSON objects.")
        if error is not None and not isinstance(error, str):
            raise StoreError("Job errors must be text.")
        payload = _json(result) if result is not None else None
        with self._transaction() as connection:
            now = self._now()
            row = self._owned(connection, job_id, lease_token, now)
            if row["state"] == "cancel_requested":
                self._end(connection, job_id, "cancelled", now)
                return
            if now >= row["deadline_at"]:
                self._end(connection, job_id, "failed", now, error="The job reached its time limit.")
                return
            open_dispatch = connection.execute(
                "SELECT 1 FROM dispatches WHERE job_id = ? AND state = 'reserved' LIMIT 1", (job_id,),
            ).fetchone()
            if payload is not None and error is None and open_dispatch is None:
                self._end(connection, job_id, "succeeded", now, result=payload)
            else:
                self._end(connection, job_id, "failed", now, error=(error or "The job ended before completion.")[:1024])

    def reap(self) -> int:
        """Recover expired work without replaying a possibly billable dispatch."""
        with self._transaction() as connection:
            now = self._now()
            rows = connection.execute(
                """SELECT id, state, deadline_at FROM jobs
                   WHERE (state IN ('running', 'cancel_requested') AND lease_until <= ?)
                      OR (state IN ('queued', 'running', 'cancel_requested') AND deadline_at <= ?)""",
                (now, now),
            ).fetchall()
            for row in rows:
                job_id = row["id"]
                if row["state"] == "cancel_requested":
                    self._end(connection, job_id, "cancelled", now)
                elif row["deadline_at"] <= now:
                    self._end(connection, job_id, "failed", now, error="The job reached its time limit.")
                elif self._job_usage(connection, job_id)["calls"]:
                    self._end(
                        connection, job_id, "failed", now,
                        error="Worker interrupted after dispatch. Review usage before restarting the job.",
                    )
                else:
                    connection.execute(
                        """UPDATE jobs SET state = 'queued', progress = NULL, updated_at = ?,
                                   lease_token = NULL, lease_until = NULL WHERE id = ?""",
                        (now, job_id),
                    )
            return len(rows)

    def usage(self, workspace_id: str) -> dict:
        with self._transaction(write=False) as connection:
            return self._workspace_usage(connection, workspace_id)

    def ledger(self, workspace_id: str, limit: int = 100) -> list[dict]:
        _integer(limit, "Ledger list limit", 1, 1000)
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT d.* FROM dispatches AS d JOIN jobs AS j ON d.job_id = j.id
                   WHERE j.workspace_id = ? ORDER BY d.created_at DESC, d.rowid DESC LIMIT ?""",
                (workspace_id, limit),
            )
            return [dict(row) for row in rows]

    def reconcile(self, reservation_id: str, cost_microusd: int) -> dict:
        """Apply a confirmed provider charge to an interrupted, uncertain dispatch."""
        _integer(cost_microusd, "Actual dispatch cost", 0, MAX_COST_MICROUSD)
        with self._transaction() as connection:
            now = self._now()
            entry = connection.execute(
                "SELECT d.*, j.state AS job_state FROM dispatches AS d JOIN jobs AS j ON d.job_id = j.id WHERE d.id = ?",
                (reservation_id,),
            ).fetchone()
            if entry is None:
                raise StoreError("Unknown dispatch reservation.")
            if entry["state"] == "settled" and entry["cost_microusd"] == cost_microusd:
                return {key: entry[key] for key in entry.keys() if key != "job_state"}
            if entry["state"] != "uncertain" or entry["job_state"] not in _TERMINAL:
                raise Conflict("Only uncertain dispatches on terminal jobs can be reconciled.")
            connection.execute(
                "UPDATE dispatches SET state = 'settled', cost_microusd = ?, settled_at = ? WHERE id = ?",
                (cost_microusd, now, reservation_id),
            )
            if cost_microusd > entry["max_cost_microusd"]:
                self._end(connection, entry["job_id"], "failed", now, error="Provider charge exceeded its dispatch reservation.")
            else:
                connection.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (now, entry["job_id"]))
            row = connection.execute("SELECT * FROM dispatches WHERE id = ?", (reservation_id,)).fetchone()
            return dict(row)

    @staticmethod
    def _request_view(row: sqlite3.Row) -> dict:
        return {key: row[key] for key in row.keys()}

    def request_access(self, email: str, name: str, use_case: str) -> dict:
        """Queue a request for a workspace key. A repeat request from a pending address is not duplicated."""
        if not isinstance(email, str) or len(email.strip()) > 254 or not _EMAIL.fullmatch(email.strip()):
            raise InvalidInput("Enter a valid email address.")
        if name is not None and (not isinstance(name, str) or len(name) > 120):
            raise InvalidInput("Keep your name to 120 characters or fewer.")
        if not isinstance(use_case, str) or not 10 <= len(use_case.strip()) <= 2000:
            raise InvalidInput("Describe what you want to improve in 10 to 2,000 characters.")
        email = email.strip().lower()
        name = name.strip() if isinstance(name, str) and name.strip() else "Not given"
        use_case = use_case.strip()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM access_requests WHERE email = ? AND state = 'pending'", (email,),
            ).fetchone()
            if existing is not None:
                return self._request_view(existing)
            pending = connection.execute("SELECT COUNT(*) FROM access_requests WHERE state = 'pending'").fetchone()[0]
            if pending >= MAX_PENDING_REQUESTS:
                raise Conflict("The request queue is full. Please try again later.")
            request_id = _id("req")
            connection.execute(
                """INSERT INTO access_requests (id, email, name, use_case, state, created_at)
                   VALUES (?, ?, ?, ?, 'pending', ?)""",
                (request_id, email, name, use_case, self._now()),
            )
            return self._request_view(connection.execute("SELECT * FROM access_requests WHERE id = ?", (request_id,)).fetchone())

    def access_requests(self, state: str | None = "pending", limit: int = 100) -> list[dict]:
        _integer(limit, "Access request list limit", 1, 500)
        if state not in (None, "pending", "approved", "declined"):
            raise StoreError("Filter access requests by pending, approved, or declined.")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT * FROM access_requests WHERE (? IS NULL OR state = ?)
                   ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                (state, state, limit),
            )
            return [self._request_view(row) for row in rows]

    def approve_access(self, request_id: str, spend_limit_microusd: int) -> dict:
        """Create the requester's workspace and write key in one step. The token is returned only here."""
        _integer(spend_limit_microusd, "Workspace spend limit", 0, MAX_COST_MICROUSD)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM access_requests WHERE id = ?", (request_id,)).fetchone()
            if row is None:
                raise InvalidInput("Unknown access request.")
            if row["state"] != "pending":
                raise Conflict(f"This request was already {row['state']}.")
            now = self._now()
            workspace_id = _id("ws")
            connection.execute(
                "INSERT INTO workspaces (id, name, created_at) VALUES (?, ?, ?)",
                (workspace_id, row["email"][:128], now),
            )
            connection.execute(
                "INSERT INTO workspace_limits (workspace_id, spend_limit_microusd) VALUES (?, ?)",
                (workspace_id, spend_limit_microusd),
            )
            key_id = _id("key")
            token = "rsi_" + secrets.token_urlsafe(32)
            connection.execute(
                "INSERT INTO workspace_keys (id, workspace_id, role, token_hash, created_at) VALUES (?, ?, 'write', ?, ?)",
                (key_id, workspace_id, hashlib.sha256(token.encode("ascii")).hexdigest(), now),
            )
            connection.execute(
                "UPDATE access_requests SET state = 'approved', workspace_id = ?, decided_at = ? WHERE id = ?",
                (workspace_id, now, request_id),
            )
            request = self._request_view(connection.execute("SELECT * FROM access_requests WHERE id = ?", (request_id,)).fetchone())
        return {"request": request, "workspace_id": workspace_id, "spend_limit_microusd": spend_limit_microusd,
                "key": {"id": key_id, "token": token}}

    def decline_access(self, request_id: str) -> dict:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM access_requests WHERE id = ?", (request_id,)).fetchone()
            if row is None:
                raise InvalidInput("Unknown access request.")
            if row["state"] != "pending":
                raise Conflict(f"This request was already {row['state']}.")
            connection.execute(
                "UPDATE access_requests SET state = 'declined', decided_at = ? WHERE id = ?", (self._now(), request_id),
            )
            return self._request_view(connection.execute("SELECT * FROM access_requests WHERE id = ?", (request_id,)).fetchone())

    def backup(self, path: str | os.PathLike[str]) -> None:
        """Write a consistent online backup into a new owner-readable file."""
        destination = Path(path)
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with closing(self._connect()) as source, closing(sqlite3.connect(destination)) as target:
                source.backup(target)
        except BaseException:
            destination.unlink()
            raise
