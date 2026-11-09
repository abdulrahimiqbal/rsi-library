"""Claim durable jobs and meter provider calls before sending them."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid

from .methods import verify_candidate
from .providers import Model, ProviderError
from .routing import Adapter, call_capacity, training_request
from .store import BudgetExceeded, Cancelled, LeaseLost, Store, StoreError


log = logging.getLogger(__name__)
_LEASE_SECONDS = 45


class Worker:
    def __init__(self, store: Store, adapters: dict[str, Adapter], models: dict[str, Model]):
        self.store = store
        self.adapters = adapters
        self.models = models
        self.id = uuid.uuid4().hex
        self.stop = threading.Event()

    def once(self) -> bool:
        self.store.worker_seen(self.id)
        self.store.reap()
        job = self.store.claim(_LEASE_SECONDS)
        if not job:
            return False
        self._run(job)
        return True

    def _run(self, job: dict) -> None:
        job_id, lease = job["id"], job["lease_token"]
        request = job["request"]
        route = request["_routing"]
        model = self.models.get(route["model"])
        adapter = self.adapters.get(route["method"])
        if model is None or adapter is None or model.fingerprint != route["model_fingerprint"] or not model.available():
            self.store.finish(job_id, lease, error="The selected optimizer or model configuration changed. Submit a new job.")
            return

        done = threading.Event()

        def monitor() -> None:
            while not done.wait(5):
                try:
                    self.store.heartbeat(job_id, lease, _LEASE_SECONDS)
                    self.store.worker_seen(self.id)
                except LeaseLost:
                    return
                except StoreError:
                    log.warning("Could not renew lease for job %s", job_id)

        watch = threading.Thread(target=monitor, daemon=True)
        watch.start()
        deadline = time.monotonic() + max(0, job["deadline_at"] - time.time())

        def check() -> None:
            if self.stop.is_set():
                raise StoreError("Worker shutdown interrupted the job. Review usage before submitting again.")
            self.store.check(job_id, lease)

        def complete(stage: str, messages: list[dict[str, str]]) -> str:
            check()
            model.check_input(messages)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded("The job reached its time limit.")
            reservation = self.store.reserve(job_id, lease, model.reservation, stage)
            try:
                reply = model.complete(stage, messages, min(model.timeout_seconds, remaining), dispatch_id=reservation)
            except Exception:
                self.store.settle(job_id, lease, reservation, None)
                raise
            self.store.settle(job_id, lease, reservation, reply.cost_microusd)
            if reply.input_tokens > model.max_input_tokens or reply.output_tokens > model.max_output_tokens:
                raise ProviderError("The model reported token usage beyond its configured ceiling.")
            if reply.text is None:
                raise ProviderError("The model did not return a text completion.")
            check()
            return reply.text

        def progress(event: dict) -> None:
            self.store.record_progress(job_id, lease, event)

        def fail(message: str) -> None:
            try:
                self.store.finish(job_id, lease, error=message)
            except LeaseLost:
                # A reaper or a provider charge breach may already have closed this job.
                return

        try:
            capacity = call_capacity(request, model)
            if adapter.external:
                result = self._execute_external(adapter, request, capacity, complete, check, progress)
            else:
                result = adapter.execute(
                    request, route["method"], complete, check, call_capacity=capacity, progress=progress,
                )
            if not isinstance(result, dict):
                raise ValueError("The optimizer did not return a result object.")
            check()
            self.store.finish(job_id, lease, result={**result, "routing": route})
            log.info("Job finished %s", job_id)
        except LeaseLost:
            log.warning("Lost lease for job %s", job_id)
        except Cancelled:
            fail("Job cancelled.")
        except (BudgetExceeded, ProviderError, StoreError, ValueError) as exc:
            fail(str(exc))
            log.info("Job stopped %s (%s)", job_id, type(exc).__name__)
        except Exception as exc:
            fail("The optimizer stopped unexpectedly. See the run usage before retrying.")
            log.error("Job failed %s (%s)", job_id, type(exc).__name__)
        finally:
            done.set()
            watch.join()

    @staticmethod
    def _execute_external(adapter, request, capacity, complete, check, progress) -> dict:
        development = training_request(request, capacity)
        remaining_calls = development["budget"]["max_calls"]

        def train_complete(stage: str, messages: list[dict[str, str]]) -> str:
            nonlocal remaining_calls
            if remaining_calls <= 0:
                raise BudgetExceeded("The optimizer used its search call allocation.")
            remaining_calls -= 1
            return complete(stage, messages)

        method = adapter.metadata["id"]
        proposed = adapter.execute(
            development, method, train_complete, check, call_capacity=remaining_calls, progress=progress,
        )
        if not isinstance(proposed, dict) or "candidate" not in proposed:
            raise ValueError("The external optimizer did not return a candidate.")
        progress({"stage": "baseline_holdout"})
        return {
            "method": method,
            "trials": proposed.get("trials", []),
            **verify_candidate(request, proposed["candidate"], complete),
        }

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                if self.once():
                    continue
            except (StoreError, sqlite3.Error):
                log.warning("Worker could not read or update the queue")
            self.stop.wait(1)
