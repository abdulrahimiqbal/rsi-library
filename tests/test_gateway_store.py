"""Regression coverage for job ownership, dispatch accounting, and crash recovery."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from rsi_library.gateway.store import BudgetExceeded, Cancelled, Conflict, LeaseLost, Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "gateway.sqlite"
        self.now = 10.0
        self.store = Store(self.path, clock=lambda: self.now)
        self.workspace = self.store.create_workspace("Example", spend_limit_microusd=1000)
        self.request = {
            "method": "auto",
            "budget": {"max_calls": 3, "max_cost_microusd": 100, "max_seconds": 30},
        }

    def submit(self, **request):
        return self.store.submit(self.workspace, {**self.request, **request})

    def test_credentials_and_jobs_are_workspace_scoped(self):
        other = self.store.create_workspace("Another")
        read_key = self.store.create_key(self.workspace, "read")
        write_key = self.store.create_key(other)
        self.assertEqual(self.store.authenticate(read_key["token"]), {"workspace_id": self.workspace, "role": "read"})
        self.assertEqual(self.store.authenticate(write_key["token"]), {"workspace_id": other, "role": "write"})
        self.assertTrue(self.store.revoke_key(read_key["id"]))
        self.assertFalse(self.store.revoke_key(read_key["id"]))
        self.assertIsNone(self.store.authenticate(read_key["token"]))

        job = self.submit()
        self.assertIsNone(self.store.get(other, job["id"]))
        self.assertIsNone(self.store.cancel(other, job["id"]))
        self.assertEqual(self.store.list(other), [])
        self.assertEqual(self.store.get(self.workspace, job["id"])["state"], "queued")

    def test_hosted_bootstrap_preserves_the_workspace_and_does_not_revive_a_key(self):
        original = "rsi_" + "a" * 43
        replacement = "rsi_" + "b" * 43
        digest = hashlib.sha256(original.encode()).hexdigest()
        self.store.bootstrap_workspace(digest)
        owner = self.store.authenticate(original)
        self.assertEqual(owner["role"], "write")
        self.assertEqual(self.store.usage(owner["workspace_id"])["spend_limit_microusd"], 0)
        restarted = Store(self.path)
        restarted.bootstrap_workspace(hashlib.sha256(replacement.encode()).hexdigest())
        self.assertEqual(restarted.authenticate(original), owner)
        self.assertIsNone(restarted.authenticate(replacement))
        with closing(sqlite3.connect(self.path)) as connection:
            key_id = connection.execute("SELECT id FROM workspace_keys WHERE token_hash = ?", (digest,)).fetchone()[0]
        self.assertTrue(restarted.revoke_key(key_id))
        restarted.bootstrap_workspace(digest)
        self.assertIsNone(restarted.authenticate(original))

    def test_concurrent_submission_and_claim_do_not_duplicate_jobs(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(
                lambda index: self.store.submit(self.workspace, {**self.request, "_routing": [index]}, "one"),
                range(8),
            ))
            leases = list(pool.map(lambda _: self.store.claim(), range(8)))
        self.assertEqual(len({job["id"] for job in jobs}), 1)
        self.assertEqual(len({tuple(job["request"]["_routing"]) for job in jobs}), 1)
        self.assertEqual(sum(lease is not None for lease in leases), 1)
        self.assertNotIn("lease_token", self.store.get(self.workspace, jobs[0]["id"]))
        self.assertEqual(self.store.find_submission(self.workspace, "one", self.request)["id"], jobs[0]["id"])
        with self.assertRaises(Conflict):
            self.store.submit(self.workspace, {**self.request, "method": "another"}, "one")

    def test_reservations_are_atomic_and_include_uncertain_spend(self):
        self.submit()
        job = self.store.claim()
        job_id, lease = job["id"], job["lease_token"]

        def dispatch(_):
            try:
                return self.store.reserve(job_id, lease, 60, "proposal")
            except BudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            reservations = [result for result in pool.map(dispatch, range(6)) if result is not None]
        self.assertEqual(len(reservations), 1)
        self.store.settle(job_id, lease, reservations[0], None)
        with self.assertRaises(BudgetExceeded):
            self.store.reserve(job_id, lease, 41, "judge")
        second = self.store.reserve(job_id, lease, 40, "judge")
        self.store.settle(job_id, lease, second, 20)
        third = self.store.reserve(job_id, lease, 20, "judge")
        self.store.settle(job_id, lease, third, 15)
        with self.assertRaises(BudgetExceeded):
            self.store.reserve(job_id, lease, 0, "another")
        self.store.finish(job_id, lease, {"verified": True})
        self.assertEqual(self.store.get(self.workspace, job_id)["usage"], {
            "calls": 3, "spent_microusd": 35, "reserved_microusd": 0, "uncertain_microusd": 60,
        })
        self.assertEqual(self.store.ledger(self.workspace)[-1]["state"], "uncertain")
        self.store.reconcile(reservations[0], 45)
        self.assertEqual(self.store.usage(self.workspace)["spent_microusd"], 80)
        self.assertEqual(self.store.usage(self.workspace)["uncertain_microusd"], 0)

    def test_provider_breach_records_actual_charge_and_fails_job(self):
        self.submit()
        job = self.store.claim()
        job_id, lease = job["id"], job["lease_token"]
        reservation = self.store.reserve(job_id, lease, 30, "proposal")
        with self.assertRaises(BudgetExceeded):
            self.store.settle(job_id, lease, reservation, 39)
        terminal = self.store.get(self.workspace, job_id)
        self.assertEqual(terminal["state"], "failed")
        self.assertEqual(terminal["usage"]["spent_microusd"], 39)
        with self.assertRaises(LeaseLost):
            self.store.reserve(job_id, lease, 0, "judge")

    def test_workspace_spend_cap_covers_concurrent_jobs_and_uncertain_charges(self):
        workspace = self.store.create_workspace("Spend limited")
        self.store.submit(workspace, self.request)
        self.store.submit(workspace, self.request)
        jobs = [self.store.claim(), self.store.claim()]
        with self.assertRaises(BudgetExceeded):
            self.store.reserve(jobs[0]["id"], jobs[0]["lease_token"], 1, "proposal")
        self.store.set_spend_limit(workspace, 80)

        def reserve(job):
            try:
                return self.store.reserve(job["id"], job["lease_token"], 60, "proposal")
            except BudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            reservations = list(pool.map(reserve, jobs))
        self.assertEqual(sum(reservation is not None for reservation in reservations), 1)
        winner = next(index for index, reservation in enumerate(reservations) if reservation is not None)
        job = jobs[winner]
        self.store.settle(job["id"], job["lease_token"], reservations[winner], None)
        self.store.finish(job["id"], job["lease_token"], error="Unconfirmed usage")
        other = jobs[1 - winner]
        with self.assertRaises(BudgetExceeded):
            self.store.reserve(other["id"], other["lease_token"], 21, "proposal")
        self.store.reconcile(reservations[winner], 40)
        self.store.reserve(other["id"], other["lease_token"], 40, "proposal")
        self.assertEqual(self.store.usage(workspace)["spend_limit_microusd"], 80)

    def test_lease_recovery_requeues_only_undispatched_work(self):
        first = self.submit()
        lease1 = self.store.claim(lease_seconds=2)
        self.now += 3
        reopened = Store(self.path, clock=lambda: self.now)
        self.assertEqual(reopened.reap(), 1)
        lease2 = reopened.claim(lease_seconds=2)
        self.assertEqual(lease2["id"], first["id"])
        self.assertEqual(lease2["deadline_at"], lease1["deadline_at"])
        with self.assertRaises(LeaseLost):
            reopened.reserve(first["id"], lease1["lease_token"], 20, "proposal")

        reservation = reopened.reserve(first["id"], lease2["lease_token"], 75, "proposal")
        self.now += 3
        self.assertEqual(reopened.reap(), 1)
        self.assertIsNone(reopened.claim())
        terminal = reopened.get(self.workspace, first["id"])
        self.assertEqual(terminal["state"], "failed")
        self.assertEqual(terminal["usage"]["uncertain_microusd"], 75)
        reopened.reconcile(reservation, 50)
        self.assertEqual(reopened.usage(self.workspace)["spent_microusd"], 50)

    def test_cancel_allows_settlement_without_further_dispatch(self):
        self.submit()
        job = self.store.claim()
        job_id, lease = job["id"], job["lease_token"]
        reservation = self.store.reserve(job_id, lease, 50, "judge")
        self.assertEqual(self.store.cancel(self.workspace, job_id)["state"], "cancel_requested")
        with self.assertRaises(Cancelled):
            self.store.reserve(job_id, lease, 0, "proposal")
        self.store.settle(job_id, lease, reservation, 15)
        self.store.finish(job_id, lease, {"candidate": "never deliver"})
        terminal = self.store.get(self.workspace, job_id)
        self.assertEqual(terminal["state"], "cancelled")
        self.assertIsNone(terminal["result"])
        self.assertEqual(terminal["usage"]["spent_microusd"], 15)

    def test_backup_preserves_ledger_and_job_result(self):
        self.submit()
        job = self.store.claim()
        job_id, lease = job["id"], job["lease_token"]
        reservation = self.store.reserve(job_id, lease, 25, "judge")
        self.store.settle(job_id, lease, reservation, 11)
        self.store.finish(job_id, lease, {"candidate": "backup artifact"})
        destination = self.path.with_name("backup.sqlite")
        self.store.backup(destination)
        backup = Store(destination)
        self.assertEqual(backup.get(self.workspace, job_id), self.store.get(self.workspace, job_id))
        self.assertEqual(backup.ledger(self.workspace), self.store.ledger(self.workspace))

    def test_time_limit_and_worker_readiness(self):
        self.assertFalse(self.store.worker_ready())
        self.store.worker_seen("worker-1")
        self.assertTrue(self.store.worker_ready())
        self.submit(budget={**self.request["budget"], "max_seconds": 3})
        job = self.store.claim()
        self.now += 31
        self.assertFalse(self.store.worker_ready())
        with self.assertRaises(BudgetExceeded):
            self.store.reserve(job["id"], job["lease_token"], 0, "proposal")
        self.assertEqual(self.store.reap(), 1)
        self.assertEqual(self.store.get(self.workspace, job["id"])["state"], "failed")


if __name__ == "__main__":
    unittest.main()
