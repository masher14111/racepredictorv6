"""Prove the hosted-trial budget ledger enforces its caps BEFORE any live call.

AGENTS.md/D19/prompt 15 scope 5: budget enforcement must be proven with
mocked responses first. These tests never touch a network.
"""
import json

import pytest

from llm.hosted_budget import BudgetExceeded, HostedBudgetLedger


def _ledger(tmp_path, max_cost_usd=1.0, max_requests=10):
    return HostedBudgetLedger(
        tmp_path / "ledger.json", max_cost_usd=max_cost_usd, max_requests=max_requests
    )


def test_fresh_ledger_starts_at_zero_spend(tmp_path):
    led = _ledger(tmp_path)
    snap = led.snapshot()
    assert snap["committed_cost_usd"] == 0.0
    assert snap["committed_requests"] == 0
    assert snap["remaining_cost_usd"] == 1.0
    assert snap["remaining_requests"] == 10


def test_reserve_then_commit_moves_reserved_to_committed(tmp_path):
    led = _ledger(tmp_path)
    token = led.reserve(0.10, requests=1, note="test")
    snap = led.snapshot()
    assert snap["reserved_cost_usd"] == pytest.approx(0.10)
    assert snap["remaining_cost_usd"] == pytest.approx(0.90)
    led.commit(token, actual_cost_usd=0.07, requests=1)
    snap = led.snapshot()
    assert snap["committed_cost_usd"] == pytest.approx(0.07)
    assert snap["reserved_cost_usd"] == pytest.approx(0.0)
    assert snap["remaining_cost_usd"] == pytest.approx(0.93)


def test_release_returns_unused_reservation(tmp_path):
    led = _ledger(tmp_path)
    token = led.reserve(0.50, requests=1)
    led.release(token)
    snap = led.snapshot()
    assert snap["reserved_cost_usd"] == 0.0
    assert snap["remaining_cost_usd"] == pytest.approx(1.0)


def test_reserve_refuses_when_cost_cap_would_be_exceeded(tmp_path):
    led = _ledger(tmp_path, max_cost_usd=0.05)
    with pytest.raises(BudgetExceeded):
        led.reserve(0.06, requests=1)
    # nothing was reserved by the failed call
    assert led.snapshot()["reserved_cost_usd"] == 0.0


def test_reserve_refuses_when_request_cap_would_be_exceeded(tmp_path):
    led = _ledger(tmp_path, max_cost_usd=10.0, max_requests=2)
    led.reserve(0.01, requests=2)
    with pytest.raises(BudgetExceeded):
        led.reserve(0.01, requests=1)


def test_commit_cannot_exceed_its_own_reservation_fail_closed(tmp_path):
    led = _ledger(tmp_path)
    token = led.reserve(0.10, requests=1)
    with pytest.raises(BudgetExceeded):
        led.commit(token, actual_cost_usd=0.50, requests=1)


def test_ledger_persists_and_never_resets_on_resume(tmp_path):
    path = tmp_path / "ledger.json"
    led1 = HostedBudgetLedger(path, max_cost_usd=5.0, max_requests=1200)
    token = led1.reserve(1.0)
    led1.commit(token, actual_cost_usd=1.0)
    assert led1.snapshot()["committed_cost_usd"] == pytest.approx(1.0)

    # simulate a brand-new process resuming: same path, fresh instance
    led2 = HostedBudgetLedger(path, max_cost_usd=5.0, max_requests=1200)
    assert led2.snapshot()["committed_cost_usd"] == pytest.approx(1.0)
    assert led2.snapshot()["remaining_cost_usd"] == pytest.approx(4.0)

    # spend right up to the remaining cap...
    token2 = led2.reserve(4.0)
    led2.commit(token2, actual_cost_usd=4.0)
    # ...and a third resume must refuse ANY further spend, even $0.01
    led3 = HostedBudgetLedger(path, max_cost_usd=5.0, max_requests=1200)
    assert led3.snapshot()["remaining_cost_usd"] == pytest.approx(0.0)
    with pytest.raises(BudgetExceeded):
        led3.reserve(0.01)


def test_ledger_file_is_plain_json_with_full_audit_history(tmp_path):
    path = tmp_path / "ledger.json"
    led = HostedBudgetLedger(path, max_cost_usd=5.0, max_requests=1200)
    token = led.reserve(0.02, note="unit-test-reserve")
    led.commit(token, actual_cost_usd=0.015, note="unit-test-commit")
    raw = json.loads(path.read_text(encoding="utf-8"))
    kinds = [h["kind"] for h in raw["history"]]
    assert "reserve" in kinds and "commit" in kinds


def test_concurrent_reservations_never_exceed_the_cap(tmp_path):
    """Stage 19 needs the ledger safe under concurrent reserve() calls from
    multiple threads (e.g. a batch extractor is not strictly serial). The
    lock in HostedBudgetLedger.reserve must make the check-then-reserve
    sequence atomic -- without it, two threads could both read "room for one
    more" and both reserve, breaching the cap."""
    import threading

    led = _ledger(tmp_path, max_cost_usd=0.01, max_requests=1000)
    per_reservation = 0.002  # exactly 5 fit in the cap
    successes = []
    lock = threading.Lock()

    def worker():
        try:
            token = led.reserve(per_reservation, requests=1)
        except BudgetExceeded:
            return
        with lock:
            successes.append(token)

    threads = [threading.Thread(target=worker) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 5
    snap = led.snapshot()
    assert snap["reserved_cost_usd"] == pytest.approx(0.01)
    assert snap["remaining_cost_usd"] == pytest.approx(0.0, abs=1e-9)


def test_total_cap_holds_across_many_small_reservations(tmp_path):
    """1,200-request-shaped simulation: the running total must never exceed
    the cap even though each individual reservation is small."""
    led = _ledger(tmp_path, max_cost_usd=0.05, max_requests=50)
    per_request = 0.001
    committed = 0
    for _ in range(50):
        try:
            token = led.reserve(per_request)
        except BudgetExceeded:
            break
        led.commit(token, actual_cost_usd=per_request)
        committed += 1
    assert committed == 50  # exactly fills 0.05 / 0.001
    with pytest.raises(BudgetExceeded):
        led.reserve(per_request)
    assert led.snapshot()["committed_cost_usd"] <= 0.05 + 1e-9
