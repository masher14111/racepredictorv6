"""Persistent, fail-closed spend/request ledger for hosted LLM trials (step 15).

AGENTS.md/D19: across ALL attempts and resumes of this improvement programme,
hosted extraction spend is capped at ``max_cost_usd`` and ``max_requests``
total — the cap is on the PROGRAMME, not on one process run. The ledger file
on disk is the sole source of truth: a fresh process re-reads it and adds to
whatever is already committed, it never resets counters to zero on start.

Reserve-then-commit, not spend-then-check
------------------------------------------
A caller must ``reserve()`` the WORST-CASE cost of a request (max output
tokens, at the highest applicable price tier, multiplied by
``1 + max_retries`` attempts) *before* sending it. ``reserve`` raises
:class:`BudgetExceeded` synchronously if that would breach either cap — the
request is never sent in that case. After the real call finishes,
``commit()`` replaces the reservation with the ACTUAL billed cost (which must
not exceed what was reserved — a fail-closed :class:`BudgetExceeded` if the
provider bills more than the reservation covered, rather than silently
letting it through). ``release()`` gives back a reservation that was never
used (e.g. every retry failed before any billable call went out), so
accounting stays honest without ever letting the in-flight total exceed the
cap even transiently.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BASE = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _BASE / "data" / "cache" / "llm" / "hosted_budget_ledger.json"
_MAX_HISTORY = 5000  # bounded audit trail; oldest entries drop first


class BudgetExceeded(RuntimeError):
    """A reservation or commit would breach the persistent programme cap."""


class PricingUnbounded(RuntimeError):
    """Provider pricing is unknown or exceeds the configured ceiling."""


@dataclass
class _Reservation:
    cost_usd: float
    requests: int
    note: str


@dataclass
class LedgerState:
    schema_version: int = 1
    max_cost_usd: float = 0.0
    max_requests: int = 0
    committed_cost_usd: float = 0.0
    committed_requests: int = 0
    reservations: dict = field(default_factory=dict)  # token -> {cost_usd, requests, note}
    history: list = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HostedBudgetLedger:
    """File-backed ledger. Safe for one process; callers must not run two
    concurrent processes against the same ``path`` (no cross-process lock —
    matches this programme's single-writer-per-file rule, AGENTS.md)."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        max_cost_usd: Optional[float] = None,
        max_requests: Optional[int] = None,
    ) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self._lock = threading.Lock()
        self._load(max_cost_usd, max_requests)

    # -- persistence ----------------------------------------------------------
    def _load(self, max_cost_usd: Optional[float], max_requests: Optional[int]) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.state = LedgerState(
                schema_version=raw.get("schema_version", 1),
                max_cost_usd=raw.get("max_cost_usd", 0.0),
                max_requests=raw.get("max_requests", 0),
                committed_cost_usd=raw.get("committed_cost_usd", 0.0),
                committed_requests=raw.get("committed_requests", 0),
                reservations=raw.get("reservations", {}),
                history=raw.get("history", []),
            )
            # The configured cap may only ever be REQUESTED tighter or equal to
            # what governed prior spend; a caller asking for a higher cap than
            # was ever persisted cannot retroactively license already-approved
            # spend to grow — but the caps themselves are config, not spend, so
            # we always adopt the currently configured value and log the change
            # rather than silently keeping a stale number.
            if max_cost_usd is not None and max_cost_usd != self.state.max_cost_usd:
                self._log("cap_change", 0.0, 0, f"max_cost_usd {self.state.max_cost_usd} -> {max_cost_usd}")
                self.state.max_cost_usd = max_cost_usd
            if max_requests is not None and max_requests != self.state.max_requests:
                self._log("cap_change", 0.0, 0, f"max_requests {self.state.max_requests} -> {max_requests}")
                self.state.max_requests = max_requests
        else:
            self.state = LedgerState(
                max_cost_usd=max_cost_usd or 0.0, max_requests=max_requests or 0
            )
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".tmp{uuid.uuid4().hex[:8]}")
        payload = asdict(self.state)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _log(self, kind: str, cost_usd: float, requests: int, note: str) -> None:
        self.state.history.append(
            {"ts": _now_iso(), "kind": kind, "cost_usd": cost_usd, "requests": requests, "note": note}
        )
        if len(self.state.history) > _MAX_HISTORY:
            self.state.history = self.state.history[-_MAX_HISTORY:]

    # -- accounting -------------------------------------------------------------
    @property
    def _reserved_cost(self) -> float:
        return sum(r["cost_usd"] for r in self.state.reservations.values())

    @property
    def _reserved_requests(self) -> int:
        return sum(r["requests"] for r in self.state.reservations.values())

    @property
    def remaining_cost_usd(self) -> float:
        return self.state.max_cost_usd - self.state.committed_cost_usd - self._reserved_cost

    @property
    def remaining_requests(self) -> int:
        return self.state.max_requests - self.state.committed_requests - self._reserved_requests

    def snapshot(self) -> dict:
        return {
            "max_cost_usd": self.state.max_cost_usd,
            "max_requests": self.state.max_requests,
            "committed_cost_usd": self.state.committed_cost_usd,
            "committed_requests": self.state.committed_requests,
            "reserved_cost_usd": self._reserved_cost,
            "reserved_requests": self._reserved_requests,
            "remaining_cost_usd": self.remaining_cost_usd,
            "remaining_requests": self.remaining_requests,
        }

    def reserve(self, cost_usd: float, requests: int = 1, note: str = "") -> str:
        """Reserve worst-case cost/requests. Raises BudgetExceeded, sends nothing."""
        with self._lock:
            if cost_usd < 0 or requests < 0:
                raise ValueError("reserve: cost_usd/requests must be >= 0")
            if cost_usd - self.remaining_cost_usd > 1e-9:
                raise BudgetExceeded(
                    f"cost cap: remaining=${self.remaining_cost_usd:.6f} < "
                    f"requested=${cost_usd:.6f}"
                )
            if requests > self.remaining_requests:
                raise BudgetExceeded(
                    f"request cap: remaining={self.remaining_requests} < requested={requests}"
                )
            token = f"{time.time_ns():x}-{uuid.uuid4().hex[:8]}"
            self.state.reservations[token] = {"cost_usd": cost_usd, "requests": requests, "note": note}
            self._log("reserve", cost_usd, requests, note)
            self._save()
            return token

    def commit(self, token: str, actual_cost_usd: float, requests: int = 1, note: str = "") -> None:
        """Replace a reservation with the actual billed cost. Never exceeds
        what was reserved for that token — a provider billing MORE than the
        worst-case reservation covered is a pricing/reservation bug, and this
        fails closed instead of quietly overspending the cap."""
        with self._lock:
            resv = self.state.reservations.get(token)
            if resv is None:
                raise KeyError(f"commit: unknown reservation token {token!r}")
            if actual_cost_usd - resv["cost_usd"] > 1e-9:
                raise BudgetExceeded(
                    f"commit ${actual_cost_usd:.6f} exceeds reservation ${resv['cost_usd']:.6f} "
                    f"for token {token} — reservation sizing is unbound, refusing to overspend"
                )
            del self.state.reservations[token]
            self.state.committed_cost_usd += actual_cost_usd
            self.state.committed_requests += requests
            self._log("commit", actual_cost_usd, requests, note)
            self._save()

    def release(self, token: str, note: str = "") -> None:
        """Give back an unused reservation (all retries failed before billing)."""
        with self._lock:
            resv = self.state.reservations.pop(token, None)
            if resv is None:
                return
            self._log("release", resv["cost_usd"], resv["requests"], note)
            self._save()
