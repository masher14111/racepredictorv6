"""Stage-5 realistic execution, forward validation and betting safeguards.

Stage 4 (``reports/calibration_audit_20260727.md``) issued **MODEL NO-GO**: on an
untouched 904-race window the independent price-free line loses to the de-vigged
pre-off market by 0.186 race log-loss (95% CI [-0.225, -0.146]) and closing-line
value is -13.4%. Everything in this package therefore exists to make **paper**
betting honest and **"no bet" the default answer** — not to find a threshold that
makes a backtest look profitable.

Modules
-------
config      — the ``execution:`` config contract (risk ceilings, gates, forward
              criteria) with provisional clamps that can only be lifted by a
              passing forward gate.
snapshots   — append-only point-in-time odds store (migration 4). A backtest may
              only ever read quotes with ``fetched_at <= as_of``; the closing
              price is never available as the price supposedly offered earlier.
frictions   — decision/placement latency, price movement, rejections,
              suspensions, bookmaker limits, exchange commission, BOG-only-if-
              recorded.
settlement  — non-runners, Rule 4 deductions, dead heats, voids, each-way terms
              captured at bet time.
staking     — provisional fractional-Kelly under per-bet / per-race / per-day
              exposure ceilings. No accumulators, no correlated bets.
safeguards  — bankroll stop-loss, daily loss limit, duplicate + started-race +
              stale-source blocking, and the permanent paper-only override.
gates       — PASS ("no bet") unless every candidate condition is affirmatively
              met, including the Stage-4 model-validation gate.
tickets     — the full disclosure record every candidate/paper ticket carries.
baselines   — model-only vs de-vigged market vs simple favourite, same races.
evaluation  — CLV, ROI/yield, A/E, hit rate, drawdown, turnover, losing streaks,
              sample size, race-clustered bootstrap intervals.
selection   — tune on train folds, select once, evaluate once; count and report
              every strategy tried and the multiple-testing correction.
forward_gate— the >= 8 week forward-release criteria (config-driven, floored).
simulator   — the walk-forward execution simulator wiring the above together.
report      — dated forward-validation and "today's candidates" reports that keep
              historical backtest, shadow/paper and (hypothetical) real execution
              strictly separated.
"""
from __future__ import annotations

from execution.config import (
    ExecutionConfig,
    ForwardGateConfig,
    FrictionConfig,
    GateConfig,
    ReportConfig,
    SafeguardConfig,
    SelectionConfig,
    SnapshotConfig,
    StakingConfig,
    real_money_enabled,
)

__all__ = [
    "ExecutionConfig",
    "ForwardGateConfig",
    "FrictionConfig",
    "GateConfig",
    "ReportConfig",
    "SafeguardConfig",
    "SelectionConfig",
    "SnapshotConfig",
    "StakingConfig",
    "real_money_enabled",
]
