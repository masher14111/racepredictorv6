"""Render a :class:`~backtest.engine.BacktestRun` and persist it for the UI.

``render_report`` builds a plain-text summary (headline model metrics, a
per-strategy comparison table, and the A/E calibration tables). ``save_run``
writes a machine-readable bundle under ``data/backtests/<run_id>/`` that the
Streamlit app can list and load: ``summary.json`` (config + folds + per-strategy
metrics + A/E tables) plus one ``bets_<strategy>.parquet`` ledger each.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from backtest.engine import BacktestRun
from utils.logger import get_logger
from utils.timezone import now

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_OUT = os.path.join(_BASE, "data", "backtests")

# Summary keys shown in the per-strategy comparison, in display order.
_COMPARE_COLS = [
    "strategy", "n_bets", "staked", "profit", "yield_pct", "hit_rate",
    "bankroll_growth_pct", "max_drawdown_pct", "sharpe_per_bet",
    "clv_pct_mean", "beat_close_rate", "avg_odds", "avg_win_prob",
]


def compare_table(run: BacktestRun) -> pd.DataFrame:
    """Per-strategy summary as a DataFrame (one row per strategy)."""
    rows = [s["summary"] for s in run.strategies.values()]
    if not rows:
        return pd.DataFrame(columns=_COMPARE_COLS)
    df = pd.DataFrame(rows)
    cols = [c for c in _COMPARE_COLS if c in df.columns]
    return df[cols].sort_values("yield_pct", ascending=False, na_position="last").reset_index(drop=True)


def render_report(run: BacktestRun) -> str:
    """Human-readable text report for the console / logs."""
    L = []
    sep = "═" * 78
    L.append(sep)
    L.append("  WALK-FORWARD BACKTEST")
    L.append(sep)

    wf = run.config.get("walk_forward", {})
    scored_folds = sum(1 for f in run.folds if f.get("scored"))
    L.append(f"  Folds: {scored_folds} scored / {len(run.folds)} total   "
             f"(min_train={wf.get('min_train_days')}d, window={wf.get('test_window_days')}d, "
             f"{'rolling' if wf.get('rolling') else 'expanding'})")
    L.append(f"  Bankroll start: €{run.config.get('initial_bankroll'):,.0f}   "
             f"commission: {run.config.get('commission', 0):.1%}   "
             f"stop-loss: {run.config.get('stop_loss_pct')}")

    m = run.model_metrics
    L.append("")
    L.append(f"  Model (out-of-sample, {m.get('n', 0):,} runners, "
             f"base-rate {m.get('base_rate')}):")
    L.append(f"    AUC={m.get('auc')}  logloss={m.get('logloss')}  "
             f"Brier={m.get('brier')}  ECE={m.get('ece')}")

    L.append("")
    L.append("  Per-strategy comparison (sorted by yield):")
    cmp = compare_table(run)
    if cmp.empty:
        L.append("    (no strategies run)")
    else:
        L.append(_fmt_table(cmp))

    L.append("")
    L.append("  Model calibration — A/E by predicted-probability bucket:")
    L.append(_fmt_ae(run.ae_by_prob))
    L.append("")
    L.append("  A/E by odds band (execution price):")
    L.append(_fmt_ae(run.ae_by_odds))
    L.append(sep)
    return "\n".join(L)


def _fmt_table(df: pd.DataFrame) -> str:
    try:
        return "    " + df.to_string(index=False).replace("\n", "\n    ")
    except Exception:  # noqa: BLE001
        return "    " + str(df)


def _fmt_ae(rows: list[dict]) -> str:
    if not rows:
        return "    (none)"
    out = []
    for r in rows:
        flag = ""
        if r.get("ae") is not None:
            flag = "  <-- off" if abs(r["ae"] - 1.0) > 0.25 else ""
        out.append(f"    {r['bucket']:>16}  n={r['n']:>6}  pred={r.get('mean_pred')}  "
                   f"exp={r['expected']:>8}  act={r['actual']:>7}  A/E={r.get('ae')}{flag}")
    return "\n".join(out)


def save_run(run: BacktestRun, output_dir: Optional[str] = None,
             run_id: Optional[str] = None) -> str:
    """Persist the run; return the directory path.

    Writes ``summary.json`` (everything except the heavy frames) and one
    ``bets_<strategy>.parquet`` per strategy. A slim ``scored.parquet`` (ids +
    prob + prices + outcome) is written for offline calibration analysis.
    """
    base = output_dir or _DEFAULT_OUT
    rid = run_id or now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, rid)
    os.makedirs(out, exist_ok=True)

    summary = {
        "run_id": rid,
        "generated_at": now().isoformat(),
        "config": run.config,
        "folds": run.folds,
        "model_metrics": run.model_metrics,
        "ae_by_prob": run.ae_by_prob,
        "ae_by_odds": run.ae_by_odds,
        "strategies": {name: s["summary"] for name, s in run.strategies.items()},
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    for name, s in run.strategies.items():
        ledger = s["ledger"]
        if not ledger.empty:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            ledger.to_parquet(os.path.join(out, f"bets_{safe}.parquet"), index=False)

    slim_cols = [c for c in ("race_date", "venue", "horse_id", "horse_name", "fold",
                             "prob", "bet_price", "close_price", "won", "ev", "edge",
                             "kelly", "implied") if c in run.scored.columns]
    if slim_cols:
        run.scored[slim_cols].to_parquet(os.path.join(out, "scored.parquet"), index=False)

    logger.info("backtest: saved run to %s", out)
    return out
