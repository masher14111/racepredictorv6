"""One-off: settle the model's bets across the last 7 settled racing days, and
report whether the model is statistically well-trained (AUC / Brier / ECE /
calibration by probability band) over the same window.

Run:  python -m scripts.last_week_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.yesterday import (
    YesterdayConfig, run_yesterday, _load_matrix, _settled_days, _score_day, _price
)


def _agg(bets: list[dict], stake_total: float) -> dict:
    n = len(bets)
    if n == 0:
        return dict(n_bets=0, wins=0, places=0, win_rate=0.0, place_rate=0.0,
                    staked=0.0, profit=0.0, roi=0.0)
    wins = sum(1 for b in bets if b["outcome"] == "win")
    places = sum(1 for b in bets if b["outcome"] in ("win", "place"))
    staked = sum(b["stake"] for b in bets)
    profit = sum(b["profit"] for b in bets)
    return dict(n_bets=n, wins=wins, places=places,
                win_rate=round(wins / n * 100, 1),
                place_rate=round(places / n * 100, 1),
                staked=round(staked, 2), profit=round(profit, 2),
                roi=round(profit / staked * 100, 2) if staked else 0.0)


def run_strategy(matrix, days, **cfg_kw):
    """Run one strategy across every day; return (per-day rows, all bets)."""
    all_bets, per_day = [], []
    for d in days:
        cfg = YesterdayConfig(target_date=d, **cfg_kw)
        res = run_yesterday(cfg, df=matrix)
        s = res.summary
        per_day.append((d, res.n_races, res.n_value_races, s["n_bets"],
                        s["n_winners"], s["total_profit"], s["roi_pct"]))
        all_bets.extend(res.bets)
    return per_day, all_bets


def _quality_metrics(p, y):
    """AUC / logloss / Brier / ECE + 10-bin reliability rows for (p, y)."""
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = len(p)
    if n == 0:
        return None
    order = np.argsort(p)
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    n_pos, n_neg = y.sum(), n - y.sum()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else float("nan")
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-12
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = 0.0
    rows = []
    for b in range(10):
        m = idx == b
        if not m.any():
            continue
        conf, acc, cnt = p[m].mean(), y[m].mean(), m.sum()
        ece += cnt / n * abs(conf - acc)
        rows.append((f"{bins[b]:.1f}-{bins[b+1]:.1f}", cnt, round(conf, 3),
                     round(acc, 3), round(acc / conf, 2) if conf else float("nan")))
    return dict(n=n, n_pos=int(n_pos), mean=float(p.mean()), auc=auc,
                logloss=logloss, brier=brier, ece=ece, rows=rows)


def model_quality(matrix, days):
    """Score every runner across the week and measure win-prob calibration.

    Reports the HEADLINE win prob the UI/API present (``won_prob_normalized``,
    field-coherent) against the raw per-runner calibrated marginal (``won_prob``,
    a debug column that saturates OOS). See calib-fl-01.
    """
    dts = pd.to_datetime(matrix["race_date"], utc=True, errors="coerce").dt.date.astype(str)
    week = matrix[dts.isin(days) & matrix["position"].notna()].copy()
    week["_price"] = _price(week).to_numpy()
    scored = _score_day(week)
    if scored is None or "won_prob" not in scored.columns:
        print("  (could not score week for quality metrics)")
        return
    y = (pd.to_numeric(scored["position"], errors="coerce") == 1).astype(int).to_numpy()

    cols = [
        ("won_prob_normalized", "HEADLINE (won_prob_normalized)"),
        ("won_prob", "raw marginal (won_prob, debug)"),
    ]
    base = next((c for c, _ in cols if c in scored.columns), None)
    nb = _quality_metrics(pd.to_numeric(scored[base], errors="coerce").to_numpy(), y) if base else None
    if nb:
        print(f"\n  runners scored: {nb['n']:,}   winners: {nb['n_pos']:,}   "
              f"base rate: {nb['n_pos']/nb['n']:.3f}")

    print(f"\n  {'column':>32} {'mean':>6} {'AUC':>6} {'logloss':>8} {'Brier':>6} {'ECE':>6}")
    headline_rows = None
    for col, label in cols:
        if col not in scored.columns:
            continue
        m = _quality_metrics(pd.to_numeric(scored[col], errors="coerce").to_numpy(), y)
        if m is None:
            continue
        print(f"  {label:>32} {m['mean']:>6.3f} {m['auc']:>6.3f} "
              f"{m['logloss']:>8.3f} {m['brier']:>6.3f} {m['ece']:>6.3f}")
        if col == "won_prob_normalized":
            headline_rows = m["rows"]

    rows = headline_rows or (nb["rows"] if nb else [])
    print(f"\n  headline reliability ({'won_prob_normalized' if headline_rows else base}):")
    print(f"  {'prob band':>10} {'n':>6} {'pred':>7} {'actual':>7} {'A/E':>6}")
    for band, cnt, conf, acc, ae in rows:
        print(f"  {band:>10} {cnt:>6} {conf:>7} {acc:>7} {ae:>6}")


def main():
    matrix = _load_matrix(None)
    all_days = _settled_days(matrix)
    days = all_days[-7:]
    print(f"Last-week backtest over {days[0]} .. {days[-1]}  ({len(days)} settled days)\n")

    strategies = [
        ("VALUE picks, EACH-WAY", dict(strategy="value", bet_type="each_way", stake=10)),
        ("VALUE picks, WIN only", dict(strategy="value", bet_type="win", stake=10)),
        ("TOP win pick/race, EACH-WAY", dict(strategy="top_pick", bet_type="each_way", stake=10, top_n=1)),
        ("TOP win pick/race, WIN only", dict(strategy="top_pick", bet_type="win", stake=10, top_n=1)),
    ]

    for label, kw in strategies:
        per_day, bets = run_strategy(matrix, days, **kw)
        print(f"=== {label} ===")
        print(f"  {'date':>10} {'races':>6} {'vraces':>6} {'bets':>5} {'wins':>5} {'profit':>8} {'roi%':>7}")
        for d, nr, nv, nb, nw, pf, roi in per_day:
            print(f"  {d:>10} {nr:>6} {nv:>6} {nb:>5} {nw:>5} {pf:>8.2f} {roi:>7.2f}")
        a = _agg(bets, 0)
        print(f"  TOTAL: {a['n_bets']} bets, {a['wins']} wins ({a['win_rate']}%), "
              f"{a['places']} placed ({a['place_rate']}%), staked EUR{a['staked']}, "
              f"profit EUR{a['profit']}, ROI {a['roi']}%\n")

    print("=== MODEL QUALITY (win prob vs actual, same week, blind) ===")
    model_quality(matrix, days)


if __name__ == "__main__":
    main()
