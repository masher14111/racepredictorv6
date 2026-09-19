"""Multi-week backtest: settle the model's bets across the last N settled racing
days (default 21 = 3 weeks) and report whether it wins, using the OPTIMAL value
gates currently in config.yaml ([2,4] band + EV>=0.05, F-L recalibrated probs).

Honest by construction: bets settle at the **starting price (SP)**, which is a
conservative lower bound — the live layer takes the best board price (usually more
generous than SP), and the real profitability test is closing-line value (CLV),
which this SP-settled run does not measure. See memory/calib-fl-04/05.

Run:  python -m scripts.multi_week_backtest [--days 21]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models.yesterday import _load_matrix, _settled_days
from scripts.last_week_backtest import run_strategy, _agg, model_quality


def _print_block(label, per_day, bets, weeks):
    print(f"=== {label} ===")
    print(f"  {'date':>10} {'races':>6} {'vraces':>6} {'bets':>5} {'wins':>5} {'profit':>9} {'roi%':>7}")
    for d, nr, nv, nb, nw, pf, roi in per_day:
        print(f"  {d:>10} {nr:>6} {nv:>6} {nb:>5} {nw:>5} {pf:>9.2f} {roi:>7.2f}")
    # per-week rollup (chronological chunks of 7 settled days)
    print(f"  {'-'*54}")
    by_day = {row[0]: row for row in per_day}
    for wi, wk in enumerate(weeks, 1):
        rows = [by_day[d] for d in wk if d in by_day]
        nb = sum(r[3] for r in rows); nw = sum(r[4] for r in rows)
        pf = sum(r[5] for r in rows)
        staked = nb * 10.0  # flat EUR10/bet
        roi = pf / staked * 100 if staked else 0.0
        print(f"  week {wi} ({wk[0]}..{wk[-1]}): {nb:>4} bets, {nw:>3} wins, "
              f"profit EUR{pf:>9.2f}, ROI {roi:>7.2f}%")
    a = _agg(bets, 0)
    print(f"  {'-'*54}")
    print(f"  TOTAL: {a['n_bets']} bets, {a['wins']} wins ({a['win_rate']}%), "
          f"{a['places']} placed ({a['place_rate']}%), staked EUR{a['staked']}, "
          f"profit EUR{a['profit']}, ROI {a['roi']}%\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=21, help="number of settled days (default 21 = 3 weeks)")
    args = ap.parse_args()

    matrix = _load_matrix(None)
    all_days = _settled_days(matrix)
    days = all_days[-args.days:]
    # chronological chunks of 7 for the per-week rollup
    weeks = [days[i:i + 7] for i in range(0, len(days), 7)]
    print(f"Multi-week backtest over {days[0]} .. {days[-1]}  "
          f"({len(days)} settled days, {len(weeks)} weeks)")
    print("Value gates: the OPTIMAL config.yaml set (band [2,4], EV>=0.05, F-L recalibrated).")
    print("Settlement: at SP (conservative). CLV not measured here — see memory/calib-fl-05.\n")

    strategies = [
        ("VALUE picks, WIN only", dict(strategy="value", bet_type="win", stake=10)),
        ("VALUE picks, EACH-WAY", dict(strategy="value", bet_type="each_way", stake=10)),
        ("TOP win pick/race, WIN only", dict(strategy="top_pick", bet_type="win", stake=10, top_n=1)),
        ("TOP win pick/race, EACH-WAY", dict(strategy="top_pick", bet_type="each_way", stake=10, top_n=1)),
    ]

    for label, kw in strategies:
        per_day, bets = run_strategy(matrix, days, **kw)
        _print_block(label, per_day, bets, weeks)

    print("=== MODEL QUALITY (win prob vs actual, whole window, blind) ===")
    model_quality(matrix, days)


if __name__ == "__main__":
    main()
