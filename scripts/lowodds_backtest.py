"""Low-odds value backtest: focus the model on short-priced favourites over a
longer window, and sweep several tight low-odds bands to see which (if any) wins.

Keeps every OTHER value gate from config.yaml (EV>=0.05, support, de-vig,
confidence, F-L recalibration) and only overrides the odds band, so this is the
shipped "optimal" selection narrowed to low odds. Bets settle at the **starting
price (SP)** — a conservative lower bound; CLV is not measured (see calib-fl-05).

Each day is scored once and reused across bands/markets (fast).

Run:  python -m scripts.lowodds_backtest [--days 90]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from models.yesterday import (
    _load_matrix, _settled_days, _score_day, _price, _race_key, _settle_bet,
    YesterdayConfig,
)
from models.value import ValueConfig, find_value_bets

# (label, min_odds, max_odds) — low-odds focus, tightest first
BANDS = [
    ("odds-on  [1.01, 2.0)", 1.01, 2.0),
    ("short    [1.50, 2.5]", 1.50, 2.5),
    ("favs     [2.00, 3.0]", 2.00, 3.0),
    ("optimal  [2.00, 4.0]", 2.00, 4.0),   # current config baseline
]


def score_days(matrix: pd.DataFrame, days: list[str]) -> dict[str, pd.DataFrame]:
    """Score every settled day once; cache the scored frame keyed by date."""
    out: dict[str, pd.DataFrame] = {}
    dser = pd.to_datetime(matrix["race_date"], utc=True, errors="coerce")
    for tgt in days:
        day_df = matrix[(dser.dt.date.astype(str) == tgt) & matrix["position"].notna()].copy()
        if day_df.empty:
            continue
        day_df["_price"] = _price(day_df).to_numpy()
        scored = _score_day(day_df)
        if scored is None:
            continue
        scored["_price"] = day_df["_price"].to_numpy()
        scored["_rkey"] = _race_key(scored).to_numpy()
        out[tgt] = scored
    return out


def run_band(scored_days, base_vcfg, min_odds, max_odds, bet_type,
             stake=10.0, bankroll=1000.0) -> list[dict]:
    vcfg = replace(base_vcfg, min_odds=min_odds, max_odds=max_odds)
    ycfg = YesterdayConfig(strategy="value", bet_type=bet_type, stake=stake, bankroll=bankroll)
    bets: list[dict] = []
    for scored in scored_days.values():
        for _, grp in scored.groupby("_rkey", sort=True):
            grp = grp.reset_index(drop=True)
            frame = pd.DataFrame({
                "horse_id": grp["horse_id"].astype(str),
                "horse_name": grp["horse_name"].astype(str),
                "value_win_prob": pd.to_numeric(grp.get("value_win_prob"), errors="coerce"),
                "decimal_odds": grp["_price"],
                "value_supported": grp.get("value_supported", pd.Series(True, index=grp.index)).astype(bool),
            })
            if frame["value_win_prob"].isna().all():
                continue
            picks = find_value_bets(frame, config=vcfg, bankroll=bankroll)
            by_id = {str(r["horse_id"]): r for _, r in grp.iterrows()}
            for pick in picks:
                row = by_id.get(str(pick.get("horse_id")))
                if row is None or pd.isna(row.get("position")) or pd.isna(pick.get("decimal_odds")):
                    continue
                bets.append(_settle_bet(
                    row, float(pick["decimal_odds"]), ycfg, is_value=True,
                    model_prob=pick.get("model_prob"), edge=pick.get("edge"),
                    ev=pick.get("expected_value")))
    return bets


def agg(bets: list[dict]) -> dict:
    n = len(bets)
    if n == 0:
        return dict(n=0, wins=0, strike=0.0, avg_odds=0.0, staked=0.0, profit=0.0, roi=0.0)
    wins = sum(1 for b in bets if b["outcome"] == "win")
    staked = sum(b["stake"] for b in bets)
    profit = sum(b["profit"] for b in bets)
    avg_odds = float(np.mean([b["decimal_odds"] for b in bets]))
    return dict(n=n, wins=wins, strike=round(wins / n * 100, 1), avg_odds=round(avg_odds, 2),
                staked=round(staked, 2), profit=round(profit, 2),
                roi=round(profit / staked * 100, 2) if staked else 0.0)


def value_ae_by_band(scored_days) -> None:
    """A/E for the recalibrated value_win_prob, banded by SP, over the window."""
    rows = []
    for scored in scored_days.values():
        p = pd.to_numeric(scored.get("value_win_prob"), errors="coerce").to_numpy()
        d = pd.to_numeric(scored["_price"], errors="coerce").to_numpy()
        y = (pd.to_numeric(scored["position"], errors="coerce").to_numpy() == 1).astype(float)
        ok = np.isfinite(p) & np.isfinite(d) & np.isfinite(y)
        rows.append(np.c_[p[ok], d[ok], y[ok]])
    if not rows:
        return
    a = np.vstack(rows)
    p, d, y = a[:, 0], a[:, 1], a[:, 2]
    edges = [1.0, 2.0, 3.0, 4.0, 6.0, 10.0, np.inf]
    print(f"\n  value_win_prob A/E by SP band (recalibrated), {len(p):,} runners:")
    print(f"  {'band':>12} {'n':>6} {'pred':>7} {'actual':>7} {'A/E':>6}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 20:
            continue
        exp, act = p[m].sum(), y[m].sum()
        ae = act / exp if exp else float("nan")
        label = f"[{lo:g},{hi:g})" if np.isfinite(hi) else f">{lo:g}"
        print(f"  {label:>12} {int(m.sum()):>6} {p[m].mean():>7.3f} {y[m].mean():>7.3f} {ae:>6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=90, help="settled days to backtest (default 90)")
    args = ap.parse_args()

    matrix = _load_matrix(None)
    days = _settled_days(matrix)[-args.days:]
    print(f"Low-odds value backtest over {days[0]} .. {days[-1]}  ({len(days)} settled days)")
    print("Gates: config.yaml value set (EV>=0.05, support, de-vig, confidence, F-L recal);")
    print("only the odds band is varied. Flat EUR10/bet, settled at SP (conservative).\n")
    print("Scoring each day once...")
    scored_days = score_days(matrix, days)
    base_vcfg = ValueConfig.from_config()

    for bet_type in ("win", "each_way"):
        print(f"\n=== VALUE picks, {bet_type.upper().replace('_','-')} ===")
        print(f"  {'band':>22} {'bets':>5} {'wins':>5} {'strike':>7} {'avgOdds':>8} "
              f"{'staked':>8} {'profit':>9} {'roi%':>8}")
        for label, lo, hi in BANDS:
            a = agg(run_band(scored_days, base_vcfg, lo, hi, bet_type))
            print(f"  {label:>22} {a['n']:>5} {a['wins']:>5} {a['strike']:>6.1f}% "
                  f"{a['avg_odds']:>8} {a['staked']:>8.0f} {a['profit']:>9.2f} {a['roi']:>8.2f}")

    value_ae_by_band(scored_days)


if __name__ == "__main__":
    main()
