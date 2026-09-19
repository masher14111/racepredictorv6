"""Head-to-head metrics with race-level bootstrap intervals (requirement 8).

Everything scores a frame carrying one row per runner with a probability
column, ``won``, ``race_uid`` and the pre-off ``bet_price``. The market side is
the de-vigged pre-off line (proportional, matching the project's GO gate); both
sides are evaluated over the SAME complete-book races.

Uncertainty: race-level bootstrap (races resampled with replacement) for the
decision metrics — race log-loss (and its model-vs-market delta), runner Brier,
A/E — plus a normal-approx interval for CLV. ECE is reported as a point
estimate (its equal-frequency bins are not resample-stable).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from models.devig import devig
from models.head_to_head import ODDS_BANDS
from utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-7
_B_DEFAULT = 1000
_SEED = 42


# ── primitives ───────────────────────────────────────────────────────────────


def market_line(df: pd.DataFrame, *, odds_col: str = "bet_price",
                race_col: str = "race_uid") -> np.ndarray:
    """De-vigged (proportional) market win probability per runner; NaN on
    partial books."""
    odds = pd.to_numeric(df[odds_col], errors="coerce")
    return np.asarray(devig(odds, df[race_col], method="proportional"),
                      dtype=float)


def per_race_logloss(df: pd.DataFrame, prob_col: str, *,
                     race_col: str = "race_uid",
                     won_col: str = "won") -> pd.Series:
    """−log(p_winner) per race (index: race id). Races w/o a winner are dropped."""
    p = np.clip(pd.to_numeric(df[prob_col], errors="coerce").to_numpy(float),
                _EPS, 1.0)
    won = pd.to_numeric(df[won_col], errors="coerce").fillna(0).astype(int).to_numpy()
    rid = df[race_col].to_numpy()
    sub = pd.DataFrame({"rid": rid, "p": p, "won": won})
    winners = sub[sub["won"] == 1].groupby("rid")["p"].apply(
        lambda s: -float(np.log(s).sum()))
    return winners


def per_race_brier(df: pd.DataFrame, prob_col: str, *,
                   race_col: str = "race_uid",
                   won_col: str = "won") -> pd.DataFrame:
    """Per-race (sum squared error, n runners) — bootstrap-ready Brier parts."""
    p = pd.to_numeric(df[prob_col], errors="coerce").to_numpy(float)
    won = pd.to_numeric(df[won_col], errors="coerce").fillna(0).to_numpy(float)
    sq = (p - won) ** 2
    return pd.DataFrame({"rid": df[race_col].to_numpy(), "sq": sq}) \
        .groupby("rid")["sq"].agg(["sum", "count"])


def ece_equal_freq(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size < n_bins * 2 or np.unique(p).size < 2:
        return float("nan")
    order = np.argsort(p)
    edges = np.array_split(order, n_bins)
    tot = 0.0
    for idx in edges:
        if idx.size == 0:
            continue
        tot += idx.size / p.size * abs(y[idx].mean() - p[idx].mean())
    return float(tot)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── head-to-head with bootstrap ──────────────────────────────────────────────


def head_to_head_ci(df: pd.DataFrame, prob_col: str, *,
                    race_col: str = "race_uid", won_col: str = "won",
                    odds_col: str = "bet_price", n_boot: int = _B_DEFAULT,
                    seed: int = _SEED) -> dict:
    """Model vs de-vigged market over identical complete-book races, with
    race-bootstrap CIs on log-loss, the log-loss delta, and Brier.

    The model side is normalised within race before scoring (a probability
    vector, matching the market's construction). Returns a JSON-safe dict.
    """
    from models.calibration import normalize_within_race

    d = df.copy()
    d["_mkt"] = market_line(d, odds_col=odds_col, race_col=race_col)
    d = d[np.isfinite(pd.to_numeric(d[prob_col], errors="coerce"))
          & np.isfinite(d["_mkt"])]
    if d.empty:
        return {"n_races": 0}
    d["_model"] = normalize_within_race(
        pd.to_numeric(d[prob_col], errors="coerce").to_numpy(float),
        d[race_col].to_numpy())

    ll_model = per_race_logloss(d, "_model", race_col=race_col, won_col=won_col)
    ll_mkt = per_race_logloss(d, "_mkt", race_col=race_col, won_col=won_col)
    common = ll_model.index.intersection(ll_mkt.index)
    ll_model, ll_mkt = ll_model.loc[common], ll_mkt.loc[common]

    br_model = per_race_brier(d, "_model", race_col=race_col, won_col=won_col)
    br_mkt = per_race_brier(d, "_mkt", race_col=race_col, won_col=won_col)
    br_model, br_mkt = br_model.loc[common], br_mkt.loc[common]

    n_races = len(common)
    rng = np.random.default_rng(seed)
    deltas, lls, briers = [], [], []
    a_model, a_mkt = ll_model.to_numpy(), ll_mkt.to_numpy()
    bs_m, bn_m = br_model["sum"].to_numpy(), br_model["count"].to_numpy()
    bs_k = br_mkt["sum"].to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n_races, n_races)
        lls.append(a_model[idx].mean())
        deltas.append(a_mkt[idx].mean() - a_model[idx].mean())
        briers.append(bs_m[idx].sum() / bn_m[idx].sum()
                      - bs_k[idx].sum() / bn_m[idx].sum())
    q = lambda arr, lo, hi: (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))  # noqa: E731

    y = pd.to_numeric(d[won_col], errors="coerce").fillna(0).to_numpy(float)
    out = {
        "n_races": int(n_races),
        "n_runners": int(len(d)),
        "model_log_loss": float(a_model.mean()),
        "model_log_loss_ci95": q(lls, 2.5, 97.5),
        "market_log_loss": float(a_mkt.mean()),
        "logloss_delta_market_minus_model": float(a_mkt.mean() - a_model.mean()),
        "logloss_delta_ci95": q(deltas, 2.5, 97.5),
        "delta_positive_is_model_better": True,
        "model_beats_market_logloss": bool(a_model.mean() < a_mkt.mean()),
        "logloss_delta_frac_boot_positive": float(np.mean(np.asarray(deltas) > 0)),
        "model_brier": float(bs_m.sum() / bn_m.sum()),
        "market_brier": float(bs_k.sum() / bn_m.sum()),
        "brier_delta_market_minus_model_ci95": q(
            [-b for b in briers] if False else briers, 2.5, 97.5),
        "model_ece": ece_equal_freq(d["_model"].to_numpy(float), y),
        "market_ece": ece_equal_freq(d["_mkt"].to_numpy(float), y),
    }
    # brier delta sign fix: briers holds model - market; positive delta
    # (market - model) is model-better, so negate.
    out["brier_delta_market_minus_model"] = float(
        out["market_brier"] - out["model_brier"])
    out["brier_delta_ci95"] = tuple(sorted((-out["brier_delta_market_minus_model_ci95"][1],
                                            -out["brier_delta_market_minus_model_ci95"][0])))
    del out["brier_delta_market_minus_model_ci95"]
    return out


# ── A/E + subgroup calibration tables ────────────────────────────────────────


def ae_table(df: pd.DataFrame, prob_col: str, group: pd.Series, *,
             won_col: str = "won", min_rows: int = 50) -> pd.DataFrame:
    """Actual/Expected per group with a bootstrap-free binomial CI.

    A/E CI: treat expected as fixed; CI on actual wins via Wilson on the win
    rate, scaled by n/expected — a standard approximation at these n.
    """
    p = pd.to_numeric(df[prob_col], errors="coerce")
    y = pd.to_numeric(df[won_col], errors="coerce").fillna(0).astype(int)
    frame = pd.DataFrame({"g": group.to_numpy(), "p": p.to_numpy(),
                          "y": y.to_numpy()})
    frame = frame[np.isfinite(frame["p"])]
    rows = []
    for g, sub in frame.groupby("g", dropna=False):
        n = len(sub)
        if n < min_rows:
            continue
        exp = float(sub["p"].sum())
        act = int(sub["y"].sum())
        lo, hi = wilson_interval(act, n)
        rows.append({
            "group": str(g), "n": n, "actual": act, "expected": round(exp, 1),
            "ae": round(act / exp, 3) if exp > 0 else None,
            "ae_ci95": (round(lo * n / exp, 3), round(hi * n / exp, 3))
            if exp > 0 else None,
            "mean_prob": round(float(sub["p"].mean()), 4),
            "win_rate": round(act / n, 4),
        })
    return pd.DataFrame(rows)


def odds_band_series(odds: pd.Series) -> pd.Series:
    """Label each runner with the project's canonical odds band."""
    o = pd.to_numeric(odds, errors="coerce")
    lab = pd.Series("unknown", index=odds.index, dtype=object)
    for name, lo, hi in ODDS_BANDS:
        lab[(o >= lo) & (o < hi)] = name
    return lab


def field_size_series(fs: pd.Series) -> pd.Series:
    f = pd.to_numeric(fs, errors="coerce")
    bins = [(2, 5, "2-5"), (6, 8, "6-8"), (9, 11, "9-11"),
            (12, 15, "12-15"), (16, 40, "16+")]
    lab = pd.Series("unknown", index=fs.index, dtype=object)
    for lo, hi, name in bins:
        lab[(f >= lo) & (f <= hi)] = name
    return lab


def month_series(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, utc=True, errors="coerce").dt.strftime("%Y-%m")


def completeness_series(dc: Optional[pd.Series], index) -> pd.Series:
    if dc is None:
        return pd.Series("unknown", index=index, dtype=object)
    v = pd.to_numeric(dc, errors="coerce")
    lab = pd.Series("unknown", index=index, dtype=object)
    lab[v < 0.5] = "<0.5"
    lab[(v >= 0.5) & (v < 0.8)] = "0.5-0.8"
    lab[v >= 0.8] = ">=0.8"
    return lab


def clv_stats(bet_price: pd.Series, close_price: pd.Series) -> dict:
    """Mean log CLV + normal-approx 95% CI + beat rate, over priced rows."""
    bp = pd.to_numeric(bet_price, errors="coerce")
    cp = pd.to_numeric(close_price, errors="coerce")
    m = (bp > 1.0) & (cp > 1.0)
    if not m.any():
        return {"n": 0}
    clv = np.log(bp[m].to_numpy(float) / cp[m].to_numpy(float))
    n = clv.size
    mean = float(clv.mean())
    se = float(clv.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    k = int((clv > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"n": int(n), "mean_clv_log": mean,
            "mean_clv_ci95": (mean - 1.96 * se, mean + 1.96 * se),
            "beat_close_rate": k / n, "beat_close_ci95": (lo, hi)}
