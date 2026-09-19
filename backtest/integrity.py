"""Backtest integrity checks (ported from racing_ingestion, adapted to v4).

Post-hoc diagnostics that surface cherry-picking, leakage, and other structural
problems *after* a backtest run, so an ROI headline can no longer flatter itself.
These are not run-time guardrails (the walk-forward engine already enforces
chronological, leak-safe folds); they are transparent flags for a human reviewer
and for the holdout runner / UI to display.

Each check returns a result dict:

    {"name": str, "status": "OK" | "WARN" | "FAIL", "message": str, "detail": dict}

Severity:
    OK   — no obvious problem found.
    WARN — worth investigating; not necessarily fatal.
    FAIL — strong signal of a methodological problem.

A full OK does NOT prove the strategy works — it proves the most common failure
modes are not obviously present.

Input
-----
``ledger`` is a v4 bet ledger (see :func:`backtest.engine.Backtester._simulate`):
a row per settled bet with at least ``race_date, won, stake, profit, bet_price,
close_price``. Optional columns the checks use when present:
``venue``/``course_name`` (course concentration), ``odds_band`` (else derived
from ``bet_price``), ``clv_log`` (else derived from the prices), and a matched
volume column (``bf_matched_volume``/``matched_volume``). Everything is read-only;
where a column is absent the check degrades to OK/WARN rather than crashing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.overfitting import minimum_backtest_length, probabilistic_sharpe_ratio

# Candidate column names (v4 vs racing_ingestion naming).
_COURSE_COLS = ("course_name", "venue", "course")
_VOLUME_COLS = ("bf_matched_volume", "matched_volume")

# CLV-leakage thresholds on mean log closing-line value.
_CLV_WARN = 0.08
_CLV_FAIL = 0.15

# Concentration thresholds (share of total profit from the single best slice).
_COURSE_WARN_SHARE = 0.60
_SEASON_WARN_SHARE = 0.70
_BAND_WARN_SHARE = 0.80

# Below this many settled bets the sample is too small to trust at all.
_MIN_CREDIBLE_BETS = 200


# ── helpers ───────────────────────────────────────────────────────────────────

def _result(name: str, status: str, message: str, **detail) -> dict:
    return {"name": name, "status": status, "message": message, "detail": detail}


def _sev(cond_fail: bool, cond_warn: bool) -> str:
    if cond_fail:
        return "FAIL"
    if cond_warn:
        return "WARN"
    return "OK"


def _first_present(df: pd.DataFrame, names) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _odds_band_labels(prices: np.ndarray) -> np.ndarray:
    """Map decimal prices to v4's odds-band labels (``metrics._ODDS_BANDS``)."""
    labels = np.full(prices.shape, None, dtype=object)
    for label, lo, hi in metrics._ODDS_BANDS:
        m = np.isfinite(prices) & (prices >= lo) & (prices < hi)
        labels[m] = label
    return labels


# ── individual checks ─────────────────────────────────────────────────────────

def check_lookahead_bias(ledger: pd.DataFrame, train_cutoff=None) -> dict:
    """Structural look-ahead guard.

    FAIL on any NaN ``race_date``. If ``train_cutoff`` is given, the first bet
    must fall *strictly after* the model's training cutoff — a bet on/before the
    cutoff means the model could have trained on the very race it bet into.
    """
    name = "lookahead_bias"
    if ledger.empty:
        return _result(name, "OK", "Empty ledger — nothing to check.")

    dates = pd.to_datetime(ledger["race_date"], errors="coerce")
    n_nan = int(dates.isna().sum())
    if n_nan:
        return _result(name, "FAIL", f"FAIL: {n_nan} NaN race_date(s) in ledger.",
                       n_null_dates=n_nan)

    first_bet, last_bet = dates.min(), dates.max()
    detail = {"first_bet_date": str(first_bet.date()), "last_bet_date": str(last_bet.date())}

    if train_cutoff is not None:
        cutoff = pd.to_datetime(train_cutoff, errors="coerce")
        detail["train_cutoff"] = None if pd.isna(cutoff) else str(cutoff.date())
        if pd.isna(cutoff):
            return _result(name, "WARN",
                           "WARN: train_cutoff unparseable — could not verify look-ahead.",
                           **detail)
        if first_bet <= cutoff:
            return _result(
                name, "FAIL",
                f"FAIL: First bet ({first_bet.date()}) is on/before the train cutoff "
                f"({cutoff.date()}). The model may have trained on rows it bet into.",
                **detail)
        return _result(name, "OK",
                       f"OK: First bet ({first_bet.date()}) is after the train cutoff "
                       f"({cutoff.date()}).", **detail)

    return _result(name, "OK",
                   "OK: No NaN bet dates. (No train_cutoff supplied — strict "
                   "look-ahead guard skipped.)", **detail)


def check_course_cherrypicking(ledger: pd.DataFrame) -> dict:
    """Flag profit dominated by one course/venue.

    WARN: top course > 60% of total profit. FAIL: only one profitable course
    when more than one was bet.
    """
    name = "course_cherrypicking"
    col = _first_present(ledger, _COURSE_COLS)
    if col is None or ledger.empty:
        return _result(name, "OK", "Course column absent — course check skipped.")

    total_profit = float(ledger["profit"].sum())
    grp = (ledger.groupby(col)
           .agg(profit=("profit", "sum"), staked=("stake", "sum"), n_bets=("stake", "count"))
           .sort_values("profit", ascending=False))
    n_courses = len(grp)
    n_profitable = int((grp["profit"] > 0).sum())

    if total_profit <= 0:
        return _result(name, "OK", "Loss-making overall; course check not applicable.",
                       n_courses=n_courses, n_profitable=n_profitable)

    top_profit = float(grp["profit"].iloc[0])
    top_share = top_profit / total_profit

    sev = _sev(n_profitable == 1 and n_courses > 1, top_share > _COURSE_WARN_SHARE)
    if sev == "FAIL":
        msg = (f"FAIL: Only 1 profitable {col} of {n_courses}. "
               "Profit is course-specific — likely cherry-picking.")
    elif sev == "WARN":
        msg = (f"WARN: Top {col} ({grp.index[0]}) contributes {top_share*100:.1f}% of "
               f"total profit ({n_profitable} profitable). Possible course concentration.")
    else:
        msg = f"OK: Profit spread across {n_profitable} courses (top share {top_share*100:.1f}%)."

    return _result(name, sev, msg, course_col=col, n_courses=n_courses,
                   n_profitable_courses=n_profitable, top_course=str(grp.index[0]),
                   top_course_profit_share=top_share)


def check_season_cherrypicking(ledger: pd.DataFrame) -> dict:
    """Flag profit concentrated in one calendar quarter.

    WARN: best quarter > 70% of total profit. FAIL: only one profitable quarter
    when more than one was bet.
    """
    name = "season_cherrypicking"
    if ledger.empty:
        return _result(name, "OK", "Empty ledger.")

    df = ledger[["race_date", "profit", "stake"]].copy()
    df["quarter"] = pd.to_datetime(df["race_date"], errors="coerce").dt.to_period("Q").astype(str)
    grp = (df.groupby("quarter")
           .agg(profit=("profit", "sum"), staked=("stake", "sum"), n_bets=("stake", "count"))
           .sort_values("profit", ascending=False))
    n_q = len(grp)
    n_profitable = int((grp["profit"] > 0).sum())
    total_profit = float(df["profit"].sum())

    if total_profit <= 0:
        return _result(name, "OK", "Loss-making overall; season check not applicable.",
                       n_quarters=n_q, n_profitable=n_profitable)

    top_share = float(grp["profit"].iloc[0]) / total_profit

    sev = _sev(n_profitable <= 1 and n_q > 1, top_share > _SEASON_WARN_SHARE)
    if sev == "FAIL":
        msg = f"FAIL: Only {n_profitable} profitable quarter(s) of {n_q}. Seasonal-only edge."
    elif sev == "WARN":
        msg = (f"WARN: Best quarter contributes {top_share*100:.1f}% of total profit. "
               "Possible seasonal concentration.")
    else:
        msg = f"OK: Profit spread across {n_profitable} profitable quarters of {n_q}."

    return _result(name, sev, msg, n_quarters=n_q, n_profitable_quarters=n_profitable,
                   top_quarter_profit_share=top_share)


def check_band_cherrypicking(ledger: pd.DataFrame) -> dict:
    """Flag profit driven by a single odds band.

    Uses an ``odds_band`` column if present, else derives bands from ``bet_price``.
    WARN: best band > 80% of total profit. FAIL: only one profitable band when
    more than one was bet.
    """
    name = "band_cherrypicking"
    if ledger.empty:
        return _result(name, "OK", "Empty ledger.")

    if "odds_band" in ledger.columns:
        bands = ledger["odds_band"].astype(object).to_numpy()
    elif "bet_price" in ledger.columns:
        bands = _odds_band_labels(pd.to_numeric(ledger["bet_price"], errors="coerce").to_numpy())
    else:
        return _result(name, "OK", "No odds-band or bet_price column — band check skipped.")

    df = pd.DataFrame({"band": bands, "profit": ledger["profit"].to_numpy()})
    df = df[df["band"].notna()]
    if df.empty:
        return _result(name, "OK", "No banded bets.")

    grp = df.groupby("band")["profit"].sum().sort_values(ascending=False)
    n_bands = len(grp)
    n_profitable = int((grp > 0).sum())
    total_profit = float(grp.sum())

    if total_profit <= 0:
        return _result(name, "OK", "Loss-making overall; band check not applicable.",
                       n_bands=n_bands, n_profitable=n_profitable)

    best_band = str(grp.index[0])
    best_share = float(grp.iloc[0]) / total_profit

    sev = _sev(n_profitable == 1 and n_bands > 1, best_share > _BAND_WARN_SHARE)
    if sev == "FAIL":
        msg = (f"FAIL: Only band [{best_band}] is profitable of {n_bands}. "
               "Likely odds-band cherry-picking.")
    elif sev == "WARN":
        msg = (f"WARN: Band [{best_band}] drives {best_share*100:.1f}% of total profit. "
               "Strategy may be band-sensitive.")
    else:
        msg = f"OK: {n_profitable}/{n_bands} odds bands profitable."

    return _result(name, sev, msg, n_bands=n_bands, n_profitable_bands=n_profitable,
                   best_band=best_band, best_band_profit_share=best_share)


def check_clv_leakage(ledger: pd.DataFrame) -> dict:
    """Flag suspiciously high closing-line value — a fingerprint of using
    closing-price (future market) information.

    Uses a ``clv_log`` column if present, else log(bet_price / close_price).
    WARN: mean log CLV > 0.08. FAIL: mean log CLV > 0.15.
    """
    name = "clv_leakage"
    if ledger.empty:
        return _result(name, "OK", "Empty ledger.")

    if "clv_log" in ledger.columns:
        clv = pd.to_numeric(ledger["clv_log"], errors="coerce")
    elif {"bet_price", "close_price"}.issubset(ledger.columns):
        bp = metrics._valid_odds(ledger["bet_price"])
        cp = metrics._valid_odds(ledger["close_price"])
        with np.errstate(divide="ignore", invalid="ignore"):
            clv = pd.Series(np.log(bp / cp))
    else:
        return _result(name, "OK", "No CLV or price columns — leakage check skipped.")

    clv = clv.replace([np.inf, -np.inf], np.nan).dropna()
    if clv.empty:
        return _result(name, "OK", "No valid CLV values.")

    mean_clv = float(clv.mean())
    beat_rate = float((clv > 0).mean())

    sev = _sev(mean_clv > _CLV_FAIL, mean_clv > _CLV_WARN)
    if sev == "FAIL":
        msg = (f"FAIL: Mean log CLV = {mean_clv:.4f} (beat-close {beat_rate*100:.1f}%). "
               "Suspiciously high — likely closing-odds leakage.")
    elif sev == "WARN":
        msg = (f"WARN: Mean log CLV = {mean_clv:.4f} (beat-close {beat_rate*100:.1f}%). "
               "Above typical range — verify no closing-odds features are used.")
    else:
        msg = f"OK: Mean log CLV = {mean_clv:.4f} (beat-close rate {beat_rate*100:.1f}%)."

    return _result(name, sev, msg, mean_clv_log=mean_clv, clv_beat_rate=beat_rate,
                   n_bets=int(clv.size))


def check_non_runner_impact(ledger: pd.DataFrame, card_df: pd.DataFrame | None = None) -> dict:
    """Report exposure to non-runners. The simulator re-normalises probabilities
    after scratches; this quantifies the exposure for transparency only (never
    FAILs)."""
    name = "non_runner_impact"
    nr_col = _first_present(ledger, ("is_non_runner", "non_runner"))
    if nr_col is not None and not ledger.empty:
        n_nr = int(pd.to_numeric(ledger[nr_col], errors="coerce").fillna(0).astype(bool).sum())
        pct = n_nr / len(ledger)
        return _result(name, "OK",
                       f"OK: {n_nr}/{len(ledger)} bets ({pct*100:.1f}%) flagged near a non-runner.",
                       n_bets_with_nr=n_nr, n_total=len(ledger))

    if (card_df is not None and not card_df.empty
            and "is_non_runner" in card_df.columns and "race_id" in card_df.columns):
        nr_by_race = card_df.groupby("race_id")["is_non_runner"].any()
        n_with = int(nr_by_race.sum())
        n_total = len(nr_by_race)
        pct = n_with / n_total if n_total else 0.0
        return _result(name, "OK",
                       f"OK: {n_with}/{n_total} races ({pct*100:.1f}%) had ≥1 non-runner. "
                       "Probabilities re-normalised automatically.",
                       n_races_with_nr=n_with, n_total_races=n_total)

    return _result(name, "OK", "Non-runner data not available — check skipped.")


def check_liquidity(ledger: pd.DataFrame) -> dict:
    """Liquidity / market-impact check.

    v4 has no exchange matched-volume feed, so by default this is a transparency
    WARN ("liquidity unknown"). If a volume column is present, WARN on any bet
    staking > 5% of matched volume (real execution would move the market).
    """
    name = "liquidity"
    col = _first_present(ledger, _VOLUME_COLS)
    if col is None:
        return _result(name, "WARN",
                       "WARN: Matched-volume data absent — liquidity unknown. Real "
                       "execution costs (especially on outsiders) cannot be verified.")

    df = ledger[ledger[col].notna()]
    if df.empty:
        return _result(name, "WARN", "WARN: No matched-volume values — liquidity unknown.")

    stake = pd.to_numeric(df["stake"], errors="coerce").to_numpy()
    vol = pd.to_numeric(df[col], errors="coerce").to_numpy()
    impact = np.isfinite(vol) & (vol > 0) & (stake > 0.05 * vol)
    n_impact = int(impact.sum())
    if n_impact:
        return _result(name, "WARN",
                       f"WARN: {n_impact} bet(s) exceed 5% of matched volume. Real "
                       "execution would move the market and erode the achieved price.",
                       n_market_impact_bets=n_impact, n_total=int(len(df)))
    return _result(name, "OK", "OK: All bets within 5% of matched volume.",
                   n_total=int(len(df)))


def check_minimum_backtest_length(
    ledger: pd.DataFrame, n_trials: int = 1, min_bets: int = _MIN_CREDIBLE_BETS,
) -> dict:
    """Credibility of the backtest length against multiple-testing (Bailey-Prado).

    FAIL if the sample is below ``min_bets`` (too few bets to trust at all).
    Otherwise, with an annualised Sharpe estimated from the ledger and the number
    of optimisation trials ``n_trials``, compare the observed length against the
    Minimum Backtest Length; WARN if shorter (likely overfit). A non-positive
    Sharpe is OK here — there is no positive edge to defend.
    """
    name = "minimum_backtest_length"
    n = len(ledger)
    if n == 0:
        return _result(name, "OK", "Empty ledger.")
    if n < min_bets:
        return _result(name, "FAIL",
                       f"FAIL: Only {n} settled bets (< {min_bets}). Insufficient sample "
                       "for credible statistics.", n_bets=n)

    stake = metrics._as_float(ledger["stake"])
    profit = metrics._as_float(ledger["profit"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(stake > 0, profit / stake, np.nan)
    ret = ret[np.isfinite(ret)]
    if ret.size < 2:
        return _result(name, "OK", "Too few valued bets to estimate a Sharpe.", n_bets=n)

    sd = float(ret.std(ddof=1))
    sharpe_per_bet = float(ret.mean() / sd) if sd > 1e-12 else 0.0

    dates = pd.to_datetime(ledger["race_date"], errors="coerce").dropna()
    span_days = max((dates.max() - dates.min()).days, 1) if len(dates) else 1
    bets_per_year = n / (span_days / 365.25)
    annual_sharpe = sharpe_per_bet * np.sqrt(bets_per_year)

    psr = probabilistic_sharpe_ratio(sharpe_per_bet, n)  # P(true SR > 0)

    if annual_sharpe <= 0:
        return _result(name, "OK",
                       f"OK: {n} bets; non-positive Sharpe ({annual_sharpe:.2f}) — no "
                       "positive edge to defend against overfitting.",
                       n_bets=n, annual_sharpe=annual_sharpe, psr=psr)

    btl = minimum_backtest_length(
        n_trials=n_trials, target_annual_sharpe=annual_sharpe,
        races_per_year=bets_per_year, observed_backtest_units=n)
    status = "OK" if btl.observed_is_credible else "WARN"
    msg = (f"{status}: {n} bets, annual Sharpe ≈ {annual_sharpe:.2f}, "
           f"P(SR>0) = {psr*100:.1f}%. {btl.message}")
    return _result(name, status, msg, n_bets=n, annual_sharpe=annual_sharpe, psr=psr,
                   n_trials=n_trials, min_btl_race_units=btl.min_btl_race_units,
                   observed_is_credible=btl.observed_is_credible)


# ── aggregate runner ──────────────────────────────────────────────────────────

def run_all_integrity_checks(
    ledger: pd.DataFrame,
    *,
    train_cutoff=None,
    card_df: pd.DataFrame | None = None,
    n_trials: int = 1,
) -> list[dict]:
    """Run every integrity check and return a list of result dicts.

    Args:
        ledger:       v4 bet ledger (read-only).
        train_cutoff: the model's training cutoff date; enables the strict
                      look-ahead guard (first bet must be strictly after it).
        card_df:      optional scored-card frame for the non-runner check.
        n_trials:     number of optimisation configurations tried during model
                      selection — feeds the Minimum-Backtest-Length credibility.
    """
    return [
        check_lookahead_bias(ledger, train_cutoff),
        check_course_cherrypicking(ledger),
        check_season_cherrypicking(ledger),
        check_band_cherrypicking(ledger),
        check_clv_leakage(ledger),
        check_non_runner_impact(ledger, card_df),
        check_liquidity(ledger),
        check_minimum_backtest_length(ledger, n_trials=n_trials),
    ]
