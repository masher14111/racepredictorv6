"""Requirement-2 leakage proofs for the price-free (v3nf) feature set.

Four independent guards, from structural to empirical:

1. **Provenance (structural).** The v3nf whitelist must be disjoint from every
   price/market column, every post-off column, and every outcome/label column.
   This is the airtight by-construction check.

2. **Closing-move partial correlation (empirical).** After partialling out a
   rich pre-off price basis, no feature may retain information about the
   closing-line move ``log(odds_finish) - log(ppwap)`` — a quantity knowable
   only after the off. Same technique as ``tools/audit_lgbm_features.py``,
   applied to the CatBoost price-free columns.

3. **Outcome correlation (empirical).** No feature may be a near-copy of the
   result (|corr with won| above threshold).

4. **Append-future invariance.** Point-in-time features must not change for
   historical rows when strictly-later data is appended to the dataset. We
   compare feature values for identical (race_uid, horse_id) WIN rows between a
   matrix built BEFORE a data refresh and one built AFTER: any drift on
   pre-refresh race dates means some feature reads the future.

One deliberate channel is *documented, not failed*: ``models.train`` weights
training rows by inverse ``implied_prob`` (= 1/morningwap, a genuine PRE-OFF
price). Sample weights shape the fit but are not model inputs — at inference
the v3nf model needs no price — so this is market-informed training, not price
leakage into the probability function. ``check_weight_channel`` pins the
provenance of that price so it can never silently become a post-off one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.features import (
    FEATURE_COLS,
    MARKET_DERIVED_FEATURE_COLS,
    PRICE_FEATURE_COLS,
    PRICE_FREE_FEATURE_COLS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

MOVE_PCORR_THRESH = 0.30
WON_CORR_THRESH = 0.70
_MIN_ROWS = 200

# Columns that are prices, post-off quantities, outcomes, or identifiers of the
# outcome. None may appear in the price-free whitelist.
PRICE_OR_POSTOFF_COLS = {
    "odds_decimal", "sp", "morningwap", "ppwap", "odds_finish",
    "implied_prob", "overround_norm_prob", "log_odds", "market_rank",
    "odds_drift", "odds_value_delta", "ew_value_index", "price_steam_pct",
    "relative_market_share", "market_book_pct", "is_steaming", "is_drifting",
    "market_confidence", "form_market_disagreement",
}
OUTCOME_COLS = {"won", "placed", "position", "win_lose", "showed", "placed_2"}


@dataclass
class LeakageResult:
    passed: bool
    failures: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ── 1. structural provenance ─────────────────────────────────────────────────


def check_provenance() -> LeakageResult:
    """The price-free whitelist is disjoint from prices, post-off and outcomes."""
    failures = []
    bad_price = sorted(set(PRICE_FREE_FEATURE_COLS) & PRICE_OR_POSTOFF_COLS)
    if bad_price:
        failures.append(f"price/post-off column(s) in PRICE_FREE list: {bad_price}")
    bad_outcome = sorted(set(PRICE_FREE_FEATURE_COLS) & OUTCOME_COLS)
    if bad_outcome:
        failures.append(f"outcome column(s) in PRICE_FREE list: {bad_outcome}")
    # The strip really removed every price feature from the priced list.
    residue = sorted(set(PRICE_FREE_FEATURE_COLS) & set(PRICE_FEATURE_COLS))
    if residue:
        failures.append(f"PRICE_FEATURE_COLS not stripped: {residue}")
    # ...and every market-DERIVED one too (step 09 audit F3): a feature computed
    # from the price book is not price-free just because it is not a price.
    derived = sorted(set(PRICE_FREE_FEATURE_COLS) & set(MARKET_DERIVED_FEATURE_COLS))
    if derived:
        failures.append(f"MARKET_DERIVED_FEATURE_COLS not stripped: {derived}")
    return LeakageResult(
        passed=not failures, failures=failures,
        details={
            "n_price_free": len(PRICE_FREE_FEATURE_COLS),
            "n_full": len(FEATURE_COLS),
            "price_free_cols": list(PRICE_FREE_FEATURE_COLS),
        })


# ── 2/3. empirical guards ────────────────────────────────────────────────────


def _residual(y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    return y - basis @ coef


def closing_move_pcorr(feature: pd.Series, df: pd.DataFrame):
    """Partial corr of ``feature`` with log(odds_finish/ppwap), controlling for
    a pre-off price basis. Returns (pcorr or None, n)."""
    mw = pd.to_numeric(df.get("morningwap"), errors="coerce").where(lambda s: s > 1.0)
    pp = pd.to_numeric(df.get("ppwap"), errors="coerce").where(lambda s: s > 1.0)
    fin = pd.to_numeric(df.get("odds_finish"), errors="coerce").where(lambda s: s > 1.0)
    f = pd.to_numeric(feature, errors="coerce")

    mask = mw.notna() & pp.notna() & fin.notna() & np.isfinite(f.to_numpy(dtype=float))
    n = int(mask.sum())
    if n < _MIN_ROWS:
        return None, n
    mwv, ppv, finv = (s[mask].to_numpy(float) for s in (mw, pp, fin))
    fv = f[mask].to_numpy(float)
    move = np.log(finv) - np.log(ppv)
    basis = np.column_stack([np.ones(n), 1.0 / mwv, np.log(mwv), 1.0 / ppv, np.log(ppv)])
    rf, rm = _residual(fv, basis), _residual(move, basis)
    if np.std(rf) < 1e-9 or np.std(rm) < 1e-9:
        return 0.0, n
    return float(np.corrcoef(rf, rm)[0, 1]), n


def outcome_corr(feature: pd.Series, won: pd.Series):
    f = pd.to_numeric(feature, errors="coerce")
    w = pd.to_numeric(won, errors="coerce")
    mask = np.isfinite(f.to_numpy(dtype=float)) & w.notna().to_numpy()
    n = int(mask.sum())
    if n < _MIN_ROWS:
        return None, n
    fv, wv = f[mask].to_numpy(float), w[mask].to_numpy(float)
    if np.std(fv) < 1e-12 or np.std(wv) < 1e-12:
        return 0.0, n
    return float(np.corrcoef(fv, wv)[0, 1]), n


def check_empirical(df: pd.DataFrame,
                    cols=None) -> LeakageResult:
    """Run guards 2 and 3 over ``cols`` (default: the price-free whitelist)."""
    cols = list(cols or PRICE_FREE_FEATURE_COLS)
    won = df.get("won", pd.Series(np.nan, index=df.index))
    failures, table = [], []
    for c in cols:
        if c not in df.columns:
            table.append({"feature": c, "present": False})
            continue
        pc, n_pc = closing_move_pcorr(df[c], df)
        cw, n_cw = outcome_corr(df[c], won)
        table.append({"feature": c, "present": True, "null_pct":
                      float(pd.to_numeric(df[c], errors='coerce').isna().mean() * 100),
                      "pcorr_move": pc, "corr_won": cw, "n": n_pc or n_cw})
        if pc is not None and abs(pc) > MOVE_PCORR_THRESH:
            failures.append(f"{c}: |pcorr closing-move| {abs(pc):.3f} > {MOVE_PCORR_THRESH}")
        if cw is not None and abs(cw) > WON_CORR_THRESH:
            failures.append(f"{c}: |corr won| {abs(cw):.3f} > {WON_CORR_THRESH}")
    return LeakageResult(passed=not failures, failures=failures,
                         details={"table": table})


# ── 4. append-future invariance ──────────────────────────────────────────────


def check_append_invariance(before: pd.DataFrame, after: pd.DataFrame,
                            cols=None, *, cutoff=None,
                            atol: float = 1e-9) -> LeakageResult:
    """Features on rows dated <= cutoff must be identical before/after appending
    strictly-later data.

    ``before``/``after`` are training matrices built from the pre-/post-refresh
    datasets. ``cutoff`` defaults to ``before``'s max race_date. Rows are joined
    on (race_uid, horse_id, market_type); a feature whose value drifts on ANY
    pre-cutoff row reads the future and fails.
    """
    cols = list(cols or PRICE_FREE_FEATURE_COLS)
    cutoff = pd.to_datetime(cutoff or pd.to_datetime(before["race_date"]).max(), utc=True)

    key = ["race_uid", "horse_id", "market_type"]
    b = before[pd.to_datetime(before["race_date"], utc=True) <= cutoff]
    a = after[pd.to_datetime(after["race_date"], utc=True) <= cutoff]
    merged = b[key + [c for c in cols if c in b.columns]].merge(
        a[key + [c for c in cols if c in a.columns]],
        on=key, suffixes=("_b", "_a"), how="inner")
    if merged.empty:
        return LeakageResult(passed=False,
                             failures=["no overlapping rows to compare"],
                             details={"n_overlap": 0})

    failures, drift = [], {}
    for c in cols:
        cb, ca = f"{c}_b", f"{c}_a"
        if cb not in merged.columns or ca not in merged.columns:
            continue
        vb = pd.to_numeric(merged[cb], errors="coerce").to_numpy(dtype=float)
        va = pd.to_numeric(merged[ca], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(vb) & np.isfinite(va)
        changed = int((np.abs(vb[both] - va[both]) > atol).sum())
        nan_flip = int((np.isfinite(vb) != np.isfinite(va)).sum())
        if changed or nan_flip:
            drift[c] = {"value_drift_rows": changed, "nan_flip_rows": nan_flip}
            failures.append(
                f"{c}: {changed} value drift + {nan_flip} nan-flip rows on "
                f"pre-cutoff data (future data changed the past)")
    return LeakageResult(passed=not failures, failures=failures,
                         details={"n_overlap": len(merged), "cutoff": str(cutoff),
                                  "drift": drift})


# ── documented channel: odds-inverse sample weights ──────────────────────────


def check_weight_channel(df: pd.DataFrame) -> LeakageResult:
    """Pin the provenance of the sample-weight price: implied_prob must equal
    1/morningwap (pre-off) wherever priced — never a post-off price."""
    ip = pd.to_numeric(df.get("implied_prob"), errors="coerce")
    mw = pd.to_numeric(df.get("morningwap"), errors="coerce")
    m = ip.notna() & mw.notna() & (mw > 1.0)
    n = int(m.sum())
    if n < _MIN_ROWS:
        return LeakageResult(passed=False,
                             failures=[f"too few priced rows to verify ({n})"])
    ok = np.isclose(ip[m].to_numpy(float), 1.0 / mw[m].to_numpy(float),
                    rtol=1e-6, atol=1e-9)
    frac = float(ok.mean())
    passed = frac > 0.999
    failures = [] if passed else [
        f"implied_prob == 1/morningwap on only {frac:.2%} of priced rows"]
    return LeakageResult(passed=passed, failures=failures,
                         details={"n_priced": n, "match_frac": frac})
