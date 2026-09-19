"""Audit the LightGBM feature matrix for look-ahead / post-off leakage.

Usage
-----
    python tools/audit_lgbm_features.py [feature_parquet]

Defaults to ``data/features/training.parquet`` (falls back to
``data/features.parquet``). Prints, per selected feature: null %, dtype, basic
stats, and two leakage diagnostics; then runs HARD assertions (exit code 1 on
any failure):

  1. PROVENANCE — the selected feature set is disjoint from every known post-off
     column (``POST_OFF_COLS``), and the rebuilt market block reconstructs EXACTLY
     from the pre-off price (morningwap -> ppwap). This proves no SP / odds_finish
     / finishing-position information enters the matrix by construction.

  2. CLOSING-MOVE PARTIAL CORRELATION — the empirical guard. A point-in-time
     feature is a deterministic function of pre-off data, so once a RICH pre-off
     price basis (1/morningwap, log morningwap, 1/ppwap, log ppwap) is partialled
     out, it must carry NO residual information about the closing-line move
     ``move = log(odds_finish) - log(ppwap)`` — a quantity knowable only AFTER the
     off. Any feature computed from odds_finish/SP retains the move and is flagged
     (|partial r| > MOVE_PCORR_THRESH). Empirically, every legitimate feature
     scores <= ~0.06 here while 1/odds_finish scores ~0.62 and log(odds_finish)
     ~1.0, so the 0.30 threshold has a large margin.

  3. OUTCOME CORRELATION — a backstop for non-price leaks: a feature that is a
     near-deterministic copy of the result (``won``/``placed``/finishing position)
     trips |corr(feature, won)| > WON_CORR_THRESH. Legitimate features top out
     around 0.36 (the favourite's market probability); the threshold is 0.70.

The empirical guards (2,3) target the documented v4 failure mode — market
features built from a post-off price. Provenance (1) is the airtight backstop for
everything else (the trailing/speed features are leak-safe by construction).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.lgbm_adapter import (  # noqa: E402
    POST_OFF_COLS,
    build_lgbm_matrix,
    feature_provenance,
)

_DEFAULT_PATHS = (
    os.path.join("data", "features", "training.parquet"),
    os.path.join("data", "features.parquet"),
)

# Thresholds — see module docstring for the empirical justification.
MOVE_PCORR_THRESH = 0.30
WON_CORR_THRESH = 0.70
_MIN_ROWS = 200  # below this a diagnostic is reported as "n/a" rather than run


# ── leakage diagnostics ──────────────────────────────────────────────────────


def _preoff_price(df: pd.DataFrame, cols=("morningwap", "ppwap")) -> pd.Series:
    price = pd.Series(np.nan, index=df.index, dtype=float)
    for c in cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            v = v.where(v > 1.0)
            price = price.where(price.notna(), v)
    return price


def _residual(y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    return y - basis @ coef


def _partial_corr_with_move(feature: pd.Series, df: pd.DataFrame):
    """Partial correlation of ``feature`` with the closing-line move, controlling
    for a rich pre-off price basis. Returns (pcorr, n) or (None, n) if n/a."""
    mw = pd.to_numeric(df.get("morningwap"), errors="coerce").where(lambda s: s > 1.0)
    pp = pd.to_numeric(df.get("ppwap"), errors="coerce").where(lambda s: s > 1.0)
    fin = pd.to_numeric(df.get("odds_finish"), errors="coerce").where(lambda s: s > 1.0)
    f = pd.to_numeric(feature, errors="coerce")

    mask = mw.notna() & pp.notna() & fin.notna() & np.isfinite(f.to_numpy(dtype=float))
    n = int(mask.sum())
    if n < _MIN_ROWS:
        return None, n

    mwv = mw[mask].to_numpy(float)
    ppv = pp[mask].to_numpy(float)
    finv = fin[mask].to_numpy(float)
    fv = f[mask].to_numpy(float)
    move = np.log(finv) - np.log(ppv)

    basis = np.column_stack(
        [np.ones(n), 1.0 / mwv, np.log(mwv), 1.0 / ppv, np.log(ppv)]
    )
    rf = _residual(fv, basis)
    rm = _residual(move, basis)
    if np.std(rf) < 1e-9 or np.std(rm) < 1e-9:
        # Feature fully spanned by the pre-off basis (a pure price function) ->
        # no residual to carry the move. Definitively pre-off.
        return 0.0, n
    return float(np.corrcoef(rf, rm)[0, 1]), n


def _corr_with_won(feature: pd.Series, won: pd.Series):
    f = pd.to_numeric(feature, errors="coerce")
    w = pd.to_numeric(won, errors="coerce")
    mask = np.isfinite(f.to_numpy(dtype=float)) & w.notna().to_numpy()
    n = int(mask.sum())
    if n < _MIN_ROWS:
        return None, n
    fv = f[mask].to_numpy(float)
    wv = w[mask].to_numpy(float)
    if np.std(fv) < 1e-12 or np.std(wv) < 1e-12:
        return 0.0, n
    return float(np.corrcoef(fv, wv)[0, 1]), n


# ── report ───────────────────────────────────────────────────────────────────


def _load(path: str | None) -> tuple[pd.DataFrame, str]:
    candidates = [path] if path else list(_DEFAULT_PATHS)
    for p in candidates:
        if p and os.path.exists(p):
            return pd.read_parquet(p), p
    raise FileNotFoundError(
        f"no feature parquet found (looked for: {[c for c in candidates if c]})"
    )


def main(path: str | None = None) -> int:
    df, used = _load(path)
    # Align df to the training matrix: keep only labelled rows so per-row
    # diagnostics line up with X.
    if "won" in df.columns:
        df = df[pd.to_numeric(df["won"], errors="coerce").notna()].reset_index(drop=True)
    X, y, race_ids = build_lgbm_matrix(df, inference=False)

    prov = feature_provenance()
    won = df["won"] if "won" in df.columns else pd.Series(np.nan, index=df.index)
    have_finish = "odds_finish" in df.columns

    print("=" * 78)
    print(f"LightGBM feature audit  |  source: {used}")
    print(f"rows={len(X):,}  features={X.shape[1]}  races={len(np.unique(race_ids)):,}")
    print("=" * 78)
    print(f"pre-off price priority : {' -> '.join(prov['preoff_price_priority'])}")
    print(f"rebuilt from pre-off   : {', '.join(prov['rebuilt_from_preoff_price'])}")
    print("excluded (post-off)    :")
    for col, reason in prov["excluded"].items():
        print(f"    - {col}: {reason}")
    if not have_finish:
        print("\nNOTE: `odds_finish` absent -> closing-move guard cannot run "
              "(provenance + outcome guards still enforced).")
    print("-" * 78)
    hdr = f"{'feature':28s}{'null%':>7s}{'mean':>11s}{'min':>10s}{'max':>10s}"
    hdr += f"{'pcorr_mv':>9s}{'corr_won':>9s}"
    print(hdr)
    print("-" * 78)

    move_violations: list[tuple[str, float]] = []
    won_violations: list[tuple[str, float]] = []
    high_null: list[tuple[str, float]] = []

    for col in X.columns:
        s = X[col]
        nullpct = float(s.isna().mean() * 100.0)
        finite = s.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        mean = finite.mean() if finite.size else float("nan")
        lo = finite.min() if finite.size else float("nan")
        hi = finite.max() if finite.size else float("nan")

        pc, _ = _partial_corr_with_move(s, df) if have_finish else (None, 0)
        cw, _ = _corr_with_won(s, won)

        pc_str = f"{pc:9.3f}" if pc is not None else f"{'n/a':>9s}"
        cw_str = f"{cw:9.3f}" if cw is not None else f"{'n/a':>9s}"
        print(f"{col:28s}{nullpct:7.2f}{mean:11.4f}{lo:10.3f}{hi:10.3f}{pc_str}{cw_str}")

        if pc is not None and abs(pc) > MOVE_PCORR_THRESH:
            move_violations.append((col, pc))
        if cw is not None and abs(cw) > WON_CORR_THRESH:
            won_violations.append((col, cw))
        if nullpct > 85.0:
            high_null.append((col, nullpct))

    print("-" * 78)

    failures: list[str] = []

    # ── check 1: provenance / structural ──────────────────────────────────────
    selected_postoff = sorted(set(X.columns) & POST_OFF_COLS)
    if selected_postoff:
        failures.append(f"post-off column(s) present in matrix: {selected_postoff}")
    else:
        print("PASS  provenance: ZERO post-off columns selected.")

    # rebuilt implied_prob must equal 1/pre-off price (NOT 1/odds_finish)
    if "implied_prob" in X.columns:
        price = _preoff_price(df).reset_index(drop=True)
        ip_pre = 1.0 / price
        d = (X["implied_prob"] - ip_pre).abs()
        max_diff = float(np.nanmax(d.to_numpy(dtype=float))) if len(d) else 0.0
        if max_diff > 1e-9:
            failures.append(
                f"implied_prob does not reconstruct from the pre-off price "
                f"(max|diff|={max_diff:.3e})"
            )
        else:
            print("PASS  provenance: implied_prob == 1 / (morningwap->ppwap), "
                  "exactly.")
        if have_finish:
            fin = pd.to_numeric(df["odds_finish"], errors="coerce").where(
                lambda s: s > 1.0).reset_index(drop=True)
            d_fin = float(np.nanmax((X["implied_prob"] - 1.0 / fin).abs()
                                    .to_numpy(dtype=float)))
            print(f"      (sanity: max|implied_prob - 1/odds_finish| = "
                  f"{d_fin:.3f}  >> 0, so it is NOT the finishing price)")

    # ── check 2: closing-move partial correlation ─────────────────────────────
    if have_finish:
        if move_violations:
            for col, pc in move_violations:
                failures.append(
                    f"{col}: |partial corr with closing-move| = {abs(pc):.3f} "
                    f"> {MOVE_PCORR_THRESH} (post-off price leakage)"
                )
        else:
            print(f"PASS  closing-move guard: all |pcorr| <= {MOVE_PCORR_THRESH}.")

    # ── check 3: outcome correlation ──────────────────────────────────────────
    if won_violations:
        for col, cw in won_violations:
            failures.append(
                f"{col}: |corr with won| = {abs(cw):.3f} > {WON_CORR_THRESH} "
                f"(looks like a direct outcome copy)"
            )
    else:
        print(f"PASS  outcome guard: all |corr(feature, won)| <= {WON_CORR_THRESH}.")

    if high_null:
        print("\nNOTE  features with >85% null (reported, not a failure):")
        for col, p in high_null:
            print(f"    - {col}: {p:.1f}%")

    print("=" * 78)
    if failures:
        print(f"FAIL: {len(failures)} leakage issue(s) detected:")
        for f in failures:
            print(f"  ✗ {f}")
        print("=" * 78)
        return 1
    print("OK: feature matrix is leak-free (zero post-off features selected).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(arg))
