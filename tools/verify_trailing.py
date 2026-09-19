"""Equivalence check: fast _trailing_fast core vs the original O(n^2) scan.

Reconstructs the OLD implementations inline and compares their output to the new
vectorized derive/engine functions on a real-data sample. Any mismatch prints the
offending rows; otherwise prints VERIFY_OK.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features import derive, engine
from features.fuse import fuse_sources
from features.builder import _load_cfg

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
cfg = _load_cfg()
uni = pd.read_parquet("data/unified_races.parquet").head(N).copy()

base = fuse_sources(uni)
base["race_date"] = pd.to_datetime(base["race_date"], utc=True, errors="coerce")
base = derive.add_going_speed(base, cfg["going_speed_map"])
base = derive.add_odds_features(base)  # field_size for speed figures


# ---- OLD implementations (verbatim copies of the pre-refactor loops) ----
def old_trailing(df, entity, win_col, place_col, months, runs, places, strict):
    out = df.copy().reset_index(drop=True)
    out[win_col] = pd.NA
    out[place_col] = pd.NA
    out["_runs"] = 0
    window = pd.DateOffset(months=months)
    if len(entity) == 1:
        key = out[entity[0]]
    else:
        key = out[entity].astype(object).agg(tuple, axis=1)
    for _, grp in out.groupby(key, sort=False):
        g = grp.sort_values("race_date")
        idx = list(g.index)
        for pos_i, i in enumerate(idx):
            cutoff = out.at[i, "race_date"]
            prior = g.loc[idx[:pos_i]]
            prior = prior[prior["race_date"] >= (cutoff - window)]
            if strict:
                prior = prior[prior["race_date"] < cutoff]
            if runs:
                prior = prior.tail(runs)
            prior = prior[prior["position"].notna()]
            n = len(prior)
            out.at[i, "_runs"] = n
            if n:
                out.at[i, win_col] = (prior["position"] == 1).sum() / n
                out.at[i, place_col] = (prior["position"] <= places).sum() / n
    return out


def cmp(name, a, b, valid=None):
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64")
    bad = ~((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=True))
    if valid is not None:
        bad = bad & valid  # ignore rows whose grouping key is null (old pooled them)
    n_bad = int(bad.sum())
    print(f"  {name}: {n_bad} mismatches / {len(a)}", flush=True)
    if n_bad:
        idx = np.where(bad)[0][:10]
        for i in idx:
            print(f"    row {i}: old={a[i]!r} new={b[i]!r}", flush=True)
    return n_bad


m, r, p = cfg["lookback_months"], cfg["lookback_runs"], cfg["place_positions"]
total = 0

# derive horse trailing (non-strict)
old = old_trailing(base, ["horse_id"], "w", "pl", m, r, p, strict=False)
new = derive.add_trailing_rates(base, "horse_id", "w", "pl", m, r, p)
total += cmp("derive horse win", old["w"], new["w"])
total += cmp("derive horse place", old["pl"], new["pl"])
total += cmp("derive horse runs", old["_runs"], new["runs_in_window"])

# engine combo (strict, 2-key). Compare only rows with BOTH ids present; the old
# loop pooled null-id rows into one fake entity (the bug the new path drops).
valid_combo = (base["jockey_id"].notna() & base["trainer_id"].notna()).to_numpy()
old = old_trailing(base, ["jockey_id", "trainer_id"], "w", "pl", m, r, p, strict=True)
new = engine.add_combo_win_rate(base, cfg)
total += cmp("combo win", old["w"], new["jt_combo_win_rate"], valid_combo)
total += cmp("combo runs", old["_runs"], new["jt_combo_runs"], valid_combo)

# going preference: new grouped fast path vs the original predicate scan (kept in
# _trailing_rate). Both null out NaN-horse and NaN-going rows, so compare all rows.
old_gp = engine._trailing_rate(
    base, ["horse_id"], "w", "pl", "_runs", cfg,
    predicate=lambda prior, cur: prior["going_speed"] == cur["going_speed"])
new_gp = engine.add_going_preference(base, cfg)
total += cmp("going_pref win", old_gp["w"], new_gp["going_pref_win_rate"])
total += cmp("going_pref place", old_gp["pl"], new_gp["going_pref_place_rate"])


# speed proxy: reconstruct the original per-horse strict-mean loop as the oracle.
def old_speed_proxy(df):
    out = df.copy().reset_index(drop=True)
    pos = pd.to_numeric(out.get("position"), errors="coerce")
    fs = pd.to_numeric(out.get("field_size"), errors="coerce")
    valid = pos.notna() & fs.notna() & (fs > 1)
    out["_run_speed"] = pd.Series(
        np.where(valid, (fs - pos) / (fs - 1.0) * 100.0, np.nan), index=out.index)
    out["_sp"] = np.nan
    window = pd.DateOffset(months=cfg["lookback_months"])
    runs = cfg["lookback_runs"]
    for _, grp in out.groupby("horse_id", sort=False):
        g = grp.sort_values("race_date")
        idx = list(g.index)
        for pos_i, i in enumerate(idx):
            cutoff = out.at[i, "race_date"]
            prior = g.loc[idx[:pos_i]]
            prior = prior[prior["race_date"] >= (cutoff - window)]
            prior = prior[prior["race_date"] < cutoff]
            if runs:
                prior = prior.tail(runs)
            vals = prior["_run_speed"].dropna()
            if len(vals):
                out.at[i, "_sp"] = float(vals.mean())
    return out["_sp"]


from features import _trailing_fast
base2 = base.copy().reset_index(drop=True)
pos = pd.to_numeric(base2.get("position"), errors="coerce")
fs = pd.to_numeric(base2.get("field_size"), errors="coerce")
valid = pos.notna() & fs.notna() & (fs > 1)
base2["_run_speed"] = pd.Series(
    np.where(valid, (fs - pos) / (fs - 1.0) * 100.0, np.nan), index=base2.index)
new_proxy = _trailing_fast.windowed_mean(
    base2, ["horse_id"], "_run_speed", runs=cfg["lookback_runs"],
    months=cfg["lookback_months"], strict=True, dropna=True)
total += cmp("speed proxy", old_speed_proxy(base), pd.Series(new_proxy))

print("VERIFY_OK" if total == 0 else f"VERIFY_FAIL ({total} mismatches)", flush=True)
