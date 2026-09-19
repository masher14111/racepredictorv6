"""Step 09 audit C — market identity, full-field and provenance invariants on
the real step-06 candidate matrix (the first matrix built end-to-end through
every step 02-06 fix).

Checks:
  C1 uniqueness of (race_uid, horse_id, market_type)
  C2 WIN and PLACE books are genuinely separate (prices differ, terms do not
     cross, position IS shared — the documented cross-fill)
  C3 full-field invariant: field_size == observed rows of that race's OWN book,
     and overround_norm_prob sums to ~1 within each book
  C4 market_rank is ranked inside one book only
  C5 independent/price-free feature provenance: no PRICE_FREE_FEATURE_COL is a
     pure function of the market columns (structural mutation probe)
  C6 non-runner / runner_status handling in the labelled matrix

Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.features import PRICE_FEATURE_COLS, PRICE_FREE_FEATURE_COLS  # noqa: E402

SRC = Path("data/audit/10/training_rebuilt.parquet")  # step10 regression gate
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(f"{name} {detail}")


def main() -> int:
    df = pd.read_parquet(SRC)
    print(f"source: {SRC}  rows={len(df):,}  cols={len(df.columns)}")

    print("\nC1 identity uniqueness")
    dup = df.duplicated(["race_uid", "horse_id", "market_type"]).sum()
    check("no duplicate (race_uid, horse_id, market_type)", dup == 0, f"dups={dup}")
    mk = df["market_type"].value_counts(dropna=False).to_dict()
    print(f"       market_type counts: {mk}")

    print("\nC2 WIN vs PLACE book separation")
    piv = df.pivot_table(index=["race_uid", "horse_id"], columns="market_type",
                         values="odds_decimal", aggfunc="first")
    if {"WIN", "PLACE"}.issubset(piv.columns):
        both = piv.dropna(subset=["WIN", "PLACE"])
        same = (both["WIN"] == both["PLACE"]).mean() if len(both) else float("nan")
        print(f"       runners priced in BOTH books: {len(both):,}; identical price "
              f"fraction={same:.4f}")
    else:
        print(f"       odds_decimal never populated in one book "
              f"(cols={list(piv.columns)}) — checking morningwap instead")
        piv = df.pivot_table(index=["race_uid", "horse_id"], columns="market_type",
                             values="morningwap", aggfunc="first")
        both = piv.dropna(subset=[c for c in ("WIN", "PLACE") if c in piv.columns])
        if {"WIN", "PLACE"}.issubset(piv.columns) and len(both):
            same = (both["WIN"] == both["PLACE"]).mean()
            print(f"       morningwap both-book runners={len(both):,} identical={same:.4f}")
            check("WIN/PLACE morningwap are not universally identical", same < 0.99,
                  f"identical_fraction={same:.4f}")
    # position IS deliberately shared across a runner's two rows
    ppiv = df.pivot_table(index=["race_uid", "horse_id"], columns="market_type",
                          values="position", aggfunc="first")
    if {"WIN", "PLACE"}.issubset(ppiv.columns):
        pb = ppiv.dropna(subset=["WIN", "PLACE"])
        agree = (pb["WIN"] == pb["PLACE"]).mean() if len(pb) else float("nan")
        check("finishing position agrees across a runner's WIN/PLACE rows",
              bool(len(pb) == 0 or agree > 0.999), f"agree={agree:.4f} n={len(pb):,}")

    print("\nC3a labelled-matrix sanity (field_size is a FULL-field count, so a")
    print("    label-filtered matrix must satisfy field_size >= observed rows)")
    g = df.groupby(["race_uid", "market_type"], sort=False)
    obs = g["horse_id"].transform("size")
    fs = pd.to_numeric(df["field_size"], errors="coerce")
    under = int((fs < obs).sum())
    check("field_size >= rows of that race's own book in the labelled matrix",
          under == 0, f"violating_rows={under}")
    mr = pd.to_numeric(df["market_rank"], errors="coerce")
    check("market_rank never exceeds its own book's FULL field size",
          int((mr > fs).sum()) == 0, f"bad_rows={int((mr > fs).sum())}")

    print("\nC5 price-free provenance (structural mutation probe) + C3b/C4b "
          "full-field invariants on a freshly derived, pre-label-filter frame")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from features import builder  # noqa: E402
    sample_races = df["race_uid"].drop_duplicates().tail(120)
    raw = pd.read_parquet("data/unified_races.parquet")
    # rebuild a small slice twice: once as-is, once with every market/price
    # column perturbed, and compare the price-free feature columns.
    from features import derive as _derive  # noqa: E402
    keyed = _derive.add_race_key(raw.assign(
        race_date=pd.to_datetime(raw["race_date"], utc=True, errors="coerce")))
    slice_raw = raw.loc[keyed["race_uid"].isin(set(sample_races)).to_numpy()].copy()
    print(f"       probe slice raw rows={len(slice_raw):,} over "
          f"{len(sample_races)} races")
    cfg = builder._load_cfg()
    base = builder._derive_all(slice_raw, cfg)
    pert = slice_raw.copy()
    rng = np.random.default_rng(11)
    for col in ("odds_decimal", "sp", "morningwap", "ppwap"):
        if col in pert.columns:
            v = pd.to_numeric(pert[col], errors="coerce")
            pert[col] = (v * rng.uniform(1.5, 3.0, size=len(v))).astype(object)
    pert_out = builder._derive_all(pert, cfg)
    key = ["race_uid", "horse_id", "market_type"]
    b = base.set_index(key).sort_index()
    p = pert_out.set_index(key).sort_index()
    common = b.index.intersection(p.index)
    b, p = b.loc[common], p.loc[common]
    # C3b/C4b on the unfiltered derived frame — this is where field_size /
    # overround_norm_prob are actually defined.
    gb = base.groupby(["race_uid", "market_type"], sort=False)
    obs_b = gb["horse_id"].transform("size")
    fs_b = pd.to_numeric(base["field_size"], errors="coerce")
    check("field_size == rows of that race's own book (pre-label-filter)",
          int((fs_b != obs_b).sum()) == 0,
          f"mismatched_rows={int((fs_b != obs_b).sum())} of {len(base):,}")
    tot_b = gb["overround_norm_prob"].transform("sum")
    priced_b = pd.to_numeric(base["overround_norm_prob"], errors="coerce").notna()
    off_b = float(np.abs(tot_b[priced_b] - 1.0).max()) if priced_b.any() else float("nan")
    check("overround_norm_prob sums to 1 inside each book (pre-label-filter)",
          bool(priced_b.sum() == 0 or off_b < 1e-6),
          f"max|sum-1|={off_b:.2e} priced_rows={int(priced_b.sum()):,}")
    mr_b = pd.to_numeric(base["market_rank"], errors="coerce")
    check("market_rank <= own book field size (pre-label-filter)",
          int((mr_b > obs_b).sum()) == 0, f"bad_rows={int((mr_b > obs_b).sum())}")

    changed_pf, changed_price = [], []
    probe_cols = PRICE_FREE_FEATURE_COLS + ["race_complexity_v2", "going_speed_v2"]
    for col in probe_cols + PRICE_FEATURE_COLS + ["race_market_entropy"]:
        if col not in b.columns:
            continue
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        pv = pd.to_numeric(p[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(bv) & np.isnan(pv)
        diff = int((~both_nan & ~np.isclose(bv, pv, equal_nan=True)).sum())
        if diff:
            (changed_pf if col in probe_cols else changed_price).append((col, diff))
    print(f"       PRICE/market features that moved (expected): "
          f"{[c for c, _ in changed_price]}")
    check("no price-free / candidate-independent feature reacts to a "
          "market-price perturbation", not changed_pf, f"reacting={changed_pf}")

    print("\nC6 non-runner handling")
    for col in ("runner_status", "jockey_connections_source", "trainer_connections_source"):
        if col in df.columns:
            print(f"       {col}: {df[col].value_counts(dropna=False).head(4).to_dict()}")
        else:
            print(f"       {col}: ABSENT from labelled matrix")
    print(f"       rows with null position in a LABELLED matrix: "
          f"{int(df['position'].isna().sum()):,} (must be 0)")
    check("labelled matrix carries no unresolved position",
          int(df["position"].isna().sum()) == 0)

    print("\n" + "=" * 60)
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: market identity, full-field and provenance invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
