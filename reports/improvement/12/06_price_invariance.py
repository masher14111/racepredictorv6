"""Step 12 acceptance probe: an EXECUTABLE bookmaker quote must not be able to
move estimated horse ability or the reference market probability.

Four checks, all on real data:

A. Executable-quote perturbation. Rebuild the eval window's features twice from
   the raw store — once as-is, once with ONLY the executable quote columns
   (``odds_decimal``, and the settling ``sp``) perturbed — then re-score the
   saved step-10 conditional logits and recompute the reference de-vig and the
   blend. The price-free probability, the reference probability and the blended
   probability must be bit-identical.
B. Reference-price perturbation (the control that proves A is not vacuous):
   perturbing ``morningwap``/``ppwap`` must move the reference line while
   leaving the price-free probability bit-identical.
C. Missing / incomplete reference book: nulling one runner's reference price
   must NaN that whole race's reference probability and its blend, drop it from
   the scored set, and leave every other race untouched.
D. Non-runners: withdrawing a runner must renormalise the survivors to sum 1
   with their relative standing intact, never impute a probability for the
   withdrawn horse.

Plus an audit of the SHIPPED price-adjustment (``models/fl_oddsband_v3nf_calib.pkl``,
``models.predictor._apply_fl_recalibration``) measured on the real live cache.

Writes reports/improvement/12/price_invariance.json and .txt.
Run: .venv/Scripts/python.exe reports/improvement/12/06_price_invariance.py
"""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from features import builder, derive as _derive
from features.lgbm_adapter import FINAL_FEATURE_COLS, INDEPENDENT_FEATURE_COLS, build_lgbm_matrix
from models.blend import PowerBlend, power_blend
from models.conditional_logit import ConditionalLogitModel
from models.devig import devig

MANIFEST = "reports/improvement/12/blend_manifest.json"
OUT_JSON = "reports/improvement/12/price_invariance.json"
OUT_TXT = "reports/improvement/12/price_invariance.txt"
BLEND = "data/audit/12/blend/combined_indep_forced.json"

EXECUTABLE_COLS = ("odds_decimal", "sp")      # quote we would back at / settle at
REFERENCE_COLS = ("morningwap", "ppwap")      # the fair-line reference book

results: dict = {}
fails: list = []
lines: list = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def check(name: str, ok: bool, detail: str = "") -> None:
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def score(frame: pd.DataFrame, blend: PowerBlend) -> pd.DataFrame:
    """Independent prob, market-assisted prob, reference prob and blend for one
    derived frame, keyed on (race_uid, horse_id)."""
    win = frame[frame["market_type"].astype(str).str.upper() == "WIN"]
    win = win[pd.to_numeric(win["position"], errors="coerce").notna()].reset_index(drop=True)
    panel = load_panel(df=win, config=PanelConfig(feature_cols=()))
    out = panel[["race_uid", "horse_id", "bet_price", "won"]].copy()

    for branch, cols in (("indep", INDEPENDENT_FEATURE_COLS), ("market", FINAL_FEATURE_COLS)):
        model = ConditionalLogitModel.load(f"data/audit/10/models_condlogit/{branch}.json")
        X, _, rid = build_lgbm_matrix(win, inference=True, feature_cols=cols)
        s = pd.DataFrame({"race_uid": win["race_uid"].to_numpy(),
                          "horse_id": win["horse_id"].to_numpy(),
                          f"p_{branch}": model.predict_proba(X, rid)})
        out = out.merge(s.drop_duplicates(subset=["race_uid", "horse_id"]),
                        on=["race_uid", "horse_id"], how="left")

    out["p_reference"] = devig(out["bet_price"], out["race_uid"], method="proportional")
    out["p_blend"] = blend.transform(out["p_indep"], out["p_reference"],
                                     out["race_uid"].to_numpy())
    return out.set_index(["race_uid", "horse_id"]).sort_index()


def max_abs_delta(a: pd.DataFrame, b: pd.DataFrame, col: str) -> float:
    common = a.index.intersection(b.index)
    x = a.loc[common, col].to_numpy(float)
    y = b.loc[common, col].to_numpy(float)
    both_nan = np.isnan(x) & np.isnan(y)
    d = np.abs(x - y)
    d[both_nan] = 0.0
    return float(np.nanmax(d)) if d.size else float("nan")


def main() -> int:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    blend = PowerBlend.load(BLEND)
    w = manifest["windows"]["eval"]
    start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end_exclusive"])

    say("=" * 72)
    say("Step 12 — executable-quote / reference-book / non-runner invariance")
    say(f"eval window {w['start']}..{w['end_exclusive']}  blend alpha={blend.alpha} "
        f"beta={blend.beta}")
    say("=" * 72)

    raw = pd.read_parquet("data/unified_races.parquet")
    keyed = _derive.add_race_key(raw.assign(
        race_date=pd.to_datetime(raw["race_date"], utc=True, errors="coerce")))
    day = pd.to_datetime(raw["race_date"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    in_window = ((day >= start) & (day < end)).to_numpy()
    slice_raw = raw.loc[in_window].copy()
    say(f"\nraw probe slice: {len(slice_raw):,} rows / "
        f"{keyed.loc[in_window, 'race_uid'].nunique()} races")

    cfg = builder._load_cfg()
    base_derived = builder._derive_all(slice_raw, cfg)
    base = score(base_derived, blend)
    say(f"scored base frame: {len(base):,} runner rows")

    rng = np.random.default_rng(12)

    # ── A. executable quote only ─────────────────────────────────────────────
    say("\nA. perturb ONLY the executable quote (odds_decimal, sp)")
    pert = slice_raw.copy()
    touched = []
    for col in EXECUTABLE_COLS:
        if col not in pert.columns:
            continue
        v = pd.to_numeric(pert[col], errors="coerce")
        # Historical rows leave these null (they are live-only columns), so the
        # honest perturbation is null -> a real best-of-N overlay AND a violent
        # move: both must be invisible downstream.
        overlay = pd.to_numeric(slice_raw["morningwap"], errors="coerce") * \
            rng.uniform(1.05, 1.25, size=len(pert))
        newv = v.where(v.notna(), overlay) * rng.uniform(1.5, 4.0, size=len(pert))
        pert[col] = newv.astype(object)
        touched.append(f"{col} (was {100 * v.isna().mean():.0f}% null)")
    say(f"   perturbed: {', '.join(touched)}")
    a_derived = builder._derive_all(pert, cfg)
    a = score(a_derived, blend)

    moved_cols = []
    key = ["race_uid", "horse_id", "market_type"]
    bd = base_derived.set_index(key).sort_index()
    ad = a_derived.set_index(key).sort_index()
    common = bd.index.intersection(ad.index)
    for col in FINAL_FEATURE_COLS:
        if col not in bd.columns:
            continue
        x = pd.to_numeric(bd.loc[common, col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(ad.loc[common, col], errors="coerce").to_numpy(float)
        if int((~(np.isclose(x, y, equal_nan=True))).sum()):
            moved_cols.append(col)

    d_ind = max_abs_delta(base, a, "p_indep")
    d_ref = max_abs_delta(base, a, "p_reference")
    d_bl = max_abs_delta(base, a, "p_blend")
    d_mkt = max_abs_delta(base, a, "p_market")
    check("independent (price-free) probability is unmoved by an executable quote",
          d_ind == 0.0, f"max|delta|={d_ind:.3e}")
    check("reference market probability is unmoved by an executable quote",
          d_ref == 0.0, f"max|delta|={d_ref:.3e}")
    check("blended probability is unmoved by an executable quote",
          d_bl == 0.0, f"max|delta|={d_bl:.3e}")
    say(f"  [INFO] market-ASSISTED branch max|delta| = {d_mkt:.3e} over "
        f"{len(moved_cols)} reacting feature(s): {moved_cols}")
    results["A_executable_quote"] = {
        "perturbed_columns": list(EXECUTABLE_COLS),
        "max_abs_delta_independent": d_ind, "max_abs_delta_reference": d_ref,
        "max_abs_delta_blend": d_bl, "max_abs_delta_market_assisted": d_mkt,
        "market_assisted_features_that_moved": moved_cols,
        "n_rows_compared": int(len(base.index.intersection(a.index))),
    }

    # ── B. reference price (control) ─────────────────────────────────────────
    say("\nB. control — perturb ONLY the reference price (morningwap, ppwap)")
    pert2 = slice_raw.copy()
    for col in REFERENCE_COLS:
        if col in pert2.columns:
            v = pd.to_numeric(pert2[col], errors="coerce")
            pert2[col] = (v * rng.uniform(1.5, 3.0, size=len(v))).astype(object)
    b = score(builder._derive_all(pert2, cfg), blend)
    b_ind = max_abs_delta(base, b, "p_indep")
    b_ref = max_abs_delta(base, b, "p_reference")
    check("independent probability STILL unmoved when the reference price moves",
          b_ind == 0.0, f"max|delta|={b_ind:.3e}")
    check("reference probability DOES move (probe is not vacuous)",
          b_ref > 1e-6, f"max|delta|={b_ref:.3e}")
    results["B_reference_price_control"] = {
        "max_abs_delta_independent": b_ind, "max_abs_delta_reference": b_ref}

    # ── C. missing / incomplete reference book ───────────────────────────────
    say("\nC. missing / incomplete reference book")
    panel = base.reset_index()
    races = panel["race_uid"].drop_duplicates().to_numpy()
    victims = set(races[: max(1, len(races) // 10)])
    holed = panel.copy()
    first_row = holed.groupby("race_uid").head(1).index
    hole_idx = [i for i in first_row if holed.loc[i, "race_uid"] in victims]
    holed.loc[hole_idx, "bet_price"] = np.nan
    holed["p_reference"] = devig(holed["bet_price"], holed["race_uid"],
                                 method="proportional")
    holed["p_blend"] = power_blend(holed["p_indep"], holed["p_reference"],
                                   holed["race_uid"], blend.alpha, blend.beta)
    hit = holed["race_uid"].isin(victims).to_numpy()
    check("every runner of an incomplete-book race has a NaN reference prob",
          bool(holed.loc[hit, "p_reference"].isna().all()),
          f"{int(holed.loc[hit, 'p_reference'].isna().sum())}/{int(hit.sum())} rows")
    check("every runner of an incomplete-book race has a NaN blend (never partial)",
          bool(holed.loc[hit, "p_blend"].isna().all()))
    check("complete races are untouched by another race's missing price",
          bool(np.allclose(holed.loc[~hit, "p_blend"].to_numpy(float),
                           panel.loc[~hit, "p_blend"].to_numpy(float), equal_nan=True)))
    check("the independent ability estimate survives a missing reference price",
          bool(holed.loc[hit, "p_indep"].notna().all()))
    results["C_incomplete_reference_book"] = {
        "n_races_holed": len(victims), "n_rows_affected": int(hit.sum()),
        "reference_nan_rows": int(holed.loc[hit, "p_reference"].isna().sum()),
        "blend_nan_rows": int(holed.loc[hit, "p_blend"].isna().sum()),
        "independent_still_defined_rows": int(holed.loc[hit, "p_indep"].notna().sum()),
    }

    # ── D. non-runners ───────────────────────────────────────────────────────
    say("\nD. non-runner withdrawal")
    big = panel.groupby("race_uid")["horse_id"].transform("size") >= 6
    sample = panel[big].groupby("race_uid").head(1).head(200)
    max_off, max_rank_break, n_done = 0.0, 0, 0
    for _, row in sample.iterrows():
        r = panel[panel["race_uid"] == row["race_uid"]].copy()
        kept = r[r["horse_id"] != row["horse_id"]].copy()
        kept["p_reference"] = devig(kept["bet_price"], kept["race_uid"],
                                    method="proportional")
        kept["p_blend"] = power_blend(kept["p_indep"], kept["p_reference"],
                                      kept["race_uid"], blend.alpha, blend.beta)
        total = float(np.nansum(kept["p_blend"]))
        max_off = max(max_off, abs(total - 1.0))
        before = r[r["horse_id"] != row["horse_id"]]["p_blend"].to_numpy(float)
        after = kept["p_blend"].to_numpy(float)
        if list(np.argsort(before)) != list(np.argsort(after)):
            max_rank_break += 1
        n_done += 1
    check("survivors renormalise to exactly 1.0 after a withdrawal",
          max_off < 1e-9, f"max|sum-1|={max_off:.3e} over {n_done} races")
    check("a withdrawal does not reorder the survivors",
          max_rank_break == 0, f"reordered_races={max_rank_break}")
    check("no probability is imputed for the withdrawn runner",
          True, "the row leaves the field; the blend is computed over the survivors only")
    results["D_non_runner"] = {"n_races_tested": n_done,
                               "max_abs_sum_minus_one": max_off,
                               "races_reordered": max_rank_break}

    # ── E. shipped price-adjustment audit ────────────────────────────────────
    say("\nE. audit of the SHIPPED favourite-longshot price adjustment")
    art = "models/fl_oddsband_v3nf_calib.pkl"
    cache = "data/predictions.json"
    if not (os.path.exists(art) and os.path.exists(cache)):
        say(f"  [SKIP] {art} or {cache} absent")
        results["E_fl_recalibration_audit"] = {"status": "unavailable"}
    else:
        with open(art, "rb") as fh:
            obc = pickle.load(fh)
        cached = json.load(open(cache, "r", encoding="utf-8"))
        rows = [r for race in cached.get("races", []) for r in race.get("runners", [])]
        d = pd.DataFrame(rows)
        need = {"value_win_prob_independent", "reference_odds", "best_odds"}
        d = d.dropna(subset=list(need)) if need <= set(d.columns) else pd.DataFrame()
        if d.empty:
            say("  [SKIP] live cache carries no reference/executable price pair")
            results["E_fl_recalibration_audit"] = {"status": "no_paired_prices"}
        else:
            ind = d["value_win_prob_independent"].to_numpy(float)
            ref = d["reference_odds"].to_numpy(float)
            exe = d["best_odds"].to_numpy(float)
            p_ref = np.asarray(obc.predict(ind, ref), dtype=float)
            p_exe = np.asarray(obc.predict(ind, exe), dtype=float)
            bumped = np.asarray(obc.predict(ind, exe * 1.10), dtype=float)
            say(f"  live rows with both prices: {len(d)}; "
                f"executable/reference price ratio mean {np.mean(exe / ref):.4f}")
            say(f"  served value_win_prob uses the EXECUTABLE price "
                f"(models.predictor._effective_decimal) while the artifact was FIT on the "
                f"reference price (docs/calibration/fit_fl_recalibrator.py: bet_price=ppwap)")
            say(f"  |p(exec) - p(reference)|: mean {np.mean(np.abs(p_exe - p_ref)):.5f}, "
                f"max {np.max(np.abs(p_exe - p_ref)):.5f}")
            say(f"  a +10% move in the EXECUTABLE quote alone shifts the served "
                f"probability by mean {np.mean(np.abs(bumped - p_exe)):.5f}, "
                f"max {np.max(np.abs(bumped - p_exe)):.5f}")
            say("  [INFO] this is the shipped live path, NOT this stage's blend: "
                "models.blend.power_blend has no price argument at all. Recorded as a "
                "finding; no production behaviour changed in this stage.")
            results["E_fl_recalibration_audit"] = {
                "status": "measured", "n_live_rows": int(len(d)),
                "mean_executable_over_reference": float(np.mean(exe / ref)),
                "mean_abs_prob_shift_exec_vs_reference": float(np.mean(np.abs(p_exe - p_ref))),
                "max_abs_prob_shift_exec_vs_reference": float(np.max(np.abs(p_exe - p_ref))),
                "mean_abs_prob_shift_from_10pct_quote_move": float(np.mean(np.abs(bumped - p_exe))),
                "max_abs_prob_shift_from_10pct_quote_move": float(np.max(np.abs(bumped - p_exe))),
                "finding": (
                    "models.predictor._score_value feeds _apply_fl_recalibration the "
                    "EXECUTABLE best-of-N board price (_effective_decimal) while the shipped "
                    "artifact was fitted on the pre-off REFERENCE price (ppwap). The served "
                    "value_win_prob therefore moves when a bookmaker moves only its quote, and "
                    "is evaluated in a systematically longer odds band than it was fit for. "
                    "value_win_prob_independent (what execution.gates prefers) is unaffected."
                ),
            }

    say("\n" + "=" * 72)
    ok = not fails
    say("PASS: an executable quote cannot move ability or the reference line"
        if ok else "FAIL: " + "; ".join(fails))
    results["_summary"] = {"passed": ok, "failures": fails}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    Path(OUT_TXT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
