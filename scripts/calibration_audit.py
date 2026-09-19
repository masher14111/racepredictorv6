"""Stage-4 leak-free calibration and model-validity audit — the driver CLI.

Phases (each idempotent, artifacts under ``data/audit/<run>/``):

    panel        validated evaluation panel (requirement 1) + exclusion account
    leakage      requirement-2 proofs for the price-free feature set
    walkforward  expanding-window walk-forward, everything fit inside folds
                 (requirements 3/5/6); folds end BEFORE the final test window
    select       calibration-method comparison on the walk-forward folds ONLY
                 (requirement 5), F-L circularity ablation (requirement 3),
                 probability-extreme / support / portability analysis (req 4)
    final        ONE-SHOT evaluation on the untouched final window
                 (requirement 7): audit walk-forward line + frozen v3nf/v3/
                 LightGBM artifacts + de-vigged market on identical races
                 (requirements 8/10), with race-bootstrap intervals
    report       assemble the dated markdown calibration report (acceptance 3)

Run everything:

    python -m scripts.calibration_audit --run stage4 --all

or phase by phase (later phases read earlier phases' artifacts):

    python -m scripts.calibration_audit --run stage4 --phase panel
    python -m scripts.calibration_audit --run stage4 --phase walkforward
    ...

The final test window is frozen by --final-start/--final-end and is scored
exactly once, by the ``final`` phase. Nothing in ``select`` reads it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

from utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.calibration_audit")

TRAINING_PARQUET = _BASE / "data" / "features" / "training.parquet"
BACKUP_PARQUET = _BASE / "data" / "backups" / "training_20260618_pre_stage4.parquet"
AUDIT_ROOT = _BASE / "data" / "audit"

# The frozen final window (requirement 7). Chosen as the span of settled results
# fetched 2026-07-27 that postdates EVERY existing model's training data
# (max 2026-06-12), every holdout already published (max end 2026-06-12), and
# every value-gate tuning frame (max 2026-06-16 build). Scored once, in `final`.
FINAL_START_DEFAULT = "2026-06-13"
FINAL_END_DEFAULT = "2026-07-25"

# race_complexity is excluded from the audit walk-forward feature set: the
# append-invariance check proved it is built from dataset-global z-scores
# (future data shifts historical values — mean |drift| 0.0024, corr 1.0), and
# one of its components is the race-level market entropy. See leakage.json.
EXCLUDE_FEATURES_WF = ["race_complexity"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_BASE,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")
    logger.info("audit: wrote %s", path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


# ── phase: panel ─────────────────────────────────────────────────────────────


def phase_panel(run_dir: Path, args) -> None:
    from audit.panel import build_audit_panel

    df = pd.read_parquet(TRAINING_PARQUET)
    res = build_audit_panel(df)
    res.panel.to_parquet(run_dir / "panel.parquet", index=False)
    res.exclusions.to_csv(run_dir / "panel_exclusions.csv", index=False)
    rd = pd.to_datetime(res.panel["race_date"], utc=True)
    _write_json(run_dir / "panel_meta.json", {
        "source": str(TRAINING_PARQUET),
        "source_sha256": _sha256(TRAINING_PARQUET),
        "n_input_rows": res.n_input_rows,
        "n_panel_rows": len(res.panel),
        "n_races": int(res.panel["race_uid"].nunique()),
        "date_min": str(rd.min()), "date_max": str(rd.max()),
        "book_complete_frac": float(res.panel["book_complete"].mean()),
        "exclusions": res.exclusions.to_dict(orient="records"),
    })


# ── phase: leakage ───────────────────────────────────────────────────────────


def phase_leakage(run_dir: Path, args) -> None:
    from audit import leakage
    from models.features import FEATURE_COLS

    after = pd.read_parquet(TRAINING_PARQUET)
    payload = {"provenance": None, "empirical": None,
               "append_invariance": None, "weight_channel": None}

    r = leakage.check_provenance()
    payload["provenance"] = {"passed": r.passed, "failures": r.failures,
                             "details": r.details}

    panel = pd.read_parquet(run_dir / "panel.parquet")
    r = leakage.check_empirical(panel)
    payload["empirical"] = {"passed": r.passed, "failures": r.failures,
                            "table": r.details["table"]}

    if BACKUP_PARQUET.exists():
        before = pd.read_parquet(BACKUP_PARQUET)
        r_pf = leakage.check_append_invariance(before, after)
        r_full = leakage.check_append_invariance(before, after,
                                                 cols=FEATURE_COLS)
        payload["append_invariance"] = {
            "price_free": {"passed": r_pf.passed, "failures": r_pf.failures,
                           "details": r_pf.details},
            "full_list": {"passed": r_full.passed, "failures": r_full.failures},
            "backup_sha256": _sha256(BACKUP_PARQUET),
        }
    else:
        payload["append_invariance"] = {"skipped": "no backup matrix on disk"}

    r = leakage.check_weight_channel(after)
    payload["weight_channel"] = {"passed": r.passed, "failures": r.failures,
                                 "details": r.details}
    _write_json(run_dir / "leakage.json", _jsonable(payload))


# ── phase: walkforward ───────────────────────────────────────────────────────


def phase_walkforward(run_dir: Path, args) -> None:
    from audit.walkforward import AuditWFConfig, run_walkforward_audit
    from models.features import PRICE_FREE_FEATURE_COLS

    panel = pd.read_parquet(run_dir / "panel.parquet")
    meta = _read_json(_BASE / "models" / "catboost_v3nf_meta.json")
    best = dict(meta.get("targets", {}).get("won", {}).get("best_params") or {})

    cols = [c for c in PRICE_FREE_FEATURE_COLS if c not in EXCLUDE_FEATURES_WF]
    cfg = AuditWFConfig(
        feature_cols=cols,
        min_train_days=int(args.min_train_days),
        test_window_days=int(args.window_days),
        final_test_start=args.final_start,
        best_params=best,
        task_type=args.task_type,
    )
    scored, fold_meta = run_walkforward_audit(panel, cfg)
    scored.to_parquet(run_dir / "wf_scored.parquet", index=False)
    _write_json(run_dir / "wf_meta.json", _jsonable({
        "config": asdict(cfg),
        "excluded_features": EXCLUDE_FEATURES_WF,
        "n_scored": len(scored),
        "n_folds": len(fold_meta),
        "folds": fold_meta,
    }))


# ── phase: select (validation-fold decisions + ablations + extremes) ─────────


def phase_select(run_dir: Path, args) -> None:
    from audit import extremes as ex
    from audit.metrics_panel import (
        ae_table,
        ece_equal_freq,
        head_to_head_ci,
        odds_band_series,
        per_race_logloss,
    )
    from models.calibration import OddsBandCalibrator

    wf = pd.read_parquet(run_dir / "wf_scored.parquet")
    payload: dict = {}

    # 1. Requirement 5 — calibration-method comparison, VALIDATION FOLDS ONLY.
    arms = {
        "raw_normalized": "norm_raw",
        "marginal_calib_then_normalize (production)": "norm_ind",
        "grouped_softmax_temperature": "norm_temp",
        "fl_market_adjusted_then_normalize": "norm_adj",
    }
    comp = {}
    y = wf["won"].to_numpy(dtype=float)
    for name, col in arms.items():
        ll = per_race_logloss(wf, col)
        comp[name] = {
            "race_log_loss": float(ll.mean()),
            "runner_brier": float(np.nanmean(
                (pd.to_numeric(wf[col], errors="coerce").to_numpy(float) - y) ** 2)),
            "ece": ece_equal_freq(
                pd.to_numeric(wf[col], errors="coerce").to_numpy(float), y),
        }
    order = sorted(comp, key=lambda k: comp[k]["race_log_loss"])
    payload["calibration_method_comparison"] = {
        "note": "walk-forward validation folds only; the final window is untouched",
        "arms": comp, "winner_by_race_log_loss": order[0],
    }

    # 2. Requirement 3 — F-L cross-fit vs in-sample optimism + circularity.
    p_ind = wf["p_ind"].to_numpy(dtype=float)
    p_adj = wf["p_adj"].to_numpy(dtype=float)
    odds = wf["bet_price"].to_numpy(dtype=float)
    ok = np.isfinite(p_ind) & np.isfinite(odds) & (odds > 1.0)
    fl_full = OddsBandCalibrator.fit(p_ind[ok], odds[ok], y[ok].astype(int),
                                     method="isotonic", min_rows=400)
    p_adj_insample = p_ind.copy()
    p_adj_insample[ok] = np.asarray(fl_full.predict(p_ind[ok], odds[ok]), float)

    def _brier(p):
        return float(np.nanmean((p - y) ** 2))

    payload["fl_ablation"] = {
        "brier_independent": _brier(p_ind),
        "brier_crossfit_adjusted": _brier(p_adj),
        "brier_insample_adjusted": _brier(p_adj_insample),
        "insample_optimism_brier": _brier(p_adj) - _brier(p_adj_insample),
        "note": ("crossfit = per-fold F-L (every row scored by a recalibrator "
                 "fit strictly before its fold); insample = one F-L fit on all "
                 "OOS rows then applied to the same rows"),
    }

    # Price-collapse: does the adjustment drag the prob onto the price?
    inv = 1.0 / odds[ok]
    payload["fl_ablation"]["corr_with_inverse_price"] = {
        "independent": float(np.corrcoef(p_ind[ok], inv)[0, 1]),
        "crossfit_adjusted": float(np.corrcoef(p_adj[ok], inv)[0, 1]),
    }

    # Same-price discrimination: AUC within fine price bins.
    def _within_price_auc(p):
        from sklearn.metrics import roc_auc_score
        bins = pd.qcut(pd.Series(odds[ok]), q=25, duplicates="drop")
        aucs, ns = [], []
        frame = pd.DataFrame({"b": bins.to_numpy(), "p": p[ok], "y": y[ok]})
        for _, sub in frame.groupby("b", observed=True):
            if sub["y"].nunique() < 2 or len(sub) < 200:
                continue
            aucs.append(roc_auc_score(sub["y"], sub["p"]))
            ns.append(len(sub))
        return float(np.average(aucs, weights=ns)) if aucs else float("nan")

    payload["fl_ablation"]["within_price_bin_auc"] = {
        "independent": _within_price_auc(p_ind),
        "crossfit_adjusted": _within_price_auc(p_adj),
    }

    # Manufactured-edge bets: EV flips sign from the price-conditioned remap.
    ev_ind = p_ind * odds - 1.0
    ev_adj = p_adj * odds - 1.0
    man = ok & (ev_adj > 0) & (ev_ind <= 0)
    band = (odds >= 2.0) & (odds <= 4.0)
    man_band = man & band
    payload["fl_ablation"]["manufactured_edge"] = {
        "n_all": int(man.sum()),
        "n_band_2_4": int(man_band.sum()),
        "realized_ae_band_2_4": (float(y[man_band].sum()
                                       / p_adj[man_band].sum())
                                 if man_band.any() else None),
        "flat_yield_band_2_4": (float(np.mean(
            np.where(y[man_band] > 0, odds[man_band] - 1.0, -1.0)))
            if man_band.any() else None),
        "note": ("rows whose EV is positive ONLY after the F-L market "
                 "adjustment; their realized A/E and flat yield measure whether "
                 "price-conditioning manufactures phantom edge"),
    }

    # Price-perturbation: how much of a price improvement does the F-L map
    # claw back? (For a price-free prob, EV rises 1:1 with price.)
    d_up = odds[ok] * 1.05
    p_up = np.asarray(fl_full.predict(p_ind[ok], d_up), dtype=float)
    ev_up_adj = p_up * d_up - 1.0
    ev_up_ind = p_ind[ok] * d_up - 1.0
    dev_adj = ev_up_adj - ev_adj[ok]
    dev_ind = ev_up_ind - ev_ind[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = 1.0 - np.where(dev_ind != 0, dev_adj / dev_ind, np.nan)
    payload["fl_ablation"]["price_shock_offset_frac"] = {
        "median": float(np.nanmedian(offset)),
        "note": ("fraction of a +5% price improvement's EV gain absorbed by "
                 "the price-conditioned recalibration (0 = none, price-free "
                 "behaviour; 1 = fully absorbed, pure price echo)"),
    }

    # 3. Requirement 4 — extremes / support / portability.
    fl_art = ex.load_pickle(_BASE / "models" / "fl_oddsband_v3nf_calib.pkl")
    base_art = ex.load_pickle(_BASE / "models" / "catboost_won_v3nf_calib.pkl")
    payload["extremes"] = {
        "base_calibrator": ex.describe_base_calibrator(base_art),
        "fl_bands": ex.describe_fl_bands(fl_art).to_dict(orient="records"),
        "grid": ex.grid_extremes(fl_art),
        "support_hit_rate_oos": ex.support_hit_rate(fl_art, p_ind, odds),
    }
    # Portability: fit-domain (ppwap OOS) vs live-domain (best-board odds from
    # the shipped cache) band occupancy.
    cache_path = _BASE / "data" / "predictions.json"
    if cache_path.exists():
        cache = _read_json(cache_path)
        live_odds = [r.get("best_odds") for race in cache.get("races", [])
                     for r in (race.get("runners") or []) if r.get("best_odds")]
        payload["extremes"]["portability"] = ex.portability_table(
            pd.Series(odds[ok]), pd.Series(live_odds, dtype=float)
        ).to_dict(orient="records")
        payload["extremes"]["live_cache_extremes"] = ex.cache_extremes(cache)

    # Per-arm zero/one counts on the OOS frame (which lines can claim 0/1).
    payload["extremes"]["oos_zero_one_counts"] = {
        col: {"zeros": int((wf[col] == 0.0).sum()),
              "ones": int((wf[col] == 1.0).sum())}
        for col in ("p_ind", "p_adj", "p_adj_sig", "norm_ind", "norm_adj")
    }

    # 3b. F-L band method selection (extremes fix) — VALIDATION FOLDS ONLY.
    from audit.metrics_panel import per_race_logloss as _prll
    from models.calibration import normalize_within_race as _nwr

    wf["norm_adj_sig"] = _nwr(wf["p_adj_sig"].to_numpy(float),
                              wf["race_uid"].to_numpy())
    sel_arms = {}
    for col in ("norm_adj", "norm_adj_sig"):
        p_col = col.replace("norm_", "p_")
        sel_arms[col] = {
            "race_log_loss": float(_prll(wf, col).mean()),
            "runner_brier": float(np.nanmean(
                (pd.to_numeric(wf[col], errors="coerce").to_numpy(float) - y) ** 2)),
            "ece": ece_equal_freq(
                pd.to_numeric(wf[col], errors="coerce").to_numpy(float), y),
            "exact_zeros": int((wf[p_col] == 0.0).sum()),
            "exact_ones": int((wf[p_col] == 1.0).sum()),
        }
    payload["fl_band_method_selection"] = {
        "note": ("extremes fix selected on VALIDATION folds only: sigmoid F-L "
                 "bands vs production isotonic bands"),
        "arms": sel_arms,
        "decision": ("sigmoid" if sel_arms["norm_adj_sig"]["race_log_loss"]
                     <= sel_arms["norm_adj"]["race_log_loss"] + 0.002
                     else "isotonic"),
        "rationale": ("sigmoid bands emit no exact 0/1 (a 0.0 win claim is "
                      "unfalsifiable overconfidence); adopted unless "
                      "materially worse on race log-loss (+0.002 tolerance)"),
    }

    # 4. A/E by odds band for the two decision lines (validation folds).
    band = odds_band_series(wf["bet_price"])
    payload["ae_by_odds_band"] = {
        "independent": ae_table(wf, "p_ind", band).to_dict(orient="records"),
        "market_adjusted": ae_table(wf, "p_adj", band).to_dict(orient="records"),
        "market_adjusted_sigmoid_bands":
            ae_table(wf, "p_adj_sig", band).to_dict(orient="records"),
    }

    # 5. Head-to-head vs market on the validation folds (context, not verdict).
    payload["wf_head_to_head"] = {
        "independent": head_to_head_ci(wf, "p_ind", n_boot=int(args.n_boot)),
        "market_adjusted": head_to_head_ci(wf, "p_adj", n_boot=int(args.n_boot)),
    }

    _write_json(run_dir / "selection.json", _jsonable(payload))


# ── phase: final (the one-shot untouched window) ─────────────────────────────


def phase_final(run_dir: Path, args) -> None:
    from audit.final_eval import run_final_evaluation

    payload = run_final_evaluation(
        run_dir=run_dir,
        final_start=args.final_start,
        final_end=args.final_end,
        n_boot=int(args.n_boot),
        task_type=args.task_type,
    )
    _write_json(run_dir / "final_evaluation.json", _jsonable(payload))


# ── phase: report ────────────────────────────────────────────────────────────


def phase_report(run_dir: Path, args) -> None:
    from audit.report import build_report

    out = build_report(run_dir, base=_BASE, git_head=_git_head())
    print(f"report written: {out}")


PHASES = {
    "panel": phase_panel,
    "leakage": phase_leakage,
    "walkforward": phase_walkforward,
    "select": phase_select,
    "final": phase_final,
    "report": phase_report,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scripts.calibration_audit")
    ap.add_argument("--run", default="stage4", help="run name under data/audit/")
    ap.add_argument("--phase", choices=list(PHASES), default=None)
    ap.add_argument("--all", action="store_true", help="run every phase in order")
    ap.add_argument("--final-start", default=FINAL_START_DEFAULT)
    ap.add_argument("--final-end", default=FINAL_END_DEFAULT)
    ap.add_argument("--min-train-days", default=365, type=int)
    ap.add_argument("--window-days", default=30, type=int)
    ap.add_argument("--task-type", default="GPU")
    ap.add_argument("--n-boot", default=1000, type=int)
    args = ap.parse_args(argv)

    run_dir = AUDIT_ROOT / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    names = list(PHASES) if args.all else ([args.phase] if args.phase else [])
    if not names:
        ap.error("pass --phase <name> or --all")
    for name in names:
        logger.info("audit: ── phase %s ──", name)
        PHASES[name](run_dir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
