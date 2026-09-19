"""One-shot evaluation on the untouched final window (requirements 7/8/10/11).

The final window postdates every model's training data, every published
holdout, and every value-gate tuning frame. It is read exactly once, by this
module. Nothing here tunes anything: every model, calibrator and threshold
arrives frozen.

Lines evaluated on IDENTICAL eligible races (complete pre-off books):

* ``audit_independent``      — CatBoost refit on the validated WIN-only panel
                               through the day before the window (audit recipe,
                               ``race_complexity`` excluded), base-calibrated.
* ``audit_market_adjusted``  — the same line after the fold-fitted F-L
                               (isotonic bands; the production method).
* ``frozen_v3nf_independent``     — the shipped price-free artifact, base only.
* ``frozen_v3nf_market_adjusted`` — shipped artifact + shipped F-L pkl.
* ``frozen_v3_priced``       — the shipped priced CatBoost line.
* ``frozen_lgbm``            — the shipped LightGBM softmax line.
* ``market``                 — the de-vigged pre-off market itself.

Plus: subgroup calibration (odds band / field size / venue / month / region /
completeness), CLV, an EV>0 simulation with the integrity suite for the two
decision lines, and a feature-drift (PSI) gate between the audit train span and
the final window.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from audit.metrics_panel import (
    ae_table,
    clv_stats,
    completeness_series,
    field_size_series,
    head_to_head_ci,
    market_line,
    month_series,
    odds_band_series,
    wilson_interval,
)
from audit.walkforward import (
    AuditWFConfig,
    _matrix,
    _sample_weights,
    fit_temperature,
    grouped_softmax,
    _logit,
)
from models.calibration import (
    OddsBandCalibrator,
    fit_calibrator,
    normalize_within_race,
)
from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parent.parent


def _load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _apply_fl(fl, p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    out = np.array(p, dtype=float, copy=True)
    if fl is None:
        return out
    ok = np.isfinite(out) & np.isfinite(odds) & (odds > 1.0)
    if ok.any():
        recal = np.asarray(fl.predict(out[ok], odds[ok]), dtype=float)
        good = np.isfinite(recal)
        out[np.flatnonzero(ok)[good]] = recal[good]
    return out


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (same construction as models.retrain_trigger)."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 50 or cur.size < 50:
        return float("nan")
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if qs.size < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    r, _ = np.histogram(ref, bins=qs)
    c, _ = np.histogram(cur, bins=qs)
    rp = np.clip(r / max(r.sum(), 1), 1e-6, None)
    cp = np.clip(c / max(c.sum(), 1), 1e-6, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def _train_audit_final_model(train: pd.DataFrame, cfg: AuditWFConfig):
    """The walk-forward recipe, fit once on everything before the final window."""
    from catboost import CatBoostClassifier

    train = train.sort_values("race_date").reset_index(drop=True)
    X = _matrix(train, cfg.feature_cols)
    y = pd.to_numeric(train["won"], errors="coerce").fillna(0).astype(int).to_numpy()
    w = _sample_weights(train, cfg.max_sample_weight) if cfg.use_sample_weights \
        else np.ones(len(train))

    n_cal = int(len(X) * cfg.calibration_frac)
    X_core, y_core, w_core = X[:-n_cal], y[:-n_cal], w[:-n_cal]
    X_cal, y_cal = X[-n_cal:], y[-n_cal:]
    cal_df = train.iloc[-n_cal:]

    n_val = max(1, int(len(X_core) * 0.10))
    params = {
        "iterations": cfg.iterations, "loss_function": "Logloss",
        "eval_metric": "AUC", "early_stopping_rounds": cfg.early_stopping_rounds,
        "random_seed": cfg.random_seed, "verbose": False,
        "allow_writing_files": False,
        **({"task_type": "GPU", "devices": cfg.devices}
           if str(cfg.task_type).upper() == "GPU"
           else {"task_type": "CPU", "thread_count": cfg.thread_count}),
        **{k: v for k, v in (cfg.best_params or {}).items() if v is not None},
    }
    model = CatBoostClassifier(**params)
    model.fit(X_core[:-n_val], y_core[:-n_val], sample_weight=w_core[:-n_val],
              eval_set=(X_core[-n_val:], y_core[-n_val:]), verbose=False)

    raw_cal = model.predict_proba(X_cal)[:, 1]
    calibrator, method = fit_calibrator(cfg.calibration_method, raw_cal, y_cal)
    ind_cal = np.asarray(calibrator.predict(raw_cal), dtype=float)
    odds_cal = pd.to_numeric(cal_df["bet_price"], errors="coerce").to_numpy()
    fl_iso = OddsBandCalibrator.fit(ind_cal, odds_cal, y_cal,
                                    method="isotonic", min_rows=cfg.fl_min_rows)
    fl_sig = OddsBandCalibrator.fit(ind_cal, odds_cal, y_cal,
                                    method="sigmoid", min_rows=cfg.fl_min_rows)
    temperature = fit_temperature(ind_cal, y_cal, cal_df["race_uid"].to_numpy())
    return model, calibrator, method, fl_iso, fl_sig, temperature


def _score_frozen_catboost(window: pd.DataFrame, tag: str, cols: list):
    from catboost import CatBoostClassifier

    path = _BASE / "models" / f"catboost_won_{tag}.bin"
    model = CatBoostClassifier()
    model.load_model(str(path))
    calib = _load_pickle(_BASE / "models" / f"catboost_won_{tag}_calib.pkl")
    X = _matrix(window, cols)
    raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    ind = np.asarray(calib.predict(raw), dtype=float) if calib is not None else raw
    return ind


def _score_frozen_lgbm(window: pd.DataFrame) -> np.ndarray | None:
    try:
        from features.lgbm_adapter import build_lgbm_matrix
        from models.lgbm_softmax import LGBMSoftmaxModel, softmax_by_race
    except Exception as exc:  # noqa: BLE001
        logger.warning("final_eval: lightgbm unavailable (%s)", exc)
        return None
    path = _BASE / "models" / "lgbm_won_v3.txt"
    if not path.exists():
        return None
    model = LGBMSoftmaxModel.load(path)
    X, _, _ = build_lgbm_matrix(window, inference=True)
    if len(X) != len(window):
        logger.warning("final_eval: lgbm matrix mismatch (%d vs %d)",
                       len(X), len(window))
        return None
    raw = model.predict_raw(X)
    return softmax_by_race(np.asarray(raw, dtype=float),
                           window["race_uid"].to_numpy())


def run_final_evaluation(*, run_dir: Path, final_start: str, final_end: str,
                         n_boot: int = 1000, task_type: str = "GPU") -> dict:
    panel = pd.read_parquet(run_dir / "panel.parquet")
    rd = pd.to_datetime(panel["race_date"], utc=True, errors="coerce")
    start = pd.to_datetime(final_start, utc=True)
    end = pd.to_datetime(final_end, utc=True) + pd.Timedelta(days=1)

    train = panel.loc[rd < start].copy()
    window = panel.loc[(rd >= start) & (rd < end)].copy().reset_index(drop=True)
    if window.empty:
        raise SystemExit(f"final window {final_start}..{final_end} is empty")

    meta = json.loads((_BASE / "models" / "catboost_v3nf_meta.json")
                      .read_text(encoding="utf-8"))
    best = dict(meta.get("targets", {}).get("won", {}).get("best_params") or {})
    # race_complexity is market-derived and is now stripped from the price-free
    # whitelist at source (models/features.MARKET_DERIVED_FEATURE_COLS, step 09
    # audit F3); this local exclusion is kept only so the list is explicit here.
    cols = [c for c in PRICE_FREE_FEATURE_COLS if c != "race_complexity"]
    cfg = AuditWFConfig(feature_cols=cols, best_params=best, task_type=task_type)

    logger.info("final_eval: training audit final model on %d rows (< %s)",
                len(train), final_start)
    model, calibrator, method, fl_iso, fl_sig, temperature = \
        _train_audit_final_model(train, cfg)

    odds = pd.to_numeric(window["bet_price"], errors="coerce").to_numpy(float)
    rid = window["race_uid"].to_numpy()

    lines: dict[str, np.ndarray] = {}
    Xw = _matrix(window, cfg.feature_cols)
    raw = np.asarray(model.predict_proba(Xw)[:, 1], dtype=float)
    lines["audit_independent"] = np.asarray(calibrator.predict(raw), dtype=float)
    lines["audit_market_adjusted"] = _apply_fl(fl_iso, lines["audit_independent"], odds)
    lines["audit_market_adjusted_sigmoid"] = _apply_fl(
        fl_sig, lines["audit_independent"], odds)
    lines["audit_grouped_temperature"] = grouped_softmax(
        _logit(lines["audit_independent"]), rid, temperature=temperature)

    lines["frozen_v3nf_independent"] = _score_frozen_catboost(
        window, "v3nf", list(meta.get("feature_cols") or PRICE_FREE_FEATURE_COLS))
    # The CURRENT shipped F-L artifact (post-Stage-4 this is the sigmoid-band
    # refit; its meta sidecar names the method), plus — when the Stage-4 backup
    # of the original isotonic artifact exists — that prior artifact too, so a
    # re-run reproduces both rows of the report.
    fl_art = _load_pickle(_BASE / "models" / "fl_oddsband_v3nf_calib.pkl")
    fl_meta_p = _BASE / "models" / "fl_oddsband_v3nf_calib_meta.json"
    fl_method = "isotonic"
    if fl_meta_p.exists():
        try:
            fl_method = json.loads(
                fl_meta_p.read_text(encoding="utf-8")).get("method", "isotonic")
        except Exception:  # noqa: BLE001
            pass
    if fl_method == "sigmoid":
        lines["frozen_v3nf_market_adjusted_sigmoid_refit"] = _apply_fl(
            fl_art, lines["frozen_v3nf_independent"], odds)
    else:
        lines["frozen_v3nf_market_adjusted"] = _apply_fl(
            fl_art, lines["frozen_v3nf_independent"], odds)
    backup = _BASE / "data" / "backups" / \
        "fl_oddsband_v3nf_calib_isotonic_20260617.pkl"
    if backup.exists() and "frozen_v3nf_market_adjusted" not in lines:
        lines["frozen_v3nf_market_adjusted"] = _apply_fl(
            _load_pickle(backup), lines["frozen_v3nf_independent"], odds)

    meta_v3 = json.loads((_BASE / "models" / "catboost_v3_meta.json")
                         .read_text(encoding="utf-8"))
    lines["frozen_v3_priced"] = _score_frozen_catboost(
        window, "v3", list(meta_v3.get("feature_cols") or FEATURE_COLS))

    lgbm = _score_frozen_lgbm(window)
    if lgbm is not None:
        lines["frozen_lgbm"] = lgbm

    # ── identical eligible races (requirement 10) ─────────────────────────────
    mkt = market_line(window)
    elig = np.isfinite(mkt) & np.isfinite(odds) & (odds > 1.0)
    for arr in lines.values():
        elig &= np.isfinite(np.asarray(arr, dtype=float))
    keep_races = pd.Series(elig).groupby(pd.Series(rid)).transform("all")
    common = window.loc[keep_races.to_numpy()].copy().reset_index(drop=True)
    logger.info("final_eval: %d/%d rows on identical eligible races",
                len(common), len(window))
    for name, arr in list(lines.items()):
        common[name] = np.asarray(arr, dtype=float)[keep_races.to_numpy()]

    payload: dict = {
        "window": {"start": final_start, "end": final_end},
        "n_window_rows": int(len(window)),
        "n_common_rows": int(len(common)),
        "n_common_races": int(common["race_uid"].nunique()),
        "audit_final_model": {
            "train_rows": int(len(train)),
            "train_max_date": str(pd.to_datetime(train["race_date"]).max()),
            "calibration_method": method,
            "temperature": float(temperature),
            "feature_count": len(cfg.feature_cols),
            "excluded_features": ["race_complexity"],
        },
        "head_to_head": {},
        "subgroups": {},
        "clv": {},
        "ev_simulation": {},
        "drift": {},
    }

    # ── head-to-head vs market with CIs, every line, identical races ─────────
    for name in lines:
        payload["head_to_head"][name] = head_to_head_ci(
            common, name, n_boot=n_boot)

    # ── market de-vig method baselines (the fair benchmark for the
    # market-adjusted lines: a price-conditioned recalibration is itself a
    # margin-removal correction, so it must be read against the BEST de-vig,
    # not only proportional) ─────────────────────────────────────────────────
    from models.devig import devig
    from audit.metrics_panel import per_race_logloss

    bl = {}
    for method in ("proportional", "power", "shin"):
        mk = np.asarray(devig(pd.to_numeric(common["bet_price"],
                                            errors="coerce"),
                              common["race_uid"], method=method), float)
        tmp = common.assign(_m=mk)
        tmp = tmp[np.isfinite(mk)]
        bl[method] = {"n_races": int(tmp["race_uid"].nunique()),
                      "race_log_loss": float(per_race_logloss(tmp, "_m").mean())}
    payload["market_devig_method_baselines"] = {
        **bl,
        "note": ("the F-L market-adjusted lines must be read against the BEST "
                 "de-vig baseline, not proportional: price-conditioned "
                 "recalibration is itself a margin-removal correction"),
    }

    # ── subgroup calibration for the decision lines ───────────────────────────
    groups = {
        "odds_band": odds_band_series(common["bet_price"]),
        "field_size": field_size_series(common.get(
            "field_size_panel", pd.Series(np.nan, index=common.index))),
        "month": month_series(common["race_date"]),
        "venue_top12": common["venue"].where(
            common["venue"].isin(common["venue"].value_counts().head(12).index),
            other="OTHER") if "venue" in common.columns else None,
        "region": common.get("region"),
        "data_completeness": completeness_series(
            common.get("data_completeness"), common.index),
    }
    for gname, series in groups.items():
        if series is None:
            continue
        payload["subgroups"][gname] = {
            "audit_independent": ae_table(common, "audit_independent",
                                          series).to_dict(orient="records"),
            "audit_market_adjusted": ae_table(common, "audit_market_adjusted",
                                              series).to_dict(orient="records"),
            "market_note": "market A/E == 1 by construction after de-vig",
        }

    # ── CLV (all common rows; and on each decision line's EV>0 picks) ────────
    payload["clv"]["all_common_rows"] = clv_stats(common["bet_price"],
                                                  common["close_price"])
    from backtest.holdout import simulate_ev_bets
    from backtest.integrity import run_all_integrity_checks

    for name in ("audit_independent", "audit_market_adjusted",
                 "frozen_v3nf_market_adjusted",
                 "frozen_v3nf_market_adjusted_sigmoid_refit"):
        if name not in common.columns:
            continue
        norm = normalize_within_race(common[name].to_numpy(float),
                                     common["race_uid"].to_numpy())
        ledger = simulate_ev_bets(common, norm)
        n = len(ledger)
        roi = (float(ledger["profit"].sum() / ledger["stake"].sum())
               if n else None)
        wins = int(ledger["won"].sum()) if n else 0
        lo, hi = wilson_interval(wins, n) if n else (None, None)
        clv_led = clv_stats(ledger["bet_price"], ledger["close_price"]) if n else {"n": 0}
        integ = run_all_integrity_checks(
            ledger, train_cutoff=pd.to_datetime(train["race_date"]).max()
            .tz_localize(None))
        payload["ev_simulation"][name] = {
            "n_bets": n, "roi": roi,
            "strike_rate": float(ledger["won"].mean()) if n else None,
            "strike_rate_ci95": (lo, hi),
            "clv": clv_led,
            "integrity": integ,
        }

    # ── drift gate: PSI train-span vs final window on the audit features ─────
    psi_rows = []
    for c in cfg.feature_cols:
        if c not in panel.columns:
            continue
        ref = pd.to_numeric(train[c], errors="coerce").to_numpy(float)
        cur = pd.to_numeric(window[c], errors="coerce").to_numpy(float)
        val = _psi(ref, cur)
        if np.isfinite(val):
            psi_rows.append({"feature": c, "psi": round(val, 4)})
    psi_rows.sort(key=lambda r: -r["psi"])
    worst = psi_rows[0]["psi"] if psi_rows else float("nan")
    payload["drift"] = {
        "psi_by_feature": psi_rows,
        "worst_psi": worst,
        "gate": "OK" if worst < 0.2 else ("WARN" if worst < 0.3 else "FAIL"),
        "note": "PSI < 0.1 stable, 0.1-0.2 moderate, > 0.2 major shift "
                "(models.retrain_trigger convention)",
        "monthly_race_logloss": {},
    }
    # Monthly stability of the decision line vs the market.
    mon = month_series(common["race_date"])
    for m in sorted(mon.dropna().unique()):
        sub = common[mon == m]
        if sub["race_uid"].nunique() < 30:
            continue
        h_ind = head_to_head_ci(sub, "audit_independent", n_boot=200)
        payload["drift"]["monthly_race_logloss"][m] = {
            "n_races": h_ind.get("n_races"),
            "model_log_loss": h_ind.get("model_log_loss"),
            "market_log_loss": h_ind.get("market_log_loss"),
            "delta": h_ind.get("logloss_delta_market_minus_model"),
        }
    return payload
