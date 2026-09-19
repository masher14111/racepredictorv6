"""Train the v3 LightGBM grouped-softmax win model on a FRESH out-of-sample window.

This is the headline retrain of the v4 rebuild. racing_ingestion's v3 retrain was
blocked because its holdout was already spent; v4 carries its own labelled data to
a recent date, so we can cut a brand-new, never-touched evaluation window and ask
the only question that matters honestly: *does the model beat the de-vigged market
on log-loss?* (see ``models.head_to_head`` / ``backtest.holdout``).

What this script does
---------------------
1. Build the leak-free matrix via :func:`features.lgbm_adapter.build_lgbm_matrix`
   (market features rebuilt from a PRE-OFF price — morningwap/ppwap — never the
   finishing SP).
2. Split STRICTLY CHRONOLOGICALLY — never randomly::

       train  <  split_date  <=  validation  <=  max_date  <  holdout_start  <=  holdout_end

   The most recent ~3 weeks (``[D-20d .. D]`` by default) are reserved as an
   UNTOUCHED holdout and are never passed to ``fit``.
3. Train :class:`models.lgbm_softmax.LGBMSoftmaxModel` with the per-race softmax
   cross-entropy objective, early-stopping on the validation slice, ``num_threads=-1``.
4. Save the booster (``--output``) + its sidecar ``.meta.json``.
5. Score the holdout window and report the GO / NO-GO verdict THROUGH the shared
   holdout harness (:func:`backtest.holdout.evaluate_scored_window`): model-vs-market
   log-loss / Brier / ECE, EV>0 ROI, CLV, and the integrity suite — writing the same
   ``summary.json`` / ``ledger.csv`` the CatBoost line produces.
6. Write ``models/lgbm_v3_meta.json`` including the verdict.

Environment
-----------
``models.lgbm_softmax`` takes the **3.14-native** path (LightGBM 4.6.0 imports
cleanly on CPython 3.14), so this trainer imports it directly — no ``.venv-lgbm``
subprocess hand-off. The interpreter is recorded in the model metadata.

Leakage is the enemy
--------------------
The run log restates the train max date, split date and holdout start, and asserts
``train_max < holdout_start`` before any fitting. Market features are pre-off only.

Expected outcome: most likely the model TRACKS the market but does not beat it
(NO-GO). That is an honest, correct result — it is reported plainly and the holdout
is never tuned against to force a GO.

Usage
-----
    python -m models.train_lgbm --output models/lgbm_won_v3.txt --importance
    python -m models.train_lgbm --max-date 2026-05-22 --holdout-start 2026-05-23 \
        --model-version v3-lgbm-20260618 --output models/lgbm_won_v3.txt --importance
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from backtest.holdout import evaluate_scored_window
from features.lgbm_adapter import build_lgbm_matrix
from models.calibration import normalize_within_race
from models.lgbm_softmax import LGBMSoftmaxModel, softmax_by_race
from utils.logger import get_logger

logger = get_logger("models.train_lgbm")

_BASE = Path(__file__).resolve().parent.parent
# data/features.parquet (repo root) is a stale 5-row artefact; the real labelled
# matrix — the one backtest.data also defaults to — is data/features/training.parquet.
_DEFAULT_FEATURES = _BASE / "data" / "features" / "training.parquet"
_DEFAULT_OUTPUT = _BASE / "models" / "lgbm_won_v3.txt"
_DEFAULT_META = _BASE / "models" / "lgbm_v3_meta.json"

# Holdout = the most recent ~3 weeks: holdout_start = D - 20d → [D-20d .. D] = 21 days.
_HOLDOUT_DAYS = 20
# Validation = the most recent ~15% of TRAIN dates (the early-stopping slice).
_VAL_DATE_FRAC = 0.15


def _as_day(value) -> pd.Timestamp:
    """Coerce a scalar to a tz-naive, midnight-normalised day Timestamp."""
    ts = pd.to_datetime(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _day_series(race_dates) -> pd.Series:
    """Normalise a column of race dates to tz-naive midnight days (NaT dropped)."""
    s = pd.to_datetime(pd.Series(race_dates), utc=True, errors="coerce")
    return s.dt.tz_localize(None).dt.normalize()


@dataclass
class Windows:
    """The resolved chronological boundaries (all tz-naive midnight days)."""

    train_start: pd.Timestamp
    split_date: pd.Timestamp      # train < split_date <= validation
    max_date: pd.Timestamp        # last TRAIN+VAL date, inclusive
    holdout_start: pd.Timestamp   # first holdout date
    holdout_end: pd.Timestamp     # last holdout date
    data_max_date: pd.Timestamp   # max race_date present in the data


def resolve_windows(
    race_dates,
    *,
    max_date: Optional[str] = None,
    split_date: Optional[str] = None,
    holdout_start: Optional[str] = None,
    holdout_end: Optional[str] = None,
    holdout_days: int = _HOLDOUT_DAYS,
    val_date_frac: float = _VAL_DATE_FRAC,
) -> Windows:
    """Derive the train / validation / holdout day boundaries from the data.

    Resolution rules (any explicitly-passed boundary is honoured as-is):

    * ``holdout_start``: explicit, else ``max_date + 1d``, else ``D - holdout_days``.
    * ``max_date``     : explicit, else ``holdout_start - 1d``.
    * ``holdout_end``  : explicit, else ``D`` (the max race date in the data).
    * ``split_date``   : explicit, else the ``(1 - val_date_frac)`` quantile of the
      unique TRAIN+VAL days (``<= max_date``) so the most recent ~``val_date_frac``
      of those days form the early-stopping validation slice.

    where ``D`` is the maximum race date present in ``race_dates``. Guarantees
    ``train_start <= split_date <= max_date < holdout_start <= holdout_end``.
    """
    rd = _day_series(race_dates).dropna()
    if rd.empty:
        raise SystemExit("train_lgbm: no valid race_date values in the data.")
    data_max = rd.max()

    if holdout_start is not None:
        hs = _as_day(holdout_start)
    elif max_date is not None:
        hs = _as_day(max_date) + pd.Timedelta(days=1)
    else:
        hs = data_max - pd.Timedelta(days=holdout_days)

    md = _as_day(max_date) if max_date is not None else hs - pd.Timedelta(days=1)
    he = _as_day(holdout_end) if holdout_end is not None else data_max

    if md >= hs:
        raise SystemExit(
            f"train_lgbm: max_date {md.date()} must be strictly before holdout_start "
            f"{hs.date()} (the holdout must not be trained on).")

    train_val_days = np.sort(rd[rd <= md].unique())
    if train_val_days.size == 0:
        raise SystemExit(
            f"train_lgbm: no training rows on/before max_date {md.date()}.")

    if split_date is not None:
        sd = _as_day(split_date)
    else:
        idx = int(train_val_days.size * (1.0 - val_date_frac))
        idx = min(max(idx, 1), train_val_days.size - 1)  # >=1 train day, >=1 val day
        sd = pd.Timestamp(train_val_days[idx])

    return Windows(
        train_start=pd.Timestamp(train_val_days[0]),
        split_date=sd,
        max_date=md,
        holdout_start=hs,
        holdout_end=he,
        data_max_date=data_max,
    )


def _load_win_market(features_path: Path | str) -> pd.DataFrame:
    """Load the labelled matrix and keep the WIN market only.

    features/fuse.py produces one row per runner PER MARKET: a runner racing
    under both books gets a WIN row (WIN odds/terms) and a PLACE row (PLACE
    odds/terms), sharing the same non-market horse/race attributes. The win
    model needs exactly one row per runner, so PLACE rows are dropped —
    mirroring ``backtest.data.load_panel`` and ``models.train``'s CatBoost path.
    """
    df = pd.read_parquet(features_path)
    if "market_type" in df.columns:
        df = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    df["_day"] = _day_series(df["race_date"]).to_numpy()
    df = df[pd.notna(df["_day"])].copy()
    return df


def chronological_split(
    df: pd.DataFrame, w: Windows
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Slice ``df`` (which must carry a normalised ``_day`` column) by date.

    Returns ``(train_df, val_df, holdout_df, train_max)`` where ``train_max`` is the
    latest day actually present in train+val. Raises if the leakage invariant
    ``train_max < holdout_start`` does not hold.
    """
    day = df["_day"]
    train_df = df[day < w.split_date].copy()
    val_df = df[(day >= w.split_date) & (day <= w.max_date)].copy()
    holdout_df = df[(day >= w.holdout_start) & (day <= w.holdout_end)].copy()

    trainval_days = df.loc[day <= w.max_date, "_day"]
    if trainval_days.empty:
        raise SystemExit("train_lgbm: empty train+validation set after the split.")
    train_max = pd.Timestamp(trainval_days.max())

    if not (train_max < w.holdout_start):
        raise SystemExit(
            f"LEAKAGE GUARD: train max date {train_max.date()} is not strictly "
            f"before holdout start {w.holdout_start.date()}. Refusing to train.")
    return train_df, val_df, holdout_df, train_max


def _score_holdout(model: LGBMSoftmaxModel, holdout_df: pd.DataFrame,
                   feature_cols: Optional[list] = None) -> pd.DataFrame:
    """Score the holdout window: leak-free matrix → raw scores → per-race softmax.

    Returns a small frame keyed by ``(race_uid, horse_id)`` carrying the
    ``model_win_prob`` so it can be merged onto the bet panel. ``feature_cols``
    must match whatever the model was trained with (see :func:`run_training`).
    """
    h = holdout_df.reset_index(drop=True)
    X, _, race_ids = build_lgbm_matrix(h, inference=True, feature_cols=feature_cols)
    raw = model.predict_raw(X)
    prob = softmax_by_race(raw, race_ids)
    horse_id = (h["horse_id"].to_numpy() if "horse_id" in h.columns
                else np.arange(len(h)))
    return pd.DataFrame({
        "race_uid": np.asarray(race_ids),
        "horse_id": horse_id,
        "model_win_prob": np.asarray(prob, dtype=float),
    })


def _holdout_verdict(
    model: LGBMSoftmaxModel,
    holdout_df: pd.DataFrame,
    *,
    out_dir: str,
    model_summary: dict,
    train_cutoff: pd.Timestamp,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    devig_method: str,
    stake: float,
    commission: float,
    feature_cols: Optional[list] = None,
) -> dict:
    """Build the bet panel, attach model probs, and run the shared GO/NO-GO core.

    The panel (pre-off ``bet_price`` = ppwap→morningwap, closing ``close_price`` =
    odds_finish, WIN market, priced + settled rows only) is built by the SAME
    ``backtest.data.load_panel`` the CatBoost holdout uses, so the LightGBM verdict
    is directly comparable. Model probabilities are merged on ``(race_uid, horse_id)``.
    """
    if holdout_df.empty:
        raise SystemExit(
            f"train_lgbm: no holdout rows in window "
            f"{holdout_start.date()}..{holdout_end.date()}.")

    scored = _score_holdout(model, holdout_df, feature_cols=feature_cols)
    panel = load_panel(df=holdout_df, config=PanelConfig(feature_cols=()))
    if panel.empty:
        raise SystemExit(
            "train_lgbm: holdout panel is empty after price/outcome filtering.")

    panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
    missing = int(panel["model_win_prob"].isna().sum())
    if missing:
        logger.warning("train_lgbm: %d holdout panel rows had no model score "
                       "(dropped before scoring).", missing)
        panel = panel[panel["model_win_prob"].notna()].copy()
    panel["norm_prob"] = normalize_within_race(
        panel["model_win_prob"].to_numpy(), panel["race_uid"].to_numpy())

    return evaluate_scored_window(
        panel,
        out_dir=out_dir,
        model_summary=model_summary,
        train_cutoff=train_cutoff,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        devig_method=devig_method,
        stake=stake,
        commission=commission,
        settle_price_col="ppwap",
        close_price_col="odds_finish",
    )


def run_training(
    *,
    features_path: Path | str = _DEFAULT_FEATURES,
    output_path: Path | str = _DEFAULT_OUTPUT,
    meta_path: Path | str = _DEFAULT_META,
    holdout_out: Path | str,
    df: Optional[pd.DataFrame] = None,
    max_date: Optional[str] = None,
    split_date: Optional[str] = None,
    holdout_start: Optional[str] = None,
    holdout_end: Optional[str] = None,
    holdout_days: int = _HOLDOUT_DAYS,
    val_date_frac: float = _VAL_DATE_FRAC,
    model_version: Optional[str] = None,
    model_params: Optional[dict] = None,
    early_stopping_rounds: int = 80,
    verbose_eval: int = 0,
    show_importance: bool = False,
    devig_method: str = "proportional",
    stake: float = 10.0,
    commission: float = 0.02,
    feature_cols: Optional[list] = None,
) -> dict:
    """End-to-end: load → split → train → frozen holdout GO/NO-GO → write meta.

    Pass ``df`` to supply a pre-loaded matrix (tests do this); otherwise it is read
    from ``features_path``. Returns a dict with the holdout ``summary``, the written
    ``meta`` and ``meta_path``, the resolved ``windows`` and the trained ``model``.

    ``feature_cols`` overrides ``features.lgbm_adapter.FINAL_FEATURE_COLS`` (the
    default, market-assisted whitelist) — pass
    ``features.lgbm_adapter.INDEPENDENT_FEATURE_COLS`` to train the price-free
    branch (step 10's independent-vs-market-assisted comparison).
    """
    output_path = Path(output_path)
    meta_path = Path(meta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_version = model_version or f"v3-lgbm-{datetime.now():%Y%m%d_%H%M%S}"

    # ── Load (WIN market only) ────────────────────────────────────────────────
    if df is None:
        logger.info("train_lgbm: loading %s", features_path)
        df = _load_win_market(features_path)
    else:
        df = df.copy()
        if "market_type" in df.columns:
            df = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
        if "_day" not in df.columns:
            df["_day"] = _day_series(df["race_date"]).to_numpy()
        df = df[pd.notna(df["_day"])].copy()

    w = resolve_windows(
        df["_day"], max_date=max_date, split_date=split_date,
        holdout_start=holdout_start, holdout_end=holdout_end,
        holdout_days=holdout_days, val_date_frac=val_date_frac)

    train_df, val_df, holdout_df, train_max = chronological_split(df, w)

    # ── Leakage discipline: restate the boundaries in the run log ─────────────
    logger.info("=" * 72)
    logger.info("train_lgbm LEAKAGE LEDGER (assert train_max < holdout_start)")
    logger.info("  data max race date : %s", w.data_max_date.date())
    logger.info("  train window       : %s .. < %s  (%d rows, %d races)",
                w.train_start.date(), w.split_date.date(),
                len(train_df), train_df["race_uid"].nunique() if len(train_df) else 0)
    logger.info("  validation window  : %s .. %s  (%d rows, %d races)",
                w.split_date.date(), w.max_date.date(),
                len(val_df), val_df["race_uid"].nunique() if len(val_df) else 0)
    logger.info("  >>> train MAX date : %s", train_max.date())
    logger.info("  >>> split date     : %s", w.split_date.date())
    logger.info("  >>> holdout start  : %s  (UNTOUCHED, not loaded in fit)", w.holdout_start.date())
    logger.info("  holdout window     : %s .. %s  (%d rows, %d races)",
                w.holdout_start.date(), w.holdout_end.date(),
                len(holdout_df), holdout_df["race_uid"].nunique() if len(holdout_df) else 0)
    logger.info("  ASSERT train_max(%s) < holdout_start(%s) -> %s",
                train_max.date(), w.holdout_start.date(), train_max < w.holdout_start)
    logger.info("  market features    : pre-off only (morningwap/ppwap); odds_finish = CLV ref only")
    logger.info("=" * 72)

    # ── Build leak-free matrices (build_lgbm_matrix rejects post-off columns) ──
    X_tr, y_tr, rid_tr = build_lgbm_matrix(train_df, inference=False, feature_cols=feature_cols)
    X_val, y_val, rid_val = build_lgbm_matrix(val_df, inference=False, feature_cols=feature_cols)
    if len(X_tr) == 0 or len(X_val) == 0:
        raise SystemExit(
            f"train_lgbm: empty train ({len(X_tr)}) or validation ({len(X_val)}) "
            "matrix — widen the data or adjust the split.")

    # ── Train (per-race softmax, early-stop on validation, all cores) ─────────
    model = LGBMSoftmaxModel(**(model_params or {}))
    model.fit(
        X_tr, y_tr, rid_tr,
        X_val=X_val, y_val=y_val, race_ids_val=rid_val,
        race_dates=train_df["race_date"],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    model.save(output_path)

    importance_records: list[dict] = []
    if show_importance:
        imp = model.feature_importance(importance_type="gain")
        importance_records = imp.to_dict(orient="records")
        logger.info("train_lgbm: top features by gain:")
        for _, row in imp.head(20).iterrows():
            logger.info("    %-32s %10.1f  (%5.2f%%)",
                        row["feature"], row["importance"], row["importance_pct"])

    # ── Frozen holdout GO/NO-GO through the shared harness ────────────────────
    model_summary = {
        "path": str(output_path),
        "model_type": "lgbm_softmax",
        "model_version": model_version,
        "train_cutoff": str(w.max_date.date()),
        "holdout_start": str(w.holdout_start.date()),
        "leakage_verified": True,
    }
    summary = _holdout_verdict(
        model, holdout_df,
        out_dir=str(holdout_out),
        model_summary=model_summary,
        train_cutoff=w.max_date,           # integrity look-ahead guard fires
        holdout_start=w.holdout_start,
        holdout_end=w.holdout_end,
        devig_method=devig_method,
        stake=stake,
        commission=commission,
        feature_cols=feature_cols,
    )

    h2h = summary["head_to_head"]
    betting = summary["betting"]
    go = bool(summary["model_beats_market_logloss"])
    verdict = {
        "go": go,
        "model_beats_market_logloss": go,
        "model_log_loss": h2h["model"]["log_loss"],
        "market_log_loss": h2h["market"]["log_loss"],
        "model_brier": h2h["model"]["brier_runner_level"],
        "market_brier": h2h["market"]["brier_runner_level"],
        "model_ece": h2h["model"]["ece"],
        "market_ece": h2h["market"]["ece"],
        "logloss_gap_market_minus_model": h2h["gaps"]["log_loss"],
        "n_races_kept": h2h["n_races_kept"],
        "ev_roi": betting["roi"],
        "ev_n_bets": betting["n_bets"],
        "mean_clv_log": betting["mean_clv_log"],
        "clv_beat_rate": betting["clv_beat_rate"],
    }

    # ── Write models/lgbm_v3_meta.json (with the verdict) ─────────────────────
    meta = {
        "model_version": model_version,
        "model_type": "lgbm_softmax",
        "model_path": str(output_path),
        "model_meta_sidecar": str(Path(output_path).with_suffix(".meta.json")),
        "env": model.metadata.get("env"),
        "lgbm_version": model.metadata.get("lgbm_version"),
        "python_executable": model.metadata.get("python_executable"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "leakage": {
            "train_max_date": str(train_max.date()),
            "split_date": str(w.split_date.date()),
            "holdout_start": str(w.holdout_start.date()),
            "holdout_end": str(w.holdout_end.date()),
            "assert_train_max_lt_holdout_start": bool(train_max < w.holdout_start),
            "market_features": ("pre-off only (morningwap/ppwap); odds_finish used "
                                "as the CLV reference, never a feature"),
        },
        "windows": {
            "data_max_date": str(w.data_max_date.date()),
            "max_date_inclusive": str(w.max_date.date()),
            "train": {
                "start": str(w.train_start.date()),
                "end_exclusive": str(w.split_date.date()),
                "n_rows": int(len(train_df)),
                "n_races": int(train_df["race_uid"].nunique()) if len(train_df) else 0,
            },
            "validation": {
                "start": str(w.split_date.date()),
                "end": str(w.max_date.date()),
                "n_rows": int(len(val_df)),
                "n_races": int(val_df["race_uid"].nunique()) if len(val_df) else 0,
            },
            "holdout": {
                "start": str(w.holdout_start.date()),
                "end": str(w.holdout_end.date()),
                "n_rows": int(len(holdout_df)),
                "n_races": int(holdout_df["race_uid"].nunique()) if len(holdout_df) else 0,
            },
        },
        "n_features": model.metadata.get("n_features"),
        "feature_name": model.feature_name,
        "best_iteration": model.metadata.get("best_iteration"),
        "model_params": model.metadata.get("params"),
        "holdout_out_dir": str(holdout_out),
        "holdout_summary": summary,
        "verdict": verdict,
    }
    if show_importance:
        meta["feature_importance"] = importance_records

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("train_lgbm: wrote %s", meta_path)

    verdict_word = "GO  (model beats market)" if go else "NO-GO (model does not beat market)"
    print(f"\n=== v3 LightGBM verdict: {verdict_word} ===")
    print(f"  model log-loss {verdict['model_log_loss']} vs market "
          f"{verdict['market_log_loss']}  over {verdict['n_races_kept']} races")
    print(f"  EV>0 bets {verdict['ev_n_bets']}  ROI {verdict['ev_roi']}  "
          f"mean CLV {verdict['mean_clv_log']}")
    print(f"  meta: {meta_path}")

    return {
        "summary": summary,
        "verdict": verdict,
        "meta": meta,
        "meta_path": str(meta_path),
        "windows": w,
        "model": model,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="models.train_lgbm",
        description="Train v3 LightGBM softmax win model + honest holdout GO/NO-GO verdict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features", default=str(_DEFAULT_FEATURES),
                   help="labelled feature parquet (default: data/features/training.parquet)")
    p.add_argument("--max-date", default=None,
                   help="YYYY-MM-DD last TRAIN+VAL date (inclusive); default = holdout_start - 1d")
    p.add_argument("--split-date", default=None,
                   help="YYYY-MM-DD train/validation boundary; default = auto (last ~15%% of train dates)")
    p.add_argument("--holdout-start", default=None,
                   help="YYYY-MM-DD first holdout date; default = D - 20d (most recent ~3 weeks)")
    p.add_argument("--holdout-end", default=None,
                   help="YYYY-MM-DD last holdout date; default = max race date in data")
    p.add_argument("--holdout-days", type=int, default=_HOLDOUT_DAYS,
                   help="holdout length in days when --holdout-start is auto (default 20 → ~3 weeks)")
    p.add_argument("--val-date-frac", type=float, default=_VAL_DATE_FRAC,
                   help="fraction of train dates used as the early-stopping slice (default 0.15)")
    p.add_argument("--model-version", default=None, help="version string (default v3-lgbm-<stamp>)")
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT), help="booster output path (.txt)")
    p.add_argument("--meta", default=str(_DEFAULT_META),
                   help="verdict metadata json (default models/lgbm_v3_meta.json)")
    p.add_argument("--holdout-out", default=None,
                   help="holdout artefact dir (default data/backtests/holdout_lgbm_<stamp>)")
    p.add_argument("--devig-method", default="proportional",
                   choices=["proportional", "power", "shin"])
    p.add_argument("--stake", type=float, default=10.0)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--early-stopping-rounds", type=int, default=80)
    p.add_argument("--n-estimators", type=int, default=None,
                   help="override booster rounds (default: model default 1500)")
    p.add_argument("--seed", type=int, default=42, help="LightGBM seed (default 42)")
    p.add_argument("--importance", action="store_true", help="log + record top feature importances")
    p.add_argument("--verbose-eval", type=int, default=0, help="log every N boosting rounds (0=silent)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    holdout_out = args.holdout_out or str(_BASE / "data" / "backtests" / f"holdout_lgbm_{stamp}")

    model_params: dict = {"seed": args.seed}
    if args.n_estimators is not None:
        model_params["n_estimators"] = args.n_estimators

    run_training(
        features_path=args.features,
        output_path=args.output,
        meta_path=args.meta,
        holdout_out=holdout_out,
        max_date=args.max_date,
        split_date=args.split_date,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        holdout_days=args.holdout_days,
        val_date_frac=args.val_date_frac,
        model_version=args.model_version,
        model_params=model_params,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
        show_importance=args.importance,
        devig_method=args.devig_method,
        stake=args.stake,
        commission=args.commission,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
