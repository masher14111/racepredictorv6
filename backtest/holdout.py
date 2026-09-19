"""Phase-4 honest holdout backtest — the GO / NO-GO harness for a frozen model.

This is v4's port of racing_ingestion's ``backtest/phase4_holdout_backtest.py``:
the one evaluation that cannot flatter itself. It loads a **frozen** model (one
trained strictly before a holdout window), scores every runner in the window,
and answers the only question that matters — *does the model beat the de-vigged
market on log-loss?* — before any P&L is even computed. Everything downstream
(EV-positive simulation, CLV, the integrity suite) is supporting evidence for
that single ``model_beats_market_logloss`` verdict.

Price hygiene (the whole point)
-------------------------------
* The **decision and settlement price** is a PRE-OFF price — ``ppwap`` (Betfair
  pre-play weighted-average), falling back to ``morningwap``. That is the price
  ``backtest.data.load_panel`` resolves into ``bet_price``; EV is computed on it
  and bets settle at it, so we never settle a bet at a price that postdates the
  decision. THIS IS NOT THE SAME CLAIM AS "FILLABLE": ``ppwap`` is a
  volume-weighted AVERAGE over the whole pre-off period, not a quote that stood
  at one instant a real order could have matched — see
  ``execution.snapshots.SnapshotStore``, the append-only *captured* quote
  series, for that. This harness's price is a chronologically-safe
  reconstruction, i.e. a diagnostic proxy for execution-timing purposes; only a
  captured snapshot series is genuine executable-price evidence.
* ``odds_finish`` (Betfair SP) is the **closing line** — used ONLY as the CLV
  reference. It is never a feature and never the decision price. (The training
  matrix's ``implied_prob`` is ``1/morningwap`` — pre-off — since ``3eca232``;
  verified empirically in the Stage-4 audit.)

Leakage guard
-------------
The model's training cutoff is read from its metadata; if the holdout starts on
or before that cutoff the run HARD-FAILS, because the model could have trained on
the very races it is being asked to bet into. When the metadata records no cutoff
the guard cannot fire — it degrades to a loud WARNING and records
``leakage_verified: false`` in the summary rather than silently asserting safety.

Usage
-----
    python -m backtest.holdout --model models/catboost_won_v3nf.bin \
        --holdout-start 2026-05-22 --holdout-end 2026-06-12 \
        --price-col ppwap --close-col odds_finish \
        --out data/backtests/holdout_<stamp>

Depends on :mod:`models.head_to_head` (Prompt 2) and
:mod:`backtest.integrity` (Prompt 3).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from backtest.integrity import run_all_integrity_checks
from models.calibration import normalize_within_race
from models.head_to_head import head_to_head
from utils.logger import get_logger

logger = get_logger("backtest.holdout")

# Frozen-model filename grammar: catboost_<target>_<tag>.bin (e.g.
# catboost_won_v3nf.bin → target="won", tag="v3nf").
_MODEL_RE = re.compile(r"^catboost_(?P<target>.+)_(?P<tag>[^_]+)\.bin$")

# Metadata keys that may carry the training cutoff date, in priority order. The
# current v3/v3nf meta carries none of these (cutoff unknown → guard degrades to
# a warning); a future retrain (Prompt 7) records one and the guard goes live.
_CUTOFF_KEYS: Sequence[str] = (
    "train_cutoff",
    "train_max_date",
    "max_train_date",
    "train_end",
    "holdout_start",
)

_DEFAULT_STAKE = 10.0
_DEFAULT_COMMISSION = 0.02


# ── frozen model ──────────────────────────────────────────────────────────────


@dataclass
class FrozenModel:
    """A trained CatBoost win model plus its calibration stack and metadata."""

    model: object                       # CatBoostClassifier (or any predict_proba)
    calibrator: Optional[object]        # base per-prob calibrator (isotonic/sigmoid)
    fl_calibrator: Optional[object]     # favourite-longshot OddsBandCalibrator
    feature_cols: list[str]
    target: str
    tag: str
    meta: dict
    train_cutoff: Optional[pd.Timestamp]


def _load_pickle(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("holdout: could not read %s (%s)", path.name, exc)
        return None


def _train_cutoff_from_meta(meta: dict) -> Optional[pd.Timestamp]:
    """Pull the model's training cutoff out of metadata, or None if unrecorded.

    Accepts a flat ``train_cutoff``-style key or a nested
    ``data_window.max_race_date`` (racing_ingestion's shape). Returns the cutoff
    as a tz-naive day; unparseable values are treated as absent.
    """
    candidates = [meta.get(k) for k in _CUTOFF_KEYS]
    window = meta.get("data_window")
    if isinstance(window, dict):
        candidates += [window.get("max_race_date"), window.get("train_end")]

    for raw in candidates:
        if raw in (None, ""):
            continue
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.notna(ts):
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return None


def load_frozen_model(model_path: str) -> FrozenModel:
    """Load the frozen win model, its calibrators, feature list and cutoff.

    ``model_path`` points at the ``.bin``; the sibling
    ``catboost_<tag>_meta.json``, ``catboost_<target>_<tag>_calib.pkl`` and
    ``fl_oddsband_<tag>_calib.pkl`` are resolved alongside it (each optional bar
    the model and meta).
    """
    from catboost import CatBoostClassifier

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")

    m = _MODEL_RE.match(path.name)
    if not m:
        raise ValueError(
            f"model filename {path.name!r} is not catboost_<target>_<tag>.bin")
    target, tag = m.group("target"), m.group("tag")
    model_dir = path.parent

    model = CatBoostClassifier()
    model.load_model(str(path))

    meta_path = model_dir / f"catboost_{tag}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    feature_cols = list(meta.get("feature_cols") or [])
    if not feature_cols:
        raise ValueError(f"{meta_path.name} has no feature_cols")

    calibrator = _load_pickle(model_dir / f"catboost_{target}_{tag}_calib.pkl")
    fl_calibrator = _load_pickle(model_dir / f"fl_oddsband_{tag}_calib.pkl")
    cutoff = _train_cutoff_from_meta(meta)

    logger.info(
        "holdout: loaded frozen model %s (target=%s tag=%s, %d features, "
        "calibrator=%s, fl=%s, train_cutoff=%s)",
        path.name, target, tag, len(feature_cols),
        type(calibrator).__name__ if calibrator else None,
        type(fl_calibrator).__name__ if fl_calibrator else None,
        cutoff.date() if cutoff is not None else "UNRECORDED",
    )
    return FrozenModel(model, calibrator, fl_calibrator, feature_cols,
                       target, tag, meta, cutoff)


# ── scoring ───────────────────────────────────────────────────────────────────


def _feature_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Column-align to the model's feature order, NaN-impute, float — the same
    matrix shape the trainer/predictor builds. Absent columns become all-NaN
    (CatBoost handles NaN natively)."""
    frame = pd.DataFrame(index=df.index)
    for c in cols:
        frame[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return frame.astype(float).to_numpy()


def score(frozen: FrozenModel, df: pd.DataFrame, price_col: str = "bet_price") -> np.ndarray:
    """Per-runner calibrated win probability for ``df`` (NOT yet race-normalised).

    Mirrors the live value layer exactly: raw CatBoost ``P(win)`` → base
    calibrator → favourite-longshot recalibrator keyed on the PRE-OFF execution
    price (``price_col``). Rows with a missing/≤1 price keep the
    market-independent calibrated probability (never NaN), matching
    ``models.predictor._apply_fl_recalibration``.
    """
    X = _feature_matrix(df, frozen.feature_cols)
    raw = np.asarray(frozen.model.predict_proba(X)[:, 1], dtype=float)
    p = (np.asarray(frozen.calibrator.predict(raw), dtype=float)
         if frozen.calibrator is not None else raw)

    if frozen.fl_calibrator is not None and price_col in df.columns:
        odds = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(p) & np.isfinite(odds) & (odds > 1.0)
        if usable.any():
            recal = np.asarray(
                frozen.fl_calibrator.predict(p[usable], odds[usable]), dtype=float)
            good = np.isfinite(recal)
            p = p.copy()
            p[np.flatnonzero(usable)[good]] = recal[good]
    return p


# ── EV-positive simulation ────────────────────────────────────────────────────


def simulate_ev_bets(
    df: pd.DataFrame,
    norm_prob: np.ndarray,
    *,
    stake: float = _DEFAULT_STAKE,
    commission: float = _DEFAULT_COMMISSION,
    price_col: str = "bet_price",
    close_col: str = "close_price",
) -> pd.DataFrame:
    """Flat-stake EV>0 ledger settled at the PRE-OFF price.

    For each runner ``EV = norm_prob * price − 1``; a bet is placed iff ``EV > 0``
    and the price is valid. Winners return ``stake·(price−1)`` less ``commission``
    on the winnings; losers forfeit the stake. ``clv_log = log(price / close)`` is
    the closing-line value against ``close_col`` (Betfair SP) — reference only.
    """
    out = df.copy()
    out["norm_prob"] = np.asarray(norm_prob, dtype=float)
    price = pd.to_numeric(out[price_col], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out.get(close_col), errors="coerce").to_numpy(dtype=float) \
        if close_col in out.columns else np.full(len(out), np.nan)

    ev = out["norm_prob"].to_numpy() * price - 1.0
    bet = np.isfinite(ev) & np.isfinite(price) & (price > 1.0) & (ev > 0.0)

    sel = out.loc[bet].copy()
    sel_price = price[bet]
    sel_close = close[bet]
    won = pd.to_numeric(sel["won"], errors="coerce").fillna(0).to_numpy(dtype=float)

    # tz-naive dates so the integrity suite's cutoff comparison (a tz-naive
    # Timestamp) never trips on a tz-aware vs tz-naive mismatch.
    race_date = pd.to_datetime(sel.get("race_date"), errors="coerce")
    if getattr(race_date.dt, "tz", None) is not None:
        race_date = race_date.dt.tz_localize(None)

    gross = stake * (sel_price - 1.0)
    profit = np.where(won > 0, gross * (1.0 - commission), -float(stake))

    with np.errstate(divide="ignore", invalid="ignore"):
        clv_log = np.log(sel_price / sel_close)
    clv_log = np.where(np.isfinite(clv_log), clv_log, np.nan)

    ledger = pd.DataFrame({
        "race_date": race_date.to_numpy(),
        "race_uid": sel["race_uid"].to_numpy() if "race_uid" in sel else None,
        "venue": sel["venue"].to_numpy() if "venue" in sel else None,
        "horse_name": sel["horse_name"].to_numpy() if "horse_name" in sel else None,
        "won": won.astype(int),
        "norm_prob": sel["norm_prob"].to_numpy(),
        "ev": ev[bet],
        "stake": float(stake),
        "bet_price": sel_price,
        "close_price": sel_close,
        "clv_log": clv_log,
        "profit": profit,
    })
    return ledger.reset_index(drop=True)


# ── orchestration ─────────────────────────────────────────────────────────────


def _roi(ledger: pd.DataFrame) -> Optional[float]:
    staked = float(ledger["stake"].sum())
    return float(ledger["profit"].sum() / staked) if staked > 0 else None


def evaluate_scored_window(
    window: pd.DataFrame,
    *,
    out_dir: str,
    model_summary: dict,
    train_cutoff: Optional[pd.Timestamp],
    holdout_start,
    holdout_end,
    devig_method: str = "proportional",
    stake: float = _DEFAULT_STAKE,
    commission: float = _DEFAULT_COMMISSION,
    settle_price_col: str = "ppwap",
    close_price_col: str = "odds_finish",
) -> dict:
    """Model-agnostic core of the holdout harness: score in, GO/NO-GO out.

    Given a ``window`` that already carries the per-runner model probability
    (``model_win_prob``), its within-race normalisation (``norm_prob``), the
    pre-off decision price (``bet_price``), the closing line (``close_price``),
    plus ``race_uid`` / ``won`` / ``race_date``, this runs the head-to-head GO
    gate, the EV>0 simulation, the integrity suite, writes ``summary.json`` /
    ``ledger.csv`` / ``odds_band_calibration.csv`` and returns the summary dict.

    Both the frozen-CatBoost path (:func:`run_holdout`) and the LightGBM v3
    trainer (``models.train_lgbm``) score their window and then funnel through
    here, so the single ``model_beats_market_logloss`` verdict — and the exact
    summary shape the UI reads — is produced in one place.

    Args:
        window:         One row per scored runner (see required columns above).
        out_dir:        Directory for the persisted artefacts (created if absent).
        model_summary:  Arbitrary, JSON-safe description of the model; copied
                        verbatim into ``summary["model"]`` (the CatBoost and
                        LightGBM lines carry different keys here).
        train_cutoff:   Model training cutoff; enables the integrity look-ahead
                        guard (first bet must fall strictly after it).
        holdout_start / holdout_end:
                        Window bounds, recorded in ``summary["window"]``.
        devig_method:   ``"proportional"`` / ``"power"`` / ``"shin"`` for the
                        de-vigged market benchmark.
        stake, commission:
                        EV>0 flat-stake simulation parameters.
        settle_price_col / close_price_col:
                        Source-column labels recorded in the betting summary (the
                        execution price already lives in ``window['bet_price']``).
    """
    # ── Head-to-head vs the de-vigged PRE-OFF market (the GO gate) ─────────────
    h2h = head_to_head(
        window, prob_col="model_win_prob", odds_col="bet_price",
        race_id_col="race_uid", label_col="won", devig_method=devig_method)
    band_table = h2h.pop("odds_band_table")
    n_races = int(window["race_uid"].nunique())
    logger.info("holdout: model log-loss %.5f vs market %.5f → %s",
                h2h["model"]["log_loss"], h2h["market"]["log_loss"],
                "GO" if h2h["model_beats_market_logloss"] else "NO-GO")

    # ── EV>0 flat-bet simulation settled at the pre-off price ──────────────────
    ledger = simulate_ev_bets(window, window["norm_prob"].to_numpy(),
                              stake=stake, commission=commission)
    mean_clv = float(np.nanmean(ledger["clv_log"])) if len(ledger) else None
    clv_beat = float(np.nanmean(ledger["clv_log"].to_numpy() > 0)) if len(ledger) else None
    logger.info("holdout: %d EV>0 bets, ROI=%s, strike=%s, mean CLV=%s",
                len(ledger), _roi(ledger),
                float(ledger["won"].mean()) if len(ledger) else None, mean_clv)

    # ── Integrity suite over the ledger ────────────────────────────────────────
    integrity = run_all_integrity_checks(ledger, train_cutoff=train_cutoff)
    for chk in integrity:
        logger.info("holdout: integrity [%s] %s", chk["status"], chk["name"])

    # ── Persist ────────────────────────────────────────────────────────────────
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(holdout_start)
    end = pd.to_datetime(holdout_end)
    summary = {
        "model": dict(model_summary),
        "window": {"start": str(start.date()), "end": str(end.date())},
        "n_runners": int(len(window)),
        "n_races": n_races,
        "devig_method": devig_method,
        "head_to_head": h2h,
        "model_beats_market_logloss": bool(h2h["model_beats_market_logloss"]),
        "betting": {
            "stake": stake,
            "commission": commission,
            "settle_price_col": settle_price_col,
            "close_price_col": close_price_col,
            "n_bets": int(len(ledger)),
            "total_staked": float(ledger["stake"].sum()) if len(ledger) else 0.0,
            "total_pnl": float(ledger["profit"].sum()) if len(ledger) else 0.0,
            "roi": _roi(ledger),
            "strike_rate": float(ledger["won"].mean()) if len(ledger) else None,
            "mean_clv_log": mean_clv,
            "clv_beat_rate": clv_beat,
        },
        "integrity": integrity,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    ledger.to_csv(out / "ledger.csv", index=False)
    band_table.to_csv(out / "odds_band_calibration.csv", index=False)
    logger.info("holdout: wrote %s", out / "summary.json")

    # Console summary.
    print("\n=== Holdout head-to-head (the GO gate) ===")
    print(json.dumps({
        "window": summary["window"],
        "n_races_kept": h2h["n_races_kept"],
        "model_log_loss": h2h["model"]["log_loss"],
        "market_log_loss": h2h["market"]["log_loss"],
        "model_beats_market_logloss": summary["model_beats_market_logloss"],
    }, indent=2, default=str))
    print("\n=== EV>0 betting (settled pre-off) ===")
    print(json.dumps(summary["betting"], indent=2, default=str))
    print("\n=== Integrity ===")
    for chk in integrity:
        print(f"  [{chk['status']:4}] {chk['name']}: {chk['message']}")
    print("\nArtifacts:", out / "summary.json", "|", out / "ledger.csv")
    return summary


def run_holdout(
    *,
    model_path: str,
    holdout_start: str,
    holdout_end: str,
    out_dir: str,
    price_col: str = "ppwap",
    close_col: str = "odds_finish",
    features_path: Optional[str] = None,
    devig_method: str = "proportional",
    stake: float = _DEFAULT_STAKE,
    commission: float = _DEFAULT_COMMISSION,
    panel_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Full holdout evaluation; writes summary.json + ledger.csv and returns the
    summary dict."""
    frozen = load_frozen_model(model_path)

    start = pd.to_datetime(holdout_start).tz_localize(None)
    end = pd.to_datetime(holdout_end).tz_localize(None)

    # ── 1. Leakage guard ──────────────────────────────────────────────────────
    leakage_verified = frozen.train_cutoff is not None
    if leakage_verified and start <= frozen.train_cutoff:
        raise SystemExit(
            f"LEAKAGE GUARD: holdout start {start.date()} is on/before the model's "
            f"train cutoff {frozen.train_cutoff.date()}. The model may have trained "
            "on rows it would bet into. Refusing to run.")
    if not leakage_verified:
        logger.warning(
            "holdout: model metadata records NO train cutoff — leakage CANNOT be "
            "verified. Treat results with suspicion (leakage_verified=false).")

    # ── 2. Load panel, score, restrict to the window ──────────────────────────
    cfg = PanelConfig(price_col=price_col, close_col=close_col,
                      feature_cols=frozen.feature_cols)
    panel = load_panel(df=panel_df, path=features_path, config=cfg)

    rd = pd.to_datetime(panel["race_date"], utc=True, errors="coerce").dt.tz_localize(None)
    window = panel.loc[(rd >= start) & (rd <= end)].copy().reset_index(drop=True)
    if window.empty:
        raise SystemExit(
            f"No runners in holdout window {start.date()}..{end.date()} "
            f"(panel spans {rd.min()}..{rd.max()}).")

    cal_prob = score(frozen, window, price_col="bet_price")
    window["model_win_prob"] = cal_prob
    window["norm_prob"] = normalize_within_race(cal_prob, window["race_uid"].to_numpy())
    logger.info("holdout: scored %d runners / %d races in window",
                len(window), int(window["race_uid"].nunique()))

    # ── 3-6. Shared GO gate → EV sim → integrity → summary.json/ledger.csv ─────
    return evaluate_scored_window(
        window,
        out_dir=out_dir,
        model_summary={
            "path": str(model_path),
            "target": frozen.target,
            "tag": frozen.tag,
            "train_cutoff": (str(frozen.train_cutoff.date())
                             if frozen.train_cutoff is not None else None),
            "leakage_verified": leakage_verified,
        },
        train_cutoff=frozen.train_cutoff,
        holdout_start=start,
        holdout_end=end,
        devig_method=devig_method,
        stake=stake,
        commission=commission,
        settle_price_col=price_col,
        close_price_col=close_col,
    )


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="backtest.holdout",
                                description="Honest holdout GO/NO-GO backtest for a frozen model")
    p.add_argument("--model", required=True, help="path to the frozen catboost_<target>_<tag>.bin")
    p.add_argument("--holdout-start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--holdout-end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--price-col", default="ppwap", help="pre-off decision/settlement price column")
    p.add_argument("--close-col", default="odds_finish", help="closing-line column (CLV reference)")
    p.add_argument("--features-path", default=None,
                   help="training parquet (default: data/features/training.parquet)")
    p.add_argument("--devig-method", default="proportional",
                   choices=["proportional", "power", "shin"])
    p.add_argument("--stake", type=float, default=_DEFAULT_STAKE)
    p.add_argument("--commission", type=float, default=_DEFAULT_COMMISSION)
    p.add_argument("--out", default=None,
                   help="output dir (default: data/backtests/holdout_<stamp>)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    out_dir = args.out or os.path.join(
        "data", "backtests", f"holdout_{datetime.now():%Y%m%d_%H%M%S}")
    run_holdout(
        model_path=args.model,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        out_dir=out_dir,
        price_col=args.price_col,
        close_col=args.close_col,
        features_path=args.features_path,
        devig_method=args.devig_method,
        stake=args.stake,
        commission=args.commission,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
