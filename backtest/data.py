"""Build the backtest bet-candidate panel from the training matrix.

The walk-forward engine bets on *runners*: one row per (race_date, venue,
horse_id) carrying the point-in-time model features, the realised outcome, a
leak-safe **execution price** (taken pre-off) and the **closing price** (Betfair
SP) used only for closing-line value.

Why we re-use ``data/features/training.parquet`` directly
---------------------------------------------------------
Its features are computed point-in-time (trailing rates use only strictly-prior
runs — verified leak-safe in the baseline audit), so a feature value for a race
on date *t* never depends on anything after *t*. That means the same precomputed
matrix is valid for *every* walk-forward fold: only model *fitting* and
*calibration* must respect the train/test cut, which the engine enforces. This
avoids re-deriving 248k rows of features per fold.

Price hygiene (the whole point of a value backtester)
-----------------------------------------------------
The execution price must be one we could actually have matched pre-off, and the
closing line must only ever score CLV. (Historical note: the matrix's
``implied_prob`` was once ``1/odds_finish``; since commit ``3eca232`` it is
``1/morningwap`` — pre-off — verified empirically in the Stage-4 audit. We
still resolve our own price columns here rather than trusting it.)

* **execution price** = ``ppwap`` (Betfair pre-play weighted-average price —
  a chronologically-safe pre-off number, but a volume-weighted AVERAGE over
  the whole pre-off period, not a single quote proven fillable at one instant;
  treat it as a diagnostic proxy, not captured executable-price evidence —
  see ``execution.snapshots.SnapshotStore`` for the real thing), falling back
  to ``morningwap``;
* **closing price**   = ``odds_finish`` (BSP) — used only to score CLV.

We also keep the WIN market only (the PLACE rows duplicate every runner with the
place-market price), giving exactly one win-bet candidate per runner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models.features import PRICE_FREE_FEATURE_COLS
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_TRAINING = os.path.join(_BASE, "data", "features", "training.parquet")

# Columns the engine/metrics need beyond the model features.
_ID_COLS = ["race_date", "venue", "horse_id", "horse_name", "race_uid"]


@dataclass
class PanelConfig:
    """How to turn the training matrix into a bet panel."""

    price_col: str = "ppwap"                       # pre-off execution price
    fallback_price_cols: Sequence[str] = ("morningwap",)
    close_col: str = "odds_finish"                 # closing line (BSP) for CLV
    market_type: Optional[str] = "WIN"             # keep one row per runner
    max_price: float = 1000.0                      # drop absurd/garbage prices
    feature_cols: Sequence[str] = field(default_factory=lambda: list(PRICE_FREE_FEATURE_COLS))


def _race_uid(df: pd.DataFrame) -> pd.Series:
    """venue + race-time minute key; falls back to venue+date (audit C3: race_time
    is currently 100% null, so this degrades to venue-day — fine for per-runner
    betting, only de-vig needs a true race split)."""
    venue = df.get("venue", pd.Series(["?"] * len(df), index=df.index)).astype(str)
    rt = df.get("race_time")
    has_time = rt is not None and rt.notna().any() and rt.astype(str).str.len().gt(0).any()
    if has_time:
        slot = rt.astype(str)
    else:
        slot = pd.to_datetime(df["race_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    return venue + "|" + slot.astype(str)


def load_panel(df: Optional[pd.DataFrame] = None,
               path: Optional[str] = None,
               config: Optional[PanelConfig] = None) -> pd.DataFrame:
    """Return the bet-candidate panel (sorted by race_date).

    Pass ``df`` to supply a pre-loaded matrix (tests do this); otherwise it is
    read from ``path`` or the default training parquet.

    Output columns: the model ``feature_cols``, the id columns, ``won`` (0/1),
    ``bet_price`` (execution decimal odds), and ``close_price`` (BSP). Rows with
    no usable execution price or no outcome are dropped.
    """
    cfg = config or PanelConfig()
    if df is None:
        p = path or _DEFAULT_TRAINING
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"training matrix not found at {p}; run build_training_matrix() first")
        df = pd.read_parquet(p)
    df = df.copy()

    if cfg.market_type and "market_type" in df.columns:
        df = df[df["market_type"].astype(str).str.upper() == cfg.market_type.upper()]

    df["race_date"] = pd.to_datetime(df["race_date"], utc=True, errors="coerce")

    # Outcome: prefer the explicit `won` label, else derive from finishing position.
    if "won" in df.columns:
        won = pd.to_numeric(df["won"], errors="coerce")
    else:
        won = (pd.to_numeric(df["position"], errors="coerce") == 1).astype(float)

    # Execution price: first non-null, valid (>1, <=max) price across the chosen
    # column then the fallbacks. This is the price we *back at* — never a feature.
    bet_price = _first_valid_price(df, [cfg.price_col, *cfg.fallback_price_cols], cfg.max_price)
    close_price = _clean_price(df.get(cfg.close_col), cfg.max_price)

    if "race_uid" not in df.columns:
        df["race_uid"] = _race_uid(df)

    feat_cols = [c for c in cfg.feature_cols if c in df.columns]
    missing = set(cfg.feature_cols) - set(feat_cols)
    if missing:
        logger.warning("backtest.data: %d feature cols absent: %s",
                       len(missing), sorted(missing))

    id_cols = [c for c in _ID_COLS if c in df.columns]
    panel = df[id_cols + feat_cols].copy()
    panel["won"] = won.to_numpy()
    panel["bet_price"] = bet_price
    panel["close_price"] = close_price

    before = len(panel)
    panel = panel[panel["won"].notna() & np.isfinite(panel["bet_price"])].copy()
    panel["won"] = panel["won"].astype(int)
    panel = panel.sort_values("race_date").reset_index(drop=True)
    logger.info("backtest.data: panel %d rows (dropped %d w/o price+outcome); "
                "%d features; %s -> %s",
                len(panel), before - len(panel), len(feat_cols),
                panel["race_date"].min(), panel["race_date"].max())
    return panel


def _clean_price(series, max_price: float) -> np.ndarray:
    if series is None:
        return np.full(0, np.nan)
    v = np.array(pd.to_numeric(series, errors="coerce"), dtype=float)  # writable copy
    v[~((v > 1.0) & (v <= max_price))] = np.nan
    return v


def _first_valid_price(df: pd.DataFrame, cols, max_price: float) -> np.ndarray:
    """Coalesce several price columns into one valid execution price per row."""
    n = len(df)
    out = np.full(n, np.nan)
    for c in cols:
        if c not in df.columns:
            continue
        v = _clean_price(df[c], max_price)
        take = ~np.isfinite(out) & np.isfinite(v)
        out[take] = v[take]
    return out
