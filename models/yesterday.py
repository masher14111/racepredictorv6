"""Yesterday's Bet Predictor — settle the model's picks against a past day's results.

What it does: take a single *completed* racing day (default: the most recent day that
actually has finishing positions), score every runner **blind to the result** with the
same models the live UI uses, select bets exactly as the live value layer would (or the
model's top win pick per race), stake a fixed amount, then settle each bet against the
real result. The output is the profit you would have made — an honest, fully
out-of-sample one-day backtest.

Honesty / data notes (all surfaced in the UI):
  * **Results lag the calendar.** Live racecards are scraped daily, but finishing
    positions arrive from the historical results feeds a few days later. So "yesterday"
    is really *the most recent day with settled results* — the engine picks it and the UI
    states the date plainly. ``available_dates`` lists every settleable day.
  * **Settled rows only carry the starting price** (``odds_finish`` / SP); the live
    bookmaker board prices exist only for *upcoming* races. So a bet is both *selected*
    and *settled* at SP. In reality you'd usually take a bigger board price than SP, so
    the value-bet profit here is, if anything, conservative — but closing-line value
    (CLV) is not measurable on this data.
  * **Selection never sees the result.** The scorer reads features only; value / top-pick
    selection ranks on model output. ``position`` is used solely for settlement.

Public API
----------
    from models.yesterday import run_yesterday, YesterdayConfig
    result = run_yesterday(YesterdayConfig(bet_type="each_way", stake=10))
    result.summary["roi_pct"], result.bets, result.date
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.predictor import Predictor
from models.value import ValueConfig, find_value_bets
from utils.backtester import _gross_return, _settle_outcome
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parent.parent
# The labelled training matrix is persisted here by build_training_matrix(write=True);
# loading it is instant, vs. minutes to rebuild from scratch.
_MATRIX_PATH = _BASE / "data" / "features" / "training.parquet"


@dataclass
class YesterdayConfig:
    """Tunables for a single-day settlement run."""

    strategy: str = "value"        # "value" (live value layer) | "top_pick" (best win prob)
    bet_type: str = "win"          # "win" | "each_way"
    stake: float = 10.0            # flat stake per bet, in the day's currency units
    bankroll: float = 1000.0       # starting bankroll (for the equity readout only)
    min_odds: float = 1.01         # top_pick: skip runners priced below this
    top_n: int = 1                 # top_pick: number of bets per race
    ew_places: int = 3             # each-way: places paid
    ew_fraction: float = 0.20      # each-way: place-leg fraction of the win odds
    target_date: Optional[str] = None  # "YYYY-MM-DD"; None ⇒ most recent settled day


@dataclass
class YesterdayResult:
    """Outputs from a settlement run."""

    config: YesterdayConfig
    date: Optional[str]
    available_dates: list = field(default_factory=list)
    bets: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    n_races: int = 0
    n_value_races: int = 0
    has_results: bool = False
    message: str = ""


# ── matrix loading ─────────────────────────────────────────────────────────────

def _load_matrix(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return the labelled matrix: caller-supplied, else the persisted parquet, else
    a fresh (slow) rebuild."""
    if df is not None:
        return df
    if _MATRIX_PATH.exists():
        try:
            return pd.read_parquet(_MATRIX_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yesterday: could not read %s: %s — rebuilding", _MATRIX_PATH.name, exc)
    from features.builder import build_training_matrix
    return build_training_matrix(write=False)


def _settled_days(df: pd.DataFrame) -> list[str]:
    """Sorted list of 'YYYY-MM-DD' days that carry at least one finishing position."""
    if df.empty or "position" not in df.columns:
        return []
    d = pd.to_datetime(df["race_date"], utc=True, errors="coerce")
    has_pos = df["position"].notna()
    days = sorted({str(x) for x in d[has_pos].dt.date.dropna().unique()})
    return days


# ── price + race key ───────────────────────────────────────────────────────────

def _price(df: pd.DataFrame) -> pd.Series:
    """Settlement/selection price for each runner: the starting price (``odds_finish``),
    falling back to the price implied by ``implied_prob``. NaN when neither is usable."""
    sp = pd.to_numeric(df.get("odds_finish", pd.Series(np.nan, index=df.index)), errors="coerce")
    ip = pd.to_numeric(df.get("implied_prob", pd.Series(np.nan, index=df.index)), errors="coerce")
    derived = 1.0 / ip.where(ip > 0)
    price = sp.where(sp > 1.0, derived)
    return price.where(price > 1.0)


def _race_key(df: pd.DataFrame) -> pd.Series:
    """(venue | race-time) string per runner; race_time falls back to race_date."""
    rd = df.get("race_date")
    rt = df["race_time"].where(df["race_time"].notna(), rd) if "race_time" in df.columns else rd
    return df["venue"].astype(str) + " | " + pd.Series(rt, index=df.index).astype(str)


# ── scoring ────────────────────────────────────────────────────────────────────

def _score_day(day_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Score a single day's runners with the live models + price-free value model.

    Returns the frame with ``won_prob`` / ``value_win_prob`` / ``expected_value`` etc.
    appended, or None if no models could be loaded.
    """
    p = Predictor()
    # Force the price-free value model to load even if the config has the value layer
    # off — this page is *about* the value picks.
    p._value_enabled = True
    if not p.load():
        return None
    return p._score(day_df.copy())


# ── selection ──────────────────────────────────────────────────────────────────

def _value_picks(grp: pd.DataFrame, cfg: YesterdayConfig) -> list[dict]:
    """Runners in this race that clear the live value gates (de-vig, EV, band, support)."""
    frame = pd.DataFrame({
        "horse_id": grp["horse_id"].astype(str),
        "horse_name": grp["horse_name"].astype(str),
        "value_win_prob": pd.to_numeric(grp.get("value_win_prob"), errors="coerce"),
        "decimal_odds": grp["_price"],
        "value_supported": grp.get("value_supported", pd.Series(True, index=grp.index)).astype(bool),
    })
    if frame["value_win_prob"].isna().all():
        return []
    vcfg = ValueConfig.from_config()
    return find_value_bets(frame, config=vcfg, bankroll=cfg.bankroll)


def _top_pick_rows(grp: pd.DataFrame, cfg: YesterdayConfig) -> pd.DataFrame:
    """The model's top-N win picks in this race, gated on min odds."""
    rank_col = "won_prob" if "won_prob" in grp.columns else "composite_score"
    elig = grp[grp["_price"].notna() & (grp["_price"] >= cfg.min_odds)].copy()
    if elig.empty:
        return elig
    return elig.sort_values(rank_col, ascending=False).head(max(1, cfg.top_n))


# ── settlement ─────────────────────────────────────────────────────────────────

def _settle_bet(row: pd.Series, price: float, cfg: YesterdayConfig, *,
                is_value: bool, model_prob: float, edge=None, ev=None) -> dict:
    position = int(row["position"])
    outcome = _settle_outcome(position, cfg.bet_type, cfg.ew_places)
    gross = _gross_return(outcome, cfg.stake, price, cfg.bet_type, cfg.ew_fraction)
    profit = gross - cfg.stake
    return {
        "venue": str(row.get("venue", "") or ""),
        "race_time": str(row.get("race_time") or row.get("race_date") or ""),
        "horse_name": str(row.get("horse_name", "") or ""),
        "horse_id": str(row.get("horse_id", "") or ""),
        "decimal_odds": round(float(price), 2),
        "model_win_prob": round(float(model_prob), 4) if model_prob is not None and np.isfinite(model_prob) else None,
        "value_edge": round(float(edge), 4) if edge is not None and np.isfinite(edge) else None,
        "expected_value": round(float(ev), 4) if ev is not None and np.isfinite(ev) else None,
        "is_value": bool(is_value),
        "position": position,
        "outcome": outcome,
        "stake": round(float(cfg.stake), 2),
        "gross_return": round(float(gross), 2),
        "profit": round(float(profit), 2),
    }


def _summarise(bets: list[dict], cfg: YesterdayConfig) -> dict:
    n = len(bets)
    if n == 0:
        return {"n_bets": 0, "n_winners": 0, "win_rate": 0.0, "place_rate": 0.0,
                "total_staked": 0.0, "total_profit": 0.0, "roi_pct": 0.0,
                "final_bankroll": round(cfg.bankroll, 2)}
    wins = sum(1 for b in bets if b["outcome"] == "win")
    places = sum(1 for b in bets if b["outcome"] in ("win", "place"))
    staked = sum(b["stake"] for b in bets)
    profit = sum(b["profit"] for b in bets)
    return {
        "n_bets": n,
        "n_winners": wins,
        "win_rate": round(wins / n * 100.0, 2),
        "place_rate": round(places / n * 100.0, 2),
        "total_staked": round(staked, 2),
        "total_profit": round(profit, 2),
        "roi_pct": round(profit / staked * 100.0, 2) if staked > 0 else 0.0,
        "final_bankroll": round(cfg.bankroll + profit, 2),
    }


# ── main entry ─────────────────────────────────────────────────────────────────

def run_yesterday(config: Optional[YesterdayConfig] = None,
                  df: Optional[pd.DataFrame] = None) -> YesterdayResult:
    """Score one settled day blind, place the model's bets, settle against results.

    Parameters
    ----------
    config : selection / staking knobs (see :class:`YesterdayConfig`).
    df : pre-loaded labelled matrix (with ``position``). None ⇒ load the persisted
         training matrix, else rebuild.
    """
    cfg = config or YesterdayConfig()
    matrix = _load_matrix(df)
    days = _settled_days(matrix)

    if not days:
        return YesterdayResult(config=cfg, date=None, available_dates=[],
                               has_results=False,
                               message="No settled results are available yet — "
                                       "finishing positions haven't been scraped for any day.")

    target = cfg.target_date if (cfg.target_date and cfg.target_date in days) else days[-1]

    d = pd.to_datetime(matrix["race_date"], utc=True, errors="coerce")
    day_df = matrix[(d.dt.date.astype(str) == target) & matrix["position"].notna()].copy()
    if day_df.empty:
        return YesterdayResult(config=cfg, date=target, available_dates=days,
                               has_results=False,
                               message=f"No settled runners found for {target}.")

    day_df["_price"] = _price(day_df).to_numpy()

    scored = _score_day(day_df)
    if scored is None:
        return YesterdayResult(config=cfg, date=target, available_dates=days,
                               has_results=False,
                               message="No trained models found — run `python -m models.train` first.")
    scored["_price"] = day_df["_price"].to_numpy()
    scored["_rkey"] = _race_key(scored).to_numpy()

    bets: list[dict] = []
    n_value_races = 0
    race_groups = list(scored.groupby("_rkey", sort=True))

    for _, grp in race_groups:
        grp = grp.reset_index(drop=True)
        if cfg.strategy == "value":
            picks = _value_picks(grp, cfg)
            if picks:
                n_value_races += 1
            by_id = {str(r["horse_id"]): r for _, r in grp.iterrows()}
            for pick in picks:
                row = by_id.get(str(pick.get("horse_id")))
                if row is None or pd.isna(row.get("position")) or pd.isna(pick.get("decimal_odds")):
                    continue
                bets.append(_settle_bet(
                    row, float(pick["decimal_odds"]), cfg, is_value=True,
                    model_prob=pick.get("model_prob"), edge=pick.get("edge"),
                    ev=pick.get("expected_value"),
                ))
        else:  # top_pick
            top = _top_pick_rows(grp, cfg)
            for _, row in top.iterrows():
                if pd.isna(row.get("position")) or pd.isna(row.get("_price")):
                    continue
                mp = row.get("won_prob")
                ev = row.get("expected_value")
                bets.append(_settle_bet(
                    row, float(row["_price"]), cfg, is_value=bool(row.get("value_bet", False)),
                    model_prob=mp, edge=row.get("value_edge"), ev=ev,
                ))

    summary = _summarise(bets, cfg)
    return YesterdayResult(
        config=cfg,
        date=target,
        available_dates=days,
        bets=bets,
        summary=summary,
        n_races=len(race_groups),
        n_value_races=n_value_races,
        has_results=True,
        message="",
    )
