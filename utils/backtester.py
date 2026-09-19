"""Historical prediction backtester.

Walk-forward simulation: score historical runners with the trained CatBoost model (or
implied-probability market baseline when no models are available), apply flat/Kelly staking,
settle against actual finishing positions, and report ROI, win rate, drawdown, and
venue/odds-band breakdowns.

Honesty guarantees:
  * **No look-ahead.** The scoring step (`_score_model` / `_score_market`) only ever reads
    feature columns / `implied_prob`; the finishing `position` and the derived `outcome`
    never reach the scorer. Selection (top-N by score) therefore cannot see the result.
  * **Settlement at a realistic executable price.** Bets are *selected and staked* at the
    price you could actually take (`bet_odds`, from the board/market price) but *settled* at
    the starting/closing price (`settle_odds`, from BSP → SP → returned SP). The two are
    resolved from different columns so a backtest never credits a stale board price you
    could not have been matched at. See `_resolve_prices`.
  * **CLV** (closing-line value) is reported wherever an independent board price *and* a
    closing/SP price both exist.
  * **Per-odds-band bootstrapped median ROI with a 95% CI** quantifies how noisy each band's
    edge is, and a **longshot gate** flags any positive ROI that collapses once a handful of
    big-priced winners are removed.

Usage (programmatic):
    from utils.backtester import Backtester, BacktestConfig
    result = Backtester().run()
    Backtester().report(result)           # writes reports/backtest.md

Usage (CLI):
    python -m utils.backtester [--strategy flat] [--stake 10] [--output reports/backtest.md]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from features.builder import build_training_matrix
from models.features import FEATURE_COLS
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.timezone import local_month, now

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_DEFAULT_REPORT = _BASE / "reports" / "backtest.md"
_COMPOSITE_W = {"won": 0.50, "placed_2": 0.30, "showed": 0.20}
_RACE_KEY = ["venue", "race_date"]

# Genuine pre-race board/market price the strategy could actually *take* when placing the
# bet. Used for selection, the min-odds gate, and Kelly staking — never for settlement.
# NB: the legacy/synthesised `decimal_odds` is deliberately NOT here — in the backtester it
# is derived from implied_prob (i.e. from the SP), so counting it as a board quote would
# fabricate non-zero CLV against a price it was itself derived from.
_BET_PRICE_COLS = ("odds_decimal", "board_odds")
# Realistic *executable* settlement price (the off / closing line), best first.
_SETTLE_PRICE_COLS = ("bsp", "betfair_sp", "sp", "odds_finish")


@dataclass
class BacktestConfig:
    """All tunable parameters for a single backtest run."""
    strategy: str = "flat"            # flat | fractional_kelly
    flat_stake: float = 10.0
    kelly_fraction: float = 0.25
    initial_bankroll: float = 1000.0
    min_selection_odds: float = 2.50
    top_n: int = 3
    bet_type: str = "win"             # win | each_way
    ew_fraction: float = 0.20
    ew_places: int = 3
    train_split_pct: float = 0.80
    stop_loss_pct: float = 0.20
    model_dir: Optional[str] = None
    output_path: Optional[str] = None
    # honest-evaluation knobs
    n_bootstrap: int = 2000           # resamples for per-band median ROI CI
    bootstrap_seed: int = 12345       # deterministic CI
    ci_alpha: float = 0.05            # 95% CI
    longshot_threshold: float = 10.0  # decimal odds above which a winner is a "longshot"
    robust_top_k: int = 3             # winners removed by the robustness/longshot gate
    longshot_dominance: float = 0.50  # flag if ≥ this share of winnings comes from longshots


@dataclass
class BacktestResult:
    """Outputs from a completed backtest run."""
    config: BacktestConfig
    model_bets: pd.DataFrame
    market_bets: pd.DataFrame
    model_summary: dict
    market_summary: dict
    metadata: dict


# ── pure helpers ────────────────────────────────────────────────────────────────

def _settle_outcome(position: int, bet_type: str, ew_places: int) -> str:
    """Map finishing position → 'win' | 'place' | 'lose'."""
    if position == 1:
        return "win"
    if bet_type == "each_way" and 1 < position <= ew_places:
        return "place"
    return "lose"


def _gross_return(
    outcome: str,
    stake: float,
    decimal_odds: float,
    bet_type: str,
    ew_fraction: float,
) -> float:
    """Gross return (stake back + winnings) for a settled bet."""
    if bet_type == "win":
        return stake * decimal_odds if outcome == "win" else 0.0
    half = stake / 2.0
    place_odds = (decimal_odds - 1.0) * ew_fraction + 1.0
    win_ret = half * decimal_odds if outcome == "win" else 0.0
    place_ret = half * place_odds if outcome in ("win", "place") else 0.0
    return win_ret + place_ret


def _kelly_stake(
    odds_decimal: float,
    win_prob: float,
    bankroll: float,
    kelly_fraction: float,
) -> float:
    """Fractional Kelly stake in currency units. Returns 0 when there is no edge."""
    b = odds_decimal - 1.0
    if b <= 0.0 or win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    q = 1.0 - win_prob
    f = max(0.0, (b * win_prob - q) / b)
    return round(f * bankroll * kelly_fraction, 2)


def _classify_odds_band(decimal_odds: float) -> str:
    if decimal_odds <= 4.0:
        return "Favourite (≤4.0)"
    if decimal_odds <= 10.0:
        return "Mid (4.1–10.0)"
    return "Longshot (>10.0)"


# ── main class ───────────────────────────────────────────────────────────────────

class Backtester:
    """Walk-forward backtesting engine."""

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        self._cfg = config or BacktestConfig()
        if model_dir:
            self._cfg.model_dir = model_dir
        self._models: dict = {}
        self._feature_cols: list[str] = list(FEATURE_COLS)

    # ── model loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> bool:
        """Attempt to load trained CatBoost models. Returns True if ≥1 loaded."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.warning("backtester: catboost not installed — market baseline only")
            return False

        raw = get_config()
        m = raw.get("model", {})
        vtag = m.get("version_tag", "v3")
        targets = list(m.get("targets", ["won", "placed_2", "showed"]))
        model_dir = Path(self._cfg.model_dir) if self._cfg.model_dir else _BASE / m.get("model_dir", "models")

        meta_path = model_dir / f"catboost_{vtag}_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                fc = meta.get("feature_cols")
                if fc:
                    self._feature_cols = fc
            except Exception as exc:
                logger.warning("backtester: could not read meta.json: %s", exc)

        loaded = 0
        for target in targets:
            path = model_dir / f"catboost_{target}_{vtag}.bin"
            if not path.exists():
                continue
            try:
                model = CatBoostClassifier()
                model.load_model(str(path))
                self._models[target] = model
                loaded += 1
            except Exception as exc:
                logger.warning("backtester: failed to load %s: %s", path.name, exc)

        if loaded:
            logger.info("backtester: %d model(s) loaded", loaded)
        else:
            logger.info("backtester: no trained models found — using market baseline")
        return loaded > 0

    # ── scoring ────────────────────────────────────────────────────────────────

    def _score_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add *_prob columns and composite_score using CatBoost models."""
        out = df.copy()
        avail = [c for c in self._feature_cols if c in out.columns]
        X = out[avail].fillna(np.nan).astype(float).values

        for target, model in self._models.items():
            col = f"{target}_prob"
            try:
                out[col] = model.predict_proba(X)[:, 1]
            except Exception as exc:
                logger.warning("backtester: predict_proba(%s) failed: %s", target, exc)
                out[col] = np.nan

        composite = np.zeros(len(out))
        wsum = 0.0
        for t, w in _COMPOSITE_W.items():
            col = f"{t}_prob"
            if col in out.columns:
                composite += w * pd.to_numeric(out[col], errors="coerce").fillna(0.0).values
                wsum += w
        out["composite_score"] = composite / wsum if wsum else 0.0
        return out

    def _score_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """Baseline: rank runners by implied_prob (market favourite first)."""
        out = df.copy()
        ip = pd.to_numeric(out.get("implied_prob", pd.Series(dtype=float)), errors="coerce")
        out["composite_score"] = ip.fillna(0.0)
        out["won_prob"] = ip
        return out

    # ── price resolution ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Attach `bet_odds`, `settle_odds`, `has_clv` (and source tags).

        `bet_odds`  — the price the strategy could take when placing the bet (board/market
                      price; falls back to a price derived from `implied_prob`). Drives
                      selection, the min-odds gate and Kelly staking.
        `settle_odds` — the realistic executable settlement price (BSP → SP → returned SP).
                      When no independent closing/SP price exists it falls back to `bet_odds`
                      so the simulation still settles, but `has_clv` is then False.
        `has_clv`   — True only when a genuine board price *and* a distinct closing/SP price
                      both exist, i.e. when CLV is actually measurable.
        """
        out = df.copy()

        def _first_valid(cols):
            val = pd.Series(np.nan, index=out.index, dtype="float64")
            src = pd.Series("", index=out.index, dtype="object")
            for col in cols:
                if col not in out.columns:
                    continue
                v = pd.to_numeric(out[col], errors="coerce")
                take = val.isna() & v.notna() & (v > 1.0)
                val = val.where(~take, v)
                src = src.where(~take, col)
            return val, src

        bet, bet_src = _first_valid(_BET_PRICE_COLS)
        # last-resort bet price: invert implied_prob (still a pre-race quantity)
        if "implied_prob" in out.columns:
            ip = pd.to_numeric(out["implied_prob"], errors="coerce")
            derived = 1.0 / ip.replace(0.0, np.nan)
            take = bet.isna() & derived.notna() & (derived > 1.0)
            bet = bet.where(~take, derived)
            bet_src = bet_src.where(~take, "implied_prob")

        settle, settle_src = _first_valid(_SETTLE_PRICE_COLS)
        # no closing/SP price → settle at the bet price (CLV unavailable)
        take = settle.isna() & bet.notna()
        settle = settle.where(~take, bet)
        settle_src = settle_src.where(~take, bet_src)

        out["bet_odds"] = bet
        out["settle_odds"] = settle
        out["_bet_src"] = bet_src
        out["_settle_src"] = settle_src
        # CLV is only meaningful with a real board price AND a distinct closing/SP price.
        out["has_clv"] = (
            bet_src.isin(list(_BET_PRICE_COLS))
            & settle_src.isin(list(_SETTLE_PRICE_COLS))
            & (bet_src != settle_src)
            & bet.notna()
            & settle.notna()
        )
        return out

    # ── simulation ─────────────────────────────────────────────────────────────

    def _simulate(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Walk through races chronologically, select top-N, simulate staking and settlement.

        Returns a DataFrame of one row per simulated bet.
        """
        cfg = self._cfg
        bankroll = cfg.initial_bankroll
        floor = bankroll * (1.0 - cfg.stop_loss_pct)
        rows = []

        for _, grp in df.groupby(_RACE_KEY, sort=True):
            if bankroll < floor:
                logger.info("backtester: stop-loss triggered at %.2f — halting %s", bankroll, mode)
                break

            grp = grp.reset_index(drop=True)
            # Gate on the price the strategy could actually TAKE (bet_odds), not the SP.
            valid = grp[
                grp["bet_odds"].notna()
                & (pd.to_numeric(grp["bet_odds"], errors="coerce") >= cfg.min_selection_odds)
                & grp["settle_odds"].notna()
                & grp["position"].notna()
            ].copy()
            if valid.empty:
                continue

            # Selection is blind to the result — it ranks on composite_score only.
            selected = valid.sort_values("composite_score", ascending=False).head(cfg.top_n)

            for _, runner in selected.iterrows():
                bet_odds = float(runner["bet_odds"])
                settle_odds = float(runner["settle_odds"])
                position = int(runner["position"])
                win_prob = float(
                    runner.get("won_prob") if pd.notna(runner.get("won_prob")) else runner.get("composite_score") or 0.0
                )

                # Stake on the price you take (bet_odds) …
                if cfg.strategy == "flat":
                    stake = cfg.flat_stake
                else:
                    stake = _kelly_stake(bet_odds, win_prob, bankroll, cfg.kelly_fraction)

                if stake <= 0.0:
                    continue

                # … but settle the return at the realistic executable price (SP/BSP).
                outcome = _settle_outcome(position, cfg.bet_type, cfg.ew_places)
                ret = _gross_return(outcome, stake, settle_odds, cfg.bet_type, cfg.ew_fraction)
                profit = ret - stake
                bankroll = bankroll - stake + ret

                has_clv = bool(runner.get("has_clv", False))
                clv = (bet_odds / settle_odds - 1.0) if (has_clv and settle_odds > 0) else np.nan

                rows.append({
                    "mode": mode,
                    "race_date": runner.get("race_date"),
                    "venue": str(runner.get("venue", "") or ""),
                    "horse_name": str(runner.get("horse_name", "") or ""),
                    "horse_id": str(runner.get("horse_id", "") or ""),
                    "jockey": str(runner.get("jockey", "") or ""),
                    "trainer": str(runner.get("trainer", "") or ""),
                    "decimal_odds": bet_odds,           # back-compat: the price banded/reported on
                    "bet_odds": bet_odds,
                    "settle_odds": settle_odds,
                    "clv": clv,
                    "has_clv": has_clv,
                    "implied_prob": runner.get("implied_prob"),
                    "composite_score": float(runner.get("composite_score") or 0.0),
                    "won_prob": win_prob,
                    "position": position,
                    "stake": stake,
                    "outcome": outcome,
                    "gross_return": round(ret, 4),
                    "profit": round(profit, 4),
                    "bankroll": round(bankroll, 4),
                })

        if not rows:
            return pd.DataFrame(columns=[
                "mode", "race_date", "venue", "horse_name", "decimal_odds",
                "composite_score", "stake", "outcome", "profit", "bankroll",
            ])

        bets = pd.DataFrame(rows)
        bets["cumulative_profit"] = bets["profit"].cumsum()
        return bets

    # ── metrics ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_summary(bets: pd.DataFrame, cfg: BacktestConfig) -> dict:
        if bets.empty:
            return {
                "total_bets": 0,
                "win_rate": 0.0,
                "place_rate": 0.0,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "final_bankroll": cfg.initial_bankroll,
                "n_clv": 0,
                "clv_mean_pct": 0.0,
                "clv_positive_rate": 0.0,
            }
        total = len(bets)
        wins = int((bets["outcome"] == "win").sum())
        places = int(((bets["outcome"] == "win") | (bets["outcome"] == "place")).sum())
        staked = float(bets["stake"].sum())
        profit = float(bets["profit"].sum())
        roi = (profit / staked * 100.0) if staked > 0 else 0.0

        equity = pd.to_numeric(bets["bankroll"], errors="coerce").values
        peak = np.maximum.accumulate(equity)
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
        max_dd = float(drawdown.max()) * 100.0 if len(drawdown) else 0.0

        # Closing-line value over bets that actually have an independent SP/closing price.
        n_clv, clv_mean_pct, clv_positive_rate = 0, 0.0, 0.0
        if "clv" in bets.columns and "has_clv" in bets.columns:
            cl = pd.to_numeric(
                bets.loc[bets["has_clv"].astype(bool), "clv"], errors="coerce"
            ).dropna()
            n_clv = int(len(cl))
            if n_clv:
                clv_mean_pct = round(float(cl.mean()) * 100.0, 2)
                clv_positive_rate = round(float((cl > 0).mean()) * 100.0, 2)

        return {
            "total_bets": total,
            "win_rate": round(wins / total * 100, 2),
            "place_rate": round(places / total * 100, 2),
            "total_staked": round(staked, 2),
            "total_profit": round(profit, 2),
            "roi_pct": round(roi, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "final_bankroll": round(float(bets["bankroll"].iloc[-1]), 2),
            "n_clv": n_clv,
            "clv_mean_pct": clv_mean_pct,
            "clv_positive_rate": clv_positive_rate,
        }

    @staticmethod
    def _monthly_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
        if bets.empty:
            return pd.DataFrame()
        df = bets.copy()
        # tz_convert(None) drops to UTC, which buckets a late-evening Irish card
        # into the wrong month at a month boundary. Convert to local first.
        df["month"] = local_month(df["race_date"])
        df = df.dropna(subset=["month"])
        if df.empty:
            return pd.DataFrame()
        g = (
            df.groupby("month")
            .agg(
                bets=("profit", "count"),
                wins=("outcome", lambda x: (x == "win").sum()),
                staked=("stake", "sum"),
                profit=("profit", "sum"),
            )
            .reset_index()
        )
        g["win_rate"] = (g["wins"] / g["bets"] * 100).round(1)
        g["roi_pct"] = (g["profit"] / g["staked"] * 100).round(2)
        g["month"] = g["month"].astype(str)
        return g

    @staticmethod
    def _odds_band_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
        if bets.empty:
            return pd.DataFrame()
        df = bets.copy()
        df["odds_band"] = pd.to_numeric(df["decimal_odds"], errors="coerce").apply(
            lambda v: _classify_odds_band(v) if pd.notna(v) else "Unknown"
        )
        g = (
            df.groupby("odds_band")
            .agg(
                bets=("profit", "count"),
                wins=("outcome", lambda x: (x == "win").sum()),
                staked=("stake", "sum"),
                profit=("profit", "sum"),
            )
            .reset_index()
        )
        g["win_rate"] = (g["wins"] / g["bets"] * 100).round(1)
        g["roi_pct"] = (g["profit"] / g["staked"] * 100).round(2)
        return g

    @staticmethod
    def _band_ci(
        bets: pd.DataFrame,
        n_boot: int = 2000,
        seed: int = 12345,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Per-odds-band bootstrapped median ROI with a (1-alpha) confidence interval.

        For each band the per-bet (profit, stake) pairs are resampled with replacement
        `n_boot` times; each resample's ROI = Σprofit / Σstake. We report the point ROI,
        the bootstrap median, and the alpha/2 … 1-alpha/2 percentile interval. A wide CI
        that straddles 0 means the band's apparent edge is not distinguishable from noise.
        Deterministic for a fixed `seed`.
        """
        if bets.empty:
            return pd.DataFrame()
        df = bets.copy()
        df["odds_band"] = pd.to_numeric(df["decimal_odds"], errors="coerce").apply(
            lambda v: _classify_odds_band(v) if pd.notna(v) else "Unknown"
        )
        rng = np.random.default_rng(seed)
        lo_pct, hi_pct = alpha / 2.0 * 100.0, (1.0 - alpha / 2.0) * 100.0
        rows = []
        for band, g in df.groupby("odds_band", sort=True):
            profit = pd.to_numeric(g["profit"], errors="coerce").fillna(0.0).to_numpy(float)
            stake = pd.to_numeric(g["stake"], errors="coerce").fillna(0.0).to_numpy(float)
            n = len(profit)
            if n == 0:
                continue
            tot_stake = stake.sum()
            point_roi = (profit.sum() / tot_stake * 100.0) if tot_stake > 0 else 0.0
            idx = rng.integers(0, n, size=(n_boot, n))
            bs_stake = stake[idx].sum(axis=1)
            bs_roi = np.where(bs_stake > 0, profit[idx].sum(axis=1) / bs_stake * 100.0, 0.0)
            rows.append({
                "odds_band": band,
                "bets": n,
                "roi_pct": round(float(point_roi), 2),
                "roi_median": round(float(np.median(bs_roi)), 2),
                "ci_low": round(float(np.percentile(bs_roi, lo_pct)), 2),
                "ci_high": round(float(np.percentile(bs_roi, hi_pct)), 2),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _robustness_gate(
        bets: pd.DataFrame,
        longshot_threshold: float = 10.0,
        top_k: int = 3,
        dominance: float = 0.50,
    ) -> dict:
        """Longshot gate: flag positive ROI that rests on a few big-priced winners.

        Reports the share of total winnings coming from longshots (decimal odds >
        `longshot_threshold`) and from the single most profitable `top_k` bets, plus the
        ROI recomputed with those top winners removed. `flagged` is True when ROI is
        positive overall but either (a) collapses to ≤0 once the top `top_k` winners are
        dropped, or (b) longshots supply ≥ `dominance` of all winnings. A flagged result
        means the edge is concentration-driven, not broad-based.
        """
        base = {
            "flagged": False, "reason": "none", "n_bets": 0,
            "total_profit": 0.0, "roi_full": 0.0, "roi_ex_top_k": 0.0, "top_k": int(top_k),
            "n_longshot_winners": 0, "longshot_profit_share": 0.0, "top_k_profit_share": 0.0,
        }
        if bets.empty:
            return base

        profit = pd.to_numeric(bets["profit"], errors="coerce").fillna(0.0)
        stake = pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0)
        odds = pd.to_numeric(bets["decimal_odds"], errors="coerce")

        total_profit = float(profit.sum())
        total_stake = float(stake.sum())
        roi_full = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0

        winners = profit > 0
        gross_winnings = float(profit[winners].sum())
        long_win_mask = winners & (odds > longshot_threshold)
        n_long = int(long_win_mask.sum())
        longshot_winnings = float(profit[long_win_mask].sum())
        longshot_share = (longshot_winnings / gross_winnings) if gross_winnings > 0 else 0.0

        # Drop the top_k most profitable individual bets and recompute ROI.
        drop_idx = profit.sort_values(ascending=False).index[: max(0, int(top_k))]
        keep = ~bets.index.isin(drop_idx)
        s2 = float(stake[keep].sum())
        roi_ex = (float(profit[keep].sum()) / s2 * 100.0) if s2 > 0 else 0.0
        top_k_share = (
            float(profit[~keep].clip(lower=0).sum()) / gross_winnings
            if gross_winnings > 0 else 0.0
        )

        flips = roi_full > 0 and roi_ex <= 0
        dominated = roi_full > 0 and longshot_share >= dominance
        reasons = []
        if flips:
            reasons.append(f"ROI {roi_full:+.1f}% → {roi_ex:+.1f}% after removing top {top_k} winners")
        if dominated:
            reasons.append(f"{longshot_share * 100:.0f}% of winnings from longshots (>{longshot_threshold:g})")

        return {
            "flagged": bool(flips or dominated),
            "reason": "; ".join(reasons) if reasons else "none",
            "n_bets": int(len(bets)),
            "total_profit": round(total_profit, 2),
            "roi_full": round(roi_full, 2),
            "roi_ex_top_k": round(roi_ex, 2),
            "top_k": int(top_k),
            "n_longshot_winners": n_long,
            "longshot_profit_share": round(longshot_share, 4),
            "top_k_profit_share": round(top_k_share, 4),
        }

    @staticmethod
    def _venue_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
        if bets.empty:
            return pd.DataFrame()
        g = (
            bets.groupby("venue")
            .agg(
                bets=("profit", "count"),
                wins=("outcome", lambda x: (x == "win").sum()),
                staked=("stake", "sum"),
                profit=("profit", "sum"),
            )
            .reset_index()
        )
        g["win_rate"] = (g["wins"] / g["bets"] * 100).round(1)
        g["roi_pct"] = (g["profit"] / g["staked"] * 100).round(2)
        return g.sort_values("roi_pct", ascending=False).head(10).reset_index(drop=True)

    # ── main entry ─────────────────────────────────────────────────────────────

    def run(self, df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Build (or accept) the labelled training matrix, split chronologically,
        score, simulate, and return a BacktestResult.

        Parameters
        ----------
        df:
            Pre-loaded labelled DataFrame with `position` column. If None,
            calls build_training_matrix(write=False).
        """
        logger.info("backtester: building training matrix …")
        if df is None:
            df = build_training_matrix(write=False)

        _empty_sum = self._compute_summary(pd.DataFrame(), self._cfg)

        if df.empty:
            logger.warning("backtester: training matrix empty — no historical positions available")
            return BacktestResult(
                config=self._cfg,
                model_bets=pd.DataFrame(),
                market_bets=pd.DataFrame(),
                model_summary=_empty_sum,
                market_summary=_empty_sum,
                metadata={"total_races": 0, "total_runners": 0, "date_range": "N/A",
                          "has_models": False, "train_rows": 0, "test_rows": 0},
            )

        df = df.copy()
        df["race_date"] = pd.to_datetime(df["race_date"], utc=True, errors="coerce")
        df = df.dropna(subset=["race_date", "position"]).reset_index(drop=True)
        df = df.sort_values("race_date").reset_index(drop=True)

        split_idx = int(len(df) * self._cfg.train_split_pct)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        logger.info(
            "backtester: %d total rows — %d train, %d test (%s → %s)",
            len(df), split_idx, len(test_df),
            df["race_date"].iloc[0].date() if len(df) else "—",
            df["race_date"].iloc[-1].date() if len(df) else "—",
        )

        if test_df.empty:
            logger.warning("backtester: test window is empty (all data is in train split)")
            return BacktestResult(
                config=self._cfg,
                model_bets=pd.DataFrame(),
                market_bets=pd.DataFrame(),
                model_summary=_empty_sum,
                market_summary=_empty_sum,
                metadata={"total_races": 0, "total_runners": len(df), "date_range": "N/A",
                          "has_models": False, "train_rows": split_idx, "test_rows": 0},
            )

        # Ensure decimal_odds column (derived from implied_prob when absent)
        if "decimal_odds" not in test_df.columns:
            ip = pd.to_numeric(test_df.get("implied_prob", pd.Series(dtype=float)), errors="coerce")
            test_df["decimal_odds"] = (1.0 / ip.replace(0.0, np.nan)).round(3)

        # Separate the price we BET at (board) from the price we SETTLE at (SP/BSP).
        test_df = self._resolve_prices(test_df)
        clv_n = int(test_df["has_clv"].sum())
        logger.info(
            "backtester: settle source(s)=%s — %d/%d runners have an independent SP/CLV price",
            sorted(set(test_df.loc[test_df["settle_odds"].notna(), "_settle_src"]) - {""}) or "none",
            clv_n, len(test_df),
        )

        has_models = self._load_models()
        model_scored = self._score_model(test_df) if has_models else self._score_market(test_df)
        market_scored = self._score_market(test_df)

        model_bets = self._simulate(model_scored, "model")
        market_bets = self._simulate(market_scored, "market")

        date_range = (
            f"{test_df['race_date'].min().date()} → {test_df['race_date'].max().date()}"
        )
        n_races = test_df.groupby(_RACE_KEY).ngroups

        return BacktestResult(
            config=self._cfg,
            model_bets=model_bets,
            market_bets=market_bets,
            model_summary=self._compute_summary(model_bets, self._cfg),
            market_summary=self._compute_summary(market_bets, self._cfg),
            metadata={
                "has_models": has_models,
                "total_runners": len(test_df),
                "total_races": n_races,
                "date_range": date_range,
                "train_rows": split_idx,
                "test_rows": len(test_df),
                "clv_available": clv_n > 0,
                "settle_sources": sorted(
                    set(test_df.loc[test_df["settle_odds"].notna(), "_settle_src"]) - {""}
                ),
            },
        )

    # ── report ─────────────────────────────────────────────────────────────────

    def report(
        self,
        result: BacktestResult,
        output_path: Optional[str] = None,
    ) -> str:
        """Render a markdown report and write it to disk. Returns the markdown string."""
        cfg = result.config
        meta = result.metadata
        ms = result.model_summary
        mkt = result.market_summary
        mode_label = "Model (CatBoost)" if meta.get("has_models") else "Model (market fallback)"

        monthly = self._monthly_breakdown(result.model_bets)
        bands = self._odds_band_breakdown(result.model_bets)
        venues = self._venue_breakdown(result.model_bets)
        band_ci = self._band_ci(
            result.model_bets, cfg.n_bootstrap, cfg.bootstrap_seed, cfg.ci_alpha
        )
        gate = self._robustness_gate(
            result.model_bets, cfg.longshot_threshold, cfg.robust_top_k, cfg.longshot_dominance
        )

        def _sign(v: float) -> str:
            return f"+€{v:.2f}" if v >= 0 else f"-€{abs(v):.2f}"

        def _clv_cell(s: dict) -> str:
            return "n/a" if not s.get("n_clv") else f"{s['clv_mean_pct']:+.2f}%"

        def _clv_pos_cell(s: dict) -> str:
            return "n/a" if not s.get("n_clv") else f"{s['clv_positive_rate']:.1f}% (n={s['n_clv']})"

        lines = [
            "# Backtest Report",
            "",
            f"**Generated:** {now().strftime('%Y-%m-%d %H:%M %Z')}  ",
            f"**Date range:** {meta.get('date_range', 'N/A')}  ",
            (
                f"**Races:** {meta.get('total_races', 0)} | "
                f"**Test runners:** {meta.get('total_runners', 0)} | "
                f"**Train rows:** {meta.get('train_rows', 0)} | "
                f"**Test rows:** {meta.get('test_rows', 0)}"
            ),
            "",
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Strategy | {cfg.strategy} |",
            f"| Flat stake | €{cfg.flat_stake:.2f} |",
            f"| Kelly fraction | {cfg.kelly_fraction} |",
            f"| Initial bankroll | €{cfg.initial_bankroll:.2f} |",
            f"| Min selection odds | {cfg.min_selection_odds:.2f} |",
            f"| Top N per race | {cfg.top_n} |",
            f"| Bet type | {cfg.bet_type} |",
            f"| Train/test split | {int(cfg.train_split_pct * 100)}/{int((1 - cfg.train_split_pct) * 100)} by date |",
            f"| Stop-loss | {int(cfg.stop_loss_pct * 100)}% drawdown |",
            "",
            "## Overall Performance",
            "",
            f"| Metric | {mode_label} | Market Baseline |",
            "|--------|-------------|-----------------|",
            f"| Bets placed | {ms['total_bets']} | {mkt['total_bets']} |",
            f"| Win rate | {ms['win_rate']:.1f}% | {mkt['win_rate']:.1f}% |",
            f"| Place rate | {ms['place_rate']:.1f}% | {mkt['place_rate']:.1f}% |",
            f"| Total staked | €{ms['total_staked']:.2f} | €{mkt['total_staked']:.2f} |",
            f"| Total profit | {_sign(ms['total_profit'])} | {_sign(mkt['total_profit'])} |",
            f"| ROI | {ms['roi_pct']:+.2f}% | {mkt['roi_pct']:+.2f}% |",
            f"| Max drawdown | {ms['max_drawdown_pct']:.2f}% | {mkt['max_drawdown_pct']:.2f}% |",
            f"| CLV (mean) | {_clv_cell(ms)} | {_clv_cell(mkt)} |",
            f"| CLV > 0 (beat close) | {_clv_pos_cell(ms)} | {_clv_pos_cell(mkt)} |",
            f"| Final bankroll | €{ms['final_bankroll']:.2f} | €{mkt['final_bankroll']:.2f} |",
            "",
        ]

        if not monthly.empty:
            lines += [
                f"## Monthly P&L — {mode_label}",
                "",
                "| Month | Bets | Win % | Staked | Profit | ROI |",
                "|-------|------|-------|--------|--------|-----|",
            ]
            for _, row in monthly.iterrows():
                lines.append(
                    f"| {row['month']} | {row['bets']} | {row['win_rate']:.1f}% "
                    f"| €{row['staked']:.2f} | {_sign(row['profit'])} | {row['roi_pct']:+.2f}% |"
                )
            lines.append("")

        if not bands.empty:
            lines += [
                "## Performance by Odds Band",
                "",
                "| Band | Bets | Win % | Staked | Profit | ROI |",
                "|------|------|-------|--------|--------|-----|",
            ]
            for _, row in bands.iterrows():
                lines.append(
                    f"| {row['odds_band']} | {row['bets']} | {row['win_rate']:.1f}% "
                    f"| €{row['staked']:.2f} | {_sign(row['profit'])} | {row['roi_pct']:+.2f}% |"
                )
            lines.append("")

        if not band_ci.empty:
            ci_lvl = int(round((1.0 - cfg.ci_alpha) * 100))
            lines += [
                f"## Per-Band ROI — bootstrapped {ci_lvl}% CI ({cfg.n_bootstrap} resamples)",
                "",
                "| Band | Bets | ROI | Median ROI | CI low | CI high |",
                "|------|------|-----|-----------|--------|---------|",
            ]
            for _, row in band_ci.iterrows():
                lines.append(
                    f"| {row['odds_band']} | {row['bets']} | {row['roi_pct']:+.2f}% "
                    f"| {row['roi_median']:+.2f}% | {row['ci_low']:+.2f}% | {row['ci_high']:+.2f}% |"
                )
            lines += [
                "",
                "_A CI that straddles 0% means the band's edge is not distinguishable from "
                "noise at this sample size._",
                "",
            ]

        # Edge-robustness / longshot gate
        gate_status = "⚠️ FLAGGED" if gate["flagged"] else "✓ robust"
        lines += [
            "## Edge Robustness (Longshot Gate)",
            "",
            f"**Status:** {gate_status}",
            "",
            "| Check | Value |",
            "|-------|-------|",
            f"| ROI (all bets) | {gate['roi_full']:+.2f}% |",
            f"| ROI excl. top {gate['top_k']} winners | {gate['roi_ex_top_k']:+.2f}% |",
            f"| Top {gate['top_k']} winners' share of winnings | {gate['top_k_profit_share'] * 100:.1f}% |",
            f"| Longshot winners (>{cfg.longshot_threshold:g}) | {gate['n_longshot_winners']} |",
            f"| Longshot share of winnings | {gate['longshot_profit_share'] * 100:.1f}% |",
            "",
        ]
        if gate["flagged"]:
            lines += [
                f"> **Edge is concentration-driven, treat with caution:** {gate['reason']}.",
                "",
            ]

        if not venues.empty:
            lines += [
                "## Top 10 Venues by ROI",
                "",
                "| Venue | Bets | Win % | Profit | ROI |",
                "|-------|------|-------|--------|-----|",
            ]
            for _, row in venues.iterrows():
                lines.append(
                    f"| {row['venue']} | {row['bets']} | {row['win_rate']:.1f}% "
                    f"| {_sign(row['profit'])} | {row['roi_pct']:+.2f}% |"
                )
            lines.append("")

        lines += [
            "## Notes",
            "",
            f"- Scoring mode: **{mode_label}**.",
            "- Market baseline ranks runners by implied probability (favourite first).",
            f"- Walk-forward: model evaluated on the last "
            f"{int((1 - cfg.train_split_pct) * 100)}% of historical races by date.",
            f"- Bets selected/staked at the takeable board price (≥ {cfg.min_selection_odds}); "
            "**settled at the realistic executable price (BSP → SP → returned SP)** — not the "
            "price used to pick the bet.",
            f"- Settlement price source(s) this run: "
            f"{', '.join(meta.get('settle_sources', [])) or 'none'}.",
            (
                "- CLV (closing-line value) reported above where an independent board price and "
                "a closing/SP price both exist."
                if meta.get("clv_available")
                else "- **CLV not measurable** this run: no independent board price distinct "
                "from the closing/SP price in the data (both collapse to the same source)."
            ),
            "- Selection is blind to the result: the scorer never sees finishing position.",
        ]
        if not meta.get("has_models"):
            lines += [
                "- **No trained models found.** Both simulations used market implied probability.",
                "  Run `python -m models.train` once results data is available, then re-run.",
            ]

        md = "\n".join(lines) + "\n"
        out = Path(output_path or cfg.output_path or _DEFAULT_REPORT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        logger.info("backtester: report written → %s", out)
        return md


# ── module-level convenience ──────────────────────────────────────────────────────

def run_backtest(
    config: Optional[BacktestConfig] = None,
    df: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Run a full backtest and write the report. Returns the BacktestResult."""
    bt = Backtester(config)
    result = bt.run(df)
    bt.report(result)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────────

def _main() -> None:
    ap = argparse.ArgumentParser(description="Backtest predictions on historical race data")
    ap.add_argument("--strategy", default="flat", choices=["flat", "fractional_kelly"])
    ap.add_argument("--stake", type=float, default=10.0, metavar="EUR")
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--bankroll", type=float, default=1000.0, metavar="EUR")
    ap.add_argument("--min-odds", type=float, default=2.50)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--bet-type", default="win", choices=["win", "each_way"])
    ap.add_argument("--split", type=float, default=0.80)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--output", default=None, help="Report output path (default: reports/backtest.md)")
    args = ap.parse_args()

    cfg = BacktestConfig(
        strategy=args.strategy,
        flat_stake=args.stake,
        kelly_fraction=args.kelly,
        initial_bankroll=args.bankroll,
        min_selection_odds=args.min_odds,
        top_n=args.top_n,
        bet_type=args.bet_type,
        train_split_pct=args.split,
        model_dir=args.model_dir,
        output_path=args.output,
    )

    result = run_backtest(cfg)
    ms = result.model_summary
    mkt = result.market_summary
    meta = result.metadata

    out = Path(cfg.output_path or _DEFAULT_REPORT)
    print(f"\nBacktest complete — {meta.get('date_range', 'N/A')}")
    print(f"  Races: {meta.get('total_races', 0)}  |  Test runners: {meta.get('total_runners', 0)}")
    print(f"  Model   — ROI: {ms['roi_pct']:+.2f}%  Win: {ms['win_rate']:.1f}%  "
          f"Bets: {ms['total_bets']}  Profit: €{ms['total_profit']:.2f}")
    print(f"  Market  — ROI: {mkt['roi_pct']:+.2f}%  Win: {mkt['win_rate']:.1f}%  "
          f"Bets: {mkt['total_bets']}  Profit: €{mkt['total_profit']:.2f}")
    print(f"  Report  → {out}")


if __name__ == "__main__":
    _main()
