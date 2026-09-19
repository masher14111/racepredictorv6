"""Walk-forward backtest engine: train -> predict -> bet -> measure.

Pipeline per run
----------------
1. Split the panel's timeline into walk-forward folds (:mod:`backtest.splitter`).
2. For each fold: fit the model factory on the *train* slice, predict win
   probabilities for the *test* slice. The model never sees a test row, and the
   precomputed features are point-in-time, so a fold's prediction depends only on
   data on/before its origin T — the no-leakage guarantee.
3. Concatenate the scored test rows into one out-of-sample frame and derive the
   value signals (implied prob from the *execution* price, EV, edge, Kelly).
4. For each (strategy, stake-scheme): select bets, simulate the bankroll in
   chronological order (Kelly stakes compound; a stop-loss can halt new bets),
   and settle each bet at its execution price net of commission.
5. Summarise every strategy and the model's own calibration (A/E by probability
   bucket and by odds band) into a :class:`BacktestRun`.

Scoring the model ONCE per fold and replaying every strategy over the same frame
keeps a multi-strategy sweep cheap (one CatBoost fit per fold, not per strategy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.model import ModelFactory
from backtest.splitter import Fold, WalkForwardConfig, walk_forward_folds
from backtest.staking import Strategy
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Bankroll/execution settings shared by every strategy in a run."""

    initial_bankroll: float = 1000.0
    commission: float = 0.0           # exchange commission on net winnings (0-1)
    stop_loss_pct: Optional[float] = None  # halt new bets at this drawdown (e.g. 0.20)
    one_bet_per_runner: bool = True   # dedupe overlapping fold test windows


@dataclass
class BacktestRun:
    """Everything a report or the UI needs from one backtest."""

    config: dict
    folds: list[dict]
    scored: pd.DataFrame                       # out-of-sample rows + prob/ev/edge/kelly
    model_metrics: dict                        # OOS AUC / Brier / logloss / ECE-ish
    ae_by_prob: list[dict]
    ae_by_odds: list[dict]
    strategies: dict = field(default_factory=dict)  # name -> {summary, ledger}


class Backtester:
    def __init__(self, panel: pd.DataFrame, model_factory: ModelFactory,
                 walk_forward: WalkForwardConfig,
                 config: Optional[BacktestConfig] = None) -> None:
        if "race_date" not in panel.columns:
            raise ValueError("panel must have a race_date column")
        self.panel = panel.sort_values("race_date").reset_index(drop=True)
        self.factory = model_factory
        self.wf = walk_forward
        self.config = config or BacktestConfig()

    # ── walk-forward scoring ────────────────────────────────────────────────

    def _score_folds(self) -> tuple[pd.DataFrame, list[dict]]:
        """Fit per fold and return (scored OOS frame, fold metadata)."""
        folds = walk_forward_folds(self.panel["race_date"], self.wf)
        if not folds:
            raise ValueError(
                "no walk-forward folds — data span is shorter than "
                f"min_train_days ({self.wf.min_train_days}) + one window")

        rd = self.panel["race_date"]
        scored_parts: list[pd.DataFrame] = []
        meta: list[dict] = []
        for fold in folds:
            tr = fold.train_mask(rd).to_numpy()
            te = fold.test_mask(rd).to_numpy()
            n_tr, n_te = int(tr.sum()), int(te.sum())
            info = {
                "index": fold.index,
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "n_train": n_tr,
                "n_test": n_te,
            }
            # Need both classes in train and at least one test row to score.
            train_df = self.panel.loc[tr]
            if n_te == 0 or n_tr == 0 or train_df["won"].nunique() < 2:
                info["scored"] = False
                meta.append(info)
                logger.info("backtest: fold %d skipped (n_train=%d n_test=%d)",
                            fold.index, n_tr, n_te)
                continue

            model = self.factory.fit(train_df)
            test_df = self.panel.loc[te].copy()
            test_df["prob"] = model.predict(test_df)
            test_df["fold"] = fold.index
            scored_parts.append(test_df)
            info["scored"] = True
            meta.append(info)
            logger.info("backtest: fold %d scored — train=%d test=%d (<=%s)",
                        fold.index, n_tr, n_te, fold.test_end.date())

        if not scored_parts:
            raise ValueError("no fold produced predictions (insufficient data)")

        scored = pd.concat(scored_parts, ignore_index=True)
        if self.config.one_bet_per_runner:
            keys = [c for c in ("race_date", "venue", "horse_id") if c in scored.columns]
            if keys:
                before = len(scored)
                # Keep the LAST fold to score a runner (most training history).
                scored = (scored.sort_values("fold")
                          .drop_duplicates(keys, keep="last")
                          .sort_values("race_date").reset_index(drop=True))
                if before != len(scored):
                    logger.info("backtest: deduped %d overlapping fold rows",
                                before - len(scored))
        return scored, meta

    # ── value signals + model calibration ───────────────────────────────────

    @staticmethod
    def _attach_signals(scored: pd.DataFrame) -> pd.DataFrame:
        p = pd.to_numeric(scored["prob"], errors="coerce").to_numpy(dtype=float)
        d = pd.to_numeric(scored["bet_price"], errors="coerce").to_numpy(dtype=float)
        scored = scored.copy()
        scored["implied"] = metrics.implied_prob(d)
        scored["ev"] = metrics.expected_value(p, d)
        scored["edge"] = metrics.edge(p, d)
        scored["kelly"] = metrics.kelly_fraction(p, d)
        return scored

    @staticmethod
    def _model_metrics(scored: pd.DataFrame) -> dict:
        from sklearn.metrics import log_loss, roc_auc_score

        y = pd.to_numeric(scored["won"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(scored["prob"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
        out = {"n": int(y.size), "base_rate": round(float(y.mean()), 4) if y.size else None}
        if y.size and len(np.unique(y)) > 1:
            out["auc"] = round(float(roc_auc_score(y, p)), 4)
            out["logloss"] = round(float(log_loss(y, p)), 4)
        out["brier"] = round(float(np.mean((p - y) ** 2)), 4) if y.size else None
        # A coarse expected-calibration error over 10 equal-width bins.
        if y.size:
            bins = np.clip((p * 10).astype(int), 0, 9)
            ece = 0.0
            for b in range(10):
                m = bins == b
                if m.any():
                    ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
            out["ece"] = round(float(ece), 4)
        return out

    # ── bankroll simulation for one strategy ─────────────────────────────────

    def _simulate(self, scored: pd.DataFrame, strategy: Strategy, scheme) -> pd.DataFrame:
        """Select bets, walk the bankroll forward, settle each — return a ledger."""
        mask = strategy.select(scored).to_numpy()
        bets = scored.loc[mask].sort_values(
            [c for c in ("race_date", "venue", "horse_id") if c in scored.columns]
        ).reset_index(drop=True)

        init = self.config.initial_bankroll
        floor = init * (1.0 - self.config.stop_loss_pct) if self.config.stop_loss_pct else None
        bankroll = init
        halted = False

        rows = []
        for _, r in bets.iterrows():
            prob = float(r["prob"])
            price = float(r["bet_price"])
            stake = 0.0
            if not halted and bankroll > 0 and (floor is None or bankroll > floor):
                stake = float(scheme.stake(bankroll, prob, price))
                stake = min(stake, bankroll)  # never stake more than we hold
            profit = float(metrics.settle(stake, price, r["won"], self.config.commission)) \
                if stake > 0 else 0.0
            bankroll += profit
            if floor is not None and bankroll <= floor:
                halted = True
            rows.append({
                "race_date": r["race_date"],
                "venue": r.get("venue"),
                "horse_id": r.get("horse_id"),
                "horse_name": r.get("horse_name"),
                "prob": prob,
                "bet_price": price,
                "close_price": float(r["close_price"]) if pd.notna(r.get("close_price")) else np.nan,
                "won": int(r["won"]),
                "stake": stake,
                "profit": profit,
                "bankroll_after": bankroll,
                "ev": float(r.get("ev")) if pd.notna(r.get("ev")) else np.nan,
                "edge": float(r.get("edge")) if pd.notna(r.get("edge")) else np.nan,
            })
        ledger = pd.DataFrame(rows)
        # A stake of 0 means the bankroll guard blocked the bet — keep only real bets.
        if not ledger.empty:
            ledger = ledger[ledger["stake"] > 0].reset_index(drop=True)
        return ledger

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, strategies: dict[str, tuple]) -> BacktestRun:
        """Run the walk-forward and evaluate every ``name -> (Strategy, scheme)``.

        Returns a :class:`BacktestRun`. Scoring happens once; each strategy is a
        cheap replay over the shared out-of-sample frame.
        """
        scored, fold_meta = self._score_folds()
        scored = self._attach_signals(scored)
        model_metrics = self._model_metrics(scored)

        ae_prob = metrics.ae_table(scored["prob"], scored["won"])
        ae_odds = metrics.ae_table(scored["prob"], scored["won"],
                                   group_values=scored["bet_price"], bands=metrics._ODDS_BANDS)

        out: dict = {}
        for name, (strategy, scheme) in strategies.items():
            ledger = self._simulate(scored, strategy, scheme)
            summary = metrics.summarize_bets(ledger, self.config.initial_bankroll)
            summary["strategy"] = name
            out[name] = {"summary": summary, "ledger": ledger}
            logger.info("backtest: [%s] %d bets, yield=%s%%, CLV=%s%%, bankroll=%s",
                        name, summary["n_bets"], summary["yield_pct"],
                        summary["clv_pct_mean"], summary["final_bankroll"])

        return BacktestRun(
            config={
                "initial_bankroll": self.config.initial_bankroll,
                "commission": self.config.commission,
                "stop_loss_pct": self.config.stop_loss_pct,
                "walk_forward": vars(self.wf),
                "n_features": len([c for c in self.factory.feature_cols if c in self.panel.columns]),
            },
            folds=fold_meta,
            scored=scored,
            model_metrics=model_metrics,
            ae_by_prob=ae_prob,
            ae_by_odds=ae_odds,
            strategies=out,
        )
