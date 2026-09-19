"""CLI: run a walk-forward value-betting backtest and save the results.

    python -m backtest                       # defaults from config.yaml
    python -m backtest --min-train-days 365 --window-days 30 --task-type GPU
    python -m backtest --no-save --min-train-days 540 --window-days 60

The default strategy panel mirrors the live config: an EV-threshold and a
probability-edge rule, each under flat and quarter-Kelly staking, plus a
"back everything in band" control so value-add is visible against a baseline.
"""
from __future__ import annotations

import argparse
import sys

from backtest.data import PanelConfig, load_panel
from backtest.engine import Backtester, BacktestConfig
from backtest.model import CatBoostFactory
from backtest.report import render_report, save_run
from backtest.splitter import WalkForwardConfig
from backtest.staking import FlatStake, KellyStake, Strategy
from models.features import PRICE_FREE_FEATURE_COLS
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("backtest.cli")


def _default_strategies(value_cfg: dict, flat_unit: float, kelly_frac: float) -> dict:
    min_ev = float(value_cfg.get("min_expected_value", 0.05))
    min_odds = float(value_cfg.get("min_odds", 2.0))
    max_odds = float(value_cfg.get("max_odds", 4.0))
    band = dict(min_odds=min_odds, max_odds=max_odds)
    return {
        "ev_flat": (Strategy(f"ev>={min_ev}", min_ev=min_ev, **band), FlatStake(flat_unit)),
        "ev_kelly": (Strategy(f"ev>={min_ev}", min_ev=min_ev, **band), KellyStake(kelly_frac)),
        "edge10_flat": (Strategy("edge>=10%", min_edge_pct=0.10, **band), FlatStake(flat_unit)),
        "edge10_kelly": (Strategy("edge>=10%", min_edge_pct=0.10, **band), KellyStake(kelly_frac)),
        "band_all_flat": (Strategy("band-only", **band), FlatStake(flat_unit)),
    }


def main(argv=None) -> int:
    cfg = get_config()
    model_cfg = cfg.get("model", {}) or {}
    value_cfg = cfg.get("value", {}) or {}
    bt_cfg = cfg.get("bet_tracker", {}) or {}

    ap = argparse.ArgumentParser(description="Walk-forward value-betting backtest")
    ap.add_argument("--features-path", default=None, help="training parquet (default: data/features/training.parquet)")
    ap.add_argument("--min-train-days", type=int, default=365)
    ap.add_argument("--window-days", type=int, default=30, help="out-of-sample test window per fold")
    ap.add_argument("--step-days", type=int, default=None, help="origin roll step (default: window)")
    ap.add_argument("--rolling", action="store_true", help="rolling train window instead of expanding")
    ap.add_argument("--train-window-days", type=int, default=None, help="rolling train span")
    ap.add_argument("--price-col", default="ppwap", help="pre-off execution price column")
    ap.add_argument("--close-col", default="odds_finish", help="closing-line column for CLV")
    ap.add_argument("--initial-bankroll", type=float, default=float(bt_cfg.get("initial_bankroll", 1000.0)))
    ap.add_argument("--flat-stake", type=float, default=float(bt_cfg.get("flat_stake", 10.0)))
    ap.add_argument("--kelly-fraction", type=float, default=float(bt_cfg.get("kelly_fraction", 0.25)))
    ap.add_argument("--commission", type=float, default=0.0, help="exchange commission on winnings (0-1)")
    ap.add_argument("--stop-loss", type=float, default=None, help="halt new bets at this drawdown fraction")
    ap.add_argument("--iterations", type=int, default=500, help="CatBoost boosting rounds per fold")
    ap.add_argument("--task-type", default=str(model_cfg.get("task_type", "CPU")), help="GPU | CPU")
    ap.add_argument("--fl-recalibrate", action="store_true",
                    help="apply the favourite-longshot OddsBandCalibrator per fold "
                         "(mirrors the live value layer)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    panel = load_panel(path=args.features_path,
                       config=PanelConfig(price_col=args.price_col, close_col=args.close_col))

    factory = CatBoostFactory(
        feature_cols=list(PRICE_FREE_FEATURE_COLS),
        iterations=args.iterations,
        task_type=args.task_type,
        devices=str(model_cfg.get("devices", "0")),
        thread_count=int(model_cfg.get("thread_count", -1)),
        calibration_method=str(model_cfg.get("calibration_method", "auto")),
        random_seed=int(model_cfg.get("random_seed", 42)),
        fl_recalibrate=args.fl_recalibrate,
    )

    wf = WalkForwardConfig(
        min_train_days=args.min_train_days,
        test_window_days=args.window_days,
        step_days=args.step_days,
        rolling=args.rolling,
        train_window_days=args.train_window_days,
    )
    bt = Backtester(panel, factory, wf, BacktestConfig(
        initial_bankroll=args.initial_bankroll,
        commission=args.commission,
        stop_loss_pct=args.stop_loss,
    ))

    strategies = _default_strategies(value_cfg, args.flat_stake, args.kelly_fraction)
    run = bt.run(strategies)
    print(render_report(run))

    if not args.no_save:
        path = save_run(run, output_dir=args.output_dir)
        print(f"\nSaved run -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
