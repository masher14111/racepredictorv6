"""Task 12 — price-free (value-betting) model comparison.

Trains the price-free variant (version_tag ``v3nf`` — all market-price features
dropped) and a same-run full-feature control, then prints overall AUC and
per-odds-band AUC side by side. The point of value betting is that the model's
probability must be *independent of the market price*; otherwise EV-vs-market is
circular. AUC is expected to drop — the question is by how much, and where.

The deliverable v3nf artefacts are written to ``models/``. The full-feature
control is written to a temp dir (throwaway) so it shares the exact CatBoost
version / code path as v3nf, giving a confound-free head-to-head. The on-disk
``v3`` meta (current production model) is also shown for reference.

Run:  python -m scripts.compare_pricefree
This does NOT change the default model — config still points at ``v3``.
"""
import json
import os
import tempfile

from features.builder import build_training_matrix
from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS, PRICE_FEATURE_COLS
from models.train import train
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_BANDS = ["favourite", "mid", "underdog"]


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else " n/a "


def _row(label, full, nf):
    """One metric row: full value, price-free value, delta."""
    delta = ""
    if isinstance(full, (int, float)) and isinstance(nf, (int, float)):
        delta = f"{nf - full:+.4f}"
    return f"  {label:<12} {_fmt(full):>8}   {_fmt(nf):>8}   {delta:>8}"


def _print_block(title, full_meta, nf_meta):
    print(f"\n=== {title} ===")
    print(f"  {'metric':<12} {'full':>8}   {'price-free':>8}   {'delta':>8}")
    for target in ("won", "placed_2", "showed"):
        ft = (full_meta or {}).get("targets", {}).get(target, {})
        nt = (nf_meta or {}).get("targets", {}).get(target, {})
        print(f"  -- {target} --")
        print(_row("AUC", ft.get("test_auc"), nt.get("test_auc")))
        fb = ft.get("band_aucs", {}) or {}
        nb = nt.get("band_aucs", {}) or {}
        for band in _BANDS:
            print(_row(band, fb.get(band), nb.get(band)))


def main():
    logger.info("compare_pricefree: building training matrix once")
    df = build_training_matrix(write=False)
    print(f"training matrix: {len(df):,} rows")
    print(f"dropped price features ({len(PRICE_FEATURE_COLS)}): {PRICE_FEATURE_COLS}")
    print(f"price-free whitelist: {len(PRICE_FREE_FEATURE_COLS)} cols "
          f"(full: {len(FEATURE_COLS)})")

    # Price-free variant — the deliverable, saved to models/ as v3nf.
    logger.info("compare_pricefree: training v3nf (price-free)")
    nf_meta = train(df=df, no_tune=True,
                    feature_cols=PRICE_FREE_FEATURE_COLS, version_tag="v3nf")

    # Full-feature control — same code path / CatBoost version, throwaway dir.
    with tempfile.TemporaryDirectory() as tmp:
        logger.info("compare_pricefree: training full-feature control (temp)")
        full_meta = train(df=df, no_tune=True,
                          feature_cols=FEATURE_COLS, version_tag="v3cmp",
                          model_dir=tmp)

    # On-disk current production model for reference.
    v3_path = os.path.join(_BASE, "models", "catboost_v3_meta.json")
    v3_meta = None
    if os.path.exists(v3_path):
        with open(v3_path, "r", encoding="utf-8") as fh:
            v3_meta = json.load(fh)

    _print_block("full (this run, untuned) vs price-free v3nf", full_meta, nf_meta)
    if v3_meta is not None:
        _print_block("on-disk v3 (current default) vs price-free v3nf",
                     v3_meta, nf_meta)

    print("\nv3nf artefacts written to models/catboost_{won,placed_2,showed}_v3nf.bin")
    print("default model UNCHANGED (config version_tag still 'v3').")


if __name__ == "__main__":
    main()
