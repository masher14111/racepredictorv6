"""Full setup pipeline — run once to go from zero to trained model.

Usage:
    python pipeline.py [--skip-scrape] [--no-tune]

    --skip-scrape   Skip the historical data fetch (use if parquets already exist)
    --no-tune       Skip Optuna hyperparameter search (much faster, slightly worse model)
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger("pipeline")


def step(label: str):
    logger.info("=" * 60)
    logger.info("STEP: %s", label)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    # ── 1. Initialise the database schema ────────────────────────────────────
    step("1/5  Initialise database")
    from utils.storage.migrations import _main as migrations_main
    migrations_main(["migrate"])
    logger.info("Database ready.")

    # ── 2. Scrape historical data ─────────────────────────────────────────────
    if not args.skip_scrape:
        step("2/5  Scrape Betfair SP historical data (this may take several minutes)")
        from scraper.betsp_historical import fetch as fetch_betsp
        df_betsp = fetch_betsp()
        logger.info("betsp_historical: %d rows", len(df_betsp))

        step("3/5  Scrape Timeform historical data (this may take several minutes)")
        from scraper.timeform_historical import fetch as fetch_tf
        df_tf = fetch_tf()
        logger.info("timeform_historical: %d rows", len(df_tf))
    else:
        step("2/5  Skipping scrape (--skip-scrape)")
        step("3/5  Skipping scrape (--skip-scrape)")

    # ── 3. Normalise sources into unified_races.parquet ───────────────────────
    step("4/5  Normalise sources → data/unified_races.parquet")
    from utils.normalizer import normalize
    unified = normalize(write=True)
    logger.info("normalizer: %d unified rows", len(unified))

    # ── 4. Build training feature matrix ──────────────────────────────────────
    step("5/5  Build training feature matrix → data/features/training.parquet")
    from features.builder import build_training_matrix
    training = build_training_matrix(unified=unified, write=True)
    logger.info("training matrix: %d labelled rows", len(training))

    if len(training) == 0:
        logger.error(
            "Training matrix is empty — no labelled rows (finishing positions) found. "
            "The historical scrapers returned data but none of it has results yet. "
            "Wait for today's races to complete, then re-run."
        )
        sys.exit(1)

    # ── 5. Train the model ───────────────────────────────────────────────────
    step("5/5  Train model")
    from models.train import train
    train(df=training, no_tune=args.no_tune)
    logger.info("Training complete. Models saved to models/")

    logger.info("")
    logger.info("Pipeline finished. Now run:")
    logger.info("  python -m models.predictor   <- generate today's predictions")
    logger.info("  streamlit run ui/app.py       <- open the dashboard")


if __name__ == "__main__":
    main()
