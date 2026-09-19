"""One-shot daily refresh: scrape live racecards -> normalize -> build -> predict.

This is the intended daily flow that keeps `data/unified_races.parquet` (and the
prediction cache) current. Without it the GUI shows nothing once the on-disk data
ages past today.

    python -m scripts.refresh            # full pipeline
    python -m scripts.refresh --no-scrape  # skip scraping, just re-normalize+predict

Pipeline:
  1. scrape  — live odds scrapers write data/live_odds.parquet (livescorebet is the
               verified racecard source; boylesports is best-effort) and paddy_power.json;
               then a best-effort Timeform fetch (Stage 5) writes today's declared
               jockey/trainer/draw/weight/OR/equipment into data/historical/timeform.parquet
  2. normalize — utils.normalizer merges those sources into data/unified_races.parquet
                 (race_date derived from race_time, position left null => "upcoming")
  3. build   — features.builder.build_inference_matrix() derives today's live runners
  4. predict — models.predictor writes data/predictions.json
  5. settle  — auto-settle pending paper bets against newly-scraped SP results
               (records closing-line value; powers the live-CLV test)

Each scraper is best-effort: a failure is logged and the pipeline continues, so one
dead bookie never blocks the refresh.
"""
import argparse
import os
import sys

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils.logger import get_logger
from utils.proxy_manager import redact_secrets

logger = get_logger(__name__)


def _result_size(result) -> int:
    """How much a scraper actually returned, for the console line.

    livescorebet/boylesports hand back a DataFrame, where ``len`` is the row
    count. Paddy Power hands back ``{"fetched_at": ..., "races": [...]}`` — so a
    bare ``len`` counted its two dict KEYS and reported "2 rows" on every run,
    however much it had scraped. That made the healthiest source look dead.
    """
    if result is None:
        return 0
    if isinstance(result, dict):
        return sum(
            len(market.get("selections") or [])
            for race in (result.get("races") or [])
            for market in (race.get("markets") or [])
        )
    try:
        return len(result)
    except TypeError:
        return 0


def _scrape() -> None:
    """Run the live odds scrapers, tolerating per-source failure."""
    from scraper import livescorebet, boylesports, paddy_power

    for name, fn in (
        ("livescorebet", livescorebet.scrape),
        ("boylesports", boylesports.scrape),
        ("paddy_power", paddy_power.scrape),
    ):
        try:
            result = fn(force=True)
            print(f"  scrape {name:<13} -> {_result_size(result)} rows/selections")
        except Exception as exc:  # noqa: BLE001
            # redact_secrets: a connection-layer error from the HTTP client can
            # embed the proxy URL (with credentials) in its own str(); scraper
            # modules sanitize their own known error paths, but this catch-all
            # sees whatever bubbles up, so it must not trust it unredacted.
            msg = redact_secrets(str(exc))
            logger.warning("refresh: %s scrape failed (%s)", name, msg)
            print(f"  scrape {name:<13} -> FAILED ({msg})")

    _scrape_declarations()
    _scrape_spotlight()
    _check_destination_health()
    _print_source_health()
    _capture_snapshots()
    _shadow_extract_text()


def _scrape_declarations() -> None:
    """Fetch today's declared racecard (jockey/trainer/draw/weight/OR/equipment)
    from Timeform, best-effort (Stage 5). The odds scrapers above carry only
    horse + price, so without this step every live jockey/trainer field is a
    last-known historical guess (features.builder._fill_last_known_connections)
    rather than today's actual declaration. A failure here (WAF block, no
    browser fallback available, etc.) must not break the odds-only refresh."""
    from scraper.timeform_historical import fetch as fetch_declarations

    try:
        df = fetch_declarations()
        print(f"  scrape {'timeform':<13} -> {len(df)} declared runner row(s)")
    except Exception as exc:  # noqa: BLE001
        msg = redact_secrets(str(exc))
        logger.warning("refresh: timeform declarations failed (%s)", msg)
        print(f"  scrape {'timeform':<13} -> FAILED ({msg})")


def _scrape_spotlight() -> None:
    """Fetch today's Sporting Life spotlight commentary and append it into the
    immutable text archive (Stage 14), best-effort (Stage 19). Without this
    call the archive never gains a new dated day -- it stayed frozen at its
    single 2026-06-17 backfilled sample (D46/D48) because nothing in the daily
    pipeline had ever invoked ``scraper.spotlight`` before this stage. A
    failure here (WAF block, no UK/IRE cards yet) must not break the rest of
    the refresh; it only means today has nothing new to shadow-extract."""
    from scraper import spotlight

    try:
        df = spotlight.scrape()
        print(f"  scrape {'spotlight':<13} -> {len(df)} archived comment(s)")
    except Exception as exc:  # noqa: BLE001
        msg = redact_secrets(str(exc))
        logger.warning("refresh: spotlight scrape failed (%s)", msg)
        print(f"  scrape {'spotlight':<13} -> FAILED ({msg})")


def _shadow_extract_text() -> None:
    """Run one bounded batch of capped hosted DeepSeek shadow extraction over
    unprocessed archived comments (Stage 19). Best-effort and fail-closed:
    disabled config, an absent OPENROUTER_API_KEY, unverifiable pricing or an
    exhausted programme cap each print a labelled, non-secret status line and
    never raise -- this step must never abort the odds-only refresh it rides
    on, exactly like every scraper above it."""
    from llm.shadow_extraction import run_shadow_extraction, write_health_report

    try:
        summary = run_shadow_extraction()
        write_health_report(summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh: shadow extraction step raised (%s)", exc)
        print(f"  shadow extract -> FAILED ({exc})")
        return
    print(
        f"  shadow extract -> stop={summary.stop_reason} "
        f"processed={summary.processed} succeeded={summary.succeeded} "
        f"failed={summary.failed} cache_hits={summary.cache_hits} "
        f"cost=${summary.cost_usd_this_run:.4f} remaining_unprocessed={summary.unprocessed_remaining}"
    )


def _capture_snapshots() -> None:
    """Append this poll's board to the immutable odds_snapshots store (Stage 6).

    Best-effort: a capture failure must never break the scrape it rides on. Each
    poll gets its own ``fetched_at``, so re-running this within the same day adds
    new rows rather than overwriting the last one — that is what turns one
    pre-off quote into a real intraday price series.
    """
    from execution.snapshots import ingest_live_odds_parquet

    try:
        result = ingest_live_odds_parquet()
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh: snapshot capture failed (%s)", exc)
        print(f"  capture       -> FAILED ({exc})")
        return
    print(
        f"  capture       -> {result.inserted} inserted, "
        f"{result.duplicates} duplicate, {result.rejected} rejected"
    )


def _check_destination_health() -> None:
    """Probe each bookmaker's own URL and record it against source_health.

    ``check_destinations()`` never mutates proxy blacklist state (a bookmaker
    being briefly down is not the proxy's fault) — it only reports reachability
    for the operator. Best-effort: a probe failure must not fail the refresh.
    """
    from utils import source_health
    from utils.proxy_manager import get_proxy_manager

    try:
        results = get_proxy_manager().check_destinations()
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh: destination health check failed (%s)", redact_secrets(str(exc)))
        return
    for name, reachable in results.items():
        source_health.set_proxy_reachable(name, reachable)


def _print_source_health() -> None:
    """Per-source telemetry after a scrape pass (audit req 5) — the cron/CLI
    path has no dashboard, so this is its only visibility into which sources
    are actually healthy versus quietly degraded."""
    from utils import source_health

    health = source_health.get_health()
    if not health:
        return
    # `success_rate` is a LIFETIME PER-REQUEST ratio: every retried 403 on an
    # individual event page counts as a failure, while a whole scrape counts as
    # one success. So a partly-blocked source like BoyleSports can sit at 1%
    # while its scrape genuinely succeeded — "status=ok success_rate=1%" read as
    # a contradiction when it was really two different units side by side.
    # Label it for what it is and lead with what this run actually returned.
    print("  source health:")
    for name, rec in sorted(health.items()):
        age = rec.get("age_seconds")
        age_str = f"{age:.0f}s" if isinstance(age, (int, float)) else "never"
        rate = rec.get("success_rate")
        rate_str = f"{rate:.0%}" if isinstance(rate, (int, float)) else "n/a"
        rows = rec.get("last_row_count")
        races = rec.get("last_race_count")
        got = (
            f"{rows if rows is not None else '?'} rows"
            f"/{races if races is not None else '?'} races"
        )
        print(
            f"    {name:<13} status={rec.get('status') or 'unknown':<11} "
            f"last_run={got:<20} age={age_str:<8} "
            f"request_success={rate_str:<5} "
            f"last_rejection={rec.get('last_rejection_reason')}"
        )
    print(
        "    (request_success is per HTTP request over all time — individual "
        "event-page retries count, so a blocked-but-recovered source reads low)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-scrape", action="store_true",
                    help="skip the scraping step (re-normalize + predict only)")
    args = ap.parse_args()

    print("Race Predictor v3 — daily refresh")

    if args.no_scrape:
        print("[1/5] scrape   — skipped (--no-scrape)")
    else:
        print("[1/5] scrape   — fetching live racecards")
        _scrape()

    print("[2/5] normalize — merging sources into unified_races.parquet")
    from utils.normalizer import normalize
    unified = normalize(write=True)

    print("[3/5] build     — deriving today's live inference matrix")
    from features.builder import build_inference_matrix
    live = build_inference_matrix(unified=unified)
    print(f"  live runners: {len(live)} across "
          f"{live[['race_date', 'venue', 'race_time']].drop_duplicates().shape[0] if len(live) else 0} races")
    if live.empty:
        print("  WARNING: no upcoming races — scrapers returned nothing for today.")

    print("[4/5] predict   — scoring and writing predictions.json")
    from models.predictor import Predictor
    predictor = Predictor()
    if not predictor.load():
        print("  ERROR: no models loaded — cannot predict.")
        return 1
    races = predictor.predict(unified=unified)
    print(f"  predicted {len(races)} races -> data/predictions.json")

    print("[5/5] settle    — settling pending paper bets vs SP results")
    _settle()
    return 0


def _settle() -> None:
    """Settle pending paper bets against newly-scraped SP results (best-effort).

    Records closing-line value on each settled bet, which is what the live-CLV
    test (``scripts.clv_live``) reads. Never fatal — a settlement failure must not
    break the daily refresh."""
    try:
        from utils.bet_tracker import BetTracker
        from utils.bet_settlement import settle_from_storage

        settled = settle_from_storage(BetTracker())
        print(f"  settled {len(settled)} pending bet(s) against SP results")
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh: settlement step failed (%s)", exc)
        print(f"  settle -> skipped ({exc})")


if __name__ == "__main__":
    raise SystemExit(main())
