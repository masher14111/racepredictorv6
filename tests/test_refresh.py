"""Tests for scripts/refresh.py — destination-health wiring and telemetry
printing (Stage 2 acceptance items 1 and 5), plus live snapshot capture
(Stage 6 requirement 1/6). No network access."""
from unittest.mock import patch

from scripts.refresh import (
    _capture_snapshots, _check_destination_health, _print_source_health,
    _scrape_declarations, _scrape_spotlight, _shadow_extract_text,
)
from utils import source_health


def test_check_destination_health_records_results_per_source():
    fake_results = {"livescorebet": True, "paddy_power": True, "boylesports": False}
    with patch("utils.proxy_manager.get_proxy_manager") as mock_get_mgr:
        mock_get_mgr.return_value.check_destinations.return_value = fake_results
        _check_destination_health()

    health = source_health.get_health()
    assert health["livescorebet"]["proxy_reachable"] is True
    assert health["paddy_power"]["proxy_reachable"] is True
    assert health["boylesports"]["proxy_reachable"] is False


def test_check_destination_health_probe_failure_is_non_fatal():
    with patch("utils.proxy_manager.get_proxy_manager") as mock_get_mgr:
        mock_get_mgr.return_value.check_destinations.side_effect = RuntimeError("boom")
        _check_destination_health()  # must not raise

    assert source_health.get_health() == {}


def test_print_source_health_handles_empty_and_populated(capsys):
    _print_source_health()
    assert capsys.readouterr().out == ""

    source_health.record_success("livescorebet", rows=10, races=2)
    source_health.record_failure("boylesports", reason="bot_detected")
    _print_source_health()
    out = capsys.readouterr().out
    assert "livescorebet" in out and "status=ok" in out
    assert "boylesports" in out and "last_rejection=bot_detected" in out


def test_capture_snapshots_failure_does_not_break_the_scrape(capsys):
    """A capture failure must never propagate out and fail the scrape it rides
    on — it is best-effort telemetry, not a precondition for the day's card."""
    with patch("execution.snapshots.ingest_live_odds_parquet") as mock_ingest:
        mock_ingest.side_effect = RuntimeError("boom")
        _capture_snapshots()  # must not raise

    assert "FAILED" in capsys.readouterr().out


def test_scrape_declarations_failure_is_non_fatal(capsys):
    """Stage 05: Timeform (WAF/paywall/no-browser-fallback) is best-effort — a
    failed declared-card fetch must never break the odds-only refresh."""
    with patch("scraper.timeform_historical.fetch") as mock_fetch:
        mock_fetch.side_effect = RuntimeError("all tiers failed/blocked")
        _scrape_declarations()  # must not raise

    assert "FAILED" in capsys.readouterr().out


def test_scrape_declarations_reports_row_count(capsys):
    import pandas as pd

    with patch("scraper.timeform_historical.fetch") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame([{"horse_name": "A"}])
        _scrape_declarations()

    out = capsys.readouterr().out
    assert "timeform" in out and "1 declared runner row(s)" in out


def test_scrape_spotlight_failure_is_non_fatal(capsys):
    """Stage 19: without this best-effort call the text archive never gains a
    new dated day (it never ran anywhere in the daily pipeline before this
    stage) -- but a scrape failure (WAF block, no cards yet) must still never
    break the odds-only refresh it rides alongside."""
    with patch("scraper.spotlight.scrape") as mock_scrape:
        mock_scrape.side_effect = RuntimeError("blocked")
        _scrape_spotlight()  # must not raise

    assert "FAILED" in capsys.readouterr().out


def test_scrape_spotlight_reports_row_count(capsys):
    import pandas as pd

    with patch("scraper.spotlight.scrape") as mock_scrape:
        mock_scrape.return_value = pd.DataFrame([{"horse_name": "A"}, {"horse_name": "B"}])
        _scrape_spotlight()

    out = capsys.readouterr().out
    assert "spotlight" in out and "2 archived comment(s)" in out


def test_shadow_extract_text_failure_is_non_fatal(capsys):
    """A hosted-provider/network exception raised out of run_shadow_extraction
    itself (not one of its own labelled fail-closed returns) must still never
    break the refresh it rides on."""
    with patch("llm.shadow_extraction.run_shadow_extraction") as mock_run:
        mock_run.side_effect = RuntimeError("boom")
        _shadow_extract_text()  # must not raise

    assert "FAILED" in capsys.readouterr().out


def test_shadow_extract_text_prints_non_secret_summary(capsys):
    from llm.shadow_extraction import ShadowExtractionSummary

    summary = ShadowExtractionSummary(
        enabled=True, api_key_present=True, pricing_verified=True,
        model="deepseek/deepseek-v4.1-flash", reason=None, stop_reason="no_candidates",
        requested=0, processed=0, succeeded=0, failed=0, cache_hits=0,
        unprocessed_remaining=0, cost_usd_this_run=0.0, budget_snapshot={},
        started_at="t0", finished_at="t1",
    )
    with patch("llm.shadow_extraction.run_shadow_extraction", return_value=summary), \
         patch("llm.shadow_extraction.write_health_report") as mock_write:
        _shadow_extract_text()

    mock_write.assert_called_once_with(summary)
    out = capsys.readouterr().out
    assert "shadow extract" in out and "stop=no_candidates" in out
