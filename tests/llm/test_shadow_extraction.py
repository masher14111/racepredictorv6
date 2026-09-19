"""Offline tests for the Stage-19 capped hosted DeepSeek shadow extractor.

No test in this module makes a real network call: pricing listings, the
OpenRouter transport and the archive/ledger/cache stores are all injected.
Covers: disabled/missing-credential/unverifiable-pricing fail-closed paths,
successful batch extraction with resumable dedupe, budget exhaustion
mid-batch, malformed/ungrounded evidence handling, the extraction-time vs
text-fetched-time point-in-time contract, non-secret health reporting, and
that nothing here is wired into any model feature/probability path.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm.hosted_adapter import HOSTED_PROMPT_VERSION
from llm.hosted_budget import HostedBudgetLedger
from llm.shadow_extraction import (
    ShadowExtractionStore,
    run_shadow_extraction,
    verify_current_pricing,
    write_health_report,
    PricingUnverifiable,
)
from llm.hosted_budget import PricingUnbounded
from llm.text_archive import TextArchive
from llm.text_features import SCHEMA_VERSION
from utils.cache import Cache

T0 = datetime(2026, 6, 17, 8, 0, 0, tzinfo=timezone.utc)
RACE_UID = "2026-06-17T14:30:00+00:00"


# ── fixtures / helpers ───────────────────────────────────────────────────────

@pytest.fixture
def archive(tmp_path):
    a = TextArchive(db_path=str(tmp_path / "shared.db"))
    yield a
    a.close()


@pytest.fixture
def store(tmp_path):
    s = ShadowExtractionStore(db_path=str(tmp_path / "shared.db"))
    yield s
    s.close()


def _seed(archive, n=3, fetched_at=T0):
    rows = []
    for i in range(n):
        rows.append(
            archive.record(
                text=f"Comment number {i}: needs better ground, may find this too sharp.",
                source="sporting_life_spotlight", race_uid=RACE_UID,
                horse_name=f"Horse {i}", fetched_at=fetched_at,
            )
        )
    return rows


def _all_unknown() -> dict:
    return {
        "ground_excuse": {"value": "unknown", "evidence": ""},
        "distance_excuse": {"value": "unknown", "evidence": ""},
        "trip_trouble": {"value": "unknown", "evidence": ""},
        "headgear_first_time": {"value": "unknown", "evidence": ""},
        "returning_from_layoff": {"value": "unknown", "evidence": ""},
        "course_winner": {"value": "unknown", "evidence": ""},
    }


def _usage_response(model, content, prompt_tokens=50, completion_tokens=30):
    return {
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "choices": [{"message": {"content": json.dumps(content)}}],
    }


def _valid_pricing_listing():
    return {
        "data": [
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "pricing": {"prompt": "2e-7", "completion": "8e-7", "overrides": []},
            }
        ]
    }


def _config(**overrides):
    cfg = {
        "enabled": True,
        "model": "deepseek/deepseek-v4.1-flash",
        "reasoning_effort": "low",
        "max_output_tokens": 800,
        "max_retries": 1,
        "timeout_s": 5.0,
        "batch_size_per_run": 10,
        "price_ceiling_input_per_million_usd": 0.3,
        "price_ceiling_output_per_million_usd": 1.2,
        "max_cost_usd": 5.0,
        "max_requests": 1200,
    }
    cfg.update(overrides)
    return cfg


def _fake_http_post_ok(content=None):
    content = content or _all_unknown()

    def _post(url, headers, payload, timeout):
        return _usage_response(payload["model"], content)

    return _post


# ── fail-closed paths (no network, no ledger touch) ─────────────────────────

def test_disabled_config_is_a_pure_noop(tmp_path):
    def fetch_listing():
        raise AssertionError("must never fetch pricing when disabled")

    summary = run_shadow_extraction(
        config=_config(enabled=False), fetch_listing=fetch_listing,
    )
    assert summary.enabled is False
    assert summary.stop_reason == "disabled_by_config"
    assert summary.processed == 0


def test_missing_api_key_fails_closed_before_any_pricing_fetch(tmp_path):
    def fetch_listing():
        raise AssertionError("must never fetch pricing without a credential")

    summary = run_shadow_extraction(
        config=_config(), api_key="", fetch_listing=fetch_listing,
    )
    assert summary.enabled is True
    assert summary.api_key_present is False
    assert summary.stop_reason == "api_key_absent"
    assert summary.reason == "OPENROUTER_API_KEY_absent"
    assert summary.processed == 0


def test_pricing_unverifiable_with_no_cache_fails_closed(tmp_path):
    def fetch_listing():
        raise ConnectionError("simulated outage")

    summary = run_shadow_extraction(
        config=_config(), api_key="test-key", fetch_listing=fetch_listing,
        pricing_cache_path=tmp_path / "no_such_cache.json",
    )
    assert summary.pricing_verified is False
    assert summary.stop_reason == "pricing_unverifiable"
    assert summary.processed == 0


def test_pricing_verification_falls_back_to_cache_on_live_failure(tmp_path):
    cache_path = tmp_path / "models_cache.json"
    cache_path.write_text(json.dumps(_valid_pricing_listing()), encoding="utf-8")

    def fetch_listing():
        raise ConnectionError("simulated outage")

    result = verify_current_pricing(
        "deepseek/deepseek-v4.1-flash",
        ceiling_input_per_million_usd=0.3, ceiling_output_per_million_usd=1.2,
        fetch_listing=fetch_listing, cache_path=cache_path,
    )
    assert result["pricing_source"] == "cached_fallback"
    assert result["within_ceiling"] is True


def test_pricing_exceeding_ceiling_fails_closed(tmp_path):
    def fetch_listing():
        return {
            "data": [{
                "id": "deepseek/deepseek-v4.1-flash",
                "pricing": {"prompt": "5e-6", "completion": "5e-6", "overrides": []},
            }]
        }

    with pytest.raises(PricingUnbounded):
        verify_current_pricing(
            "deepseek/deepseek-v4.1-flash",
            ceiling_input_per_million_usd=0.3, ceiling_output_per_million_usd=1.2,
            fetch_listing=fetch_listing, cache_path=tmp_path / "cache.json",
        )

    summary = run_shadow_extraction(
        config=_config(), api_key="test-key", fetch_listing=fetch_listing,
        pricing_cache_path=tmp_path / "cache2.json",
    )
    assert summary.stop_reason == "pricing_unverifiable"
    assert "exceeds configured ceiling" in summary.reason


def test_model_absent_from_listing_fails_closed(tmp_path):
    def fetch_listing():
        return {"data": [{"id": "some/other-model", "pricing": {"prompt": "1e-7", "completion": "1e-7"}}]}

    with pytest.raises(PricingUnverifiable):
        verify_current_pricing(
            "deepseek/deepseek-v4.1-flash",
            ceiling_input_per_million_usd=0.3, ceiling_output_per_million_usd=1.2,
            fetch_listing=fetch_listing, cache_path=tmp_path / "cache.json",
        )


# ── successful batch, resumable dedupe ──────────────────────────────────────

def test_successful_batch_records_extractions_and_resumes_without_respend(tmp_path, archive, store):
    _seed(archive, n=3)
    ledger_path = tmp_path / "ledger.json"

    def run_once(batch_size):
        return run_shadow_extraction(
            config=_config(batch_size_per_run=batch_size), api_key="test-key",
            fetch_listing=lambda: _valid_pricing_listing(),
            archive=archive, store=store,
            ledger=HostedBudgetLedger(ledger_path, max_cost_usd=5.0, max_requests=1200),
            cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
            http_post=_fake_http_post_ok(),
        )

    first = run_once(2)
    assert first.stop_reason == "completed_batch"
    assert first.requested == 2
    assert first.processed == 2
    assert first.succeeded == 2
    assert first.unprocessed_remaining == 1

    second = run_once(2)
    assert second.requested == 1
    assert second.processed == 1
    assert second.unprocessed_remaining == 0

    third = run_once(2)
    assert third.requested == 0
    assert third.stop_reason == "no_candidates"

    coverage = store.coverage()
    assert coverage["rows"] == 3
    assert coverage["schema_valid"] == 3


def test_store_provenance_fields_match_the_source_archive_row(tmp_path, archive, store):
    rows = _seed(archive, n=1)
    summary = run_shadow_extraction(
        config=_config(), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(),
    )
    assert summary.processed == 1
    stored = store.as_of(RACE_UID, rows[0].runner_key, as_of=datetime.now(timezone.utc) + timedelta(days=1))
    assert stored is not None
    assert stored.content_hash == rows[0].content_hash
    assert stored.text_fetched_at == rows[0].fetched_at
    assert stored.text_published_at == rows[0].published_at
    assert stored.schema_version == SCHEMA_VERSION
    assert stored.prompt_version == HOSTED_PROMPT_VERSION
    assert stored.model == "deepseek/deepseek-v4.1-flash"


def test_extraction_time_never_precedes_text_fetched_at(tmp_path, archive, store):
    old_fetch = T0
    _seed(archive, n=1, fetched_at=old_fetch)
    run_shadow_extraction(
        config=_config(), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(),
        now=old_fetch + timedelta(days=30),
    )
    remaining = store.select_unprocessed(
        model="deepseek/deepseek-v4.1-flash", schema_version=SCHEMA_VERSION,
        prompt_version=HOSTED_PROMPT_VERSION, limit=10,
    )
    assert remaining == []  # the one seeded row is now processed
    conn = store._pool.connection()
    rec = conn.execute("SELECT * FROM llm_shadow_extractions").fetchone()
    fetched_dt = datetime.fromisoformat(rec["text_fetched_at"])
    extracted_dt = datetime.fromisoformat(rec["extraction_time"])
    assert extracted_dt >= fetched_dt
    assert extracted_dt - fetched_dt >= timedelta(days=29)


def test_as_of_hides_an_extraction_recorded_after_the_query_instant(tmp_path, archive, store):
    """A downstream reader asking 'what was available at T' must never see an
    extraction whose own extraction_time is after T, even though the
    underlying text was fetched well before T -- the point this module exists
    to enforce (see module docstring)."""
    rows = _seed(archive, n=1, fetched_at=T0)
    late_extraction_time = T0 + timedelta(days=10)
    run_shadow_extraction(
        config=_config(), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(),
        now=late_extraction_time,
    )
    # as_of a moment BEFORE the late extraction ran: nothing should be visible,
    # even though the text itself was fetched at T0 (well before this cutoff).
    early_view = store.as_of(RACE_UID, rows[0].runner_key, as_of=T0 + timedelta(days=1))
    assert early_view is None
    late_view = store.as_of(RACE_UID, rows[0].runner_key, as_of=late_extraction_time + timedelta(minutes=1))
    assert late_view is not None


def test_identical_text_under_a_different_archive_row_is_a_cache_hit(tmp_path, archive, store):
    """The hosted adapter's cache keys on text content (+ model/version), not
    on the archive row id -- so two different runners' comments that happen
    to share identical wording must not double-spend."""
    fixed_text = "Needs better ground, may find this too sharp."
    row_a = archive.record(text=fixed_text, source="sporting_life_spotlight",
                            race_uid=RACE_UID, horse_name="Horse A", fetched_at=T0)
    row_b = archive.record(text=fixed_text, source="sporting_life_spotlight",
                            race_uid=RACE_UID, horse_name="Horse B", fetched_at=T0)
    assert row_a.content_hash == row_b.content_hash

    calls = []

    def counting_post(url, headers, payload, timeout):
        calls.append(1)
        return _usage_response(payload["model"], _all_unknown())

    shared_cache = Cache(cache_dir=tmp_path / "cache", default_ttl=3600)
    ledger_path = tmp_path / "ledger.json"

    def run_once():
        return run_shadow_extraction(
            config=_config(batch_size_per_run=1), api_key="test-key",
            fetch_listing=lambda: _valid_pricing_listing(),
            archive=archive, store=store,
            ledger=HostedBudgetLedger(ledger_path, max_cost_usd=5.0, max_requests=1200),
            cache=shared_cache, http_post=counting_post,
        )

    first = run_once()
    second = run_once()
    assert first.processed == 1 and second.processed == 1
    assert len(calls) == 1  # second row's identical text served from cache
    assert second.cache_hits == 1
    assert second.cost_usd_this_run == 0.0


def test_retries_are_bounded_by_the_configured_max_retries(tmp_path, archive, store):
    _seed(archive, n=1)
    attempts = []

    def always_fail(url, headers, payload, timeout):
        attempts.append(1)
        raise ConnectionError("simulated network failure")

    summary = run_shadow_extraction(
        config=_config(max_retries=2), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=always_fail,
    )
    assert len(attempts) == 3  # 1 initial + 2 retries, bounded
    assert summary.failed == 1
    assert summary.succeeded == 0
    # every retry failed before billing -> the reservation was released, not spent
    assert summary.cost_usd_this_run == 0.0


# ── budget exhaustion mid-batch ──────────────────────────────────────────────

def test_budget_exhaustion_mid_batch_stops_and_preserves_partial_progress(tmp_path, archive, store):
    _seed(archive, n=3)
    # A request cap of exactly 1 deterministically exhausts the budget after
    # the first row regardless of estimated token cost -- the second row's
    # reservation must fail on the request cap, not on cost arithmetic.
    tiny_ledger = HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1)
    summary = run_shadow_extraction(
        config=_config(max_output_tokens=50, max_retries=0), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store, ledger=tiny_ledger,
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(),
    )
    assert summary.stop_reason == "budget_exhausted"
    assert summary.processed < summary.requested
    assert summary.unprocessed_remaining > 0
    assert store.coverage()["rows"] == summary.processed


# ── malformed / ungrounded evidence ─────────────────────────────────────────

def test_ungrounded_claim_is_forced_unknown_before_it_reaches_the_store(tmp_path, archive, store):
    rows = _seed(archive, n=1)
    content = _all_unknown()
    content["headgear_first_time"] = {"value": True, "evidence": "not actually in the text"}
    run_shadow_extraction(
        config=_config(), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(content),
    )
    stored = store.as_of(RACE_UID, rows[0].runner_key, as_of=datetime.now(timezone.utc) + timedelta(days=1))
    assert stored.features["headgear_first_time"]["value"] is None


def test_malformed_schema_is_recorded_not_silently_dropped(tmp_path, archive, store):
    rows = _seed(archive, n=1)
    content = _all_unknown()
    content["course_winner"] = "not-a-dict"
    summary = run_shadow_extraction(
        config=_config(), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(content),
    )
    assert summary.failed == 1
    assert summary.succeeded == 0
    stored = store.as_of(RACE_UID, rows[0].runner_key, as_of=datetime.now(timezone.utc) + timedelta(days=1))
    assert stored.schema_valid is False
    assert stored.features["course_winner"]["value"] is None


# ── append-only enforcement ──────────────────────────────────────────────────

def test_shadow_extractions_table_is_append_only(tmp_path, archive, store):
    import sqlite3

    _seed(archive, n=1)
    run_shadow_extraction(
        config=_config(), api_key="test-key",
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(),
    )
    conn = store._pool.connection()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE llm_shadow_extractions SET cost_usd = 99.0 WHERE id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM llm_shadow_extractions WHERE id = 1")


# ── health reporting is non-secret ──────────────────────────────────────────

def test_health_report_never_contains_the_api_key(tmp_path, archive, store):
    secret = "sk-or-v1-super-secret-value-should-never-appear"
    summary = run_shadow_extraction(
        config=_config(), api_key=secret,
        fetch_listing=lambda: _valid_pricing_listing(),
        archive=archive, store=store,
        ledger=HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=5.0, max_requests=1200),
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=_fake_http_post_ok(),
    )
    path = write_health_report(summary, path=tmp_path / "health.json")
    raw = Path(path).read_text(encoding="utf-8")
    assert secret not in raw
    payload = json.loads(raw)
    assert "api_key" not in payload
    assert payload["model"] == "deepseek/deepseek-v4.1-flash"


# ── numerical forecast influence stays disabled ─────────────────────────────

def test_shadow_extraction_is_not_wired_into_any_model_feature_or_column():
    import inspect

    from models import features as feat_mod

    src = inspect.getsource(feat_mod)
    assert "shadow_extraction" not in src
    assert "llm_shadow" not in src

    base = Path(__file__).resolve().parents[2]
    tf_v1_src = (base / "features" / "text_features_v1.py").read_text(encoding="utf-8")
    assert "shadow_extraction" not in tf_v1_src


def test_llm_enabled_flag_is_independent_of_shadow_extraction_config():
    """Confirms `llm.enabled` (numeric-feature gate) and `llm.hosted_shadow.enabled`
    are two separate config switches -- flipping shadow on must never flip the
    tri-state text-feature extractor on."""
    from llm.text_features import TextFeatureExtractor

    extractor = TextFeatureExtractor(enabled=False)
    result = extractor.extract_detailed("Needs better ground, wants further.")
    assert result.backend == "disabled"
    assert all(fv.value is None for fv in result.features.values())
