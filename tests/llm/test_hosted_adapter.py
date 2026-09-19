"""Mocked tests for the hosted extraction adapter — no network calls.

Proves (prompts/15 scope 4-5, before any live spend): strict grounding
validation, bounded retries, budget reservation/commit wiring, per-record
caching that never re-spends, pricing-ceiling fail-closed construction, and
the identity-substitution probe.
"""
import json

import pytest

from llm.hosted_adapter import OpenRouterBackend, PricingUnbounded, substitute_identity
from llm.hosted_budget import BudgetExceeded, HostedBudgetLedger
from utils.cache import Cache


def _usage_response(model, content, prompt_tokens=50, completion_tokens=30):
    return {
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "choices": [{"message": {"content": json.dumps(content)}}],
    }


def _all_unknown() -> dict:
    return {
        "ground_excuse": {"value": "unknown", "evidence": ""},
        "distance_excuse": {"value": "unknown", "evidence": ""},
        "trip_trouble": {"value": "unknown", "evidence": ""},
        "headgear_first_time": {"value": "unknown", "evidence": ""},
        "returning_from_layoff": {"value": "unknown", "evidence": ""},
        "course_winner": {"value": "unknown", "evidence": ""},
    }


def _ledger(tmp_path, max_cost_usd=5.0, max_requests=1200):
    return HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=max_cost_usd, max_requests=max_requests)


def _backend(tmp_path, http_post, **overrides):
    kwargs = dict(
        api_key="test-key-not-real",
        model="deepseek/deepseek-v4.1-flash",
        ledger=_ledger(tmp_path),
        price_input_per_token=0.3e-6,
        price_output_per_token=1.2e-6,
        price_ceiling_input_per_token=0.3e-6,
        price_ceiling_output_per_token=1.2e-6,
        max_output_tokens=800,
        max_retries=2,
        cache=Cache(cache_dir=tmp_path / "cache", default_ttl=3600),
        http_post=http_post,
    )
    kwargs.update(overrides)
    return OpenRouterBackend(**kwargs)


def test_pricing_above_ceiling_refuses_construction(tmp_path):
    with pytest.raises(PricingUnbounded):
        _backend(tmp_path, http_post=None, price_input_per_token=0.31e-6)
    with pytest.raises(PricingUnbounded):
        _backend(tmp_path, http_post=None, price_output_per_token=1.21e-6)


def test_grounded_true_claim_is_accepted(tmp_path):
    text = "Cheekpieces are added for the first time here."
    content = _all_unknown()
    content["headgear_first_time"] = {"value": True, "evidence": "first time"}
    calls = []

    def fake_post(url, headers, payload, timeout):
        calls.append(payload)
        return _usage_response("deepseek/deepseek-v4.1-flash", content)

    backend = _backend(tmp_path, http_post=fake_post)
    result = backend.extract(text)
    assert result.schema_valid
    assert result.features["headgear_first_time"].value is True
    assert result.features["headgear_first_time"].evidence == "first time"
    assert len(calls) == 1


def test_ungrounded_true_claim_is_forced_unknown(tmp_path):
    text = "Nothing about headgear in this comment at all."
    content = _all_unknown()
    # model claims a headgear fact with evidence NOT present in the text
    content["headgear_first_time"] = {"value": True, "evidence": "blinkers fitted"}

    def fake_post(url, headers, payload, timeout):
        return _usage_response("deepseek/deepseek-v4.1-flash", content)

    backend = _backend(tmp_path, http_post=fake_post)
    result = backend.extract(text)
    assert result.features["headgear_first_time"].value is None
    assert result.features["headgear_first_time"].evidence is None


def test_malformed_field_marks_schema_invalid_not_silently_dropped(tmp_path):
    content = _all_unknown()
    content["course_winner"] = "not-a-dict"  # malformed entry

    def fake_post(url, headers, payload, timeout):
        return _usage_response("deepseek/deepseek-v4.1-flash", content)

    backend = _backend(tmp_path, http_post=fake_post)
    result = backend.extract("some text")
    assert result.schema_valid is False
    assert result.features["course_winner"].value is None


def test_transport_failure_retries_bounded_then_fails_closed(tmp_path):
    attempts = []

    def always_fail(url, headers, payload, timeout):
        attempts.append(1)
        raise ConnectionError("simulated network failure")

    backend = _backend(tmp_path, http_post=always_fail, max_retries=2)
    result = backend.extract("text")
    assert len(attempts) == 3  # 1 initial + 2 retries, bounded
    assert result.schema_valid is False
    assert result.cost_usd == 0.0
    assert "transport_failed" in result.failure_reason
    # the failed reservation must have been released, not left dangling
    assert backend.ledger.snapshot()["reserved_cost_usd"] == 0.0


def test_cache_hit_never_reserves_or_spends_budget_again(tmp_path):
    content = _all_unknown()
    calls = []

    def fake_post(url, headers, payload, timeout):
        calls.append(1)
        return _usage_response("deepseek/deepseek-v4.1-flash", content)

    ledger_path = tmp_path / "ledger.json"
    cache = Cache(cache_dir=tmp_path / "cache", default_ttl=3600)
    backend = _backend(tmp_path, http_post=fake_post, ledger=HostedBudgetLedger(ledger_path, max_cost_usd=5.0, max_requests=1200), cache=cache)
    r1 = backend.extract("same text twice")
    spend_after_first = backend.ledger.snapshot()["committed_cost_usd"]
    assert spend_after_first > 0
    r2 = backend.extract("same text twice")
    assert r2.is_cache_hit is True
    assert len(calls) == 1  # second call never hit the network
    assert backend.ledger.snapshot()["committed_cost_usd"] == pytest.approx(spend_after_first)


def test_reservation_blocks_call_when_budget_exhausted(tmp_path):
    content = _all_unknown()

    def fake_post(url, headers, payload, timeout):
        return _usage_response("deepseek/deepseek-v4.1-flash", content)

    tiny_ledger = HostedBudgetLedger(tmp_path / "ledger.json", max_cost_usd=1e-9, max_requests=1200)
    backend = _backend(tmp_path, http_post=fake_post, ledger=tiny_ledger)
    with pytest.raises(BudgetExceeded):
        backend.extract("this call must never reach the network")


def test_provider_identity_mismatch_is_recorded_not_silently_trusted(tmp_path, caplog):
    content = _all_unknown()

    def fake_post(url, headers, payload, timeout):
        # provider silently served a different model than requested
        return _usage_response("some/other-model", content)

    backend = _backend(tmp_path, http_post=fake_post)
    result = backend.extract("text")
    assert result.model == "some/other-model"  # actual identity recorded, not assumed


def test_substitute_identity_replaces_horse_name_only():
    text = "Shout was a smooth winner here in September and shaped well after."
    out = substitute_identity(text, "Shout")
    assert "Shout" not in out
    assert "This Runner" in out
    assert "smooth winner here" in out  # rest of the sentence is untouched


def test_substitute_identity_noop_without_horse_name():
    text = "No name substitution should occur."
    assert substitute_identity(text, None) == text
