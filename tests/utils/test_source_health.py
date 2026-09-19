"""Tests for utils/source_health.py."""
import json

import pytest

import utils.source_health as sh


@pytest.fixture(autouse=True)
def _isolated_path(tmp_path, monkeypatch):
    """Point the module's disk-backed store at a per-test tmp file."""
    monkeypatch.setattr(sh, "_PATH", tmp_path / "source_health.json")
    yield


class TestRecordSuccess:
    def test_first_success_sets_status_ok(self):
        sh.record_success("boylesports", rows=120, races=8)
        rec = sh.get_health("boylesports")
        assert rec["status"] == sh.STATUS_OK
        assert rec["total_successes"] == 1
        assert rec["total_failures"] == 0
        assert rec["success_rate"] == 1.0
        assert rec["last_row_count"] == 120
        assert rec["last_race_count"] == 8
        assert rec["last_rejection_reason"] is None
        assert rec["last_success_at"] is not None

    def test_success_after_failures_clears_rejection_reason(self):
        sh.record_failure("boylesports", reason="timeout")
        sh.record_success("boylesports", rows=5, races=1)
        rec = sh.get_health("boylesports")
        assert rec["status"] == sh.STATUS_OK
        assert rec["last_rejection_reason"] is None

    def test_success_rate_is_cumulative(self):
        sh.record_success("boylesports")
        sh.record_success("boylesports")
        sh.record_failure("boylesports", reason="timeout")
        rec = sh.get_health("boylesports")
        assert abs(rec["success_rate"] - 2 / 3) < 1e-9
        assert rec["total_successes"] == 2
        assert rec["total_failures"] == 1


class TestRecordFailure:
    def test_failure_sets_status_failing(self):
        sh.record_failure("paddy_power", reason="bot_detected")
        rec = sh.get_health("paddy_power")
        assert rec["status"] == sh.STATUS_FAILING
        assert rec["total_failures"] == 1
        assert rec["last_rejection_reason"] == "bot_detected"

    def test_failure_does_not_touch_last_success_at(self):
        sh.record_success("paddy_power", rows=10, races=2)
        first = sh.get_health("paddy_power")["last_success_at"]
        sh.record_failure("paddy_power", reason="timeout")
        rec = sh.get_health("paddy_power")
        assert rec["last_success_at"] == first
        assert rec["status"] == sh.STATUS_FAILING


class TestRecordUnavailable:
    def test_unavailable_sets_status_and_reason(self):
        sh.record_unavailable("livescorebet", reason="proxy_unavailable")
        rec = sh.get_health("livescorebet")
        assert rec["status"] == sh.STATUS_UNAVAILABLE
        assert rec["last_rejection_reason"] == "proxy_unavailable"
        assert rec["total_failures"] == 1

    def test_unavailable_counts_toward_failure_total(self):
        sh.record_failure("livescorebet", reason="timeout")
        sh.record_unavailable("livescorebet", reason="proxy_unavailable")
        rec = sh.get_health("livescorebet")
        assert rec["total_failures"] == 2


class TestProxyReachable:
    def test_set_and_read_back(self):
        sh.set_proxy_reachable("boylesports", True)
        assert sh.get_health("boylesports")["proxy_reachable"] is True
        sh.set_proxy_reachable("boylesports", False)
        assert sh.get_health("boylesports")["proxy_reachable"] is False

    def test_defaults_to_none_when_never_set(self):
        sh.record_success("boylesports")
        assert sh.get_health("boylesports")["proxy_reachable"] is None


class TestGetHealth:
    def test_unknown_source_returns_default_record(self):
        rec = sh.get_health("nonexistent")
        assert rec["status"] is None
        assert rec["total_successes"] == 0
        assert rec["age_seconds"] is None

    def test_get_all_sources(self):
        sh.record_success("boylesports")
        sh.record_failure("paddy_power", reason="timeout")
        all_health = sh.get_health()
        assert set(all_health.keys()) == {"boylesports", "paddy_power"}
        assert all_health["boylesports"]["status"] == sh.STATUS_OK
        assert all_health["paddy_power"]["status"] == sh.STATUS_FAILING

    def test_age_seconds_derived_from_last_success(self):
        sh.record_success("boylesports")
        rec = sh.get_health("boylesports")
        assert rec["age_seconds"] is not None
        assert 0 <= rec["age_seconds"] < 5

    def test_age_seconds_none_before_any_success(self):
        sh.record_failure("boylesports", reason="timeout")
        rec = sh.get_health("boylesports")
        assert rec["age_seconds"] is None


class TestPersistence:
    def test_round_trip_across_module_calls(self):
        sh.record_success("boylesports", rows=50, races=6)
        # Simulate a second process reading the same file fresh.
        rec = sh.get_health("boylesports")
        assert rec["last_row_count"] == 50
        assert rec["last_race_count"] == 6

    def test_write_is_valid_json_on_disk(self):
        sh.record_success("boylesports", rows=1, races=1)
        raw = json.loads(sh._PATH.read_text(encoding="utf-8"))
        assert "boylesports" in raw
        assert raw["boylesports"]["total_successes"] == 1

    def test_no_leftover_tmp_file(self):
        sh.record_success("boylesports")
        sh.record_failure("boylesports", reason="timeout")
        assert not list(sh._PATH.parent.glob("*.tmp"))

    def test_missing_file_returns_empty_all(self):
        assert sh.get_health() == {}
