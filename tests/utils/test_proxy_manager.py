"""Tests for utils/proxy_manager.py."""
import logging
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from utils.proxy_manager import (
    ProxyManager,
    ProxyUnavailableError,
    get_proxy_manager,
    redact_proxy_url,
    redact_secrets,
    _on_config_change,
    _reset_singleton,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mgr(proxies=None, enabled=True, max_failures=3, blacklist_cooldown=300,
         health_check_interval=0):
    cfg = {
        "enabled": enabled,
        "proxies": proxies if proxies is not None else ["http://proxy1:8080", "http://proxy2:8080"],
        "max_failures": max_failures,
        "blacklist_cooldown": blacklist_cooldown,
        "health_check_interval": health_check_interval,
        "health_check_url": "https://icanhazip.com",
    }
    return ProxyManager(cfg=cfg)


# ---------------------------------------------------------------------------
# next() — basic rotation
# ---------------------------------------------------------------------------

class TestNext:
    def test_returns_none_when_disabled(self):
        m = _mgr(enabled=False)
        assert m.next() is None

    def test_returns_none_when_no_proxies(self):
        m = _mgr(proxies=[], enabled=True)
        assert m.next() is None

    def test_round_robin(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"])
        assert m.next() == "http://p1:1"
        assert m.next() == "http://p2:2"
        assert m.next() == "http://p1:1"

    def test_skips_blacklisted(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"])
        # blacklist p1 manually
        m._entries[0].blacklisted_until = time.monotonic() + 9999
        assert m.next() == "http://p2:2"
        assert m.next() == "http://p2:2"

    def test_returns_none_when_all_blacklisted(self):
        m = _mgr(proxies=["http://p1:1"])
        m._entries[0].blacklisted_until = time.monotonic() + 9999
        assert m.next() is None

    def test_returns_proxy_after_blacklist_expires(self):
        m = _mgr(proxies=["http://p1:1"])
        m._entries[0].blacklisted_until = time.monotonic() - 1  # already expired
        assert m.next() == "http://p1:1"


# ---------------------------------------------------------------------------
# report_success / report_failure
# ---------------------------------------------------------------------------

class TestReporting:
    def test_success_increments_counter(self):
        m = _mgr(proxies=["http://p1:1"])
        m.report_success("http://p1:1")
        assert m._entries[0].total_successes == 1
        assert m._entries[0].consecutive_failures == 0

    def test_success_clears_blacklist(self):
        m = _mgr(proxies=["http://p1:1"])
        m._entries[0].blacklisted_until = time.monotonic() + 9999
        m.report_success("http://p1:1")
        assert m._entries[0].blacklisted_until == 0.0

    def test_failure_increments_counter(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=5)
        m.report_failure("http://p1:1")
        assert m._entries[0].total_failures == 1
        assert m._entries[0].consecutive_failures == 1
        assert m._entries[0].blacklisted_until == 0.0  # not yet blacklisted

    def test_failure_blacklists_after_threshold(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=3, blacklist_cooldown=300)
        for _ in range(3):
            m.report_failure("http://p1:1")
        assert m._entries[0].blacklisted_until > time.monotonic()

    def test_noop_on_none_proxy(self):
        m = _mgr(proxies=["http://p1:1"])
        m.report_success(None)
        m.report_failure(None)
        assert m._entries[0].total_successes == 0
        assert m._entries[0].total_failures == 0

    def test_noop_on_unknown_proxy(self):
        m = _mgr(proxies=["http://p1:1"])
        m.report_success("http://unknown:9")
        m.report_failure("http://unknown:9")
        assert m._entries[0].total_successes == 0

    def test_success_after_failures_resets_consecutive(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=5)
        m.report_failure("http://p1:1")
        m.report_failure("http://p1:1")
        m.report_success("http://p1:1")
        assert m._entries[0].consecutive_failures == 0


# ---------------------------------------------------------------------------
# Blacklist cooldown — failover then recovery
# ---------------------------------------------------------------------------

class TestBlacklistFailover:
    def test_falls_back_to_second_proxy_after_blacklist(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"], max_failures=2)
        m.report_failure("http://p1:1")
        m.report_failure("http://p1:1")  # now blacklisted
        assert m.next() == "http://p2:2"

    def test_all_proxies_blacklisted_returns_none(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"], max_failures=1)
        m.report_failure("http://p1:1")
        m.report_failure("http://p2:2")
        assert m.next() is None

    def test_blacklist_with_zero_cooldown_expires_immediately(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=1, blacklist_cooldown=0)
        m.report_failure("http://p1:1")
        # With cooldown=0, blacklisted_until = monotonic() + 0 ≤ monotonic() now
        assert m.next() == "http://p1:1"


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------

class TestHealthCheck:
    CREDS = "http://someuser123:somepass456@gw.dataimpulse.com:823"

    @pytest.mark.parametrize("probe_ok", [True, False])
    def test_health_check_never_logs_the_credential(self, caplog, probe_ok):
        """redact_proxy_url existed and was used in the DEBUG paths, but these
        INFO/WARNING lines passed entry.url raw — and those are the ones that
        reach the file log. 48 plaintext credentials had accumulated in
        logs/race_predictor.log before this was caught."""
        m = _mgr(proxies=[self.CREDS])
        with caplog.at_level(logging.DEBUG):
            with patch.object(m, "_probe", return_value=probe_ok):
                m.health_check()
        blob = chr(10).join(r.getMessage() for r in caplog.records)
        assert "somepass456" not in blob
        assert "someuser123" not in blob
        assert "gw.dataimpulse.com:823" in blob, "the host must still be logged"

    def test_healthy_proxy_re_enabled(self):
        m = _mgr(proxies=["http://p1:1"])
        m._entries[0].blacklisted_until = time.monotonic() + 9999
        m._entries[0].consecutive_failures = 5

        with patch.object(m, "_probe", return_value=True):
            results = m.health_check()

        assert results["http://p1:1"] is True
        assert m._entries[0].blacklisted_until == 0.0
        assert m._entries[0].consecutive_failures == 0

    def test_unhealthy_proxy_gets_blacklisted(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch.object(m, "_probe", return_value=False):
            results = m.health_check()

        assert results["http://p1:1"] is False
        assert m._entries[0].blacklisted_until > time.monotonic()

    def test_multiple_proxies_mixed_results(self):
        m = _mgr(proxies=["http://good:1", "http://bad:2"])
        probe_results = {"http://good:1": True, "http://bad:2": False}
        with patch.object(m, "_probe", side_effect=lambda u, t: probe_results[u]):
            results = m.health_check()

        assert results["http://good:1"] is True
        assert results["http://bad:2"] is False
        assert m._entries[0].blacklisted_until == 0.0
        assert m._entries[1].blacklisted_until > time.monotonic()


# ---------------------------------------------------------------------------
# _probe_destination() — low-level per-URL reachability GET
# ---------------------------------------------------------------------------

def _mock_client(status_code=None, raises=None):
    """Build a MagicMock standing in for `httpx.Client(...)` used as a context manager."""
    client = MagicMock()
    if raises is not None:
        client.get.side_effect = raises
    else:
        client.get.return_value = MagicMock(status_code=status_code)
    ctx = MagicMock()
    ctx.__enter__.return_value = client
    ctx.__exit__.return_value = False
    return ctx


class TestProbeDestination:
    def test_2xx_is_reachable(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch("httpx.Client", return_value=_mock_client(status_code=200)):
            assert m._probe_destination("https://example.com", "http://p1:1", 5.0) is True

    def test_3xx_is_reachable(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch("httpx.Client", return_value=_mock_client(status_code=302)):
            assert m._probe_destination("https://example.com", "http://p1:1", 5.0) is True

    def test_4xx_challenge_is_unreachable(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch("httpx.Client", return_value=_mock_client(status_code=403)):
            assert m._probe_destination("https://example.com", "http://p1:1", 5.0) is False

    def test_5xx_is_unreachable(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch("httpx.Client", return_value=_mock_client(status_code=503)):
            assert m._probe_destination("https://example.com", "http://p1:1", 5.0) is False

    def test_exception_is_unreachable_not_raised(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch("httpx.Client", return_value=_mock_client(raises=TimeoutError("boom"))):
            assert m._probe_destination("https://example.com", "http://p1:1", 5.0) is False

    def test_no_proxy_probes_direct(self):
        m = _mgr(proxies=["http://p1:1"])
        captured = {}
        with patch("httpx.Client", side_effect=lambda **kw: (captured.update(kw), _mock_client(status_code=200))[1]):
            assert m._probe_destination("https://example.com", None, 5.0) is True
        assert captured["mounts"] is None

    def test_with_proxy_mounts_it(self):
        m = _mgr(proxies=["http://p1:1"])
        captured = {}
        with patch("httpx.Client", side_effect=lambda **kw: (captured.update(kw), _mock_client(status_code=200))[1]):
            m._probe_destination("https://example.com", "http://p1:1", 5.0)
        assert captured["mounts"] is not None
        assert "all://" in captured["mounts"]


# ---------------------------------------------------------------------------
# check_destinations() / destination_status() — per-bookmaker reachability
# ---------------------------------------------------------------------------

class TestCheckDestinations:
    def test_probes_each_configured_destination(self):
        m = _mgr(proxies=["http://p1:1"])
        m._destination_urls = {"boylesports": "https://boylesports.com", "paddy_power": "https://paddypower.com"}
        with patch.object(m, "_probe_destination", side_effect=lambda url, proxy, t: "boyle" in url):
            results = m.check_destinations()
        assert results == {"boylesports": True, "paddy_power": False}

    def test_explicit_destinations_overrides_configured(self):
        m = _mgr(proxies=["http://p1:1"])
        m._destination_urls = {"boylesports": "https://boylesports.com"}
        with patch.object(m, "_probe_destination", return_value=True) as probe:
            results = m.check_destinations({"livescorebet": "https://livescorebet.com"})
        assert results == {"livescorebet": True}
        probe.assert_called_once()
        assert probe.call_args[0][0] == "https://livescorebet.com"

    def test_result_cached_and_readable_via_destination_status(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch.object(m, "_probe_destination", return_value=True):
            m.check_destinations({"boylesports": "https://boylesports.com"})
        assert m.destination_status() == {"boylesports": True}

    def test_destination_status_empty_before_any_check(self):
        m = _mgr(proxies=["http://p1:1"])
        assert m.destination_status() == {}

    def test_destination_status_does_not_reprobe(self):
        m = _mgr(proxies=["http://p1:1"])
        with patch.object(m, "_probe_destination", return_value=True) as probe:
            m.check_destinations({"boylesports": "https://boylesports.com"})
            probe.reset_mock()
            m.destination_status()
            m.destination_status()
        probe.assert_not_called()

    def test_never_mutates_blacklist_state_on_failure(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=1)
        with patch.object(m, "_probe_destination", return_value=False):
            m.check_destinations({"boylesports": "https://boylesports.com"})
        assert m._entries[0].blacklisted_until == 0.0
        assert m._entries[0].total_failures == 0
        assert m._entries[0].consecutive_failures == 0

    def test_uses_pool_proxy_for_every_destination(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"])
        seen_proxies = []
        with patch.object(m, "_probe_destination", side_effect=lambda url, proxy, t: (seen_proxies.append(proxy), True)[1]):
            m.check_destinations({"a": "https://a.com", "b": "https://b.com"})
        # One next() draw per check_destinations() call, reused for every
        # destination in that batch (not re-drawn per destination).
        assert seen_proxies == ["http://p1:1", "http://p1:1"]

    def test_successive_checks_draw_a_fresh_proxy(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"])
        draws = []
        with patch.object(m, "_probe_destination", side_effect=lambda url, proxy, t: (draws.append(proxy), True)[1]):
            m.check_destinations({"a": "https://a.com"})
            m.check_destinations({"a": "https://a.com"})
        assert draws == ["http://p1:1", "http://p2:2"]

    def test_disabled_pool_probes_direct(self):
        m = _mgr(proxies=["http://p1:1"], enabled=False)
        seen_proxies = []
        with patch.object(m, "_probe_destination", side_effect=lambda url, proxy, t: (seen_proxies.append(proxy), True)[1]):
            m.check_destinations({"a": "https://a.com"})
        assert seen_proxies == [None]

    def test_empty_destinations_returns_empty_dict(self):
        m = _mgr(proxies=["http://p1:1"])
        assert m.check_destinations({}) == {}


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_pool(self):
        m = _mgr(proxies=[])
        assert m.stats() == []

    def test_active_proxy_stats(self):
        m = _mgr(proxies=["http://p1:1"])
        m.report_success("http://p1:1")
        m.report_success("http://p1:1")
        m.report_failure("http://p1:1")
        stats = m.stats()
        assert len(stats) == 1
        s = stats[0]
        assert s["url"] == "http://p1:1"
        assert s["status"] == "active"
        assert s["total_successes"] == 2
        assert s["total_failures"] == 1
        assert abs(s["success_rate"] - 2/3) < 0.001

    def test_blacklisted_proxy_shows_status(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=1)
        m.report_failure("http://p1:1")
        stats = m.stats()
        assert stats[0]["status"] == "blacklisted"
        assert stats[0]["blacklisted_until"] is not None

    def test_success_rate_none_when_no_requests(self):
        m = _mgr(proxies=["http://p1:1"])
        assert m.stats()[0]["success_rate"] is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_report_calls(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=1000)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    m.report_success("http://p1:1")
                    m.report_failure("http://p1:1")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        total = m._entries[0].total_successes + m._entries[0].total_failures
        assert total == 1000  # 10 threads × 100 calls each


# ---------------------------------------------------------------------------
# Rotating gateway mode
# ---------------------------------------------------------------------------

def _rot_mgr(proxies=None, **kw):
    cfg = {
        "enabled": True,
        "rotating_gateway": True,
        "proxies": proxies if proxies is not None else ["http://gw:823"],
        "max_failures": 3,
        "blacklist_cooldown": 300,
        "health_check_interval": 0,
        "health_check_url": "https://icanhazip.com",
    }
    cfg.update(kw)
    return ProxyManager(cfg=cfg)


class TestRotatingGateway:
    def test_failures_never_blacklist(self):
        m = _rot_mgr(max_failures=2)
        for _ in range(10):
            m.report_failure("http://gw:823")
        assert m._entries[0].total_failures == 10
        assert m._entries[0].blacklisted_until == 0.0

    def test_next_never_falls_back_to_direct_on_failures(self):
        m = _rot_mgr(max_failures=1)
        for _ in range(5):
            m.report_failure("http://gw:823")
        # Non-rotating this would be blacklisted -> None; rotating keeps the gateway.
        assert m.next() == "http://gw:823"
        assert m.next() == "http://gw:823"

    def test_next_ignores_manual_blacklist(self):
        m = _rot_mgr()
        m._entries[0].blacklisted_until = time.monotonic() + 9999
        assert m.next() == "http://gw:823"

    def test_round_robin_across_multiple_gateways(self):
        m = _rot_mgr(proxies=["http://gw1:823", "http://gw2:823"])
        assert m.next() == "http://gw1:823"
        assert m.next() == "http://gw2:823"
        assert m.next() == "http://gw1:823"

    def test_health_check_fail_does_not_blacklist(self):
        m = _rot_mgr()
        with patch.object(m, "_probe", return_value=False):
            results = m.health_check()
        assert results["http://gw:823"] is False
        assert m._entries[0].blacklisted_until == 0.0

    def test_disabled_still_returns_none(self):
        m = _rot_mgr(enabled=False)
        assert m.next() is None


# ---------------------------------------------------------------------------
# Country geo-targeting
# ---------------------------------------------------------------------------

class TestCountryTargeting:
    def test_injects_country_token_into_username(self):
        m = ProxyManager(cfg={
            "enabled": True, "country": "ie", "health_check_interval": 0,
            "proxies": ["http://user:pass@gw.dataimpulse.com:823"],
        })
        assert m._entries[0].url == "http://user__cr.ie:pass@gw.dataimpulse.com:823"

    def test_country_is_idempotent(self):
        once = ProxyManager._apply_country("http://user:pass@gw:823", "ie")
        twice = ProxyManager._apply_country(once, "ie")
        assert once == twice == "http://user__cr.ie:pass@gw:823"

    def test_country_replaces_existing_token(self):
        out = ProxyManager._apply_country("http://user__cr.gb:pass@gw:823", "ie")
        assert out == "http://user__cr.ie:pass@gw:823"

    def test_country_normalised_to_lowercase(self):
        m = ProxyManager(cfg={
            "enabled": True, "country": "IE", "health_check_interval": 0,
            "proxies": ["http://user:pass@gw:823"],
        })
        assert m._entries[0].url == "http://user__cr.ie:pass@gw:823"

    def test_no_userinfo_is_noop(self):
        assert ProxyManager._apply_country("http://gw:823", "ie") == "http://gw:823"

    def test_no_country_leaves_url_unchanged(self):
        m = ProxyManager(cfg={
            "enabled": True, "health_check_interval": 0,
            "proxies": ["http://user:pass@gw:823"],
        })
        assert m._entries[0].url == "http://user:pass@gw:823"


# ---------------------------------------------------------------------------
# next(required=True) — typed failure, never a silent direct fallback
# ---------------------------------------------------------------------------

class TestRequiredProxy:
    def test_disabled_returns_none_even_when_required(self):
        m = _mgr(enabled=False)
        assert m.next(required=True) is None

    def test_no_proxies_configured_raises_when_required(self):
        m = _mgr(proxies=[], enabled=True)
        with pytest.raises(ProxyUnavailableError):
            m.next(required=True)

    def test_no_proxies_configured_returns_none_when_not_required(self):
        m = _mgr(proxies=[], enabled=True)
        assert m.next(required=False) is None

    def test_all_blacklisted_raises_when_required(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"], max_failures=1)
        m.report_failure("http://p1:1")
        m.report_failure("http://p2:2")
        with pytest.raises(ProxyUnavailableError):
            m.next(required=True)

    def test_all_blacklisted_returns_none_when_not_required(self):
        m = _mgr(proxies=["http://p1:1"], max_failures=1)
        m.report_failure("http://p1:1")
        assert m.next(required=False) is None

    def test_healthy_pool_never_raises_when_required(self):
        m = _mgr(proxies=["http://p1:1", "http://p2:2"])
        assert m.next(required=True) == "http://p1:1"
        assert m.next(required=True) == "http://p2:2"

    def test_rotating_gateway_never_raises_even_when_required(self):
        m = _rot_mgr(max_failures=1)
        for _ in range(10):
            m.report_failure("http://gw:823")
        # Rotating gateways never exhaust — always draws a fresh exit IP.
        assert m.next(required=True) == "http://gw:823"

    def test_rotating_gateway_disabled_returns_none_when_required(self):
        m = _rot_mgr(enabled=False)
        assert m.next(required=True) is None


# ---------------------------------------------------------------------------
# redact_proxy_url()
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_masks_userinfo(self):
        out = redact_proxy_url("http://someuser123:somepass456@gw.dataimpulse.com:823")
        assert "someuser123" not in out
        assert "somepass456" not in out
        assert out == "http://***:***@gw.dataimpulse.com:823"

    def test_no_userinfo_returned_unchanged(self):
        assert redact_proxy_url("http://gw.dataimpulse.com:823") == "http://gw.dataimpulse.com:823"

    def test_empty_or_none_returns_empty_string(self):
        assert redact_proxy_url("") == ""
        assert redact_proxy_url(None) == ""

    def test_stats_never_leaks_raw_credentials(self):
        m = ProxyManager(cfg={
            "enabled": True, "health_check_interval": 0,
            "proxies": ["http://secretuser:secretpass@gw.dataimpulse.com:823"],
        })
        stats = m.stats()
        assert "secretuser" not in stats[0]["url"]
        assert "secretpass" not in stats[0]["url"]
        assert stats[0]["url"] == "http://***:***@gw.dataimpulse.com:823"


# ---------------------------------------------------------------------------
# redact_secrets() — defense-in-depth for credentials embedded in free text
# ---------------------------------------------------------------------------

class TestRedactSecrets:
    def test_masks_credential_embedded_mid_string(self):
        out = redact_secrets(
            "connection failed: http://secretuser:secretpass@gw.dataimpulse.com:823 timed out"
        )
        assert "secretuser" not in out
        assert "secretpass" not in out
        assert out == "connection failed: http://***:***@gw.dataimpulse.com:823 timed out"

    def test_no_credential_returned_unchanged(self):
        msg = "connection failed: https://www.boylesports.com/sports/horse-racing timed out"
        assert redact_secrets(msg) == msg

    def test_empty_or_none_returns_empty_string(self):
        assert redact_secrets("") == ""
        assert redact_secrets(None) == ""

    def test_masks_multiple_credentials_in_one_message(self):
        out = redact_secrets(
            "tried http://u1:p1@proxy1.example.com:1 then http://u2:p2@proxy2.example.com:2"
        )
        assert "u1" not in out and "p1" not in out
        assert "u2" not in out and "p2" not in out
        assert out == (
            "tried http://***:***@proxy1.example.com:1 then "
            "http://***:***@proxy2.example.com:2"
        )


# ---------------------------------------------------------------------------
# _on_config_change() — hot reload only when proxy_pool actually changed
# ---------------------------------------------------------------------------

class TestConfigHotReload:
    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    def test_rebuilds_when_proxy_pool_changes(self):
        get_proxy_manager(cfg={
            "enabled": True, "proxies": ["http://p1:1"], "health_check_interval": 0,
        })
        import utils.proxy_manager as pm
        before = pm._manager
        _on_config_change({"proxy_pool": {
            "enabled": True, "proxies": ["http://p2:2"], "health_check_interval": 0,
        }})
        assert pm._manager is not before
        assert pm._manager._entries[0].url == "http://p2:2"

    def test_noop_when_proxy_pool_unchanged(self):
        cfg = {"enabled": True, "proxies": ["http://p1:1"], "health_check_interval": 0}
        get_proxy_manager(cfg=cfg)
        import utils.proxy_manager as pm
        before = pm._manager
        before.report_failure("http://p1:1")  # live state that must survive a no-op reload
        _on_config_change({"proxy_pool": cfg, "unrelated": {"foo": "bar"}})
        assert pm._manager is before
        assert pm._manager._entries[0].total_failures == 1

    def test_noop_when_no_manager_yet_and_config_empty(self):
        import utils.proxy_manager as pm
        assert pm._manager is None
        _on_config_change({})
        # An empty proxy_pool config still creates a (disabled) manager on first change.
        assert pm._manager is not None
        assert pm._manager.next() is None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    def test_same_instance_returned(self):
        a = get_proxy_manager()
        b = get_proxy_manager()
        assert a is b

    def test_reset_gives_new_instance(self):
        a = get_proxy_manager()
        _reset_singleton()
        b = get_proxy_manager()
        assert a is not b

    def test_singleton_accepts_cfg_override(self):
        cfg = {
            "enabled": True,
            "proxies": ["http://custom:9"],
            "max_failures": 1,
            "blacklist_cooldown": 60,
            "health_check_interval": 0,
        }
        m = get_proxy_manager(cfg=cfg)
        assert m._entries[0].url == "http://custom:9"
