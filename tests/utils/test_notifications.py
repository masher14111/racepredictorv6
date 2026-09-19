"""Tests for utils/notifications.py."""
import os
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import respx
import httpx

from utils.notifications import (
    Notifier,
    _Message,
    _NotificationQueue,
    _TelegramChannel,
    get_notifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(hour=14, minute=30):
    return datetime(2026, 6, 13, hour, minute, tzinfo=timezone.utc)


def _notifier_with_mock_channel():
    """Return a Notifier whose queue feeds a MagicMock channel."""
    ch = MagicMock()
    n = Notifier.__new__(Notifier)
    n.race_soon_minutes = 10
    n.odds_drop_threshold_pct = 10
    n._queue = _NotificationQueue([ch])
    n._active = True
    return n, ch


def _drain(nq: _NotificationQueue):
    """Block until the queue is fully drained."""
    nq._q.join()


# ---------------------------------------------------------------------------
# _TelegramChannel
# ---------------------------------------------------------------------------

class TestTelegramChannel:
    @respx.mock
    def test_sends_correct_payload(self):
        route = respx.post("https://api.telegram.org/botTOKEN/sendMessage").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        ch = _TelegramChannel("TOKEN", "12345")
        ch.send(_Message(text="Hello"))
        assert route.called
        payload = route.calls[0].request
        import json
        body = json.loads(payload.content)
        assert body["chat_id"] == "12345"
        assert body["text"] == "Hello"
        assert body["parse_mode"] == "HTML"

    @respx.mock
    def test_non_2xx_logs_warning_does_not_raise(self, caplog):
        respx.post("https://api.telegram.org/botTOK/sendMessage").mock(
            return_value=httpx.Response(400, json={"ok": False, "description": "bad"})
        )
        ch = _TelegramChannel("TOK", "999")
        ch.send(_Message(text="x"))  # must not raise

    @respx.mock
    def test_network_error_does_not_raise(self):
        respx.post("https://api.telegram.org/botTOK/sendMessage").mock(
            side_effect=httpx.ConnectError("refused")
        )
        ch = _TelegramChannel("TOK", "999")
        ch.send(_Message(text="x"))  # must not raise


# ---------------------------------------------------------------------------
# _NotificationQueue
# ---------------------------------------------------------------------------

class TestNotificationQueue:
    def test_enqueue_calls_channel(self):
        ch = MagicMock()
        nq = _NotificationQueue([ch])
        nq.enqueue(_Message(text="ping"))
        nq._q.join()
        ch.send.assert_called_once()
        nq.shutdown()

    def test_enqueue_is_non_blocking(self):
        """enqueue() must return immediately even with a slow channel."""
        slow_ch = MagicMock()
        slow_ch.send.side_effect = lambda m: time.sleep(0.5)
        nq = _NotificationQueue([slow_ch])
        t0 = time.monotonic()
        nq.enqueue(_Message(text="slow"))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"enqueue blocked for {elapsed:.3f}s"
        nq._q.join()
        nq.shutdown()

    def test_multiple_channels_all_receive(self):
        ch1, ch2 = MagicMock(), MagicMock()
        nq = _NotificationQueue([ch1, ch2])
        nq.enqueue(_Message(text="both"))
        nq._q.join()
        ch1.send.assert_called_once()
        ch2.send.assert_called_once()
        nq.shutdown()

    def test_channel_exception_does_not_stop_worker(self):
        """A raising channel must not deadlock the queue or kill the worker thread."""
        bad_ch = MagicMock()
        bad_ch.send.side_effect = RuntimeError("boom")
        nq = _NotificationQueue([bad_ch])
        nq.enqueue(_Message(text="first"))
        nq._q.join()  # must not hang even though channel raised
        nq.enqueue(_Message(text="second"))
        nq._q.join()  # worker must still be alive for a second message
        assert bad_ch.send.call_count == 2
        nq.shutdown()

    def test_shutdown_joins_thread(self):
        ch = MagicMock()
        nq = _NotificationQueue([ch])
        nq.shutdown()
        assert not nq._thread.is_alive()


# ---------------------------------------------------------------------------
# Notifier — inactive (no channels configured)
# ---------------------------------------------------------------------------

class TestNotifierInactive:
    def _inactive(self) -> Notifier:
        cfg = {"notifications": {"enabled": False}}
        with patch("utils.notifications._load_cfg", return_value=cfg):
            return Notifier()

    def test_active_false_when_disabled(self):
        n = self._inactive()
        assert not n._active

    def test_send_does_not_enqueue_when_inactive(self):
        n = self._inactive()
        n.notify_race_soon("Curragh", _utc(), "Test Horse", 0.32, 6.0, 8)
        assert n._queue._q.empty()

    def test_all_notify_methods_safe_when_inactive(self):
        n = self._inactive()
        n.notify_race_soon("V", _utc(), "H", 0.3, 5.0, 5)
        n.notify_new_top_pick("V", _utc(), "H", 0.3, 5.0)
        n.notify_settle_prompt(1, "H", "V", _utc(), 10.0, 5.0)
        n.notify_stop_loss(900.0, 950.0)
        n.notify_odds_drop("H", "V", 8.0, 6.0, 25.0)
        n.notify_bet_outcome("H", "win", 40.0, 1040.0)


# ---------------------------------------------------------------------------
# Notifier — active (mock channel)
# ---------------------------------------------------------------------------

class TestNotifierActive:
    def test_notify_race_soon_queues_message(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_race_soon("Curragh", _utc(), "Mighty Oak", 0.35, 7.5, 9)
        _drain(n._queue)
        ch.send.assert_called_once()
        text = ch.send.call_args[0][0].text
        assert "Mighty Oak" in text
        assert "Curragh" in text
        assert "Race Starting Soon" in text
        assert "9 min" in text

    def test_notify_new_top_pick_queues_message(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_new_top_pick("Leopardstown", _utc(15, 0), "Silver Arrow", 0.42, 5.0)
        _drain(n._queue)
        ch.send.assert_called_once()
        text = ch.send.call_args[0][0].text
        assert "Silver Arrow" in text
        assert "New Top Pick" in text

    def test_notify_settle_prompt_includes_bet_id(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_settle_prompt(99, "Fleet Wind", "Naas", _utc(), 20.0, 4.5)
        _drain(n._queue)
        text = ch.send.call_args[0][0].text
        assert "id=99" in text
        assert "Fleet Wind" in text
        assert "Settle Your Bet" in text

    def test_notify_stop_loss_includes_bankroll(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_stop_loss(880.0, 900.0)
        _drain(n._queue)
        text = ch.send.call_args[0][0].text
        assert "Stop-Loss" in text
        assert "880.00" in text
        assert "900.00" in text

    def test_notify_odds_drop_shows_direction(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_odds_drop("Dark Prince", "Galway", 12.0, 8.0, 33.3)
        _drain(n._queue)
        text = ch.send.call_args[0][0].text
        assert "Dark Prince" in text
        assert "12.00" in text
        assert "8.00" in text
        assert "Odds Drop" in text

    @pytest.mark.parametrize("result,icon", [
        ("win", "✅"),
        ("place", "🟡"),
        ("lose", "❌"),
    ])
    def test_notify_bet_outcome_all_results(self, result, icon):
        n, ch = _notifier_with_mock_channel()
        n.notify_bet_outcome("Golden Ray", result, 35.0 if result != "lose" else -10.0, 1035.0)
        _drain(n._queue)
        text = ch.send.call_args[0][0].text
        assert icon in text
        assert "Golden Ray" in text

    def test_positive_pl_shows_plus_sign(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_bet_outcome("H", "win", 50.0, 1050.0)
        _drain(n._queue)
        assert "+€50.00" in ch.send.call_args[0][0].text

    def test_negative_pl_no_plus_sign(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_bet_outcome("H", "lose", -10.0, 990.0)
        _drain(n._queue)
        assert "-€10.00" in ch.send.call_args[0][0].text

    def test_message_parse_mode_is_html(self):
        n, ch = _notifier_with_mock_channel()
        n.notify_stop_loss(800.0, 850.0)
        _drain(n._queue)
        assert ch.send.call_args[0][0].parse_mode == "HTML"


# ---------------------------------------------------------------------------
# Notifier — config loading
# ---------------------------------------------------------------------------

class TestNotifierConfig:
    def _make(self, cfg_override: dict) -> Notifier:
        # These tests exercise real channel construction against a mocked
        # endpoint (respx), so lift the suite-wide self-disable safety net.
        with patch.dict(
            os.environ,
            {"PYTEST_CURRENT_TEST": "", "RP_DISABLE_NOTIFICATIONS": ""},
            clear=False,
        ):
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            os.environ.pop("RP_DISABLE_NOTIFICATIONS", None)
            with patch("utils.notifications._load_cfg", return_value=cfg_override):
                return Notifier()

    def test_defaults_when_no_notifications_block(self):
        n = self._make({})
        assert n.race_soon_minutes == 10
        assert n.odds_drop_threshold_pct == 10.0
        assert not n._active

    def test_custom_thresholds_applied(self):
        cfg = {
            "notifications": {
                "enabled": True,
                "events": {"race_soon_minutes": 15, "odds_drop_threshold_pct": 20},
                "channels": {"telegram": {"enabled": False}},
            }
        }
        n = self._make(cfg)
        assert n.race_soon_minutes == 15
        assert n.odds_drop_threshold_pct == 20.0

    @respx.mock
    def test_telegram_channel_created_when_configured(self):
        respx.post("https://api.telegram.org/botMYTOKEN/sendMessage").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        cfg = {
            "notifications": {
                "enabled": True,
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "bot_token": "MYTOKEN",
                        "chat_id": "42",
                    }
                },
                "events": {},
            }
        }
        n = self._make(cfg)
        assert n._active
        n.notify_stop_loss(800.0, 850.0)
        n._queue._q.join()
        n.shutdown()

    def test_telegram_not_active_without_token(self):
        cfg = {
            "notifications": {
                "enabled": True,
                "channels": {"telegram": {"enabled": True, "bot_token": "", "chat_id": "42"}},
                "events": {},
            }
        }
        n = self._make(cfg)
        assert not n._active


# ---------------------------------------------------------------------------
# get_notifier singleton
# ---------------------------------------------------------------------------

class TestGetNotifierSingleton:
    def test_returns_same_instance(self):
        n1 = get_notifier()
        n2 = get_notifier()
        assert n1 is n2


# ── friendly Dublin date/time formatting ───────────────────────────────────────

def test_fmt_time_iso_string_to_dublin():
    from utils.notifications import Notifier
    # +01:00 (Dublin BST) → shown as-is in Dublin local, with a friendly date.
    assert Notifier._fmt_time("2026-06-19T14:30:00+01:00") == "Fri 19 Jun, 14:30"


def test_fmt_time_utc_shifts_to_dublin():
    from utils.notifications import Notifier
    # 13:30 UTC in June = 14:30 Dublin (BST, +1).
    assert Notifier._fmt_time("2026-06-19T13:30:00Z") == "Fri 19 Jun, 14:30"


def test_fmt_time_unparseable_falls_back():
    from utils.notifications import Notifier
    assert Notifier._fmt_time("not-a-date") == "not-a-date"
