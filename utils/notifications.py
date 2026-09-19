"""
Notification system for race alerts and bet outcomes.

Architecture:
  - NotificationQueue: non-blocking queue.Queue + daemon worker thread
  - TelegramChannel: httpx POST to Telegram Bot API
  - Notifier: façade; reads config, owns the queue, exposes typed .notify_*() methods
  - get_notifier(): module-level singleton factory (safe to import anywhere)

Config block in config.yaml:
  notifications:
    enabled: true
    channels:
      telegram:
        enabled: true
        bot_token: ""      # BotFather token
        chat_id: ""        # your chat / group / channel ID
    events:
      race_soon_minutes: 10          # fire alert when race_time - now() <= this
      odds_drop_threshold_pct: 10    # fire alert when odds fall by >= this percent
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from utils.config_loader import get_config
from utils.logger import get_logger
from utils.timezone import TZ, to_local

_log = get_logger("notifications")

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_WORKER_SLEEP = 1.2  # seconds between sends — respects Telegram 30 msg/min limit
_HTTP_TIMEOUT = 8.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    try:
        return get_config()
    except Exception:
        return {}


def _notif_cfg(cfg: dict) -> dict:
    return cfg.get("notifications", {})


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    text: str
    parse_mode: str = "HTML"
    # future: per-message channel overrides could go here


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

class _TelegramChannel:
    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat_id = str(chat_id)
        self._url = _TELEGRAM_API.format(token=token)

    def send(self, msg: _Message) -> None:
        try:
            resp = httpx.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": msg.text,
                    "parse_mode": msg.parse_mode,
                },
                timeout=_HTTP_TIMEOUT,
            )
            if not resp.is_success:
                _log.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
        except Exception as exc:  # network errors must not crash the worker
            _log.error("Telegram send failed: %s", exc)


# ---------------------------------------------------------------------------
# Queue + worker
# ---------------------------------------------------------------------------

class _NotificationQueue:
    def __init__(self, channels: list) -> None:
        self._channels = channels
        self._q: queue.Queue[Optional[_Message]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="notif-worker")
        self._thread.start()

    def enqueue(self, msg: _Message) -> None:
        """Non-blocking — caller returns immediately."""
        self._q.put_nowait(msg)

    def _worker(self) -> None:
        while True:
            try:
                msg = self._q.get(timeout=5)
            except queue.Empty:
                continue
            if msg is None:  # sentinel for graceful shutdown
                self._q.task_done()
                break
            try:
                for ch in self._channels:
                    try:
                        ch.send(msg)
                    except Exception as exc:  # one bad channel must not skip others
                        _log.error("Channel send failed: %s", exc)
            finally:
                self._q.task_done()  # always mark done — prevents join() deadlock
            time.sleep(_WORKER_SLEEP)

    def shutdown(self) -> None:
        self._q.put_nowait(None)
        self._thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Notifier façade
# ---------------------------------------------------------------------------

class Notifier:
    """Main notification interface.  Instantiate via get_notifier() only."""

    def __init__(self) -> None:
        cfg = _load_cfg()
        ncfg = _notif_cfg(cfg)

        # Safety net: never send real notifications under pytest, or when
        # RP_DISABLE_NOTIFICATIONS is set. Keeps production behaviour unchanged.
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("RP_DISABLE_NOTIFICATIONS"):
            self._enabled = False
        else:
            self._enabled = ncfg.get("enabled", False)
        ev = ncfg.get("events", {})
        self.race_soon_minutes: int = int(ev.get("race_soon_minutes", 10))
        self.odds_drop_threshold_pct: float = float(ev.get("odds_drop_threshold_pct", 10))

        channels = []
        if self._enabled:
            tg = ncfg.get("channels", {}).get("telegram", {})
            if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
                channels.append(_TelegramChannel(tg["bot_token"], tg["chat_id"]))
                _log.info("Telegram notification channel active (chat_id=%s)", tg["chat_id"])
            else:
                _log.debug("Telegram channel disabled or not configured")

        self._queue = _NotificationQueue(channels)
        self._active = bool(channels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, text: str) -> None:
        if not self._active:
            _log.debug("Notifications inactive — dropped: %s", text[:80])
            return
        self._queue.enqueue(_Message(text=text))

    @staticmethod
    def _fmt_time(race_time) -> str:
        """Return a friendly Dublin-local date+time, e.g. 'Thu 19 Jun, 14:30'.

        Accepts a tz-aware/naive ``datetime`` or an ISO-8601 string (the form the
        predictor/feeds pass, e.g. ``2026-06-19T14:00:00+01:00``). Naive values
        are assumed to be Dublin local. Falls back to the raw value only if it is
        wholly unparseable."""
        from datetime import datetime

        dt = race_time
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return str(race_time)
        if not isinstance(dt, datetime):
            return str(race_time)
        try:
            if dt.tzinfo is None:
                dt = TZ.localize(dt)
            return to_local(dt).strftime("%a %d %b, %H:%M")
        except Exception:
            return str(race_time)

    # ------------------------------------------------------------------
    # Typed event methods
    # ------------------------------------------------------------------

    def notify_race_soon(
        self,
        venue: str,
        race_time,
        horse_name: str,
        composite_score: float,
        odds: float,
        minutes_to_go: int,
    ) -> None:
        """Fire when a predicted horse's race is within race_soon_minutes."""
        local_t = self._fmt_time(race_time)
        text = (
            f"🏇 <b>Race Starting Soon</b> — {minutes_to_go} min\n"
            f"<b>{horse_name}</b> @ {venue} ({local_t})\n"
            f"Odds: {odds:.2f} | Score: {composite_score:.3f}"
        )
        self._send(text)

    def notify_new_top_pick(
        self,
        venue: str,
        race_time,
        horse_name: str,
        composite_score: float,
        odds: float,
    ) -> None:
        """Fire when predictor.refresh() produces a new composite_score leader."""
        local_t = self._fmt_time(race_time)
        text = (
            f"⭐ <b>New Top Pick</b>\n"
            f"<b>{horse_name}</b> @ {venue} ({local_t})\n"
            f"Odds: {odds:.2f} | Score: {composite_score:.3f}"
        )
        self._send(text)

    def notify_settle_prompt(
        self,
        bet_id: int,
        horse_name: str,
        venue: str,
        race_time,
        stake: float,
        odds: float,
    ) -> None:
        """Fire when a pending bet's race_time has passed — prompt user to settle."""
        local_t = self._fmt_time(race_time)
        text = (
            f"📋 <b>Settle Your Bet</b> (id={bet_id})\n"
            f"<b>{horse_name}</b> @ {venue} ({local_t})\n"
            f"Stake: €{stake:.2f} @ {odds:.2f}\n"
            f"Open the Performance Dashboard → Settlement panel."
        )
        self._send(text)

    def notify_stop_loss(self, bankroll: float, floor: float) -> None:
        """Fire immediately when bankroll drops below the stop-loss floor."""
        text = (
            f"🚨 <b>Stop-Loss Triggered</b>\n"
            f"Bankroll: €{bankroll:.2f} (floor: €{floor:.2f})\n"
            f"New bets are <b>blocked</b> until bankroll recovers."
        )
        self._send(text)

    def notify_odds_drop(
        self,
        horse_name: str,
        venue: str,
        old_odds: float,
        new_odds: float,
        drop_pct: float,
    ) -> None:
        """Fire when odds contract by >= odds_drop_threshold_pct."""
        text = (
            f"📉 <b>Odds Drop Detected</b>\n"
            f"<b>{horse_name}</b> @ {venue}\n"
            f"{old_odds:.2f} → {new_odds:.2f} ({drop_pct:.1f}% shorter)"
        )
        self._send(text)

    def notify_bet_outcome(
        self,
        horse_name: str,
        result: str,          # 'win' | 'place' | 'lose'
        pl: float,
        bankroll: float,
    ) -> None:
        """Fire when a bet is settled (win/place/lose)."""
        icons = {"win": "✅", "place": "🟡", "lose": "❌"}
        icon = icons.get(result, "ℹ️")
        sign = "+" if pl >= 0 else "-"
        text = (
            f"{icon} <b>Bet Settled — {result.upper()}</b>\n"
            f"<b>{horse_name}</b>\n"
            f"P&L: {sign}€{abs(pl):.2f} | Bankroll: €{bankroll:.2f}"
        )
        self._send(text)

    def shutdown(self) -> None:
        """Drain the queue and stop the worker thread.  Call on process exit."""
        self._queue.shutdown()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_notifier: Optional[Notifier] = None
_notifier_lock = threading.Lock()


def get_notifier() -> Notifier:
    """Return the process-wide Notifier singleton (lazy-initialised, thread-safe)."""
    global _notifier
    if _notifier is None:
        with _notifier_lock:
            if _notifier is None:
                _notifier = Notifier()
    return _notifier
