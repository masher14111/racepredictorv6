"""Central proxy pool with health checks, blacklisting, and per-proxy stats.

Public API
----------
get_proxy_manager() -> ProxyManager   # module-level singleton
ProxyManager.next(required=False) -> Optional[str]  # pick next healthy proxy
ProxyManager.report_success(proxy)    # record a successful request
ProxyManager.report_failure(proxy)    # record a failure; may blacklist
ProxyManager.health_check()           # probe all proxies now; returns {url: bool}
ProxyManager.check_destinations(urls) # read-only per-destination reachability probe
ProxyManager.stats() -> list[dict]    # per-proxy diagnostic data (credentials redacted)
redact_proxy_url(url) -> str          # mask userinfo in a proxy URL
redact_secrets(text) -> str           # mask any proxy-URL credential embedded in free text
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from utils.config_loader import get_config, register_callback
from utils.logger import get_logger

logger = get_logger(__name__)


class ProxyUnavailableError(Exception):
    """Raised by next(required=True) when the pool is enabled but no proxy can
    be returned right now (misconfigured or, in non-rotating mode, exhausted).

    Callers that require a proxy (Cloudflare-protected bookmakers) must treat
    this as a hard failure — never silently retry over a direct connection.
    """


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _proxy_cfg() -> dict:
    try:
        return get_config().get("proxy_pool", {})
    except Exception:
        return {}


def redact_proxy_url(url: Optional[str]) -> str:
    """Mask userinfo in a proxy URL so credentials never reach logs/UI."""
    if not url:
        return ""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    _, host = parts.netloc.rsplit("@", 1)
    new_netloc = f"***:***@{host}"
    return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))


_URL_CREDENTIAL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")


def redact_secrets(text: Optional[str]) -> str:
    """Mask any ``scheme://user:pass@host`` credential embedded in free text.

    Defense-in-depth for callers that surface a raw exception message (a
    connection-layer error from an HTTP client can embed the proxy URL,
    including its credentials, in its own string representation) to a log,
    stdout, or the UI without going through ``redact_proxy_url`` first.
    """
    if not text:
        return ""
    return _URL_CREDENTIAL_RE.sub(lambda m: f"{m.group(1)}***:***@", text)


# ---------------------------------------------------------------------------
# ProxyEntry — per-proxy mutable state
# ---------------------------------------------------------------------------

@dataclass
class ProxyEntry:
    url: str
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0
    blacklisted_until: float = 0.0   # monotonic; 0 = not blacklisted


# ---------------------------------------------------------------------------
# ProxyManager
# ---------------------------------------------------------------------------

class ProxyManager:
    """Thread-safe proxy pool with health checks and automatic blacklisting."""

    def __init__(self, cfg: dict = None):
        if cfg is None:
            cfg = _proxy_cfg()
        self._raw_cfg: dict = dict(cfg)

        self._enabled: bool = cfg.get("enabled", False)
        # Rotating gateway mode: the configured proxy is a single rotating
        # residential gateway (e.g. DataImpulse) that hands out a fresh exit IP
        # per request. A 403/blip then means *that random exit IP* was flagged —
        # not the gateway — so blacklisting the endpoint (and worse, falling back
        # to a direct, proxy-less connection that Cloudflare blocks outright) is
        # wrong. In this mode failures are counted but never blacklist, and
        # next() always returns the gateway so a retry simply draws a new IP.
        self._rotating: bool = bool(cfg.get("rotating_gateway", False))
        # Optional exit-IP country (ISO-3166 alpha-2, e.g. "ie"). DataImpulse
        # geo-targets via a username suffix (LOGIN__cr.ie), so we can pin the
        # country from committed config while the credential stays in the
        # git-ignored config.local.yaml.
        self._country: Optional[str] = (cfg.get("country") or "").strip().lower() or None

        raw: list[str] = [u for u in cfg.get("proxies", []) if u]
        if self._country:
            raw = [self._apply_country(u, self._country) for u in raw]
        self._entries: list[ProxyEntry] = [ProxyEntry(url=u) for u in raw]

        self._max_failures: int = int(cfg.get("max_failures", 3))
        self._blacklist_cooldown: float = float(cfg.get("blacklist_cooldown", 300))
        self._health_check_url: str = cfg.get("health_check_url", "https://icanhazip.com")
        health_interval: float = float(cfg.get("health_check_interval", 300))
        # Per-bookmaker probe targets (req 3): a 200 from icanhazip only proves
        # the gateway itself is up, not that a given bookmaker endpoint is
        # reachable through it. Populated from proxy_pool.destination_urls.
        self._destination_urls: dict[str, str] = dict(cfg.get("destination_urls") or {})
        self._last_destination_check: dict[str, bool] = {}

        self._lock = threading.Lock()
        self._index: int = 0

        if self._enabled and self._entries and health_interval > 0:
            self._start_background_checker(health_interval)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def next(self, required: bool = False) -> Optional[str]:
        """Return the next healthy proxy URL, or None to use a direct connection.

        If the pool is disabled, always returns None regardless of `required`
        — that's an explicit operator choice, not a failure. If the pool is
        enabled but no proxy can be returned right now (misconfigured with no
        entries, or in non-rotating mode all entries are blacklisted) and
        `required=True`, raises ProxyUnavailableError instead of returning
        None — callers for Cloudflare-protected sources must never silently
        fall back to a direct connection that will just get blocked.
        """
        if not self._enabled:
            return None
        if not self._entries:
            if required:
                raise ProxyUnavailableError("proxy_pool enabled but no proxies configured")
            return None

        now = time.monotonic()
        with self._lock:
            if self._rotating:
                # Rotating gateway: every request already draws a fresh exit IP,
                # so never demote to direct on transient failures — round-robin
                # the gateway(s) regardless of blacklist state.
                entry = self._entries[self._index % len(self._entries)]
                self._index += 1
                logger.debug("proxy_manager: selected %s (rotating)", redact_proxy_url(entry.url))
                return entry.url

            for _ in range(len(self._entries)):
                entry = self._entries[self._index % len(self._entries)]
                self._index += 1
                if entry.blacklisted_until <= now:
                    logger.debug("proxy_manager: selected %s", redact_proxy_url(entry.url))
                    return entry.url

            if required:
                raise ProxyUnavailableError(
                    f"all {len(self._entries)} proxies blacklisted"
                )
            # All proxies blacklisted — log and fall back to direct
            logger.warning(
                "proxy_manager: all %d proxies blacklisted, falling back to direct",
                len(self._entries),
            )
            return None

    def report_success(self, proxy: Optional[str]) -> None:
        if not proxy:
            return
        with self._lock:
            entry = self._entry(proxy)
            if entry is None:
                return
            entry.total_successes += 1
            entry.consecutive_failures = 0
            entry.blacklisted_until = 0.0

    def report_failure(self, proxy: Optional[str]) -> None:
        if not proxy:
            return
        with self._lock:
            entry = self._entry(proxy)
            if entry is None:
                return
            entry.total_failures += 1
            entry.consecutive_failures += 1
            if self._rotating:
                # A flagged exit IP isn't the gateway's fault — count it for
                # diagnostics but never blacklist (the next request draws a new IP).
                return
            if entry.consecutive_failures >= self._max_failures:
                until = time.monotonic() + self._blacklist_cooldown
                entry.blacklisted_until = until
                logger.warning(
                    "proxy_manager: blacklisted %s for %.0fs after %d failures",
                    entry.url, self._blacklist_cooldown, entry.consecutive_failures,
                )

    def health_check(self, timeout: float = 10.0) -> dict[str, bool]:
        """Probe every proxy; re-enable blacklisted ones that pass. Returns {url: ok}."""
        results: dict[str, bool] = {}
        for entry in self._entries:
            ok = self._probe(entry.url, timeout)
            results[entry.url] = ok
            with self._lock:
                if ok:
                    entry.consecutive_failures = 0
                    entry.blacklisted_until = 0.0
                    logger.info("proxy_manager: health_check OK — %s",
                                redact_proxy_url(entry.url))
                elif self._rotating:
                    # Don't blacklist a rotating gateway on a single bad probe —
                    # that would drop every scraper to a direct connection.
                    logger.warning(
                        "proxy_manager: health_check FAIL (rotating, not blacklisting) — %s",
                        redact_proxy_url(entry.url),
                    )
                else:
                    entry.consecutive_failures = max(
                        entry.consecutive_failures, self._max_failures
                    )
                    entry.blacklisted_until = time.monotonic() + self._blacklist_cooldown
                    logger.warning("proxy_manager: health_check FAIL — %s",
                                   redact_proxy_url(entry.url))
        return results

    def check_destinations(self, destinations: dict[str, str] = None, timeout: float = 10.0) -> dict[str, bool]:
        """Read-only reachability probe per bookmaker destination URL.

        Fetches each destination through the pool (the same proxy `next()`
        would hand out; direct if the pool is disabled) and checks for a
        non-error response. Unlike health_check(), this never mutates
        blacklist state — a bookmaker being briefly down is not the proxy's
        fault. Results are cached on the instance and returned to the caller
        (typically utils.source_health) to record as `proxy_reachable`.
        """
        targets = destinations if destinations is not None else self._destination_urls
        proxy_url = self.next()
        results: dict[str, bool] = {}
        for name, url in targets.items():
            results[name] = self._probe_destination(url, proxy_url, timeout)
        with self._lock:
            self._last_destination_check = dict(results)
        return results

    def destination_status(self) -> dict[str, bool]:
        """Most recent check_destinations() result, without re-probing."""
        with self._lock:
            return dict(self._last_destination_check)

    def stats(self) -> list[dict]:
        now = time.monotonic()
        out = []
        with self._lock:
            for e in self._entries:
                total = e.total_successes + e.total_failures
                out.append({
                    "url": redact_proxy_url(e.url),
                    "status": "blacklisted" if e.blacklisted_until > now else "active",
                    "total_successes": e.total_successes,
                    "total_failures": e.total_failures,
                    "success_rate": e.total_successes / total if total else None,
                    "consecutive_failures": e.consecutive_failures,
                    "blacklisted_until": e.blacklisted_until if e.blacklisted_until > now else None,
                })
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def sticky(self, session_id: str, ttl_minutes: Optional[int] = None,
               required: bool = False) -> Optional[str]:
        """A proxy URL pinned to ONE exit IP for the life of ``session_id``.

        The rotating gateway hands out a fresh IP per request, which is the right
        default against per-IP rate limits but is counter-productive against
        Cloudflare: a visitor whose address changes on every request and who
        never carries a clearance cookie looks exactly like a bot farm, so the
        challenge never stops being served. A sticky session holds one IP long
        enough for a cookie jar to be worth having.

        DataImpulse pins the exit via username tokens — ``__cr.ie;sessid.<id>``,
        plus an optional ``;sessttl.<minutes>`` rotation interval. The same
        ``session_id`` returns the same IP (~30 min by default).
        """
        base = self.next(required=required)
        if not base:
            return None
        return self._apply_session(base, session_id, ttl_minutes)

    @staticmethod
    def _apply_session(url: str, session_id: str, ttl_minutes: Optional[int] = None) -> str:
        """Add ``;sessid.<id>`` (and optional ``;sessttl.<n>``) to the username.

        DataImpulse's first username token is introduced by ``__`` and any
        further ones by ``;``. _apply_country has usually already added
        ``__cr.<cc>``, so this appends with ``;``; when it hasn't, this token
        becomes the first and takes the ``__``. Idempotent: existing sessid /
        sessttl tokens are stripped before the new ones go on.
        """
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        userinfo, host = parts.netloc.rsplit("@", 1)
        user, sep, pwd = userinfo.partition(":")

        user = re.sub(r"[;_]{0,2}sessid\.[^;:@]*", "", user, flags=re.IGNORECASE)
        user = re.sub(r"[;_]{0,2}sessttl\.[^;:@]*", "", user, flags=re.IGNORECASE)

        safe_id = re.sub(r"[^A-Za-z0-9]", "", str(session_id)) or "rp"
        token = f"sessid.{safe_id}"
        if ttl_minutes:
            token += f";sessttl.{int(ttl_minutes)}"
        # "__" introduces the first token; ";" chains the rest.
        user = f"{user};{token}" if "__" in user else f"{user}__{token}"

        new_netloc = f"{user}{sep}{pwd}@{host}"
        return urlunsplit(
            (parts.scheme, new_netloc, parts.path, parts.query, parts.fragment)
        )

    @staticmethod
    def _apply_country(url: str, country: str) -> str:
        """Geo-target a DataImpulse proxy URL to a country via its username suffix.

        DataImpulse selects the exit-IP country with a ``LOGIN__cr.<cc>`` token in
        the username. This injects (or normalises) that token so the same
        credential in config.local.yaml can be pointed at a country from committed
        config. No-op if the URL carries no userinfo (can't geo-target then).
        """
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        userinfo, host = parts.netloc.rsplit("@", 1)
        user, sep, pwd = userinfo.partition(":")
        # Strip any existing __cr.<cc> token so re-pointing the country is idempotent.
        user = re.sub(r"__cr\.[a-z]{2}", "", user, flags=re.IGNORECASE)
        user = f"{user}__cr.{country}"
        new_netloc = f"{user}{sep}{pwd}@{host}"
        return urlunsplit(
            (parts.scheme, new_netloc, parts.path, parts.query, parts.fragment)
        )

    def _entry(self, proxy_url: str) -> Optional[ProxyEntry]:
        for e in self._entries:
            if e.url == proxy_url:
                return e
        return None

    def _probe(self, proxy_url: str, timeout: float) -> bool:
        try:
            mounts = {"all://": httpx.HTTPTransport(proxy=proxy_url)}
            with httpx.Client(mounts=mounts, timeout=timeout) as client:
                resp = client.get(self._health_check_url)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("proxy_manager: probe failed for %s: %s", redact_proxy_url(proxy_url), exc)
            return False

    def _probe_destination(self, target_url: str, proxy_url: Optional[str], timeout: float) -> bool:
        """GET a bookmaker destination URL (through `proxy_url`, or direct if None)."""
        try:
            mounts = {"all://": httpx.HTTPTransport(proxy=proxy_url)} if proxy_url else None
            with httpx.Client(mounts=mounts, timeout=timeout) as client:
                resp = client.get(target_url)
            return resp.status_code < 400
        except Exception as exc:
            logger.debug(
                "proxy_manager: destination probe failed for %s (proxy=%s): %s",
                target_url, redact_proxy_url(proxy_url), exc,
            )
            return False

    def _start_background_checker(self, interval: float) -> None:
        def _loop():
            while True:
                time.sleep(interval)
                try:
                    self.health_check()
                except Exception as exc:
                    logger.warning("proxy_manager: background health check error: %s", exc)
                if self._destination_urls:
                    try:
                        self.check_destinations()
                    except Exception as exc:
                        logger.warning("proxy_manager: background destination check error: %s", exc)

        t = threading.Thread(target=_loop, daemon=True, name="proxy-health-checker")
        t.start()
        logger.debug("proxy_manager: background health-check thread started (interval=%.0fs)", interval)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[ProxyManager] = None
_manager_lock = threading.Lock()
_callback_registered: bool = False


def _on_config_change(cfg: dict) -> None:
    """Rebuild the singleton if proxy_pool config actually changed.

    utils.config_loader re-checks the file mtime and fires registered
    callbacks on every get_config() call, so this is invoked continuously —
    it must be cheap and a no-op when nothing relevant changed, so that
    unrelated config edits don't discard live blacklist/success stats.
    """
    global _manager
    new_proxy_cfg = (cfg or {}).get("proxy_pool", {})
    with _manager_lock:
        if _manager is not None and _manager._raw_cfg == new_proxy_cfg:
            return
        logger.info("proxy_manager: proxy_pool config changed, rebuilding manager")
        _manager = ProxyManager(cfg=new_proxy_cfg)


def get_proxy_manager(cfg: dict = None) -> ProxyManager:
    """Return the module-level ProxyManager singleton (created on first call)."""
    global _manager, _callback_registered
    if not _callback_registered:
        # register_callback dedupes by function identity, so this is safe to
        # call repeatedly; the flag just avoids the lookup on the hot path.
        register_callback(_on_config_change)
        _callback_registered = True
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ProxyManager(cfg=cfg)
    return _manager


def _reset_singleton() -> None:
    """Reset the singleton — for tests only."""
    global _manager, _callback_registered
    with _manager_lock:
        _manager = None
        _callback_registered = False
