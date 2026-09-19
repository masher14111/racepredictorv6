"""
Central rate limiter and concurrency controller for all scrapers.

Sync:  RateLimiter / get_rate_limiter()            – threading-based (current sync scrapers)
Async: AsyncRateLimiter / get_async_rate_limiter() – asyncio-based (future async scrapers)

Algorithm: token-bucket per domain.
  - Tokens refill at `rps` tokens/second (rps=0 → unlimited).
  - Per-domain semaphore caps simultaneous in-flight requests.
  - Optional global bucket limits total cross-domain RPS.
  - First request on a fresh bucket never waits (burst tolerance preserves test speed).

Config (config.yaml → rate_limits:):
  global:
    max_rps: 0          # 0 = disabled
    max_concurrent: 0   # 0 = unlimited
  domains:
    "gateway-ie.livescorebet.com":
      rps: 1.0
      max_concurrent: 1
"""
import asyncio
import random
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Optional, TypeVar
from urllib.parse import urlparse

from utils.logger import get_logger

_log = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_RPS = 1.0
_DEFAULT_CONCURRENT = 3


# ---------------------------------------------------------------------------
# Token bucket – sync, thread-safe
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe fixed-rate token bucket. acquire() blocks until a token is available."""

    def __init__(self, rps: float):
        self._rps = rps
        if rps <= 0:
            return
        self._capacity = max(rps, 1.0)
        self._tokens = self._capacity          # start full → first call never waits
        self._refill_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self._rps <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._refill_at) * self._rps,
                )
                self._refill_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rps
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Token bucket – asyncio-compatible
# ---------------------------------------------------------------------------

class _AsyncTokenBucket:
    """asyncio-safe token bucket; acquire() is a coroutine."""

    def __init__(self, rps: float):
        self._rps = rps
        if rps <= 0:
            return
        self._capacity = max(rps, 1.0)
        self._tokens = self._capacity
        self._refill_at: float = 0.0
        self._lock: Optional[asyncio.Lock] = None

    def _lock_(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        if self._rps <= 0:
            return
        loop = asyncio.get_event_loop()
        if self._refill_at == 0.0:
            self._refill_at = loop.time()
        while True:
            async with self._lock_():
                now = loop.time()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._refill_at) * self._rps,
                )
                self._refill_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rps
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Per-domain state – sync
# ---------------------------------------------------------------------------

class _DomainLimiter:
    def __init__(self, rps: float, max_concurrent: int):
        self._bucket = _TokenBucket(rps)
        self._sem = threading.Semaphore(max_concurrent) if max_concurrent > 0 else None

    @contextmanager
    def acquire(self):
        if self._sem is not None:
            self._sem.acquire()
        try:
            self._bucket.acquire()
            yield
        finally:
            if self._sem is not None:
                self._sem.release()


# ---------------------------------------------------------------------------
# Sync RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe, per-domain rate limiter for sync scrapers."""

    def __init__(self) -> None:
        self._domains: dict[str, _DomainLimiter] = {}
        self._lock = threading.Lock()
        self._cfg: dict = {}
        self._global_bucket: _TokenBucket = _TokenBucket(0)
        self._global_sem: Optional[threading.Semaphore] = None

    def configure(self, cfg: dict) -> None:
        g = cfg.get("global", {})
        self._global_bucket = _TokenBucket(float(g.get("max_rps", 0) or 0))
        n = int(g.get("max_concurrent", 0) or 0)
        self._global_sem = threading.Semaphore(n) if n > 0 else None
        self._cfg = cfg.get("domains", {})

    def _limiter(self, domain: str) -> _DomainLimiter:
        with self._lock:
            if domain not in self._domains:
                dcfg = self._cfg.get(domain, {})
                rps = float(dcfg.get("rps", _DEFAULT_RPS) or 0)
                mc = int(dcfg.get("max_concurrent", _DEFAULT_CONCURRENT) or 0)
                self._domains[domain] = _DomainLimiter(rps, mc)
        return self._domains[domain]

    @contextmanager
    def acquire(self, domain: str):
        """Block until global + per-domain limits permit a request to `domain`."""
        if self._global_sem is not None:
            self._global_sem.acquire()
        try:
            self._global_bucket.acquire()
            with self._limiter(domain).acquire():
                yield
        finally:
            if self._global_sem is not None:
                self._global_sem.release()

    @contextmanager
    def acquire_for_url(self, url: str):
        """Convenience wrapper: extracts hostname from URL then acquires."""
        domain = urlparse(url).hostname or url
        with self.acquire(domain):
            yield

    def stats(self) -> list[dict]:
        """Return per-domain rps config for diagnostics."""
        with self._lock:
            return [
                {"domain": d, "rps": lim._bucket._rps}
                for d, lim in self._domains.items()
            ]


# ---------------------------------------------------------------------------
# Per-domain state – async
# ---------------------------------------------------------------------------

class _AsyncDomainLimiter:
    def __init__(self, rps: float, max_concurrent: int):
        self._bucket = _AsyncTokenBucket(rps)
        self._max_concurrent = max_concurrent
        self._sem: Optional[asyncio.Semaphore] = None

    def _sem_(self) -> Optional[asyncio.Semaphore]:
        if self._max_concurrent > 0 and self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrent)
        return self._sem

    @asynccontextmanager
    async def acquire(self):
        sem = self._sem_()
        if sem is not None:
            await sem.acquire()
        try:
            await self._bucket.acquire()
            yield
        finally:
            if sem is not None:
                sem.release()


# ---------------------------------------------------------------------------
# Async RateLimiter
# ---------------------------------------------------------------------------

class AsyncRateLimiter:
    """asyncio-compatible per-domain rate limiter for future async scrapers."""

    def __init__(self) -> None:
        self._domains: dict[str, _AsyncDomainLimiter] = {}
        self._cfg: dict = {}
        self._global_bucket: _AsyncTokenBucket = _AsyncTokenBucket(0)
        self._global_max_concurrent: int = 0
        self._global_sem: Optional[asyncio.Semaphore] = None

    def configure(self, cfg: dict) -> None:
        g = cfg.get("global", {})
        self._global_bucket = _AsyncTokenBucket(float(g.get("max_rps", 0) or 0))
        self._global_max_concurrent = int(g.get("max_concurrent", 0) or 0)
        self._cfg = cfg.get("domains", {})

    def _limiter(self, domain: str) -> _AsyncDomainLimiter:
        if domain not in self._domains:
            dcfg = self._cfg.get(domain, {})
            rps = float(dcfg.get("rps", _DEFAULT_RPS) or 0)
            mc = int(dcfg.get("max_concurrent", _DEFAULT_CONCURRENT) or 0)
            self._domains[domain] = _AsyncDomainLimiter(rps, mc)
        return self._domains[domain]

    def _global_sem_(self) -> Optional[asyncio.Semaphore]:
        if self._global_max_concurrent > 0 and self._global_sem is None:
            self._global_sem = asyncio.Semaphore(self._global_max_concurrent)
        return self._global_sem

    @asynccontextmanager
    async def acquire(self, domain: str):
        sem = self._global_sem_()
        if sem is not None:
            await sem.acquire()
        try:
            await self._global_bucket.acquire()
            async with self._limiter(domain).acquire():
                yield
        finally:
            if sem is not None:
                sem.release()

    @asynccontextmanager
    async def acquire_for_url(self, url: str):
        domain = urlparse(url).hostname or url
        async with self.acquire(domain):
            yield


# ---------------------------------------------------------------------------
# Retry helper (sync)
# ---------------------------------------------------------------------------

def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: bool = True,
) -> T:
    """Call fn() with exponential back-off + full jitter. Raises last exception on exhaustion."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                if jitter:
                    delay *= random.uniform(0.5, 1.5)
                _log.debug(
                    "with_retry attempt %d/%d sleeping %.2fs: %s",
                    attempt + 1, max_retries - 1, delay, exc,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_sync_instance: Optional[RateLimiter] = None
_sync_lock = threading.Lock()

_async_instance: Optional[AsyncRateLimiter] = None


def _load_cfg() -> dict:
    try:
        from utils.config_loader import get_config
        return get_config().get("rate_limits", {})
    except Exception as exc:
        _log.warning("rate_limiter: config load failed (%s) — using defaults", exc)
        return {}


def get_rate_limiter() -> RateLimiter:
    """Return the process-wide sync RateLimiter singleton."""
    global _sync_instance
    if _sync_instance is None:
        with _sync_lock:
            if _sync_instance is None:
                rl = RateLimiter()
                rl.configure(_load_cfg())
                _sync_instance = rl
    return _sync_instance


def get_async_rate_limiter() -> AsyncRateLimiter:
    """Return the asyncio-scoped AsyncRateLimiter singleton (call from async context only)."""
    global _async_instance
    if _async_instance is None:
        rl = AsyncRateLimiter()
        rl.configure(_load_cfg())
        _async_instance = rl
    return _async_instance
