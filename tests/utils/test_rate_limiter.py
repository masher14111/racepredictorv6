"""Tests for utils/rate_limiter.py — token bucket, concurrency, retry, singletons."""
import asyncio
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# _TokenBucket
# ---------------------------------------------------------------------------

def test_token_bucket_unlimited_never_blocks():
    from utils.rate_limiter import _TokenBucket
    b = _TokenBucket(rps=0)
    start = time.monotonic()
    for _ in range(10):
        b.acquire()
    assert time.monotonic() - start < 0.05


def test_token_bucket_first_call_immediate():
    from utils.rate_limiter import _TokenBucket
    b = _TokenBucket(rps=1.0)
    start = time.monotonic()
    b.acquire()
    assert time.monotonic() - start < 0.05  # bucket starts full


def test_token_bucket_second_call_waits():
    from utils.rate_limiter import _TokenBucket
    b = _TokenBucket(rps=1.0)
    b.acquire()
    start = time.monotonic()
    b.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.9, f"expected ~1s wait, got {elapsed:.2f}s"


def test_token_bucket_high_rate_does_not_wait():
    from utils.rate_limiter import _TokenBucket
    b = _TokenBucket(rps=100.0)
    b.acquire()
    start = time.monotonic()
    b.acquire()
    assert time.monotonic() - start < 0.1


def test_token_bucket_capacity_caps_burst():
    from utils.rate_limiter import _TokenBucket
    # rps=2.0 → capacity=2; 3 calls: 2 immediate, 3rd waits
    b = _TokenBucket(rps=2.0)
    start = time.monotonic()
    b.acquire()
    b.acquire()
    elapsed_two = time.monotonic() - start
    assert elapsed_two < 0.1  # two burst tokens
    b.acquire()
    elapsed_three = time.monotonic() - start
    assert elapsed_three >= 0.4  # must wait for refill


# ---------------------------------------------------------------------------
# RateLimiter — acquire context manager
# ---------------------------------------------------------------------------

def test_rate_limiter_acquire_by_domain():
    from utils.rate_limiter import RateLimiter
    rl = RateLimiter()
    rl.configure({"domains": {"test.example": {"rps": 100.0, "max_concurrent": 2}}})
    with rl.acquire("test.example"):
        pass  # should not raise


def test_rate_limiter_acquire_for_url_extracts_hostname():
    from utils.rate_limiter import RateLimiter
    rl = RateLimiter()
    rl.configure({"domains": {"example.com": {"rps": 100.0, "max_concurrent": 1}}})
    with rl.acquire_for_url("https://example.com/path?q=1"):
        pass


def test_rate_limiter_unknown_domain_uses_defaults():
    from utils.rate_limiter import RateLimiter
    rl = RateLimiter()
    rl.configure({})
    # Default rps=1.0, bucket starts full → immediate
    start = time.monotonic()
    with rl.acquire("new-domain.ie"):
        pass
    assert time.monotonic() - start < 0.1


def test_rate_limiter_concurrency_semaphore_blocks_excess():
    from utils.rate_limiter import RateLimiter
    rl = RateLimiter()
    rl.configure({"domains": {"slow.ie": {"rps": 0, "max_concurrent": 1}}})

    entry_times = []
    lock = threading.Lock()

    def worker():
        with rl.acquire("slow.ie"):
            with lock:
                entry_times.append(time.monotonic())
            time.sleep(0.1)  # hold slot so second thread must wait

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(entry_times) == 2  # both completed
    # max_concurrent=1: second entry must be at least 0.09s after the first
    assert abs(entry_times[1] - entry_times[0]) >= 0.09


def test_rate_limiter_stats_returns_configured_domains():
    from utils.rate_limiter import RateLimiter
    rl = RateLimiter()
    rl.configure({"domains": {
        "a.com": {"rps": 2.0, "max_concurrent": 1},
        "b.com": {"rps": 0.5, "max_concurrent": 1},
    }})
    with rl.acquire("a.com"):
        pass
    with rl.acquire("b.com"):
        pass
    stats = {s["domain"]: s["rps"] for s in rl.stats()}
    assert stats["a.com"] == 2.0
    assert stats["b.com"] == 0.5


def test_rate_limiter_global_rps_applied():
    from utils.rate_limiter import RateLimiter
    rl = RateLimiter()
    rl.configure({"global": {"max_rps": 100.0, "max_concurrent": 0}, "domains": {}})
    start = time.monotonic()
    with rl.acquire("any.ie"):
        pass
    assert time.monotonic() - start < 0.1


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------

def test_with_retry_success_first_attempt():
    from utils.rate_limiter import with_retry
    calls = []
    result = with_retry(lambda: calls.append(1) or "ok", max_retries=3)
    assert result == "ok"
    assert len(calls) == 1


def test_with_retry_succeeds_on_second_attempt():
    from utils.rate_limiter import with_retry
    attempts = [0]

    def fn():
        attempts[0] += 1
        if attempts[0] < 2:
            raise ValueError("boom")
        return "done"

    result = with_retry(fn, max_retries=3, base_delay=0.01, jitter=False)
    assert result == "done"
    assert attempts[0] == 2


def test_with_retry_raises_after_exhaustion():
    from utils.rate_limiter import with_retry
    with pytest.raises(RuntimeError, match="exhausted"):
        with_retry(lambda: (_ for _ in ()).throw(RuntimeError("exhausted")),
                   max_retries=3, base_delay=0.01, jitter=False)


def test_with_retry_no_jitter_delay_is_deterministic():
    from utils.rate_limiter import with_retry
    calls = []

    def fn():
        calls.append(time.monotonic())
        raise OSError("retry")

    with pytest.raises(OSError):
        with_retry(fn, max_retries=3, base_delay=0.05, jitter=False)

    assert len(calls) == 3
    # Gap between attempt 1→2 should be ~0.05s, attempt 2→3 should be ~0.10s
    assert calls[1] - calls[0] >= 0.04
    assert calls[2] - calls[1] >= 0.08


# ---------------------------------------------------------------------------
# get_rate_limiter singleton
# ---------------------------------------------------------------------------

def test_get_rate_limiter_returns_singleton(monkeypatch):
    import utils.rate_limiter as _mod
    monkeypatch.setattr(_mod, "_sync_instance", None)
    from utils.rate_limiter import get_rate_limiter, RateLimiter
    a = get_rate_limiter()
    b = get_rate_limiter()
    assert a is b
    assert isinstance(a, RateLimiter)


def test_get_rate_limiter_loads_without_config(tmp_path, monkeypatch):
    """Should not raise if config.yaml is absent; falls back to defaults."""
    import utils.rate_limiter as _mod
    monkeypatch.setattr(_mod, "_sync_instance", None)
    monkeypatch.setattr(_mod, "_load_cfg", lambda: {})
    from utils.rate_limiter import get_rate_limiter
    rl = get_rate_limiter()
    with rl.acquire("fallback.ie"):
        pass  # should not raise


# ---------------------------------------------------------------------------
# AsyncRateLimiter
# ---------------------------------------------------------------------------

def test_async_rate_limiter_acquire():
    from utils.rate_limiter import AsyncRateLimiter

    async def _run():
        rl = AsyncRateLimiter()
        rl.configure({"domains": {"async.ie": {"rps": 100.0, "max_concurrent": 2}}})
        async with rl.acquire("async.ie"):
            pass

    asyncio.run(_run())


def test_async_rate_limiter_first_call_immediate():
    from utils.rate_limiter import AsyncRateLimiter

    async def _run():
        rl = AsyncRateLimiter()
        rl.configure({"domains": {"t.ie": {"rps": 1.0, "max_concurrent": 1}}})
        start = asyncio.get_event_loop().time()
        async with rl.acquire("t.ie"):
            pass
        assert asyncio.get_event_loop().time() - start < 0.1

    asyncio.run(_run())


def test_async_rate_limiter_acquire_for_url():
    from utils.rate_limiter import AsyncRateLimiter

    async def _run():
        rl = AsyncRateLimiter()
        rl.configure({})
        async with rl.acquire_for_url("https://example.ie/path"):
            pass

    asyncio.run(_run())


def test_get_async_rate_limiter_singleton(monkeypatch):
    import utils.rate_limiter as _mod
    monkeypatch.setattr(_mod, "_async_instance", None)
    from utils.rate_limiter import get_async_rate_limiter, AsyncRateLimiter
    a = get_async_rate_limiter()
    b = get_async_rate_limiter()
    assert a is b
    assert isinstance(a, AsyncRateLimiter)
