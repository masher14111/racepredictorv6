# Race Predictor v3 — Utility Layer Review

_Reviewed: 2026-06-13 | Test baseline: 604/607 pass (3 skipped — reportlab missing)_

---

## Config Loader

**Verdict: GOOD with one real bug.**

**What works:**

- Double-checked locking singleton is correct.
- `_reload()` holds `self._lock` for the entire stat + load cycle — thread-safe.
- `get_config()` calls `_reload()` on every invocation, giving lazy mtime polling with no separate watcher thread required.
- Startup raises `ConfigError` on invalid config; hot-reload keeps the previous good copy — correct defensive behavior.
- Callback list is snapshot-copied (`list(self._callbacks)`) before iteration — safe against mutation during callback.
- `start_watching()` is idempotent (guarded by `_watch_started`).

**Bugs:**

1. **`reporter` missing from `_KNOWN_TOP_LEVEL_KEYS`** (`config_loader.py:16–21`). `config.yaml` has a `reporter:` block; the loader logs `WARNING Unknown top-level config key: reporter` on every load. Harmless but pollutes logs.
2. **`start_watching()` is never called automatically.** The watchdog observer only fires if a caller explicitly invokes it. mtime polling still works (via `_reload()` on each `get_config()` call), but watchdog's faster inotify/FSEvents path is dead unless the application explicitly opts in.
3. **`get()` returns `self._cfg` without holding the lock.** Between `_reload()` releasing the lock and the caller using the returned dict, another thread could swap `_cfg`. In CPython the GIL makes dict-reference assignment atomic so this is safe in practice, but it is a correctness gap for non-CPython runtimes.

---

## Cache

**Verdict: SOLID — no critical bugs.**

**What works:**

- All reads/writes/deletes acquire `self._lock` — thread-safe.
- `set()` uses atomic `.tmp` → `Path.replace()` — no partial-write corruption.
- Legacy `fetched_at` format handled transparently in `_expiry()` and `get_stale()`.
- `clear()` skips `.gitkeep` — intentional.
- Singleton TTL bootstraps from `scrape_interval` in config, with fallback to 3600 s.
- `get_stale()` is a correct fallback for graceful degradation after a live fetch failure.

**Edge-case gaps (non-critical):**

1. **Key collision via sanitization.** `_path()` maps any non-alphanumeric character to `_`. Two distinct keys that differ only in special characters (e.g. `race/2026-06-13` and `race|2026-06-13`) both resolve to `race_2026_06_13`. Callers should use only alphanumeric + `-_.` keys to avoid silent overwrites.
2. **`keys()` returns sanitized stems, not original keys.** If callers need to reconstruct the original key string they cannot — stems are information-lossy. Document this contract.

---

## Notifications / Telegram

**Verdict: WORKS END-TO-END — confirmed send path is correct. Two low-severity gaps.**

### Confirmed send path

```
config.yaml
  notifications.enabled: true                              ✓
  channels.telegram.enabled: true                          ✓
  channels.telegram.bot_token: "8648732935:AAErLnp..."    ✓
  channels.telegram.chat_id:   "6618994572"               ✓
        ↓
get_notifier() → Notifier.__init__()
  self._enabled = True
  tg.get("enabled")    → True
  tg.get("bot_token")  → truthy string
  tg.get("chat_id")    → "6618994572"
        ↓
_TelegramChannel("8648732935:AAErLnp...", "6618994572")
  self._url = "https://api.telegram.org/bot8648732935:AAErLnp.../sendMessage"
        ↓
_NotificationQueue([telegram_channel])  →  daemon worker thread started
  self._active = True
        ↓
notify_*() → _send(text) → queue.put_nowait(_Message(text))
        ↓  (non-blocking; returns immediately)
worker thread: q.get(timeout=5) → ch.send(msg)
        ↓
httpx.post(url, json={chat_id, text, parse_mode="HTML"}, timeout=8.0)
        ↓
Telegram API  →  message delivered
```

Config validator in `config_loader._validate()` checks `notifications.channels.telegram.bot_token` and `chat_id` via the same dotted-path traversal that matches the YAML structure — no mismatch.

**What works:**

- Channel activation guard (`tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id")`) is correct.
- `str(chat_id)` cast in `_TelegramChannel.__init__` handles both string and integer chat IDs.
- Worker thread is a daemon — will not block process exit.
- `queue.Empty` timeout loop is correct; `None` sentinel provides graceful shutdown.
- Per-channel exception isolation (`try/except` inside the channel loop) prevents one bad channel from silently skipping others.
- All `notify_*` message templates use HTML parse mode with correct `<b>` tags.

**Gaps (non-critical):**

1. **No retry on failed sends.** `_TelegramChannel.send()` logs the error and drops the message. A transient network hiccup silently loses the alert. For a bet-outcome notification this is significant — consider a simple 1-retry before drop.
2. **Worker thread always starts even when `channels = []`** (i.e. notifications disabled). The thread loops on `q.get(timeout=5)` and does nothing, consuming ~1 thread slot indefinitely. Trivial fix: only start the thread when `channels` is non-empty.

---

## Bet Tracker

**Verdict: CORRECT — edge cases handled. One data-race on bankroll read.**

**What works:**

- `all_bets(settled_only=False)` returns `[]` on empty DB — correct.
- `summary()` early-returns a zeroed dict when `df.empty` — correct shape, all required dashboard keys present.
- `pl_series()` returns an empty DataFrame with the correct 8-column schema on empty DB — caller can safely call `.empty` check before plotting.
- `breakdown(by=)` returns empty DataFrame when DB is empty or `by` column is absent — caller must handle empty result but won't raise.
- Each-way settlement: `win_return + place_return` where win leg = `half * odds` and place leg = `half * ((odds-1)*ew_fraction+1)` — matches standard UK/Ireland EW terms correctly.
- `settle_bet()` guards re-settlement (`if row["outcome"] is not None`) and validates outcome string.
- Stop-loss check fires `notify_stop_loss()` immediately after the INSERT commits — correct sequencing.

**Bugs:**

1. **TOCTOU on bankroll read in `record_bet()`** (`bet_tracker.py:189`). `bankroll_before = self.bankroll` is evaluated outside the `write_lock()` context. If two bets are placed concurrently, both could read the same `bankroll_before`, and the second INSERT would use a stale pre-deduction balance. Fix: move `bankroll_before = self.bankroll` inside the `write_lock()` block, or read `balance` from the last `bankroll_log` row inside the same transaction.
2. **`export_csv()` writes placeholder text on empty DB** (`bet_tracker.py:411`). The file contains `"no bets recorded\n"`, not a valid CSV. Any caller that subsequently does `pd.read_csv(path)` will get a confusing one-row DataFrame. Prefer writing an empty CSV with headers: `pd.DataFrame(columns=[...]).to_csv(out, index=False)`.

---

## Reporter

**Verdict: SOLID — one wasted BetTracker instantiation, scheduler first-run delay is by design.**

**What works:**

- `generate()` wraps each format in its own try/except — a PDF failure never aborts CSV/JSON export.
- `IntegrityChecker.check_bets()` raises `IntegrityError` on missing required columns (fatal) and returns warnings for soft violations (stake ≤ 0, odds < 1, bad outcomes) — correct severity split.
- `IntegrityChecker.check_referential()` runs a LEFT JOIN SQLite query for orphaned `bankroll_log.bet_id` — correct approach, direct connection not via ORM.
- SHA-256 manifest uses 65536-byte chunked reads — memory-safe for large files.
- JSON and manifest use atomic `.tmp` → `.replace()` writes.
- `_load_predictions()` falls back to `data/predictions.json` on cache miss.
- PDF `ImportError` is caught and added to `result.warnings` — clean degradation when `reportlab` is absent.
- Scheduler uses `Event.wait(timeout=interval)` — correct; `stop_event.set()` terminates cleanly.

**Gaps (non-critical):**

1. **`export_pdf()` creates a second `BetTracker` instance** (`reporter.py:464`). `_load_bets()` already constructs one; `export_pdf()` constructs another to get `pl_series()` and `breakdown()`. No correctness issue — both read the same DB — but it opens a second connection pool unnecessarily. Pass the `BetTracker` instance from `_load_bets()` into `export_pdf()`.
2. **Scheduler first run is delayed by `interval_hours`** (24 h by default). There is no immediate run on scheduler start. If the process restarts mid-day, the next report is up to 24 h away. Consider calling `generate()` once synchronously at startup before entering the wait loop, or document this behavior.
3. **Manifest row-count for placeholder CSVs**: When bets CSV contains `"no bets recorded\n"`, the row count computes as `max(0, 1 - 1) = 0`. Correct, but only because the placeholder is a single non-header line.
4. **`reporter` block absent from config_loader's `_KNOWN_TOP_LEVEL_KEYS`** (see Config Loader bug #1 above) — causes spurious log warnings on every config load.

---

## Critical Fixes Needed (ordered by severity)

| #   | Severity   | File                         | Description                                                                                                                                                                               |
| --- | ---------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **HIGH**   | `utils/bet_tracker.py:189`   | TOCTOU bankroll read: `bankroll_before` read outside `write_lock()`. Two concurrent bets can both deduct from the same pre-bet balance, overcrediting the bankroll. Move inside the lock. |
| 2   | **MEDIUM** | `utils/config_loader.py:16`  | `reporter` missing from `_KNOWN_TOP_LEVEL_KEYS`. Logs a spurious warning on every config load. Add `"reporter"` to the set.                                                               |
| 3   | **LOW**    | `utils/notifications.py:101` | Worker thread always starts even when `channels = []` (notifications disabled). No functional harm but wastes a thread. Guard: `if channels: self._thread.start()`.                       |
| 4   | **LOW**    | `utils/bet_tracker.py:411`   | `export_csv()` writes non-CSV placeholder text on empty DB. Break downstream `pd.read_csv()` callers. Write an empty-with-headers CSV instead.                                            |
| 5   | **LOW**    | `utils/notifications.py:80`  | No retry on failed Telegram sends. Transient network error silently drops bet-outcome / stop-loss notifications. Add one retry with short sleep before dropping.                          |

---

## Next

**Prompt 3:** Fix the bankroll TOCTOU (bug #1) and `reporter` key warning (bug #2), then install `reportlab` (`pip install reportlab`) to un-skip the 3 PDF tests and confirm the full reporter pipeline end-to-end.
