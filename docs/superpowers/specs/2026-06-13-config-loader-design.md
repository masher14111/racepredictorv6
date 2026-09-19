# Config Loader Design

**Date:** 2026-06-13  
**Status:** Approved  
**Scope:** `utils/config_loader.py` — centralised config parsing, validation, env-var overrides, and hot-reload

---

## Problem

Config loading is duplicated across ~15 modules. Each opens `config.yaml` independently with `yaml.safe_load`, no validation, no env-var override support, and no hot-reload. A misconfigured field silently returns `None` at runtime.

---

## Goals

- Single source of truth for parsed config
- Fail fast at startup on invalid/missing required fields
- Env var overrides: `RP_<SECTION>__<KEY>=value` (double-underscore nesting, `RP_` prefix)
- Hot-reload: mtime-poll on every `get_config()` call; optional watchdog callbacks for Streamlit
- Structured logging via `utils/logger.py`
- Drop-in replacement: `from utils.config_loader import get_config` returns the same plain dict

---

## Architecture

### `ConfigLoader` (singleton class)

Internal state:

| Field            | Type                           | Purpose                                          |
| ---------------- | ------------------------------ | ------------------------------------------------ |
| `_cfg`           | `dict`                         | Active validated config                          |
| `_last_good`     | `dict`                         | Last known-good config (for hot-reload rollback) |
| `_mtime`         | `float`                        | File mtime at last load                          |
| `_callbacks`     | `list[Callable[[dict], None]]` | Hot-reload listeners                             |
| `_watch_started` | `bool`                         | Guard for idempotent `start_watching()`          |

### Module-level public API

```python
get_config() -> dict                          # mtime-checked on every call
get(key_path: str, default=None) -> Any       # dotted accessor: get("model.test_size")
register_callback(fn: Callable[[dict], None]) # register hot-reload listener
start_watching(interval: float = 2.0)         # spawn watchdog observer (idempotent)
```

### `ConfigError(Exception)`

Raised only on first load (startup) when required fields are missing or invalid. Never raised during hot-reload cycles — errors are logged and the previous config is retained.

---

## Load Sequence

```
startup                      hot-reload (get_config())
   │                               │
open config.yaml              check mtime
   │                               │ changed?
yaml.safe_load                     │ yes
   │                          yaml.safe_load
apply env overrides           apply env overrides
   │                               │
_validate(cfg)                _validate(cfg)
   │                               │ errors?
   │ errors?                       │ yes → log ERROR, keep _last_good
   │ yes → raise ConfigError       │ no  → swap _cfg, call callbacks
   │ no  → set _cfg, _last_good    │
   ▼                               ▼
return cfg                    return _cfg
```

---

## Env Var Overrides

**Format:** `RP_` prefix + section + `__` separator + key (case-insensitive matching):

```
RP_MODEL__TEST_SIZE=0.15          → model.test_size  (float, cast from YAML type)
RP_SCRAPE_INTERVAL=1800           → scrape_interval  (int)
RP_NOTIFICATIONS__ENABLED=false   → notifications.enabled  (bool)
RP_MODEL__TARGETS=won,placed_2    → model.targets  (list, comma-split)
```

**Type casting rules:**

- Cast to the Python type of the existing YAML value
- `bool`: `"true"/"1"/"yes"` → `True`; `"false"/"0"/"no"` → `False` (case-insensitive)
- `list`: comma-split the string
- Unknown path (no matching YAML key): log `WARNING`, skip
- Cast failure: log `WARNING` with original value, skip

Applied after YAML parse, before validation.

---

## Validation

`_validate(cfg: dict) -> list[str]` returns a list of error strings. Empty = valid.

**Required fields:**

| Field                                       | Rule                                                     |
| ------------------------------------------- | -------------------------------------------------------- |
| `database_path`                             | non-empty string                                         |
| `scrape_interval`                           | int > 0                                                  |
| `each_way_threshold`                        | float > 0                                                |
| `model.test_size`                           | 0 < float < 1                                            |
| `model.cv_folds`                            | int ≥ 2                                                  |
| `model.optuna_trials`                       | int ≥ 1                                                  |
| `model.iterations`                          | int ≥ 1                                                  |
| `notifications.channels.telegram.bot_token` | non-empty string, only if `notifications.enabled = true` |
| `notifications.channels.telegram.chat_id`   | non-empty string, only if `notifications.enabled = true` |

**Warnings (logged, not errors):**

- Unknown top-level keys (not in a known set)
- `proxy_pool.enabled = true` with empty `proxies` list

---

## Hot-Reload: Mtime Poll (default)

`get_config()` reads `os.stat(config_path).st_mtime`. If different from `_mtime`, calls internal `_reload()`. Cost: one `stat` syscall per call — negligible.

## Hot-Reload: Watchdog (optional, Streamlit)

`start_watching(interval=2.0)` imports `watchdog` and spawns an `Observer` thread watching the config file's directory for `FileModifiedEvent`. On event it calls `_reload()`. Idempotent — second call is a no-op.

**Usage in Streamlit pages:**

```python
from utils.config_loader import register_callback, start_watching

def _on_config_change(cfg):
    st.cache_data.clear()

register_callback(_on_config_change)
start_watching()
```

Watchdog is an optional dep — `ImportError` is caught and logged as `WARNING`; the loader falls back to mtime-poll only.

---

## Logging

All via `get_logger("config")` from `utils/logger.py`:

| Event                                                     | Level                            |
| --------------------------------------------------------- | -------------------------------- |
| Config loaded / reloaded successfully                     | `INFO`                           |
| Env var override applied                                  | `DEBUG`                          |
| Env var override skipped (unknown path / cast fail)       | `WARNING`                        |
| Unknown top-level config key                              | `WARNING`                        |
| `proxy_pool.enabled` with no proxies                      | `WARNING`                        |
| Validation errors on hot-reload (keeping previous config) | `ERROR`                          |
| Validation errors on startup                              | `ERROR` then raise `ConfigError` |

---

## Migration Path

Existing `_load_config()` functions across scrapers/models/utils can be replaced one-by-one. No flag day required.

```python
# before (in every module)
with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# after
from utils.config_loader import get_config
_cfg = get_config()
```

`get_config()` returns the same plain dict the modules already expect.

---

## Files Changed

| File                     | Action                                               |
| ------------------------ | ---------------------------------------------------- |
| `utils/config_loader.py` | **New** — full implementation                        |
| `requirements.txt`       | Add `watchdog>=4.0` (optional, for `start_watching`) |

Existing callers (`scraper/`, `models/`, `utils/`, `ui/`) are **not** changed as part of this task — migration is a separate step.

---

## Testing

- Load valid config → returns dict with correct values
- Startup with missing required field → raises `ConfigError`
- Startup with invalid type (e.g. `test_size: 2.0`) → raises `ConfigError`
- Env var override: `RP_SCRAPE_INTERVAL=999` → `get_config()["scrape_interval"] == 999`
- Env var unknown path → warning logged, ignored
- Env var bad cast → warning logged, ignored
- Mtime unchanged → same object returned (no re-parse)
- Mtime changed, valid update → new config returned, callbacks fired
- Mtime changed, invalid update → error logged, previous config retained
- `start_watching()` called twice → one observer thread only
- Watchdog not installed → falls back to mtime-poll, logs warning
