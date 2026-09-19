# Config Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `utils/config_loader.py` — a centralised singleton that parses `config.yaml`, validates required fields, applies `RP_`-prefixed env var overrides, hot-reloads on mtime change, and optionally fires callbacks via watchdog.

**Architecture:** A `ConfigLoader` class is constructed once and stored as a module-level singleton. `get_config()` checks the file's mtime on every call and re-parses when it changes — keeping the last-good config if the new version fails validation. An optional `start_watching()` method layers a `watchdog` `Observer` thread on top for push-based callbacks (used by Streamlit pages).

**Tech Stack:** Python stdlib (`os`, `threading`), `pyyaml` (already in requirements), `watchdog>=4.0` (new optional dep).

---

## File Map

| File                                | Action                           |
| ----------------------------------- | -------------------------------- |
| `utils/config_loader.py`            | **Create** — full implementation |
| `tests/utils/test_config_loader.py` | **Create** — all tests           |
| `requirements.txt`                  | **Modify** — add `watchdog>=4.0` |

---

### Task 1: `ConfigError` and `_validate()`

**Files:**

- Create: `utils/config_loader.py`
- Create: `tests/utils/test_config_loader.py`

- [ ] **Step 1: Write failing tests for `_validate`**

Create `tests/utils/test_config_loader.py`:

```python
import pytest
from utils.config_loader import _validate, ConfigError

VALID_CFG = {
    "database_path": "data/races.db",
    "scrape_interval": 3600,
    "each_way_threshold": 8.0,
    "model": {
        "test_size": 0.2,
        "cv_folds": 5,
        "optuna_trials": 50,
        "iterations": 1000,
    },
    "notifications": {"enabled": False},
    "proxy_pool": {"enabled": False, "proxies": []},
}


def test_validate_valid_config():
    assert _validate(VALID_CFG) == []


def test_validate_missing_database_path():
    cfg = {**VALID_CFG, "database_path": ""}
    errors = _validate(cfg)
    assert any("database_path" in e for e in errors)


def test_validate_scrape_interval_zero():
    cfg = {**VALID_CFG, "scrape_interval": 0}
    errors = _validate(cfg)
    assert any("scrape_interval" in e for e in errors)


def test_validate_test_size_out_of_range():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["test_size"] = 2.0
    errors = _validate(cfg)
    assert any("test_size" in e for e in errors)


def test_validate_cv_folds_too_low():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["cv_folds"] = 1
    errors = _validate(cfg)
    assert any("cv_folds" in e for e in errors)


def test_validate_optuna_trials_zero():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["optuna_trials"] = 0
    errors = _validate(cfg)
    assert any("optuna_trials" in e for e in errors)


def test_validate_iterations_zero():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["iterations"] = 0
    errors = _validate(cfg)
    assert any("iterations" in e for e in errors)


def test_validate_notifications_enabled_missing_token():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {
        "enabled": True,
        "channels": {"telegram": {"bot_token": "", "chat_id": "123"}},
    }
    errors = _validate(cfg)
    assert any("bot_token" in e for e in errors)


def test_validate_notifications_enabled_missing_chat_id():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {
        "enabled": True,
        "channels": {"telegram": {"bot_token": "tok", "chat_id": ""}},
    }
    errors = _validate(cfg)
    assert any("chat_id" in e for e in errors)


def test_validate_notifications_disabled_no_token_ok():
    import copy
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {"enabled": False}
    assert _validate(cfg) == []
```

- [ ] **Step 2: Run tests — verify they all fail**

```
pytest tests/utils/test_config_loader.py -v
```

Expected: `ImportError` or collection error — `utils/config_loader.py` does not exist yet.

- [ ] **Step 3: Implement `ConfigError` and `_validate()`**

Create `utils/config_loader.py`:

```python
from __future__ import annotations

import os
import threading
from typing import Any, Callable

import yaml

from utils.logger import get_logger

_log = get_logger("config")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

_KNOWN_TOP_LEVEL_KEYS = {
    "database_path", "proxy_pool", "timezone", "scrape_interval",
    "each_way_threshold", "low_odds_threshold", "gbp_eur_rate",
    "model_weights", "livescorebet", "betsp_historical", "boylesports",
    "timeform", "model", "retrain", "rate_limits", "notifications",
    "bet_tracker",
}


class ConfigError(Exception):
    pass


def _validate(cfg: dict) -> list[str]:
    errors: list[str] = []

    def _get(path: str):
        node = cfg
        for part in path.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        return node

    # Required scalar fields
    db = _get("database_path")
    if not db or not isinstance(db, str):
        errors.append("database_path must be a non-empty string")

    si = _get("scrape_interval")
    if not isinstance(si, (int, float)) or si <= 0:
        errors.append("scrape_interval must be a number > 0")

    ewt = _get("each_way_threshold")
    if not isinstance(ewt, (int, float)) or ewt <= 0:
        errors.append("each_way_threshold must be a number > 0")

    ts = _get("model.test_size")
    if not isinstance(ts, (int, float)) or not (0 < ts < 1):
        errors.append("model.test_size must be a float between 0 and 1 (exclusive)")

    cf = _get("model.cv_folds")
    if not isinstance(cf, int) or cf < 2:
        errors.append("model.cv_folds must be an int >= 2")

    ot = _get("model.optuna_trials")
    if not isinstance(ot, int) or ot < 1:
        errors.append("model.optuna_trials must be an int >= 1")

    it = _get("model.iterations")
    if not isinstance(it, int) or it < 1:
        errors.append("model.iterations must be an int >= 1")

    # Conditional: telegram credentials required only when notifications enabled
    if _get("notifications.enabled"):
        tok = _get("notifications.channels.telegram.bot_token")
        if not tok or not isinstance(tok, str):
            errors.append("notifications.channels.telegram.bot_token must be non-empty when notifications.enabled=true")
        cid = _get("notifications.channels.telegram.chat_id")
        if not cid or not isinstance(cid, str):
            errors.append("notifications.channels.telegram.chat_id must be non-empty when notifications.enabled=true")

    # Warnings (not errors)
    for key in cfg:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            _log.warning("Unknown top-level config key: %s", key)

    pp = cfg.get("proxy_pool", {})
    if pp.get("enabled") and not pp.get("proxies"):
        _log.warning("proxy_pool.enabled=true but proxies list is empty — will connect directly")

    return errors
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/utils/test_config_loader.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```
git add utils/config_loader.py tests/utils/test_config_loader.py
git commit -m "feat(config): add ConfigError and _validate()"
```

---

### Task 2: `_apply_env_overrides()`

**Files:**

- Modify: `utils/config_loader.py`
- Modify: `tests/utils/test_config_loader.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/utils/test_config_loader.py`:

```python
import copy
from utils.config_loader import _apply_env_overrides


def test_env_override_top_level_int(monkeypatch):
    monkeypatch.setenv("RP_SCRAPE_INTERVAL", "999")
    cfg = copy.deepcopy(VALID_CFG)
    _apply_env_overrides(cfg)
    assert cfg["scrape_interval"] == 999


def test_env_override_nested_float(monkeypatch):
    monkeypatch.setenv("RP_MODEL__TEST_SIZE", "0.3")
    cfg = copy.deepcopy(VALID_CFG)
    _apply_env_overrides(cfg)
    assert cfg["model"]["test_size"] == pytest.approx(0.3)


def test_env_override_bool_false(monkeypatch):
    monkeypatch.setenv("RP_NOTIFICATIONS__ENABLED", "false")
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {"enabled": True}
    _apply_env_overrides(cfg)
    assert cfg["notifications"]["enabled"] is False


def test_env_override_bool_true(monkeypatch):
    monkeypatch.setenv("RP_NOTIFICATIONS__ENABLED", "1")
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {"enabled": False}
    _apply_env_overrides(cfg)
    assert cfg["notifications"]["enabled"] is True


def test_env_override_list(monkeypatch):
    monkeypatch.setenv("RP_MODEL__TARGETS", "won,placed_2")
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["targets"] = ["won", "placed_2", "showed"]
    _apply_env_overrides(cfg)
    assert cfg["model"]["targets"] == ["won", "placed_2"]


def test_env_override_unknown_path_ignored(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("RP_NONEXISTENT__KEY", "value")
    cfg = copy.deepcopy(VALID_CFG)
    with caplog.at_level(logging.WARNING, logger="config"):
        _apply_env_overrides(cfg)
    assert any("NONEXISTENT__KEY" in r.message for r in caplog.records)


def test_env_override_bad_cast_ignored(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("RP_SCRAPE_INTERVAL", "not_a_number")
    cfg = copy.deepcopy(VALID_CFG)
    with caplog.at_level(logging.WARNING, logger="config"):
        _apply_env_overrides(cfg)
    assert cfg["scrape_interval"] == 3600  # unchanged
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/utils/test_config_loader.py -k "env_override" -v
```

Expected: `ImportError` on `_apply_env_overrides`.

- [ ] **Step 3: Implement `_apply_env_overrides()`**

Append to `utils/config_loader.py` (after `_validate`):

```python
def _apply_env_overrides(cfg: dict) -> None:
    prefix = "RP_"
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        without_prefix = raw_key[len(prefix):]
        parts = without_prefix.lower().split("__")

        # Walk the config dict to find the target node and leaf key
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]

        leaf = parts[-1]
        if node is None or not isinstance(node, dict) or leaf not in node:
            _log.warning("RP env var %s has no matching config path — skipped", raw_key)
            continue

        existing = node[leaf]
        try:
            casted = _cast(raw_val, existing)
        except (ValueError, TypeError) as exc:
            _log.warning(
                "RP env var %s: cannot cast %r to %s (%s) — skipped",
                raw_key, raw_val, type(existing).__name__, exc,
            )
            continue

        node[leaf] = casted
        _log.debug("RP env override: %s = %r", ".".join(parts), casted)


def _cast(value: str, existing: Any) -> Any:
    if isinstance(existing, bool):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot parse {value!r} as bool")
    if isinstance(existing, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value  # str
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/utils/test_config_loader.py -v
```

Expected: all 17 tests PASS.

- [ ] **Step 5: Commit**

```
git add utils/config_loader.py tests/utils/test_config_loader.py
git commit -m "feat(config): add _apply_env_overrides() with type casting"
```

---

### Task 3: `ConfigLoader` singleton — load, mtime-poll, `get_config()`, `get()`

**Files:**

- Modify: `utils/config_loader.py`
- Modify: `tests/utils/test_config_loader.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/utils/test_config_loader.py`:

```python
import time
import utils.config_loader as cl_module
from utils.config_loader import get_config, get, ConfigError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ConfigLoader singleton between tests."""
    cl_module._loader = None
    yield
    cl_module._loader = None


def _write_config(tmp_path, data: dict):
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def test_get_config_returns_dict(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    cfg = get_config()
    assert isinstance(cfg, dict)
    assert cfg["scrape_interval"] == 3600


def test_get_config_raises_on_invalid_startup(tmp_path, monkeypatch):
    import copy
    bad = copy.deepcopy(VALID_CFG)
    bad["scrape_interval"] = -1
    p = _write_config(tmp_path, bad)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    with pytest.raises(ConfigError):
        get_config()


def test_get_dotted_key(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    assert get("model.test_size") == pytest.approx(0.2)


def test_get_missing_key_returns_default(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    assert get("model.nonexistent", default="fallback") == "fallback"


def test_mtime_unchanged_no_reparse(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2  # same object — no reparse


def test_mtime_changed_valid_reloads(tmp_path, monkeypatch):
    import yaml, copy
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    get_config()

    updated = copy.deepcopy(VALID_CFG)
    updated["scrape_interval"] = 7200
    time.sleep(0.01)  # ensure mtime changes
    p.write_text(yaml.dump(updated), encoding="utf-8")
    # touch mtime to be strictly newer
    os.utime(str(p), (time.time() + 1, time.time() + 1))

    cfg2 = get_config()
    assert cfg2["scrape_interval"] == 7200


def test_mtime_changed_invalid_keeps_previous(tmp_path, monkeypatch, caplog):
    import logging, yaml, copy
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    get_config()

    bad = copy.deepcopy(VALID_CFG)
    bad["scrape_interval"] = -1
    os.utime(str(p), (time.time() + 1, time.time() + 1))
    p.write_text(yaml.dump(bad), encoding="utf-8")
    os.utime(str(p), (time.time() + 2, time.time() + 2))

    with caplog.at_level(logging.ERROR, logger="config"):
        cfg2 = get_config()

    assert cfg2["scrape_interval"] == 3600  # previous good config retained
    assert any("scrape_interval" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/utils/test_config_loader.py -k "get_config or get_dotted or mtime" -v
```

Expected: FAIL — `get_config`, `get`, `_loader` not defined.

- [ ] **Step 3: Implement `ConfigLoader` and module-level API**

Append to `utils/config_loader.py`:

```python
class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        self._path = config_path
        self._cfg: dict = {}
        self._last_good: dict = {}
        self._mtime: float = -1.0
        self._callbacks: list[Callable[[dict], None]] = []
        self._watch_started: bool = False
        self._lock = threading.Lock()
        # Startup load — raises ConfigError on failure
        self._cfg = self._load(startup=True)
        self._last_good = self._cfg
        self._mtime = os.stat(self._path).st_mtime

    def _load(self, startup: bool = False) -> dict:
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _apply_env_overrides(raw)
        errors = _validate(raw)
        if errors:
            for e in errors:
                _log.error("Config error: %s", e)
            if startup:
                raise ConfigError(f"Invalid config: {'; '.join(errors)}")
            return {}  # sentinel: caller checks empty dict
        return raw

    def _reload(self) -> None:
        with self._lock:
            new_mtime = os.stat(self._path).st_mtime
            if new_mtime == self._mtime:
                return
            self._mtime = new_mtime
            result = self._load(startup=False)
            if not result:
                _log.error("Hot-reload aborted — keeping previous config")
                return
            self._cfg = result
            self._last_good = result
            _log.info("Config reloaded from %s", self._path)
            for cb in list(self._callbacks):
                try:
                    cb(self._cfg)
                except Exception as exc:
                    _log.warning("Config callback %s raised: %s", cb, exc)

    def get_config(self) -> dict:
        self._reload()
        return self._cfg

    def get(self, key_path: str, default: Any = None) -> Any:
        cfg = self.get_config()
        node = cfg
        for part in key_path.split("."):
            if not isinstance(node, dict):
                return default
            node = node.get(part, default)
            if node is default:
                return default
        return node

    def register_callback(self, fn: Callable[[dict], None]) -> None:
        with self._lock:
            if fn not in self._callbacks:
                self._callbacks.append(fn)

    def start_watching(self, interval: float = 2.0) -> None:
        with self._lock:
            if self._watch_started:
                return
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler

                config_dir = os.path.dirname(os.path.abspath(self._path))
                config_name = os.path.basename(self._path)
                loader_ref = self

                class _Handler(FileSystemEventHandler):
                    def on_modified(self, event):
                        if os.path.basename(event.src_path) == config_name:
                            loader_ref._reload()

                observer = Observer()
                observer.schedule(_Handler(), config_dir, recursive=False)
                observer.daemon = True
                observer.start()
                self._watch_started = True
                _log.info("Watchdog observer started for %s (interval=%.1fs)", self._path, interval)
            except ImportError:
                _log.warning("watchdog not installed — hot-reload via mtime-poll only")


# Module-level singleton
_loader: ConfigLoader | None = None
_loader_lock = threading.Lock()


def _get_loader() -> ConfigLoader:
    global _loader
    if _loader is None:
        with _loader_lock:
            if _loader is None:
                _loader = ConfigLoader(_CONFIG_PATH)
    return _loader


def get_config() -> dict:
    return _get_loader().get_config()


def get(key_path: str, default: Any = None) -> Any:
    return _get_loader().get(key_path, default)


def register_callback(fn: Callable[[dict], None]) -> None:
    _get_loader().register_callback(fn)


def start_watching(interval: float = 2.0) -> None:
    _get_loader().start_watching(interval)
```

- [ ] **Step 4: Run all tests — verify they pass**

```
pytest tests/utils/test_config_loader.py -v
```

Expected: all 25+ tests PASS.

- [ ] **Step 5: Commit**

```
git add utils/config_loader.py tests/utils/test_config_loader.py
git commit -m "feat(config): add ConfigLoader singleton with mtime hot-reload"
```

---

### Task 4: `register_callback()` and `start_watching()` tests

**Files:**

- Modify: `tests/utils/test_config_loader.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/utils/test_config_loader.py`:

```python
from utils.config_loader import register_callback, start_watching
import utils.config_loader as cl_module


def test_register_callback_called_on_reload(tmp_path, monkeypatch):
    import yaml, copy
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))

    calls = []
    register_callback(lambda cfg: calls.append(cfg["scrape_interval"]))

    updated = copy.deepcopy(VALID_CFG)
    updated["scrape_interval"] = 9999
    os.utime(str(p), (time.time() + 1, time.time() + 1))
    p.write_text(yaml.dump(updated), encoding="utf-8")
    os.utime(str(p), (time.time() + 2, time.time() + 2))

    get_config()
    assert calls == [9999]


def test_register_callback_not_duplicated(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    fn = lambda cfg: None
    register_callback(fn)
    register_callback(fn)
    assert cl_module._get_loader()._callbacks.count(fn) == 1


def test_start_watching_idempotent(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    start_watching()
    start_watching()  # second call must not raise or spawn second observer
    assert cl_module._get_loader()._watch_started is True


def test_start_watching_without_watchdog(tmp_path, monkeypatch, caplog):
    import logging
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("watchdog"):
            raise ImportError("watchdog not installed")
        return real_import(name, *args, **kwargs)

    with caplog.at_level(logging.WARNING, logger="config"):
        monkeypatch.setattr(builtins, "__import__", mock_import)
        start_watching()

    assert any("watchdog" in r.message for r in caplog.records)
    assert cl_module._get_loader()._watch_started is False
```

- [ ] **Step 2: Run tests — verify they pass (implementation already complete)**

```
pytest tests/utils/test_config_loader.py -v
```

Expected: all tests PASS (implementation from Task 3 covers these).

- [ ] **Step 3: Commit**

```
git add tests/utils/test_config_loader.py
git commit -m "test(config): add callback and start_watching tests"
```

---

### Task 5: Add `watchdog` to requirements and run full suite

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1: Add watchdog**

In `requirements.txt`, append:

```
watchdog>=4.0
```

- [ ] **Step 2: Install**

```
pip install watchdog>=4.0
```

- [ ] **Step 3: Run full test suite**

```
pytest --tb=short -q
```

Expected: 487 existing tests + all new config_loader tests PASS. No regressions.

- [ ] **Step 4: Commit**

```
git add requirements.txt
git commit -m "chore: add watchdog>=4.0 for config hot-reload"
```

---

### Task 6: Smoke-test with real `config.yaml`

**Files:** none modified

- [ ] **Step 1: Run smoke test**

```
python -c "
from utils.config_loader import get_config, get
cfg = get_config()
print('scrape_interval:', cfg['scrape_interval'])
print('model.test_size via get():', get('model.test_size'))
print('missing key default:', get('does.not.exist', 'ok'))
print('Config loaded OK')
"
```

Expected output (exact values from your `config.yaml`):

```
scrape_interval: 3600
model.test_size via get(): 0.2
missing key default: ok
Config loaded OK
```

- [ ] **Step 2: Test env var override end-to-end**

```
RP_SCRAPE_INTERVAL=999 python -c "
from utils.config_loader import get_config
cfg = get_config()
assert cfg['scrape_interval'] == 999, cfg['scrape_interval']
print('Env var override OK:', cfg['scrape_interval'])
"
```

On Windows PowerShell:

```powershell
$env:RP_SCRAPE_INTERVAL = "999"
python -c "from utils.config_loader import get_config; cfg = get_config(); print(cfg['scrape_interval'])"
$env:RP_SCRAPE_INTERVAL = ""
```

Expected: `999`

- [ ] **Step 3: Final commit**

```
git add .
git commit -m "feat(config): config_loader complete — parse, validate, env overrides, hot-reload"
```

---

## Self-Review Against Spec

| Spec requirement                                | Covered in                                   |
| ----------------------------------------------- | -------------------------------------------- |
| Parse `config.yaml`                             | Task 3 `ConfigLoader.__init__` → `_load()`   |
| Validate required fields                        | Task 1 `_validate()`                         |
| Fail fast at startup                            | Task 3 `startup=True` → raises `ConfigError` |
| Keep last-good on hot-reload failure            | Task 3 `_reload()` rollback path             |
| Env var overrides `RP_X__Y` format              | Task 2 `_apply_env_overrides()`              |
| Type casting (int, float, bool, list)           | Task 2 `_cast()`                             |
| Unknown env var path → WARNING                  | Task 2 test + implementation                 |
| Bad cast → WARNING, skip                        | Task 2 test + implementation                 |
| `get_config()` mtime-poll                       | Task 3 `_reload()`                           |
| `get(key_path, default)`                        | Task 3 `ConfigLoader.get()`                  |
| `register_callback()`                           | Task 3 + Task 4 tests                        |
| `start_watching()` watchdog                     | Task 3 `ConfigLoader.start_watching()`       |
| `start_watching()` idempotent                   | Task 4 test                                  |
| Watchdog `ImportError` fallback + WARNING       | Task 4 test                                  |
| Logging via `get_logger("config")`              | All tasks — `_log` used throughout           |
| `watchdog>=4.0` in requirements                 | Task 5                                       |
| Singleton pattern (`get_config()` module-level) | Task 3 `_get_loader()`                       |
| Unknown top-level keys → WARNING                | Task 1 `_validate()`                         |
| `proxy_pool.enabled` with no proxies → WARNING  | Task 1 `_validate()`                         |
