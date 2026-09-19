from __future__ import annotations

import os
import threading
from typing import Any, Callable

import yaml

from utils.logger import get_logger

_log = get_logger("config")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
# Git-ignored overlay for secrets (bot tokens, proxy creds). Deep-merged over
# config.yaml before env overrides, so it wins over committed placeholders.
_LOCAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.local.yaml")

_KNOWN_TOP_LEVEL_KEYS = {
    "database_path", "proxy_pool", "timezone", "scrape_interval",
    "each_way_threshold", "low_odds_threshold", "gbp_eur_rate", "uk_ire_only",
    "livescorebet", "betsp_historical", "boylesports", "paddypower",
    "timeform", "model", "retrain", "rate_limits", "notifications",
    "bet_tracker", "reporter", "llm", "value", "logging",
    "features", "orchestration", "staleness", "execution",
    "firecrawl", "staking",
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

    if _get("notifications.enabled"):
        tok = _get("notifications.channels.telegram.bot_token")
        if not tok or not isinstance(tok, str):
            errors.append(
                "notifications.channels.telegram.bot_token must be non-empty "
                "when notifications.enabled=true"
            )
        cid = _get("notifications.channels.telegram.chat_id")
        if not cid or not isinstance(cid, str):
            errors.append(
                "notifications.channels.telegram.chat_id must be non-empty "
                "when notifications.enabled=true"
            )

    for key in cfg:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            _log.warning("Unknown top-level config key: %s", key)

    pp = cfg.get("proxy_pool", {})
    if pp.get("enabled") and not pp.get("proxies"):
        _log.warning("proxy_pool.enabled=true but proxies list is empty — will connect directly")

    return errors


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
    return value


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge overlay into base in-place (dicts merge, scalars/lists replace)."""
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _apply_local_overlay(cfg: dict) -> None:
    """Merge git-ignored config.local.yaml (secrets) over the committed config."""
    if not os.path.exists(_LOCAL_CONFIG_PATH):
        return
    try:
        with open(_LOCAL_CONFIG_PATH, "r", encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        _log.warning("Could not read config.local.yaml — skipped (%s)", exc)
        return
    if isinstance(local, dict):
        _deep_merge(cfg, local)
        _log.info("Merged secrets overlay from config.local.yaml")


def _apply_env_overrides(cfg: dict) -> None:
    prefix = "RP_"
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        without_prefix = raw_key[len(prefix):]
        parts = without_prefix.lower().split("__")

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


class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        self._path = config_path
        self._cfg: dict = {}
        self._last_good: dict = {}
        self._mtime: float = -1.0
        self._callbacks: list[Callable[[dict], None]] = []
        self._watch_started: bool = False
        self._lock = threading.Lock()
        self._cfg = self._load(startup=True)
        self._last_good = self._cfg
        self._mtime = os.stat(self._path).st_mtime
        _log.info("Config loaded from %s", self._path)

    def _load(self, startup: bool = False) -> dict:
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _apply_local_overlay(raw)
        _apply_env_overrides(raw)
        errors = _validate(raw)
        if errors:
            for e in errors:
                _log.error("Config error: %s", e)
            if startup:
                raise ConfigError(f"Invalid config: {'; '.join(errors)}")
            return {}
        return raw

    def _reload(self) -> None:
        with self._lock:
            try:
                new_mtime = os.stat(self._path).st_mtime
            except OSError:
                return
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
                _log.info(
                    "Watchdog observer started for %s (interval=%.1fs)",
                    self._path, interval,
                )
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
