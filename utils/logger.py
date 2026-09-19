import logging
import os

import yaml

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_PATH = os.path.join(LOG_DIR, "race_predictor.log")

# Read rotation settings directly from config.yaml (cannot use config_loader here
# because config_loader imports this module, which would create a circular import).
def _read_log_config() -> tuple[int, int]:
    _cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        _log_cfg = _cfg.get("logging", {})
        return (
            int(_log_cfg.get("max_bytes", 5 * 1024 * 1024)),
            int(_log_cfg.get("backup_count", 2)),
        )
    except Exception:
        return 5 * 1024 * 1024, 2


_MAX_BYTES, _BACKUP_COUNT = _read_log_config()

_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Shared handler instances — created once and attached to the root logger.
# Why not RotatingFileHandler: it rotates by os.rename(), which fails on Windows
# with WinError 32 whenever any other handle (another thread, Streamlit, a second
# process) has the file open — a constant condition under parallel scraping. A
# plain FileHandler never renames, so rotation can never crash logging. Growth is
# bounded by a one-time startof-run size check below.
_file_handler: logging.FileHandler | None = None
_console_handler: logging.StreamHandler | None = None


def _roll_if_large() -> None:
    """One-shot size cap at import time, before any threads start.

    Renaming here is safe: no handler holds the file open yet. If even this
    fails (another process has it open), we silently keep appending.
    """
    try:
        if not (os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _MAX_BYTES):
            return
        # Rotate: .2 ← .1 ← live (up to _BACKUP_COUNT slots)
        for i in range(_BACKUP_COUNT - 1, 0, -1):
            src = f"{_LOG_PATH}.{i}"
            dst = f"{_LOG_PATH}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.replace(src, dst)
        os.replace(_LOG_PATH, f"{_LOG_PATH}.1")
    except OSError:
        pass


def _ensure_root_handlers() -> None:
    global _file_handler, _console_handler
    root = logging.getLogger()
    if _file_handler is not None:
        return

    root.setLevel(logging.DEBUG)

    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(_FMT)

    _roll_if_large()
    _file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8", delay=True)
    # File at INFO: DEBUG lines (e.g. per-request proxy selection) flooded the
    # file. Raise to DEBUG here if you need verbose file logs.
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(_FMT)

    root.addHandler(_console_handler)
    root.addHandler(_file_handler)

    # Third-party libraries that log one line per HTTP request flood the console
    # and bury our own progress lines. Quiet them to WARNING.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    _ensure_root_handlers()
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Propagate to the shared root handlers instead of attaching our own.
    logger.propagate = True
    return logger
