"""Forward-validation window freeze/reset (Stage 6, requirement 4).

Eight weeks of paper evidence is only honest if it was collected against one
unchanged model and one unchanged rule set. This module hashes the frozen
model artifacts, the selection lock, and the ``execution:`` config block, and
resets the accumulated window to zero the instant any of them change — a
retrain or a threshold edit invalidates prior evidence mechanically, rather
than relying on whoever made the change to remember to say so.

Starting the window is a **separate, deliberate action** from verifying it
(:func:`start_window` vs :func:`verify_window`). The daily loop calls
``verify_window`` every run; nothing in this stage calls ``start_window``
automatically. Capture begins immediately; the formal clock does not, until a
human explicitly starts it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WINDOW_PATH = os.path.join("data", "execution", "forward_window.json")
DEFAULT_SELECTION_LOCK_PATH = os.path.join("data", "execution", "selection_lock.json")
DEFAULT_MODELS_DIR = "models"

# The frozen artifacts that define "the model" for forward-validation purposes.
# Retraining any target/tag regenerates these files, which is exactly the event
# that must reset the window. Listed explicitly rather than globbed so an
# unrelated new file in models/ cannot silently join (or fail to join) the set.
FROZEN_MODEL_FILES: tuple[str, ...] = (
    "catboost_won_v3.bin",
    "catboost_won_v3_calib.pkl",
    "catboost_won_v3nf.bin",
    "catboost_won_v3nf_calib.pkl",
    "catboost_placed_2_v3.bin",
    "catboost_placed_2_v3_calib.pkl",
    "catboost_placed_2_v3nf.bin",
    "catboost_placed_2_v3nf_calib.pkl",
    "catboost_showed_v3.bin",
    "catboost_showed_v3_calib.pkl",
    "catboost_showed_v3nf.bin",
    "catboost_showed_v3nf_calib.pkl",
    "catboost_v3_meta.json",
    "catboost_v3nf_meta.json",
    "fl_oddsband_v3nf_calib.pkl",
    "fl_oddsband_v3nf_calib_meta.json",
    "lgbm_won_v3.txt",
    "lgbm_won_v3.meta.json",
    "lgbm_v3_meta.json",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_state_hashes(
    *,
    models_dir: str = DEFAULT_MODELS_DIR,
    selection_lock_path: str = DEFAULT_SELECTION_LOCK_PATH,
    raw_config: Optional[dict] = None,
) -> dict:
    """Hash every input whose change must reset the forward window.

    A missing model file hashes to ``None`` rather than raising — a partially
    trained model directory is itself a state worth detecting a change against,
    not a reason to crash the daily loop.
    """
    model_hashes = {
        name: _sha256_file(os.path.join(models_dir, name)) for name in FROZEN_MODEL_FILES
    }
    selection_hash = _sha256_file(selection_lock_path)
    raw = raw_config if raw_config is not None else get_config()
    execution_block = (raw or {}).get("execution") or {}
    config_hash = _sha256_json(execution_block)
    return {
        "model_files": model_hashes,
        "selection_lock": selection_hash,
        "execution_config": config_hash,
    }


@dataclass(frozen=True)
class WindowState:
    """Persisted forward-validation window state."""

    status: str = "not_started"  # "not_started" | "running" | "reset"
    window_start: Optional[str] = None  # ISO date the current (post-reset) window began
    hashes: dict = field(default_factory=dict)
    history: tuple[dict, ...] = ()

    @property
    def running(self) -> bool:
        return self.status in ("running", "reset")

    def days_elapsed(self, *, now: Optional[datetime] = None) -> int:
        if not self.window_start:
            return 0
        now = now or _now()
        start = date.fromisoformat(self.window_start)
        return max(0, (now.date() - start).days)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "window_start": self.window_start,
            "hashes": self.hashes,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "WindowState":
        return cls(
            status=str(raw.get("status") or "not_started"),
            window_start=raw.get("window_start"),
            hashes=raw.get("hashes") or {},
            history=tuple(raw.get("history") or ()),
        )


def load_window(path: str = DEFAULT_WINDOW_PATH) -> WindowState:
    """Read the window file. Missing or unreadable -> a fresh not-started state."""
    if not os.path.exists(path):
        return WindowState()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return WindowState.from_dict(json.load(fh))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("window: %s unreadable (%s); treating as not started", path, exc)
        return WindowState()


def _write(path: str, state: WindowState) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _diff_reasons(old_hashes: dict, new_hashes: dict) -> list[str]:
    reasons: list[str] = []
    old_models = (old_hashes or {}).get("model_files") or {}
    new_models = (new_hashes or {}).get("model_files") or {}
    for name in sorted(set(old_models) | set(new_models)):
        if old_models.get(name) != new_models.get(name):
            reasons.append(f"model_file_changed:{name}")
    if (old_hashes or {}).get("selection_lock") != (new_hashes or {}).get("selection_lock"):
        reasons.append("selection_lock_changed")
    if (old_hashes or {}).get("execution_config") != (new_hashes or {}).get("execution_config"):
        reasons.append("execution_config_changed")
    return reasons


def start_window(
    path: str = DEFAULT_WINDOW_PATH, *, hashes: dict, now: Optional[datetime] = None
) -> WindowState:
    """Deliberately begin the formal forward-validation window.

    A no-op if the window is already running or reset — that state can only be
    changed by :func:`verify_window` detecting an input change, never by
    calling this again. This keeps "start" a single, explicit, named action.
    """
    now = now or _now()
    existing = load_window(path)
    if existing.running:
        logger.info("window: already %s since %s; start_window is a no-op", existing.status,
                    existing.window_start)
        return existing
    state = WindowState(
        status="running",
        window_start=now.date().isoformat(),
        hashes=hashes,
        history=existing.history + (
            {"at": now.isoformat(), "event": "started", "reason": None},
        ),
    )
    _write(path, state)
    logger.info("window: started %s", state.window_start)
    return state


def verify_window(
    path: str = DEFAULT_WINDOW_PATH, *, hashes: dict, now: Optional[datetime] = None
) -> tuple[WindowState, list[str]]:
    """Re-verify the frozen hashes every run. Reset to zero on any change.

    Returns ``(state, changes)``. ``changes`` is empty unless a reset just
    happened, in which case it names exactly what changed so the report and
    dashboard can say so prominently rather than silently zeroing the clock.
    """
    now = now or _now()
    existing = load_window(path)
    if existing.status == "not_started":
        # Nothing running yet to invalidate; verification is a no-op until the
        # window is explicitly started.
        return existing, []

    changes = _diff_reasons(existing.hashes, hashes)
    if not changes:
        return existing, []

    state = WindowState(
        status="reset",
        window_start=now.date().isoformat(),
        hashes=hashes,
        history=existing.history + (
            {"at": now.isoformat(), "event": "reset", "reason": ", ".join(changes)},
        ),
    )
    _write(path, state)
    logger.warning("window: RESET (%s)", "; ".join(changes))
    return state, changes
