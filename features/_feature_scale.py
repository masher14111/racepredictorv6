"""Frozen (train-fitted) standardization constants for independent features.

Some independent features (e.g. race_complexity_v2's field-size component) need
a stable population reference to be a meaningful "how unusual is this race"
scalar — a per-race quantity like field_size is constant within its own race,
so it cannot be standardized against ITS OWN race alone.

The legacy `engine.add_race_complexity` standardized against the mean/std of
whatever frame happened to be passed to it (the WHOLE call), which makes an
admissible historical row's value depend on which OTHER rows — including rows
appended later — were present in that call (see DECISIONS D27, CONTRACTS.md).
This module fits that reference ONCE from admissible training data, freezes it
to a small JSON artifact, and every future call (train or serve) loads the same
frozen values instead of recomputing them live. Refitting is a deliberate,
versioned action (see engine.fit_race_complexity_v2_scale), never an implicit
side effect of a feature-build call.
"""
from __future__ import annotations

import json
import os

import pandas as pd

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SCALE_PATH = os.path.join(
    _BASE, "data", "feature_artifacts", "race_complexity_v2_scale.json")


def fit_scale(values: dict) -> dict:
    """values: {component_name: pandas Series of ONE representative row per
    real race}. Returns {component: {mean, std, n}} using only non-null
    entries of each series. A component with fewer than 2 non-null values gets
    an inert identity scale (mean 0, std 1) rather than raising, so a
    thin/degenerate fit still produces a well-formed artifact."""
    out = {}
    for name, s in values.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 2:
            out[name] = {"mean": 0.0, "std": 1.0, "n": int(len(s))}
            continue
        std = float(s.std())
        out[name] = {"mean": float(s.mean()), "std": std if std > 0 else 1.0, "n": int(len(s))}
    return out


def save_scale(scale: dict, path: str | None = None, version: str = "v1") -> None:
    path = path if path is not None else DEFAULT_SCALE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"version": version, "components": scale}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_scale(path: str | None = None) -> dict | None:
    """Returns {component: {mean, std, n}}, or None when no artifact has been
    fitted yet (callers must degrade gracefully, never fabricate a scale).

    ``path`` resolves to the module-level ``DEFAULT_SCALE_PATH`` at CALL time
    (not as a bound default parameter), so tests can monkeypatch
    ``features._feature_scale.DEFAULT_SCALE_PATH`` and have every caller that
    omits ``path`` pick up the override."""
    path = path if path is not None else DEFAULT_SCALE_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("components")


def apply_scale(series: pd.Series, component: dict) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s - component["mean"]) / component["std"]
