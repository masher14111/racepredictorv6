"""Backend operations for the Scrape & Refresh page.

Thin, Streamlit-free wrappers around the live-odds scrapers and the
normalize -> build -> predict pipeline (the same steps as scripts/refresh.py),
so the UI buttons and a unit test can drive them without a running server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# module suffix -> display label. One source of truth so the page and tests agree
# on which books exist and how they're labelled.
SCRAPERS: dict[str, str] = {
    "livescorebet": "LiveScore Bet",
    "paddy_power": "Paddy Power",
    "boylesports": "BoyleSports",
}


def scrape_one(key: str, force: bool = True) -> dict:
    """Run a single book's scraper. Returns {ok, rows, races, summary, error,
    stale, age_seconds}.

    livescorebet/boylesports return a DataFrame of runner rows; paddy_power
    returns a ``{"races": [...]}`` dict — both are normalised to row/race counts
    so the UI can report a uniform summary. Any failure is caught and surfaced in
    ``error`` rather than raised (a button shouldn't crash the page).

    ``stale``/``age_seconds`` surface a degraded stale-cache-fallback scrape (only
    boylesports.py can produce one; see req 8) so the UI never presents cache data
    as if it were a fresh live fetch. ``age_seconds`` is the oldest ``fetched_at``
    across the returned rows."""
    from importlib import import_module

    result: dict = {
        "ok": False, "rows": 0, "races": 0, "summary": "", "error": None,
        "stale": False, "age_seconds": None,
    }
    try:
        mod = import_module(f"scraper.{key}")
        out = mod.scrape(force=force)
        if isinstance(out, dict):  # paddy_power cache-schema dict
            races = out.get("races") or []
            rows = sum(
                len(r.get("runners") or r.get("selections") or []) for r in races
            )
            result.update(
                rows=rows, races=len(races),
                summary=f"{len(races)} races, {rows} runners",
            )
        else:  # DataFrame (livescorebet / boylesports)
            rows = len(out)
            cols = getattr(out, "columns", [])
            venues = int(out["venue"].nunique()) if "venue" in cols else 0
            if rows and "stale" in cols:
                result["stale"] = bool(out["stale"].fillna(False).astype(bool).any())
            if rows and "fetched_at" in cols:
                import pandas as pd
                fetched = pd.to_datetime(out["fetched_at"], utc=True, errors="coerce")
                ages = (pd.Timestamp.now(tz="UTC") - fetched).dt.total_seconds()
                if ages.notna().any():
                    result["age_seconds"] = float(ages.max())
            result.update(
                rows=rows, races=venues,
                summary=f"{rows} rows across {venues} venues",
            )
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 — surface to the UI, never crash the page
        from utils.proxy_manager import redact_secrets
        result["error"] = redact_secrets(str(exc))
    return result


def rebuild_predictions(progress: Callable[[str], None] = lambda _m: None) -> dict:
    """normalize -> build -> predict (no scrape), writing data/predictions.json.

    Mirrors steps 2-4 of scripts.refresh. ``progress`` is called with a status
    string before each step so the UI can show where it is. Returns
    {ok, unified_rows, live_runners, races, error}."""
    out: dict = {
        "ok": False, "unified_rows": 0, "live_runners": 0, "races": 0, "error": None,
    }
    try:
        progress("Normalizing sources into unified_races.parquet…")
        from utils.normalizer import normalize
        unified = normalize(write=True)
        out["unified_rows"] = len(unified)

        progress("Building today's live inference matrix…")
        from features.builder import build_inference_matrix
        live = build_inference_matrix(unified=unified)
        out["live_runners"] = len(live)

        progress("Scoring and writing predictions.json…")
        from models.predictor import Predictor
        predictor = Predictor()
        if not predictor.load():
            out["error"] = "No trained models found — run `python -m models.train` first."
            return out
        # Reuse the matrix we just built — predict() would otherwise re-derive it,
        # doubling the slowest step of the rebuild.
        races = predictor.predict(unified=unified, live=live)
        out["races"] = len(races)
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 — surface to the UI
        from utils.proxy_manager import redact_secrets
        out["error"] = redact_secrets(str(exc))
    return out
