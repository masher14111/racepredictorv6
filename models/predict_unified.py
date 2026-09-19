"""Unified multi-line enrichment for ``data/predictions.json``.

Pure wiring on top of the CatBoost prediction path in :mod:`models.predictor`.
For every upcoming runner already scored by the CatBoost line, this module
attaches, **additively**:

* ``catboost_win_prob`` — the existing within-race-normalised CatBoost win prob
  (``won_prob_normalized``), surfaced under a line-named key so the three lines
  can be compared side by side.
* ``lgbm_win_prob``     — the LightGBM v3 grouped-softmax win line, within-race
  normalised (omitted entirely when the model file is absent / unscorable).
* ``market_prob``       — the de-vigged pre-off market probability (best board
  price else the fused market odds), via :func:`models.devig.devig`.
* ``ev_catboost`` / ``ev_lgbm`` — expected value ``prob * price - 1`` at the
  executable board price (best board price else fused odds).

and a top-level ``verdict`` block: the latest holdout GO / NO-GO summary for each
model line (read from the line's meta json plus the most recent
``data/backtests/holdout_*`` summary), so the UI can show how honest each line's
edge really is.

No new modelling. LightGBM is imported lazily and every step degrades
gracefully — a missing model, missing features, or an unscored race simply omits
the affected key rather than failing the prediction run. The whole entry point
(:func:`enrich`) is wrapped so it can never break the CatBoost cache write.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model artefacts (relative to the model dir).
_LGBM_MODEL_FILENAME = "lgbm_won_v3.txt"
_LGBM_META_FILENAME = "lgbm_v3_meta.json"

# De-vig method for the market line. Proportional matches the holdout head-to-head
# benchmark (see data/backtests/holdout_lgbm_*/summary.json -> devig_method).
_DEVIG_METHOD = "proportional"

# Cache loaded LightGBM boosters by (path, mtime) so a polling loop does not
# reload the model from disk on every refresh.
_LGBM_CACHE: dict[str, tuple[float, object]] = {}


# ── group-id flattening ───────────────────────────────────────────────────────

def _flat_keys(group_ids: Sequence) -> np.ndarray:
    """Flatten per-runner race ids to 1-D hashable string keys.

    ``group_ids`` may be a list of ``(venue, race_time)`` tuples; ``np.asarray``
    on those yields a 2-D array that breaks both ``np.unique`` grouping (softmax)
    and the ``race_arr == rid`` mask (devig). Joining each id to a single string
    keeps every grouping 1-D and order-preserving.
    """
    return np.array(
        [
            "\x1f".join(map(str, k)) if isinstance(k, (tuple, list)) else str(k)
            for k in group_ids
        ],
        dtype=object,
    )


# ── LightGBM win line ─────────────────────────────────────────────────────────

def _load_lgbm(model_path: Path):
    """Load (and cache) the LightGBM softmax model, or None when unavailable."""
    if not model_path.exists():
        return None
    key = str(model_path)
    try:
        mtime = model_path.stat().st_mtime
    except OSError:
        mtime = -1.0
    cached = _LGBM_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        from models.lgbm_softmax import LGBMSoftmaxModel  # lazy: heavy import
    except Exception as exc:  # noqa: BLE001 — lightgbm absent / venv-gated
        logger.warning("predict_unified: lightgbm unavailable (%s) — lgbm line skipped", exc)
        return None
    model = LGBMSoftmaxModel.load(model_path)
    _LGBM_CACHE[key] = (mtime, model)
    return model


def score_lgbm_win_prob(
    scored: pd.DataFrame,
    flat_keys: np.ndarray,
    model_dir: Path,
) -> Optional[np.ndarray]:
    """Within-race-normalised LightGBM win probabilities aligned to ``scored`` rows.

    Returns ``None`` (graceful no-op) when the model file is missing, lightgbm is
    unavailable, or scoring raises for any reason. Uses
    :func:`features.lgbm_adapter.build_lgbm_matrix` so the served features match
    the leak-free, pre-off matrix the model was trained on (train/serve parity).
    """
    model_path = model_dir / _LGBM_MODEL_FILENAME
    model = _load_lgbm(model_path)
    if model is None:
        return None
    try:
        from features.lgbm_adapter import build_lgbm_matrix
        from models.lgbm_softmax import softmax_by_race

        X, _, _ = build_lgbm_matrix(scored, inference=True)
        if len(X) != len(scored):
            logger.warning(
                "predict_unified: lgbm matrix row mismatch (%d vs %d) — lgbm skipped",
                len(X), len(scored),
            )
            return None
        raw = model.predict_raw(X)
        return softmax_by_race(np.asarray(raw, dtype=float), flat_keys)
    except Exception as exc:  # noqa: BLE001 — never break the CatBoost line
        logger.warning("predict_unified: lgbm scoring failed (%s) — lgbm line omitted", exc)
        return None


# ── De-vigged market line ─────────────────────────────────────────────────────

def market_prob(decimals, flat_keys: np.ndarray) -> np.ndarray:
    """De-vigged pre-off market probability per runner (NaN for partial books)."""
    from models.devig import devig

    odds = pd.Series(pd.to_numeric(pd.Series(decimals), errors="coerce").to_numpy())
    try:
        return devig(odds, pd.Series(flat_keys), method=_DEVIG_METHOD)
    except Exception as exc:  # noqa: BLE001
        logger.warning("predict_unified: devig failed (%s) — market_prob NaN", exc)
        return np.full(len(odds), np.nan)


# ── Public enrichment entry point ─────────────────────────────────────────────

def enrich(
    df: pd.DataFrame,
    group_ids: Sequence,
    reference_decimals=None,
    executable_decimals=None,
    model_dir: Path = None,
    effective_decimals=None,
) -> None:
    """Attach the unified multi-line columns to ``df`` in place (best-effort).

    Two prices are threaded through, kept strictly separate (audit req 5/6):

    * ``reference_decimals`` — the consensus / fused single-line price. This is the
      ONLY price de-vigged into ``market_prob``. De-vigging a "best price across
      every book" overlay would treat a synthetic book as one bookmaker's quote and
      understate the overround, corrupting the fair-probability line.
    * ``executable_decimals`` — the best board price we would actually back. EV
      (``ev_catboost`` / ``ev_lgbm``) is computed against this, since EV must
      reflect the return actually obtainable.

    ``effective_decimals`` is a deprecated back-compat alias: when the split pair is
    not supplied it seeds both prices (preserving the historical single-price
    behaviour). Columns added: ``catboost_win_prob``, ``market_prob``,
    ``ev_catboost`` always; ``lgbm_win_prob`` and ``ev_lgbm`` only when the LightGBM
    line scored. Any failure is logged and swallowed so it can never interfere with
    the CatBoost prediction cache.
    """
    try:
        if df is None or df.empty:
            return
        exec_src = executable_decimals if executable_decimals is not None else effective_decimals
        ref_src = reference_decimals if reference_decimals is not None else exec_src
        if exec_src is None:
            logger.warning("predict_unified: no executable prices supplied — enrichment skipped")
            return
        exec_dec = pd.to_numeric(pd.Series(exec_src).reset_index(drop=True), errors="coerce")
        ref_dec = pd.to_numeric(pd.Series(ref_src).reset_index(drop=True), errors="coerce")
        flat = _flat_keys(group_ids)

        # EV inputs are quantised to the exact precision they are persisted at
        # (prob 4dp, price 3dp — see models.predictor._prob/_clean_odds) BEFORE
        # the multiply, so every cached EV is exactly reproducible from its
        # cached inputs: ev == round(prob * price - 1, 4). (Audit req 8/10.)
        exec_q = np.round(exec_dec.to_numpy(dtype=float), 3)

        # CatBoost line: alias the existing within-race-normalised headline so the
        # three lines live under parallel, self-describing keys.
        cb = pd.to_numeric(df.get("won_prob_normalized"), errors="coerce") \
            if "won_prob_normalized" in df.columns else pd.Series(np.nan, index=df.index)
        df["catboost_win_prob"] = cb.to_numpy()
        # EV against the EXECUTABLE price.
        df["ev_catboost"] = np.round(cb.to_numpy(dtype=float), 4) * exec_q - 1.0

        # De-vigged market line — from the REFERENCE book only (one complete
        # bookmaker board, else the fused consensus — never the best-price overlay).
        df["market_prob"] = market_prob(ref_dec, flat)

        # LightGBM line — omitted entirely when unavailable.
        lgbm = score_lgbm_win_prob(df, flat, Path(model_dir))
        if lgbm is not None:
            lgbm_q = np.round(np.asarray(lgbm, dtype=float), 4)
            df["lgbm_win_prob"] = lgbm_q
            df["ev_lgbm"] = lgbm_q * exec_q - 1.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("predict_unified: enrichment skipped (%s)", exc)


# ── Verdict block ─────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 — absent/corrupt verdict is a normal state
        return None


def _latest_holdout_summary(base: Path, line: str) -> tuple[Optional[dict], Optional[str]]:
    """Most recent ``data/backtests/holdout_<line>_*/summary.json`` + its dir name.

    Directories are timestamp-suffixed (``holdout_lgbm_YYYYMMDD_HHMMSS``), so the
    lexicographically greatest name is the newest run.
    """
    bt = base / "data" / "backtests"
    if not bt.is_dir():
        return None, None
    dirs = sorted(
        (p for p in bt.glob(f"holdout_{line}_*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for d in dirs:
        summary = _read_json(d / "summary.json")
        if summary is not None:
            return summary, d.name
    return None, None


def _lgbm_verdict(base: Path, model_dir: Path) -> Optional[dict]:
    """GO / NO-GO verdict for the LightGBM line from its meta + latest holdout."""
    meta = _read_json(model_dir / _LGBM_META_FILENAME) or {}
    verdict = dict(meta.get("verdict") or {})
    summary, source = _latest_holdout_summary(base, "lgbm")

    if not verdict and summary is None:
        return None

    # Backfill the headline metrics from the holdout summary if the meta verdict
    # is absent/partial, so the block is populated either way.
    if summary is not None:
        h2h = summary.get("head_to_head") or {}
        betting = summary.get("betting") or {}
        model_m = h2h.get("model") or {}
        market_m = h2h.get("market") or {}
        verdict.setdefault("model_beats_market_logloss",
                           summary.get("model_beats_market_logloss"))
        verdict.setdefault("go", verdict.get("model_beats_market_logloss"))
        verdict.setdefault("model_log_loss", model_m.get("log_loss"))
        verdict.setdefault("market_log_loss", market_m.get("log_loss"))
        verdict.setdefault("n_races_kept", h2h.get("n_races_kept"))
        verdict.setdefault("ev_roi", betting.get("roi"))
        verdict.setdefault("mean_clv_log", betting.get("mean_clv_log"))
        verdict.setdefault("clv_beat_rate", betting.get("clv_beat_rate"))
        verdict["holdout_window"] = summary.get("window")
        verdict["model_version"] = (summary.get("model") or {}).get("model_version") \
            or meta.get("model_version")
        verdict["source"] = source

    verdict.setdefault("model_version", meta.get("model_version"))
    return verdict


def load_verdicts(base: Path, model_dir: Path) -> dict:
    """Latest GO / NO-GO verdict per model line for the predictions payload.

    ``lgbm`` carries the real holdout verdict; ``catboost`` has no holdout
    programme on disk yet, so it is marked unavailable rather than faked.
    """
    out: dict = {}
    lgbm = _lgbm_verdict(Path(base), Path(model_dir))
    out["lgbm"] = lgbm if lgbm is not None else {
        "available": False,
        "note": "no LightGBM holdout verdict on disk",
    }
    # CatBoost: no holdout GO/NO-GO summary is produced for the priced line yet.
    cb_summary, cb_source = _latest_holdout_summary(Path(base), "catboost")
    if cb_summary is not None:
        h2h = cb_summary.get("head_to_head") or {}
        out["catboost"] = {
            "go": cb_summary.get("model_beats_market_logloss"),
            "model_beats_market_logloss": cb_summary.get("model_beats_market_logloss"),
            "holdout_window": cb_summary.get("window"),
            "source": cb_source,
        }
    else:
        out["catboost"] = {
            "available": False,
            "note": "no CatBoost holdout verdict on disk",
        }
    return out


def unified_runner_keys(row: pd.Series) -> dict:
    """The additive unified keys for one runner row, ready to merge into its dict.

    ``catboost_win_prob`` / ``market_prob`` / ``ev_catboost`` are always present
    (value may be ``None``); ``lgbm_win_prob`` / ``ev_lgbm`` appear only when the
    LightGBM line scored this runner — graceful omission, never null padding.
    """
    out: dict = {
        "catboost_win_prob": _prob(row.get("catboost_win_prob")),
        "market_prob": _prob(row.get("market_prob")),
        "ev_catboost": _prob(row.get("ev_catboost")),
    }
    lgbm = _prob(row.get("lgbm_win_prob"))
    if lgbm is not None:
        out["lgbm_win_prob"] = lgbm
        out["ev_lgbm"] = _prob(row.get("ev_lgbm"))
    return out


def _prob(val) -> Optional[float]:
    """Round to 4dp, or None when missing/non-finite."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None
