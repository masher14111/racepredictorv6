"""Public API: build the training and inference feature matrices.

Both builders run the same fuse -> derive pipeline so train/serve features match.
"""
import os

import pandas as pd

from features import derive, engine
from features._race_reconcile import reconcile_live_race_times
from features._trainer_form import add_trainer_form
from features._jockey_form import add_jockey_form
from features._market_movement import add_market_movement
from features._class_par import add_class_par
from features._interaction import add_interactions
from features._freshness import add_freshness_and_market
from features.fuse import fuse_sources
from features.labels import add_labels
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.timezone import now

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_UNIFIED = os.path.join(_BASE, "data", "unified_races.parquet")
_DEFAULT_TRAINING = os.path.join(_BASE, "data", "features", "training.parquet")
_DEFAULT_FEATURES = os.path.join(_BASE, "data", "features.parquet")


def _load_cfg() -> dict:
    tf = get_config().get("timeform", {})
    return {
        "lookback_months": tf.get("lookback_months", 12),
        "lookback_runs": tf.get("lookback_runs", 20),
        "place_positions": tf.get("place_positions", 3),
        "going_speed_map": tf.get("going_speed_map", {}),
    }


def _load_unified(unified) -> pd.DataFrame:
    if unified is not None:
        return unified
    if not os.path.exists(_DEFAULT_UNIFIED):
        raise FileNotFoundError(
            f"unified dataset not found at {_DEFAULT_UNIFIED}; run the normalizer first")
    return pd.read_parquet(_DEFAULT_UNIFIED)


def _derive_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Shared feature pipeline: fuse then derive every feature. Train/serve identical."""
    out = fuse_sources(df)
    # Historical rows (training, and a live row's own prior runs) are real,
    # already-happened results — their connections are genuinely known, never a
    # last-known guess — so default/backfill to "declared" here rather than in
    # _fill_last_known_connections (which only ever sees TODAY's live rows).
    # This also keeps train/serve column parity (see build_inference_matrix).
    for _role in _CONNECTION_ROLES:
        _col = f"{_role}_connections_source"
        if _col not in out.columns:
            out[_col] = pd.NA
        out[_col] = out[_col].fillna("declared")
    out["race_date"] = pd.to_datetime(out["race_date"], utc=True, errors="coerce")
    out = derive.add_race_key(out)            # per-race id (fixes field_size/ranks)
    out = derive.add_going_speed(out, cfg["going_speed_map"])
    # Step 06: corrected-coverage companion (same frozen numbers, fixed string
    # matching) — additive, not yet in models.features.FEATURE_COLS; see
    # features/derive.py's block comment above add_going_speed_v2.
    out = derive.add_going_speed_v2(out, cfg["going_speed_map"])
    out = derive.add_going_band(out)
    out = derive.add_class_change(out)
    out = derive.add_distance_furlongs(out)
    out = derive.add_recent_form(out)
    out = derive.add_odds_features(out)
    out = derive.add_rating_rank(out)
    out = derive.add_all_trailing_rates(out, cfg)
    out = derive.add_days_since_last_run(out)
    out = derive.add_career_runs(out)
    # higher-order engine features (build on the derived columns above)
    out = engine.add_speed_figures(out, cfg)
    out = engine.add_combo_win_rate(out, cfg)
    out = engine.add_going_preference(out, cfg)
    out = engine.add_course_suitability(out, cfg)
    out = engine.add_distance_suitability(out, cfg)
    out = engine.add_pace_bias(out)
    out = engine.add_ew_value_index(out)
    out = engine.add_odds_delta(out)
    out = engine.add_race_complexity(out)
    # Step 04: corrected, additive replacements — race_complexity above is left
    # unchanged (bound to existing model bundles' feature schema); these two are
    # candidate features for a future retrain (step 10+), not yet in
    # models.features.FEATURE_COLS. See features/engine.py's block comment.
    out = engine.add_race_complexity_v2(out)
    out = engine.add_race_market_entropy(out)
    # new predictive signals (post model-10)
    out = add_trainer_form(out)
    out = add_jockey_form(out)
    out = add_market_movement(out)
    out = add_class_par(out)
    out = add_interactions(out)
    out = add_freshness_and_market(out)
    return out.reset_index(drop=True)


def build_training_matrix(unified=None, write=True, output_path=None,
                          features_path=None) -> pd.DataFrame:
    """Labelled historical matrix: fuse -> derive -> label -> drop null-position rows.

    Also writes the FULL derived matrix (labelled history + live) to
    data/features.parquet when write=True."""
    cfg = _load_cfg()
    full = _derive_all(_load_unified(unified), cfg)
    full = add_labels(full, cfg["place_positions"])
    if write:
        fpath = features_path or _DEFAULT_FEATURES
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        full.to_parquet(fpath, engine="pyarrow", index=False)
        logger.info("features: wrote %d full-matrix rows to %s", len(full), fpath)
    df = full[full["position"].notna()].reset_index(drop=True)
    if write:
        path = output_path or _DEFAULT_TRAINING
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)
        logger.info("features: wrote %d training rows to %s", len(df), path)
    return df


def _attach_true_run_speed(frame: pd.DataFrame, raw: pd.DataFrame,
                           in_window: pd.Series) -> pd.DataFrame:
    """Precompute each historical row's finish percentile (``_run_speed``) using its
    race's TRUE field size, taken over the full in-window data before history is
    pruned to live entities.

    The horse-speed proxy is the trailing mean of prior ``_run_speed``; that
    percentile's denominator is the race's field size. Pruning history to only the
    live horses' own rows shrinks each historical race to ~1 runner, so a field_size
    recomputed over the pruned frame is wrong and the proxy degenerates (negative /
    nonsense). Computing it here, over the unpruned field, keeps live horse_speed on
    the same scale the model trained on. Live rows (no position) stay NaN."""
    cols = ["race_date", "venue", "horse_id", "position"]
    if not set(cols).issubset(raw.columns):
        return frame
    # Use the recovered per-race key (venue + off-time) so field_size is the true
    # ~9-runner race, not the ~72-runner venue-day. Distinct horses per race ==
    # fused field size; drop the WIN/PLACE market duplication (same position).
    keyed = derive.add_race_key(raw.loc[in_window, cols + (
        ["race_time"] if "race_time" in raw.columns else [])])
    inw = keyed.drop_duplicates(["race_uid", "horse_id"])
    if inw.empty:
        return frame
    fs = inw.groupby("race_uid", sort=False)["horse_id"].transform("size")
    rs = engine.finish_percentile(inw["position"], fs)
    rs_map = {(rd, v, h): val for rd, v, h, val
              in zip(inw["race_date"], inw["venue"], inw["horse_id"], rs)
              if pd.notna(val)}
    out = frame.copy()
    out["_run_speed"] = [rs_map.get((rd, v, h), float("nan")) for rd, v, h
                         in zip(out["race_date"], out["venue"], out["horse_id"])]
    return out


_CONNECTION_ROLES = ("jockey", "trainer")


def _fill_last_known_connections(live_rows: pd.DataFrame,
                                 prior: pd.DataFrame) -> pd.DataFrame:
    """Carry each live horse's most-recent known jockey/trainer onto its live row,
    ONLY when today's row does not already carry one — and record which case
    applied, per role, in a new ``{role}_connections_source`` column:
    "declared" (this row's source already reported it — e.g. today's Timeform
    card fused ahead of the odds-only rows via features/fuse.py's source
    priority), "historical_fallback" (invented from the horse's own past runs
    below) or "missing" (neither available).

    Live odds feeds emit only horse + price — no jockey/trainer — so the
    jockey/trainer/jt-combo trailing-rate features (a quarter of the price-free
    model's signal) are null for every live runner absent a racecard source. A
    horse's connections are stable, so its last-known jockey_id/trainer_id is a
    leak-free prior (strictly past runs) that revives those features for the
    runners we DO have history for — but it is a GUESS, not today's actual
    declaration, and callers/UI must not present it as one (Stage 5)."""
    out = live_rows.copy()
    cols = ["jockey_id", "jockey_name", "trainer_id", "trainer_name"]
    have = [c for c in cols if c in prior.columns and c in out.columns]
    last = None
    if have and not prior.empty:
        # groupby.last() takes the most recent NON-NULL value per column per horse.
        last = prior.sort_values("race_date").groupby("horse_id")[have].last()

    for role in _CONNECTION_ROLES:
        id_col, name_col = f"{role}_id", f"{role}_name"
        declared = out[id_col].notna() if id_col in out.columns else pd.Series(
            False, index=out.index)
        source = pd.Series("missing", index=out.index, dtype="object")
        source[declared] = "declared"
        for c in (id_col, name_col):
            if c not in out.columns or last is None or c not in last.columns:
                continue
            fill = out["horse_id"].map(last[c])
            filled = out[c].isna() & fill.notna()
            out[c] = out[c].where(~filled, fill)
            if c == id_col:
                source[filled] = "historical_fallback"
        out[f"{role}_connections_source"] = source
    return out


_IDENTITY_QUARANTINE_PATH = os.path.join(_BASE, "data", "execution", "identity_quarantine.json")


def _record_identity_quarantine(quarantined: pd.DataFrame) -> None:
    """Best-effort, overwritten-every-cycle record of rows excluded this run
    because a close off-time pairing at one venue could not be confirmed as
    the same physical race (Stage 20 / B5). Never blocks inference on a write
    failure — this is operational visibility, not a gate."""
    try:
        import json as _json
        from datetime import timezone as _timezone

        from utils.timezone import now as _now

        cols = [c for c in ("venue", "race_time", "horse_name", "source", "_quarantine_reason")
                if c in quarantined.columns]
        payload = {
            "recorded_at": _now().astimezone(_timezone.utc).isoformat(),
            "n_rows": int(len(quarantined)),
            "rows": quarantined[cols].astype(str).to_dict("records"),
        }
        os.makedirs(os.path.dirname(_IDENTITY_QUARANTINE_PATH), exist_ok=True)
        tmp = _IDENTITY_QUARANTINE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2)
        os.replace(tmp, _IDENTITY_QUARANTINE_PATH)
        logger.warning(
            "features: quarantined %d live row(s) with an ambiguous close "
            "off-time — see %s", len(quarantined), _IDENTITY_QUARANTINE_PATH,
        )
    except OSError as exc:
        logger.warning("features: could not write identity quarantine record: %s", exc)


def _live_mask(race_date: pd.Series, position: pd.Series) -> pd.Series:
    """A row is a live (inference) runner iff it has no finishing position AND its
    race has not happened yet (race_date >= start of today, Europe/Dublin).

    The date guard is essential: after a historical backfill the unified dataset is
    dominated by past rows whose results failed to join (null position). Those are
    NOT upcoming races — keying inference off `position.isna()` alone would score
    tens of thousands of historical join-misses as phantom 'live' races and derive
    features over the entire history on every call."""
    today = pd.Timestamp(now().date(), tz="UTC")
    rd = pd.to_datetime(race_date, utc=True, errors="coerce")
    return position.isna() & (rd >= today)


def build_inference_matrix(unified=None) -> pd.DataFrame:
    """Unlabelled live matrix: derive over today's racecards plus the history those
    runners need, and return only the live runners.

    Identical feature columns to the training matrix (no won/placed). Only upcoming
    races (race_date >= today, no result) count as live — see _live_mask. The derive
    input is pruned to the live rows + the in-window prior runs of the live horses/
    jockeys/trainers, so inference stays fast regardless of total history size."""
    cfg = _load_cfg()
    raw = _load_unified(unified)
    if "position" not in raw.columns or "race_date" not in raw.columns:
        return _derive_all(raw, cfg).iloc[0:0].reset_index(drop=True)

    live_mask = _live_mask(raw["race_date"], raw["position"])
    if not live_mask.any():
        logger.info("features: no upcoming races found — inference matrix is empty")
        # Derive over an empty frame to return the correct (zero-row) column schema.
        return _derive_all(raw.iloc[0:0], cfg).reset_index(drop=True)

    rd = pd.to_datetime(raw["race_date"], utc=True, errors="coerce")
    today = pd.Timestamp(now().date(), tz="UTC")
    lower = today - pd.DateOffset(months=cfg["lookback_months"])
    # Today's ALREADY-RUN races are admissible history for a later race on the
    # same card, and the training matrix counts them (features/_trailing_fast
    # cuts priors on the real off-time, step 09 audit F1) — so excluding them
    # here would make the serving path compute the same feature from a smaller
    # history than training did. A race that has not run yet still carries no
    # position and cannot enter: the `position.notna()` guard is what makes this
    # point-in-time, and the off-time cut stops an earlier race being scored on
    # a later one.
    in_window = (raw["position"].notna() & (rd >= lower)
                 & (rd < today + pd.Timedelta(days=1)))

    # Live odds feeds carry no jockey/trainer, so carry each live horse's last-known
    # connections forward from its own history first — this both revives the
    # jockey/trainer features AND lets those entities' history survive the prune below.
    live_rows = _fill_last_known_connections(raw[live_mask], raw[in_window])

    # Stage 20 (B5/GAP-C): two odds sources scraping the SAME physical race
    # sometimes disagree on the off-time by a minute; left unreconciled,
    # fuse_sources's exact-race_time grouping turns that into two DIFFERENT
    # race_uid values downstream, and the daily loop issues two ticket sets for
    # the same runners. Reconciliation runs ONLY on today's live rows — never
    # on build_training_matrix's frozen historical pipeline (D42) — so it can
    # only affect today's predictions, not any evaluated window. Ambiguous
    # close-off-time pairs (no confirmed shared runner) are quarantined, not
    # guessed either way, and recorded for operational visibility.
    live_rows, _quarantined = reconcile_live_race_times(live_rows)
    if not _quarantined.empty:
        _record_identity_quarantine(_quarantined)

    # History the live runners need: prior known-result runs of any live horse/
    # jockey/trainer, within the trailing lookback window. Trailing rates for a live
    # row depend only on that row's own entities' priors, so pruning history to these
    # entities leaves them unchanged. (The horse-speed proxy is the exception — its
    # percentile needs each prior race's true field size — so we precompute it below.)
    ids = {}
    for col in ("horse_id", "jockey_id", "trainer_id"):
        ids[col] = set(live_rows[col].dropna()) if col in live_rows.columns else set()
    relevant = pd.Series(False, index=raw.index)
    for col, vals in ids.items():
        if vals:
            relevant |= raw[col].isin(vals)
    hist_mask = in_window & relevant

    frame = pd.concat([raw[hist_mask], live_rows]).reset_index(drop=True)
    frame = _attach_true_run_speed(frame, raw, in_window)
    out = _derive_all(frame, cfg)
    live = out[_live_mask(out["race_date"], out["position"])].reset_index(drop=True)
    return live
