"""Unify scraped + historical race sources into one canonical schema.

Public API: ``normalize(sources=None, write=True) -> pd.DataFrame``.
Output: year-partitioned parquet at ``data/unified_races.parquet``.
"""
import hashlib
import json
import os
import re
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

from utils.logger import get_logger
from utils.market_validation import LIVE_SOURCES, validate_live_dataframe
from utils.storage.parquet_store import write_partitioned_parquet
from utils.text_norm import norm_horse
from utils.timezone import to_local

logger = get_logger(__name__)

CANONICAL_COLUMNS = [
    # keys / identity
    "race_id", "race_date", "race_time", "venue", "region", "horse_name", "horse_id",
    "jockey_name", "jockey_id", "trainer_name", "trainer_id",
    # source-native ids
    "selection_id", "src_horse_id", "src_jockey_id", "src_trainer_id", "src_race_id",
    # declared-card facts (free-tier racecard fields; see scraper/timeform/parser.py)
    "draw", "weight_lbs", "age", "official_rating", "equipment", "colour", "sex",
    "runner_status",
    # market / odds
    "market_id", "market_name", "market_type", "odds_decimal", "sp",
    "ew_places", "ew_reduction",
    "ew_margin", "is_low_odds", "currency", "morningwap", "ppwap", "odds_finish",
    # outcome
    "position", "win_lose",
    # ratings / form
    "timeform_rating", "pace_rating", "race_class", "going", "going_speed",
    "distance", "class_change", "recent_form", "historical_win_rate",
    "historical_place_rate", "jockey_win_rate", "runs_in_window",
    # provenance
    "source", "fetched_at", "stale", "validation_status", "validation_reasons",
    "market_field_size", "market_booksum", "year",
]

_DEDUPE_KEY = [
    "race_date", "venue", "race_time", "race_id", "horse_id",
    "market_type", "source",
]


class UnifiedRaceRow(BaseModel):
    """Validation contract for one unified runner row."""
    model_config = ConfigDict(extra="ignore")

    race_date: str
    venue: str
    horse_name: str
    horse_id: str
    market_type: Literal["WIN", "EACH_WAY", "PLACE"]
    source: str
    fetched_at: str
    race_id: Optional[str] = None
    market_id: Optional[str] = None
    market_name: Optional[str] = None
    selection_id: Optional[str] = None
    validation_status: Optional[str] = None
    odds_decimal: Optional[float] = None
    position: Optional[int] = None

    @field_validator("odds_decimal")
    @classmethod
    def _odds_gt_one(cls, v):
        if v is not None and v <= 1.0:
            raise ValueError("odds_decimal must be > 1.0")
        return v

    @field_validator("position")
    @classmethod
    def _position_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("position must be >= 1")
        return v


_ROW_VALIDATOR = TypeAdapter(UnifiedRaceRow)


def _reindex_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing canonical columns as NA and order to CANONICAL_COLUMNS."""
    out = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            # "stale" is a boolean gate (features/fuse.py, models/value.py TTL
            # checks), not free-form provenance — a source that never degrades
            # to cache (livescorebet, paddy_power) truly means "not stale",
            # so it must default to False, never an ambiguous NA.
            out[col] = False if col == "stale" else pd.NA
    if "stale" in out.columns and len(out):
        out["stale"] = out["stale"].fillna(False).astype(bool)
    if "race_date" in out.columns and len(out):
        out["race_date"] = _date_str_series(out["race_date"])
    # Native bookmaker IDs are strings while historical provider IDs are often
    # integers. A single stable string dtype prevents pyarrow from attempting to
    # coerce live IDs such as "SBTS_2_..." into int64 during merged writes.
    for col in (
        "race_id", "horse_id", "jockey_id", "trainer_id", "selection_id",
        "src_horse_id", "src_jockey_id", "src_trainer_id", "src_race_id",
        "market_id", "market_name", "validation_status", "validation_reasons",
        "equipment", "colour", "sex", "runner_status",
    ):
        out[col] = out[col].astype("string")
    return out[CANONICAL_COLUMNS]


def _coerce_col(df: pd.DataFrame, col: str, fn) -> None:
    """Apply fn to df[col] in place when the column exists (no-op otherwise)."""
    if col in df.columns:
        df[col] = df[col].map(fn)


def _date_str(value) -> str | None:
    """Coerce any date-ish value to a 'YYYY-MM-DD' string. Null on empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _date_str_series(s: pd.Series) -> pd.Series:
    """Vectorized :func:`_date_str` over a whole column → 'YYYY-MM-DD' (None on NaT).

    Element-wise ``.map(_date_str)`` calls ``pd.to_datetime`` once per row, which
    cost ~190s on the ~580k-row unified frame and ran twice per ``normalize`` (here
    and again inside ``_write``). ``format="mixed"`` parses each value with its own
    format — byte-for-byte identical to the scalar path — but vectorized (~3s).
    """
    ts = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return ts.dt.strftime("%Y-%m-%d").where(ts.notna(), None)


def _name_ids(df: pd.DataFrame, out: pd.DataFrame, role: str) -> None:
    """Derive canonical {role}_id from {role}_name when the name column exists."""
    name_col = f"{role}_name"
    if name_col in df.columns:
        out[f"{role}_id"] = df[name_col].map(_canonical_id)
    else:
        out[f"{role}_id"] = pd.NA


def _race_date_from(race_time, fallback=None) -> str | None:
    iso = _to_dublin(race_time)
    if iso:
        return iso[:10]
    return fallback


# --- per-source adapters -------------------------------------------------


# Special-market placeholder selections ("1st Favourite", "2nd Favourite", …) that
# some live odds feeds emit alongside real runners. Not horses — guarded so they
# never reach unified (and the predictor's race cards).
_PLACEHOLDER_NAME = re.compile(r"^\d+(st|nd|rd|th)\s+favourite$", re.IGNORECASE)


def _drop_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Drop placeholder selection rows (e.g. "1st Favourite") by horse_name."""
    if df is None or df.empty or "horse_name" not in df.columns:
        return df
    mask = df["horse_name"].astype("string").str.match(
        _PLACEHOLDER_NAME.pattern, case=False, na=False)
    if mask.any():
        logger.info("normalizer: dropped %d placeholder selection(s)", int(mask.sum()))
        return df[~mask].copy()
    return df


def _from_live_odds(df: pd.DataFrame) -> pd.DataFrame:
    """data/live_odds.parquet (covers boylesports + livescorebet rows)."""
    if df is None or df.empty:
        return _reindex_canonical(pd.DataFrame())
    out = _fail_closed_live(df.copy())
    if out.empty:
        return _reindex_canonical(pd.DataFrame())
    _coerce_col(out, "race_time", _to_dublin)
    _coerce_col(out, "fetched_at", _to_dublin)
    out["race_date"] = out["race_time"].map(_race_date_from)
    _coerce_col(out, "odds_decimal", _to_decimal)
    _coerce_col(out, "sp", _to_decimal)
    out["src_horse_id"] = out.get("selection_id")
    out["horse_id"] = out["horse_name"].map(_canonical_id)
    return _reindex_canonical(out)


def _from_paddy_power(cache: dict) -> pd.DataFrame:
    """Flatten the nested paddy_power.json cache (races -> markets -> selections)."""
    fetched = (cache or {}).get("fetched_at")
    rows = []
    for race in (cache or {}).get("races", []):
        for market in race.get("markets", []):
            terms = market.get("each_way_terms") or {}
            for sel in market.get("selections", []):
                rows.append({
                    "race_id": race.get("race_id"),
                    "race_time": _to_dublin(race.get("race_time")),
                    "race_date": _race_date_from(race.get("race_time")),
                    "venue": race.get("venue"),
                    "market_id": market.get("market_id"),
                    "market_name": market.get("market_name"),
                    "market_type": market.get("market_type"),
                    "ew_places": terms.get("places"),
                    "ew_reduction": terms.get("reduction"),
                    "selection_id": sel.get("selection_id"),
                    "src_horse_id": sel.get("selection_id"),
                    "horse_name": sel.get("horse_name"),
                    "horse_id": _canonical_id(sel.get("horse_name")),
                    "sp": _to_decimal(sel.get("sp")),
                    "odds_decimal": _to_decimal(sel.get("odds_decimal")),
                    "source": "paddy_power",
                    "fetched_at": _to_dublin(fetched),
                    "validation_status": sel.get(
                        "validation_status", market.get(
                            "validation_status", race.get("validation_status")
                        )
                    ),
                    "validation_reasons": sel.get(
                        "validation_reasons", market.get(
                            "validation_reasons", race.get("validation_reasons")
                        )
                    ),
                })
    out = _fail_closed_live(pd.DataFrame(rows))
    return _reindex_canonical(out)


def _from_betsp(df: pd.DataFrame) -> pd.DataFrame:
    """data/historical/betsp.parquet (Betfair SP + results enrichment)."""
    if df is None or df.empty:
        return _reindex_canonical(pd.DataFrame())
    out = df.copy()
    out["src_horse_id"] = df.get("horse_id")
    out["src_jockey_id"] = df.get("jockey_id")
    out["src_trainer_id"] = df.get("trainer_id")
    out["horse_id"] = out["horse_name"].map(_canonical_id)
    # The Betfair SP backbone stores the race OFF-TIME in race_date (e.g.
    # 2024-03-01 15:45). _reindex_canonical floors race_date to a YYYY-MM-DD
    # string, which collapses every race at a venue-day into one group and
    # corrupts field_size / market_rank / the per-race ranks (audit C3). Capture
    # the off-time into race_time FIRST so the feature pipeline can rebuild a
    # true per-race key. The off-time is published in advance → leak-free.
    if "race_time" not in df.columns or df["race_time"].isna().all():
        out["race_time"] = df["race_date"].map(_to_dublin)
    # jockey_name and trainer_name (now enriched by results sources) carry through
    # via df.copy(); derive the canonical {jockey,trainer}_id from each so trailing
    # jockey/trainer/jt-combo features — which key on those ids — are populated
    # instead of dead. Native source ids are preserved under src_*_id above.
    # When the name column is absent (pre-enrichment parquet) _name_ids sets NA.
    _name_ids(df, out, "jockey")
    _name_ids(df, out, "trainer")
    _coerce_col(out, "odds_finish", _to_decimal)
    _coerce_col(out, "fetched_at", _to_dublin)
    # C2 fix: betSP's ``sp`` column IS the Betfair Starting Price (BSP) — the
    # closing price at the off, outcome-correlated. Using it to build market
    # features (implied_prob, log_odds, odds_drift) would leak the result into
    # training. Null it so the feature pipeline falls back to ``morningwap`` /
    # ``ppwap`` (genuine pre-off prices). ``odds_finish`` stays populated as the
    # CLV reference (never fed to the model).
    # Live odds feeds write their own ``sp`` = board price (pre-off), which is
    # safe — this only targets the betSP source.
    out["sp"] = pd.NA
    out["source"] = "betsp"
    return _reindex_canonical(out)


def _from_timeform(df: pd.DataFrame) -> pd.DataFrame:
    """data/historical/timeform.parquet.

    Also carries today's live declared-card fetch (scraper.timeform_historical
    writes today's rows, position=null, into this same parquet) — those rows
    reach the live/inference frame through the same adapter, so a declared
    jockey/trainer/draw/weight/OR/equipment/runner_status this function maps
    below is a genuine current-card fact, not a historical-fallback value.
    ``features/fuse.py`` gives this source top priority, so it wins any
    conflict with a same-runner odds-source row that lacks these fields.
    """
    if df is None or df.empty:
        return _reindex_canonical(pd.DataFrame())
    out = df.copy()
    out["src_horse_id"] = df.get("horse_id")
    out["src_jockey_id"] = df.get("jockey_id")
    out["src_trainer_id"] = df.get("trainer_id")
    out["src_race_id"] = df.get("timeform_race_id")
    out["horse_id"] = out["horse_name"].map(_canonical_id)
    _name_ids(df, out, "jockey")
    _name_ids(df, out, "trainer")
    _coerce_col(out, "race_time", _to_dublin)
    _coerce_col(out, "fetched_at", _to_dublin)
    # Timeform is a form/ratings source with no market column; the runner/outcome
    # row maps to the WIN market grain.
    if "market_type" not in out.columns or out["market_type"].isna().all():
        out["market_type"] = "WIN"
    out["source"] = "timeform"
    return _reindex_canonical(out)


# --- orchestration -------------------------------------------------------

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_OUTPUT = os.path.join(_BASE, "data", "unified_races.parquet")
_DEFAULT_INPUTS = {
    "live_odds": os.path.join(_BASE, "data", "live_odds.parquet"),
    "paddy_power": os.path.join(_BASE, "data", "cache", "paddy_power.json"),
    "betsp": os.path.join(_BASE, "data", "historical", "betsp.parquet"),
    "timeform": os.path.join(_BASE, "data", "historical", "timeform.parquet"),
}
_ADAPTERS = {
    "live_odds": _from_live_odds,
    "paddy_power": _from_paddy_power,
    "betsp": _from_betsp,
    "timeform": _from_timeform,
}


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per grain key, preferring the latest fetched_at."""
    if df.empty:
        return df
    out = df.sort_values("fetched_at", kind="stable", na_position="first")
    key = [col for col in _DEDUPE_KEY if col in out.columns]
    return out.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def _load_default(name: str, path: str):
    if not os.path.exists(path):
        logger.info("normalizer: source %s missing at %s, skipping", name, path)
        return None
    if name == "paddy_power":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # The scraper persists via utils.cache.Cache, which wraps the payload in
        # an envelope {key, cached_at, expires_at, ttl, data}. Unwrap it so the
        # adapter sees the raw {fetched_at, races} (a raw file passes through).
        if isinstance(data, dict) and "races" not in data and isinstance(data.get("data"), dict):
            data = data["data"]
        return data
    return pd.read_parquet(path)


def _write(df: pd.DataFrame, path: str) -> None:
    """Year-partitioned read-merge-write, swapped in atomically (crash-safe)."""
    out = df.copy()
    out["year"] = pd.to_datetime(out["race_date"], utc=True, errors="coerce").dt.year
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    if os.path.exists(path) and os.listdir(path):
        try:
            existing = pd.read_parquet(path)
            # Live bookmaker inputs are complete point-in-time snapshots, not
            # append-only history. Remove the previous snapshot for each live
            # source present in this normalize run before concatenating; otherwise
            # a stale bad selection can contaminate and fail-close its clean
            # replacement race.
            current_live_sources = set(
                out.loc[out["source"].isin(LIVE_SOURCES), "source"]
                .dropna().astype(str)
            )
            if current_live_sources and "source" in existing.columns:
                existing = existing[
                    ~existing["source"].astype(str).isin(current_live_sources)
                ]
            out = pd.concat([existing, out], ignore_index=True)
            out = _dedupe(out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("normalizer: could not read existing dataset (%s)", exc)
    # Read-merge-write re-reads the prior unified, so placeholder rows persisted by
    # an older normalize survive a mere source-side filter — drop them on the merged
    # frame too, flushing any already baked in. Same for non-UK/IRE rows baked into
    # an older unified before uk_ire_only was enabled.
    out = _fail_closed_live(out)
    out = _drop_placeholders(out)
    out = _filter_uk_ire(out)
    out = _reindex_canonical(out)
    out["year"] = pd.to_datetime(out["race_date"], utc=True, errors="coerce").dt.year.astype(int)
    # write_partitioned_parquet writes the full replacement dataset to a sibling
    # temp dir and swaps it in with two directory renames — never clears `path`
    # in place, so a crash mid-write can never leave the unified dataset empty
    # or half-written (see utils/storage/parquet_store.py).
    write_partitioned_parquet(out, path, "year")
    logger.debug("normalizer: wrote %d rows", len(out))


def normalize(sources=None, write=True, output_path=None) -> pd.DataFrame:
    """Unify all available sources into the canonical schema.

    sources: optional dict mapping source name -> pre-loaded input (DataFrame for
        parquet sources, dict for paddy_power). When None, reads default files,
        skipping any that are missing. Use to restrict inputs (e.g. in tests).
    write: persist the year-partitioned parquet at output_path.
    output_path: defaults to data/unified_races.parquet.
    """
    output_path = output_path or _DEFAULT_OUTPUT
    if sources is None:
        sources = {name: _load_default(name, path)
                   for name, path in _DEFAULT_INPUTS.items()}

    frames = []
    for name, data in sources.items():
        if data is None:
            continue
        adapter = _ADAPTERS.get(name)
        if adapter is None:
            logger.warning("normalizer: unknown source %s, skipping", name)
            continue
        adapted = adapter(data)
        # Skip empty adapter output: an absent source yields a 0-row, all-NA-column
        # frame, which contributes nothing but triggers pandas' "concatenation with
        # empty or all-NA entries is deprecated" FutureWarning at the concat below.
        if not adapted.empty:
            frames.append(adapted)

    if not frames:
        return _reindex_canonical(pd.DataFrame())

    combined = pd.concat(frames, ignore_index=True)
    combined = _validate(combined).reset_index(drop=True)
    combined = _fail_closed_live(combined)
    combined = _dedupe(combined)
    combined = _filter_uk_ire(combined)
    combined = _reindex_canonical(combined)

    if write:
        _write(combined, output_path)
    return combined


# Historical sources are scraped UK/IRE-only; never foreign-filtered (and they
# seed the set of known UK/IRE courses live rows are matched against).
_HIST_SOURCES = frozenset({"betsp", "timeform"})


def _canon_venue(s: pd.Series) -> pd.Series:
    """Canonical venue key for cross-source matching: lowercase, slug separators →
    spaces, a trailing ' park' dropped (history's 'Sandown' vs feeds'
    'sandown-park'), whitespace collapsed."""
    out = (s.astype("string").str.lower()
           .str.replace(r"[-_]", " ", regex=True)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip())
    return out.str.replace(r"\s+park$", "", regex=True)


def _filter_uk_ire(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only UK & Irish racing (the model's whole world).

    Live feeds list global racing (US/AUS/FRA/...); historical sources are
    UK/IRE-only and carry ``region`` in {UK, IRE}. A row is kept iff its region
    is UK/IRE **or** its venue is a known UK/IRE course (seen in a region-tagged
    row) — so live rows with a blank region still pass when they name a real
    UK/IRE track, while foreign tracks (never region-tagged) are dropped. Gated
    by the top-level ``uk_ire_only`` config flag (default on)."""
    from utils.config_loader import get_config

    if not bool(get_config().get("uk_ire_only", True)):
        return df
    if df.empty or "venue" not in df.columns:
        return df

    region = df["region"].astype("string").str.strip().str.upper() \
        if "region" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    is_uk_ire = region.isin(["UK", "IRE"])
    # Historical feeds are region-filtered at scrape (UK/IRE only) — trust them and
    # never drop their rows; the foreign contamination only enters via live odds.
    is_hist = df["source"].isin(_HIST_SOURCES) if "source" in df.columns \
        else pd.Series(False, index=df.index)

    # Match live venues to known UK/IRE courses by a canonical form so slug
    # spellings still resolve (paddy's "sandown-park" → history's "Sandown").
    canon = _canon_venue(df["venue"])
    known = set(canon[is_uk_ire | is_hist].dropna().unique())
    keep = is_hist | is_uk_ire | canon.isin(known)

    dropped = int((~keep).sum())
    if dropped:
        foreign = sorted(df.loc[~keep, "venue"].dropna().unique())
        logger.info("normalizer: dropped %d non-UK/IRE rows across %d venues (%s)",
                    dropped, len(foreign), ", ".join(foreign[:12]))
    return df.loc[keep].reset_index(drop=True)


_REQUIRED_STR_FIELDS = ["race_date", "venue", "horse_name", "horse_id",
                        "market_type", "source", "fetched_at"]
_MARKET_TYPES = ("WIN", "EACH_WAY", "PLACE")


def _fail_closed_live(df: pd.DataFrame) -> pd.DataFrame:
    """Drop an entire live source race on any primary-WIN integrity failure."""
    if df is None or df.empty or "source" not in df.columns:
        return df
    before = len(df)
    clean = validate_live_dataframe(df)
    if "field_size" in clean.columns:
        clean["market_field_size"] = clean["field_size"]
        clean = clean.drop(columns=["field_size"])
    if "booksum" in clean.columns:
        clean["market_booksum"] = clean["booksum"]
        clean = clean.drop(columns=["booksum"])
    dropped = before - len(clean)
    if dropped:
        # WARNING, not ERROR: this is the fail-closed guard doing its job, and it
        # fires on every ordinary run (a race that has gone off has its prices
        # pulled, and a partially-blocked source leaves unpriced rows behind).
        # Logging it as an error trained the eye to ignore real errors. Name the
        # sources and venues so it is actionable instead of just alarming.
        detail = ""
        try:
            if "source" in df.columns:
                kept = clean["source"].value_counts() if "source" in clean.columns else {}
                per_source = {
                    str(src): int(n) - int(kept.get(src, 0))
                    for src, n in df["source"].value_counts().items()
                    if int(n) - int(kept.get(src, 0)) > 0
                }
                if per_source:
                    detail = f" by source: {per_source}"
                    if "venue" in df.columns and len(clean) < before:
                        lost = df.loc[~df.index.isin(clean.index), "venue"]
                        venues = sorted({str(v) for v in lost.dropna().unique()})[:6]
                        if venues:
                            detail += f"; venues: {', '.join(venues)}"
        except Exception:  # noqa: BLE001 — diagnostics must never break normalize
            detail = ""
        logger.warning(
            "normalizer: fail-closed %d of %d live row(s) with no confirmed "
            "price%s", dropped, before, detail,
        )
    return clean


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows failing the UnifiedRaceRow contract; log how many were dropped.

    Vectorized mirror of ``UnifiedRaceRow`` — kept in sync with that model by
    hand (a row-by-row ``validate_python`` loop here cost ~15 CPU-min on the
    full ~582k-row frame). Rules replicated:
      - required str fields must be present and non-null;
      - market_type in {WIN, EACH_WAY, PLACE};
      - odds_decimal null OR a number > 1.0;
      - position null OR an integer >= 1.
    """
    if df.empty:
        return df
    keep = pd.Series(True, index=df.index)
    for col in _REQUIRED_STR_FIELDS:
        present = df[col].notna() if col in df.columns else False
        keep &= present
    keep &= df["market_type"].isin(_MARKET_TYPES) if "market_type" in df.columns else False

    if "odds_decimal" in df.columns:
        od = pd.to_numeric(df["odds_decimal"], errors="coerce")
        od_present = df["odds_decimal"].notna()
        keep &= ~od_present | (od > 1.0)  # null ok; else must coerce and be > 1.0

    if "position" in df.columns:
        pos = pd.to_numeric(df["position"], errors="coerce")
        pos_present = df["position"].notna()
        pos_ok = (pos >= 1) & (pos == pos.round())  # integral and >= 1
        keep &= ~pos_present | pos_ok

    keep = keep.fillna(False).astype(bool)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("normalizer: dropped %d invalid row(s)", dropped)
    return df[keep]

# --- shared transforms ---------------------------------------------------


# Strings that are a rendered ABSENCE, not a name. Some upstream stores write a
# missing jockey/trainer as the literal text "nan" (a float NaN that went through
# str()), and hashing that minted a perfectly ordinary-looking canonical id:
# step 09's audit found 1,002 labelled rows pooling 501 distinct horses into one
# fake jockey and one fake trainer, which then shared a trailing form history.
# Treated exactly like the empty string, which this function already rejected.
# Deliberately narrow: only tokens that cannot plausibly be a real runner or
# connection name. "unknown" is NOT included — it is a judgement call about a
# real value, not a rendered null, and belongs to whichever source emitted it.
_NULL_NAME_TOKENS = {"nan", "none", "null", "n/a", "n.a.", "-", "--"}


def _canonical_id(name) -> str | None:
    """Stable, cross-process hash of a normalized name. Null on empty or on a
    string that merely spells a missing value."""
    norm = norm_horse(name)
    if not norm:
        return None
    # Check the raw text as well as the normalized form: norm_horse strips
    # punctuation, so "n/a" would otherwise survive as "na".
    raw = "" if name is None else str(name).strip().lower()
    if raw in _NULL_NAME_TOKENS or norm.strip().lower() in _NULL_NAME_TOKENS:
        return None
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=8).hexdigest()


def _to_decimal(value) -> float | None:
    """Coerce odds to decimal. Handles decimals, fractional, EVS, NR/null."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in {"EVS", "EVENS"}:
        return 2.0
    if upper in {"NR", "SP", "-"}:
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            return round(float(num) / float(den) + 1.0, 4)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_dublin(value) -> str | None:
    """Align a timestamp to Europe/Dublin and return ISO text. Null on empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, utc=False, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return to_local(ts.to_pydatetime()).isoformat()
