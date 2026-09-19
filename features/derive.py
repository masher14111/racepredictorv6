"""Pure feature functions added to a runner-race frame.

Canonical home for the leak-safe primitives (going_speed, class_change,
trailing rates) plus odds/distance/form/rank features. ``scraper/timeform/
features.py`` re-imports the leak-safe primitives from here.
"""
import math
import re

import numpy as np
import pandas as pd

from features import _trailing_fast
from utils.text_norm import minute_key as _minute_key

_RACE_KEY = ["race_date", "venue"]


def _race_cols(df: pd.DataFrame) -> list:
    """Columns that identify a single race's ONE market's book. Prefer the
    recovered per-race key ``race_uid`` (venue + off-time); fall back to the
    venue-day [race_date, venue] when no off-time was available (audit C3).
    Used everywhere a per-race grouping or within-race rank is computed so
    field_size / market_rank / the ranks are on the right unit (~9 runners), not
    collapsed over the whole venue-day (~72).

    Also splits on ``market_type`` when present: a runner now carries one row
    per market it races under (features/fuse.py), so a race's WIN and PLACE
    books are two disjoint populations sharing one race_uid. Grouping on
    race_uid alone would rank/normalize a WIN row's implied_prob against the
    PLACE book's odds (and double-count field_size) — exactly the market
    contamination DESIGN.md rules out."""
    key = ["race_uid"] if "race_uid" in df.columns else list(_RACE_KEY)
    if "market_type" in df.columns:
        key = key + ["market_type"]
    return key


def add_race_key(df: pd.DataFrame) -> pd.DataFrame:
    """race_uid = venue + the race OFF-TIME (minute resolution), falling back to
    the race_date when no time is known. The Betfair backbone carries the off-time
    in race_time (recovered in the normalizer); without this, every race at a
    venue-day collapsed into one group and field_size/ranks were meaningless."""
    out = df.copy()
    n = len(out)
    venue = (out["venue"] if "venue" in out.columns
             else pd.Series([""] * n, index=out.index)).astype(str)
    rd = (pd.to_datetime(out["race_date"], utc=True, errors="coerce")
          if "race_date" in out.columns else pd.Series([pd.NaT] * n, index=out.index))
    date_slot = rd.dt.strftime("%Y-%m-%d")
    if "race_time" in out.columns:
        time_slot = [_minute_key(v) if (v is not None and not pd.isna(v))
                     else None for v in out["race_time"]]
    else:
        time_slot = [None] * n
    slot = [t if t is not None else d for t, d in zip(time_slot, date_slot)]
    out["race_uid"] = venue + "|" + pd.Series(slot, index=out.index).astype(str)
    return out


# --- leak-safe primitives (canonical home) -------------------------------


def add_going_speed(df: pd.DataFrame, going_map: dict) -> pd.DataFrame:
    out = df.copy()
    norm = {str(k).lower(): v for k, v in (going_map or {}).items()}
    # object dtype keeps mapped ints as ints and misses as None (avoid float/NaN coercion)
    out["going_speed"] = pd.Series(
        [norm.get(str(g).strip().lower()) if pd.notna(g) else None
         for g in out["going"]],
        index=out.index, dtype="object")
    return out


# Step 06: `add_going_speed` above only matched a going string that was
# EXACTLY one of `going_speed_map`'s hyphenated keys ("good-to-firm") — but
# the real raw `going` text uses spaces and "to" ("Good to Firm"), and often
# carries a parenthetical/slash qualifier ("Good (Good to Soft in places)",
# "Standard / Slow"). Measured on `data/unified_races.parquet` (651,154 rows,
# stage 06): only the 4 single-word exact matches (good/soft/heavy/firm) ever
# hit, leaving `good-to-firm`/`good-to-soft` — and every composite/Irish
# variant — permanently unmapped. `going_band` (below) already worked around
# this for going-preference by keying off just the leading token, but that
# deliberately loses the good-vs-good-to-firm distinction `going_speed` exists
# to keep. `add_going_speed_v2` fixes the STRING MATCHING only — it resolves
# a going string to the same frozen `going_speed_map` numbers, never invents a
# new physical scale — and is additive (CANDIDATE_INDEPENDENT_FEATURE_COLS),
# not adopted into any trained bundle's `going_speed`, per the D26/D27 pattern.
_GOING_SYNONYM_TO_CANONICAL = {
    "firm": "firm",
    "good to firm": "good-to-firm",
    "good": "good",
    "good to soft": "good-to-soft",
    "good-to-soft": "good-to-soft",  # observed stray already-hyphenated variant
    # Irish going terminology: "yielding" is the official Irish-turf equivalent
    # of the British "soft" band (both describe the same underlying going
    # state under each jurisdiction's own vocabulary), so it reuses soft's
    # existing number rather than inventing a new one.
    "yielding": "soft",
    "good to yielding": "good-to-soft",
    # A two-band transition string ("Soft to Heavy", "Yielding to Soft")
    # states the HEADLINE (leading) band the same way a parenthetical
    # "(Heavy in places)" qualifier does; both take the leading band's
    # existing number, not an invented midpoint.
    "soft to heavy": "soft",
    "yielding to soft": "soft",
    "soft": "soft",
    "heavy": "heavy",
}
# All-weather going terms describe a genuinely different surface (no
# firm..heavy turf scale applies) — never coerced onto the turf numbers.
_ALL_WEATHER_GOING_TERMS = {"standard", "slow", "fast"}


def _primary_going_phrase(raw) -> str | None:
    """Leading published going state: drop a parenthetical/slash qualifier,
    collapse whitespace, lowercase. 'Good (Good to Soft in places)' -> 'good';
    'Standard / Slow' -> 'standard'; 'Good to Firm' -> 'good to firm'."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = re.split(r"\s*\(", text, maxsplit=1)[0]
    text = re.split(r"\s*/\s*", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text or None


def add_going_speed_v2(df: pd.DataFrame, going_map: dict) -> pd.DataFrame:
    """Corrected-coverage companion to `add_going_speed` (see block comment
    above): same frozen numeric scale, broader/correct string recognition.
    Also adds `going_is_all_weather` so an AW fixture's going is distinguished
    from genuinely unknown/unparsed going, rather than silently null either way."""
    out = df.copy()
    norm = {str(k).lower(): v for k, v in (going_map or {}).items()}
    n = len(out)
    if "going" not in out.columns:
        out["going_speed_v2"] = pd.Series([None] * n, index=out.index, dtype="object")
        out["going_is_all_weather"] = pd.Series([None] * n, index=out.index, dtype="object")
        return out
    phrases = [_primary_going_phrase(g) for g in out["going"]]
    values, is_aw = [], []
    for phrase in phrases:
        if phrase is None:
            values.append(None)
            is_aw.append(None)
            continue
        if phrase in _ALL_WEATHER_GOING_TERMS:
            values.append(None)
            is_aw.append(True)
            continue
        canonical = _GOING_SYNONYM_TO_CANONICAL.get(phrase)
        values.append(norm.get(canonical) if canonical is not None else None)
        is_aw.append(False)
    out["going_speed_v2"] = pd.Series(values, index=out.index, dtype="object")
    out["going_is_all_weather"] = pd.Series(is_aw, index=out.index, dtype="object")
    return out


def add_class_change(df: pd.DataFrame) -> pd.DataFrame:
    """class_change = this race_class minus the horse's previous race_class.
    Negative = stepping UP in class (lower band number is higher class).

    A runner's WIN and PLACE row of the SAME race (features/fuse.py) share one
    race_class; they must not count as two separate steps in the horse's class
    history (DECISIONS D25/D26) — collapse to one row per (horse_name, real
    race) before the shift, then broadcast the result back to every row."""
    out = df.copy()
    n = len(out)
    if n == 0 or "horse_name" not in out.columns or "race_date" not in out.columns:
        out["class_change"] = pd.Series([pd.NA] * n, index=out.index, dtype="object")
        return out
    event = _trailing_fast._race_event_key(out)
    rc = pd.to_numeric(out["race_class"], errors="coerce").to_numpy() \
        if "race_class" in out.columns else np.full(n, np.nan)
    tmp = pd.DataFrame({"h": out["horse_name"].to_numpy(), "rd": out["race_date"].to_numpy(),
                       "e": event, "rc": rc})
    dedup = tmp.drop_duplicates(subset=["h", "e"], keep="first").sort_values(
        ["h", "rd"], kind="stable")
    prev = dedup.groupby("h")["rc"].shift(1)
    change_vals = dedup["rc"].to_numpy() - prev.to_numpy()
    lookup = pd.Series(change_vals,
                       index=pd.MultiIndex.from_arrays([dedup["h"], dedup["e"]]))
    out_key = pd.MultiIndex.from_arrays([tmp["h"], tmp["e"]])
    out["class_change"] = lookup.reindex(out_key).to_numpy()
    return out


def add_trailing_rates(df: pd.DataFrame, entity: str, win_col: str, place_col: str,
                       lookback_months: int, lookback_runs: int,
                       place_positions: int) -> pd.DataFrame:
    """Point-in-time win/place rate from runs before each row's race_date, bounded
    by lookback_months AND lookback_runs (tighter wins). Leak-free.

    Only known-result priors count toward the denominator; NA-position priors
    (most historical rows until results are scraped) would otherwise deflate every
    rate toward 0. Vectorized via _trailing_fast (was an O(n^2) per-row scan)."""
    out = df.copy().reset_index(drop=True)
    win, plc, runs = _trailing_fast.windowed_rates(
        out, entity, runs=lookback_runs, months=lookback_months,
        places=place_positions, strict=False, dropna=True)
    out[win_col] = win
    out[place_col] = plc
    out["runs_in_window"] = runs
    return out


def add_all_trailing_rates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply add_trailing_rates for horse, jockey and trainer entities."""
    months = cfg["lookback_months"]
    runs = cfg["lookback_runs"]
    places = cfg["place_positions"]
    out = add_trailing_rates(df, "horse_id", "historical_win_rate",
                             "historical_place_rate", months, runs, places)
    out = add_trailing_rates(out, "jockey_id", "jockey_win_rate",
                             "_jockey_place_rate", months, runs, places)
    out = add_trailing_rates(out, "trainer_id", "trainer_win_rate",
                             "_trainer_place_rate", months, runs, places)
    return out.drop(columns=["_jockey_place_rate", "_trainer_place_rate"])


# --- odds features -------------------------------------------------------


def _best_odds(row) -> float | None:
    # Pre-off prices ONLY. odds_finish (returned/finishing SP) is deliberately
    # excluded: it is the outcome-correlated settling price, so deriving market
    # features from it leaks the result (audit C2). morningwap is the Betfair
    # morning weighted-average price — a genuine pre-off price, 100% populated in
    # the betSP backbone — and matches the live odds_decimal the scrapers write at
    # prediction time, so train/serve market features come from the same regime.
    for col in ("odds_decimal", "sp", "morningwap"):
        v = row.get(col)
        if v is not None and not pd.isna(v) and float(v) > 1.0:
            return float(v)
    return None


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """implied_prob, overround-normalized prob, log_odds, market_rank, field_size.

    Uses the live odds_decimal, falling back to sp then the pre-off morningwap.
    Never uses odds_finish (finishing SP) — that would leak the outcome (C2)."""
    out = df.copy()
    best = out.apply(_best_odds, axis=1)
    out["implied_prob"] = best.map(lambda v: 1.0 / v if v else pd.NA)
    out["log_odds"] = best.map(lambda v: math.log(v) if v else pd.NA)
    grp = out.groupby(_race_cols(out), sort=False)
    out["field_size"] = grp["horse_id"].transform("size")
    out["market_rank"] = grp["implied_prob"].rank(ascending=False, method="min")
    totals = grp["implied_prob"].transform("sum")
    out["overround_norm_prob"] = out["implied_prob"] / totals
    return out


# --- distance ------------------------------------------------------------

_DIST_RE = re.compile(r"(?:(\d+)m)?\s*(?:(\d+)f)?\s*(?:(\d+)y)?", re.IGNORECASE)


def _to_furlongs(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    m = _DIST_RE.fullmatch(text.replace(" ", ""))
    if not m or not any(m.groups()):
        return None
    miles = int(m.group(1)) if m.group(1) else 0
    furlongs = int(m.group(2)) if m.group(2) else 0
    yards = int(m.group(3)) if m.group(3) else 0
    return miles * 8 + furlongs + yards / 220.0


def add_distance_furlongs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["distance_furlongs"] = out["distance"].map(_to_furlongs)
    return out


# --- recent form ---------------------------------------------------------


def _parse_form(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0, 0, float("nan")
    figures = [c for c in str(value) if c.isalnum()]
    digits = [int(c) for c in figures if c.isdigit()]
    runs = len(figures)
    wins = sum(1 for d in digits if d == 1)
    avg = (sum(digits) / len(digits)) if digits else float("nan")
    return runs, wins, avg


def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["recent_form"].map(_parse_form)
    out["recent_form_runs"] = parsed.map(lambda t: t[0])
    out["recent_form_wins"] = parsed.map(lambda t: t[1])
    out["recent_form_avg"] = parsed.map(lambda t: t[2])
    return out


# --- rating rank ---------------------------------------------------------


def add_rating_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rating_rank"] = out.groupby(_race_cols(out), sort=False)["timeform_rating"].rank(
        ascending=False, method="min")
    return out


# --- going band ----------------------------------------------------------

# Map the leading ground descriptor of a free-text going string to a coarse band.
# The raw `going` is ~100% populated (the config-driven going_speed map only hit
# ~21%), so a band keyed off the first descriptor revives going-preference form.
_GOING_BANDS = {
    "heavy": "heavy", "soft": "soft", "yielding": "soft", "holding": "soft",
    "good": "good", "standard": "standard", "slow": "standard", "fast": "firm",
    "firm": "firm", "hard": "firm",
}


def _going_band(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    # First alphabetic token captures the primary ground (e.g. "Good to Firm" ->
    # good, "Standard / Slow" -> standard, "Soft (Heavy in places)" -> soft).
    m = re.match(r"[a-z]+", text)
    return _GOING_BANDS.get(m.group(0), None) if m else None


def add_going_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["going_band"] = pd.Series(
        [_going_band(g) for g in out["going"]] if "going" in out.columns
        else [None] * len(out), index=out.index, dtype="object")
    return out


# --- freshness / experience ----------------------------------------------


def add_days_since_last_run(df: pd.DataFrame) -> pd.DataFrame:
    """days_since_last_run = gap (days) from a horse's previous run to this race.

    Leak-free: reads only the strictly-prior run's date. NaN for a horse's first
    seen run. Captures layoff/freshness (both the well-known ~21-90-day sweet spot
    and the rust of a very long absence).

    A runner's WIN and PLACE row of the SAME race (features/fuse.py) share one
    race_date; naively shifting within the horse group let a row see its own
    market-sibling as a spurious zero-day "prior run" (DECISIONS D25/D26).
    Collapse to one row per (horse, real race) before the shift, then broadcast
    the computed gap back onto every row of that race."""
    out = df.copy()
    n = len(out)
    if n == 0 or "horse_id" not in out.columns or "race_date" not in out.columns:
        out["days_since_last_run"] = pd.Series([pd.NA] * n, index=out.index, dtype="object")
        return out
    rd = pd.to_datetime(out["race_date"], utc=True, errors="coerce")
    event = _trailing_fast._race_event_key(out)
    tmp = pd.DataFrame({"hid": out["horse_id"].to_numpy(), "rd": rd.to_numpy(), "e": event})
    dedup = tmp.drop_duplicates(subset=["hid", "e"], keep="first").sort_values(
        "rd", kind="stable")
    prev = dedup.groupby("hid", sort=False)["rd"].shift(1)
    gap = (dedup["rd"] - prev).dt.total_seconds() / 86400.0
    lookup = pd.Series(gap.to_numpy(),
                       index=pd.MultiIndex.from_arrays([dedup["hid"], dedup["e"]]))
    out_key = pd.MultiIndex.from_arrays([tmp["hid"], tmp["e"]])
    out["days_since_last_run"] = lookup.reindex(out_key).to_numpy()
    return out


def add_career_runs(df: pd.DataFrame) -> pd.DataFrame:
    """horse_career_runs = count of a horse's strictly-prior known-result runs.

    Leak-free running count (self excluded). Unlike runs_in_window it is uncapped
    and unwindowed, so it separates unexposed/early-career types from veterans —
    a strong interaction with form-rate reliability.

    Same market-duplicate collapse as add_days_since_last_run: a runner's
    WIN/PLACE row of the SAME race must count as ONE career run, not two
    (DECISIONS D25/D26)."""
    out = df.copy()
    n = len(out)
    if n == 0 or "horse_id" not in out.columns or "race_date" not in out.columns:
        out["horse_career_runs"] = pd.Series([0] * n, index=out.index, dtype="int64")
        return out
    rd = pd.to_datetime(out["race_date"], utc=True, errors="coerce")
    event = _trailing_fast._race_event_key(out)
    contrib = pd.to_numeric(out.get("position"), errors="coerce").notna().astype(int) \
        if "position" in out.columns else pd.Series(0, index=out.index)
    tmp = pd.DataFrame({"hid": out["horse_id"].to_numpy(), "rd": rd.to_numpy(),
                        "e": event, "c": contrib.to_numpy()})
    dedup = tmp.drop_duplicates(subset=["hid", "e"], keep="first").sort_values(
        "rd", kind="stable")
    # cumulative prior known runs = cumsum within horse minus self's own contribution
    career = dedup.groupby("hid", sort=False)["c"].cumsum() - dedup["c"]
    lookup = pd.Series(career.to_numpy(),
                       index=pd.MultiIndex.from_arrays([dedup["hid"], dedup["e"]]))
    out_key = pd.MultiIndex.from_arrays([tmp["hid"], tmp["e"]])
    out["horse_career_runs"] = lookup.reindex(out_key).to_numpy().astype("int64")
    return out
