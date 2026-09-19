"""Strict validation for live primary horse-race WIN markets.

Market/container identity is the primary defence against special-market
contamination.  Runner-name checks are deliberately only a secondary guard for
the case where a bookmaker places a special selection inside a market that was
otherwise identified as the primary WIN market.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from utils.text_norm import norm_horse

VALID = "VALID"
INVALID = "INVALID"
LIVE_SOURCES = frozenset({"livescorebet", "boylesports", "paddy_power"})

MIN_FIELD_SIZE = 2
MAX_FIELD_SIZE = 40
MIN_BOOKSUM = 0.80
MAX_BOOKSUM = 1.80

# Secondary only: primary market IDs/types/containers must already have matched.
_SPECIAL_SELECTION_RE = re.compile(
    r"(?:"
    r"\b(?:both|all)\s+to\s+finish\b|"
    r"\bto\s+finish\s+1\s*[-–]\s*2\b|"
    r"\bbetting\s+without\b|"
    r"\bwinning\s+distance\b|"
    r"\bto\s+win\s+by\s+\d|"
    # winning-distance selections read as "<horse> by 2 Lengths or more"
    r"\bby\s+\d[\d.\s+\-–]*lengths?\b|"
    r"\bany\s+horse\s+to\s+win\b|"
    r"\b(?:either|or)\b.+\bto\s+finish\b|"
    r"\bunnamed\s+(?:\d(?:st|nd|rd|th)\s+)?favourite\b|"
    r"\([^()]+\s+-\s+[^()]+\)\s*$"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RaceValidation:
    status: str
    reasons: tuple[str, ...]
    field_size: int
    booksum: float | None


def looks_like_special_selection(name: object) -> bool:
    """Return whether a primary-market selection still looks like a special."""
    return bool(_SPECIAL_SELECTION_RE.search(str(name or "").strip()))


def _reason_text(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return [text]
    return [str(v) for v in parsed] if isinstance(parsed, list) else [text]


def parse_validation_reasons(value: object) -> list[str]:
    """Decode a stored ``validation_reasons`` cell back into reason codes."""
    return _reason_text(value)


def reason_family(code: object) -> str:
    """Strip the measured suffix so ``extreme_booksum:9.36`` groups by kind."""
    return str(code or "").split(":", 1)[0]


def validate_primary_win_rows(
    rows: Sequence[Mapping],
    *,
    primary_market_count: int | None = None,
    min_field_size: int = MIN_FIELD_SIZE,
    max_field_size: int = MAX_FIELD_SIZE,
    min_booksum: float = MIN_BOOKSUM,
    max_booksum: float = MAX_BOOKSUM,
) -> RaceValidation:
    """Validate one source race after primary-WIN market selection.

    The result is race-level. Any issue invalidates the whole source race; callers
    must never salvage individual rows from an invalid result.
    """
    reasons: list[str] = []
    field_size = len(rows)

    for row in rows:
        if str(row.get("validation_status") or "").upper() == INVALID:
            reasons.extend(_reason_text(row.get("validation_reasons")))

    if primary_market_count is not None and primary_market_count != 1:
        reasons.append(f"primary_win_market_count:{primary_market_count}")

    market_types = {str(row.get("market_type") or "").strip().upper() for row in rows}
    if market_types != {"WIN"}:
        reasons.append("non_primary_win_market_type")

    market_ids = {str(row.get("market_id") or "").strip() for row in rows}
    if "" in market_ids:
        reasons.append("missing_market_id")
        market_ids.discard("")
    if len(market_ids) != 1:
        reasons.append(f"primary_win_market_ids:{len(market_ids)}")

    selection_ids = [str(row.get("selection_id") or "").strip() for row in rows]
    if any(not sid for sid in selection_ids):
        reasons.append("missing_selection_id")
    nonempty_ids = [sid for sid in selection_ids if sid]
    if len(nonempty_ids) != len(set(nonempty_ids)):
        reasons.append("duplicate_selection_id")

    names = [norm_horse(row.get("horse_name")) for row in rows]
    if any(not name for name in names):
        reasons.append("missing_selection_name")
    nonempty_names = [name for name in names if name]
    if len(nonempty_names) != len(set(nonempty_names)):
        reasons.append("duplicate_selection_name")

    if not min_field_size <= field_size <= max_field_size:
        reasons.append(f"implausible_field_size:{field_size}")

    if any(looks_like_special_selection(row.get("horse_name")) for row in rows):
        reasons.append("special_selection_in_primary_win")

    prices: list[float] = []
    for row in rows:
        try:
            price = float(row.get("odds_decimal"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 1.0:
            prices.append(price)
    booksum = sum(1.0 / price for price in prices) if prices else None
    # Individual SP/unpriced runners are legitimate and simply cannot carry EV;
    # booksum is computed from the available board. A wholly absent or
    # implausible board still fails through the booksum guard below.
    if booksum is None or not min_booksum <= booksum <= max_booksum:
        rendered = "none" if booksum is None else f"{booksum:.6f}"
        reasons.append(f"extreme_booksum:{rendered}")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return RaceValidation(
        status=VALID if not unique_reasons else INVALID,
        reasons=unique_reasons,
        field_size=field_size,
        booksum=booksum,
    )


def annotate_primary_win_rows(
    rows: Sequence[Mapping],
    *,
    primary_market_count: int | None = None,
) -> list[dict]:
    """Copy and annotate all rows with their shared race-level validation."""
    result = validate_primary_win_rows(
        rows, primary_market_count=primary_market_count
    )
    reason_json = json.dumps(list(result.reasons), separators=(",", ":"))
    annotated = []
    for row in rows:
        item = dict(row)
        item["validation_status"] = result.status
        item["validation_reasons"] = reason_json
        item["field_size"] = result.field_size
        item["booksum"] = result.booksum
        annotated.append(item)
    return annotated


def validate_live_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Revalidate and fail-close live source races in a flat DataFrame.

    Historical rows pass through untouched. Live groups are keyed by
    ``(source, race_id)`` where possible, with venue/time as a defensive fallback.
    Only fully valid primary WIN races are returned.
    """
    if df is None or df.empty or "source" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    live_mask = out["source"].isin(LIVE_SOURCES)
    historical = out.loc[~live_mask]
    live = out.loc[live_mask]
    if live.empty:
        return out

    kept: list[pd.DataFrame] = []
    key_cols = ["source"]
    if "race_id" in live.columns and live["race_id"].notna().any():
        key_cols.append("race_id")
    else:
        key_cols.extend(c for c in ("venue", "race_time") if c in live.columns)

    for _, grp in live.groupby(key_cols, dropna=False, sort=False):
        records = grp.to_dict("records")
        market_count = None
        if "market_id" in grp.columns:
            market_count = int(
                grp["market_id"].dropna().astype(str).replace("", pd.NA).dropna().nunique()
            )
        annotated = annotate_primary_win_rows(
            records, primary_market_count=market_count
        )
        if annotated and annotated[0]["validation_status"] == VALID:
            kept.append(pd.DataFrame(annotated, index=grp.index))

    pieces = [historical] + kept
    if not pieces:
        return out.iloc[0:0].copy()
    clean = pd.concat(pieces, axis=0, sort=False).sort_index(kind="stable")
    return clean.reset_index(drop=True)


def invalid_race_keys(df: pd.DataFrame) -> set[tuple]:
    """Return source/race IDs explicitly marked invalid (audit helper)."""
    if df is None or df.empty or "validation_status" not in df.columns:
        return set()
    bad = df[df["validation_status"].astype(str).str.upper().eq(INVALID)]
    return set(zip(bad.get("source", ""), bad.get("race_id", "")))
