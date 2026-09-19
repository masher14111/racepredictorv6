"""Audit live primary-WIN race fields by source.

Reports, per source: race counts, runner-count and booksum distributions,
rejected races with their rejection reasons, a rollup of reason families, and
concrete suspicious-selection examples.

Paddy Power and BoyleSports are read from their quarantine-aware caches;
LivescoreBet is read from data/live_odds.parquet.

Note on coverage: markets rejected at *parse* time (an unknown LivescoreBet
marketGroupId, a BoyleSports container that is not the primary race-winner
market) never reach a cache or the parquet, so they are visible only in the
scraper logs. What this report can evidence is every race rejected by the
race-level contract, plus any special-looking selection that survived into a
retained artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.market_validation import (
    VALID,
    looks_like_special_selection,
    parse_validation_reasons,
    reason_family,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = ROOT / "data" / "live_odds.parquet"
DEFAULT_PADDY = ROOT / "data" / "cache" / "paddy_power.json"
DEFAULT_BOYLE = ROOT / "data" / "cache" / "boylesports.json"

AUDIT_COLUMNS = [
    "source", "race_id", "venue", "race_time", "market_id", "market_name",
    "validation_status", "validation_reasons", "runners", "unique_selections",
    "booksum",
]
SUSPECT_COLUMNS = [
    "source", "race_id", "venue", "race_time", "market_name", "selection_id",
    "horse_name", "validation_status", "flag",
]


def _unwrap_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data if isinstance(data, dict) else {}


def _inherited(*candidates):
    """First non-null value walking selection -> market -> race."""
    for value in candidates:
        if value is not None:
            return value
    return None


def _paddy_rows(path: Path) -> pd.DataFrame:
    cache = _unwrap_cache(path)
    rows = []
    fetched_at = cache.get("fetched_at")
    for race in cache.get("races", []):
        for market in race.get("markets", []):
            for selection in market.get("selections", []):
                rows.append({
                    "fetched_at": fetched_at,
                    "source": "paddy_power",
                    "race_id": race.get("race_id"),
                    "race_time": race.get("race_time"),
                    "venue": race.get("venue"),
                    "market_type": market.get("market_type"),
                    "market_id": market.get("market_id"),
                    "market_name": market.get("market_name"),
                    "selection_id": selection.get("selection_id"),
                    "horse_name": selection.get("horse_name"),
                    "odds_decimal": selection.get("odds_decimal"),
                    "validation_status": _inherited(
                        selection.get("validation_status"),
                        market.get("validation_status"),
                        race.get("validation_status"),
                    ),
                    "validation_reasons": _inherited(
                        selection.get("validation_reasons"),
                        market.get("validation_reasons"),
                        race.get("validation_reasons"),
                    ),
                })
    return pd.DataFrame(rows)


def _boyle_rows(path: Path) -> pd.DataFrame:
    cache = _unwrap_cache(path)
    fetched_at = cache.get("fetched_at")
    rows = []
    for item in cache.get("rows", []):
        row = dict(item)
        row["source"] = "boylesports"
        row["fetched_at"] = fetched_at
        rows.append(row)
    return pd.DataFrame(rows)


def load_rows(
    parquet_path: Path, paddy_path: Path, boyle_path: Path = DEFAULT_BOYLE
) -> pd.DataFrame:
    frames = []
    if parquet_path.exists():
        parquet = pd.read_parquet(parquet_path)
        boyle = _boyle_rows(boyle_path)
        if not boyle.empty and "source" in parquet.columns:
            parquet = parquet[~parquet["source"].eq("boylesports")]
        frames.append(parquet)
        if not boyle.empty:
            frames.append(boyle)
    paddy = _paddy_rows(paddy_path)
    if not paddy.empty:
        frames.append(paddy)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _join_unique(series) -> str:
    return ",".join(sorted(set(series.dropna().astype(str))))


def _merge_reasons(series) -> str:
    codes: list[str] = []
    for value in series:
        for code in parse_validation_reasons(value):
            if code not in codes:
                codes.append(code)
    return ";".join(codes)


def race_audit(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    live = rows.copy()
    for column in (
        "market_type", "market_id", "market_name", "validation_status",
        "validation_reasons", "race_id", "venue", "race_time", "horse_name",
        "odds_decimal",
    ):
        if column not in live.columns:
            live[column] = pd.NA
    live = live[live["market_type"].astype("string").eq("WIN")].copy()
    if live.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    live["_inverse_odds"] = 1.0 / pd.to_numeric(
        live["odds_decimal"], errors="coerce"
    )
    audit = live.groupby(
        ["source", "race_id"], dropna=False, sort=True, as_index=False
    ).agg(
        venue=("venue", "first"),
        race_time=("race_time", "first"),
        market_id=("market_id", _join_unique),
        market_name=("market_name", _join_unique),
        validation_status=(
            "validation_status", lambda s: _join_unique(s) or "MISSING"
        ),
        validation_reasons=("validation_reasons", _merge_reasons),
        runners=("horse_name", "size"),
        unique_selections=("horse_name", "nunique"),
        booksum=("_inverse_odds", "sum"),
    )
    return audit[AUDIT_COLUMNS]


def source_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    valid = audit[audit["validation_status"].eq(VALID)]
    if valid.empty:
        return pd.DataFrame()
    return valid.groupby("source").agg(
        races=("race_id", "size"),
        median_runners=("runners", "median"),
        min_runners=("runners", "min"),
        max_runners=("runners", "max"),
        median_booksum=("booksum", "median"),
        min_booksum=("booksum", "min"),
        max_booksum=("booksum", "max"),
    )


def distribution(audit: pd.DataFrame, column: str) -> pd.DataFrame:
    """Quantile spread of ``column`` per source, over VALID races only."""
    if audit.empty or column not in audit.columns:
        return pd.DataFrame()
    valid = audit[audit["validation_status"].eq(VALID)]
    if valid.empty:
        return pd.DataFrame()
    values = pd.to_numeric(valid[column], errors="coerce")
    frame = pd.DataFrame({"source": valid["source"], column: values})
    return frame.groupby("source")[column].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )


def rejected_races(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    bad = audit[~audit["validation_status"].eq(VALID)].copy()
    return bad.sort_values(["source", "race_time"], kind="stable")


def rejection_reason_counts(audit: pd.DataFrame) -> pd.DataFrame:
    """Rejected-race counts per source and reason family."""
    columns = ["source", "reason", "races"]
    bad = rejected_races(audit)
    if bad.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for _, race in bad.iterrows():
        codes = [c for c in str(race["validation_reasons"] or "").split(";") if c]
        for code in dict.fromkeys(reason_family(c) for c in codes) or ["unspecified"]:
            records.append({"source": race["source"], "reason": code})
    counts = (
        pd.DataFrame(records)
        .groupby(["source", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "races"})
    )
    return counts.sort_values(
        ["source", "races", "reason"], ascending=[True, False, True]
    )[columns]


def suspicious_selections(rows: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Selection-level examples worth eyeballing, capped per source.

    Two independent flags: a selection whose name matches the secondary
    special-market pattern, and any retained row whose market_type is not WIN.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(columns=SUSPECT_COLUMNS)
    frame = rows.copy()
    for column in SUSPECT_COLUMNS + ["market_type"]:
        if column not in frame.columns:
            frame[column] = pd.NA

    flagged = frame[frame["horse_name"].map(looks_like_special_selection)].copy()
    flagged["flag"] = "special_selection_pattern"

    non_win = frame[~frame["market_type"].astype("string").eq("WIN")].copy()
    non_win["flag"] = "non_primary_win_market_type"

    suspect = pd.concat([flagged, non_win], ignore_index=True, sort=False)
    if suspect.empty:
        return pd.DataFrame(columns=SUSPECT_COLUMNS)
    suspect = suspect[SUSPECT_COLUMNS].drop_duplicates()
    return (
        suspect.sort_values(["source", "flag", "horse_name"], kind="stable")
        .groupby("source", sort=True)
        .head(limit)
        .reset_index(drop=True)
    )


def _show(title: str, frame: pd.DataFrame, empty: str, index: bool = True) -> None:
    print(f"\n{title}")
    if frame is None or frame.empty:
        print(f"  {empty}")
        return
    print(frame.to_string(index=index))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--paddy-cache", type=Path, default=DEFAULT_PADDY)
    parser.add_argument("--boyle-cache", type=Path, default=DEFAULT_BOYLE)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--max-examples", type=int, default=10,
        help="suspicious selection examples to show per source",
    )
    parser.add_argument(
        "--no-races", action="store_true", help="omit the per-race table",
    )
    args = parser.parse_args()

    rows = load_rows(args.parquet, args.paddy_cache, args.boyle_cache)
    audit = race_audit(rows)

    _show("VALID SOURCE SUMMARY", source_summary(audit), "no VALID races")
    _show("RUNNER-COUNT DISTRIBUTION (VALID)", distribution(audit, "runners"),
          "no VALID races")
    _show("BOOKSUM DISTRIBUTION (VALID)", distribution(audit, "booksum"),
          "no VALID races")

    status = (
        audit.groupby(["source", "validation_status"]).size().rename("races")
        if not audit.empty else pd.DataFrame()
    )
    _show("VALIDATION STATUS", status, "no races")
    _show("REJECTED MARKETS / RACES", rejected_races(audit),
          "none rejected", index=False)
    _show("REJECTION REASONS BY SOURCE", rejection_reason_counts(audit),
          "none rejected", index=False)
    _show(
        f"SUSPICIOUS SELECTION EXAMPLES (max {args.max_examples}/source)",
        suspicious_selections(rows, limit=args.max_examples),
        "none found in retained rows", index=False,
    )
    if not args.no_races:
        _show("RACES", audit, "no races", index=False)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.output_csv, index=False)
        print(f"\nWrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
