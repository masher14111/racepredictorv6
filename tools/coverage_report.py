"""Reusable coverage/freshness report for any of the app's dated parquet stores.

Detects the failure modes a silent data-pipeline regression usually hides:
staleness (no new rows landing), all-null columns (a parser/join defect
masquerading as "no data"), and a sudden provider mix shift (one source
quietly stopped contributing, or started dominating).

Library entry point: ``build_coverage_report(df, ...)`` returns a plain,
JSON-serializable dict — safe to diff before/after a pipeline change.

CLI:
    python -m tools.coverage_report --path data/unified_races.parquet --name unified_races
    python -m tools.coverage_report --path data/features/training.parquet --name training \\
        --watch-cols timeform_rating race_class going_speed going_speed_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

import pandas as pd

# Columns worth watching by default across this app's dated stores (per
# memory/improvement/docs/improvement/CONTRACTS.md's measured missingness).
DEFAULT_WATCH_COLS = [
    "timeform_rating", "rating_rank", "pace_bias", "race_class", "class_change",
    "recent_form_avg", "going_speed", "going_speed_v2", "position",
]

# A source's day-over-day share moving by more than this (absolute
# percentage points, vs the prior 7-day trailing average) is flagged as a
# possible silent provider change (site redesign, block, config drift).
_PROVIDER_SHIFT_THRESHOLD = 0.30


def _to_date_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], utc=True, errors="coerce").dt.date


def _null_fill(df: pd.DataFrame, cols: list) -> dict:
    n = len(df)
    out = {}
    for col in cols:
        if col not in df.columns:
            out[col] = {"present": False}
            continue
        nn = int(df[col].notna().sum())
        out[col] = {"present": True, "non_null": nn, "rows": n,
                    "fill_fraction": (nn / n) if n else 0.0}
    return out


def _all_null_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if df[c].notna().sum() == 0]


def _by_source(df: pd.DataFrame, source_col: str, date_col: str) -> dict:
    if source_col not in df.columns:
        return {}
    out = {}
    for src, sub in df.groupby(source_col, dropna=False):
        d = _to_date_series(sub, date_col) if date_col in sub.columns else pd.Series([], dtype="object")
        out[str(src)] = {
            "rows": int(len(sub)),
            "date_min": str(d.min()) if len(d) and d.notna().any() else None,
            "date_max": str(d.max()) if len(d) and d.notna().any() else None,
        }
    return out


def _missing_calendar_days(df: pd.DataFrame, date_col: str, lookback_days: int = 60) -> list:
    """Calendar days with zero rows inside [max_date - lookback_days, max_date] —
    bounded so a multi-year store doesn't force a full-history day-by-day scan;
    a stale/short window is exactly where a silent acquisition gap matters most."""
    if date_col not in df.columns or df.empty:
        return []
    d = _to_date_series(df, date_col)
    d = d.dropna()
    if d.empty:
        return []
    hi = d.max()
    lo = max(d.min(), hi - timedelta(days=lookback_days))
    present = set(d[(d >= lo) & (d <= hi)])
    all_days = {lo + timedelta(days=i) for i in range((hi - lo).days + 1)}
    return sorted(str(x) for x in (all_days - present))


def _provider_shift_alerts(df: pd.DataFrame, source_col: str, date_col: str,
                           lookback_days: int = 30) -> list:
    """Flag a day where a source's share of that day's rows moved more than
    ``_PROVIDER_SHIFT_THRESHOLD`` from its trailing-7-day average share —
    catches a source silently going dark or suddenly dominating."""
    if source_col not in df.columns or date_col not in df.columns or df.empty:
        return []
    work = df[[source_col, date_col]].copy()
    work["d"] = _to_date_series(work, date_col)
    work = work.dropna(subset=["d"])
    if work.empty:
        return []
    hi = work["d"].max()
    lo = max(work["d"].min(), hi - timedelta(days=lookback_days))
    work = work[(work["d"] >= lo) & (work["d"] <= hi)]
    if work.empty:
        return []
    daily = work.groupby(["d", source_col]).size().unstack(fill_value=0)
    daily = daily.sort_index()
    share = daily.div(daily.sum(axis=1).replace(0, pd.NA), axis=0)
    trailing_avg = share.rolling(window=7, min_periods=1).mean().shift(1)
    alerts = []
    for d in share.index:
        for src in share.columns:
            cur = share.at[d, src]
            base = trailing_avg.at[d, src] if d in trailing_avg.index else None
            if pd.isna(cur) or base is None or pd.isna(base):
                continue
            if abs(float(cur) - float(base)) >= _PROVIDER_SHIFT_THRESHOLD:
                alerts.append({
                    "date": str(d), "source": str(src),
                    "day_share": round(float(cur), 3),
                    "trailing_7d_share": round(float(base), 3),
                })
    return alerts


def build_coverage_report(df: pd.DataFrame, name: str, *, date_col: str = "race_date",
                          source_col: str = "source", watch_cols: list | None = None,
                          today: date | None = None, stale_after_days: int = 2) -> dict:
    """Build a JSON-serializable coverage/freshness report for one dataset."""
    watch_cols = list(watch_cols or DEFAULT_WATCH_COLS)
    today = today or date.today()
    n = len(df)
    report: dict = {"dataset": name, "rows": n, "columns": int(df.shape[1]) if n or df.shape[1] else 0}

    if date_col in df.columns and n:
        d = _to_date_series(df, date_col).dropna()
        date_min = d.min() if len(d) else None
        date_max = d.max() if len(d) else None
        report["date_min"] = str(date_min) if date_min else None
        report["date_max"] = str(date_max) if date_max else None
        report["days_stale"] = (today - date_max).days if date_max else None
        report["is_stale"] = bool(report["days_stale"] is not None and report["days_stale"] > stale_after_days)
        report["missing_calendar_days_recent"] = _missing_calendar_days(df, date_col)
    else:
        report.update({"date_min": None, "date_max": None, "days_stale": None,
                       "is_stale": None, "missing_calendar_days_recent": []})

    report["all_null_columns"] = _all_null_columns(df) if n else []
    report["watched_fill"] = _null_fill(df, watch_cols) if n else {c: {"present": False} for c in watch_cols}
    report["by_source"] = _by_source(df, source_col, date_col) if n else {}
    report["provider_shift_alerts"] = _provider_shift_alerts(df, source_col, date_col) if n else []
    if "region" in df.columns and n:
        report["by_region"] = {str(k): int(v) for k, v in df["region"].value_counts(dropna=False).items()}
    return report


def _print_report(report: dict) -> None:
    print(f"=== {report['dataset']} ===")
    print(f"  rows: {report['rows']:,}  columns: {report['columns']}")
    print(f"  dates: {report.get('date_min')} -> {report.get('date_max')}"
          f"  (stale: {report.get('is_stale')}, {report.get('days_stale')} day(s) behind)")
    gaps = report.get("missing_calendar_days_recent") or []
    print(f"  missing calendar days (recent window): {len(gaps)}"
          + (f" e.g. {gaps[:5]}" if gaps else ""))
    print(f"  all-null columns: {report['all_null_columns'] or 'none'}")
    print("  watched column fill:")
    for col, info in report["watched_fill"].items():
        if not info.get("present"):
            print(f"    {col:<20} absent")
        else:
            print(f"    {col:<20} {info['non_null']:>10,} / {info['rows']:,} "
                  f"({info['fill_fraction']*100:6.2f}%)")
    if report.get("by_region"):
        print(f"  by region: {report['by_region']}")
    if report.get("provider_shift_alerts"):
        print(f"  provider shift alerts: {len(report['provider_shift_alerts'])}")
        for a in report["provider_shift_alerts"][:10]:
            print(f"    {a}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--date-col", default="race_date")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--watch-cols", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    name = args.name or args.path
    try:
        df = pd.read_parquet(args.path)
    except Exception as exc:  # noqa: BLE001
        print(f"MISSING/UNREADABLE: {args.path} ({exc})")
        return 1

    report = build_coverage_report(
        df, name, date_col=args.date_col, source_col=args.source_col,
        watch_cols=args.watch_cols)
    _print_report(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"  written -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
