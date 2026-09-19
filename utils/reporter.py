"""
utils/reporter.py — Unified report generator for Race Predictor v3.

Exports predictions, bets, and performance to CSV/JSON and generates a PDF
summary with charts and key metrics.  Can be triggered manually (CLI or API)
or on a recurring schedule via a background daemon thread.

Outputs (written to reports/ by default):
    bets_{ts}.csv           — all bets (one row per bet)
    predictions_{ts}.csv    — one row per selection, flattened from predictions cache
    snapshot_{ts}.json      — unified bundle: predictions + bets + performance + model_meta
    report_{ts}.pdf         — KPI table + equity-curve chart + P&L-by-venue bar chart
    manifest.json           — SHA-256 + row count per file, overwritten each run

Config block (config.yaml):
    reporter:
      enabled: false        # set true to enable the background scheduler
      interval_hours: 24    # how often the background thread runs
      output_dir: reports   # relative to project root

CLI:
    python -m utils.reporter [--output-dir DIR] [--format csv|json|pdf|all]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.bet_tracker import BetTracker
from utils.cache import get_cache
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.storage import DEFAULT_DB_PATH
from utils.timezone import now

_log = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "reports"
_PREDICTIONS_PATH = _PROJECT_ROOT / "data" / "predictions.json"


# ── Exceptions ────────────────────────────────────────────────────────────────


class IntegrityError(ValueError):
    """Raised when an export data integrity check fails fatally."""


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ReportResult:
    """Paths and metadata produced by a single Reporter.generate() call."""

    csv_paths: list[Path] = field(default_factory=list)
    json_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    generated_at: str = ""
    warnings: list[str] = field(default_factory=list)


# ── Integrity checking ────────────────────────────────────────────────────────


class IntegrityChecker:
    """Validates data before export.  All check_* methods return a list of warning strings."""

    _BETS_REQUIRED = frozenset({
        "id", "placed_at", "horse_name", "stake", "odds_decimal",
        "bet_type", "strategy", "outcome",
    })
    _VALID_OUTCOMES = frozenset({"win", "place", "lose"})

    @staticmethod
    def check_bets(df: pd.DataFrame) -> list[str]:
        """Schema + null/range guards on the bets DataFrame.

        Raises IntegrityError for missing required columns (fatal).
        Returns a list of non-fatal warning strings for soft violations.
        """
        if df.empty:
            return []

        missing = IntegrityChecker._BETS_REQUIRED - set(df.columns)
        if missing:
            raise IntegrityError(
                f"bets DataFrame missing required columns: {sorted(missing)}"
            )

        warnings: list[str] = []

        n_bad_stake = int((df["stake"] <= 0).sum())
        if n_bad_stake:
            warnings.append(f"{n_bad_stake} bet(s) with stake <= 0")

        n_bad_odds = int((df["odds_decimal"] < 1.0).sum())
        if n_bad_odds:
            warnings.append(f"{n_bad_odds} bet(s) with odds_decimal < 1.0")

        if "composite_score" in df.columns:
            cs = df["composite_score"].dropna()
            oob = int(((cs < 0.0) | (cs > 1.0)).sum())
            if oob:
                warnings.append(f"{oob} bet(s) with composite_score outside [0, 1]")

        bad_outcomes = (
            df["outcome"]
            .dropna()
            .loc[lambda s: ~s.isin(IntegrityChecker._VALID_OUTCOMES)]
        )
        if not bad_outcomes.empty:
            warnings.append(
                f"{len(bad_outcomes)} bet(s) with unrecognised outcome value"
            )

        return warnings

    @staticmethod
    def check_predictions(data: dict) -> list[str]:
        """Schema + range guards on the predictions dict.

        Raises IntegrityError for structurally broken data (fatal).
        Returns non-fatal warnings for out-of-range values.
        """
        if not isinstance(data, dict):
            raise IntegrityError("predictions data is not a dict")

        missing = {"generated_at", "races"} - set(data.keys())
        if missing:
            raise IntegrityError(
                f"predictions missing top-level keys: {sorted(missing)}"
            )

        races = data.get("races", [])
        if not isinstance(races, list):
            raise IntegrityError("predictions.races is not a list")

        warnings: list[str] = []
        for i, race in enumerate(races):
            label = race.get("venue", f"race[{i}]")
            for sel in list(race.get("selections", [])) + list(race.get("excluded_low_odds", [])):
                cs = sel.get("composite_score")
                if cs is not None and not (0.0 <= cs <= 1.0):
                    warnings.append(f"{label}: composite_score {cs:.4f} outside [0, 1]")
                odds = sel.get("decimal_odds")
                if odds is not None and odds < 1.0:
                    warnings.append(f"{label}: decimal_odds {odds} < 1.0")

        return warnings

    @staticmethod
    def check_referential(db_path: str | Path) -> list[str]:
        """Check bankroll_log.bet_id → bets.id integrity via a direct SQLite query."""
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM bankroll_log bl
                    LEFT JOIN bets b ON bl.bet_id = b.id
                    WHERE bl.bet_id IS NOT NULL AND b.id IS NULL
                    """
                ).fetchone()
            finally:
                conn.close()
        except Exception as exc:
            return [f"referential integrity check skipped ({exc})"]

        orphans = row[0] if row else 0
        if orphans:
            return [f"{orphans} bankroll_log row(s) reference non-existent bet_id"]
        return []


# ── PDF chart helpers ─────────────────────────────────────────────────────────


def _equity_curve_png(pl_series: pd.DataFrame) -> bytes:
    """Render cumulative P&L line chart to PNG bytes via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(pl_series)))
    y = pl_series["cumulative_profit"].tolist()

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(x, y, color="#2563eb", linewidth=1.5)
    ax.axhline(0, color="#d8dee8", linewidth=0.8, linestyle="--")
    ax.fill_between(x, y, 0,
                    where=[v >= 0 for v in y],
                    alpha=0.12, color="#16a34a", interpolate=True)
    ax.fill_between(x, y, 0,
                    where=[v < 0 for v in y],
                    alpha=0.12, color="#dc2626", interpolate=True)
    ax.set_title("Cumulative P&L", fontsize=10, color="#172033", pad=6)
    ax.set_xlabel("Bet #", fontsize=8, color="#6b7689")
    ax.set_ylabel("€", fontsize=8, color="#6b7689")
    ax.tick_params(labelsize=7, colors="#6b7689")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d8dee8")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _venue_bar_png(breakdown: pd.DataFrame) -> bytes:
    """Render horizontal P&L-by-venue bar chart to PNG bytes via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = breakdown.head(10).sort_values("total_profit")
    labels = top["venue"].astype(str).tolist()
    values = top["total_profit"].tolist()
    colours = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
    max_abs = max((abs(v) for v in values), default=1.0) or 1.0
    offset = max_abs * 0.02

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(labels))))
    bars = ax.barh(labels, values, color=colours, height=0.6)
    ax.axvline(0, color="#d8dee8", linewidth=0.8)

    for bar, val in zip(bars, values):
        sign = "+" if val >= 0 else ""
        ax.text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}€{val:,.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
            color="#3b4658",
        )

    ax.set_title("P&L by Venue (top 10)", fontsize=10, color="#172033", pad=6)
    ax.tick_params(labelsize=7, colors="#6b7689")
    ax.set_xlabel("Profit (€)", fontsize=8, color="#6b7689")
    for spine in ax.spines.values():
        spine.set_edgecolor("#d8dee8")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Reporter ──────────────────────────────────────────────────────────────────


class Reporter:
    """
    Generates CSV, JSON, and PDF reports from live predictions and bet history.

    Instantiate via get_reporter(); call generate() to produce all outputs at once.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        cfg = get_config().get("reporter", {})
        raw_dir = output_dir or cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        self._output_dir = Path(raw_dir)
        if not self._output_dir.is_absolute():
            self._output_dir = _PROJECT_ROOT / self._output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = Path(db_path or DEFAULT_DB_PATH)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_predictions(self) -> dict | None:
        """Load predictions from cache, falling back to data/predictions.json."""
        try:
            data = get_cache().get("predictions")
            if data:
                return data
            if _PREDICTIONS_PATH.exists():
                raw = json.loads(_PREDICTIONS_PATH.read_text(encoding="utf-8"))
                return raw.get("data", raw)
        except Exception as exc:
            _log.warning("reporter: could not load predictions: %s", exc)
        return None

    def _load_bets(self) -> tuple[pd.DataFrame, dict]:
        """Return (bets_df, summary) from BetTracker."""
        try:
            tracker = BetTracker(db_path=str(self._db_path))
            bets = tracker.all_bets()
            df = pd.DataFrame(bets) if bets else pd.DataFrame()
            summary = tracker.summary()
            return df, summary
        except Exception as exc:
            _log.warning("reporter: could not load bets: %s", exc)
            return pd.DataFrame(), {}

    def _load_model_meta(self) -> dict:
        """Read catboost_{vtag}_meta.json if present."""
        try:
            cfg = get_config()
            model_cfg = cfg.get("model", {})
            vtag = model_cfg.get("version_tag", "v3")
            model_dir = _PROJECT_ROOT / model_cfg.get("model_dir", "models")
            meta_path = model_dir / f"catboost_{vtag}_meta.json"
            if meta_path.exists():
                return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log.debug("reporter: model meta unavailable: %s", exc)
        return {}

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(
        self,
        output_dir: Path | None = None,
        ts: str | None = None,
    ) -> list[Path]:
        """Write bets_{ts}.csv and predictions_{ts}.csv. Returns list of paths written."""
        out = output_dir or self._output_dir
        ts = ts or now().strftime("%Y%m%d_%H%M%S")
        paths: list[Path] = []

        # bets CSV
        bets_df, _ = self._load_bets()
        if not bets_df.empty:
            for w in IntegrityChecker.check_bets(bets_df):
                _log.warning("reporter (bets): %s", w)
        bets_path = out / f"bets_{ts}.csv"
        if bets_df.empty:
            bets_path.write_text("no bets recorded\n", encoding="utf-8")
        else:
            bets_df.to_csv(bets_path, index=False)
        _log.info("reporter: bets CSV → %s (%d rows)", bets_path, len(bets_df))
        paths.append(bets_path)

        # predictions CSV (one row per selection)
        pred_data = self._load_predictions()
        pred_rows: list[dict] = []
        if pred_data:
            for w in IntegrityChecker.check_predictions(pred_data):
                _log.warning("reporter (predictions): %s", w)
            gen_at = pred_data.get("generated_at", "")
            for race in pred_data.get("races", []):
                base = {
                    "venue": race.get("venue"),
                    "race_time": race.get("race_time"),
                    "field_size": race.get("field_size"),
                    "each_way_available": race.get("each_way_available"),
                    "generated_at": gen_at,
                }
                for sel in race.get("selections", []):
                    pred_rows.append({**base, **sel})

        pred_path = out / f"predictions_{ts}.csv"
        if pred_rows:
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
        else:
            pred_path.write_text("no predictions available\n", encoding="utf-8")
        _log.info(
            "reporter: predictions CSV → %s (%d rows)", pred_path, len(pred_rows)
        )
        paths.append(pred_path)

        return paths

    # ------------------------------------------------------------------
    # JSON snapshot export
    # ------------------------------------------------------------------

    def export_json(
        self,
        output_dir: Path | None = None,
        ts: str | None = None,
    ) -> Path:
        """Write snapshot_{ts}.json (unified bundle). Returns path written."""
        out = output_dir or self._output_dir
        ts = ts or now().strftime("%Y%m%d_%H%M%S")

        bets_df, summary = self._load_bets()
        pred_data = self._load_predictions()
        model_meta = self._load_model_meta()

        if not bets_df.empty:
            for w in IntegrityChecker.check_bets(bets_df):
                _log.warning("reporter (bets): %s", w)
        if pred_data:
            for w in IntegrityChecker.check_predictions(pred_data):
                _log.warning("reporter (predictions): %s", w)
        for w in IntegrityChecker.check_referential(self._db_path):
            _log.warning("reporter (referential): %s", w)

        snapshot = {
            "generated_at": now().isoformat(),
            "predictions": pred_data or {},
            "bets": bets_df.to_dict(orient="records") if not bets_df.empty else [],
            "performance_summary": summary,
            "model_meta": model_meta,
        }

        path = out / f"snapshot_{ts}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        _log.info("reporter: snapshot JSON → %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def export_pdf(
        self,
        output_dir: Path | None = None,
        ts: str | None = None,
    ) -> Path:
        """Generate PDF summary with KPI table and charts. Returns path written."""
        try:
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                HRFlowable, Image, Paragraph, SimpleDocTemplate,
                Spacer, Table, TableStyle,
            )
        except ImportError as exc:
            raise ImportError(
                "reportlab is required for PDF export: pip install reportlab"
            ) from exc

        out = output_dir or self._output_dir
        ts = ts or now().strftime("%Y%m%d_%H%M%S")
        path = out / f"report_{ts}.pdf"

        bets_df, summary = self._load_bets()
        pred_data = self._load_predictions()

        pl_series: pd.DataFrame = pd.DataFrame()
        venue_breakdown: pd.DataFrame = pd.DataFrame()
        if not bets_df.empty:
            try:
                tracker = BetTracker(db_path=str(self._db_path))
                pl_series = tracker.pl_series()
                venue_breakdown = tracker.breakdown(by="venue")
            except Exception as exc:
                _log.warning("reporter: chart data unavailable: %s", exc)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2.0 * cm, rightMargin=2.0 * cm,
            topMargin=2.5 * cm, bottomMargin=2.5 * cm,
        )

        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                            fontSize=18, textColor=rl_colors.HexColor("#172033"),
                            spaceAfter=4)
        h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                            fontSize=13, textColor=rl_colors.HexColor("#2563eb"),
                            spaceBefore=14, spaceAfter=6)
        muted = ParagraphStyle("Muted", parent=styles["Normal"],
                               fontSize=9, textColor=rl_colors.HexColor("#6b7689"),
                               spaceAfter=2)

        story: list = []

        # Title block
        story.append(Paragraph("Race Predictor v3 — Report", h1))
        story.append(Paragraph(
            f"Generated: {now().strftime('%d %b %Y %H:%M')} (Dublin)", muted
        ))
        if pred_data:
            story.append(Paragraph(
                f"Predictions snapshot: {pred_data.get('generated_at', 'N/A')}", muted
            ))
        story.append(HRFlowable(
            width="100%", thickness=1,
            color=rl_colors.HexColor("#d8dee8"), spaceAfter=12,
        ))

        # KPI table
        story.append(Paragraph("Performance Metrics", h2))
        profit = summary.get("total_profit", 0.0)
        kpi_rows = [
            ["Metric", "Value"],
            ["Total P&L", f"{'+'if profit >= 0 else ''}€{profit:,.2f}"],
            ["ROI", f"{summary.get('roi_pct', 0.0):+.1f}%"],
            ["Bets Settled", str(summary.get("total_bets", 0))],
            ["Pending Bets", str(summary.get("pending_bets", 0))],
            ["Win Rate", f"{summary.get('win_rate', 0.0) * 100:.0f}%"],
            ["Win + Place Rate", f"{summary.get('place_rate', 0.0) * 100:.0f}%"],
            ["Total Staked", f"€{summary.get('total_staked', 0.0):,.2f}"],
            ["Current Bankroll", f"€{summary.get('bankroll', 0.0):,.2f}"],
            ["Max Drawdown", f"{summary.get('max_drawdown_pct', 0.0):.1f}%"],
        ]
        if pred_data:
            kpi_rows += [
                ["Races in Snapshot", str(pred_data.get("total_races", 0))],
                ["Selections", str(
                    sum(len(r.get("selections", [])) for r in pred_data.get("races", []))
                )],
            ]

        kpi_table = Table(kpi_rows, colWidths=[8 * cm, 7 * cm])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), rl_colors.HexColor("#2563eb")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#f6f7f9"), rl_colors.white]),
            ("FONTSIZE",       (0, 1), (-1, -1), 9),
            ("TEXTCOLOR",      (0, 1), (-1, -1), rl_colors.HexColor("#3b4658")),
            ("GRID",           (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#d8dee8")),
            ("TOPPADDING",     (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ]))
        story.append(kpi_table)

        # Equity curve chart
        if not pl_series.empty and len(pl_series) >= 2:
            story.append(Paragraph("Equity Curve", h2))
            try:
                equity_png = _equity_curve_png(pl_series)
                equity_buf = io.BytesIO(equity_png)
                story.append(Image(equity_buf, width=15 * cm, height=6 * cm))
            except Exception as exc:
                _log.warning("reporter: equity curve chart failed: %s", exc)
                story.append(Paragraph(f"(chart unavailable: {exc})", muted))

        # Venue P&L bar chart + table
        if not venue_breakdown.empty:
            story.append(Paragraph("P&L by Venue", h2))
            try:
                venue_png = _venue_bar_png(venue_breakdown)
                venue_buf = io.BytesIO(venue_png)
                chart_h = min(8 * cm, max(3 * cm, len(venue_breakdown) * 0.55 * cm))
                story.append(Image(venue_buf, width=15 * cm, height=chart_h))
            except Exception as exc:
                _log.warning("reporter: venue chart failed: %s", exc)
                story.append(Paragraph(f"(chart unavailable: {exc})", muted))

            story.append(Spacer(1, 0.3 * cm))
            vd_rows = [["Venue", "Bets", "Win Rate", "Staked", "Profit", "ROI"]]
            for _, row in venue_breakdown.iterrows():
                vp = row.get("total_profit", 0.0)
                vd_rows.append([
                    str(row.get("venue", "")),
                    str(int(row.get("bets", 0))),
                    f"{row.get('win_rate', 0.0) * 100:.0f}%",
                    f"€{row.get('total_staked', 0.0):,.2f}",
                    f"{'+'if vp >= 0 else ''}€{abs(vp):,.2f}",
                    f"{row.get('roi_pct', 0.0):+.1f}%",
                ])
            vd_table = Table(
                vd_rows,
                colWidths=[5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 2 * cm],
            )
            vd_table.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), rl_colors.HexColor("#2563eb")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",       (0, 0), (-1, 0), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [rl_colors.HexColor("#f6f7f9"), rl_colors.white]),
                ("FONTSIZE",       (0, 1), (-1, -1), 8),
                ("TEXTCOLOR",      (0, 1), (-1, -1), rl_colors.HexColor("#3b4658")),
                ("GRID",           (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#d8dee8")),
                ("TOPPADDING",     (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
                ("LEFTPADDING",    (0, 0), (-1, -1), 6),
            ]))
            story.append(vd_table)

        doc.build(story)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)
        _log.info("reporter: PDF → %s", path)
        return path

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_manifest(self, result: ReportResult) -> Path:
        manifest_path = self._output_dir / "manifest.json"
        all_paths = list(result.csv_paths)
        if result.json_path:
            all_paths.append(result.json_path)
        if result.pdf_path:
            all_paths.append(result.pdf_path)

        entries = []
        for p in all_paths:
            if not p.exists():
                continue
            row_count: int | None = None
            if p.suffix == ".csv":
                try:
                    with open(p, encoding="utf-8") as fh:
                        row_count = max(0, sum(1 for _ in fh) - 1)
                except Exception:
                    pass
            try:
                display_path = str(p.relative_to(_PROJECT_ROOT))
            except ValueError:
                display_path = str(p)
            entries.append({
                "path": display_path,
                "sha256": self._sha256(p),
                "row_count": row_count,
                "size_bytes": p.stat().st_size,
            })

        manifest = {
            "generated_at": result.generated_at,
            "files": entries,
            "warnings": result.warnings,
        }
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(manifest_path)
        _log.info("reporter: manifest → %s", manifest_path)
        return manifest_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        output_dir: str | Path | None = None,
        fmt: str = "all",
    ) -> ReportResult:
        """Run a full report cycle.

        Parameters
        ----------
        output_dir:
            Override output directory for this run; defaults to reporter.output_dir
            from config.yaml (or reports/).
        fmt:
            Which formats to produce: ``'csv'``, ``'json'``, ``'pdf'``, or ``'all'``.
        """
        out = self._output_dir
        if output_dir is not None:
            out = Path(output_dir)
            if not out.is_absolute():
                out = _PROJECT_ROOT / out
            out.mkdir(parents=True, exist_ok=True)

        ts = now().strftime("%Y%m%d_%H%M%S")
        result = ReportResult(generated_at=now().isoformat())

        # Referential integrity runs unconditionally
        for w in IntegrityChecker.check_referential(self._db_path):
            _log.warning("reporter: %s", w)
            result.warnings.append(w)

        if fmt in ("csv", "all"):
            try:
                result.csv_paths = self.export_csv(out, ts)
            except IntegrityError as exc:
                _log.error("reporter: CSV aborted — integrity error: %s", exc)
                result.warnings.append(f"CSV integrity error: {exc}")

        if fmt in ("json", "all"):
            try:
                result.json_path = self.export_json(out, ts)
            except IntegrityError as exc:
                _log.error("reporter: JSON aborted — integrity error: %s", exc)
                result.warnings.append(f"JSON integrity error: {exc}")

        if fmt in ("pdf", "all"):
            try:
                result.pdf_path = self.export_pdf(out, ts)
            except IntegrityError as exc:
                _log.error("reporter: PDF aborted — integrity error: %s", exc)
                result.warnings.append(f"PDF integrity error: {exc}")
            except ImportError as exc:
                _log.warning("reporter: PDF skipped — %s", exc)
                result.warnings.append(f"PDF skipped: {exc}")
            except Exception as exc:
                _log.error("reporter: PDF failed: %s", exc)
                result.warnings.append(f"PDF error: {exc}")

        try:
            result.manifest_path = self._write_manifest(result)
        except Exception as exc:
            _log.error("reporter: manifest write failed: %s", exc)

        if result.warnings:
            _log.warning(
                "reporter: generate() completed with %d warning(s)", len(result.warnings)
            )
        return result


# ── Scheduler ─────────────────────────────────────────────────────────────────


class _SchedulerThread(threading.Thread):
    def __init__(self, interval_hours: float) -> None:
        super().__init__(daemon=True, name="reporter-scheduler")
        self._interval_secs = interval_hours * 3600.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        _log.info(
            "reporter: scheduler started (interval=%.1fh)", self._interval_secs / 3600
        )
        # wait() returns False on timeout (next run) and True when stopped
        while not self._stop_event.wait(timeout=self._interval_secs):
            try:
                _log.info("reporter: scheduled run starting")
                get_reporter().generate()
            except Exception as exc:
                _log.error("reporter: scheduled run failed: %s", exc)
        _log.info("reporter: scheduler stopped")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)


_scheduler: Optional[_SchedulerThread] = None
_scheduler_lock = threading.Lock()


def start_scheduler() -> None:
    """Start the background report scheduler if enabled in config and not already running."""
    global _scheduler
    cfg = get_config().get("reporter", {})
    if not cfg.get("enabled", False):
        _log.debug("reporter: scheduler disabled (reporter.enabled=false in config.yaml)")
        return
    interval = float(cfg.get("interval_hours", 24))
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.is_alive():
            _log.debug("reporter: scheduler already running")
            return
        _scheduler = _SchedulerThread(interval)
        _scheduler.start()


def stop_scheduler() -> None:
    """Stop the background scheduler if it is running."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.is_alive():
            _scheduler.stop()
        _scheduler = None


# ── Singleton ─────────────────────────────────────────────────────────────────


_reporter: Optional[Reporter] = None
_reporter_lock = threading.Lock()


def get_reporter() -> Reporter:
    """Return the module-level Reporter singleton (thread-safe, double-check lock)."""
    global _reporter
    if _reporter is None:
        with _reporter_lock:
            if _reporter is None:
                _reporter = Reporter()
    return _reporter


# ── CLI ───────────────────────────────────────────────────────────────────────


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate prediction / bet / performance reports"
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: reports/)",
    )
    ap.add_argument(
        "--format", default="all", choices=["csv", "json", "pdf", "all"],
        dest="fmt",
        help="Which formats to generate (default: all)",
    )
    args = ap.parse_args()

    result = get_reporter().generate(output_dir=args.output_dir, fmt=args.fmt)

    print(f"\nReport generated — {result.generated_at}")
    for p in result.csv_paths:
        print(f"  CSV      → {p}")
    if result.json_path:
        print(f"  JSON     → {result.json_path}")
    if result.pdf_path:
        print(f"  PDF      → {result.pdf_path}")
    if result.manifest_path:
        print(f"  Manifest → {result.manifest_path}")
    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    !  {w}")


if __name__ == "__main__":
    _main()
