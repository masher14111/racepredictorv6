"""Tests for utils/reporter.py."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.reporter import (
    IntegrityChecker,
    IntegrityError,
    Reporter,
    ReportResult,
    _SchedulerThread,
    get_reporter,
    start_scheduler,
    stop_scheduler,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_bets_df(**overrides) -> pd.DataFrame:
    base = {
        "id": [1, 2],
        "placed_at": ["2026-01-01T10:00:00", "2026-01-02T11:00:00"],
        "horse_name": ["Horse A", "Horse B"],
        "stake": [10.0, 15.0],
        "odds_decimal": [4.0, 6.5],
        "bet_type": ["win", "each_way"],
        "strategy": ["flat", "flat"],
        "outcome": ["win", "lose"],
        "composite_score": [0.42, 0.27],
        "venue": ["Leopardstown", "Curragh"],
        "race_time": ["2026-01-01T14:30:00", "2026-01-02T15:00:00"],
        "profit": [30.0, -15.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_predictions() -> dict:
    return {
        "generated_at": "2026-01-01T09:00:00+00:00",
        "total_races": 2,
        "total_runners": 3,
        "races": [
            {
                "venue": "Leopardstown",
                "race_time": "2026-01-01T14:30:00+00:00",
                "field_size": 10,
                "each_way_available": True,
                "selections": [
                    {
                        "rank": 1,
                        "horse_name": "Horse A",
                        "horse_id": "ha1",
                        "jockey": "J. Smith",
                        "trainer": "T. Jones",
                        "decimal_odds": 4.0,
                        "implied_prob": 0.25,
                        "won_prob": 0.30,
                        "placed_2_prob": 0.50,
                        "showed_prob": 0.65,
                        "composite_score": 0.42,
                        "each_way_value": False,
                        "low_odds": False,
                    }
                ],
                "excluded_low_odds": [],
            },
            {
                "venue": "Curragh",
                "race_time": "2026-01-01T15:30:00+00:00",
                "field_size": 8,
                "each_way_available": True,
                "selections": [
                    {
                        "rank": 1,
                        "horse_name": "Horse B",
                        "horse_id": "hb1",
                        "jockey": "K. Lee",
                        "trainer": "M. Walsh",
                        "decimal_odds": 6.5,
                        "implied_prob": 0.154,
                        "won_prob": 0.20,
                        "placed_2_prob": 0.40,
                        "showed_prob": 0.55,
                        "composite_score": 0.31,
                        "each_way_value": True,
                        "low_odds": False,
                    },
                    {
                        "rank": 2,
                        "horse_name": "Horse C",
                        "horse_id": "hc1",
                        "jockey": "P. Doe",
                        "trainer": "R. Burke",
                        "decimal_odds": 9.0,
                        "implied_prob": 0.111,
                        "won_prob": 0.15,
                        "placed_2_prob": 0.30,
                        "showed_prob": 0.45,
                        "composite_score": 0.22,
                        "each_way_value": True,
                        "low_odds": False,
                    },
                ],
                "excluded_low_odds": [],
            },
        ],
    }


def _make_db(tmp_path: Path) -> Path:
    """Create a minimal races.db with bets + bankroll_log tables."""
    db = tmp_path / "races.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        PRAGMA user_version = 2;
        CREATE TABLE bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placed_at TEXT NOT NULL,
            race_id TEXT, horse_id TEXT, horse_name TEXT NOT NULL,
            venue TEXT, race_time TEXT,
            bet_type TEXT NOT NULL DEFAULT 'win',
            stake REAL NOT NULL, odds_decimal REAL NOT NULL,
            composite_score REAL, won_prob REAL,
            strategy TEXT NOT NULL DEFAULT 'flat',
            bankroll_before REAL, outcome TEXT,
            settled_at TEXT, gross_return REAL, profit REAL, notes TEXT
        );
        CREATE TABLE bankroll_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            bet_id INTEGER REFERENCES bets(id),
            amount REAL NOT NULL,
            balance REAL NOT NULL
        );
        INSERT INTO bankroll_log(logged_at, event_type, amount, balance)
            VALUES ('2026-01-01T00:00:00', 'init', 1000.0, 1000.0);
        INSERT INTO bets(placed_at, horse_name, stake, odds_decimal, bet_type, strategy, outcome, profit)
            VALUES ('2026-01-01T10:00:00', 'Horse A', 10.0, 4.0, 'win', 'flat', 'win', 30.0);
        INSERT INTO bankroll_log(logged_at, event_type, bet_id, amount, balance)
            VALUES ('2026-01-01T10:00:00', 'bet_placed', 1, -10.0, 990.0);
        INSERT INTO bankroll_log(logged_at, event_type, bet_id, amount, balance)
            VALUES ('2026-01-01T15:00:00', 'bet_settled', 1, 40.0, 1030.0);
        """
    )
    conn.commit()
    conn.close()
    return db


# ── IntegrityChecker.check_bets ───────────────────────────────────────────────


def test_check_bets_empty_returns_no_warnings():
    assert IntegrityChecker.check_bets(pd.DataFrame()) == []


def test_check_bets_valid_returns_no_warnings():
    assert IntegrityChecker.check_bets(_make_bets_df()) == []


def test_check_bets_missing_column_raises():
    df = _make_bets_df().drop(columns=["stake"])
    with pytest.raises(IntegrityError, match="stake"):
        IntegrityChecker.check_bets(df)


def test_check_bets_bad_stake_warns():
    df = _make_bets_df(stake=[0.0, 10.0])
    warnings = IntegrityChecker.check_bets(df)
    assert any("stake" in w for w in warnings)


def test_check_bets_bad_odds_warns():
    df = _make_bets_df(odds_decimal=[0.5, 4.0])
    warnings = IntegrityChecker.check_bets(df)
    assert any("odds_decimal" in w for w in warnings)


def test_check_bets_composite_out_of_range_warns():
    df = _make_bets_df(composite_score=[1.5, 0.3])
    warnings = IntegrityChecker.check_bets(df)
    assert any("composite_score" in w for w in warnings)


def test_check_bets_invalid_outcome_warns():
    df = _make_bets_df(outcome=["win", "refund"])
    warnings = IntegrityChecker.check_bets(df)
    assert any("outcome" in w for w in warnings)


def test_check_bets_none_outcome_allowed():
    df = _make_bets_df(outcome=["win", None])
    warnings = IntegrityChecker.check_bets(df)
    assert not any("outcome" in w for w in warnings)


# ── IntegrityChecker.check_predictions ───────────────────────────────────────


def test_check_predictions_valid_returns_no_warnings():
    assert IntegrityChecker.check_predictions(_make_predictions()) == []


def test_check_predictions_not_dict_raises():
    with pytest.raises(IntegrityError):
        IntegrityChecker.check_predictions([])


def test_check_predictions_missing_key_raises():
    with pytest.raises(IntegrityError, match="races"):
        IntegrityChecker.check_predictions({"generated_at": "x"})


def test_check_predictions_composite_out_of_range_warns():
    data = _make_predictions()
    data["races"][0]["selections"][0]["composite_score"] = 1.5
    warnings = IntegrityChecker.check_predictions(data)
    assert any("composite_score" in w for w in warnings)


def test_check_predictions_bad_odds_warns():
    data = _make_predictions()
    data["races"][0]["selections"][0]["decimal_odds"] = 0.5
    warnings = IntegrityChecker.check_predictions(data)
    assert any("decimal_odds" in w for w in warnings)


def test_check_predictions_none_odds_allowed():
    data = _make_predictions()
    data["races"][0]["selections"][0]["decimal_odds"] = None
    assert IntegrityChecker.check_predictions(data) == []


# ── IntegrityChecker.check_referential ───────────────────────────────────────


def test_check_referential_clean_db_no_warnings(tmp_path):
    db = _make_db(tmp_path)
    assert IntegrityChecker.check_referential(db) == []


def test_check_referential_orphaned_bet_id_warns(tmp_path):
    db = _make_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO bankroll_log(logged_at, event_type, bet_id, amount, balance) "
        "VALUES ('2026-01-02T00:00:00', 'bet_placed', 9999, -10.0, 990.0)"
    )
    conn.commit()
    conn.close()
    warnings = IntegrityChecker.check_referential(db)
    assert any("orphaned" in w or "non-existent" in w for w in warnings)


def test_check_referential_missing_db_returns_warning(tmp_path):
    warnings = IntegrityChecker.check_referential(tmp_path / "nonexistent.db")
    assert len(warnings) == 1
    assert "skipped" in warnings[0]


# ── Reporter.export_csv ───────────────────────────────────────────────────────


def _reporter(tmp_path, pred_data=None) -> Reporter:
    reporter = Reporter(output_dir=tmp_path / "reports", db_path=tmp_path / "races.db")
    return reporter


def test_export_csv_no_data_writes_placeholder(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None):
        paths = r.export_csv()
    assert len(paths) == 2
    assert "no bets" in paths[0].read_text()
    assert "no predictions" in paths[1].read_text()


def test_export_csv_writes_bets(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(_make_bets_df(), {})), \
         patch.object(r, "_load_predictions", return_value=None):
        paths = r.export_csv()
    bets_path = paths[0]
    df = pd.read_csv(bets_path)
    assert len(df) == 2
    assert "horse_name" in df.columns


def test_export_csv_writes_predictions(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=_make_predictions()):
        paths = r.export_csv()
    pred_path = paths[1]
    df = pd.read_csv(pred_path)
    assert len(df) == 3  # 1 + 2 selections across 2 races
    assert "venue" in df.columns
    assert "composite_score" in df.columns


def test_export_csv_uses_shared_timestamp(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None):
        paths = r.export_csv(ts="20260101_120000")
    assert paths[0].name == "bets_20260101_120000.csv"
    assert paths[1].name == "predictions_20260101_120000.csv"


# ── Reporter.export_json ──────────────────────────────────────────────────────


def test_export_json_structure(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(_make_bets_df(), {"roi_pct": 5.0})), \
         patch.object(r, "_load_predictions", return_value=_make_predictions()), \
         patch.object(r, "_load_model_meta", return_value={"test_auc": 0.72}), \
         patch.object(IntegrityChecker, "check_referential", return_value=[]):
        path = r.export_json(ts="20260101_120000")

    assert path.name == "snapshot_20260101_120000.json"
    data = json.loads(path.read_text())
    assert "generated_at" in data
    assert "predictions" in data
    assert "bets" in data
    assert len(data["bets"]) == 2
    assert data["performance_summary"]["roi_pct"] == 5.0
    assert data["model_meta"]["test_auc"] == 0.72


def test_export_json_atomic_write(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch.object(r, "_load_model_meta", return_value={}), \
         patch.object(IntegrityChecker, "check_referential", return_value=[]):
        path = r.export_json(ts="20260101_120000")
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


# ── Reporter.export_pdf ───────────────────────────────────────────────────────


def test_export_pdf_raises_on_missing_reportlab(tmp_path):
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch("builtins.__import__", side_effect=_import_blocker("reportlab")):
        with pytest.raises(ImportError, match="reportlab"):
            r.export_pdf()


def _import_blocker(blocked_prefix: str):
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _fake(name, *args, **kwargs):
        if name.startswith(blocked_prefix):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    return _fake


def test_export_pdf_produces_file(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    r = _reporter(tmp_path)
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None):
        path = r.export_pdf(ts="20260101_120000")
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.name == "report_20260101_120000.pdf"
    assert not path.with_suffix(".tmp").exists()


def test_export_pdf_with_data_produces_file(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    r = _reporter(tmp_path)

    pl = pd.DataFrame({
        "settled_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "horse_name": ["A", "B", "C"],
        "venue": ["Leopardstown"] * 3,
        "odds_decimal": [4.0, 6.0, 3.0],
        "stake": [10.0, 10.0, 10.0],
        "outcome": ["win", "lose", "win"],
        "profit": [30.0, -10.0, 20.0],
        "cumulative_profit": [30.0, 20.0, 40.0],
    })
    venue_bd = pd.DataFrame({
        "venue": ["Leopardstown", "Curragh"],
        "bets": [2, 1],
        "wins": [2, 0],
        "win_rate": [1.0, 0.0],
        "total_staked": [20.0, 10.0],
        "total_profit": [50.0, -10.0],
        "roi_pct": [250.0, -100.0],
    })

    mock_tracker = MagicMock()
    mock_tracker.pl_series.return_value = pl
    mock_tracker.breakdown.return_value = venue_bd

    with patch.object(r, "_load_bets", return_value=(_make_bets_df(), {"total_profit": 40.0})), \
         patch.object(r, "_load_predictions", return_value=_make_predictions()), \
         patch("utils.reporter.BetTracker", return_value=mock_tracker):
        path = r.export_pdf(ts="20260101_120000")

    assert path.exists()
    assert path.stat().st_size > 1000


# ── Reporter._write_manifest ──────────────────────────────────────────────────


def test_manifest_contains_all_files(tmp_path):
    r = _reporter(tmp_path)
    out = tmp_path / "reports"
    out.mkdir(parents=True, exist_ok=True)
    csv1 = out / "bets_x.csv"
    csv2 = out / "predictions_x.csv"
    jsn = out / "snapshot_x.json"
    csv1.write_text("id,horse_name\n1,A\n2,B\n", encoding="utf-8")
    csv2.write_text("venue,horse_name\nLeopo,A\n", encoding="utf-8")
    jsn.write_text("{}", encoding="utf-8")

    result = ReportResult(
        csv_paths=[csv1, csv2],
        json_path=jsn,
        generated_at="2026-01-01T12:00:00",
    )
    manifest_path = r._write_manifest(result)

    data = json.loads(manifest_path.read_text())
    assert data["generated_at"] == "2026-01-01T12:00:00"
    assert len(data["files"]) == 3
    paths_in_manifest = [f["path"] for f in data["files"]]
    assert any("bets_x.csv" in p for p in paths_in_manifest)
    assert any("snapshot_x.json" in p for p in paths_in_manifest)


def test_manifest_sha256_is_stable(tmp_path):
    r = _reporter(tmp_path)
    out = tmp_path / "reports"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "bets_x.csv"
    f.write_bytes(b"id\n1\n")

    result = ReportResult(csv_paths=[f], generated_at="now")
    r._write_manifest(result)
    manifest1 = json.loads((out.parent / "reports" / "manifest.json").read_text())

    r._write_manifest(result)
    manifest2 = json.loads((out.parent / "reports" / "manifest.json").read_text())

    sha1 = manifest1["files"][0]["sha256"]
    sha2 = manifest2["files"][0]["sha256"]
    assert sha1 == sha2
    assert len(sha1) == 64  # SHA-256 hex digest length


def test_manifest_row_count_for_csv(tmp_path):
    r = _reporter(tmp_path)
    out = tmp_path / "reports"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "bets_x.csv"
    f.write_text("id,horse_name\n1,A\n2,B\n3,C\n", encoding="utf-8")

    result = ReportResult(csv_paths=[f], generated_at="now")
    r._write_manifest(result)
    data = json.loads((out.parent / "reports" / "manifest.json").read_text())
    assert data["files"][0]["row_count"] == 3


def test_manifest_atomic_write_no_tmp_left(tmp_path):
    r = _reporter(tmp_path)
    result = ReportResult(generated_at="now")
    r._write_manifest(result)
    assert not (tmp_path / "reports" / "manifest.tmp").exists()


# ── Reporter.generate ─────────────────────────────────────────────────────────


def test_generate_all_produces_four_outputs(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    r = Reporter(output_dir=tmp_path / "reports", db_path=tmp_path / "races.db")
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch.object(r, "_load_model_meta", return_value={}), \
         patch.object(IntegrityChecker, "check_referential", return_value=[]):
        result = r.generate()

    assert len(result.csv_paths) == 2
    assert result.json_path is not None
    assert result.pdf_path is not None
    assert result.manifest_path is not None
    assert result.manifest_path.exists()


def test_generate_csv_only(tmp_path):
    r = Reporter(output_dir=tmp_path / "reports", db_path=tmp_path / "races.db")
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch.object(IntegrityChecker, "check_referential", return_value=[]):
        result = r.generate(fmt="csv")

    assert len(result.csv_paths) == 2
    assert result.json_path is None
    assert result.pdf_path is None


def test_generate_integrity_error_adds_warning(tmp_path):
    r = Reporter(output_dir=tmp_path / "reports", db_path=tmp_path / "races.db")

    def _bad_bets():
        raise IntegrityError("missing column: stake")

    with patch.object(r, "_load_bets", side_effect=_bad_bets), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch.object(IntegrityChecker, "check_referential", return_value=[]):
        result = r.generate(fmt="csv")

    assert any("integrity" in w.lower() for w in result.warnings)


def test_generate_referential_warning_surfaced(tmp_path):
    r = Reporter(output_dir=tmp_path / "reports", db_path=tmp_path / "races.db")
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch.object(r, "_load_model_meta", return_value={}), \
         patch.object(IntegrityChecker, "check_referential",
                      return_value=["1 bankroll_log row(s) reference non-existent bet_id"]):
        result = r.generate(fmt="json")

    assert any("non-existent" in w for w in result.warnings)


def test_generate_custom_output_dir(tmp_path):
    r = Reporter(output_dir=tmp_path / "reports", db_path=tmp_path / "races.db")
    custom = tmp_path / "custom_out"
    with patch.object(r, "_load_bets", return_value=(pd.DataFrame(), {})), \
         patch.object(r, "_load_predictions", return_value=None), \
         patch.object(r, "_load_model_meta", return_value={}), \
         patch.object(IntegrityChecker, "check_referential", return_value=[]):
        result = r.generate(output_dir=custom, fmt="csv")

    for p in result.csv_paths:
        assert p.parent == custom


# ── Scheduler ─────────────────────────────────────────────────────────────────


def test_scheduler_thread_is_daemon():
    t = _SchedulerThread(interval_hours=1.0)
    assert t.daemon


def test_scheduler_runs_and_stops():
    calls = []
    with patch("utils.reporter.get_reporter") as mock_get:
        mock_reporter = MagicMock()
        mock_reporter.generate.side_effect = lambda: calls.append(1)
        mock_get.return_value = mock_reporter

        t = _SchedulerThread(interval_hours=0.0001)  # ~0.36 s
        t.start()
        time.sleep(0.5)
        t.stop(timeout=3.0)

    assert not t.is_alive()
    assert len(calls) >= 1


def test_scheduler_stops_cleanly_before_first_run():
    t = _SchedulerThread(interval_hours=999)
    t.start()
    assert t.is_alive()
    t.stop(timeout=3.0)
    assert not t.is_alive()


def test_start_stop_scheduler_module_functions():
    import utils.reporter as mod

    with patch.object(mod, "get_config", return_value={"reporter": {"enabled": True, "interval_hours": 999}}):
        start_scheduler()
        assert mod._scheduler is not None
        assert mod._scheduler.is_alive()

        stop_scheduler()
        assert mod._scheduler is None


def test_start_scheduler_noop_when_disabled():
    import utils.reporter as mod

    with patch.object(mod, "get_config", return_value={"reporter": {"enabled": False}}):
        start_scheduler()
    assert mod._scheduler is None


def test_start_scheduler_noop_if_already_running():
    import utils.reporter as mod

    with patch.object(mod, "get_config", return_value={"reporter": {"enabled": True, "interval_hours": 999}}):
        start_scheduler()
        first = mod._scheduler
        start_scheduler()  # second call — should be a no-op
        assert mod._scheduler is first

    stop_scheduler()


# ── Singleton ─────────────────────────────────────────────────────────────────


def test_get_reporter_returns_same_instance():
    import utils.reporter as mod
    mod._reporter = None  # reset for test isolation
    r1 = get_reporter()
    r2 = get_reporter()
    assert r1 is r2
    mod._reporter = None


def test_get_reporter_thread_safety():
    import utils.reporter as mod
    mod._reporter = None

    instances: list = []

    def _get():
        instances.append(get_reporter())

    threads = [threading.Thread(target=_get) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(i is instances[0] for i in instances)
    mod._reporter = None
