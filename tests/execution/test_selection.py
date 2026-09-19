"""Requirement 4: the selection protocol must make threshold mining impossible.

Every test here targets a specific way a search over thresholds can manufacture a
result: a random split that leaks, an uncapped grid that hides the search size, a
test window scored until it agrees, or a bias note that reads like an endorsement.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from execution.config import ExecutionConfig
from execution.selection import (
    SHORTLIST_FRACTION,
    StrategyCandidate,
    adjusted_alpha,
    bonferroni_alpha,
    build_grid,
    chronological_split,
    load_lock,
    run_selection,
    sidak_alpha,
)
# Aliased: pytest would otherwise try to collect the exception as a test class.
from execution.selection import TestWindowAlreadyScored as AlreadyScored

N_DATES = 20
ROWS_PER_DATE = 5


def make_cfg(tmp_path, **selection_overrides):
    cfg = ExecutionConfig.from_config({})
    sel = replace(
        cfg.selection, lock_file=str(tmp_path / "selection_lock.json"), **selection_overrides
    )
    return replace(cfg, selection=sel)


def make_frame(n_dates: int = N_DATES, rows_per_date: int = ROWS_PER_DATE) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    rows = []
    for d_i, day in enumerate(dates):
        for r in range(rows_per_date):
            rows.append(
                {
                    "race_date": day,
                    "race_uid": f"{day:%Y%m%d}-{r}",
                    "bet_price": 3.0 + r,
                    "edge": 0.01 * (r + 1),
                }
            )
    return pd.DataFrame(rows)


class Recorder:
    """Wraps an ``evaluate_fn`` and records which window each call saw."""

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, frame: pd.DataFrame, candidate: StrategyCandidate) -> float:
        dates = pd.to_datetime(frame["race_date"])
        self.calls.append(
            (
                dates.min().strftime("%Y-%m-%d"),
                dates.max().strftime("%Y-%m-%d"),
                candidate.name,
            )
        )
        return self.score_fn(frame, candidate)

    def calls_starting(self, start: str) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[0] == start]


def score_prefers(target_edge: float, target_ev: float, target_band: str):
    """Deterministic scorer whose maximum sits on one named grid point."""

    def _fn(frame: pd.DataFrame, candidate: StrategyCandidate) -> float:
        p = candidate.params
        return -(
            abs(p["min_edge"] - target_edge) * 100.0
            + abs(p["min_expected_value"] - target_ev) * 100.0
            + (0.0 if p["odds_band"] == target_band else 1.0)
        )

    return _fn


# ── chronological split ───────────────────────────────────────────────────────
def test_chronological_split_has_no_date_overlap_and_respects_fractions(tmp_path):
    cfg = make_cfg(tmp_path)
    frame = make_frame()

    train, validation, test = chronological_split(frame, cfg=cfg)

    tr = set(pd.to_datetime(train["race_date"]).dt.normalize())
    va = set(pd.to_datetime(validation["race_date"]).dt.normalize())
    te = set(pd.to_datetime(test["race_date"]).dt.normalize())

    # No date may appear in two splits, and together they partition the input.
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()
    assert len(tr | va | te) == N_DATES

    # 0.6 / 0.2 / remainder of the 20 distinct dates.
    assert (len(tr), len(va), len(te)) == (12, 4, 4)
    assert len(train) + len(validation) + len(test) == len(frame)

    # Strictly chronological: every train date precedes every validation date,
    # which precedes every test date.
    assert max(tr) < min(va) < max(va) < min(te)


def test_chronological_split_rejects_windows_too_short_to_hold_out(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(ValueError, match="3 distinct dates"):
        chronological_split(make_frame(n_dates=2), cfg=cfg)
    with pytest.raises(ValueError, match="empty"):
        chronological_split(pd.DataFrame(), cfg=cfg)


# ── grid ──────────────────────────────────────────────────────────────────────
def test_build_grid_truncates_at_max_strategies(tmp_path):
    full = build_grid(make_cfg(tmp_path))
    assert len(full) == 64, "default grid should be 4 edges x 4 EVs x 4 bands"

    capped = build_grid(make_cfg(tmp_path, max_strategies=10))
    assert len(capped) == 10
    # Truncation is from the deterministic enumeration order, never by score.
    assert [c.name for c in capped] == [c.name for c in full[:10]]


def test_build_grid_never_proposes_a_threshold_looser_than_the_gate(tmp_path):
    cfg = make_cfg(tmp_path)
    tight = replace(cfg, gates=replace(cfg.gates, min_edge=0.05, min_expected_value=0.12))
    grid = build_grid(tight)
    assert grid, "grid must never be empty"
    assert all(c.params["min_edge"] >= 0.05 for c in grid)
    assert all(c.params["min_expected_value"] >= 0.12 for c in grid)


# ── corrections ───────────────────────────────────────────────────────────────
def test_bonferroni_and_sidak_maths():
    assert bonferroni_alpha(0.05, 10) == pytest.approx(0.005)
    assert sidak_alpha(0.05, 10) == pytest.approx(1.0 - 0.95 ** (1 / 10))
    # Sidak is exact and therefore never stricter than Bonferroni.
    assert sidak_alpha(0.05, 64) > bonferroni_alpha(0.05, 64)

    assert adjusted_alpha("bonferroni", 0.05, 4) == pytest.approx(0.0125)
    assert adjusted_alpha("sidak", 0.05, 4) == pytest.approx(1.0 - 0.95 ** 0.25)
    assert adjusted_alpha("none", 0.05, 4) == pytest.approx(0.05)

    # m <= 0 is treated as a single test; a single test is uncorrected.
    for m in (0, -3):
        assert bonferroni_alpha(0.05, m) == pytest.approx(0.05)
        assert sidak_alpha(0.05, m) == pytest.approx(0.05)

    # An unknown correction falls back to the most conservative one.
    assert adjusted_alpha("typo", 0.05, 8) == pytest.approx(bonferroni_alpha(0.05, 8))


# ── run_selection ─────────────────────────────────────────────────────────────
def test_run_selection_reports_grid_size_and_scores_test_exactly_once(tmp_path):
    cfg = make_cfg(tmp_path)
    frame = make_frame()
    grid = build_grid(cfg)
    rec = Recorder(score_prefers(0.05, 0.12, "mid_4_12"))

    result = run_selection(frame, cfg=cfg, evaluate_fn=rec)

    assert result.n_strategies_tried == len(grid) == 64
    assert result.correction == "sidak"
    assert result.adjusted_alpha == pytest.approx(sidak_alpha(0.05, 64))
    assert result.selected is not None
    assert result.selected.name == "edge0.050_ev0.120_mid_4_12"
    assert result.reused_lock is False
    assert result.test_scored_at is not None

    # TRAIN saw every candidate; VALIDATION saw only the shortlist.
    train_calls = rec.calls_starting("2026-01-01")
    validation_calls = rec.calls_starting("2026-01-13")
    test_calls = rec.calls_starting("2026-01-17")
    assert len(train_calls) == 64
    assert len(validation_calls) == int(64 * SHORTLIST_FRACTION) == 16

    # The whole point: the test window is scored once, for one strategy.
    assert len(test_calls) == 1
    assert test_calls[0][2] == result.selected.name

    # Every candidate is reported with its train score; only the shortlist has a
    # validation score, so the search is fully visible in the artifact.
    assert len(result.candidates) == 64
    assert all(c["train_score"] is not None for c in result.candidates)
    assert sum(c["validation_score"] is not None for c in result.candidates) == 16
    assert sum(bool(c["selected"]) for c in result.candidates) == 1

    assert result.windows["train"][:2] == ["2026-01-01", "2026-01-12"]
    assert result.windows["validation"][:2] == ["2026-01-13", "2026-01-16"]
    assert result.windows["test"] == ["2026-01-17", "2026-01-20", 4 * ROWS_PER_DATE]

    # The lock is written and round-trips through to_dict/JSON.
    lock = load_lock(cfg.selection.lock_file)
    assert lock is not None
    assert lock["selected"]["name"] == result.selected.name
    assert lock["test_scored_at"] == result.test_scored_at
    json.dumps(result.to_dict())  # strictly serialisable (no inf/NaN leaking out)


def test_rerunning_reuses_the_lock_and_does_not_rescore_test(tmp_path):
    cfg = make_cfg(tmp_path)
    frame = make_frame()
    scorer = score_prefers(0.05, 0.12, "mid_4_12")

    first = run_selection(frame, cfg=cfg, evaluate_fn=Recorder(scorer))

    rec = Recorder(scorer)
    second = run_selection(frame, cfg=cfg, evaluate_fn=rec)

    assert second.reused_lock is True
    assert second.selected.name == first.selected.name
    assert second.test_score == pytest.approx(first.test_score)
    assert second.test_scored_at == first.test_scored_at
    # Train/validation may be re-scored freely; the test window may not.
    assert rec.calls_starting("2026-01-17") == []


def test_scoring_test_for_a_different_strategy_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    frame = make_frame()

    run_selection(frame, cfg=cfg, evaluate_fn=Recorder(score_prefers(0.05, 0.12, "mid_4_12")))

    rec = Recorder(score_prefers(0.02, 0.05, "all"))
    with pytest.raises(AlreadyScored) as excinfo:
        run_selection(frame, cfg=cfg, evaluate_fn=rec)

    assert "edge0.050_ev0.120_mid_4_12" in str(excinfo.value)
    assert "edge0.020_ev0.050_all" in str(excinfo.value)
    assert rec.calls_starting("2026-01-17") == [], "test window must not be touched"

    # force=True is the only way through, and it re-scores explicitly.
    forced = run_selection(
        frame, cfg=cfg, evaluate_fn=rec, force=True
    )
    assert forced.reused_lock is False
    assert forced.selected.name == "edge0.020_ev0.050_all"
    assert len(rec.calls_starting("2026-01-17")) == 1


def test_selection_bias_note_states_the_search_size_without_endorsing_it(tmp_path):
    cfg = make_cfg(tmp_path)
    result = run_selection(
        make_frame(), cfg=cfg, evaluate_fn=score_prefers(0.03, 0.08, "all")
    )
    note = result.selection_bias_note

    assert "64" in note
    assert "sidak" in note
    assert f"{result.adjusted_alpha:.4g}" in note
    assert "single draw" in note

    lowered = note.lower()
    for banned in ("profitab", "safe", "guarantee", "edge is real", "validated"):
        assert banned not in lowered, f"bias note must not endorse the result: {note}"
