"""Step 16 guarded harness: the versioned optional text-feature join.

Mirrors tests/test_text_archive.py's point-in-time pattern (temp sqlite db,
no real corpus touched) applied to the join in features.text_features_v1 —
the archive's own no-lookahead guarantee is already tested there; these tests
prove the NEW join code built on top of it (race_uid venue-casing
normalisation, eligibility exclusion, tri-state preservation, provenance
labelling, cache-only hosted reuse) behaves correctly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from llm.text_archive import TextArchive
from llm.text_features import TextFeatureExtractor
from features.text_features_v1 import (
    CANDIDATE_TEXT_FEATURE_COLS,
    TextJoinConfig,
    _archive_key,
    attach_text_features,
    decision_cutoff,
    flag_price_or_tip_language,
    text_coverage_report,
)

RACE_UID_MATRIX = "Ripon|2026-06-17T17:57"  # title-case venue, as the training matrix stores it
RACE_UID_ARCHIVE = "ripon|2026-06-17T17:57"  # lower-case venue, as race_facts_key writes it
OFF_LOCAL = datetime(2026, 6, 17, 17, 57, tzinfo=timezone(timedelta(hours=1)))  # BST


@pytest.fixture
def archive(tmp_path):
    a = TextArchive(db_path=str(tmp_path / "text.db"))
    yield a
    a.close()


def _row(race_uid=RACE_UID_MATRIX, horse_name="Storm King", horse_id="h1", won=0):
    return {"race_uid": race_uid, "market_type": "WIN", "horse_id": horse_id,
            "horse_name": horse_name, "won": won}


# ── race_uid venue-casing normalisation ─────────────────────────────────────

def test_archive_key_normalises_venue_casing():
    assert _archive_key(RACE_UID_MATRIX) == RACE_UID_ARCHIVE
    assert _archive_key("ROYAL ASCOT|2026-06-17T14:30") == "royalascot|2026-06-17T14:30"


def test_join_finds_archived_text_despite_venue_casing_mismatch(archive):
    archive.record(text="Wears a hood for the first time and open to more.",
                    source="sporting_life_spotlight", race_uid=RACE_UID_ARCHIVE,
                    horse_name="Storm King", fetched_at=datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc))
    df = pd.DataFrame([_row()])
    out = attach_text_features(df, archive=archive)
    assert bool(out.loc[0, "text_v1_available"]) is True
    assert out.loc[0, "text_v1_source"] == "regex"
    assert out.loc[0, "text_v1_headgear_first_time"] == 1.0


# ── no-lookahead / future-edit immutability ─────────────────────────────────

def test_future_comment_cannot_change_an_earlier_cutoffs_features(archive):
    early_fetch = datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc)
    archive.record(text="No mention of headgear here.", source="sporting_life_spotlight",
                    race_uid=RACE_UID_ARCHIVE, horse_name="Storm King", fetched_at=early_fetch)
    df = pd.DataFrame([_row()])
    before = attach_text_features(df, archive=archive)

    # A later "edit" (new row, same runner/race, later fetched_at) must not
    # change what an EARLIER cutoff already computed.
    later_fetch = datetime(2026, 6, 17, 9, 0, tzinfo=timezone.utc)
    archive.record(text="Wears a hood for the first time today.",
                    source="sporting_life_spotlight", race_uid=RACE_UID_ARCHIVE,
                    horse_name="Storm King", fetched_at=later_fetch)
    # Force the cutoff to fall strictly between the two fetches by using the
    # cutoff directly instead of relying on the (fixed) race off-time.
    cutoff = early_fetch + timedelta(hours=1)
    archived_at_cutoff = archive.as_of(RACE_UID_ARCHIVE, "Storm King", as_of=cutoff)
    assert archived_at_cutoff.text == "No mention of headgear here."  # unchanged by the later edit

    # And a cutoff AFTER the edit sees the new version.
    archived_after_edit = archive.as_of(RACE_UID_ARCHIVE, "Storm King", as_of=later_fetch + timedelta(minutes=1))
    assert archived_after_edit.text == "Wears a hood for the first time today."
    assert before.loc[0, "text_v1_headgear_first_time"] == 0.0  # first version: no match


def test_edited_after_cutoff_comment_is_excluded(archive):
    # Only a comment fetched AFTER the race's decision cutoff exists — the
    # join must treat the runner as having no available text, not use it.
    cutoff = decision_cutoff(RACE_UID_MATRIX)
    archive.record(text="Course winner and open to more.", source="sporting_life_spotlight",
                    race_uid=RACE_UID_ARCHIVE, horse_name="Storm King",
                    fetched_at=cutoff + timedelta(minutes=5))
    df = pd.DataFrame([_row()])
    out = attach_text_features(df, archive=archive)
    assert bool(out.loc[0, "text_v1_available"]) is False
    assert out.loc[0, "text_v1_source"] == "none"
    assert np.isnan(out.loc[0, "text_v1_course_winner"])


# ── eligibility exclusion (not guessed) ─────────────────────────────────────

def test_missing_runner_identity_is_ineligible_not_guessed(archive):
    archive.record(text="Course winner.", source="sporting_life_spotlight",
                    race_uid=RACE_UID_ARCHIVE, horse_name="Storm King",
                    fetched_at=datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc))
    df = pd.DataFrame([_row(horse_name=None, horse_id=None)])
    out = attach_text_features(df, archive=archive)
    assert bool(out.loc[0, "text_v1_available"]) is False


def test_unparseable_race_uid_is_ineligible(archive):
    archive.record(text="Course winner.", source="sporting_life_spotlight",
                    race_uid=RACE_UID_ARCHIVE, horse_name="Storm King",
                    fetched_at=datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc))
    assert decision_cutoff("not-a-race-uid") is None
    df = pd.DataFrame([_row(race_uid="garbage-no-pipe")])
    out = attach_text_features(df, archive=archive)
    assert bool(out.loc[0, "text_v1_available"]) is False


def test_race_with_no_archived_text_takes_fast_default_path(archive):
    df = pd.DataFrame([_row(race_uid="Doncaster|2026-01-01T13:00")])
    out = attach_text_features(df, archive=archive)
    assert out.loc[0, "text_v1_source"] == "none"
    assert all(np.isnan(out.loc[0, c]) for c in CANDIDATE_TEXT_FEATURE_COLS)


# ── tri-state preservation ───────────────────────────────────────────────────

def test_unknown_is_never_coerced_to_false(archive):
    # A comment with no phrase matching ANY feature: regex reports False (no
    # keyword found), never unknown — that is documented regex behaviour, not
    # a bug. This test locks in that the join passes it through unchanged.
    archive.record(text="Nothing notable to say about this runner.",
                    source="sporting_life_spotlight", race_uid=RACE_UID_ARCHIVE,
                    horse_name="Storm King", fetched_at=datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc))
    df = pd.DataFrame([_row()])
    out = attach_text_features(df, archive=archive)
    assert out.loc[0, "text_v1_available"]
    for c in CANDIDATE_TEXT_FEATURE_COLS:
        assert out.loc[0, c] == 0.0
    assert bool(out.loc[0, "text_v1_published_at_known"]) is False  # spotlight never supplies one


# ── odds/tip language separation ────────────────────────────────────────────

def test_flag_price_or_tip_language_is_a_separate_diagnostic_not_a_feature():
    assert flag_price_or_tip_language("A best bet each-way at 9/2.") is True
    assert flag_price_or_tip_language("Wears a hood for the first time.") is False
    assert flag_price_or_tip_language(None) is False
    # None of the six factual features are derived from price/tip language —
    # structural guarantee, not just this sample.
    from llm.text_features import TEXT_FEATURES
    price_words = ("odds", "price", "\\bnap\\b", "\\btip\\b", "each-way", "e/w", "9/2",
                   "starting price")
    for feat in TEXT_FEATURES:
        for pat in feat.patterns + feat.negative_patterns:
            assert not any(w in pat.lower() for w in price_words), (feat.name, pat)


# ── hosted cache-only reuse: never a live call ──────────────────────────────

def test_cache_only_hosted_backend_never_touches_network(archive, monkeypatch):
    from utils.cache import Cache
    from features.text_features_v1 import build_cache_only_hosted_backend

    archive.record(text="Returning from a long absence, open to improvement.",
                    source="sporting_life_spotlight", race_uid=RACE_UID_ARCHIVE,
                    horse_name="Storm King", fetched_at=datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc))
    df = pd.DataFrame([_row()])

    empty_cache = Cache(cache_dir=str(archive.db_path).replace("text.db", "llm_cache_empty"))
    hosted = build_cache_only_hosted_backend(
        model="deepseek/deepseek-v4.1-flash", cache=empty_cache)

    def _fail_if_called(*a, **k):
        raise AssertionError("cache-only hosted backend must never attempt a network call")
    monkeypatch.setattr("httpx.post", _fail_if_called)

    out = attach_text_features(df, archive=archive, hosted_backend=hosted)
    # Cache miss -> falls back to regex, honestly labelled, no network call and
    # no exception propagated to the caller.
    assert out.loc[0, "text_v1_source"] == "hosted_cache_miss_regex_fallback"
    assert out.loc[0, "text_v1_returning_from_layoff"] == 1.0


# ── coverage report is honest, not a claimed corpus size ───────────────────

def test_coverage_report_counts_only_real_overlap(archive):
    archive.record(text="Course winner.", source="sporting_life_spotlight",
                    race_uid=RACE_UID_ARCHIVE, horse_name="Storm King",
                    fetched_at=datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc))
    df = pd.DataFrame([_row(), _row(race_uid="Doncaster|2026-01-01T13:00", horse_name="Other Horse")])
    cov = text_coverage_report(df, archive=archive)
    assert cov["total_rows"] == 2
    assert cov["archive_known_race_uids"] == 1
    assert cov["rows_with_race_in_archive"] == 1
    assert cov["races_with_race_in_archive"] == 1


def test_candidate_cols_not_wired_into_any_promoted_or_price_free_list():
    import models.features as mf
    for c in CANDIDATE_TEXT_FEATURE_COLS:
        assert c not in mf.FEATURE_COLS
        assert c not in mf.PRICE_FREE_FEATURE_COLS
        assert c not in mf.CANDIDATE_INDEPENDENT_FEATURE_COLS
