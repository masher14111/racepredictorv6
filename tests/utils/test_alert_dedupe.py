"""Alert de-duplication store."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from utils import alert_dedupe


@pytest.fixture(autouse=True)
def _tmp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_dedupe, "_PATH", str(tmp_path / "sent_alerts.json"))
    yield


def test_first_sighting_is_not_seen_second_is():
    fp = alert_dedupe.fingerprint("drop", "Dundalk|19:00", "Beta", "4.00")
    assert alert_dedupe.seen(fp) is False
    assert alert_dedupe.seen(fp) is True


def test_different_fingerprints_are_independent():
    a = alert_dedupe.fingerprint("drop", "race1", "Beta", "4.00")
    b = alert_dedupe.fingerprint("drop", "race1", "Beta", "3.00")
    assert alert_dedupe.seen(a) is False
    assert alert_dedupe.seen(b) is False   # a further drop is a new alert


def test_fingerprint_is_stable_and_readable():
    fp = alert_dedupe.fingerprint("soon", "Dundalk|19:00")
    assert fp == "soon|Dundalk|19:00"
    assert fp == alert_dedupe.fingerprint("soon", "Dundalk|19:00")


def test_empty_fingerprint_never_suppresses():
    assert alert_dedupe.seen("") is False
    assert alert_dedupe.seen("") is False


def test_reset_clears_everything():
    fp = alert_dedupe.fingerprint("top", "r", "h")
    alert_dedupe.seen(fp)
    alert_dedupe.reset()
    assert alert_dedupe.seen(fp) is False


def test_state_survives_across_calls_on_disk(tmp_path):
    fp = alert_dedupe.fingerprint("top", "r", "h")
    alert_dedupe.seen(fp)
    on_disk = json.loads((tmp_path / "sent_alerts.json").read_text())
    assert fp in on_disk


def test_entries_older_than_the_ttl_are_pruned(tmp_path):
    """A stale entry must not silence tomorrow's identical alert."""
    fp = alert_dedupe.fingerprint("drop", "r", "h", "4.00")
    old = (datetime.now(tz=timezone.utc) - timedelta(hours=13)).isoformat()
    (tmp_path / "sent_alerts.json").write_text(json.dumps({fp: old}))
    assert alert_dedupe.seen(fp) is False


def test_recent_entries_are_kept(tmp_path):
    fp = alert_dedupe.fingerprint("drop", "r", "h", "4.00")
    recent = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
    (tmp_path / "sent_alerts.json").write_text(json.dumps({fp: recent}))
    assert alert_dedupe.seen(fp) is True


@pytest.mark.parametrize("junk", ["", "not json", "[]", '{"k": "not-a-date"}'])
def test_corrupt_state_file_degrades_to_alerting(tmp_path, junk):
    """A damaged file must mean "alert once more", never a crash or silence."""
    (tmp_path / "sent_alerts.json").write_text(junk)
    assert alert_dedupe.seen("drop|r|h|4.00") is False


def test_missing_file_is_fine():
    assert alert_dedupe.seen("anything") is False


def test_unwritable_path_does_not_raise(monkeypatch, tmp_path):
    """Bookkeeping must never be able to break a prediction run."""
    monkeypatch.setattr(alert_dedupe, "_PATH", str(tmp_path / "nodir" / "x" / "s.json"))
    monkeypatch.setattr(alert_dedupe.os, "makedirs",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    assert alert_dedupe.seen("drop|r|h") is False
