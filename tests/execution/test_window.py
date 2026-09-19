"""Forward-window freeze/reset (Stage 6, requirement 4).

Starting the window is a deliberate, separate action from verifying it, and a
change to any frozen input (a model file, the selection lock, the execution
config block) must reset the accumulated window to zero and say exactly what
changed. These are the two properties this file exists to pin down.
"""
from __future__ import annotations

from datetime import datetime, timezone

from execution.window import WindowState, start_window, verify_window

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc)

BASE_HASHES = {
    "model_files": {"catboost_won_v3.bin": "aaa", "lgbm_won_v3.txt": "bbb"},
    "selection_lock": "sel1",
    "execution_config": "cfg1",
}


def _copy_hashes(**overrides) -> dict:
    out = {
        "model_files": dict(BASE_HASHES["model_files"]),
        "selection_lock": BASE_HASHES["selection_lock"],
        "execution_config": BASE_HASHES["execution_config"],
    }
    out.update(overrides)
    return out


def test_start_window_begins_running_at_todays_date(tmp_path):
    path = str(tmp_path / "window.json")
    state = start_window(path, hashes=BASE_HASHES, now=NOW)
    assert state.status == "running"
    assert state.window_start == "2026-07-28"
    assert state.hashes == BASE_HASHES


def test_start_window_is_a_no_op_once_running(tmp_path):
    path = str(tmp_path / "window.json")
    start_window(path, hashes=BASE_HASHES, now=NOW)
    again = start_window(path, hashes=_copy_hashes(selection_lock="different"), now=LATER)
    assert again.window_start == "2026-07-28"
    assert again.hashes == BASE_HASHES  # unchanged - start_window never overwrites


def test_verify_window_before_start_is_a_no_op(tmp_path):
    path = str(tmp_path / "nope.json")
    state, changes = verify_window(
        path, hashes=_copy_hashes(selection_lock="anything"), now=NOW,
    )
    assert state.status == "not_started"
    assert changes == []


def test_verify_window_unchanged_hashes_stay_running(tmp_path):
    path = str(tmp_path / "window.json")
    start_window(path, hashes=BASE_HASHES, now=NOW)
    state, changes = verify_window(path, hashes=dict(BASE_HASHES), now=LATER)
    assert state.status == "running"
    assert state.window_start == "2026-07-28"
    assert changes == []


def test_verify_window_model_file_change_resets_and_names_it(tmp_path):
    path = str(tmp_path / "window.json")
    start_window(path, hashes=BASE_HASHES, now=NOW)
    new_hashes = _copy_hashes()
    new_hashes["model_files"]["catboost_won_v3.bin"] = "changed"
    state, changes = verify_window(path, hashes=new_hashes, now=LATER)
    assert state.status == "reset"
    assert state.window_start == "2026-08-04"  # clock zeroed to the reset date
    assert changes == ["model_file_changed:catboost_won_v3.bin"]
    assert state.history[-1]["event"] == "reset"
    assert "catboost_won_v3.bin" in state.history[-1]["reason"]


def test_verify_window_selection_lock_change_resets_and_names_it(tmp_path):
    path = str(tmp_path / "window.json")
    start_window(path, hashes=BASE_HASHES, now=NOW)
    state, changes = verify_window(
        path, hashes=_copy_hashes(selection_lock="new-lock-hash"), now=LATER
    )
    assert state.status == "reset"
    assert changes == ["selection_lock_changed"]


def test_verify_window_execution_config_change_resets_and_names_it(tmp_path):
    path = str(tmp_path / "window.json")
    start_window(path, hashes=BASE_HASHES, now=NOW)
    state, changes = verify_window(
        path, hashes=_copy_hashes(execution_config="new-cfg-hash"), now=LATER
    )
    assert state.status == "reset"
    assert changes == ["execution_config_changed"]


def test_a_reset_window_can_reset_again_and_history_accumulates(tmp_path):
    path = str(tmp_path / "window.json")
    start_window(path, hashes=BASE_HASHES, now=NOW)
    verify_window(path, hashes=_copy_hashes(selection_lock="v2"), now=LATER)
    state, changes = verify_window(
        path, hashes=_copy_hashes(selection_lock="v2", execution_config="v2"), now=LATER
    )
    assert state.status == "reset"
    assert changes == ["execution_config_changed"]
    assert [h["event"] for h in state.history] == ["started", "reset", "reset"]


def test_days_elapsed_reads_from_window_start():
    state = WindowState(status="running", window_start="2026-07-28")
    assert state.days_elapsed(now=LATER) == 7
    assert state.running is True
    assert WindowState().running is False
