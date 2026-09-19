"""Unit tests for the pipeline_full orchestrator's pure helpers.

These exercise the thread-budgeting and freshness logic without running any
training, scraping, or scoring (which need data + a GPU).
"""
import time

import pytest

import pipeline_full as pf


def test_thread_plan_gpu_gives_lgbm_full_budget():
    orch = pf.OrchConfig(max_workers=10, lgbm_threads=-1)
    cat, lgbm = pf.thread_plan(orch, gpu=True)
    # GPU present: LightGBM owns the CPU budget, CatBoost gets a token share.
    assert lgbm == 10
    assert 1 <= cat <= 10


def test_thread_plan_cpu_does_not_oversubscribe():
    orch = pf.OrchConfig(max_workers=10, lgbm_threads=-1)
    cat, lgbm = pf.thread_plan(orch, gpu=False)
    # Both on CPU: combined demand must stay within the worker ceiling.
    assert cat >= 1 and lgbm >= 1
    assert cat + lgbm <= orch.max_workers


def test_thread_plan_respects_explicit_lgbm_threads():
    orch = pf.OrchConfig(max_workers=10, lgbm_threads=3)
    _, lgbm = pf.thread_plan(orch, gpu=True)
    assert lgbm == 3


def test_thread_plan_min_one_worker():
    orch = pf.OrchConfig(max_workers=1, lgbm_threads=-1)
    cat, lgbm = pf.thread_plan(orch, gpu=False)
    assert cat >= 1 and lgbm >= 1


def test_is_fresh(tmp_path):
    inp = tmp_path / "in.txt"
    out = tmp_path / "out.txt"
    inp.write_text("x")
    time.sleep(0.01)
    out.write_text("y")
    assert pf._is_fresh(out, inp) is True

    # Touch the input so it is newer than the output → stale.
    time.sleep(0.01)
    inp.write_text("x2")
    assert pf._is_fresh(out, inp) is False


def test_is_fresh_missing_output(tmp_path):
    assert pf._is_fresh(tmp_path / "nope.txt", tmp_path / "in.txt") is False


def test_load_orch_config_defaults():
    cfg = pf.load_orch_config()
    assert cfg.max_workers >= 1
    assert isinstance(cfg.lgbm_threads, int)


def test_subprocess_env_sets_omp():
    env = pf._subprocess_env(7)
    assert env["OMP_NUM_THREADS"] == "7"
    assert str(pf.ROOT) in env["PYTHONPATH"]
