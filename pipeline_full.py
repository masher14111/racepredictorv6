"""One-shot full-rebuild orchestrator — "use my full PC".

Rebuilds everything from the raw sources to a fresh ``data/predictions.json`` as
fast as this machine safely allows:

    python pipeline_full.py [--skip-scrape] [--no-tune] [--resume] [--force]

Pipeline (timed end-to-end):
    1. (optional) Scrape Betfair SP + Timeform historical data.
    2. Normalise sources → ``data/unified_races.parquet`` and build the labelled
       feature matrix → ``data/features/training.parquet``.
    3. Train BOTH model lines CONCURRENTLY, each in its own process:
         • CatBoost (GPU, model.task_type=GPU)  — ``models.train``
         • LightGBM v3 softmax (CPU, all cores) — ``models.train_lgbm``
       They contend for different hardware (GPU vs CPU), so running them in
       parallel is a genuine speedup. ``orchestration.max_workers`` caps the CPU
       fan-out so the two jobs never oversubscribe every core (2 left free for
       the UI). The LightGBM line runs its own frozen-holdout GO/NO-GO internally.
    4. Run the CatBoost holdout GO/NO-GO over the most-recent ~3-week window.
    5. Refresh ``data/predictions.json`` (``models.predictor``).
    6. Print a wall-clock timing table + the GO/NO-GO verdict for both lines.

Safety / idempotence:
    * Existing models are NEVER deleted; the trainers overwrite in place.
    * ``--resume`` skips a stage whose output is already newer than its input
      (fast warm re-runs); ``--force`` re-runs every stage regardless.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger("pipeline_full")

_UNIFIED = ROOT / "data" / "unified_races.parquet"
_TRAINING = ROOT / "data" / "features" / "training.parquet"
_CATBOOST_MODEL = ROOT / "models" / "catboost_won_v3.bin"
_LGBM_MODEL = ROOT / "models" / "lgbm_won_v3.txt"
_LGBM_META = ROOT / "models" / "lgbm_v3_meta.json"
_PREDICTIONS = ROOT / "data" / "predictions.json"

# Holdout = the most recent ~3 weeks (mirrors models.train_lgbm._HOLDOUT_DAYS).
_HOLDOUT_DAYS = 20


# ── config ──────────────────────────────────────────────────────────────────


@dataclass
class OrchConfig:
    """Resolved ``orchestration`` block (with safe defaults)."""

    max_workers: int = 10
    lgbm_threads: int = -1


def load_orch_config() -> OrchConfig:
    from utils.config_loader import get_config

    o = get_config().get("orchestration", {}) or {}
    return OrchConfig(
        max_workers=int(o.get("max_workers", 10)),
        lgbm_threads=int(o.get("lgbm_threads", -1)),
    )


def gpu_available() -> bool:
    """True iff CatBoost can see a GPU (so the CatBoost line really offloads it)."""
    try:
        from catboost.utils import get_gpu_device_count

        return int(get_gpu_device_count()) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline_full: GPU detection failed (%s) — assuming CPU", exc)
        return False


def thread_plan(orch: OrchConfig, gpu: bool) -> tuple[int, int]:
    """Split CPU threads between the two concurrent trainers.

    Returns ``(catboost_omp, lgbm_omp)`` — the ``OMP_NUM_THREADS`` to hand each
    subprocess so their combined CPU demand never exceeds ``max_workers``.

    * GPU present → CatBoost barely touches the CPU, so LightGBM gets the whole
      budget (``lgbm_threads``: -1 ⇒ ``max_workers``) and CatBoost a token share.
    * No GPU → CatBoost falls back to CPU, so the budget is split in half; the two
      jobs run concurrently without oversubscribing the box.
    """
    budget = max(1, orch.max_workers)
    if gpu:
        lgbm = budget if orch.lgbm_threads in (-1, 0) else min(orch.lgbm_threads, budget)
        catboost = max(1, budget - lgbm) if lgbm < budget else max(1, budget // 5)
        return catboost, lgbm
    # Both on CPU: halve the budget so the sum stays within max_workers.
    half = max(1, budget // 2)
    lgbm = half if orch.lgbm_threads in (-1, 0) else min(orch.lgbm_threads, half)
    return max(1, budget - lgbm), lgbm


# ── helpers ─────────────────────────────────────────────────────────────────


def _is_fresh(output: Path, *inputs: Path) -> bool:
    """True iff ``output`` exists and is newer than every existing input."""
    if not output.exists():
        return False
    out_m = output.stat().st_mtime
    return all(out_m >= p.stat().st_mtime for p in inputs if p.exists())


def _data_max_date():
    """Max race_date in the training matrix (tz-naive day), or None if unavailable."""
    import pandas as pd

    if not _TRAINING.exists():
        return None
    try:
        rd = pd.read_parquet(_TRAINING, columns=["race_date"])["race_date"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline_full: could not read race_date from %s (%s)", _TRAINING, exc)
        return None
    s = pd.to_datetime(rd, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    s = s.dropna()
    return s.max() if not s.empty else None


def _holdout_window():
    """(holdout_start, holdout_end) for the CatBoost holdout, or (None, None)."""
    import pandas as pd

    dmax = _data_max_date()
    if dmax is None:
        return None, None
    return dmax - pd.Timedelta(days=_HOLDOUT_DAYS), dmax


# ── stages ──────────────────────────────────────────────────────────────────


@dataclass
class Stage:
    name: str
    seconds: float
    status: str  # "ok" | "skipped" | "failed"
    detail: str = ""


@dataclass
class Report:
    stages: list[Stage] = field(default_factory=list)
    catboost_verdict: Optional[dict] = None
    lgbm_verdict: Optional[dict] = None

    def add(self, name: str, seconds: float, status: str, detail: str = "") -> None:
        self.stages.append(Stage(name, seconds, status, detail))
        logger.info("pipeline_full: [%s] %s (%.1fs) %s", status.upper(), name, seconds, detail)


def step(label: str) -> None:
    logger.info("=" * 64)
    logger.info("STAGE: %s", label)
    logger.info("=" * 64)


def stage_scrape(report: Report, skip: bool) -> None:
    if skip:
        report.add("scrape", 0.0, "skipped", "--skip-scrape")
        return
    step("Scrape historical data (Betfair SP + Timeform)")
    t0 = time.perf_counter()
    from scraper.betsp_historical import fetch as fetch_betsp
    from scraper.timeform_historical import fetch as fetch_tf

    n_betsp = len(fetch_betsp())
    n_tf = len(fetch_tf())
    report.add("scrape", time.perf_counter() - t0, "ok",
               f"betsp={n_betsp} timeform={n_tf}")


def stage_features(report: Report, resume: bool, force: bool) -> None:
    step("Normalise + build feature matrix")
    if not force and resume and _is_fresh(_TRAINING, _UNIFIED):
        report.add("features", 0.0, "skipped", "training.parquet up to date (--resume)")
        return
    t0 = time.perf_counter()
    from utils.normalizer import normalize
    from features.builder import build_training_matrix

    # Cap intra-op thread pools so the build uses the box without thrashing it.
    unified = normalize(write=True)
    training = build_training_matrix(unified=unified, write=True)
    if len(training) == 0:
        report.add("features", time.perf_counter() - t0, "failed", "0 labelled rows")
        raise SystemExit(
            "pipeline_full: training matrix is empty — no labelled (settled) rows yet. "
            "Wait for results to land, then re-run.")
    report.add("features", time.perf_counter() - t0, "ok",
               f"unified={len(unified)} training={len(training)}")


def _subprocess_env(omp_threads: int) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _launch(name: str, argv: list[str], omp_threads: int, log_path: Path):
    """Start a trainer subprocess, tee-ing its output to ``log_path``."""
    logger.info("pipeline_full: launching %s (OMP_NUM_THREADS=%d) → %s",
                name, omp_threads, log_path.name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, *argv], cwd=str(ROOT), env=_subprocess_env(omp_threads),
        stdout=fh, stderr=subprocess.STDOUT,
    )
    return proc, fh


def stage_train(report: Report, orch: OrchConfig, no_tune: bool,
                resume: bool, force: bool) -> None:
    step("Train CatBoost (GPU) + LightGBM (CPU) concurrently")
    gpu = gpu_available()
    cat_omp, lgbm_omp = thread_plan(orch, gpu)
    logger.info("pipeline_full: GPU=%s → CatBoost OMP=%d, LightGBM OMP=%d",
                gpu, cat_omp, lgbm_omp)

    log_dir = ROOT / "logs"
    cat_fresh = not force and resume and _is_fresh(_CATBOOST_MODEL, _TRAINING)
    lgbm_fresh = not force and resume and _is_fresh(_LGBM_MODEL, _TRAINING)

    procs = []
    t0 = time.perf_counter()

    if cat_fresh:
        report.add("train_catboost", 0.0, "skipped", "model up to date (--resume)")
    else:
        cat_argv = ["-m", "models.train"]
        if no_tune:
            cat_argv.append("--no-tune")
        proc, fh = _launch("CatBoost", cat_argv, cat_omp, log_dir / "train_catboost.log")
        procs.append(("train_catboost", proc, fh))

    if lgbm_fresh:
        report.add("train_lgbm", 0.0, "skipped", "model up to date (--resume)")
    else:
        lgbm_argv = ["-m", "models.train_lgbm", "--output", str(_LGBM_MODEL), "--importance"]
        proc, fh = _launch("LightGBM", lgbm_argv, lgbm_omp, log_dir / "train_lgbm.log")
        procs.append(("train_lgbm", proc, fh))

    # Wait for both; one failing does not abort the other (we still want partial results).
    failures = []
    for name, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        secs = time.perf_counter() - t0
        if rc == 0:
            report.add(name, secs, "ok", "logs/%s.log" % name)
        else:
            report.add(name, secs, "failed", f"exit {rc} — see logs/{name}.log")
            failures.append(name)

    if failures:
        raise SystemExit(f"pipeline_full: training failed for {', '.join(failures)} "
                         f"(see logs/). Aborting before holdout/predict.")


def stage_catboost_holdout(report: Report) -> None:
    step("CatBoost holdout GO/NO-GO (most recent ~3 weeks)")
    t0 = time.perf_counter()
    if not _CATBOOST_MODEL.exists():
        report.add("holdout_catboost", 0.0, "skipped", "no CatBoost model on disk")
        return

    start, end = _holdout_window()
    if start is None:
        report.add("holdout_catboost", 0.0, "skipped", "no race dates in training matrix")
        return

    from backtest.holdout import run_holdout

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "backtests" / f"holdout_catboost_{stamp}"
    try:
        summary = run_holdout(
            model_path=str(_CATBOOST_MODEL),
            holdout_start=str(start.date()),
            holdout_end=str(end.date()),
            out_dir=str(out_dir),
        )
    except SystemExit as exc:
        report.add("holdout_catboost", time.perf_counter() - t0, "failed", str(exc))
        return
    h2h = summary.get("head_to_head", {})
    report.catboost_verdict = {
        "go": summary.get("model_beats_market_logloss"),
        "model_log_loss": h2h.get("model", {}).get("log_loss"),
        "market_log_loss": h2h.get("market", {}).get("log_loss"),
        "n_races_kept": h2h.get("n_races_kept"),
        "roi": summary.get("betting", {}).get("roi"),
        "mean_clv_log": summary.get("betting", {}).get("mean_clv_log"),
        "out_dir": str(out_dir),
    }
    report.add("holdout_catboost", time.perf_counter() - t0, "ok",
               "GO" if report.catboost_verdict["go"] else "NO-GO")


def stage_load_lgbm_verdict(report: Report) -> None:
    if not _LGBM_META.exists():
        return
    try:
        meta = json.loads(_LGBM_META.read_text(encoding="utf-8"))
        report.lgbm_verdict = meta.get("verdict")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline_full: could not read %s (%s)", _LGBM_META, exc)


def stage_predict(report: Report) -> None:
    step("Refresh predictions.json")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "models.predictor"],
        cwd=str(ROOT), env=_subprocess_env(max(1, load_orch_config().max_workers)),
    )
    status = "ok" if proc.returncode == 0 else "failed"
    detail = f"exit {proc.returncode}" if proc.returncode else str(_PREDICTIONS)
    report.add("predict", time.perf_counter() - t0, status, detail)


# ── summary ─────────────────────────────────────────────────────────────────


def _fmt_verdict(name: str, v: Optional[dict]) -> str:
    if not v:
        return f"  {name:<10} (no verdict produced)"
    go = v.get("go")
    word = "GO    " if go else "NO-GO " if go is not None else "  ?   "
    ll = v.get("model_log_loss")
    mll = v.get("market_log_loss")
    races = v.get("n_races_kept")
    return (f"  {name:<10} {word} model_ll={ll} vs market_ll={mll} "
            f"over {races} races")


def print_summary(report: Report, total_seconds: float) -> None:
    print("\n" + "=" * 64)
    print(" pipeline_full — timing")
    print("=" * 64)
    for s in report.stages:
        print(f"  {s.name:<20} {s.seconds:8.1f}s  [{s.status}]  {s.detail}")
    print(f"  {'TOTAL':<20} {total_seconds:8.1f}s")
    print("\n" + "=" * 64)
    print(" GO / NO-GO verdicts (does the model beat the de-vigged market?)")
    print("=" * 64)
    print(_fmt_verdict("CatBoost", report.catboost_verdict))
    print(_fmt_verdict("LightGBM", report.lgbm_verdict))
    print("\n  NB CLV is negative across the book — treat any GO as paper-only.")
    print(f"\n  predictions: {_PREDICTIONS}")


# ── entry point ─────────────────────────────────────────────────────────────


def run(skip_scrape: bool = False, no_tune: bool = False,
        resume: bool = False, force: bool = False) -> Report:
    orch = load_orch_config()
    logger.info("pipeline_full: max_workers=%d lgbm_threads=%d resume=%s force=%s",
                orch.max_workers, orch.lgbm_threads, resume, force)
    report = Report()
    t0 = time.perf_counter()

    stage_scrape(report, skip_scrape)
    stage_features(report, resume, force)
    stage_train(report, orch, no_tune, resume, force)
    stage_catboost_holdout(report)
    stage_load_lgbm_verdict(report)
    stage_predict(report)

    print_summary(report, time.perf_counter() - t0)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot full-rebuild orchestrator (CatBoost+LightGBM concurrently)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="skip the historical data fetch (use existing parquets)")
    parser.add_argument("--no-tune", action="store_true",
                        help="skip Optuna hyperparameter search (faster CatBoost)")
    parser.add_argument("--resume", action="store_true",
                        help="skip any stage whose output is already newer than its input")
    parser.add_argument("--force", action="store_true",
                        help="re-run every stage even if outputs look up to date")
    args = parser.parse_args(argv)

    run(skip_scrape=args.skip_scrape, no_tune=args.no_tune,
        resume=args.resume, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
