"""Stage 5 end-to-end: selection, walk-forward backtest, forward gate, reports.

One deterministic command that produces the two dated reports Stage 5 owes:

    python -m scripts.paper_betting            # full run
    python -m scripts.paper_betting --candidates-only

The order below is the protocol, not a convenience:

1. **Read the model verdict first.** Everything downstream is conditioned on it.
   Stage 4 says NO-GO, so nothing here can produce a real-money recommendation —
   and the run still happens, because the mechanics have to be exercised and the
   forward window has to start accumulating evidence *before* any verdict changes.
2. **Split chronologically, tune on train, select once, score the test window
   once.** ``execution.selection`` owns the lock that makes "once" enforceable.
3. **Tune on CLV, not ROI.** ROI over a few hundred bets is mostly the variance of
   a handful of long-priced winners; searching a 64-cell grid for the best ROI is
   exactly the threshold mining requirement 4 forbids. CLV is the repo's own
   leading indicator, it is far less noisy, and it cannot be inflated by luck in
   settlement. ROI, yield, A/E and drawdown are *reported* on the untouched test
   window — as outcomes, never as the search target.
4. **Run the friction simulator once per strategy**, not once per grid cell.
   Frictions (latency, rejection, suspension, Rule 4, commission) do not depend on
   which edge threshold a strategy uses; they are applied to whatever it picks. So
   the grid is scored on the fast frictionless selection path and the winner plus
   the three baselines go through the full walk. The report says so out loud.

Nothing in this script can stake real money: every ticket it writes is
``paper_only=1``, enforced by a CHECK constraint in the ticket table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution import baselines, evaluation, report as reporting, selection, simulator
from execution.config import ExecutionConfig, load_execution_config
from execution.forward_gate import evaluate_forward_gate
from execution.model_gate import load_model_verdict
from execution.race_facts import facts_from_frame, load_race_facts
from execution.snapshots import SnapshotStore
from execution.tickets import TicketStore
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ARTIFACT_DIR = os.path.join("data", "execution")
DEFAULT_BACKTEST_DB = os.path.join(DEFAULT_ARTIFACT_DIR, "backtest_snapshots.db")


# ── the selection objective ──────────────────────────────────────────────────
def _apply_candidate(cfg: ExecutionConfig, candidate) -> ExecutionConfig:
    """A config whose gates are the candidate's thresholds."""
    p = candidate.params
    return replace(
        cfg,
        gates=replace(
            cfg.gates,
            min_edge=float(p["min_edge"]),
            min_expected_value=float(p["min_expected_value"]),
        ),
    )


def _band_of(candidate) -> Optional[tuple]:
    """The candidate's ``(odds_min, odds_max)``, or ``None`` for the whole book.

    Handed to :func:`baselines.model_only_selection` as a *backable-runner*
    restriction rather than applied to the frame here. Filtering the frame would
    de-vig a field with its favourites removed and would change the eligible race
    set per strategy — see that function's docstring for what that broke.
    """
    lo = candidate.params.get("odds_min")
    hi = candidate.params.get("odds_max")
    return None if lo is None and hi is None else (lo, hi)


def _selected_run_config(
    cfg: ExecutionConfig, sel_result
) -> tuple[ExecutionConfig, Optional[tuple]]:
    """``(cfg, odds_band)`` for the friction walk: the selected strategy, or defaults.

    ``sel_result`` is ``None`` when the test window was already spent and the
    protocol refused to re-score it. There is no winner to apply in that case, so
    the walk runs the config defaults and the report labels it as such — better
    than silently reviving a strategy the protocol declined to re-certify.
    """
    selected = getattr(sel_result, "selected", None)
    if selected is None:
        return cfg, None
    return _apply_candidate(cfg, selected), _band_of(selected)


def _walk_config(cfg: ExecutionConfig, band: Optional[tuple], sel_result) -> dict:
    """What the friction walk actually ran, so the numbers can be attributed."""
    selected = getattr(sel_result, "selected", None)
    return {
        "source": "selected strategy" if selected is not None else "config defaults",
        "strategy": getattr(selected, "name", None),
        "min_edge": float(cfg.gates.min_edge),
        "min_expected_value": float(cfg.gates.min_expected_value),
        "odds_band": list(band) if band else None,
        "band_applies_to": "model_only only — baselines bet the whole book",
    }


def _superseded_locks(cfg: ExecutionConfig) -> list[dict]:
    """Earlier selection locks archived beside the live one.

    A reset lock means the untouched test window was scored again, and the
    multiple-testing correction on the current lock does not know that. Rather
    than trusting whoever performs a reset to remember the disclosure, the
    convention is mechanical: archive the old lock as
    ``<stem>_superseded_<tag>.json`` next to it and this picks it up, so the
    report cannot render without naming every extra draw.

    Deleting the archived file would remove the disclosure — which is exactly why
    the reset is archived rather than overwritten.
    """
    live = cfg.selection.lock_file
    stem = os.path.splitext(os.path.basename(live))[0]
    directory = os.path.dirname(live) or "."
    out: list[dict] = []
    for name in sorted(os.listdir(directory) if os.path.isdir(directory) else []):
        if not (name.startswith(f"{stem}_superseded") and name.endswith(".json")):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            payload = {}
        out.append({
            "path": path,
            "selected": payload.get("selected"),
            "test_score": payload.get("test_score"),
            "test_scored_at": payload.get("test_scored_at"),
            "reason": payload.get("superseded_reason"),
        })
    if out:
        logger.warning(
            "selection: %d superseded lock(s) found — the test window has been "
            "scored more than once; the report will say so", len(out)
        )
    return out


def _selection_payload(cfg: ExecutionConfig, sel_result) -> Optional[dict]:
    """The lock as the report and the run artifact both see it, resets included."""
    superseded = _superseded_locks(cfg)
    if sel_result is None:
        return {"superseded_locks": superseded} if superseded else None
    return {**sel_result.to_dict(), "superseded_locks": superseded}


MIN_OBJECTIVE_BETS = 30


def clv_objective(cfg: ExecutionConfig, *, min_bets: int = MIN_OBJECTIVE_BETS):
    """Build the ``evaluate_fn`` ``run_selection`` scores candidates with.

    The score is the mean log CLV of the picks the candidate would make. A
    candidate striking fewer than ``min_bets`` scores ``-inf``: without a floor,
    the grid reliably selects the rarest strategy, because two lucky bets produce
    a better mean than four hundred honest ones. That is the purest form of the
    overfitting this protocol exists to prevent.

    This path is deliberately frictionless. Latency, rejection, Rule 4 and
    commission do not depend on which edge threshold a strategy uses — they apply
    to whatever it picks — so running 64 candidates through the full simulator
    would cost half an hour to rank them in the same order. The winner and the
    baselines *are* run through the full walk, and the report says which numbers
    came from which path.
    """

    def evaluate(frame: pd.DataFrame, candidate) -> float:
        picks = baselines.model_only_selection(
            frame,
            cfg=_apply_candidate(cfg, candidate),
            odds_band=_band_of(candidate),
        )
        if len(picks) < int(min_bets):
            return float("-inf")
        odds = pd.to_numeric(picks["decimal_odds"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(picks["closing_odds"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            clv = np.log(np.where((odds > 1.0) & (close > 1.0), odds / close, np.nan))
        clv = clv[np.isfinite(clv)]
        return float(clv.mean()) if clv.size else float("-inf")

    return evaluate


# ── the run ──────────────────────────────────────────────────────────────────
def _summaries(results: dict) -> dict:
    return {name: res.summary for name, res in results.items()}


def _metrics(results: dict, *, cfg: ExecutionConfig, bankroll: float) -> dict:
    out = {}
    for name, res in results.items():
        out[name] = evaluation.evaluate_ledger(
            res.ledger,
            name=name,
            initial_bankroll=bankroll,
            n_boot=int(cfg.selection.bootstrap_resamples),
            seed=int(cfg.selection.seed),
        )
    return out


def _paper_ledger(store: TicketStore) -> pd.DataFrame:
    """Settled paper tickets as a ledger the forward gate can read."""
    frame = store.to_frame(decision="CANDIDATE")
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    return frame


def _paper_summary(store: TicketStore) -> dict:
    """Decisions logged vs. wagers -- kept visibly separate.

    See ``scripts.daily_paper_loop._paper_summary`` (same fix, same reason):
    ``n_open``/``n_settled`` must be scoped to CANDIDATE decisions, not every
    disclosed decision -- a PASS carries no stake and was never a bet.
    """
    frame = store.to_frame()
    empty = {
        "n_tickets": 0, "n_pass": 0, "n_candidates": 0,
        "n_open": 0, "n_settled": 0,
        "first_ticket": None, "last_ticket": None,
    }
    if frame is None or len(frame) == 0:
        return empty
    is_candidate = (
        frame["decision"].eq("CANDIDATE") if "decision" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    candidates = frame[is_candidate]
    settled = (
        candidates["outcome"].notna() if "outcome" in candidates.columns
        else pd.Series(False, index=candidates.index)
    )
    issued = frame["issued_at"] if "issued_at" in frame.columns else pd.Series(dtype=object)
    return {
        "n_tickets": int(len(frame)),
        "n_pass": int((~is_candidate).sum()),
        "n_candidates": int(is_candidate.sum()),
        "n_settled": int(settled.sum()),
        "n_open": int((~settled).sum()),
        "first_ticket": str(issued.min()) if len(issued) else None,
        "last_ticket": str(issued.max()) if len(issued) else None,
    }


def run(
    *,
    cfg: Optional[ExecutionConfig] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    bankroll: float = 1000.0,
    snapshot_db: str = DEFAULT_BACKTEST_DB,
    report_dir: Optional[str] = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    force_selection: bool = False,
    skip_backtest: bool = False,
    now: Optional[datetime] = None,
) -> dict:
    """Run the whole Stage-5 pipeline and write both reports. Returns a summary."""
    cfg = cfg or load_execution_config()
    report_dir = report_dir or cfg.reports.dir
    stamp = now or datetime.now(tz=timezone.utc)

    # 1. The verdict gates everything below it.
    verdict = load_model_verdict()
    logger.info("model gate: %s (%s)", verdict.verdict_label, verdict.headline)

    panel = pd.DataFrame()
    sel_result = None
    backtest_metrics: dict = {}
    backtest_summaries: dict = {}
    walk_cfg, walk_band = cfg, None

    if not skip_backtest:
        panel = simulator.load_scored_panel(start=start, end=end)
        logger.info("panel: %d runners over %d races",
                    len(panel), panel["race_uid"].nunique())

        # 2 + 3. Select once on CLV; the lock makes "once" mean once.
        train, _validation, test = selection.chronological_split(panel, cfg=cfg)
        try:
            sel_result = selection.run_selection(
                panel,
                cfg=cfg,
                evaluate_fn=clv_objective(cfg),
                force=force_selection,
            )
        except selection.TestWindowAlreadyScored as exc:
            # Not an error. It is the protocol working: the untouched window has
            # been spent, and re-scoring it would turn a held-out number into
            # another tuning signal.
            logger.warning("selection: %s", exc)
            sel_result = None

        # 4. The full friction walk, once per strategy, on the test window only —
        #    the same window the selected strategy was scored on, so the reported
        #    frictions belong to the reported numbers.
        #
        #    The walk runs the strategy that was *selected*, not the config
        #    defaults: reporting a friction walk of min_edge 0.02 under the
        #    heading of a strategy chosen at 0.08 would describe a strategy
        #    nothing selected. With no lock to read, the defaults stand and the
        #    report says so.
        walk_cfg, walk_band = _selected_run_config(cfg, sel_result)
        logger.info("walk: min_edge=%.3f min_ev=%.3f band=%s",
                    walk_cfg.gates.min_edge, walk_cfg.gates.min_expected_value,
                    walk_band or "whole book")
        os.makedirs(os.path.dirname(snapshot_db) or ".", exist_ok=True)
        store = SnapshotStore(snapshot_db)
        try:
            simulator.seed_snapshots(test, store)
            facts = _load_facts(test)
            results = simulator.run_walk_forward(
                test, cfg=walk_cfg, store=store, race_facts=facts,
                bankroll=bankroll, seed=int(cfg.selection.seed),
                odds_band=walk_band,
            )
        finally:
            store.close()
        backtest_metrics = _metrics(results, cfg=cfg, bankroll=bankroll)
        backtest_summaries = _summaries(results)

    # 5. Today's card through the live gate, then the forward gate — which reads
    #    paper tickets, and only paper tickets.
    ticket_store = TicketStore(cfg=cfg)
    try:
        paper = _paper_ledger(ticket_store)
        paper_stats = _paper_summary(ticket_store)
    finally:
        ticket_store.close()

    gate = evaluate_forward_gate(
        paper, cfg=cfg, model_verdict=verdict, evidence_kind="forward", now=stamp
    )
    paper_metrics = {}
    if len(paper):
        paper_metrics = {"paper": evaluation.evaluate_ledger(
            paper, name="paper", initial_bankroll=bankroll,
            n_boot=int(cfg.selection.bootstrap_resamples),
            seed=int(cfg.selection.seed),
        )}

    releasable = bool(verdict.go) and bool(gate.passed)
    deployment = "CANDIDATE-ELIGIBLE" if (releasable and not cfg.paper_only) else "PAPER-ONLY"

    # 6. The reports.
    fv_text = reporting.forward_validation_report(
        model_verdict=verdict,
        forward_gate=gate,
        backtest_metrics=backtest_metrics or None,
        backtest_summaries=backtest_summaries or None,
        paper_metrics=paper_metrics or None,
        paper_summary=paper_stats,
        selection=_selection_payload(cfg, sel_result),
        deployment_state=deployment,
        walk_config=_walk_config(walk_cfg, walk_band, sel_result),
        now=stamp,
    )
    fv_path = reporting.write_report(
        fv_text, name="forward_validation", report_dir=report_dir, now=stamp
    )

    candidates, passes = today_candidates(cfg=cfg, verdict=verdict, gate=gate, now=stamp)
    detail_path = _write_pass_detail(
        candidates + passes, report_dir=report_dir, now=stamp
    )
    cand_text = reporting.candidates_report(
        candidates=candidates,
        passes=passes,
        model_verdict=verdict,
        forward_gate=gate,
        paper_only=cfg.paper_only,
        pass_detail_path=detail_path,
        now=stamp,
    )
    cand_path = reporting.write_report(
        cand_text, name="todays_candidates", report_dir=report_dir, now=stamp
    )

    payload = {
        "generated_at": stamp.isoformat(timespec="seconds"),
        "model_verdict": verdict.to_dict(),
        "forward_gate": gate.to_dict(),
        "deployment": deployment,
        "paper_only": bool(cfg.paper_only),
        "selection": _selection_payload(cfg, sel_result),
        "walk_config": _walk_config(walk_cfg, walk_band, sel_result),
        "backtest": {k: v.to_dict() for k, v in backtest_metrics.items()},
        "backtest_summaries": backtest_summaries,
        "paper": paper_stats,
        "n_candidates": len(candidates),
        "n_passes": len(passes),
        "reports": {"forward_validation": fv_path, "candidates": cand_path},
    }
    reporting.write_json(
        payload, path=os.path.join(artifact_dir, "stage5_run.json")
    )
    return payload


def _write_pass_detail(rows: list[dict], *, report_dir: str, now: datetime) -> Optional[str]:
    """Every runner's full decision record, one row each, as CSV.

    The Markdown report summarises; this file is the receipt. Keeping the whole
    reason text out of the report and *in* a named file is not a truncation — the
    report says where it is, and every runner appears in both.
    """
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    for col in ("pass_reasons", "reasons_passed"):
        if col in frame.columns:
            frame[col] = frame[col].map(
                lambda v: "; ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
            )
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"candidate_decisions_{now.strftime('%Y%m%d')}.csv")
    frame.to_csv(path, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", path, len(frame))
    return path


def _load_facts(panel: pd.DataFrame):
    """Settlement ground truth for the walk, or ``None`` if the archive is absent.

    ``None`` is not silent: the simulator falls back to the panel's own result
    column and records the archive coverage it achieved in its summary, which the
    report prints. Settling a whole window from the panel is weaker evidence (no
    Rule 4, no non-runner detail) and the reader is told so.
    """
    try:
        dates = pd.to_datetime(panel["race_date"], errors="coerce").dropna()
        if dates.empty:
            return None
        frame = load_race_facts(str(dates.min().date()), str(dates.max().date()))
        if frame is None or len(frame) == 0:
            return None
        return facts_from_frame(frame)
    except Exception as exc:  # pragma: no cover - archive shape varies by machine
        logger.warning("race facts unavailable (%s); settling from the panel", exc)
        return None


# ── today's candidates ───────────────────────────────────────────────────────
def today_candidates(
    *,
    cfg: ExecutionConfig,
    verdict,
    gate,
    predictions_path: str = os.path.join("data", "predictions.json"),
    now: Optional[datetime] = None,
) -> tuple[list[dict], list[dict]]:
    """Run today's live card through the candidate gate.

    Returns ``(candidates, passes)``. Under Stage 4's NO-GO the first list is
    empty by construction — ``execution.gates`` refuses to issue a candidate when
    the model verdict is NO-GO, and that refusal is upstream of everything here.
    """
    from execution.gates import evaluate_race
    from execution.tickets import build_ticket

    if not os.path.exists(predictions_path):
        logger.info("no %s; today's candidates report will be empty", predictions_path)
        return [], []

    try:
        with open(predictions_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s (%s)", predictions_path, exc)
        return [], []

    races = payload.get("races") if isinstance(payload, dict) else payload
    if not isinstance(races, list):
        return [], []

    health = _source_health()
    candidates: list[dict] = []
    passes: list[dict] = []
    for race in races:
        if not isinstance(race, dict):
            continue
        try:
            results = evaluate_race(
                race, cfg=cfg, model_verdict=verdict, source_health=health, now=now
            )
        except Exception as exc:  # pragma: no cover - a malformed race is a PASS
            logger.warning("gate could not evaluate a race (%s); treated as PASS", exc)
            continue
        runners = race.get("runners") or race.get("selections") or []
        for runner, result in zip(runners, results):
            row = build_ticket(
                runner, race, result, cfg=cfg, model_verdict=verdict,
                forward_gate_state=gate.state_label, now=now,
            ).to_dict()
            (candidates if result.is_candidate else passes).append(row)
    return candidates, passes


def _source_health() -> dict:
    try:
        from utils.source_health import get_health

        return get_health() or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("source health unavailable (%s); every source reads unknown", exc)
        return {}


# ── cli ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=None, help="panel window start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="panel window end (YYYY-MM-DD)")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--snapshot-db", default=DEFAULT_BACKTEST_DB)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--candidates-only", action="store_true",
                    help="skip the backtest; write today's candidates and the gate")
    ap.add_argument("--force-selection", action="store_true",
                    help="re-score the held-out test window (breaks the lock; "
                         "the report records that it was forced)")
    args = ap.parse_args(argv)

    payload = run(
        start=args.start,
        end=args.end,
        bankroll=args.bankroll,
        snapshot_db=args.snapshot_db,
        report_dir=args.report_dir,
        force_selection=args.force_selection,
        skip_backtest=args.candidates_only,
    )
    print(f"MODEL {payload['model_verdict'].get('verdict')} · "
          f"{payload['forward_gate'].get('state_label')} · "
          f"DEPLOYMENT {payload['deployment']}")
    for name, path in payload["reports"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
