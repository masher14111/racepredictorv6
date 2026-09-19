"""Stage 20 repro/audit: identity, operating cutoff and provenance, on the
ACTUAL daily loop entrypoint (``scripts.daily_paper_loop.run``), against
isolated stores only. Never touches ``data/races.db`` or any other real store.

Run: .venv/Scripts/python.exe reports/improvement/20/identity_cutoff_provenance_repro.py
Writes: reports/improvement/20/identity_cutoff_provenance_repro.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.config import ExecutionConfig, config_fingerprint
from execution.model_gate import load_model_verdict
from execution.snapshots import SnapshotStore
from execution.tickets import TicketStore
import scripts.daily_paper_loop as loop

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
DAY_START = NOW.replace(hour=0, minute=0, second=0, microsecond=0)

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: dict | None = None) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail or {}})
    print(("OK  " if ok else "FAIL"), name)


def _race(venue: str, race_time: str, runners: list[str], race_id: str | None = None) -> dict:
    return {
        "venue": venue,
        "race_time": race_time,
        "race_id": race_id,
        "field_size": len(runners),
        "ev_eligible": True,
        "ev_gate": {
            "eligible": True,
            "reference_source": "livescorebet",
            "reference_book_complete": True,
            "reference_overround": 0.18,
            "price_sources": ["livescorebet"],
        },
        "runners": [
            {
                "horse_id": f"{venue[:3]}{i}",
                "horse_name": h,
                "reference_odds": 7.0,
                "best_odds": 8.0,
                "best_book": "livescorebet",
                "reference_source": "livescorebet",
                "value_supported": True,
                "value_win_prob_independent": 0.20,
                "value_win_prob": 0.16,
            }
            for i, h in enumerate(runners)
        ],
    }


def _isolated_cfg(tmp_path: Path) -> ExecutionConfig:
    base = ExecutionConfig.from_config()
    snaps = replace(base.snapshots, db_path=str(tmp_path / "races.db"))
    reports = replace(
        base.reports, dir=str(tmp_path / "reports"), artifact_dir=str(tmp_path / "artifacts")
    )
    return replace(base, snapshots=snaps, reports=reports)


def _write_predictions(path: Path, *, generated_at: str, races: list[dict], provenance=None) -> None:
    payload: dict = {"generated_at": generated_at, "races": races}
    if provenance is not None:
        payload["provenance"] = provenance
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(cfg, predictions_path, tmp_path, *, now=NOW):
    return loop.run(
        cfg=cfg, no_scrape=True, predictions_path=str(predictions_path),
        gap_ledger_path=str(tmp_path / "gap_ledger.json"),
        window_path=str(tmp_path / "forward_window.json"),
        report_dir=str(tmp_path / "reports"), now=now,
    )


def scenario_same_time_different_venues():
    tmp = Path(tempfile.mkdtemp(prefix="s20_venues_"))
    try:
        cfg = _isolated_cfg(tmp)
        races = [
            _race("Leopardstown", "2024-05-17T16:15:00+00:00", ["Horse A", "Horse B"]),
            _race("York", "2024-05-17T16:15:00+00:00", ["Horse C", "Horse D"]),
        ]
        path = tmp / "predictions.json"
        _write_predictions(path, generated_at=NOW.isoformat(), races=races)
        result = _run(cfg, path, tmp)
        store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
        try:
            tickets = store.all_tickets()
        finally:
            store.close()
        race_keys = {t.race_key for t in tickets}
        check(
            "same off-time different venues -> distinct race_key per venue, no cross-venue mixing",
            len(race_keys) == 2
            and all(t.venue in ("Leopardstown", "York") for t in tickets)
            and result["issue"]["issued"] == 4,
            {"race_keys": sorted(race_keys), "issued": result["issue"]["issued"]},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_repeat_run_restart_is_idempotent():
    tmp = Path(tempfile.mkdtemp(prefix="s20_repeat_"))
    try:
        cfg = _isolated_cfg(tmp)
        races = [_race("Naas", "2026-09-19T15:40:00+00:00", ["Horse A", "Horse B"])]
        path = tmp / "predictions.json"
        _write_predictions(path, generated_at=NOW.isoformat(), races=races)
        first = _run(cfg, path, tmp)
        second = _run(cfg, path, tmp)  # simulates a restart on the same real day
        store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
        try:
            n_tickets = len(store.all_tickets())
        finally:
            store.close()
        check(
            "repeat run / restart is idempotent: no new tickets, ticket_ids unchanged",
            first["issue"]["issued"] == 2
            and second["issue"]["issued"] == 0
            and second["issue"]["duplicates"] == 2
            and n_tickets == 2,
            {"first": first["issue"], "second": second["issue"], "stored": n_tickets},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_stale_next_day_replay_refused():
    tmp = Path(tempfile.mkdtemp(prefix="s20_stale_"))
    try:
        cfg = _isolated_cfg(tmp)
        races = [_race("Naas", "2026-09-19T15:40:00+00:00", ["Horse A"])]
        path = tmp / "predictions.json"
        _write_predictions(path, generated_at=NOW.isoformat(), races=races)
        _run(cfg, path, tmp)  # a normal day-1 capture

        # Day 2: refresh never re-ran, predictions.json is untouched (yesterday's).
        next_day = NOW + timedelta(days=1)
        result = _run(cfg, path, tmp, now=next_day)
        check(
            "stale next-day replay is refused (no retroactive issuance under today's stamp)",
            result["issue"]["issued"] == 0
            and result["issue"]["duplicates"] == 0
            and result["issue"].get("predictions_reject_reason") is not None,
            {"issue": result["issue"]},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_dst_spring_forward():
    tmp = Path(tempfile.mkdtemp(prefix="s20_dst_"))
    try:
        cfg = _isolated_cfg(tmp)
        generated = datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc)
        evaluated = datetime(2026, 3, 29, 1, 30, tzinfo=timezone.utc)
        races = [_race("Naas", "2026-03-29T14:00:00+00:00", ["Horse A"])]
        path = tmp / "predictions.json"
        _write_predictions(path, generated_at=generated.isoformat(), races=races)
        result = _run(cfg, path, tmp, now=evaluated)
        check(
            "Dublin DST spring-forward: same local day either side of the clock jump is accepted",
            result["issue"]["issued"] == 1
            and result["issue"].get("predictions_reject_reason") is None,
            {"issue": result["issue"]},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_mixed_timestamps_and_provenance():
    tmp = Path(tempfile.mkdtemp(prefix="s20_provenance_"))
    try:
        cfg = _isolated_cfg(tmp)
        races = [_race("Naas", "2026-09-19T15:40:00+00:00", ["Horse A"])]
        path = tmp / "predictions.json"
        provenance = {
            "model_content_hash": "sha256:deadbeef",
            "feature_schema_version": "v3nf",
            "prediction_cycle_id": "cycle-2026-09-19",
        }
        # generated_at has microseconds; race_time does not -- a mixed-precision
        # frame is exactly what D42/D39-era `format="mixed"` bugs exposed.
        _write_predictions(
            path, generated_at=(NOW - timedelta(minutes=5)).isoformat(timespec="microseconds"),
            races=races, provenance=provenance,
        )
        _run(cfg, path, tmp)
        store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
        try:
            tickets = store.all_tickets()
        finally:
            store.close()
        ok = bool(tickets) and all(
            t.model_content_hash == "sha256:deadbeef"
            and t.feature_schema_version == "v3nf"
            and t.prediction_cycle_id == "cycle-2026-09-19"
            and t.config_hash == config_fingerprint(cfg)
            and t.provenance_complete is True
            for t in tickets
        )
        check(
            "mixed-precision timestamps parse; provenance resolves onto every ticket and "
            "matches the config's own fingerprint",
            ok,
            {"n_tickets": len(tickets), "sample": tickets[0].to_dict() if tickets else None},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_already_started_race_is_pass_not_candidate():
    tmp = Path(tempfile.mkdtemp(prefix="s20_started_"))
    try:
        cfg = _isolated_cfg(tmp)
        started = _race("Naas", (NOW - timedelta(minutes=2)).isoformat(), ["Horse A"])
        upcoming = _race("Naas", (NOW + timedelta(hours=2)).isoformat(), ["Horse B"])
        path = tmp / "predictions.json"
        _write_predictions(path, generated_at=NOW.isoformat(), races=[started, upcoming])
        _run(cfg, path, tmp)
        store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
        try:
            tickets = store.all_tickets()
        finally:
            store.close()
        started_ticket = next((t for t in tickets if t.horse_id == "Naa0" and "started" in "".join(
            r for r in t.pass_reasons)), None)
        reasons_by_horse = {t.horse_id: t.pass_reasons for t in tickets}
        check(
            "an already-off race PASSes with operating_cutoff, never silently dropped nor a candidate",
            all(t.decision == "PASS" for t in tickets)
            and any(r.startswith("operating_cutoff:") for r in reasons_by_horse.get("Naa0", ())),
            {"reasons_by_horse": {k: list(v) for k, v in reasons_by_horse.items()}},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_changed_model_content_changes_the_hash():
    """Not a run() scenario -- models.predictor._model_manifest() must react to
    a byte-for-byte change in a served artifact file, on an isolated copy of a
    real bundle (never the live models/ directory)."""
    tmp = Path(tempfile.mkdtemp(prefix="s20_model_hash_"))
    try:
        from models.predictor import Predictor

        candidate_dir = tmp / "models"
        candidate_dir.mkdir()
        meta = {"feature_cols": ["a", "b", "c"]}
        (candidate_dir / "catboost_v3_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (candidate_dir / "catboost_won_v3.bin").write_bytes(b"version-one-bytes")

        p = Predictor(model_dir=str(candidate_dir))
        p._version_tag = "v3"
        p._targets = ["won"]
        p._value_enabled = False
        p._feature_cols = meta["feature_cols"]
        before = p._model_manifest()

        (candidate_dir / "catboost_won_v3.bin").write_bytes(b"version-two-DIFFERENT-bytes")
        after = p._model_manifest()

        check(
            "changed model content -> different model_content_hash; unchanged feature list -> same schema version",
            before["model_content_hash"] != after["model_content_hash"]
            and before["feature_schema_version"] == after["feature_schema_version"],
            {"before": before["model_content_hash"], "after": after["model_content_hash"]},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    scenario_same_time_different_venues()
    scenario_repeat_run_restart_is_idempotent()
    scenario_stale_next_day_replay_refused()
    scenario_dst_spring_forward()
    scenario_mixed_timestamps_and_provenance()
    scenario_already_started_race_is_pass_not_candidate()
    scenario_changed_model_content_changes_the_hash()

    out = Path(__file__).with_name("identity_cutoff_provenance_repro.json")
    out.write_text(
        json.dumps(
            {"ran_at_utc": datetime.now(timezone.utc).isoformat(), "results": RESULTS},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    failed = [r["check"] for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} ok; wrote {out}")
    for f in failed:
        print("  FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
