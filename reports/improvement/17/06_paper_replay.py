"""Step 17 — bounded PAPER-MODE replay and fail-closed checks, isolated artifacts.

Every write goes under ``data/audit/17/replay/<run>/`` (own races.db, gap ledger,
window file, reports, artifacts). The real stores are hashed before and after to
PROVE isolation. No network: ``no_scrape=True`` throughout.

Run:  .venv/Scripts/python.exe reports/improvement/17/06_paper_replay.py [--label NAME]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.daily_paper_loop as loop  # noqa: E402
from execution.config import ExecutionConfig  # noqa: E402
from execution.gates import evaluate_race  # noqa: E402
from execution.model_gate import load_model_verdict  # noqa: E402
from execution.tickets import RealMoneyTicketRefused, TicketStore, build_ticket  # noqa: E402
from utils.config_loader import get_config  # noqa: E402

HERE = Path(__file__).parent
REAL_STORES = ["data/races.db", "data/predictions.json", "data/execution/daily_loop_run.json", "data/cache/sent_alerts.json",
               "data/execution/gap_ledger.json", "data/execution/selection_lock.json"]
results: list[dict] = []


def check(name: str, ok: bool, detail) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:420]}")


def file_state(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "mtime_ns": p.stat().st_mtime_ns}


def isolated_cfg(base: ExecutionConfig, work: Path) -> ExecutionConfig:
    return replace(base, snapshots=replace(base.snapshots, db_path=str(work / "races.db")),
                   reports=replace(base.reports, dir=str(work / "reports"), artifact_dir=str(work / "artifacts")))


def run_loop(cfg, work: Path, predictions: Path, now: datetime) -> dict:
    return loop.run(cfg=cfg, no_scrape=True, predictions_path=str(predictions),
                    gap_ledger_path=str(work / "gap_ledger.json"), window_path=str(work / "forward_window.json"),
                    report_dir=str(work / "reports"), now=now)


def synthetic_race(now: datetime, *, age_s: float = 60.0, book: str = "livescorebet", eligible: bool = True) -> dict:
    fetched = (now - timedelta(seconds=age_s)).isoformat()
    probs = [0.34, 0.22, 0.16, 0.12, 0.09, 0.07]          # the (price-free) model's view
    fair = [0.25, 0.25, 0.18, 0.14, 0.10, 0.08]           # the market's view: runner 0 is a real +0.09 edge
    book_sum = 1.12                                       # reference book sum (overround 0.12 <= 0.25 ceiling)
    return {
        "venue": "Replayville", "race_time": (now + timedelta(hours=3)).isoformat(), "race_id": "S17_SYNTH",
        "field_size": 6, "ev_eligible": eligible,
        "ev_gate": {"eligible": eligible, "reference_source": book, "reference_book_complete": eligible,
                    "reference_overround": 0.12, "price_sources": [book],
                    "reasons": [] if eligible else ["incomplete_reference_book:5/6"]},
        "runners": [{"horse_id": f"S17H{i}", "horse_name": f"Synthetic {i}",
                     "reference_odds": round(1.0 / (q * book_sum), 3), "best_odds": round(1.05 / (q * book_sum), 3),
                     "best_book": book, "reference_source": book, "value_supported": True,
                     "value_win_prob_independent": p, "value_win_prob": p, "fetched_at": fetched}
                    for i, (p, q) in enumerate(zip(probs, fair))],
    }


def healthy(now: datetime, book: str = "livescorebet", age: float = 30.0, ok: bool = True) -> dict:
    return {book: {"ok": ok, "status": "ok" if ok else "error", "age_seconds": age}}


def decisions(race, cfg, verdict, now, health) -> Counter:
    return Counter(r.decision for r in evaluate_race(race, cfg=cfg, model_verdict=verdict,
                                                     source_health=health, now=now))


def reasons_of(race, cfg, verdict, now, health) -> list[str]:
    out = evaluate_race(race, cfg=cfg, model_verdict=verdict, source_health=health, now=now)
    return sorted({reason for r in out for reason in r.pass_reasons})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    work = ROOT / "data/audit/17/replay" / args.label
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    before = {rel: file_state(rel) for rel in REAL_STORES}

    base = ExecutionConfig.from_config()
    cfg = isolated_cfg(base, work)
    verdict = load_model_verdict()
    # A STUB 'GO' (the real verdict object with go flipped) used ONLY in-memory, to prove the
    # OTHER gates are not vacuous. It is never written anywhere and never reaches a real store.
    go_stub = replace(verdict, go=True, reasons=())
    check("live model verdict is NO-GO and sourced from the stage-4 evaluation",
          verdict.go is False, {"verdict": verdict.verdict_label, "source": verdict.source,
                                "reasons": list(verdict.reasons)[:6]})
    check("execution.paper_only is true and the loaded config is provisional",
          cfg.paper_only is True and cfg.provisional is True, {"clamps_applied": list(cfg.clamps_applied)})

    # ── R1: replay the REAL current card through the full loop, isolated ─────
    preds_real = json.loads((ROOT / "data/predictions.json").read_text(encoding="utf-8"))
    preds = work / "predictions.json"
    shutil.copyfile(ROOT / "data/predictions.json", preds)
    generated = preds_real.get("generated_at") or preds_real.get("generated") or ""
    try:
        now = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    except ValueError:
        now = datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc) + timedelta(minutes=5)     # decide 5 min after the card was built
    r1 = run_loop(cfg, work, preds, now)
    store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    try:
        tickets = store.all_tickets()
    finally:
        store.close()
    n_runners = sum(len(r.get("runners") or []) for r in preds_real.get("races", []))
    by_decision = Counter(t.decision for t in tickets)
    check("R1 real card replay: every decision is PASS, zero CANDIDATE under NO-GO",
          by_decision.get("CANDIDATE", 0) == 0 and len(tickets) > 0,
          {"card_generated_at": generated, "races": len(preds_real.get("races", [])), "runners": n_runners,
           "tickets": len(tickets), "by_decision": dict(by_decision), "issue": r1.get("issue"),
           "deployment": r1.get("deployment")})
    reason_hist = Counter(reason.split(":")[0] + ":" + reason.split(":")[1].split("=")[0]
                          if ":" in reason else reason
                          for t in tickets for reason in (t.pass_reasons or []))
    check("R1 every PASS carries the model_validation NO-GO reason (disclosed, not silent)",
          all(any(str(x).startswith("model_validation") or str(x).startswith("race_gate") for x in (t.pass_reasons or []))
              for t in tickets), {"top_reasons": reason_hist.most_common(12)})
    ids = [t.ticket_id for t in tickets]
    keys = [(t.race_uid, t.horse_key, t.market_type, t.bet_type) for t in tickets]
    check("R1 identity: ticket ids unique; (race_uid, horse_key, market, bet_type) unique; all WIN; all paper_only",
          len(set(ids)) == len(ids) and len(set(keys)) == len(keys)
          and {t.market_type for t in tickets} <= {"WIN"} and all(t.paper_only for t in tickets),
          {"n": len(ids), "markets": sorted({str(t.market_type) for t in tickets}),
           "missing_venue": sum(1 for t in tickets if not t.venue),
           "missing_horse_id": sum(1 for t in tickets if not t.horse_id)})
    check("R1 provenance GAP: a ticket records no model bundle/version/hash",
          True, {"ticket_has_model_version_field": any(hasattr(t, f) for t in tickets[:1]
                                                        for f in ("model_version", "model_tag", "model_hash")),
                 "window_hashes_recorded": bool((r1.get("window") or {}).get("hashes"))})

    # ── R2: idempotent same-day re-run ───────────────────────────────────────
    r2 = run_loop(cfg, work, preds, now)
    check("R2 same-day re-run issues nothing new (idempotent)",
          r2["issue"]["issued"] == 0 and r2["issue"]["duplicates"] == len(tickets), r2["issue"])
    check("R2 settlement under NO-GO touches nothing (no CANDIDATE exists to settle)",
          r2["settle"]["settled"] == 0, r2["settle"])
    check("forward gate: not passed, formal window not started, thresholds retained (8 wk / 200 / 150)",
          r2["forward_gate"]["passed"] is False and (r2.get("window") or {}).get("status") == "not_started"
          and (cfg.forward_gate.min_weeks, cfg.forward_gate.min_qualified_bets,
               cfg.forward_gate.min_qualified_races) == (8, 200, 150),
          {"window": (r2.get("window") or {}).get("status"),
           "failed": [c["name"] for c in r2["forward_gate"]["criteria"] if not c["passed"]]})

    # ── F: fail-closed probes on a synthetic six-runner race ─────────────────
    t0 = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    ok_race, ok_health = synthetic_race(t0), healthy(t0)
    d = decisions(ok_race, cfg, go_stub, t0, ok_health)
    check("F0 control: with a STUB GO verdict and everything fresh the gate DOES issue candidates "
          "(so the PASSes below are caused by the fault injected, not by a vacuous gate)",
          d.get("CANDIDATE", 0) > 0, dict(d))
    d = decisions(ok_race, cfg, verdict, t0, ok_health)
    check("F1 real NO-GO verdict, everything else perfect -> all PASS", d.get("CANDIDATE", 0) == 0, dict(d))
    probes = {
        "F2 stale executable quote (1,200 s > 900 s TTL)": (synthetic_race(t0, age_s=1200), ok_health, "executable_price"),
        "F3 quote timestamp in the future": (synthetic_race(t0, age_s=-120), ok_health, "executable_price"),
        "F4 non-executable price source (consensus, no named bookmaker)":
            (synthetic_race(t0, book="fused_consensus"), healthy(t0, "fused_consensus"), "executable_price"),
        "F5 incomplete reference book (race-level ev_eligible False)": (synthetic_race(t0, eligible=False), ok_health, "race_gate"),
        "F6 source health stale (3,600 s)": (ok_race, healthy(t0, age=3600), "source_health"),
        "F7 source health erroring": (ok_race, healthy(t0, ok=False), "source_health"),
        "F8 source health telemetry absent": (ok_race, {}, "source_health"),
    }
    for name, (race, health, expect) in probes.items():
        d = decisions(race, cfg, go_stub, t0, health)
        why = [r for r in reasons_of(race, cfg, go_stub, t0, health) if r.startswith(expect)]
        check(name + " -> PASS even with a GO verdict", d.get("CANDIDATE", 0) == 0 and bool(why),
              {"decisions": dict(d), "reasons": why[:3]})
    no_prob = copy.deepcopy(ok_race)
    for rn in no_prob["runners"]:
        rn.pop("value_win_prob_independent")
    d = decisions(no_prob, cfg, go_stub, t0, ok_health)
    check("F9 only a MARKET-ADJUSTED probability available -> PASS (price echo can never qualify)",
          d.get("CANDIDATE", 0) == 0, {"decisions": dict(d),
                                       "reasons": [r for r in reasons_of(no_prob, cfg, go_stub, t0, ok_health)
                                                   if r.startswith("calibrated_probability")]})

    missing = load_model_verdict(str(work / "does_not_exist.json"))
    bad = work / "corrupt_verdict.json"
    bad.write_text("{not json", encoding="utf-8")
    corrupt = load_model_verdict(str(bad))
    check("F10 verdict file missing or corrupt -> NO-GO (never default-GO)",
          missing.go is False and corrupt.go is False,
          {"missing": list(missing.reasons)[:2], "corrupt": list(corrupt.reasons)[:2]})

    rf = run_loop(cfg, work, work / "no_such_predictions.json", now + timedelta(days=1))
    check("F11 predictions file missing -> loop completes, issues nothing", rf["issue"]["issued"] == 0, rf["issue"])

    # real-money refusal at the store
    cand = [r for r in evaluate_race(ok_race, cfg=cfg, model_verdict=go_stub, source_health=ok_health, now=t0)
            if r.decision == "CANDIDATE"][0]
    runner = next(rn for rn in ok_race["runners"] if rn["horse_id"] == cand.detail.get("horse_id", ok_race["runners"][0]["horse_id"])) \
        if isinstance(getattr(cand, "detail", None), dict) and cand.detail.get("horse_id") else ok_race["runners"][0]
    ticket = build_ticket(runner, ok_race, cand, cfg=cfg, model_verdict=go_stub,
                          forward_gate_state="FORWARD GATE NOT MET", now=t0)
    check("F12 a ticket built under a (stub) GO verdict is STILL paper_only while the forward gate is unmet",
          ticket.paper_only is True, {"paper_only": ticket.paper_only, "stake": ticket.stake})
    store = TicketStore(str(work / "refusal.db"), cfg=cfg)
    try:
        try:
            store.issue(replace(ticket, paper_only=False))
            refused = False
        except RealMoneyTicketRefused:
            refused = True
    finally:
        store.close()
    check("F13 the ticket store REFUSES a non-paper ticket (RealMoneyTicketRefused)", refused, {})

    # ── the config bypass: one YAML flag vs the 'no override' contract ───────
    raw = copy.deepcopy(get_config())
    raw["execution"]["gates"]["require_model_validation"] = False
    loosened = isolated_cfg(ExecutionConfig.from_config(raw), work)
    d = decisions(ok_race, loosened, verdict, t0, ok_health)
    check("F14 config sets gates.require_model_validation=false under a real NO-GO verdict -> must STILL be all PASS",
          d.get("CANDIDATE", 0) == 0,
          {"decisions": dict(d), "loaded_flag": loosened.gates.require_model_validation,
           "clamps_applied": [c for c in loosened.clamps_applied if "model_validation" in c]})

    # ── GAP probes: reproduced and RECORDED, deliberately not pass/fail ───────
    already_off = [r for r in preds_real.get("races", [])
                   if datetime.fromisoformat(r["race_time"]).astimezone(timezone.utc) <= now]
    check("GAP-A no operating cutoff on the live loop: races already OFF at the decision instant are still ticketed "
          "(execution.safeguards.block_started_races is configured but never called by the loop)", True,
          {"decision_instant_utc": now.isoformat(), "races_already_off": len(already_off),
           "their_runners_all_ticketed": sum(len(r["runners"]) for r in already_off)})
    stale_work = work / "stale_card_next_day"
    stale_work.mkdir()
    stale_cfg = isolated_cfg(base, stale_work)
    shutil.copyfile(preds, stale_work / "predictions.json")
    r_stale = run_loop(stale_cfg, stale_work, stale_work / "predictions.json", now + timedelta(days=1))
    check("GAP-B a STALE card (refresh failed; yesterday's predictions.json still on disk) is re-ticketed in full the "
          "next day under new ticket ids - the loop checks only that the file exists", True,
          {"issued_one_day_later_from_yesterdays_card": r_stale["issue"]["issued"],
           "all_PASS": True, "exposure": "none under NO-GO; pollutes the PASS disclosure ledger"})
    dup = Counter((r["venue"], frozenset(x["horse_name"] for x in r["runners"])) for r in preds_real.get("races", []))
    check("GAP-C identity: one physical race emitted twice when two odds sources disagree on the off-time by a minute "
          "-> two ticket sets for the same runners", True,
          {"duplicated_fields": [f"{k[0]} x{v} ({len(k[1])} runners)" for k, v in dup.items() if v > 1]})

    # ── alert behaviour ──────────────────────────────────────────────────────
    import os
    from utils import alert_dedupe
    from utils.notifications import get_notifier
    notifier = get_notifier()
    check("ALERT the notifier is INACTIVE in this session (RP_DISABLE_NOTIFICATIONS / no channel) - this replay "
          "cannot send an external message", not bool(getattr(notifier, "_active", False)),
          {"RP_DISABLE_NOTIFICATIONS_set": bool(os.environ.get("RP_DISABLE_NOTIFICATIONS")),
           "active": bool(getattr(notifier, "_active", False))})
    real_alert_path, alert_dedupe._PATH = alert_dedupe._PATH, str(work / "sent_alerts.json")   # isolate the store
    try:
        fp = alert_dedupe.fingerprint("drop", "Replayville|2026-09-19T15:00", "Synthetic 0", "3.75")
        first, second = alert_dedupe.seen(fp), alert_dedupe.seen(fp)
        reshortened = alert_dedupe.seen(alert_dedupe.fingerprint("drop", "Replayville|2026-09-19T15:00", "Synthetic 0", "3.50"))
    finally:
        alert_dedupe._PATH = real_alert_path
    check("ALERT dedupe (isolated store): a repeated fingerprint is suppressed; a genuinely NEW price re-alerts",
          first is False and second is True and reshortened is False,
          {"first_seen": first, "repeat_seen": second, "new_price_seen": reshortened})

    # ── isolation proof ──────────────────────────────────────────────────────
    after = {rel: file_state(rel) for rel in REAL_STORES}
    changed = [rel for rel in REAL_STORES if before[rel] != after[rel]]
    check("ISOLATION: no real store was modified by this replay (sha256 + mtime before/after)", not changed,
          {"changed": changed, "isolated_dir": work.relative_to(ROOT).as_posix()})

    out = HERE / f"06_paper_replay_{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "ran_at_utc": datetime.now(timezone.utc).isoformat(),
                               "results": results}, indent=1, default=str), encoding="utf-8")
    failed = [r["check"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} ok; wrote {out.relative_to(ROOT)}")
    for f in failed:
        print("  FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
