"""Step 21 — re-run step 17's settlement probe through the REPAIRED live path.

Step 17's ``reports/improvement/17/08_settlement_probe.py`` measured B2/B3 by
minting a stub-GO candidate straight from ``build_ticket`` (never
``scripts.daily_paper_loop._today_tickets``) and settling it with
``_settle_open_tickets`` against a synthetic betSP-style result list (no
non-runner/DNF distinction available). This probe instead drives the SAME
scenario through the actual repaired live functions:

  * issuance goes through ``_today_tickets`` (so real
    ``execution.staking.StakePlanner`` sizing is exercised, not a hand-built
    ticket with no stake_decision at all), and
  * settlement is offered a real ``execution.race_facts.RaceFacts`` record
    per race (via a monkeypatched ``_load_race_facts_for``, since the real
    on-disk archive has no entry for a synthetic venue), so the genuine
    Rule 4 / dead-heat / non-runner / DNF-as-loss treatment in
    ``execution.settlement.settle_ticket`` is what actually settles the
    ticket, not the coarser betSP-position fallback.

Isolated store only; the stub GO verdict is never persisted anywhere real.

Run:  .venv/Scripts/python.exe reports/improvement/21/01_settlement_probe.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent

import scripts.daily_paper_loop as loop  # noqa: E402
from execution.config import ExecutionConfig  # noqa: E402
from execution.model_gate import load_model_verdict  # noqa: E402
from execution.race_facts import RaceFacts, RunnerFact  # noqa: E402
from execution.snapshots import SnapshotStore  # noqa: E402
from execution.tickets import TicketStore  # noqa: E402
from utils.text_norm import norm_horse  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "replay06", ROOT / "reports/improvement/17/06_paper_replay.py"
)
replay06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay06)

CASES = {"WON": 1, "LOST": 5, "DNF": None, "NONRUNNER": "nonrunner"}


def _tagged_race(t0: datetime, tag: str) -> dict:
    race = replay06.synthetic_race(t0)
    race["venue"] = f"Probe{tag}"
    race["race_id"] = f"S21_{tag}"
    for rn in race["runners"]:
        rn["horse_id"] = rn["horse_id"] + tag
        rn["horse_name"] = rn["horse_name"] + " " + tag
    return race


def _race_facts_for(race: dict, tag: str, position) -> RaceFacts:
    key = loop._race_facts_key_for(race["venue"], race["race_time"])
    runners: dict[str, RunnerFact] = {}
    for i, rn in enumerate(race["runners"]):
        name = rn["horse_name"]
        hkey = norm_horse(name)
        if i == 0:
            if position == "nonrunner":
                runners[hkey] = RunnerFact(horse_key=hkey, horse_name=name,
                                            ride_status="NONRUNNER", finish_position=None)
            elif position is None:  # DNF: ran, no finishing position
                runners[hkey] = RunnerFact(horse_key=hkey, horse_name=name,
                                            ride_status="RUNNER", finish_position=None,
                                            casualty_reason="Fell")
            else:
                runners[hkey] = RunnerFact(horse_key=hkey, horse_name=name,
                                            ride_status="RUNNER", finish_position=int(position))
        else:
            runners[hkey] = RunnerFact(horse_key=hkey, horse_name=name,
                                        ride_status="RUNNER", finish_position=i + 1)
    n_runners = sum(1 for r in runners.values() if r.ride_status == "RUNNER")
    return RaceFacts(
        race_key=key, race_date=key.split("|")[1][:10], off_time=key.split("|")[1],
        venue=race["venue"], declared_field=len(runners), n_runners=n_runners,
        runners=runners,
    )


def main() -> int:
    work = ROOT / "data/audit/21/replay/settlement_probe"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cfg = replay06.isolated_cfg(ExecutionConfig.from_config(), work)
    go_stub = replace(load_model_verdict(), go=True, reasons=())
    t0 = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")

    races = {tag: _tagged_race(t0, tag) for tag in CASES}
    facts_by_key = {}
    for tag, position in CASES.items():
        fact = _race_facts_for(races[tag], tag, position)
        facts_by_key[fact.race_key] = fact

    predictions_path = work / "predictions.json"
    predictions_path.write_text(
        json.dumps({"generated_at": t0.isoformat(), "races": list(races.values())}),
        encoding="utf-8",
    )

    store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snaps = SnapshotStore(cfg=cfg)
    out: dict = {}
    try:
        # `_today_tickets` reads real source health via `utils.source_health.
        # get_health()` (unlike `evaluate_race` called directly, which takes an
        # explicit override) -- pin it healthy for this isolated probe so the
        # `source_health` gate condition cannot fail on whatever the real repo's
        # last live scrape happened to record.
        import utils.source_health as source_health_module

        original_get_health = source_health_module.get_health
        source_health_module.get_health = lambda: replay06.healthy(t0)
        try:
            tickets, reason = loop._today_tickets(
                cfg=cfg, verdict=go_stub, gate=gate,
                predictions_path=str(predictions_path), now=t0.replace(hour=0, minute=0, second=0),
                evaluated_at=t0, ticket_store=store, bankroll=1000.0,
            )
        finally:
            source_health_module.get_health = original_get_health
        assert reason is None, reason
        candidates = [t for t in tickets if t.decision == "CANDIDATE"]
        assert len(candidates) == len(CASES), f"expected one candidate per race, got {len(candidates)}"
        issue_stats = loop._issue_tickets(store, tickets)
        out["issued"] = {
            t.horse_name.rsplit(" ", 1)[-1]: {
                "decision": t.decision, "paper_only": t.paper_only,
                "stake": t.stake, "max_stake": t.max_stake, "offered_odds": t.offered_odds,
            }
            for t in candidates
        }

        original_load_facts = loop._load_race_facts_for
        loop._load_race_facts_for = lambda race_times, _facts=facts_by_key: dict(_facts)
        try:
            settle_stats = loop._settle_open_tickets(store, snaps)
        finally:
            loop._load_race_facts_for = original_load_facts

        out["settle_stats"] = settle_stats
        frame = store.to_frame()
        frame = frame[frame["settled_at"].notna()]
        out["settled"] = {
            str(row["horse_name"]).rsplit(" ", 1)[-1]: {
                k: (None if row[k] != row[k] else row[k])
                for k in ("outcome", "stake", "returns", "profit")
            }
            for _, row in frame.iterrows()
        }
    finally:
        store.close()
        snaps.close()

    s = out["settled"]
    out["findings"] = {
        "stake_is_never_sized_live": all(v["stake"] in (None, 0, 0.0) for v in out["issued"].values()),
        "a_WINNING_candidate_records_zero_profit": s.get("WON", {}).get("profit") in (0, 0.0),
        "winner_and_loser_are_numerically_identical": (
            s.get("WON", {}).get("profit") == s.get("LOST", {}).get("profit")
        ),
        "a_non_finisher_is_settled_void_not_lost": s.get("DNF", {}).get("outcome") == "void",
        "an_explicit_non_runner_is_settled_void": s.get("NONRUNNER", {}).get("outcome") == "void",
        "settled_via_race_facts_for_every_case": out["settle_stats"].get("settled_via_race_facts") == len(CASES),
    }
    (HERE / "01_settlement_probe.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
