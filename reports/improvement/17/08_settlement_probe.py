"""Step 17 — what does the LIVE loop's settlement actually compute?

Isolated store only. A CANDIDATE is minted in-memory from a STUB GO verdict (never
persisted anywhere real), issued into a throwaway races.db, then settled through
the live loop's own ``_settle_open_tickets`` against synthetic results. The point is
to measure the gap between the terms the system CLAIMS (Rule 4, dead heats,
non-runner void, staking) and what the live path applies.

Run:  .venv/Scripts/python.exe reports/improvement/17/08_settlement_probe.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent

import scripts.daily_paper_loop as loop  # noqa: E402
from execution.config import ExecutionConfig  # noqa: E402
from execution.gates import evaluate_race  # noqa: E402
from execution.model_gate import load_model_verdict  # noqa: E402
from execution.settlement import settle_ticket  # noqa: E402,F401  (imported to show it EXISTS, unused live)
from execution.snapshots import SnapshotStore  # noqa: E402
from execution.tickets import TicketStore, build_ticket  # noqa: E402

_spec = importlib.util.spec_from_file_location("replay06", HERE / "06_paper_replay.py")
replay06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay06)


def main() -> int:
    work = ROOT / "data/audit/17/replay/settlement_probe"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cfg = replay06.isolated_cfg(ExecutionConfig.from_config(), work)
    go_stub = replace(load_model_verdict(), go=True, reasons=())
    t0 = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    out: dict = {}

    def one_candidate(tag: str):
        race = replay06.synthetic_race(t0)
        race["venue"], race["race_id"] = f"Probe{tag}", f"S17_{tag}"
        for rn in race["runners"]:
            rn["horse_id"], rn["horse_name"] = rn["horse_id"] + tag, rn["horse_name"] + " " + tag
        res = evaluate_race(race, cfg=cfg, model_verdict=go_stub, source_health=replay06.healthy(t0), now=t0)
        idx = next(i for i, r in enumerate(res) if r.decision == "CANDIDATE")
        ticket = build_ticket(race["runners"][idx], race, res[idx], cfg=cfg, model_verdict=go_stub,
                              forward_gate_state="FORWARD GATE NOT MET", now=t0)
        return race, race["runners"][idx], ticket

    store, snaps = TicketStore(cfg.snapshots.db_path, cfg=cfg), SnapshotStore(cfg=cfg)
    try:
        cases = {"WON": 1, "LOST": 5, "DNF": None}       # finishing position; None = ran but did not complete
        results = []
        for tag, position in cases.items():
            race, runner, ticket = one_candidate(tag)
            store.issue(ticket)
            out.setdefault("issued", {})[tag] = {"decision": ticket.decision, "paper_only": ticket.paper_only,
                                                 "stake": ticket.stake, "max_stake": ticket.max_stake,
                                                 "offered_odds": ticket.offered_odds}
            for rn in race["runners"]:                     # a full result card for the race
                results.append({"venue": race["venue"], "race_date": race["race_time"], "race_time": race["race_time"],
                                "horse_id": rn["horse_id"], "horse_name": rn["horse_name"],
                                "position": position if rn is runner else 3})
        real_loader = loop._load_results
        loop._load_results = lambda *a, **k: results        # synthetic results; the real parquet is not read
        try:
            out["settle_counts"] = loop._settle_open_tickets(store, snaps, now=t0.replace(hour=23))
        except TypeError:
            out["settle_counts"] = loop._settle_open_tickets(store, snaps)
        finally:
            loop._load_results = real_loader
        frame = store.to_frame()                           # settlement columns live on the row, not the Ticket
        frame = frame[frame["settled_at"].notna()]
        out["settled"] = {str(row["horse_name"]).split()[-1]: {
            k: (None if row[k] != row[k] else row[k]) for k in ("outcome", "stake", "returns", "profit", "clv_pct")}
            for _, row in frame.iterrows()}
    finally:
        store.close()
        snaps.close()

    s = out["settled"]
    out["findings"] = {
        "stake_is_never_sized_live": all(v["stake"] in (None, 0, 0.0) for v in out["issued"].values()),
        "a_WINNING_candidate_records_zero_profit": s.get("WON", {}).get("profit") in (0, 0.0),
        "winner_and_loser_are_numerically_identical": (s.get("WON", {}).get("profit") == s.get("LOST", {}).get("profit")),
        "a_non_finisher_is_settled_void_not_lost": s.get("DNF", {}).get("outcome") == "void",
        "execution.settlement.settle_ticket_called_by_live_loop": "settle_ticket" in Path(loop.__file__).read_text(encoding="utf-8"),
    }
    (HERE / "08_settlement_probe.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
