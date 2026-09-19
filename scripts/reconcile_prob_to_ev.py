"""Probability→EV reconciliation harness (Stage 3 audit deliverable).

Two modes, kept strictly apart so a synthetic run can never masquerade as live
evidence (closes HANDOFF unresolved issue #5 — the old script silently rewrote
``fetched_at``/``stale`` and so could not evidence freshness):

**live (default)** — audits ``data/predictions.json`` exactly as shipped. No
model run, no scrape, and **no provenance is touched**: the cache's own
``generated_at``, per-race ``ev_gate`` (eligibility + PASS reasons, reference
source, price ages) and per-runner typed fields are reconciled as-is. Writes
``reports/prob_to_ev_reconciliation.md`` + ``reports/predictions_reconciliation.json``.

**--synthetic** — the offline arithmetic audit: reconstructs a card from the most
recent multi-book snapshot in ``data/unified_races.parquet``, strips results,
shifts race dates +1 day (to satisfy the upcoming-guard) and **rewrites
``fetched_at``/``stale``** so the card survives the freshness gates, then runs
the real models over it. Valid ONLY as evidence that the calculation identities
hold end-to-end through the real code path; explicitly **no evidence of live
freshness** — the report says so in its banner. Writes ``*_synthetic.*`` file
names so the two kinds of evidence can never be confused.

Reconciled per race (both modes):

* ``ev_eligible`` / ``ev_gate.reasons`` — the race-level PASS decision and its
  explicit reasons;
* displayed win probabilities (``won_prob_normalized``) sum to 1 (±tol);
* a complete de-vigged reference book (``market_prob``) sums to 1 (±tol), and an
  eligible race is never partially de-vigged;
* **exact** EV identities from persisted inputs (quantised-before-multiply, so
  zero tolerance): ``expected_value == round(value_win_prob*exec − 1, 4)``,
  ``ev_catboost == round(catboost_win_prob*exec − 1, 4)``, ``ev_lgbm`` likewise,
  ``value_edge == round(value_win_prob − implied_prob, 4)``,
  ``decimal_odds == round(1/implied_prob, 3)``;
* reference odds/source carried separately from executable best odds;
* :func:`models.predictor.check_output_invariants` over the whole payload.

Representative races (first eligible, first ineligible, first with value bets)
are dumped runner-by-runner with every calculation input and output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
CACHE = BASE / "data" / "predictions.json"
BOOK_SOURCES = ["livescorebet", "boylesports", "paddy_power"]
TOL = 0.02          # sum-to-1 tolerance over 4dp-rounded persisted probabilities
N_REPRESENTATIVE = 3

# Per-runner columns dumped for representative races — every input and output on
# the probability→EV path, in calculation order.
_RUNNER_DUMP_COLS = [
    "horse_name", "implied_prob", "decimal_odds", "reference_odds",
    "reference_source", "best_odds", "best_book", "market_prob",
    "won_prob_normalized", "catboost_win_prob", "lgbm_win_prob",
    "value_win_prob", "empirical_win_rate", "value_edge",
    "expected_value", "ev_catboost", "ev_lgbm", "value_bet",
]


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _exec_price(r: dict):
    """The executable price EV is computed against: best board price else fused."""
    return _f(r.get("best_odds")) if _f(r.get("best_odds")) is not None else _f(r.get("decimal_odds"))


def _identity_errors(r: dict) -> list:
    """Exact-recompute failures for one persisted runner (empty ⇒ reproducible)."""
    errs = []
    exe = _exec_price(r)

    def _check(name, prob_key, ev_key):
        p, ev = _f(r.get(prob_key)), _f(r.get(ev_key))
        if p is None or ev is None or exe is None:
            return
        if round(p * exe - 1.0, 4) != round(ev, 4):
            errs.append(f"{name}: round({p}*{exe}-1,4)={round(p * exe - 1.0, 4)} "
                        f"!= persisted {ev}")

    _check("expected_value", "value_win_prob", "expected_value")
    _check("ev_catboost", "catboost_win_prob", "ev_catboost")
    _check("ev_lgbm", "lgbm_win_prob", "ev_lgbm")

    p, ip, edge = _f(r.get("value_win_prob")), _f(r.get("implied_prob")), _f(r.get("value_edge"))
    if p is not None and ip is not None and edge is not None:
        if round(p - ip, 4) != round(edge, 4):
            errs.append(f"value_edge: round({p}-{ip},4) != persisted {edge}")
    do = _f(r.get("decimal_odds"))
    if do is not None and ip:
        if round(1.0 / ip, 3) != round(do, 3):
            errs.append(f"decimal_odds: round(1/{ip},3)={round(1.0 / ip, 3)} "
                        f"!= persisted {do}")
    return errs


def reconcile_payload(payload: dict) -> dict:
    """Reconcile one predictions payload. Returns table rows + violation lists."""
    from models.predictor import check_output_invariants

    rows, identity_failures = [], []
    for race in payload.get("races", []):
        field = race.get("runners") or (
            (race.get("selections") or []) + (race.get("excluded_low_odds") or []))
        gate = race.get("ev_gate") or {}
        eligible = race.get("ev_eligible")

        norm = [x for x in (_f(r.get("won_prob_normalized")) for r in field) if x is not None]
        mkt = [_f(r.get("market_prob")) for r in field]
        mkt_complete = bool(field) and all(m is not None for m in mkt)

        race_errs = []
        for r in field:
            race_errs.extend(f"{r.get('horse_name', '?')}: {e}"
                             for e in _identity_errors(r))
        if race_errs:
            identity_failures.append(
                {"race": f"{race.get('venue')} {race.get('race_time')}",
                 "errors": race_errs})

        rows.append({
            "venue": race.get("venue"),
            "race_time": str(race.get("race_time"))[:16],
            "n_runners": len(field),
            "ev_eligible": eligible,
            "pass_reasons": "; ".join(gate.get("reasons", [])) or
                            ("—" if eligible else "legacy(no gate)"),
            "reference_source": gate.get("reference_source"),
            "overround": gate.get("reference_overround"),
            "sum_win_norm": round(sum(norm), 4) if norm else None,
            "win_sums_to_1": (abs(sum(norm) - 1.0) <= TOL) if len(norm) >= 2 else None,
            "market_book": "complete" if mkt_complete else "partial→PASS",
            "sum_market_prob": round(sum(m for m in mkt if m is not None), 4)
                               if mkt_complete else None,
            "market_sums_to_1": (abs(sum(m for m in mkt if m is not None) - 1.0) <= TOL)
                                if mkt_complete else None,
            "ev_identity_exact": not race_errs,
            "value_bets": sum(1 for r in field if r.get("value_bet")),
        })

    return {
        "table": pd.DataFrame(rows),
        "identity_failures": identity_failures,
        "invariant_violations": check_output_invariants(payload.get("races", [])),
    }


def _representative_races(payload: dict) -> list:
    """First eligible, first ineligible, first value-carrying race (deduped)."""
    races = payload.get("races", [])
    picks, seen = [], set()

    def _key(r):
        return (r.get("venue"), str(r.get("race_time")))

    def _add(r):
        if r is not None and _key(r) not in seen:
            seen.add(_key(r))
            picks.append(r)

    _add(next((r for r in races if r.get("ev_eligible")), None))
    _add(next((r for r in races if r.get("ev_eligible") is False), None))
    _add(next((r for r in races
               if any(x.get("value_bet") for x in (r.get("runners") or []))), None))
    for r in races:
        if len(picks) >= N_REPRESENTATIVE:
            break
        _add(r)
    return picks[:N_REPRESENTATIVE]


def _dump_race(race: dict) -> list:
    gate = race.get("ev_gate") or {}
    lines = [
        f"### {race.get('venue')} {race.get('race_time')} — "
        f"{'ELIGIBLE' if race.get('ev_eligible') else 'PASS'}",
        "",
        f"- `ev_eligible`: **{race.get('ev_eligible')}**"
        + (f" — reasons: {'; '.join(gate.get('reasons', []))}" if gate.get("reasons") else ""),
        f"- reference book: source `{gate.get('reference_source')}`, "
        f"complete={gate.get('reference_book_complete')}, "
        f"overround={gate.get('reference_overround')}",
        f"- price sources: {gate.get('price_sources')}; "
        f"max odds age at compute: {gate.get('odds_max_age_seconds')}s; "
        f"computed_at: {gate.get('computed_at')}",
        "",
    ]
    field = race.get("runners") or []
    if field:
        df = pd.DataFrame([{c: r.get(c) for c in _RUNNER_DUMP_COLS} for r in field])
        lines += [df.to_markdown(index=False), ""]
    return lines


def _write_report(payload: dict, result: dict, *, mode: str, md_path: Path,
                  json_path: Path, source_note: str) -> None:
    table: pd.DataFrame = result["table"]
    violations = result["invariant_violations"]
    failures = result["identity_failures"]

    md = ["# Probability → EV reconciliation", ""]
    if mode == "synthetic":
        md += ["> **SYNTHETIC ARITHMETIC AUDIT.** Race dates were shifted +1 day "
               "and `fetched_at`/`stale` were rewritten so the reconstructed card "
               "passes the (correctly fail-closed) freshness gates. This report "
               "evidences the calculation identities through the real code path "
               "ONLY — it is **no evidence of live freshness**.", ""]
    else:
        md += ["> **LIVE-CACHE AUDIT.** `data/predictions.json` reconciled exactly "
               "as shipped — no model rerun, no provenance rewritten.", ""]
    md += [source_note, "",
           f"Races: **{len(table)}**  Runners: **{payload.get('total_runners')}**  "
           f"Invariant violations: **{len(violations)}**  "
           f"EV identity failures: **{sum(len(f['errors']) for f in failures)}**",
           ""]
    if not table.empty:
        md += [table.to_markdown(index=False), ""]
    else:
        md += ["_No races in the payload — nothing to reconcile "
               "(a fail-closed outcome when no fresh validated data exists)._", ""]

    if violations:
        md += ["## Invariant violations", ""] + [f"- {v}" for v in violations] + [""]
    if failures:
        md += ["## EV identity failures (persisted EV not reproducible)", ""]
        for f in failures:
            md += [f"- **{f['race']}**"] + [f"  - {e}" for e in f["errors"]]
        md += [""]
    if not violations and not failures:
        md += ["All honesty invariants hold: displayed win probabilities sum to 1, "
               "complete reference books de-vig to 1, EV-ineligible races carry no "
               "EV/value fields, and **every persisted EV/edge/price is exactly "
               "reproducible from its persisted inputs**.", ""]

    md += ["## Representative races — full calculation inputs and outputs", ""]
    for race in _representative_races(payload):
        md += _dump_race(race)

    md_path.write_text("\n".join(md), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_live_card() -> pd.DataFrame:
    """SYNTHETIC mode only: most recent multi-book snapshot → pre-off card.

    Deliberately rewrites provenance (dates +1d, fetched_at=now, stale=False) so
    the card survives the upcoming-guard and freshness gates — which is exactly
    why this mode can never evidence freshness and is labelled synthetic.
    """
    import glob
    files = glob.glob(
        str(BASE / "data/unified_races.parquet/year=2026/**/*.parquet"), recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["_d"] = df["race_date"].astype(str).str[:10]
    latest = sorted(df.loc[df["source"].isin(BOOK_SOURCES), "_d"].unique())[-1]
    day = df[df["_d"] == latest].copy()
    card = day[day["source"].isin(BOOK_SOURCES + ["betsp"])].copy()
    for col in ("position", "win_lose", "odds_finish", "ppwap"):
        if col in card.columns:
            card[col] = pd.NA
    card["race_time"] = pd.to_datetime(card["race_time"], utc=True, errors="coerce") \
        + pd.Timedelta(days=1)
    card["race_date"] = card["race_time"].dt.strftime("%Y-%m-%d")
    card["fetched_at"] = pd.Timestamp.now(tz="UTC")
    if "stale" in card.columns:
        card["stale"] = False
    return card.drop(columns=["_d"])


def _run_synthetic() -> dict:
    """Rebuild a payload through the real models over the reconstructed card."""
    import models.predictor as predictor
    from models.predictor import Predictor

    card = _load_live_card()
    print(f"synthetic card: {len(card)} rows, {card['venue'].nunique()} venues, "
          f"sources={sorted(card['source'].unique())}")
    out_path = REPORTS / "predictions_reconciliation_synthetic.json"
    orig = predictor._CACHE_PATH
    predictor._CACHE_PATH = out_path       # never touch the live cache
    try:
        p = Predictor()
        if not p.load():
            raise SystemExit("models failed to load")
        p.predict(unified=card)
    finally:
        predictor._CACHE_PATH = orig
    return json.loads(out_path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true",
                    help="offline arithmetic audit over a reconstructed card "
                         "(rewrites provenance; clearly labelled; separate files)")
    args = ap.parse_args(argv)
    REPORTS.mkdir(exist_ok=True)

    if args.synthetic:
        payload = _run_synthetic()
        md_path = REPORTS / "prob_to_ev_reconciliation_synthetic.md"
        json_path = REPORTS / "predictions_reconciliation_synthetic.json"
        source_note = ("Source: reconstructed multi-book snapshot (results stripped, "
                       "dates +1d, provenance rewritten), scored by the real models.")
        mode = "synthetic"
    else:
        if not CACHE.exists():
            print(f"no {CACHE} — run the predictor first", file=sys.stderr)
            return 2
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
        gen = payload.get("generated_at")
        age_h = None
        try:
            age_h = (pd.Timestamp.now(tz="UTC")
                     - pd.Timestamp(gen).tz_convert("UTC")).total_seconds() / 3600.0
        except (TypeError, ValueError):
            pass
        md_path = REPORTS / "prob_to_ev_reconciliation.md"
        json_path = REPORTS / "predictions_reconciliation.json"
        source_note = (f"Source: `data/predictions.json` as shipped — generated_at "
                       f"**{gen}**" + (f" ({age_h:.1f}h before this audit)"
                                       if age_h is not None else ""))
        mode = "live"

    result = reconcile_payload(payload)
    _write_report(payload, result, mode=mode, md_path=md_path,
                  json_path=json_path, source_note=source_note)

    table = result["table"]
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(table.to_string(index=False) if not table.empty else "no races")
    n_viol = len(result["invariant_violations"])
    n_id = sum(len(f["errors"]) for f in result["identity_failures"])
    print(f"\ninvariant violations: {n_viol}   EV identity failures: {n_id}")
    for v in result["invariant_violations"][:20]:
        print("  -", v)
    print(f"wrote {md_path}\nwrote {json_path}")
    return 1 if (n_viol or n_id) else 0


if __name__ == "__main__":
    raise SystemExit(main())
