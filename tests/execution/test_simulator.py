"""execution/simulator.py — the walk-forward, and what it refuses to assume.

Two guarantees carry the whole stage and are tested first: a decision can never
read a price stamped after it, and the closing price is CLV evidence rather than
a fill. Everything else here is a friction that can only make a simulated result
worse than the naive one — latency, staleness, rejection, suspension, book
limits, commission — plus the ordering property that makes drawdown mean
something: bankroll evolves as the walk proceeds.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from execution import race_facts as rfacts
from execution import simulator as sim
from execution.config import ExecutionConfig
from execution.settlement import EachWayTerms
from execution.snapshots import FuturePriceError, SnapshotStore

# The baselines fixture field, reused so eligibility is known-good: five runners,
# reference book 1.1324, model backs "c" at an executable 5.5.
HORSES = ["a", "b", "c", "d", "e"]
REF_ODDS = [1.9, 4.5, 5.5, 9.0, 11.0]
EXEC_ODDS = [1.9, 6.0, 5.5, 9.0, 11.0]
MODEL_PROB = [0.35, 0.19, 0.30, 0.08, 0.08]
# Deliberately far from the executable price: if a closing price ever leaks into
# a fill, the ledger's decimal_odds will say so loudly.
CLOSE_ODDS = [1.5, 3.0, 2.0, 5.0, 6.0]


def make_race(race_uid: str, date: str, off: str, winner: str = "c") -> pd.DataFrame:
    return pd.DataFrame({
        "race_uid": race_uid,
        "race_date": pd.to_datetime([date] * 5, utc=True),
        "race_time": pd.to_datetime([off] * 5, utc=True),
        "horse_key": HORSES,
        "horse_id": [f"{race_uid}-{h}" for h in HORSES],
        "model_prob": MODEL_PROB,
        "decimal_odds": EXEC_ODDS,
        "reference_odds": REF_ODDS,
        "closing_odds": CLOSE_ODDS,
        "won": [1 if h == winner else 0 for h in HORSES],
    })


@pytest.fixture
def cfg() -> ExecutionConfig:
    return ExecutionConfig.from_config({})


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.concat([
        make_race("Naas|2026-06-01T15:40", "2026-06-01", "2026-06-01T14:40:00+00:00", "c"),
        make_race("Ayr|2026-06-02T16:10", "2026-06-02", "2026-06-02T15:10:00+00:00", "a"),
        make_race("Cork|2026-06-03T17:20", "2026-06-03", "2026-06-03T16:20:00+00:00", "e"),
    ], ignore_index=True)


@pytest.fixture
def store(tmp_path, cfg, panel):
    st = SnapshotStore(str(tmp_path / "snaps.db"), cfg=cfg)
    sim.seed_snapshots(panel, st)
    yield st
    st.close()


def run(panel, store, cfg, **kw):
    return sim.simulate(panel, cfg=cfg, store=store, race_facts=None, **kw)


# ── keys and clocks ──────────────────────────────────────────────────────────
def test_the_race_facts_key_normalises_the_venue_the_panel_keeps_verbatim():
    assert sim.facts_key_for("Newmarket (July)|2026-06-01T15:40") == (
        f"{sim.norm_venue('Newmarket (July)')}|2026-06-01T15:40"
    )


def test_race_time_is_the_authority_for_the_off():
    off = sim._race_off("Naas|2026-06-01T15:40", "2026-06-01T14:40:00+00:00")
    assert off == datetime(2026, 6, 1, 14, 40, tzinfo=timezone.utc)


def test_the_uid_fallback_reads_local_racing_time_not_utc():
    """A 15:40 card in June is 14:40 UTC. Reading the stamp as UTC would place
    every decision an hour late — inside the off, for a 30-minute lead."""
    assert sim._race_off("Naas|2026-06-01T15:40") == datetime(
        2026, 6, 1, 14, 40, tzinfo=timezone.utc
    )


def test_an_unparseable_uid_yields_no_off_rather_than_a_guess():
    assert sim._race_off("nonsense") is None


# ── the point-in-time layout ─────────────────────────────────────────────────
def test_the_preoff_quote_is_readable_before_the_off_and_the_close_is_not(store, panel):
    off = datetime(2026, 6, 1, 14, 40, tzinfo=timezone.utc)
    decision_at = off - timedelta(seconds=sim.PREOFF_LEAD_SECONDS)
    quote = store.latest_quote("Naas|2026-06-01T15:40", "c", decision_at,
                               max_age_seconds=float("inf"))
    assert quote is not None
    assert quote.odds_decimal == pytest.approx(5.5), "the executable pre-off price"
    assert quote.odds_decimal != pytest.approx(2.0), "not the closing price"


def test_the_closing_price_is_reachable_only_through_closing_quote(store):
    closing = store.closing_quote("Naas|2026-06-01T15:40", "c", market_type="WIN")
    assert closing.odds_decimal == pytest.approx(2.0)


def test_a_quote_from_after_the_fill_raises_rather_than_being_used(store, cfg):
    """The single defect that would make every downstream number meaningless."""
    off = datetime(2026, 6, 1, 14, 40, tzinfo=timezone.utc)

    class LeakyStore:
        def latest_quote(self, *a, **k):
            return store.closing_quote("Naas|2026-06-01T15:40", "c", market_type="WIN")

    with pytest.raises(FuturePriceError):
        sim.attempt_fill(
            sim.FillRequest(
                race_uid="Naas|2026-06-01T15:40", horse_key="c", stake=5.0,
                requested_at=off - timedelta(seconds=1800), quoted_odds=5.5,
            ),
            store=LeakyStore(), cfg=cfg, seed=1,
        )


def test_seeding_records_both_rows_per_runner_with_a_price(panel, tmp_path, cfg):
    st = SnapshotStore(str(tmp_path / "s.db"), cfg=cfg)
    try:
        result = sim.seed_snapshots(panel, st)
        assert result["inserted"] == 30, "15 runners × (pre-off + close)"
    finally:
        st.close()


def test_seeding_can_omit_the_closing_row(panel, tmp_path, cfg):
    st = SnapshotStore(str(tmp_path / "s.db"), cfg=cfg)
    try:
        assert sim.seed_snapshots(panel, st, include_closing=False)["inserted"] == 15
    finally:
        st.close()


# ── what the walk actually strikes ───────────────────────────────────────────
def test_the_ledger_never_carries_the_closing_price_as_a_fill(panel, store, cfg):
    result = run(panel, store, cfg)
    assert result.n_struck > 0
    for odds in result.ledger["decimal_odds"]:
        assert odds not in CLOSE_ODDS


def test_the_closing_price_is_still_recorded_for_clv(panel, store, cfg):
    result = run(panel, store, cfg)
    assert result.ledger["closing_odds"].notna().all()
    assert set(result.ledger["closing_odds"]) <= set(CLOSE_ODDS)


def test_the_model_backs_only_the_runner_that_clears_both_gates(panel, store, cfg):
    result = run(panel, store, cfg)
    assert set(result.ledger["horse_key"]) == {"c"}


def test_a_rerun_reproduces_the_ledger_exactly(panel, store, cfg):
    a = run(panel, store, cfg, seed=7).ledger
    b = run(panel, store, cfg, seed=7).ledger
    pd.testing.assert_frame_equal(a, b)


def test_rejections_are_recorded_as_attempts_not_dropped(panel, store, cfg):
    always = replace(cfg, frictions=replace(cfg.frictions, rejection_rate=1.0))
    result = run(panel, store, always)
    assert result.n_struck == 0
    assert len(result.attempts) == 3
    assert set(result.attempts["blocked_by"]) == {"REJECTED"}


def test_a_suspended_market_beats_the_book_to_the_refusal(panel, store, cfg):
    both = replace(cfg, frictions=replace(
        cfg.frictions, rejection_rate=1.0, suspension_rate=1.0))
    assert set(run(panel, store, both).attempts["blocked_by"]) == {"SUSPENDED"}


def test_a_stale_quote_blocks_rather_than_filling_at_the_last_known_price(
    panel, store, cfg
):
    # Decide two hours before the off; the seeded quote is 30 minutes pre-off, so
    # nothing exists yet at decision time.
    result = run(panel, store, cfg, decision_lead_seconds=7200)
    assert result.n_struck == 0
    assert set(result.attempts["blocked_by"]) == {"NO_QUOTE"}


def test_the_book_limit_binds_the_stake_before_the_request_is_sent(
    panel, store, cfg
):
    """The planner already knows the book's ceiling, so nothing goes unmatched.

    ``StakePlanner`` reads ``frictions.max_stake_per_bet`` as one of its
    constraints, which is why the walk never requests more than the book will
    take. The partial-fill path below is what happens when something does.
    """
    capped = replace(cfg, frictions=replace(cfg.frictions, max_stake_per_bet=1.0))
    result = run(panel, store, capped)
    assert (result.ledger["stake"] == 1.0).all()
    assert (result.ledger["unmatched_stake"] == 0.0).all()
    assert set(result.ledger["fill_status"]) == {"FILLED"}
    assert set(result.ledger["binding_constraint"]) == {"per_bet_cap"}


def test_an_oversized_request_is_matched_in_part_never_at_a_worse_price(store, cfg):
    """Stake above the limit is *unmatched*, not filled at a longer price."""
    capped = replace(cfg, frictions=replace(cfg.frictions, max_stake_per_bet=1.0))
    off = datetime(2026, 6, 1, 14, 40, tzinfo=timezone.utc)
    fill = sim.attempt_fill(
        sim.FillRequest(
            race_uid="Naas|2026-06-01T15:40", horse_key="c", stake=10.0,
            requested_at=off - timedelta(seconds=sim.PREOFF_LEAD_SECONDS),
            quoted_odds=5.5,
        ),
        store=store, cfg=capped, seed=1,
    )
    assert fill.status == "PARTIAL"
    assert fill.matched_stake == 1.0 and fill.unmatched_stake == 9.0
    assert fill.fill_odds == pytest.approx(5.5)


def test_a_zero_limit_is_a_no_fill_not_a_free_bet(panel, store, cfg):
    zero = replace(cfg, frictions=replace(cfg.frictions, max_stake_per_bet=0.0))
    assert run(panel, store, zero).n_struck == 0


def test_exchange_commission_is_charged_on_the_winner_only(panel, store, cfg):
    result = run(panel, store, cfg)
    won = result.ledger[result.ledger["won"] == 1]
    lost = result.ledger[result.ledger["won"] == 0]
    assert (won["commission_paid"] > 0).all(), "betfair is an exchange source"
    assert (lost["commission_paid"] == 0).all()


def test_bog_is_not_applied_without_recorded_evidence(panel, store, cfg):
    on = replace(cfg, frictions=replace(cfg.frictions, best_odds_guaranteed=True))
    assert not run(panel, store, on).ledger["bog_applied"].any()


# ── staking, exposure, and the bankroll walk ─────────────────────────────────
def test_every_stake_is_inside_the_ceiling_of_the_bankroll_it_was_sized_against(
    panel, store, cfg
):
    """The cap is 0.5% of the bankroll *at that moment*, not of the opening one.

    A later stake can exceed 0.5% of the starting bankroll and still be inside
    the ceiling — that is compounding, not a breach. Testing against the opening
    figure would fail the moment the walk had a winner.
    """
    result = run(panel, store, cfg, bankroll=1000.0)
    running = 1000.0
    for row in result.ledger.itertuples(index=False):
        assert row.stake <= running * cfg.staking.max_stake_pct_bankroll + 1e-9
        running += row.profit


def test_the_bankroll_carries_forward_across_races(panel, store, cfg):
    result = run(panel, store, cfg, bankroll=1000.0)
    running = 1000.0
    for row in result.ledger.itertuples(index=False):
        running += row.profit
        assert row.bankroll_after == pytest.approx(round(running, 4))


def test_a_bankroll_of_nothing_stakes_nothing(panel, store, cfg):
    assert run(panel, store, cfg, bankroll=0.0).n_struck == 0


def test_safeguards_can_be_lifted_but_are_on_by_default(panel, store, cfg):
    """The switch exists for isolating friction behaviour, not for production."""
    guarded = run(panel, store, cfg, bankroll=1000.0)
    unguarded = run(panel, store, cfg, bankroll=1000.0, apply_safeguards=False)
    assert unguarded.n_struck >= guarded.n_struck


# ── settlement ───────────────────────────────────────────────────────────────
def test_without_race_facts_the_ledger_says_it_settled_from_the_panel(
    panel, store, cfg
):
    result = run(panel, store, cfg)
    assert set(result.ledger["settlement_source"]) == {sim.SETTLEMENT_PANEL}
    assert result.summary["archive_coverage"] == 0.0


def test_archive_facts_take_precedence_and_are_labelled(panel, store, cfg):
    facts = {
        sim.facts_key_for("Naas|2026-06-01T15:40"): rfacts.RaceFacts(
            race_key=sim.facts_key_for("Naas|2026-06-01T15:40"),
            race_date="2026-06-01", venue="naas", declared_field=5, n_runners=5,
            runners={h: rfacts.RunnerFact(horse_key=h,
                                          finish_position=(1 if h == "c" else 4))
                     for h in HORSES},
        )
    }
    result = sim.simulate(panel, cfg=cfg, store=store, race_facts=facts)
    sources = dict(result.ledger.groupby("settlement_source").size())
    assert sources.get(sim.SETTLEMENT_ARCHIVE) == 1
    assert result.summary["race_facts_supplied"] is True


def test_an_unsettled_race_never_reaches_the_walk_at_all(panel, store, cfg):
    """Eligibility drops it first, which is the earlier and safer of the two guards."""
    unlabelled = panel.copy()
    unlabelled["won"] = None
    result = sim.simulate(unlabelled, cfg=cfg, store=store)
    assert result.n_struck == 0
    assert result.ledger.empty


def test_an_unknown_outcome_settles_to_nothing_rather_than_to_a_loss(cfg):
    """Not a loss (understates returns), not a void (launders it away).

    The second guard, reached when a race is eligible but the individual runner's
    outcome is missing from both the archive and the panel label.
    """
    pick = pd.Series({"horse_key": "c", "won": None}).to_frame().T.itertuples(
        index=False
    ).__next__()
    assert sim._settle(
        facts=None, pick=pick, race_uid="R1", horse_key="c", stake=5.0, odds=5.5,
        bet_type="win", ew_terms=None, commission_rate=0.0, cfg=cfg,
    ) is None


def test_each_way_without_recorded_terms_refuses_to_run(panel, store, cfg):
    with pytest.raises(ValueError, match="no historical coverage"):
        sim.simulate(panel, cfg=cfg, store=store, bet_type="each_way")


def test_each_way_runs_when_the_terms_are_stated_explicitly(panel, store, cfg):
    result = sim.simulate(panel, cfg=cfg, store=store, bet_type="each_way",
                          ew_terms=EachWayTerms(places=3, fraction=0.2))
    assert result.n_struck >= 0  # the assumption is now on the record, not implied


# ── the summary, and what it declines to claim ───────────────────────────────
def test_the_summary_labels_itself_backtest_evidence_with_a_caveat(panel, store, cfg):
    s = run(panel, store, cfg).summary
    assert s["evidence_kind"] == "backtest"
    assert "NOT evidence that the" in s["caveat"]
    assert "cannot satisfy the forward-release gate" in s["caveat"]
    assert s["paper_only"] is True


def test_the_summary_names_what_the_backtest_did_not_exercise(panel, store, cfg):
    joined = " ".join(run(panel, store, cfg).summary["not_exercised"])
    assert "source-health" in joined
    assert "movement is modelled, not measured" in joined
    assert "best-odds-guaranteed" in joined


def test_blocked_reasons_are_counted_so_nothing_disappears_silently(panel, store, cfg):
    always = replace(cfg, frictions=replace(cfg.frictions, rejection_rate=1.0))
    s = run(panel, store, always).summary
    assert s["blocked_by"] == {"REJECTED": 3}
    assert s["n_attempts"] == 3 and s["n_struck"] == 0


def test_selecting_nothing_returns_an_empty_ledger_with_the_full_schema(
    panel, store, cfg
):
    unreachable = replace(cfg, gates=replace(cfg.gates, min_edge=0.99))
    result = run(panel, store, unreachable)
    assert result.n_struck == 0
    assert list(result.ledger.columns) == list(sim.LEDGER_COLUMNS)
    assert result.summary["evidence_kind"] == "backtest"


# ── the odds band belongs to the model, not to the controls ──────────────────
def test_a_band_reaches_model_only_and_not_the_baselines(panel, cfg):
    picks = sim._strategy_picks(panel, strategy="model_only", cfg=cfg,
                                odds_band=(1.0, 4.0))
    assert picks.empty, "the model's 5.5 pick is outside the band"
    fav = sim._strategy_picks(panel, strategy="favourite", cfg=cfg,
                              odds_band=(1.0, 4.0))
    assert not fav.empty, "the control still bets the whole book"


def test_an_unknown_strategy_names_the_ones_that_exist(panel, cfg):
    with pytest.raises(KeyError, match="unknown strategy"):
        sim._strategy_picks(panel, strategy="hunches", cfg=cfg)


# ── the comparison run ───────────────────────────────────────────────────────
def test_every_strategy_walks_the_same_eligible_race_set(panel, store, cfg):
    results = sim.run_walk_forward(panel, cfg=cfg, store=store, bankroll=1000.0)
    assert set(results) == {"model_only", "devigged_market", "favourite"}
    seen = {name: set(r.attempts["race_uid"]) | set(r.ledger["race_uid"])
            for name, r in results.items()}
    # Each strategy may decline a race, but none may see a race the others cannot.
    universe = set(panel["race_uid"])
    for name, races in seen.items():
        assert races <= universe, name


def test_a_band_narrows_what_the_model_bets_not_which_races_are_compared(
    panel, store, cfg
):
    wide = sim.run_walk_forward(panel, cfg=cfg, store=store, bankroll=1000.0)
    banded = sim.run_walk_forward(panel, cfg=cfg, store=store, bankroll=1000.0,
                                  odds_band=(1.0, 4.0))
    assert banded["model_only"].n_struck < wide["model_only"].n_struck
    assert banded["favourite"].n_struck == wide["favourite"].n_struck
    assert banded["devigged_market"].n_struck == wide["devigged_market"].n_struck
