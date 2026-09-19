"""Tests for backtest.integrity — the post-hoc backtest credibility suite.

The fixtures are deterministic (seeded numpy) bet ledgers in v4's schema:
``race_date, won, stake, profit, bet_price, close_price`` plus ``venue`` and a
matched-volume column. They exercise the three headline outcomes the suite
exists to catch: a clean ledger (all OK), a look-ahead-leaky ledger
(lookahead FAIL), and a single-venue ledger (course cherry-pick FAIL).
"""
import numpy as np
import pandas as pd
import pytest

from backtest import integrity


# ── fixtures ──────────────────────────────────────────────────────────────────

def _settle(stake, price, won):
    """Net profit of a flat back-bet — mirrors metrics.settle (no commission)."""
    return np.where(won >= 0.5, stake * (price - 1.0), -stake)


def _build_ledger(rng, *, n=260, venues=8, start="2023-06-01", single_venue=False,
                  clv_edge=0.0):
    """A spread-out, mildly-profitable flat-stake ledger.

    ``clv_edge``: shifts close_price below bet_price by this log amount, raising
    mean log CLV (used to exercise the leakage check).
    """
    days = rng.integers(0, 360, size=n)
    race_date = pd.to_datetime(start) + pd.to_timedelta(np.sort(days), unit="D")
    venue = (np.full(n, "Ascot") if single_venue
             else np.array([f"Course{i}" for i in rng.integers(0, venues, size=n)]))

    # Prices spread across the odds bands (favourites → longshots).
    bet_price = rng.uniform(1.8, 30.0, size=n)
    implied = 1.0 / bet_price
    # Win with a small positive edge over the implied prob so the book is beatable
    # but profit is not concentrated in any one slice.
    won = (rng.uniform(0, 1, size=n) < implied * 1.08).astype(int)

    stake = np.full(n, 10.0)
    profit = _settle(stake, bet_price, won)
    # Close near the bet price; clv_edge makes us beat the close on average.
    close_price = bet_price / np.exp(clv_edge + rng.normal(0, 0.01, size=n))

    return pd.DataFrame({
        "race_date": race_date,
        "venue": venue,
        "won": won,
        "stake": stake,
        "profit": profit,
        "bet_price": bet_price,
        "close_price": close_price,
        "bf_matched_volume": rng.uniform(5_000, 50_000, size=n),
    })


@pytest.fixture
def clean_ledger():
    return _build_ledger(np.random.default_rng(7))


# ── clean ledger: everything OK ───────────────────────────────────────────────

class TestCleanLedger:
    def test_all_ok(self, clean_ledger):
        # train_cutoff well before the first bet → look-ahead guard satisfied.
        results = integrity.run_all_integrity_checks(
            clean_ledger, train_cutoff="2023-01-01")
        bad = [r for r in results if r["status"] != "OK"]
        assert not bad, f"expected all OK, got: {[(r['name'], r['status']) for r in bad]}"

    def test_returns_structured_results(self, clean_ledger):
        results = integrity.run_all_integrity_checks(clean_ledger)
        assert isinstance(results, list)
        for r in results:
            assert set(("name", "status", "message")).issubset(r)
            assert r["status"] in ("OK", "WARN", "FAIL")

    def test_read_only(self, clean_ledger):
        before = clean_ledger.copy(deep=True)
        integrity.run_all_integrity_checks(clean_ledger, train_cutoff="2023-01-01")
        pd.testing.assert_frame_equal(clean_ledger, before)


# ── leaky ledger: look-ahead FAIL ─────────────────────────────────────────────

class TestLookaheadLeak:
    def test_first_bet_on_or_before_cutoff_fails(self, clean_ledger):
        # Cutoff after the first bet → the model could have trained on rows it bet.
        cutoff = clean_ledger["race_date"].min()  # first bet == cutoff (not strictly after)
        res = integrity.check_lookahead_bias(clean_ledger, train_cutoff=cutoff)
        assert res["status"] == "FAIL"

    def test_cutoff_inside_betting_window_fails(self, clean_ledger):
        cutoff = clean_ledger["race_date"].quantile(0.5)
        res = integrity.check_lookahead_bias(clean_ledger, train_cutoff=cutoff)
        assert res["status"] == "FAIL"

    def test_nan_date_fails(self, clean_ledger):
        leaky = clean_ledger.copy()
        leaky.loc[0, "race_date"] = pd.NaT
        res = integrity.check_lookahead_bias(leaky, train_cutoff="2023-01-01")
        assert res["status"] == "FAIL"

    def test_in_aggregate_run(self, clean_ledger):
        cutoff = clean_ledger["race_date"].quantile(0.5)
        results = integrity.run_all_integrity_checks(clean_ledger, train_cutoff=cutoff)
        look = next(r for r in results if r["name"] == "lookahead_bias")
        assert look["status"] == "FAIL"


# ── cherry-picked ledger: concentration WARN/FAIL ─────────────────────────────

class TestCherryPicking:
    def test_single_profitable_course_fails(self):
        """Profit comes from one venue, others bleed → course cherry-pick FAIL."""
        rng = np.random.default_rng(3)
        ledger = _build_ledger(rng, single_venue=False)
        # Force all profit into 'Course0' and make every other venue loss-making.
        winners = ledger["won"] == 1
        ledger.loc[winners, "venue"] = "Course0"
        ledger.loc[~winners, "venue"] = np.where(
            np.arange((~winners).sum()) % 2 == 0, "Course1", "Course2")
        res = integrity.check_course_cherrypicking(ledger)
        assert res["status"] == "FAIL"

    def test_concentrated_course_warns(self):
        """One venue holds the majority (>60%) of profit but others also profit."""
        rng = np.random.default_rng(11)
        rows = []
        # Course0: big net profit. Course1/2: small net profit.
        for venue, wins, losses in [("Course0", 40, 5), ("Course1", 6, 5), ("Course2", 6, 5)]:
            for _ in range(wins):
                rows.append((venue, 1, 10.0, 30.0))
            for _ in range(losses):
                rows.append((venue, 0, 10.0, -10.0))
        df = pd.DataFrame(rows, columns=["venue", "won", "stake", "profit"])
        df["race_date"] = pd.to_datetime("2023-06-01")
        res = integrity.check_course_cherrypicking(df)
        assert res["status"] == "WARN"
        assert res["detail"]["top_course_profit_share"] > 0.60

    def test_single_odds_band_fails(self):
        """All profit in one odds band, others lose → band cherry-pick FAIL."""
        rng = np.random.default_rng(5)
        rows = []
        # Profit only from the 2.0-4.0 band; the longshot band only loses.
        for _ in range(60):
            rows.append((3.0, 1, 10.0, 20.0))   # winners @ 3.0
        for _ in range(40):
            rows.append((3.0, 0, 10.0, -10.0))
        for _ in range(40):
            rows.append((20.0, 0, 10.0, -10.0))  # longshots, all lose
        df = pd.DataFrame(rows, columns=["bet_price", "won", "stake", "profit"])
        df["race_date"] = pd.to_datetime("2023-06-01")
        res = integrity.check_band_cherrypicking(df)
        assert res["status"] == "FAIL"


# ── per-check behaviour ───────────────────────────────────────────────────────

class TestLiquidity:
    def test_missing_volume_warns(self, clean_ledger):
        res = integrity.check_liquidity(clean_ledger.drop(columns=["bf_matched_volume"]))
        assert res["status"] == "WARN"
        assert "liquidity unknown" in res["message"].lower()

    def test_present_volume_ok(self, clean_ledger):
        assert integrity.check_liquidity(clean_ledger)["status"] == "OK"

    def test_market_impact_warns(self, clean_ledger):
        impacted = clean_ledger.copy()
        impacted["bf_matched_volume"] = 1.0  # any €10 stake now dwarfs the volume
        assert integrity.check_liquidity(impacted)["status"] == "WARN"


class TestClvLeakage:
    def test_high_clv_fails(self):
        rng = np.random.default_rng(9)
        # Mean log CLV well above the FAIL threshold (0.15).
        ledger = _build_ledger(rng, clv_edge=0.20)
        res = integrity.check_clv_leakage(ledger)
        assert res["status"] == "FAIL"
        assert res["detail"]["mean_clv_log"] > integrity._CLV_FAIL

    def test_modest_clv_ok(self, clean_ledger):
        assert integrity.check_clv_leakage(clean_ledger)["status"] == "OK"


class TestMinimumBacktestLength:
    def test_too_few_bets_fails(self):
        rng = np.random.default_rng(1)
        small = _build_ledger(rng, n=50)
        res = integrity.check_minimum_backtest_length(small)
        assert res["status"] == "FAIL"

    def test_enough_bets_credible_with_one_trial(self, clean_ledger):
        # N=1 trial ⇒ MinBTL is 0 ⇒ any positive-Sharpe ledger is credible.
        res = integrity.check_minimum_backtest_length(clean_ledger, n_trials=1)
        assert res["status"] == "OK"

    def test_many_trials_thin_edge_warns(self):
        """A thin-edge strategy (low per-bet Sharpe) with thousands of
        optimisation trials cannot clear its Minimum Backtest Length → WARN.

        Evens-money bets with a slim positive edge: 116/220 winners gives a tiny
        but positive Sharpe — exactly the regime MinBTL is built to flag once the
        multiple-testing penalty (n_trials) is large."""
        n, n_wins = 220, 116
        won = np.array([1] * n_wins + [0] * (n - n_wins))
        stake = np.full(n, 10.0)
        price = np.full(n, 2.0)
        ledger = pd.DataFrame({
            "race_date": pd.to_datetime("2023-06-01")
            + pd.to_timedelta(np.arange(n) % 20, unit="D"),
            "won": won, "stake": stake, "bet_price": price,
            "profit": _settle(stake, price, won),
        })
        res = integrity.check_minimum_backtest_length(ledger, n_trials=5000)
        assert res["detail"]["annual_sharpe"] > 0  # a positive edge to defend
        assert res["status"] == "WARN"
        assert not res["detail"]["observed_is_credible"]


class TestEmptyLedger:
    def test_empty_is_all_ok(self):
        empty = pd.DataFrame(columns=["race_date", "won", "stake", "profit",
                                      "bet_price", "close_price"])
        results = integrity.run_all_integrity_checks(empty)
        # Empty ledger should never FAIL — nothing to flag.
        assert all(r["status"] in ("OK", "WARN") for r in results)
