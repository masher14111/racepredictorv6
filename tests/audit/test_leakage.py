"""Stage-4 leakage guards: each check must catch a planted defect and pass a
clean frame."""
import numpy as np
import pandas as pd

from audit import leakage


def _panel(n=600, seed=7):
    rng = np.random.default_rng(seed)
    ppwap = rng.uniform(2.0, 20.0, n)
    morningwap = ppwap * rng.uniform(0.9, 1.1, n)
    odds_finish = ppwap * np.exp(rng.normal(0, 0.15, n))
    won = (rng.uniform(0, 1, n) < 1.0 / ppwap).astype(int)
    return pd.DataFrame({
        "ppwap": ppwap, "morningwap": morningwap, "odds_finish": odds_finish,
        "won": won,
        "clean_feature": rng.normal(0, 1, n),
    })


class TestProvenance:
    def test_current_whitelist_passes(self):
        assert leakage.check_provenance().passed

    def test_price_free_list_is_disjoint_from_price_cols(self):
        from models.features import PRICE_FREE_FEATURE_COLS
        assert not (set(PRICE_FREE_FEATURE_COLS)
                    & leakage.PRICE_OR_POSTOFF_COLS)
        assert not (set(PRICE_FREE_FEATURE_COLS) & leakage.OUTCOME_COLS)


class TestEmpiricalGuards:
    def test_clean_feature_passes(self):
        df = _panel()
        res = leakage.check_empirical(df, cols=["clean_feature"])
        assert res.passed

    def test_outcome_copy_flagged(self):
        df = _panel()
        df["leaky"] = df["won"] + np.random.default_rng(0).normal(0, 0.05, len(df))
        res = leakage.check_empirical(df, cols=["leaky"])
        assert not res.passed
        assert any("corr won" in f for f in res.failures)

    def test_closing_price_feature_flagged(self):
        df = _panel()
        # A feature built from the finishing price retains the closing move
        # after the pre-off basis is partialled out.
        df["leaky_close"] = np.log(df["odds_finish"])
        res = leakage.check_empirical(df, cols=["leaky_close"])
        assert not res.passed
        assert any("closing-move" in f for f in res.failures)

    def test_pure_preoff_price_function_passes(self):
        df = _panel()
        df["preoff_fn"] = 1.0 / df["morningwap"]
        res = leakage.check_empirical(df, cols=["preoff_fn"])
        assert res.passed


class TestAppendInvariance:
    def _matrices(self, drift=False):
        n = 40
        base = pd.DataFrame({
            "race_uid": [f"r{i//4}" for i in range(n)],
            "horse_id": [f"h{i}" for i in range(n)],
            "market_type": ["WIN"] * n,
            "race_date": pd.to_datetime("2026-01-01", utc=True)
            + pd.to_timedelta(np.arange(n) // 4, unit="D"),
            "feat": np.linspace(0.0, 1.0, n),
        })
        after = pd.concat([base, base.iloc[[0]].assign(
            race_uid="rX", horse_id="hX",
            race_date=pd.to_datetime("2026-03-01", utc=True))],
            ignore_index=True)
        if drift:
            after = after.copy()
            after.loc[after.index[:n], "feat"] = after.loc[after.index[:n], "feat"] + 0.01
        return base, after

    def test_invariant_features_pass(self):
        before, after = self._matrices(drift=False)
        res = leakage.check_append_invariance(before, after, cols=["feat"])
        assert res.passed
        assert res.details["n_overlap"] == 40

    def test_future_dependent_feature_fails(self):
        before, after = self._matrices(drift=True)
        res = leakage.check_append_invariance(before, after, cols=["feat"])
        assert not res.passed
        assert "feat" in res.details["drift"]


class TestWeightChannel:
    def test_preoff_weight_price_passes(self):
        df = _panel()
        df["implied_prob"] = 1.0 / df["morningwap"]
        assert leakage.check_weight_channel(df).passed

    def test_postoff_weight_price_fails(self):
        df = _panel()
        df["implied_prob"] = 1.0 / df["odds_finish"]
        assert not leakage.check_weight_channel(df).passed
