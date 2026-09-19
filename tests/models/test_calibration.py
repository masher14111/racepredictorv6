"""Tests for models/calibration.py."""
import numpy as np
import pytest

from models.calibration import (
    IsotonicCalibrator,
    OddsBandCalibrator,
    SigmoidCalibrator,
    brier_score,
    fit_calibrator,
    normalize_within_race,
)


# ── SigmoidCalibrator ─────────────────────────────────────────────────────────

class TestSigmoidCalibrator:
    def test_predict_in_unit_interval(self):
        cal = SigmoidCalibrator(a=1.0, b=0.0)
        out = cal.predict(np.array([0.0, 0.01, 0.5, 0.99, 1.0]))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_identity_when_a1_b0(self):
        # sigmoid(logit(p)) == p
        cal = SigmoidCalibrator(a=1.0, b=0.0)
        p = np.array([0.1, 0.3, 0.6, 0.9])
        assert cal.predict(p) == pytest.approx(p, abs=1e-9)

    def test_monotonic_for_positive_a(self):
        cal = SigmoidCalibrator.fit(
            np.linspace(0.01, 0.99, 200),
            (np.linspace(0.01, 0.99, 200) > 0.5).astype(int),
        )
        p = np.linspace(0.01, 0.99, 50)
        out = cal.predict(p)
        assert np.all(np.diff(out) >= -1e-9)  # non-decreasing

    def test_fit_corrects_inflation(self):
        # Raw probs centred ~0.6 but true positive rate ~0.3 → sigmoid pulls down.
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.3).astype(int)
        raw = np.clip(0.6 + rng.normal(0, 0.05, 2000), 0.01, 0.99)
        cal = SigmoidCalibrator.fit(raw, y)
        assert cal.predict(np.array([0.6]))[0] < 0.5


# ── brier_score ───────────────────────────────────────────────────────────────

class TestBrier:
    def test_perfect_predictions_zero(self):
        y = np.array([1, 0, 1, 0])
        assert brier_score(y, y.astype(float)) == pytest.approx(0.0)

    def test_known_value(self):
        # constant 0.5 vs balanced labels → MSE = 0.25
        y = np.array([1, 0, 1, 0])
        assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)


# ── fit_calibrator ────────────────────────────────────────────────────────────

class TestFitCalibrator:
    def _data(self, n=4000, seed=1):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.25).astype(int)
        raw = np.clip(0.5 + rng.normal(0, 0.1, n), 0.01, 0.99)
        return raw, y

    def test_isotonic_returns_isotonic(self):
        raw, y = self._data()
        cal, method = fit_calibrator("isotonic", raw, y)
        assert method == "isotonic"
        assert isinstance(cal, IsotonicCalibrator)

    def test_isotonic_calibrator_pickles_without_numpy_internals(self):
        # The whole point of the wrapper: the pickle must carry only Python
        # floats, so it loads regardless of the numpy major version. Guard that
        # no numpy object sneaks into the serialized state.
        import pickle

        raw, y = self._data()
        cal, _ = fit_calibrator("isotonic", raw, y)
        blob = pickle.dumps(cal)
        assert b"numpy" not in blob
        reloaded = pickle.loads(blob)
        assert reloaded.predict(raw) == pytest.approx(cal.predict(raw))

    def test_sigmoid_returns_sigmoid(self):
        raw, y = self._data()
        cal, method = fit_calibrator("sigmoid", raw, y)
        assert method == "sigmoid"
        assert isinstance(cal, SigmoidCalibrator)

    def test_auto_picks_a_supported_method(self):
        raw, y = self._data()
        cal, method = fit_calibrator("auto", raw, y)
        assert method in ("isotonic", "sigmoid")

    def test_auto_falls_back_to_isotonic_when_tiny(self):
        raw, y = self._data(n=100)  # below _MIN_SELECT_ROWS val fold
        cal, method = fit_calibrator("auto", raw, y)
        assert method == "isotonic"

    def test_unknown_method_raises(self):
        raw, y = self._data(n=300)
        with pytest.raises(ValueError):
            fit_calibrator("quantile", raw, y)

    def test_calibrator_improves_brier(self):
        # Inflated raw probs; any fitted calibrator should lower test Brier.
        raw, y = self._data(n=4000)
        cal, _ = fit_calibrator("auto", raw, y)
        assert brier_score(y, cal.predict(raw)) < brier_score(y, raw)


# ── OddsBandCalibrator ────────────────────────────────────────────────────────

class TestOddsBandCalibrator:
    @staticmethod
    def _biased_fixture(seed=0, n=4000):
        """A model with favourite-longshot bias: within each price band higher
        model prob → higher true win rate (monotone), but the model *under*-rates
        favourites (true >> pred) and *over*-rates longshots (true << pred)."""
        rng = np.random.default_rng(seed)
        # favourites: odds in [2,4); model ~0.35 but truth ~0.6 (under-rated)
        u_f = rng.random(n)
        odds_f = 2.0 + 2.0 * rng.random(n)
        prob_f = np.clip(0.20 + 0.30 * u_f + rng.normal(0, 0.02, n), 0.01, 0.99)
        true_f = np.clip(0.40 + 0.40 * u_f, 0.01, 0.99)
        # longshots: odds in [8,30); model ~0.10 but truth ~0.035 (over-rated)
        u_l = rng.random(n)
        odds_l = 8.0 + 22.0 * rng.random(n)
        prob_l = np.clip(0.07 + 0.06 * u_l + rng.normal(0, 0.01, n), 0.01, 0.99)
        true_l = np.clip(0.02 + 0.03 * u_l, 0.001, 0.99)

        odds = np.concatenate([odds_f, odds_l])
        prob = np.concatenate([prob_f, prob_l])
        true = np.concatenate([true_f, true_l])
        y = (rng.random(2 * n) < true).astype(int)
        return prob, odds, y

    def _fit(self, **kw):
        prob, odds, y = self._biased_fixture()
        cal = OddsBandCalibrator.fit(prob, odds, y, edges=[1.0, 5.0, float("inf")], **kw)
        return cal, prob, odds, y

    def test_monotone_in_prob_at_fixed_odds(self):
        cal, *_ = self._fit()
        for odds in (3.0, 12.0, 25.0):
            out = cal.predict(np.linspace(0.02, 0.7, 60), np.full(60, odds))
            assert np.all(np.diff(out) >= -1e-9)  # non-decreasing within a price

    def test_corrects_favorite_longshot_bias(self):
        cal, prob, odds, y = self._fit()
        fav = odds < 5.0
        lng = odds >= 5.0
        # before: favourites under-predicted (A/E>1), longshots over-predicted (<1)
        ae_fav_before = y[fav].sum() / prob[fav].sum()
        ae_lng_before = y[lng].sum() / prob[lng].sum()
        assert ae_fav_before > 1.3 and ae_lng_before < 0.7
        # after: both A/E flatten toward 1.0
        cp = cal.predict(prob, odds)
        ae_fav_after = y[fav].sum() / cp[fav].sum()
        ae_lng_after = y[lng].sum() / cp[lng].sum()
        assert abs(ae_fav_after - 1.0) < 0.1
        assert abs(ae_lng_after - 1.0) < 0.1
        # and overall Brier improves
        assert brier_score(y, cp) < brier_score(y, prob)

    def test_continuous_across_band_edges(self):
        cal, *_ = self._fit()
        # sweep price finely at a fixed prob: output must not jump (blended in
        # log-odds, so the curve is continuous everywhere).
        sweep = np.linspace(2.0, 30.0, 2000)
        out = cal.predict(np.full(sweep.size, 0.2), sweep)
        assert np.max(np.abs(np.diff(out))) < 0.02

    def test_pickles_without_numpy_internals(self):
        import pickle
        cal, prob, odds, _ = self._fit()
        blob = pickle.dumps(cal)
        assert b"numpy" not in blob
        reloaded = pickle.loads(blob)
        assert reloaded.predict(prob, odds) == pytest.approx(cal.predict(prob, odds))

    def test_scalar_input_returns_scalar(self):
        cal, *_ = self._fit()
        out = cal.predict(0.2, 3.0)
        assert np.ndim(out) == 0
        assert 0.0 <= float(out) <= 1.0

    def test_output_in_unit_interval(self):
        cal, prob, odds, _ = self._fit()
        out = cal.predict(prob, odds)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_skips_sparse_bands(self):
        # A band thinner than min_rows is dropped; survivors still cover its range.
        prob, odds, y = self._biased_fixture()
        cal = OddsBandCalibrator.fit(prob, odds, y,
                                     edges=[1.0, 5.0, 5.5, float("inf")])
        # the [5.0,5.5) band is ~empty (longshots start at 8) → not fitted
        assert len(cal.calibrators) == 2

    def test_raises_when_no_band_fittable(self):
        prob, odds, y = self._biased_fixture(n=50)
        with pytest.raises(ValueError):
            OddsBandCalibrator.fit(prob, odds, y, min_rows=10_000)

    def test_single_band_ignores_odds(self):
        prob, odds, y = self._biased_fixture()
        cal = OddsBandCalibrator.fit(prob, odds, y, edges=[1.0, float("inf")])
        assert len(cal.calibrators) == 1
        # one band → prediction depends only on prob, not the price passed in
        a = cal.predict(0.2, 3.0)
        b = cal.predict(0.2, 25.0)
        assert float(a) == pytest.approx(float(b))


# ── normalize_within_race ─────────────────────────────────────────────────────

class TestNormalizeWithinRace:
    def test_each_group_sums_to_one(self):
        probs = [0.4, 0.3, 0.3, 0.6, 0.6]
        groups = ["A", "A", "A", "B", "B"]
        out = normalize_within_race(probs, groups)
        assert out[:3].sum() == pytest.approx(1.0)
        assert out[3:].sum() == pytest.approx(1.0)

    def test_preserves_within_group_ranking(self):
        probs = np.array([0.5, 0.2, 0.1])
        out = normalize_within_race(probs, ["R", "R", "R"])
        assert np.argsort(out)[::-1].tolist() == [0, 1, 2]

    def test_proportions_preserved(self):
        # 0.2 and 0.6 in same race → 0.25 / 0.75 after normalization
        out = normalize_within_race([0.2, 0.6], ["R", "R"])
        assert out == pytest.approx([0.25, 0.75])

    def test_zero_sum_group_left_unchanged(self):
        out = normalize_within_race([0.0, 0.0], ["R", "R"])
        assert out == pytest.approx([0.0, 0.0])

    def test_nan_prob_skipped_in_group_sum(self):
        # A NaN prob (predict_proba failed for that runner) is excluded from the
        # group total; the remaining known prob normalizes against that total.
        out = normalize_within_race([np.nan, 0.3], ["R", "R"])
        assert np.isnan(out[0]) and out[1] == pytest.approx(1.0)

    def test_reanchors_saturated_overconfident_field(self):
        """The calib-fl-01 headline fix relies on this: a field of identical,
        saturated (overconfident) marginals re-anchors to sum-to-1 — each runner
        to 1/field_size — well below the inflated marginal. This is why the
        normalized prob is well-calibrated OOS where the raw marginal is not."""
        field = [0.73] * 8  # every runner pinned at the v3 isotonic ceiling
        out = normalize_within_race(field, ["R"] * 8)
        assert out.sum() == pytest.approx(1.0)
        assert out == pytest.approx([1 / 8] * 8)
        assert out.max() < 0.73
