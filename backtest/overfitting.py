"""Statistical overfitting controls (ported from racing_ingestion).

Implements the Bailey / Borwein / López de Prado / Zhu toolkit for detecting and
quantifying backtest overfitting driven by multiple-testing:

  • expected_max_sharpe    — E[max SR] over N independent trials (true SR = 0).
  • minimum_backtest_length (MinBTL) — the minimum sample length required for an
                             observed in-sample Sharpe to be credible given N
                             optimisation trials.
  • probabilistic_sharpe_ratio (PSR) — P(true SR > benchmark), adjusting for
                             sample length and non-normality (skew/kurtosis).
  • deflated_sharpe_ratio (DSR) — PSR against the expected-max-Sharpe benchmark,
                             i.e. corrected for selection bias across N trials.

A high PSR/DSR (near 1) or an observed backtest longer than MinBTL does NOT prove
a strategy works — it proves the most common multiple-testing artefacts are not
obviously present.

References
----------
Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial
  Charlatanism", Notices of the AMS.
Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Mgmt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Euler–Mascheroni constant
_GAMMA = 0.5772156649015329


# ── Expected maximum Sharpe over N trials ─────────────────────────────────────

def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of N i.i.d. standard-normal Sharpe estimators (true SR=0).

        E[max SR_N] ≈ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e))

    This is the *standardised* expected max; multiply by the SR estimator's
    standard error (≈ 1/√T) to put it in the same units as an observed Sharpe.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be ≥ 1")
    if n_trials == 1:
        return 0.0
    e = math.e
    term1 = (1.0 - _GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
    term2 = _GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(term1 + term2)


# ── Minimum Backtest Length ───────────────────────────────────────────────────

@dataclass(frozen=True)
class MinBTLResult:
    n_trials: int
    target_annual_sharpe: float
    min_btl_years: float
    min_btl_race_units: int
    expected_max_sharpe_standardised: float
    observed_is_credible: bool
    message: str


def minimum_backtest_length(
    n_trials: int,
    target_annual_sharpe: float,
    races_per_year: float = 9000.0,
    observed_backtest_units: int | None = None,
) -> MinBTLResult:
    """Bailey-Prado Minimum Backtest Length.

    Given that *n_trials* configurations were tried during optimisation and the
    best was selected with annualised in-sample Sharpe *target_annual_sharpe*,
    MinBTL is the minimum sample length for that Sharpe to be distinguishable
    from a zero-skill strategy:

        MinBTL (years) ≈ ( E[max SR_N] / target_annual_sharpe )²

    With more trials, the best in-sample Sharpe found *by luck* grows, so a
    longer backtest is needed before a given Sharpe is meaningful. A backtest
    shorter than MinBTL is statistically indistinguishable from overfitting.

    Args:
        n_trials:                N — number of optimisation configurations tried.
        target_annual_sharpe:    The selected strategy's annualised Sharpe (> 0).
        races_per_year:          Bets/races per year; converts MinBTL years →
                                 race-units.
        observed_backtest_units: If given, compared against MinBTL to flag
                                 credibility.
    """
    if target_annual_sharpe <= 0:
        raise ValueError("target_annual_sharpe must be positive")

    e_max = expected_max_sharpe(n_trials)
    min_btl_years = (e_max / target_annual_sharpe) ** 2
    min_btl_units = int(math.ceil(min_btl_years * races_per_year))

    credible = True
    message = (
        f"With N={n_trials} trials, an annualised Sharpe of "
        f"{target_annual_sharpe:.2f} requires ≥ {min_btl_years:.2f} years "
        f"(~{min_btl_units:,} race-units) of backtest to be credible."
    )
    if observed_backtest_units is not None:
        credible = observed_backtest_units >= min_btl_units
        verdict = "CREDIBLE" if credible else "INSUFFICIENT — likely overfit"
        message += f"  Observed {observed_backtest_units:,} race-units → {verdict}."

    return MinBTLResult(
        n_trials=n_trials,
        target_annual_sharpe=target_annual_sharpe,
        min_btl_years=min_btl_years,
        min_btl_race_units=min_btl_units,
        expected_max_sharpe_standardised=e_max,
        observed_is_credible=credible,
        message=message,
    )


# ── Deflated / Probabilistic Sharpe Ratio ─────────────────────────────────────

def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark_sharpe).

        PSR = Φ( (SR − SR*) · √(n−1) / √(1 − skew·SR + ((kurt−1)/4)·SR²) )

    ``observed_sharpe`` is the per-period Sharpe (same period as ``n_obs``).
    """
    if n_obs < 2:
        return float("nan")
    denom = math.sqrt(
        max(1e-12, 1.0 - skew * observed_sharpe
            + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2)
    )
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    variance_of_trial_sharpes: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected-max-Sharpe benchmark.

        SR* = √Var(SR_trials) · E[max SR_N]
        DSR = PSR(observed_sharpe ; benchmark = SR*)

    DSR near 1 ⇒ the Sharpe is unlikely to be a multiple-testing artefact; DSR
    near 0.5 or below ⇒ it probably is.
    """
    sr_benchmark = math.sqrt(max(0.0, variance_of_trial_sharpes)) * expected_max_sharpe(n_trials)
    return probabilistic_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        benchmark_sharpe=sr_benchmark,
    )
