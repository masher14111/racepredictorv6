"""Venue currency and the EUR <-> GBP round trip.

The rule this pins down: odds are a ratio and are never converted; only stakes
and returns are. See utils.odds_format for the display side.
"""

import pytest

from utils.currency import GBP_EUR_RATE, currency_for_venue, from_eur, to_eur


@pytest.mark.parametrize("venue", ["Ascot", "ascot", " Wolverhampton ", "Newbury", "York"])
def test_uk_courses_are_gbp(venue):
    assert currency_for_venue(venue) == "GBP"


@pytest.mark.parametrize("venue", ["Leopardstown", "Curragh", "Galway", "Unknown Track"])
def test_irish_and_unknown_courses_are_eur(venue):
    assert currency_for_venue(venue) == "EUR"


@pytest.mark.parametrize("venue", ["Down Royal", "downpatrick", " Down Royal "])
def test_northern_irish_courses_are_eur(venue):
    """NI is politically UK but races under HRI and is priced in EUR."""
    assert currency_for_venue(venue) == "EUR"


def test_ni_stays_eur_even_when_the_feed_tags_it_gb():
    assert currency_for_venue("Down Royal", country_code="GB") == "EUR"


def test_country_code_overrides_an_unknown_venue():
    assert currency_for_venue("Some New Track", country_code="GB") == "GBP"
    assert currency_for_venue("Some New Track", country_code="IE") == "EUR"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def test_gbp_converts_to_eur_at_the_configured_rate():
    assert to_eur(10.0, "GBP") == pytest.approx(10.0 * GBP_EUR_RATE)


def test_eur_passes_through_untouched():
    assert to_eur(10.0, "EUR") == 10.0
    assert from_eur(10.0, "EUR") == 10.0


def test_round_trip_is_lossless():
    assert from_eur(to_eur(42.0, "GBP"), "GBP") == pytest.approx(42.0)


def test_from_eur_is_the_inverse_not_a_second_multiply():
    """Regression: the stake tooltip multiplied an already-EUR amount again,
    inflating UK returns by the rate."""
    assert from_eur(11.80, "GBP") == pytest.approx(11.80 / GBP_EUR_RATE)
    assert from_eur(11.80, "GBP") < 11.80
