import math


def test_eur_passthrough():
    from utils.currency import to_eur
    assert to_eur(10.0, "EUR") == 10.0


def test_gbp_converts_with_config_rate():
    from utils.currency import to_eur
    # default config gbp_eur_rate = 1.18
    assert math.isclose(to_eur(10.0, "GBP"), 11.8, rel_tol=1e-9)


def test_unknown_currency_passthrough():
    from utils.currency import to_eur
    assert to_eur(5.0, "USD") == 5.0


def test_uk_course_detected_as_gbp():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Ascot") == "GBP"
    assert currency_for_venue("ascot") == "GBP"


def test_irish_course_detected_as_eur():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Leopardstown") == "EUR"


def test_unknown_venue_defaults_eur():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Nowhere Park") == "EUR"


def test_country_code_overrides_venue():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Nowhere Park", country_code="GB") == "GBP"
    assert currency_for_venue("Ascot", country_code="IE") == "EUR"
