from utils.text_norm import norm_venue, norm_horse, minute_key


def test_norm_horse_strips_country_and_punct():
    assert norm_horse("Prince Of The Seas (IRE)") == "prince of the seas"


def test_norm_venue_lowercases_and_strips():
    assert norm_venue("Sandown-Park") == "sandownpark"


def test_minute_key_truncates_iso_to_minute():
    assert minute_key("2026-06-13T13:50:21+01:00") == "2026-06-13T13:50"
