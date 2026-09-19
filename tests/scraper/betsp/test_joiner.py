import pandas as pd
from scraper.betsp.results.base import ResultRow
from scraper.betsp import joiner


def _backbone():
    return pd.DataFrame([{
        "race_date": pd.Timestamp("2026-05-31T16:55", tz="Europe/Dublin"),
        "venue": "Nottingham", "horse_id": 96505154,
        "horse_name": "Sharp Romance (IRE)", "odds_finish": 2.51, "win_lose": 1,
        "distance": "1m2f", "market_type": "WIN", "region": "UK",
        "morningwap": 2.99, "ppwap": 2.49,
    }])


def test_normalize_horse_strips_country_and_punct():
    assert joiner._norm_horse("Sharp Romance (IRE)") == "sharp romance"
    assert joiner._norm_horse("O'Brien's Pride") == "obriens pride"


def test_join_fills_enrichment_on_match():
    rows = [ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                      horse_name="Sharp Romance", source="sporting_life",
                      jockey_id="oscar-456", trainer_id="charlie-789",
                      position=1, going="Good to Soft")]
    out = joiner.join(_backbone(), rows, priority=["sporting_life"])
    assert out.loc[0, "jockey_id"] == "oscar-456"
    assert out.loc[0, "position"] == 1
    assert out.loc[0, "going"] == "Good to Soft"
    assert out.loc[0, "result_source"] == "sporting_life"


def test_unmatched_backbone_keeps_nulls():
    out = joiner.join(_backbone(), [], priority=["sporting_life"])
    assert pd.isna(out.loc[0, "jockey_id"])
    assert pd.isna(out.loc[0, "result_source"])
    assert len(out) == 1


def test_source_priority_wins_conflict():
    rows = [
        ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                  horse_name="Sharp Romance", source="at_the_races", position=3),
        ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                  horse_name="Sharp Romance", source="racing_post", position=1),
    ]
    out = joiner.join(_backbone(), rows,
                      priority=["racing_post", "sporting_life", "at_the_races"])
    assert out.loc[0, "position"] == 1
    assert out.loc[0, "result_source"] == "racing_post"
