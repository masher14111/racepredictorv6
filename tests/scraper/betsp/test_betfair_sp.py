import pandas as pd
from scraper.betsp import betfair_sp


SAMPLE_CSV = (
    "event_id,menu_hint,event_name,event_dt,selection_id,selection_name,win_lose,bsp,"
    "ppwap,morningwap,ppmax,ppmin,ipmax,ipmin,morningtradedvol,pptradedvol,iptradedvol\n"
    "258715928,Nottingham 31st May,1m2f Hcap,31-05-2026 16:55,96505154,Sharp Romance (IRE),1,2.51,"
    "2.49,2.99,3.0,2.46,2.74,1.01,3337.16,77465.12,38963.54\n"
    "258715934,Nottingham 31st May,5f Nov Stks,31-05-2026 17:25,69684787,Penelope Valentine,0,1001,"
    "8.80,11.3,14.5,8.4,1000.0,3.2,886.3,12135.31,5770.85\n"
)


def test_parse_distance_variants():
    assert betfair_sp._parse_distance("1m2f Hcap") == "1m2f"
    assert betfair_sp._parse_distance("5f Nov Stks") == "5f"
    assert betfair_sp._parse_distance("2m Chase") == "2m"
    assert betfair_sp._parse_distance("Maiden") is None


def test_parse_csv_maps_columns_and_nulls_bsp_1001():
    df = betfair_sp.parse_csv(SAMPLE_CSV, region="UK", market="WIN")
    assert list(df["horse_name"]) == ["Sharp Romance (IRE)", "Penelope Valentine"]
    assert df.loc[0, "horse_id"] == 96505154
    assert df.loc[0, "odds_finish"] == 2.51
    assert pd.isna(df.loc[1, "odds_finish"])
    assert list(df["win_lose"]) == [1, 0]
    assert df.loc[0, "distance"] == "1m2f"
    assert df.loc[0, "venue"] == "Nottingham"
    assert df.loc[0, "market_type"] == "WIN"
    assert df.loc[0, "region"] == "UK"


def test_parse_csv_race_date_is_dublin_aware():
    df = betfair_sp.parse_csv(SAMPLE_CSV, region="UK", market="WIN")
    ts = df.loc[0, "race_date"]
    assert ts.year == 2026 and ts.hour == 16 and ts.minute == 55
    assert ts.tzinfo is not None


def test_file_url():
    url = betfair_sp.file_url("uk", "win", 2026, 5, 31)
    assert url == (
        "https://promo.betfair.com/betfairsp/prices/dwbfpricesukwin31052026.csv"
    )
