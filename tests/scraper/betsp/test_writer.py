import pandas as pd
from scraper.betsp import writer


def _df(horse_id, year=2026):
    return pd.DataFrame([{
        "race_date": pd.Timestamp(f"{year}-05-31T16:55", tz="Europe/Dublin"),
        "venue": "Nottingham", "horse_id": horse_id, "horse_name": "X",
        "jockey_id": None, "trainer_id": None, "odds_finish": 2.5, "position": None,
        "win_lose": 1, "going": None, "distance": "1m2f", "market_type": "WIN",
        "region": "UK", "morningwap": 2.9, "ppwap": 2.4, "result_source": None,
        "fetched_at": pd.Timestamp("2026-06-12", tz="UTC"),
    }])


def test_write_creates_year_partition(tmp_path):
    path = str(tmp_path / "betsp.parquet")
    writer.write(_df(1), path=path)
    back = pd.read_parquet(path)
    assert "year" in back.columns and set(back["year"]) == {2026}
    assert len(back) == 1


def test_write_dedupes_on_key(tmp_path):
    path = str(tmp_path / "betsp.parquet")
    writer.write(_df(1), path=path)
    writer.write(_df(1), path=path)
    back = pd.read_parquet(path)
    assert len(back) == 1


def test_write_merges_distinct_rows(tmp_path):
    path = str(tmp_path / "betsp.parquet")
    writer.write(_df(1), path=path)
    writer.write(_df(2), path=path)
    back = pd.read_parquet(path)
    assert len(back) == 2


def test_write_never_regresses_a_resolved_position_to_null(tmp_path):
    """A resumable re-fetch of an already-covered day (e.g. a partial results-site
    enrichment miss) must not overwrite an already-resolved position with null."""
    path = str(tmp_path / "betsp.parquet")
    resolved = _df(1)
    resolved["position"] = 3
    resolved["fetched_at"] = pd.Timestamp("2026-06-12", tz="UTC")
    writer.write(resolved, path=path)

    unresolved_refetch = _df(1)
    unresolved_refetch["position"] = None
    unresolved_refetch["fetched_at"] = pd.Timestamp("2026-06-15", tz="UTC")  # later, but worse
    writer.write(unresolved_refetch, path=path)

    back = pd.read_parquet(path)
    assert len(back) == 1
    assert int(back.loc[0, "position"]) == 3


def test_write_lets_a_later_resolved_position_win_over_an_earlier_one(tmp_path):
    """Among two equally-complete rows on the same key, the freshest fetch wins
    (unchanged prior behaviour) — only a regression to null is blocked."""
    path = str(tmp_path / "betsp.parquet")
    first = _df(1)
    first["position"] = 3
    first["fetched_at"] = pd.Timestamp("2026-06-12", tz="UTC")
    writer.write(first, path=path)

    corrected = _df(1)
    corrected["position"] = 1  # e.g. a stewards' inquiry amendment
    corrected["fetched_at"] = pd.Timestamp("2026-06-15", tz="UTC")
    writer.write(corrected, path=path)

    back = pd.read_parquet(path)
    assert len(back) == 1
    assert int(back.loc[0, "position"]) == 1
