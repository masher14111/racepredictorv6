import pandas as pd

from scraper.timeform import writer


def _row(**kw):
    base = dict(race_date=pd.Timestamp("2026-06-13", tz="UTC"), venue="York",
                race_time="2026-06-13T13:50", horse_name="A", source="timeform")
    base.update(kw)
    return base


def test_write_partitions_by_year_and_dedupes(tmp_path):
    path = str(tmp_path / "timeform.parquet")
    writer.write(pd.DataFrame([_row(timeform_rating=100)]), path=path)
    # same dedupe key, newer value -> last wins, not duplicated
    writer.write(pd.DataFrame([_row(timeform_rating=120)]), path=path)
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["timeform_rating"] == 120
    assert df.iloc[0]["year"] == 2026


def test_write_empty_is_noop(tmp_path):
    path = str(tmp_path / "tf.parquet")
    writer.write(pd.DataFrame(), path=path)  # must not raise
