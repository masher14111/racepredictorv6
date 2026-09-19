import pandas as pd

from features.labels import add_labels


def test_labels_win_and_place():
    df = pd.DataFrame({"position": [1, 2, 4, None]})
    out = add_labels(df, place_positions=3)
    assert list(out["won"][:3]) == [1, 0, 0]
    assert list(out["placed"][:3]) == [1, 1, 0]


def test_labels_null_position_gives_null_labels():
    df = pd.DataFrame({"position": [None]})
    out = add_labels(df, place_positions=3)
    assert pd.isna(out.iloc[0]["won"])
    assert pd.isna(out.iloc[0]["placed"])
