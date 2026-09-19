from models.features import FEATURE_COLS


def test_feature_cols_is_list_of_strings():
    assert isinstance(FEATURE_COLS, list)
    assert all(isinstance(c, str) for c in FEATURE_COLS)
    assert len(FEATURE_COLS) > 0


def test_no_leakage_cols():
    leakage = {"position", "won", "placed", "placed_2", "showed", "win_lose", "sp"}
    assert leakage.isdisjoint(set(FEATURE_COLS)), \
        f"Leakage columns found in FEATURE_COLS: {leakage & set(FEATURE_COLS)}"


def test_no_id_cols():
    id_cols = {"race_id", "horse_id", "jockey_id", "trainer_id",
               "horse_name", "venue", "fetched_at", "source"}
    assert id_cols.isdisjoint(set(FEATURE_COLS)), \
        f"ID columns found in FEATURE_COLS: {id_cols & set(FEATURE_COLS)}"


def test_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
