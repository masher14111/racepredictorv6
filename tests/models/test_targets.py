import pandas as pd

from models.targets import add_targets


def _df(positions):
    """Minimal df with position (Int64) and existing placed (top-3) column."""
    pos = pd.array(positions, dtype="Int64")
    placed = pd.array(
        [pd.NA if p is pd.NA or p is None else int(p <= 3) for p in positions],
        dtype="Int64",
    )
    return pd.DataFrame({"position": pos, "placed": placed})


def test_won_derivation():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=3)
    assert list(out["won"].fillna(-1)) == [1, 0, 0, 0, 0]


def test_placed_2_derivation():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=3)
    assert list(out["placed_2"].fillna(-1)) == [1, 1, 0, 0, 0]


def test_showed_derivation_default_3():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=3)
    assert list(out["showed"].fillna(-1)) == [1, 1, 1, 0, 0]


def test_showed_derivation_custom_positions():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=4)
    assert list(out["showed"].fillna(-1)) == [1, 1, 1, 1, 0]


def test_null_position_gives_null_labels():
    out = add_targets(_df([pd.NA, pd.NA]), show_positions=3)
    assert out["won"].isna().all()
    assert out["placed_2"].isna().all()
    assert out["showed"].isna().all()


def test_mixed_null_and_non_null():
    out = add_targets(_df([1, pd.NA, 4]), show_positions=3)
    assert list(out["won"].fillna(-1)) == [1, -1, 0]
    assert list(out["placed_2"].fillna(-1)) == [1, -1, 0]
    assert list(out["showed"].fillna(-1)) == [1, -1, 0]


def test_existing_placed_col_untouched():
    """add_targets must never overwrite the existing placed (top-3) column."""
    df = _df([1, 4])
    out = add_targets(df, show_positions=3)
    # original placed column: pos 1 → 1, pos 4 → 0
    assert list(out["placed"].fillna(-1)) == [1, 0]


def test_output_dtype_is_Int64():
    out = add_targets(_df([1, 2, None]), show_positions=3)
    assert str(out["won"].dtype) == "Int64"
    assert str(out["placed_2"].dtype) == "Int64"
    assert str(out["showed"].dtype) == "Int64"
