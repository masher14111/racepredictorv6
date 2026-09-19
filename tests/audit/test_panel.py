"""Stage-4 audit panel: the requirement-1 validation contract for the
historical evaluation frame."""
import numpy as np
import pandas as pd
import pytest

from audit.panel import build_audit_panel


def _frame(**over):
    """Two clean 3-runner WIN races + knobs to inject defects."""
    base = {
        "race_uid": ["r1"] * 3 + ["r2"] * 3,
        "race_date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3,
                                    utc=True),
        "horse_id": ["a", "b", "c", "d", "e", "f"],
        "market_type": ["WIN"] * 6,
        "won": [1, 0, 0, 0, 1, 0],
        "ppwap": [2.0, 4.0, 4.0, 3.0, 3.0, 3.0],
        "morningwap": [2.1, 4.1, 4.1, 3.1, 3.1, 3.1],
        "odds_finish": [1.9, 4.5, 4.2, 2.9, 3.2, 3.3],
        "field_size": [3, 3, 3, 3, 3, 3],
    }
    base.update(over)
    return pd.DataFrame(base)


class TestAuditPanel:
    def test_clean_frame_passes_whole(self):
        res = build_audit_panel(_frame())
        assert len(res.panel) == 6
        assert res.exclusions.empty
        assert res.panel["book_complete"].all()
        assert np.isfinite(res.panel["bet_price"]).all()

    def test_place_rows_excluded_with_reason(self):
        df = _frame(market_type=["WIN", "WIN", "WIN", "PLACE", "PLACE", "PLACE"])
        res = build_audit_panel(df)
        assert len(res.panel) == 3
        row = res.exclusions.set_index("reason").loc["non_win_market_row"]
        assert row["n_rows"] == 3

    def test_missing_outcome_cascades_to_whole_race_rejection(self):
        """Dropping the unlabeled runner breaks r1's booksum (0.75 < 0.8), so
        the WHOLE race fails closed — a race with an unlabeled runner cannot be
        scored on race-level log loss (the winner might be the missing row)."""
        df = _frame(won=[1, 0, None, 0, 1, 0])
        res = build_audit_panel(df)
        assert set(res.panel["race_uid"]) == {"r2"}
        reasons = set(res.exclusions["reason"])
        assert "missing_outcome" in reasons
        assert any(r.startswith("extreme_booksum") for r in reasons)

    def test_duplicate_runner_rows_deduped(self):
        df = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
        res = build_audit_panel(df)
        assert len(res.panel) == 6
        assert "duplicate_runner_row" in set(res.exclusions["reason"])

    def test_unpriced_row_in_small_field_fails_race_closed(self):
        """In a 3-runner race, losing one price guts the booksum, so the whole
        race is rejected — the incomplete book can never be de-vigged."""
        df = _frame(ppwap=[2.0, 4.0, np.nan, 3.0, 3.0, 3.0],
                    morningwap=[2.1, 4.1, np.nan, 3.1, 3.1, 3.1])
        res = build_audit_panel(df)
        assert set(res.panel["race_uid"]) == {"r2"}
        assert "no_valid_preoff_price" in set(res.exclusions["reason"])

    def test_missing_longshot_in_big_field_flags_book_incomplete(self):
        """A big field can lose one longshot and keep a plausible booksum —
        the race survives but book_complete goes False, so de-vig consumers
        can exclude it explicitly rather than silently."""
        n = 12
        prices = [8.0] * n            # booksum 1.5; minus one -> 1.375, in range
        df = pd.DataFrame({
            "race_uid": ["big"] * n,
            "race_date": pd.to_datetime(["2026-01-01"] * n, utc=True),
            "horse_id": [f"h{i}" for i in range(n)],
            "market_type": ["WIN"] * n,
            "won": [1] + [0] * (n - 1),
            "ppwap": prices[:-1] + [np.nan],
            "morningwap": prices[:-1] + [np.nan],
            "odds_finish": prices,
            "field_size": [n] * n,
        })
        res = build_audit_panel(df)
        assert len(res.panel) == n - 1
        assert not res.panel["book_complete"].any()

    def test_extreme_booksum_race_excluded(self):
        # Race 1's prices imply a 2.0 booksum — outside [0.8, 1.8].
        df = _frame(ppwap=[1.5, 1.5, 1.5, 3.0, 3.0, 3.0],
                    morningwap=[1.5, 1.5, 1.5, 3.1, 3.1, 3.1])
        res = build_audit_panel(df)
        assert set(res.panel["race_uid"]) == {"r2"}
        reasons = set(res.exclusions["reason"])
        assert any(r.startswith("extreme_booksum") for r in reasons)

    def test_implausible_field_size_excluded(self):
        df = _frame().iloc[[0, 3, 4, 5]]  # r1 collapses to a 1-runner "race"
        res = build_audit_panel(df)
        assert set(res.panel["race_uid"]) == {"r2"}
        reasons = set(res.exclusions["reason"])
        assert any(r.startswith("implausible_field_size") for r in reasons)

    def test_ppwap_preferred_then_morningwap(self):
        df = _frame(ppwap=[np.nan, 4.0, 4.0, 3.0, 3.0, 3.0])
        res = build_audit_panel(df)
        row = res.panel[res.panel["horse_id"] == "a"].iloc[0]
        assert row["bet_price"] == pytest.approx(2.1)   # morningwap fallback
