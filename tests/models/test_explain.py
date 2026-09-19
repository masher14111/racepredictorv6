import math

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from models.explain import Explainer, get_explainer, label_for


def _tiny_explainer(seed=42):
    """A 4-feature CatBoost model wrapped in an Explainer (no disk I/O)."""
    rng = np.random.default_rng(seed)
    X = rng.random((200, 4)).astype(float)
    # feature 0 drives the label strongly, feature 3 is pure noise
    y = (X[:, 0] + 0.3 * X[:, 1] > 0.7).astype(int)
    model = CatBoostClassifier(
        iterations=40, depth=3, verbose=False, allow_writing_files=False,
    )
    model.fit(X, y)
    cols = ["field_size", "historical_win_rate", "jockey_win_rate", "noise"]
    return Explainer(model, cols, target="won"), cols


def test_explain_structure():
    ex, cols = _tiny_explainer()
    out = ex.explain({c: 0.5 for c in cols})
    assert set(out) >= {"target", "base_value", "predicted_margin",
                        "predicted_prob", "top_positive", "top_negative"}
    assert out["target"] == "won"
    assert 0.0 <= out["predicted_prob"] <= 1.0


def test_predicted_prob_matches_sigmoid_of_margin():
    ex, cols = _tiny_explainer()
    out = ex.explain({c: 0.5 for c in cols})
    expected = 1.0 / (1.0 + math.exp(-out["predicted_margin"]))
    assert out["predicted_prob"] == pytest.approx(expected, abs=1e-3)


def test_contributions_sum_to_margin():
    """SHAP property: base + Σ contributions == predicted margin."""
    ex, cols = _tiny_explainer()
    row = {"field_size": 0.9, "historical_win_rate": 0.8,
           "jockey_win_rate": 0.2, "noise": 0.5}
    out = ex.explain(row)
    total = sum(d["contribution"] for d in out["top_positive"] + out["top_negative"])
    # top_k may truncate, so allow the truncated tail; with 4 features none drop.
    assert out["base_value"] + total == pytest.approx(out["predicted_margin"], abs=1e-2)


def test_drivers_sign_and_order():
    ex, cols = _tiny_explainer()
    out = ex.explain({"field_size": 0.95, "historical_win_rate": 0.9,
                      "jockey_win_rate": 0.5, "noise": 0.5})
    assert all(d["contribution"] > 0 for d in out["top_positive"])
    assert all(d["contribution"] < 0 for d in out["top_negative"])
    # positives sorted descending
    pos = [d["contribution"] for d in out["top_positive"]]
    assert pos == sorted(pos, reverse=True)


def test_top_k_limits_results():
    ex, cols = _tiny_explainer()
    out = ex.explain({c: 0.7 for c in cols}, top_k=1)
    assert len(out["top_positive"]) <= 1
    assert len(out["top_negative"]) <= 1


def test_explain_frame_matches_per_row():
    ex, cols = _tiny_explainer()
    df = pd.DataFrame([{c: 0.3 for c in cols}, {c: 0.8 for c in cols}])
    batch = ex.explain_frame(df)
    assert len(batch) == 2
    single = ex.explain(df.iloc[1])
    assert batch[1]["predicted_prob"] == pytest.approx(single["predicted_prob"], abs=1e-6)


def test_explain_frame_empty():
    ex, cols = _tiny_explainer()
    assert ex.explain_frame(pd.DataFrame()) == []


def test_missing_column_filled_with_nan():
    """A runner row missing a feature must not crash — column becomes NaN."""
    ex, cols = _tiny_explainer()
    out = ex.explain({"field_size": 0.5})  # other 3 cols absent
    assert 0.0 <= out["predicted_prob"] <= 1.0


def test_value_carried_through():
    ex, cols = _tiny_explainer()
    out = ex.explain({"field_size": 0.42, "historical_win_rate": 0.1,
                      "jockey_win_rate": 0.1, "noise": 0.1})
    drivers = {d["feature"]: d["value"] for d in out["top_positive"] + out["top_negative"]}
    assert drivers.get("field_size") == pytest.approx(0.42)


def test_label_for_known_and_fallback():
    assert label_for("field_size") == "Field size"
    assert label_for("some_new_feature") == "Some new feature"


def test_get_explainer_is_cached():
    pytest.importorskip("catboost")
    try:
        a = get_explainer(target="won", version_tag="v3nf")
    except FileNotFoundError:
        pytest.skip("v3nf model not on disk")
    b = get_explainer(target="won", version_tag="v3nf")
    assert a is b  # lru_cache returns the same instance


def test_get_explainer_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        get_explainer(target="won", version_tag="does_not_exist_tag")
