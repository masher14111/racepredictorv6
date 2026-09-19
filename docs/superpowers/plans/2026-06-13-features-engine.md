# Features Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `features/engine.py` module of higher-order predictive features (speed figures, jockey/trainer combo win%, each-way value index, going preference, pace bias, odds delta vs. market, race complexity) on top of the existing leak-safe `features/` pipeline, and emit the full derived matrix to `data/features.parquet`.

**Architecture:** New pure-function module `features/engine.py` is called by `builder.py`'s `_derive_all` AFTER the existing `derive.*` steps, so engine functions build on already-derived columns (`implied_prob`, `overround_norm_prob`, `market_rank`, `field_size`, `class_change`, `going_speed`, `race_class`, `recent_form_avg`). All trend features reuse one shared leak-safe helper `_trailing_rate` (strict `<` cutoff, bounded by `lookback_months` AND `lookback_runs`). Train/serve parity is preserved because engine runs inside the shared pipeline.

**Tech Stack:** Python 3.14, pandas, pyarrow, pytest. Config via existing `timeform.{lookback_months,lookback_runs,place_positions,going_speed_map}` in `config.yaml`.

> **Git note:** This project has no git repo yet (`git init` not run). The `Commit` steps below assume a repo exists. If you have not run `git init`, either do so once before starting or skip the commit steps — every other step stands on its own.

> **Unified schema facts (verified against `utils/normalizer.py CANONICAL_COLUMNS`):** columns available include `morningwap`, `sp`, `ew_places`, `ew_reduction`, `pace_rating`, `timeform_rating`, `race_class`, `going`, `position`. There is **no `bsp` column** — Betfair SP lives in `sp`, so odds drift uses `(morningwap - sp)/morningwap`. `implied_prob`, `overround_norm_prob`, `market_rank`, `field_size`, `going_speed`, `recent_form_avg` are derived earlier in `_derive_all`.

---

## File Structure

- Create: `features/engine.py` — all new feature functions + shared `_trailing_rate` helper.
- Create: `tests/features/test_engine.py` — per-function unit tests, leak-safety + null-graceful assertions.
- Modify: `features/builder.py` — call engine steps in `_derive_all`; write full matrix to `data/features.parquet`.
- Modify: `tests/features/test_builder.py` — assert new columns present + `data/features.parquet` written with full (labelled+live) row set.

---

## Task 1: Shared leak-safe trailing-rate helper

**Files:**

- Create: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_engine.py
import math

import pandas as pd

from features import engine

CFG = {"lookback_months": 12, "lookback_runs": 20, "place_positions": 3}


def test_trailing_rate_is_leak_safe_and_groups_by_key():
    df = pd.DataFrame({
        "k": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = engine._trailing_rate(
        df, ["k"], "win_rate", "place_rate", "runs", CFG).sort_values("race_date")
    # first row has no prior runs -> null
    assert pd.isna(out.iloc[0]["win_rate"])
    # third row: two prior runs (one win) -> 0.5, never sees its own result
    assert out.iloc[2]["win_rate"] == 0.5
    assert out.iloc[2]["runs"] == 2


def test_trailing_rate_predicate_filters_prior_runs():
    df = pd.DataFrame({
        "k": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 5, 1],
        "going_speed": [3, 2, 3],
    })
    # predicate: only prior runs whose going_speed equals the current row's
    out = engine._trailing_rate(
        df, ["k"], "win_rate", "place_rate", "runs", CFG,
        predicate=lambda prior, cur: prior["going_speed"] == cur["going_speed"]
    ).sort_values("race_date")
    # 3rd row going_speed=3; only the 2026-01-01 run (going_speed=3, pos 1) qualifies
    assert out.iloc[2]["win_rate"] == 1.0
    assert out.iloc[2]["runs"] == 1


def test_trailing_rate_null_when_no_position_column():
    df = pd.DataFrame({"k": ["A"], "race_date": pd.to_datetime(["2026-01-01"], utc=True)})
    out = engine._trailing_rate(df, ["k"], "win_rate", "place_rate", "runs", CFG)
    assert pd.isna(out.iloc[0]["win_rate"])
    assert out.iloc[0]["runs"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# features/engine.py
"""Higher-order predictive features built on top of features/derive.py.

Called by features/builder.py _derive_all AFTER the derive.* steps, so these
functions can rely on implied_prob, overround_norm_prob, market_rank, field_size,
class_change, going_speed, race_class and recent_form_avg already existing.

Every trend feature is leak-safe: it only ever reads runs STRICTLY before each
row's race_date, bounded by lookback_months AND lookback_runs.
"""
import math

import pandas as pd

_RACE_KEY = ["race_date", "venue"]


def _trailing_rate(df, key_cols, win_col, place_col, runs_col, cfg, predicate=None):
    """Leak-safe trailing win/place rate over strictly-prior runs of an entity.

    key_cols : grouping entity, e.g. ['horse_id'] or ['jockey_id', 'trainer_id'].
    predicate: optional fn(prior_df, current_row) -> boolean mask choosing which
               prior runs count (e.g. same going band). Applied before the
               lookback_runs tail cut.
    Only prior runs with a known finishing position contribute to the rate.
    """
    out = df.copy().reset_index(drop=True)
    out[win_col] = pd.NA
    out[place_col] = pd.NA
    out[runs_col] = 0
    if "position" not in out.columns or "race_date" not in out.columns:
        return out
    months = cfg["lookback_months"]
    runs = cfg["lookback_runs"]
    places = cfg["place_positions"]
    window = pd.DateOffset(months=months)

    if len(key_cols) == 1:
        key = out[key_cols[0]]
    else:
        key = out[key_cols].astype(object).agg(tuple, axis=1)

    for _, grp in out.groupby(key, sort=False):
        g = grp.sort_values("race_date")
        idx = list(g.index)
        for pos_i, i in enumerate(idx):
            cutoff = out.at[i, "race_date"]
            prior = g.loc[idx[:pos_i]]
            prior = prior[prior["race_date"] >= (cutoff - window)]
            if predicate is not None:
                prior = prior[predicate(prior, out.loc[i])]
            if runs:
                prior = prior.tail(runs)
            prior = prior[prior["position"].notna()]
            n = len(prior)
            out.at[i, runs_col] = n
            if n:
                wins = (prior["position"] == 1).sum()
                pl = (prior["position"] <= places).sum()
                out.at[i, win_col] = wins / n
                out.at[i, place_col] = pl / n
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add leak-safe _trailing_rate helper in engine.py"
```

---

## Task 2: Jockey/trainer combo win rate

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_combo_win_rate_keys_on_jockey_trainer_pair():
    df = pd.DataFrame({
        "jockey_id": ["J", "J", "J"],
        "trainer_id": ["T", "T", "X"],   # third row is a DIFFERENT pairing
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = engine.add_combo_win_rate(df, CFG).sort_values(["trainer_id", "race_date"])
    jt = out[(out["jockey_id"] == "J") & (out["trainer_id"] == "T")].sort_values("race_date")
    # second J/T run: one prior win for the pair -> 1.0 over 1 run
    assert jt.iloc[1]["jt_combo_win_rate"] == 1.0
    assert jt.iloc[1]["jt_combo_runs"] == 1
    # the J/X pairing has no prior runs -> null
    jx = out[(out["jockey_id"] == "J") & (out["trainer_id"] == "X")].iloc[0]
    assert pd.isna(jx["jt_combo_win_rate"])
    assert "_jt_combo_place" not in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py::test_combo_win_rate_keys_on_jockey_trainer_pair -v`
Expected: FAIL with `AttributeError: module 'features.engine' has no attribute 'add_combo_win_rate'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def add_combo_win_rate(df, cfg):
    """Leak-safe trailing win% for the (jockey_id, trainer_id) pairing."""
    out = _trailing_rate(df, ["jockey_id", "trainer_id"], "jt_combo_win_rate",
                         "_jt_combo_place", "jt_combo_runs", cfg)
    return out.drop(columns=["_jt_combo_place"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py::test_combo_win_rate_keys_on_jockey_trainer_pair -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add jockey/trainer combo win rate"
```

---

## Task 3: Going preference

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_going_preference_uses_only_matching_going_band():
    df = pd.DataFrame({
        "horse_id": ["H", "H", "H"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "going_speed": [3, 2, 3],   # today's run (3rd) is fast going, like run 1
        "position": [1, 4, 1],
    })
    out = engine.add_going_preference(df, CFG).sort_values("race_date")
    # 3rd row: only the going_speed=3 prior run (pos 1) counts -> win rate 1.0
    assert out.iloc[2]["going_pref_win_rate"] == 1.0
    assert out.iloc[2]["going_pref_place_rate"] == 1.0
    assert "_going_pref_runs" not in out.columns


def test_going_preference_null_when_today_going_unknown():
    df = pd.DataFrame({
        "horse_id": ["H", "H"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True),
        "going_speed": [3, None],
        "position": [1, None],
    })
    out = engine.add_going_preference(df, CFG).sort_values("race_date")
    assert pd.isna(out.iloc[1]["going_pref_win_rate"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py::test_going_preference_uses_only_matching_going_band -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_going_preference'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def add_going_preference(df, cfg):
    """Horse's leak-safe trailing win/place rate on runs matching today's going band."""
    out = _trailing_rate(
        df, ["horse_id"], "going_pref_win_rate", "going_pref_place_rate",
        "_going_pref_runs", cfg,
        predicate=lambda prior, cur: prior["going_speed"] == cur["going_speed"])
    return out.drop(columns=["_going_pref_runs"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -k going_preference -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add horse going preference"
```

---

## Task 4: Speed figures (TFR else derived proxy)

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_speed_figures_prefer_tfr_when_present():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13", "2026-06-13"], utc=True),
        "venue": ["X", "X"], "horse_id": ["a", "b"],
        "timeform_rating": [120, 100], "position": [None, None], "field_size": [2, 2],
    })
    out = engine.add_speed_figures(df, CFG).set_index("horse_id")
    assert out.loc["a", "horse_speed"] == 120
    assert out.loc["a", "horse_speed_rank"] == 1  # higher figure -> rank 1


def test_speed_figures_fall_back_to_leak_safe_proxy():
    # No TFR. Horse H finished 1st of 4 then races today; proxy = trailing mean of
    # the prior run's field-relative finish percentile = (4-1)/(4-1)*100 = 100.
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-01", "2026-06-13"], utc=True),
        "venue": ["X", "Y"], "horse_id": ["H", "H"],
        "timeform_rating": [None, None], "position": [1, None], "field_size": [4, 6],
    })
    out = engine.add_speed_figures(df, CFG).sort_values("race_date")
    # first run has no prior -> null; today's run sees the prior 1st-of-4 -> 100
    assert pd.isna(out.iloc[0]["horse_speed"])
    assert out.iloc[1]["horse_speed"] == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py -k speed_figures -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_speed_figures'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def _run_speed_proxy(row):
    """Field-relative finishing percentile for a single completed run, 0..100.

    Self-contained (uses only that run's own position + field_size), so taking a
    trailing mean over prior runs is inherently leak-safe."""
    pos = row.get("position")
    fs = row.get("field_size")
    if pos is None or pd.isna(pos) or fs is None or pd.isna(fs) or fs <= 1:
        return pd.NA
    return (float(fs) - float(pos)) / (float(fs) - 1.0) * 100.0


def add_speed_figures(df, cfg):
    """horse_speed = timeform_rating if present, else leak-safe trailing-mean proxy.
    horse_speed_rank = within-race descending rank of horse_speed."""
    out = df.copy().reset_index(drop=True)
    out["_run_speed"] = out.apply(_run_speed_proxy, axis=1)
    out["_speed_proxy"] = pd.NA

    runs = cfg["lookback_runs"]
    window = pd.DateOffset(months=cfg["lookback_months"])
    if "race_date" in out.columns and "horse_id" in out.columns:
        for _, grp in out.groupby("horse_id", sort=False):
            g = grp.sort_values("race_date")
            idx = list(g.index)
            for pos_i, i in enumerate(idx):
                cutoff = out.at[i, "race_date"]
                prior = g.loc[idx[:pos_i]]
                prior = prior[prior["race_date"] >= (cutoff - window)]
                if runs:
                    prior = prior.tail(runs)
                vals = prior["_run_speed"].dropna()
                if len(vals):
                    out.at[i, "_speed_proxy"] = float(vals.mean())

    if "timeform_rating" in out.columns:
        tfr = out["timeform_rating"]
    else:
        tfr = pd.Series([pd.NA] * len(out), index=out.index)
    out["horse_speed"] = [
        t if (t is not None and not pd.isna(t)) else p
        for t, p in zip(tfr, out["_speed_proxy"])
    ]
    out["horse_speed"] = pd.to_numeric(out["horse_speed"], errors="coerce")
    out["horse_speed_rank"] = out.groupby(_RACE_KEY, sort=False)["horse_speed"].rank(
        ascending=False, method="min")
    return out.drop(columns=["_run_speed", "_speed_proxy"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -k speed_figures -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add speed figures (TFR with leak-safe proxy fallback)"
```

---

## Task 5: Pace bias (pass-through)

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pace_bias_passes_through_pace_rating():
    df = pd.DataFrame({"pace_rating": [55, None]})
    out = engine.add_pace_bias(df)
    assert out.iloc[0]["pace_bias"] == 55
    assert pd.isna(out.iloc[1]["pace_bias"])


def test_pace_bias_null_when_column_absent():
    df = pd.DataFrame({"horse_id": ["a"]})
    out = engine.add_pace_bias(df)
    assert pd.isna(out.iloc[0]["pace_bias"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py -k pace_bias -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_pace_bias'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def add_pace_bias(df):
    """Pass-through of pace_rating (paywalled-null until a Timeform session is set)."""
    out = df.copy()
    if "pace_rating" in out.columns:
        out["pace_bias"] = out["pace_rating"]
    else:
        out["pace_bias"] = pd.NA
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -k pace_bias -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add pace bias pass-through"
```

---

## Task 6: Each-way value index

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ew_value_index_matches_hand_computation():
    # odds=5.0, fair win prob=0.25, ew_places=3, reduction=0.25
    # place prob proxy = min(1, 0.25*3) = 0.75
    # place_odds = 1 + (5-1)*0.25 = 2.0
    # win_ev = 0.25*5 = 1.25 ; place_ev = 0.75*2.0 = 1.5 ; index = 1.25+1.5-2 = 0.75
    df = pd.DataFrame({
        "odds_decimal": [5.0], "sp": [None],
        "ew_places": [3], "ew_reduction": [0.25],
        "overround_norm_prob": [0.25],
    })
    out = engine.add_ew_value_index(df)
    assert abs(out.iloc[0]["ew_value_index"] - 0.75) < 1e-9


def test_ew_value_index_null_without_terms_or_prob():
    df = pd.DataFrame({
        "odds_decimal": [5.0], "sp": [None],
        "ew_places": [None], "ew_reduction": [None], "overround_norm_prob": [0.25],
    })
    out = engine.add_ew_value_index(df)
    assert pd.isna(out.iloc[0]["ew_value_index"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py -k ew_value -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_ew_value_index'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def _ew_best_odds(row):
    for col in ("odds_decimal", "sp"):
        v = row.get(col)
        if v is not None and not pd.isna(v) and float(v) > 1.0:
            return float(v)
    return None


def add_ew_value_index(df):
    """Expected each-way return per 2-unit stake (1 win + 1 place) at market odds.

    Positive => positive-expectation each-way bet. Place probability is a rough
    market-implied proxy: min(1, fair_win_prob * ew_places)."""
    out = df.copy()

    def _idx(row):
        odds = _ew_best_odds(row)
        places = row.get("ew_places")
        reduction = row.get("ew_reduction")
        pwin = row.get("overround_norm_prob")
        if (odds is None or places is None or pd.isna(places)
                or reduction is None or pd.isna(reduction)
                or pwin is None or pd.isna(pwin)):
            return pd.NA
        pwin = float(pwin)
        pplace = min(1.0, pwin * float(places))
        place_odds = 1.0 + (odds - 1.0) * float(reduction)
        return pwin * odds + pplace * place_odds - 2.0

    out["ew_value_index"] = out.apply(_idx, axis=1)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -k ew_value -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add each-way value index"
```

---

## Task 7: Odds delta vs. market

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_odds_delta_drift_and_value():
    df = pd.DataFrame({
        "morningwap": [6.0, None],
        "sp": [4.0, 4.0],
        "implied_prob": [0.25, 0.10],
        "overround_norm_prob": [0.20, 0.10],
    })
    out = engine.add_odds_delta(df)
    # drift = (6-4)/6 = 0.3333...
    assert abs(out.iloc[0]["odds_drift"] - (2.0 / 6.0)) < 1e-9
    # second row has no morningwap -> drift null
    assert pd.isna(out.iloc[1]["odds_drift"])
    # value delta = implied_prob - overround_norm_prob
    assert abs(out.iloc[0]["odds_value_delta"] - 0.05) < 1e-9
    assert abs(out.iloc[1]["odds_value_delta"] - 0.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py -k odds_delta -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_odds_delta'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def add_odds_delta(df):
    """odds_drift = morning-to-SP move (betSP history); odds_value_delta = within-race
    relative mispricing (any snapshot)."""
    out = df.copy()

    def _drift(row):
        mw = row.get("morningwap")
        sp = row.get("sp")
        if mw is None or pd.isna(mw) or sp is None or pd.isna(sp) or float(mw) == 0:
            return pd.NA
        return (float(mw) - float(sp)) / float(mw)

    out["odds_drift"] = out.apply(_drift, axis=1)
    ip = pd.to_numeric(out["implied_prob"], errors="coerce") \
        if "implied_prob" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    onp = pd.to_numeric(out["overround_norm_prob"], errors="coerce") \
        if "overround_norm_prob" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    out["odds_value_delta"] = ip - onp
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -k odds_delta -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add odds drift and value delta"
```

---

## Task 8: Race complexity score

**Files:**

- Modify: `features/engine.py`
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_race_complexity_broadcasts_per_race_and_skips_null_components():
    # Two races. timeform_rating all-null (component skipped gracefully).
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 5, utc=True),
        "venue": ["A", "A", "A", "B", "B"],
        "horse_id": ["a1", "a2", "a3", "b1", "b2"],
        "overround_norm_prob": [0.34, 0.33, 0.33, 0.80, 0.20],
        "race_class": [3, 5, 4, 3, 3],
        "recent_form_avg": [2.0, 5.0, 3.0, 1.0, 1.0],
        "timeform_rating": [None, None, None, None, None],
    })
    out = engine.add_race_complexity(df)
    # every runner in race A shares one complexity value
    a_vals = out[out["venue"] == "A"]["race_complexity"].unique()
    assert len(a_vals) == 1
    # race A (3 runners, even market, wide class spread) is more complex than race B
    a = out[out["venue"] == "A"]["race_complexity"].iloc[0]
    b = out[out["venue"] == "B"]["race_complexity"].iloc[0]
    assert a > b


def test_race_complexity_single_race_does_not_crash():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 2, utc=True),
        "venue": ["A", "A"], "horse_id": ["a1", "a2"],
        "overround_norm_prob": [0.6, 0.4], "race_class": [3, 4],
        "recent_form_avg": [2.0, 3.0], "timeform_rating": [None, None],
    })
    out = engine.add_race_complexity(df)
    assert "race_complexity" in out.columns
    assert out["race_complexity"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_engine.py -k race_complexity -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_race_complexity'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to features/engine.py
def _zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / sd


def _entropy(probs):
    p = pd.to_numeric(probs, errors="coerce").dropna()
    p = p[p > 0]
    if p.empty:
        return float("nan")
    return float(-(p * p.map(math.log)).sum())


def add_race_complexity(df):
    """Per-race difficulty/competitiveness scalar broadcast to every runner.

    z-blend (mean over the non-null components) of: field size, market entropy,
    class spread, recent-form spread, ratings spread. A component whose input is
    all-null in the data is skipped gracefully."""
    out = df.copy()
    grp = out.groupby(_RACE_KEY, sort=False)
    comps = []

    comps.append(_zscore(grp["horse_id"].transform("size").astype(float)))
    if "overround_norm_prob" in out.columns:
        comps.append(_zscore(grp["overround_norm_prob"].transform(_entropy)))
    if "race_class" in out.columns:
        comps.append(_zscore(grp["race_class"].transform("std")))
    if "recent_form_avg" in out.columns:
        comps.append(_zscore(grp["recent_form_avg"].transform("std")))
    if "timeform_rating" in out.columns:
        comps.append(_zscore(grp["timeform_rating"].transform("std")))

    stacked = pd.concat(comps, axis=1)
    out["race_complexity"] = stacked.mean(axis=1, skipna=True)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_engine.py -k race_complexity -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the whole engine test module**

Run: `pytest tests/features/test_engine.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add features/engine.py tests/features/test_engine.py
git commit -m "feat(features): add race complexity score"
```

---

## Task 9: Wire engine into builder and emit data/features.parquet

**Files:**

- Modify: `features/builder.py`
- Test: `tests/features/test_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/features/test_builder.py
def test_builder_emits_full_feature_matrix_and_new_columns(tmp_path):
    out_path = tmp_path / "training.parquet"
    features_path = tmp_path / "features.parquet"
    build_training_matrix(
        unified=_unified(), write=True,
        output_path=str(out_path), features_path=str(features_path))
    full = pd.read_parquet(features_path)
    # full matrix keeps ALL rows (3 history + 2 live), unlike training (history only)
    assert len(full) == 5
    # engine columns are present on the full matrix
    for col in ["horse_speed", "jt_combo_win_rate", "going_pref_win_rate",
                "pace_bias", "ew_value_index", "odds_drift", "odds_value_delta",
                "race_complexity"]:
        assert col in full.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_builder.py::test_builder_emits_full_feature_matrix_and_new_columns -v`
Expected: FAIL — `build_training_matrix() got an unexpected keyword argument 'features_path'`

- [ ] **Step 3: Implement — add engine import + calls and the new output**

In `features/builder.py`, add the import near the other feature imports:

```python
from features import derive, engine
```

(Replace the existing `from features import derive` line.)

Add the default features path constant next to the others:

```python
_DEFAULT_FEATURES = os.path.join(_BASE, "data", "features.parquet")
```

Append the engine steps at the end of `_derive_all`, just before `return`:

```python
def _derive_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Shared feature pipeline: fuse then derive every feature. Train/serve identical."""
    out = fuse_sources(df)
    out["race_date"] = pd.to_datetime(out["race_date"], utc=True, errors="coerce")
    out = derive.add_going_speed(out, cfg["going_speed_map"])
    out = derive.add_class_change(out)
    out = derive.add_distance_furlongs(out)
    out = derive.add_recent_form(out)
    out = derive.add_odds_features(out)
    out = derive.add_rating_rank(out)
    out = derive.add_all_trailing_rates(out, cfg)
    # higher-order engine features (build on the derived columns above)
    out = engine.add_speed_figures(out, cfg)
    out = engine.add_combo_win_rate(out, cfg)
    out = engine.add_going_preference(out, cfg)
    out = engine.add_pace_bias(out)
    out = engine.add_ew_value_index(out)
    out = engine.add_odds_delta(out)
    out = engine.add_race_complexity(out)
    return out.reset_index(drop=True)
```

Replace `build_training_matrix` so it writes the full matrix to `data/features.parquet` before filtering:

```python
def build_training_matrix(unified=None, write=True, output_path=None,
                          features_path=None) -> pd.DataFrame:
    """Labelled historical matrix: fuse -> derive -> label -> drop null-position rows.

    Also writes the FULL derived matrix (labelled history + live) to
    data/features.parquet when write=True."""
    cfg = _load_cfg()
    full = _derive_all(_load_unified(unified), cfg)
    full = add_labels(full, cfg["place_positions"])
    if write:
        fpath = features_path or _DEFAULT_FEATURES
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        full.to_parquet(fpath, engine="pyarrow", index=False)
        logger.info("features: wrote %d full-matrix rows to %s", len(full), fpath)
    df = full[full["position"].notna()].reset_index(drop=True)
    if write:
        path = output_path or _DEFAULT_TRAINING
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)
        logger.info("features: wrote %d training rows to %s", len(df), path)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_builder.py::test_builder_emits_full_feature_matrix_and_new_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/builder.py tests/features/test_builder.py
git commit -m "feat(features): wire engine into _derive_all and emit data/features.parquet"
```

---

## Task 10: Parity guard + full suite

**Files:**

- Modify: `tests/features/test_builder.py`

- [ ] **Step 1: Extend the parity test to assert the new columns are shared**

Replace `test_train_and_inference_share_feature_columns` in `tests/features/test_builder.py` with:

```python
def test_train_and_inference_share_feature_columns():
    train = build_training_matrix(unified=_unified(), write=False)
    infer = build_inference_matrix(unified=_unified())
    feature_cols = lambda d: set(d.columns) - {"won", "placed"}
    assert feature_cols(train) == feature_cols(infer)
    # the new engine columns exist on BOTH paths (train/serve parity)
    for col in ["horse_speed", "horse_speed_rank", "jt_combo_win_rate",
                "jt_combo_runs", "going_pref_win_rate", "going_pref_place_rate",
                "pace_bias", "ew_value_index", "odds_drift", "odds_value_delta",
                "race_complexity"]:
        assert col in feature_cols(train)
        assert col in feature_cols(infer)
```

- [ ] **Step 2: Run the parity test**

Run: `pytest tests/features/test_builder.py::test_train_and_inference_share_feature_columns -v`
Expected: PASS

- [ ] **Step 3: Run the full test suite (regression guard)**

Run: `pytest -q`
Expected: PASS — previous 191 tests still green plus the new engine tests (≈191 + 14).

- [ ] **Step 4: Commit**

```bash
git add tests/features/test_builder.py
git commit -m "test(features): guard engine columns in train/serve parity"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** speed figures (T4), combo win% (T2), each-way value index (T6), going preference (T3), pace bias (T5), odds delta drift+value (T7), race complexity (T8), full-matrix output to `data/features.parquet` (T9), parity guard (T10). Class differential + single-entity rolling trends are pre-existing in `derive.py` and reused — no task needed.
- **Data reality:** features needing `position`/`timeform_rating`/`pace_rating` compute to null on current on-disk data and activate when that data lands. Tests inject the inputs directly so coverage does not depend on the blocked sources.
- **No `bsp` column:** odds drift uses `morningwap` vs `sp` per the verified canonical schema.
- **Type consistency:** column names are identical across tasks and the parity guard (`jt_combo_win_rate`, `going_pref_win_rate`, `ew_value_index`, `odds_drift`, `odds_value_delta`, `race_complexity`, `horse_speed`, `horse_speed_rank`).
