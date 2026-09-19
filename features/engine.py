"""Higher-order predictive features built on top of features/derive.py.

Called by features/builder.py _derive_all AFTER the derive.* steps, so these
functions can rely on implied_prob, overround_norm_prob, market_rank, field_size,
class_change, going_speed, race_class and recent_form_avg already existing.

Every trend feature is leak-safe: it only ever reads runs STRICTLY before each
row's race_date, bounded by lookback_months AND lookback_runs.
"""
import math

import numpy as np
import pandas as pd

from features import _feature_scale, _trailing_fast
from features.derive import _race_cols

_RACE_KEY = ["race_date", "venue"]


def finish_percentile(position, field_size):
    """Field-relative finish percentile of a completed run: 100 = won, lower = further
    back; NaN for non-finishers / fields of <2 runners. Vectorized over pandas Series.

    The denominator is the race's TRUE field size. Callers that derive over a pruned
    frame (e.g. build_inference_matrix, which keeps only live entities' prior rows)
    must compute this BEFORE pruning and pass it in — a pruned race has a wrong
    field_size, which would make the horse-speed proxy meaningless (or negative)."""
    pos = pd.to_numeric(position, errors="coerce")
    fs = pd.to_numeric(field_size, errors="coerce")
    valid = pos.notna() & fs.notna() & (fs > 1)
    return pd.Series(
        np.where(valid, (fs - pos) / (fs - 1.0) * 100.0, np.nan), index=pos.index)


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

    # Fast path (no predicate): the strictly-prior windowed rate vectorized via
    # _trailing_fast — O(n log n) vs the old per-row O(n^2) scan. The predicate
    # path keeps the original scan; its only caller is a tiny unit test (real
    # going-band filtering goes through add_going_preference's grouped fast path).
    if predicate is None:
        # dropna=True: a row missing any grouping id (e.g. an un-enriched
        # jockey/trainer) has no defined entity, so its rate is null. The old
        # tuple-key scan instead POOLED every NaN-id run into one fake mega-entity
        # and computed a meaningless cross-horse rate — a latent bug this fixes.
        win, plc, runs_arr = _trailing_fast.windowed_rates(
            out, list(key_cols), runs=cfg["lookback_runs"],
            months=cfg["lookback_months"], places=cfg["place_positions"],
            strict=True, dropna=True)
        out[win_col] = win
        out[place_col] = plc
        out[runs_col] = runs_arr
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
            prior = prior[prior["race_date"] < cutoff]
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


def add_combo_win_rate(df, cfg):
    """Leak-safe trailing win% for the (jockey_id, trainer_id) pairing."""
    out = _trailing_rate(df, ["jockey_id", "trainer_id"], "jt_combo_win_rate",
                         "_jt_combo_place", "jt_combo_runs", cfg)
    return out.drop(columns=["_jt_combo_place"])


def add_going_preference(df, cfg):
    """Horse's leak-safe trailing win/place rate on runs matching today's going band.

    The "same going band" predicate is realized by grouping on (horse_id, band):
    priors within a group share the horse AND the going, so the vectorized fast
    path applies directly. Prefers ``going_band`` (derived from the ~100%-populated
    raw going string) over the sparse config-mapped ``going_speed``; falls back to
    going_speed when no band column exists. Rows with an unknown band are dropped
    from grouping (dropna), leaving a null rate."""
    out = df.copy().reset_index(drop=True)
    band_col = ("going_band" if "going_band" in out.columns
                else "going_speed" if "going_speed" in out.columns else None)
    if band_col is None:
        out["going_pref_win_rate"] = pd.NA
        out["going_pref_place_rate"] = pd.NA
        return out
    win, plc, _ = _trailing_fast.windowed_rates(
        out, ["horse_id", band_col], runs=cfg["lookback_runs"],
        months=cfg["lookback_months"], places=cfg["place_positions"],
        strict=True, dropna=True)
    out["going_pref_win_rate"] = win
    out["going_pref_place_rate"] = plc
    return out


def add_course_suitability(df, cfg):
    """Horse's leak-safe trailing win/place rate at THIS venue (course suitability).

    Grouping on (horse_id, venue) means each prior in a group is the horse's own
    earlier run at the same track, so the strictly-prior vectorized rate applies
    directly. course_runs exposes the (often small) denominator to the model."""
    out = df.copy().reset_index(drop=True)
    if "venue" not in out.columns:
        out["course_win_rate"] = pd.NA
        out["course_place_rate"] = pd.NA
        out["course_runs"] = 0
        return out
    win, plc, runs = _trailing_fast.windowed_rates(
        out, ["horse_id", "venue"], runs=cfg["lookback_runs"],
        months=cfg["lookback_months"], places=cfg["place_positions"],
        strict=True, dropna=True)
    out["course_win_rate"] = win
    out["course_place_rate"] = plc
    out["course_runs"] = runs
    return out


# Distance bands (furlongs): sprint, mile, middle, staying. Course-and-distance
# aptitude is distance-specific, so a horse's form is pooled within its band.
def _dist_band(f) -> str | None:
    f = pd.to_numeric(f, errors="coerce")
    if pd.isna(f):
        return None
    if f < 7:
        return "sprint"
    if f < 9.5:
        return "mile"
    if f < 13:
        return "middle"
    return "staying"


def add_distance_suitability(df, cfg):
    """Horse's leak-safe trailing win/place rate at a similar trip (distance band).

    Grouping on (horse_id, distance band) pools the horse's prior runs over the
    same broad trip, so the strictly-prior vectorized rate applies. distance_runs
    exposes the denominator."""
    out = df.copy().reset_index(drop=True)
    if "distance_furlongs" not in out.columns:
        out["distance_win_rate"] = pd.NA
        out["distance_place_rate"] = pd.NA
        out["distance_runs"] = 0
        return out
    out["_dist_band"] = [_dist_band(f) for f in out["distance_furlongs"]]
    win, plc, runs = _trailing_fast.windowed_rates(
        out, ["horse_id", "_dist_band"], runs=cfg["lookback_runs"],
        months=cfg["lookback_months"], places=cfg["place_positions"],
        strict=True, dropna=True)
    out["distance_win_rate"] = win
    out["distance_place_rate"] = plc
    out["distance_runs"] = runs
    return out.drop(columns=["_dist_band"])


def add_speed_figures(df, cfg):
    """horse_speed = timeform_rating if present, else leak-safe trailing-mean proxy.
    horse_speed_rank = within-race descending rank of horse_speed."""
    out = df.copy().reset_index(drop=True)
    # _run_speed = each completed run's field-relative finish percentile (0..100),
    # NaN for non-finishers/non-runners. A caller may precompute it with each race's
    # true field size (build_inference_matrix does, before pruning history to live
    # entities — otherwise the pruned race's field_size is wrong); honor it if so.
    if "_run_speed" in out.columns:
        out["_run_speed"] = pd.to_numeric(out["_run_speed"], errors="coerce")
    else:
        out["_run_speed"] = finish_percentile(out.get("position"), out.get("field_size"))
    out["_speed_proxy"] = pd.NA

    # Leak-safe trailing mean of prior _run_speed, strictly before each race_date,
    # vectorized via _trailing_fast (was an O(n^2) per-row scan).
    if "race_date" in out.columns and "horse_id" in out.columns:
        out["_speed_proxy"] = _trailing_fast.windowed_mean(
            out, ["horse_id"], "_run_speed", runs=cfg["lookback_runs"],
            months=cfg["lookback_months"], strict=True, dropna=True)

    if "timeform_rating" in out.columns:
        tfr = out["timeform_rating"]
    else:
        tfr = pd.Series([pd.NA] * len(out), index=out.index)
    out["horse_speed"] = [
        t if (t is not None and not pd.isna(t)) else p
        for t, p in zip(tfr, out["_speed_proxy"])
    ]
    out["horse_speed"] = pd.to_numeric(out["horse_speed"], errors="coerce")
    out["horse_speed_rank"] = out.groupby(_race_cols(out), sort=False)["horse_speed"].rank(
        ascending=False, method="min")

    # speed_trend = recent short-window speed minus the longer baseline: positive =
    # improving, negative = regressing. Both legs are strictly-prior trailing means
    # of _run_speed (self excluded), so leak-free. The short leg caps at the most
    # recent few runs; the long leg uses the configured window.
    if "race_date" in out.columns and "horse_id" in out.columns:
        short = _trailing_fast.windowed_mean(
            out, ["horse_id"], "_run_speed", runs=3,
            months=cfg["lookback_months"], strict=True, dropna=True)
        out["speed_trend"] = short - pd.to_numeric(out["_speed_proxy"], errors="coerce")
    else:
        out["speed_trend"] = np.nan
    return out.drop(columns=["_run_speed", "_speed_proxy"])


def add_pace_bias(df):
    """Pass-through of pace_rating (paywalled-null until a Timeform session is set)."""
    out = df.copy()
    if "pace_rating" in out.columns:
        out["pace_bias"] = out["pace_rating"]
    else:
        out["pace_bias"] = pd.NA
    out["pace_bias"] = pd.to_numeric(out["pace_bias"], errors="coerce")
    return out


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
    out["ew_value_index"] = pd.to_numeric(out["ew_value_index"], errors="coerce")
    return out


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
    out["odds_drift"] = pd.to_numeric(out["odds_drift"], errors="coerce")
    ip = pd.to_numeric(out["implied_prob"], errors="coerce") \
        if "implied_prob" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    onp = pd.to_numeric(out["overround_norm_prob"], errors="coerce") \
        if "overround_norm_prob" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    out["odds_value_delta"] = ip - onp
    return out


def _zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series([float("nan")] * len(s), index=s.index)
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
    rkey = _race_cols(out)
    grp = out.groupby(rkey, sort=False)
    comps = []

    def _grp_std(col):
        # Coerce to numeric BEFORE the grouped std. A source that never supplies
        # this column (e.g. betsp has no race_class) leaves it object/NA-dtype,
        # which pandas' cython std cannot cast to float64 (TypeError on NAType).
        # to_numeric(errors="coerce") yields a clean float64 column of NaN, which
        # std handles gracefully (the component then drops out via skipna).
        s = pd.to_numeric(out[col], errors="coerce")
        return s.groupby([out[k] for k in rkey]).transform("std")

    comps.append(_zscore(grp["horse_id"].transform("size").astype(float)))
    if "overround_norm_prob" in out.columns:
        comps.append(_zscore(grp["overround_norm_prob"].transform(_entropy)))
    if "race_class" in out.columns:
        comps.append(_zscore(_grp_std("race_class")))
    if "recent_form_avg" in out.columns:
        comps.append(_zscore(_grp_std("recent_form_avg")))
    if "timeform_rating" in out.columns:
        comps.append(_zscore(_grp_std("timeform_rating")))

    stacked = pd.concat(comps, axis=1)
    out["race_complexity"] = stacked.mean(axis=1, skipna=True)
    return out


# --- race complexity v2: independent branch, train-fitted scale (step 04) ----
#
# `race_complexity` above is LEFT UNCHANGED: it is bound to the feature schema
# of every already-trained model bundle (models/train.py persists
# meta["feature_cols"]; PRICE_FREE_FEATURE_COLS already lists it as
# "independent"), and both of its defects — a market-entropy component sitting
# in a nominally independent feature, and a dataset-global z-score recomputed
# from whatever frame is passed in (so an admissible historical row's value
# silently shifted whenever later rows were appended to the same call) — are
# real, but redefining `race_complexity` in place would serve those existing
# bundles inputs with a different meaning than what they were trained on
# (DESIGN.md; DECISIONS D9). `race_complexity_v2` and `race_market_entropy`
# are the corrected, ADDITIVE replacements: available to any NEW candidate
# (step 10+) without touching what is already serving.

_RACE_COMPLEXITY_V2_COMPONENTS = (
    "field_size", "race_class_std", "recent_form_std", "rating_std")


def _race_complexity_v2_raw_components(out: pd.DataFrame) -> dict:
    """Race-local raw (pre-scale) value per independent race_complexity_v2
    component, one value per race broadcast to every runner row. No market
    input at all (contrast with the legacy race_complexity's market-entropy
    component) and no dataset-global statistic computed here — only genuinely
    race-local quantities. Cross-race comparability is added separately by a
    FROZEN scale (see fit_race_complexity_v2_scale / add_race_complexity_v2).

    Reuses the existing ``field_size`` column (derive.add_odds_features,
    computed earlier in the SAME features/builder.py::_derive_all pass, before
    any row is ever dropped) instead of recounting rows-per-race here. A
    non-runner/unresolved result has no admissible `position` and is dropped
    by `build_training_matrix`'s later `position.notna()` label filter;
    recounting group size from whatever rows happen to still be present would
    silently UNDERCOUNT that race's true field for every survivor (found this
    stage, verified in test_race_complexity_v2_field_size_reflects_full_field_
    not_a_filtered_view). Falls back to a fresh count only when no field_size
    column exists yet (e.g. a minimal synthetic caller)."""
    rkey = _race_cols(out)
    grp = out.groupby(rkey, sort=False)

    def _grp_std(col):
        s = pd.to_numeric(out[col], errors="coerce")
        return s.groupby([out[k] for k in rkey]).transform("std")

    field_size = (
        pd.to_numeric(out["field_size"], errors="coerce") if "field_size" in out.columns
        else grp["horse_id"].transform("size").astype(float))
    comps = {"field_size": field_size}
    comps["race_class_std"] = (
        _grp_std("race_class") if "race_class" in out.columns
        else pd.Series(np.nan, index=out.index))
    comps["recent_form_std"] = (
        _grp_std("recent_form_avg") if "recent_form_avg" in out.columns
        else pd.Series(np.nan, index=out.index))
    comps["rating_std"] = (
        _grp_std("timeform_rating") if "timeform_rating" in out.columns
        else pd.Series(np.nan, index=out.index))
    return comps


def fit_race_complexity_v2_scale(df: pd.DataFrame) -> dict:
    """Fit the frozen cross-race standardization constants for
    race_complexity_v2 from a (training) frame. Call ONCE on admissible
    training data and persist via features._feature_scale.save_scale; do not
    call this at serve time or inside add_race_complexity_v2 itself — the
    whole point is that the scale does not move when new rows appear.

    Fit from the FULL derived matrix (features/builder.py's ``full`` —
    every row `_derive_all` produced, BEFORE the ``position.notna()`` label
    filter that `build_training_matrix` applies to get the labelled-only
    training file), not that labelled-only file. `field_size` is computed
    from whatever rows share a race's group at the time this is called; a
    non-runner or a result that never joined has no `position` and is
    dropped by the label filter, so fitting from the labelled-only file
    silently undercounts every affected race's true field size. Both the
    live inference matrix and `_derive_all` compute `field_size` (and this
    function's components) BEFORE any such filter, so fitting from the same
    (full, unfiltered) population keeps the frozen scale consistent with
    every future call, not just numerically self-consistent within a
    biased sample."""
    out = df.copy()
    rkey = _race_cols(out)
    raw = _race_complexity_v2_raw_components(out)
    # One row per race so a big field does not over-weight the fit (field_size
    # itself is constant per race; the spread components are already per-race).
    one_per_race_idx = out.drop_duplicates(subset=rkey).index
    one_per_race_raw = {k: v.loc[one_per_race_idx] for k, v in raw.items()}
    return _feature_scale.fit_scale(one_per_race_raw)


def add_race_complexity_v2(df: pd.DataFrame, scale: dict | None = None) -> pd.DataFrame:
    """Independent race-difficulty scalar: field size + class/form/rating
    spread ONLY — no market entropy (see add_race_market_entropy for the
    explicit market-assisted companion) and no live-recomputed dataset-global
    z-score (see module comment above add_race_complexity_v2's block).

    ``scale`` defaults to the frozen artifact loaded via
    features._feature_scale.load_scale(); pass one explicitly (e.g. from
    fit_race_complexity_v2_scale) to evaluate a refit candidate without
    touching the on-disk default. A component missing from ``scale``, or whose
    input is entirely null in ``df``, drops out of the blend (skipna mean) —
    with no fitted artifact at all, the column is present but entirely null
    rather than raising, matching this codebase's degrade-gracefully
    convention for an unavailable source."""
    out = df.copy()
    if scale is None:
        scale = _feature_scale.load_scale() or {}
    raw = _race_complexity_v2_raw_components(out)
    comps = []
    for name, series in raw.items():
        if name not in scale or series.isna().all():
            continue
        comps.append(_feature_scale.apply_scale(series, scale[name]))
    if comps:
        stacked = pd.concat(comps, axis=1)
        out["race_complexity_v2"] = stacked.mean(axis=1, skipna=True)
    else:
        out["race_complexity_v2"] = pd.Series(np.nan, index=out.index)
    return out


def add_race_market_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit MARKET-ASSISTED companion to race_complexity_v2: Shannon
    entropy of the race's overround-normalized implied probabilities (a
    tightly-priced, competitive book has high entropy; a short-priced
    favourite has low entropy). Depends on odds and MUST be registered in
    PRICE_FEATURE_COLS, never PRICE_FREE_FEATURE_COLS — this is the market
    signal the legacy race_complexity silently mixed into an "independent"
    feature (CONTRACTS.md)."""
    out = df.copy()
    if "overround_norm_prob" not in out.columns:
        out["race_market_entropy"] = pd.Series(np.nan, index=out.index)
        return out
    rkey = _race_cols(out)
    grp = out.groupby(rkey, sort=False)
    out["race_market_entropy"] = grp["overround_norm_prob"].transform(_entropy)
    return out
