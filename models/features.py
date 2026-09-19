"""Input feature whitelist for CatBoost models.

Column names match the output of features/derive.py and features/engine.py exactly.
CatBoost handles NaN natively — columns present but all-null are fine.
"""

FEATURE_COLS = [
    # odds-market features
    "implied_prob",
    "overround_norm_prob",
    "log_odds",
    "market_rank",
    "field_size",
    # race context
    "going_speed",
    "class_change",
    "distance_furlongs",
    # recent form (from timeform free finishing figures)
    "recent_form_avg",
    "recent_form_wins",
    "recent_form_runs",
    # ratings
    "timeform_rating",
    "rating_rank",
    "pace_bias",
    # trailing rates — horse (historical_* from derive.add_all_trailing_rates)
    "historical_win_rate",
    "historical_place_rate",
    # trailing rates — jockey / trainer (place rates are dropped internally)
    "jockey_win_rate",
    "trainer_win_rate",
    # jockey-trainer combo
    "jt_combo_win_rate",
    "jt_combo_runs",
    # going preference (keyed on the ~100%-filled going_band, see features.derive)
    "going_pref_win_rate",
    "going_pref_place_rate",
    # freshness / experience (model-10)
    "days_since_last_run",
    "horse_career_runs",
    # course suitability — horse's trailing form at THIS venue (model-10)
    "course_win_rate",
    "course_place_rate",
    "course_runs",
    # distance suitability — trailing form at a similar trip band (model-10)
    "distance_win_rate",
    "distance_place_rate",
    "distance_runs",
    # value / drift signals
    "ew_value_index",
    "odds_drift",
    "odds_value_delta",
    # speed / complexity
    "horse_speed",
    "horse_speed_rank",
    "speed_trend",          # short-vs-long speed-proxy trajectory (model-10)
    "race_complexity",
    # class-par performance
    "class_par_speed",
    "class_par_rank",
    # form cycle (trainer + jockey)
    "trainer_hot_strike_rate",
    "trainer_form_zscore",
    "jockey_hot_strike_rate",
    "jockey_form_zscore",
    # interaction features
    "speed_distance_profile",
    "form_market_disagreement",
    "hot_connection",
    "hot_connection_z",
    # market movement (pre-off only, no odds_finish)
    "price_steam_pct",
    "relative_market_share",
    "market_book_pct",
    # freshness / market confidence
    "freshness_band",
    "is_steaming",
    "is_drifting",
    "market_confidence",
]

# Market-price features. For *value betting* the probability model must be
# price-free: if the model sees the market price, its probability is a function
# of the price and EV-vs-market becomes circular. These are stripped to build
# PRICE_FREE_FEATURE_COLS (model variant "v3nf" = no-features-from-price).
PRICE_FEATURE_COLS = [
    "implied_prob",
    "overround_norm_prob",
    "log_odds",
    "market_rank",
    "ew_value_index",
    "odds_drift",
    "odds_value_delta",
    "price_steam_pct",
    "relative_market_share",
    "market_book_pct",
    "is_steaming",
    "is_drifting",
    "market_confidence",
    "form_market_disagreement",
]

# Market-DERIVED features: not prices themselves, but computed from the price
# book, so they carry the market's opinion into a model that is supposed to be
# independent of it. `race_complexity` blends a market-entropy component into an
# otherwise field-shape scalar (features/engine.py::add_race_complexity,
# DECISIONS D27); the step 09 audit confirmed structurally that perturbing the
# price columns moves it on 100% of probe rows while it sat in the price-free
# whitelist, i.e. the "independent" model line was not actually price-free.
# It stays in FEATURE_COLS (the market model may use it, and every existing
# bundle's own meta["feature_cols"] is unchanged) — it is only stripped from the
# price-free list. The independent replacement is `race_complexity_v2`, whose
# adoption into a trained bundle remains step 10's corrected-baseline retrain.
MARKET_DERIVED_FEATURE_COLS = [
    "race_complexity",
]

# Price-free whitelist: identical to FEATURE_COLS minus every market-price and
# market-derived signal. Order preserved so artefacts/feature-importance stay
# readable.
PRICE_FREE_FEATURE_COLS = [
    c for c in FEATURE_COLS
    if c not in PRICE_FEATURE_COLS and c not in MARKET_DERIVED_FEATURE_COLS
]

# Features that carry no measurable signal on the current betSP data (see
# memory/model-15-feature-selection.md). The first five are 100% null because
# their source (Timeform free figures / racecard class) is not populated; the
# last three are present but near-zero on both SHAP and PredictionValuesChange.
# Dropping all eight kept won/placed_2/showed AUC, log-loss, Brier and ECE flat
# (deltas < 0.001) on the held-out tail while shrinking the model.
#
# They are NOT removed from FEATURE_COLS: the five null ones revive automatically
# if a Timeform/racecard source is wired up (model-10 backlog). Use the LEAN_*
# lists below to train the smaller/faster variant from today's data.
EMPIRICALLY_DEAD_COLS = [
    "timeform_rating",   # 100% null — paywalled Timeform
    "rating_rank",       # 100% null — derived from timeform_rating
    "pace_bias",         # 100% null — needs ratings
    "class_change",      # 100% null — no official class in betSP
    "recent_form_avg",   # 100% null — Timeform free finishing figures
    "recent_form_wins",  # low-signal proxy of the above
    "recent_form_runs",  # low-signal proxy of the above
    "going_speed",       # 79% null + low-signal; superseded by going_pref_* (model-10)
]

# Lean whitelists for retraining on the current data (smaller/faster, equal
# accuracy + calibration). Order preserved.
LEAN_FEATURE_COLS = [c for c in FEATURE_COLS if c not in EMPIRICALLY_DEAD_COLS]
LEAN_PRICE_FREE_FEATURE_COLS = [
    c for c in PRICE_FREE_FEATURE_COLS if c not in EMPIRICALLY_DEAD_COLS
]

# Step 04 (memory/improvement/stages/04.md): `race_complexity` mixed a
# market-entropy component into a feature FEATURE_COLS lists as price-free,
# and standardized it against whatever frame happened to be passed to
# features.engine.add_race_complexity (a dataset-global recompute, not a
# frozen train-fitted constant) — an admissible historical row's value could
# shift when later rows were appended to the same build. `race_complexity`
# itself is UNCHANGED above: every already-trained model bundle's
# meta["feature_cols"] (models/train.py) is bound to its original meaning, and
# rewriting it in place would silently serve those bundles different inputs
# than they were trained on (DESIGN.md; DECISIONS D9).
#
# race_complexity_v2 (independent: field size + class/form/rating spread only,
# frozen train-fitted scale — features/_feature_scale.py) and
# race_market_entropy (explicit market-assisted companion: entropy of
# overround_norm_prob) are the corrected replacements. Both are produced by
# features/builder.py today (additive columns) but are NOT yet in FEATURE_COLS
# / PRICE_FREE_FEATURE_COLS — adopting them into a trained model is step 10's
# corrected-baseline retrain, not this step's engineering fix.
CANDIDATE_INDEPENDENT_FEATURE_COLS = ["race_complexity_v2"]
CANDIDATE_MARKET_FEATURE_COLS = ["race_market_entropy"]

# Step 06: going_speed's config-mapped lookup only ever matched an exact
# hyphenated single-word key, missing every "X to Y"-spaced/composite/Irish
# going string — measured 20.92% fill on this training file vs going_speed_v2's
# corrected string matching over the SAME frozen going_speed_map numbers (see
# features/derive.py's block comment above add_going_speed_v2). Additive,
# not yet adopted into FEATURE_COLS (going_speed itself is EMPIRICALLY_DEAD
# below and stays byte-identical for any bundle still listing it).
CANDIDATE_INDEPENDENT_FEATURE_COLS.append("going_speed_v2")
CANDIDATE_INDEPENDENT_FEATURE_COLS.append("going_is_all_weather")

# Step 11: the MEASURED performance family (features/measured_speed.py), built
# from real published winning times, race distances and beaten-lengths margins
# mined out of the raw Sporting Life archive by features/measured_timing.py.
# Deliberately named apart from horse_speed, which stays a finishing-position
# percentile proxy with its meaning unchanged. Price-free by construction (no
# market quantity enters the par or the figure), and additive: off unless an
# experiment opts in via MEASURED_SPEED_FEATURE_COLS, exactly like the D27/D33
# candidates above. Sectionals are NOT here — the archive publishes none.
MEASURED_SPEED_FEATURE_COLS = [
    "msf_last",       # most recent prior normalized figure
    "msf_mean3",      # mean of the last 3 prior figures
    "msf_mean6",      # mean of the last 6 prior figures
    "msf_trend",      # msf_mean3 - msf_mean6 (improving vs regressing)
    "msf_distance",   # prior figures at this distance band
    "msf_going",      # prior figures on this going band
    "msf_rank",       # within-race rank of msf_mean3 (WIN/PLACE books kept apart)
]
CANDIDATE_INDEPENDENT_FEATURE_COLS.extend(MEASURED_SPEED_FEATURE_COLS)

# Step 16: the versioned, optional text-feature group (features/text_features_v1.py),
# joined from llm.text_archive using a point-in-time decision cutoff. Deliberately
# kept in ITS OWN list, never merged into CANDIDATE_INDEPENDENT_FEATURE_COLS —
# unlike race_complexity_v2/going_speed_v2/MEASURED_SPEED_FEATURE_COLS above,
# step 16 found the only archived text corpus (243 rows, one date, one source)
# has no timestamped history outside a single day inside dev_core and ZERO
# coverage in step 10's dev_oos or final_holdout windows, so the pre-registered
# chronological "does text improve future forecasts" comparison could not be
# run on any genuine out-of-sample fold. Forecast validation is DEFERRED_DATA
# (memory/improvement/stages/16.md); nothing here may be adopted into a trained
# or promoted model until a stage with real multi-day/multi-source coverage
# re-runs the frozen protocol in reports/improvement/16/frozen_protocol.json.
from features.text_features_v1 import CANDIDATE_TEXT_FEATURE_COLS  # noqa: E402
