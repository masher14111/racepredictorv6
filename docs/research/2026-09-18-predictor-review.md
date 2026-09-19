# Horse-racing predictor improvement review
Reviewed 18 September 2026. Based on the supplied Medium article, primary web sources retrieved with Firecrawl, and read-only inspection of the current project, stored data and audit reports.

**Recommendation:** Keep CatBoost and the race-level LightGBM model as the numerical prediction engines. First repair validation boundaries and market-data consistency, then improve racing data. Trial Qwen3.5-9B as a local comment extractor, with Hermes-4-14B as a challenger. There is no measured evidence yet that either LLM improves this predictor.

No predictor code, configuration or trained models were changed, and no new training or profitability backtest was run for this review.

**What the article contributes**

The [article](https://medium.com/@cagdasgul/high-precision-prediction-of-horse-racing-durations-using-ensemble-machine-learning-models-a-d6af16a1ebf1) compares regression models for finish times, including CatBoost, LightGBM and stacking. Its useful directions are normalized performance features, complementary learners and computationally efficient experiments.

Its headline claim needs independent reproduction. It reports CatBoost MAE of 0.000225 seconds alongside MAPE near 2.06%; for a roughly 100-second race, 2% represents roughly two seconds. Target scaling or unit conversion may explain the discrepancy, but the article does not establish that explanation. Time-based backtesting appears as future work, and its described percentage splits do not demonstrate chronological, race-disjoint testing. Filtering 5% of unusual observations can also change the problem being evaluated.

Treat those numbers as unverified rather than an accuracy target. Overall duration accuracy is not evidence of correct within-race rankings, calibrated win probabilities or profitable bets. A model can explain differences between short and long races while adding little information about which horse wins.

**What your project already has**

The project already contains CatBoost, a genuine grouped-softmax LightGBM model, market-margin removal, separate independent and market-assisted predictions, calibration, a walk-forward audit, race-level bootstrap intervals, immutable odds snapshots and paper-execution safeguards. There is also an older CatBoost/XGBoost/logistic ensemble experiment. These are foundations to repair and compare, not missing systems to rebuild.

The strongest saved evidence is the July audit on 904 races from 13 June to 25 July, reused in the September 18 forward report. Independent-model race log loss was **1.8740 versus 1.6878 for the market**; lower is better. The audit's independent EV-selected portfolio returned **−23.4%**, and its market-adjusted counterpart **−8.5%**, under that historical simulation. Those figures do not measure a new September test. The saved verdict remains NO-GO.

The latest report records zero qualified forward bets and 988 open decision tickets, zero settled. Tickets include PASS decisions, so this is not 988 wagers. The formal forward window is recorded as not started. [Saved audit](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/reports/calibration_audit_20260727.md:66>), [latest forward report](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/reports/forward_validation_20260918.md:35>).

**Priority 1 — repair the experiments before judging another model**

1. **Keep complete races together at every split.** The present CatBoost training function sorts by date and slices rows. Read-only reproduction on the current matrix found 28 races shared between outer training/test, 13 between fitting/calibration, and 45 between fitting/early stopping. Split unique race dates or race IDs, then map the groups back to runners. Apply the same discipline inside tuning and ensemble fitting. Let a horse's earlier runs inform later runs; universal horse-disjoint splitting would answer a different, cold-start question. [Training split](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/models/train.py:94>), [calibration split](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/models/train.py:235>), [tuner](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/models/tuner.py:136>).

2. **Reserve calibration data before hyperparameter selection.** Use chronological training/tuning, a later calibration period, and a newest test period. Fit imputers, categorical encodings, scalers, feature selection, early stopping and blending coefficients inside their appropriate training windows. The currently inspected July test is now known; subsequent decisions require another untouched period. Official [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) explains temporal splitting, but the class ignores groups and does not automatically keep races intact.

3. **Canonicalize WIN and PLACE data before fusion.** The current training file has 260,490 rows: 148,860 WIN and 111,630 PLACE. It does not contain duplicate date/venue/horse keys in the inspected matrix. The issue is that source fusion can choose fields without preserving their market association, while CatBoost training does not filter WIN as LightGBM does. Build one canonical race/horse record and explicitly attach each market's own prices and terms. Never derive win-market probabilities from place prices. [Fusion](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/features/fuse.py:35>), [CatBoost loading path](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/models/train.py:150>), [LightGBM WIN filter](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/models/train_lgbm.py:178>).

4. **Repair the nominally independent feature set.** `race_complexity` still uses dataset-global standardization and market entropy. The audit excluded it, but the production feature list retains it. Use race-local quantities or constants fitted on training only; remove market components from the independent branch. Add an append-future-data check: old feature values should not change when later races are appended. [Feature construction](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/features/engine.py:309>), [feature list](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/models/features.py:57>).

5. **Align historical availability with the actual prediction time.** “Pre-off” alone is insufficient if predictions are made earlier. Choose a defined operating cutoff, such as ten minutes before the off, and use only data available by then. Archive observation and publication times, source, race/runner IDs, declared runners, odds, terms, liquidity and model version. Weighted average historical prices are useful references but do not prove an executable quote existed at a chosen moment.

**Priority 2 — acquire information the model currently lacks**

The current training artifact ends on 25 July. Six inspected fields are entirely empty: `timeform_rating`, `rating_rank`, `pace_bias`, `race_class`, `class_change` and `recent_form_avg`. `going_speed` is only about 20.9% populated. Jockey/trainer IDs are populated now, so the older June data-health document should not be used to claim they are still absent.

| Data or feature family | Recommended experiment |
|---|---|
| Current declarations | Actual jockey, trainer, weight and claims, draw, age, sex, official rating, class, surface, equipment and non-runners; preserve missingness and provenance. |
| Measured performance | Historical race times, beaten margins and sectionals; course/distance/surface/going-adjusted performance relative to pars fitted on earlier races. |
| Pace | Historical run style, likely leaders, contested lead, pace pressure and finishing-speed residuals; interact with draw and course layout. |
| Fitness and development | Days since last run, recency-weighted form, form trend, career stage, layoff and return patterns, first-time equipment. |
| Suitability | Course × distance × going preferences, weight relative to rivals, distance/class changes, strength of previous opposition. |
| Connections | Smoothed trainer/jockey recent and long-run form, partnerships, switches and course context. Shrink sparse records toward a sensible pooled prior. |
| Market information | Timestamped prices and movement, spread, traded/available volume where available, and disagreement between independent reference books. |

The existing `horse_speed` is largely a trailing finishing-position percentile, not measured speed. Adding another learner to that feature does not create timing information. [Current speed proxy](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/features/engine.py:195>).

The live builder fills missing jockey/trainer data from a horse's previous known run. That is past information, but it is not necessarily today's declaration. Use separate “last known” fields and prefer verified current racecards. [Connection fallback](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/features/builder.py:141>).

For an article-inspired duration experiment, predict a normalized performance residual or distribution, not just raw seconds. Feed out-of-fold predictions into the win model and measure incremental gain. Account for non-finishers explicitly rather than simply deleting them, and preserve shared race/pace effects if simulating outcomes from predicted times.

[Timeform's sectional analysis](https://www.timeform.com/horse-racing/features/rowley/the-timeform-knowledge-sectional-analysis-872015) explains why overall times depend on pace and why course-specific finishing-speed pars matter. [Betfair's historical-data documentation](https://support.developer.betfair.com/hc/en-us/articles/360000402211-How-do-I-download-and-view-Betfair-Historical-Data) confirms timestamped exchange history is available. Check sample coverage and terms before purchasing anything; no subscription was purchased in this review.

**Priority 3 — improve numerical modelling with controlled comparisons**

- **Keep a compact benchmark.** Market-only, regularized race-level conditional logit, existing CatBoost, and existing grouped-softmax LightGBM should be the core. Add XGBoost only as a challenger with the same data, folds and tuning budget.
- **Fit market combination on unseen races.** A useful baseline is `p_i ∝ p_fundamental_i^a × p_market_i^b`, normalized over all runners. Learn a and b from chronological out-of-sample predictions. Compare market-only, independent-only and combined versions. [Benter's original paper](https://datagolf.com/static/blogs/benter_paper.pdf) explains this approach and the danger of apparently calibrated models overestimating their selected value bets.
- **Keep reference price separate from executable price.** A consistent complete reference book helps estimate probabilities; the available bookmaker price determines potential return. A better quote should not automatically increase estimated ability. The July audit already found that price-conditioned recalibration could create apparent edges that lost.
- **Choose ensembles by complementary errors.** Try a simple blend before complicated stacking. Train the stacker only on chronological out-of-fold predictions. More models are useful only when the combined forecast improves a new period.
- **Tune for probability quality.** Use race-level log loss as a primary selection score, supported by Brier score and calibration. Compare unweighted or race-balanced training with the current class/odds weighting; weighting changes the learned distribution and may require correction. AUC or strike rate alone cannot select the best probability model.
- **Calibrate after fitting and blending.** Compare simple temperature/logistic calibration with more flexible methods on disjoint data. Inspect selected bets, odds bands, field sizes, race codes and missing-data levels. Renormalizing probabilities does not itself establish calibration. [Official calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
- **Model racing regimes without fragmenting the data too much.** Begin with pooled models and surface/race-code interactions; test separate flat turf, all-weather, hurdle and chase models only where sample sizes support them.
- **Make place probabilities coherent.** Account for number of paid places, field size and bookmaker terms. Compare a joint finish-order model with the existing separate targets; win/top-two/top-three probabilities should obey logical ordering.
- **Treat ranking scores as scores.** Learning-to-rank can be a useful auxiliary model, but its outputs require a separately validated probability conversion. [XGBoost ranking documentation](https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html).
- **Explore newer tabular models later.** TabPFN is a relevant challenger; a general chat LLM is a different tool. Its published successes do not establish racing performance. Sequence models, graph features and neural ensembles belong after data repairs and simpler comparisons. [TabPFN paper](https://www.nature.com/articles/s41586-024-08328-6), [TabPFN-2.5 report](https://arxiv.org/abs/2511.08667).
- **Measure actual feature contribution.** Remove groups of features and use held-out permutation checks. SHAP can help explain model behavior but does not prove causality or added betting value.

**Hermes or another local LLM**

Your inspected hardware is an RTX 5070 Ti with roughly 16 GB VRAM, 32 GB system RAM and an i5-10500. It supports a sensible small-model extraction trial.

| Model | Role |
|---|---|
| [Qwen3.5-9B in Ollama](https://ollama.com/library/qwen3.5:9b) | First trial. The listed Q4_K_M model file is 6.6 GB. Runtime memory also includes context, buffers and other GPU workloads. |
| [Hermes-4-14B](https://huggingface.co/NousResearch/Hermes-4-14B) | Challenger using a supported 4-bit build. Its model card specifically describes JSON/schema training; racing extraction quality still needs measurement. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Smaller throughput candidate if it retains sufficient extraction accuracy. |

This is a fit-to-hardware shortlist, not a claim that Qwen is universally more accurate. Larger 27B/70B models are not the first investment. If “Hermes” means [Hermes Agent](https://hermes-agent.nousresearch.com/docs/guides/local-ollama-setup), that is an assistant framework, distinct from the model; its agent features are unnecessary for this narrow extraction job.

The current extractor is disabled, defaults to regex with an Ollama `llama3.1` setting, and is not called by the production feature pipeline. **Changing the model name alone currently changes no predictions.** [Configuration](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/config.yaml:500>), [extractor](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/llm/text_features.py:1>).

Its useful role is to turn dated comments into grounded fields: hampered/blocked, wide trip, run style, jumping mistakes, unsuitable ground or distance, first-time equipment and reasons for a layoff. Attach the evidence phrase, source and availability time, and permit “unknown.” Distinguish absence of mention from a confirmed negative. Keep opinions and odds references separate from factual features in a supposedly independent model.

Before integration:

- Fix string-to-Boolean conversion: `bool("false")` becomes true in the current implementation. [Parsing](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/llm/text_features.py:222>).
- Fix the regex that treats “suited by the soft ground” as a ground excuse. [Pattern](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/llm/text_features.py:72>).
- Validate a strict schema and record the actual backend. Current regex fallback can be cached under the requested LLM model key, confusing evaluation.
- Archive text versions. The inspected Spotlight file contains only 243 comments from one race day, 17 June; its scraper replaces older versions. Historical extraction needs timestamped snapshots. [Scraper](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/scraper/spotlight.py:172>).

Use [Ollama schema-constrained output](https://docs.ollama.com/capabilities/structured-outputs), bounded context/output, and non-thinking mode where supported. Extract in batches outside GPU training, cache by text/model/prompt/schema version, and keep failures visible.

Build a labelled set of roughly 500–1,000 comments across dates and sources, with enough examples of rare flags and negations. Compare repaired regex, Qwen 4B/9B and Hermes 14B on per-field precision/recall, false positives, unknown handling, valid schema, latency and memory. Freeze the prompt before evaluation. Ask for extraction from supplied text, not remembered facts; prospective testing also reduces historical-result memorization concerns.

Then compare **no text → regex text → LLM text** in the same chronological racing experiment. Promote only if the added fields improve future probabilities and paper results. Fine-tuning/LoRA becomes worthwhile only after recurring domain errors and sufficient labelled data justify it.

**Priority 4 — turn prospective records into reliable evidence**

The existing snapshot and friction machinery is useful. Verify the complete route from fresh racecards to prediction, available quote, candidate/PASS decision, result, closing line and settlement. Use canonical race-plus-horse keys; the daily loop's horse-ID-first result lookup deserves review because a horse has multiple historical runs. [Daily paper loop](<C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4/scripts/daily_paper_loop.py:227>).

Replay actual quote timing, non-runners, Rule 4 deductions, dead heats, each-way terms, commission, slippage and any liquidity constraints. For bookmaker bets, basic expected return is `p × decimal_odds − 1`; exchange costs and settlement rules require their own implementation. Small probability errors can erase a small quoted edge.

Track race log loss, Brier, calibration among selections, closing-line value, net ROI, turnover, drawdown, losing streak and race/bet counts. Resample whole races or days, not independent runners. Reconcile log-CLV and percentage-CLV labels; the saved −0.1342 log CLV is not literally a −13.42% simple return.

Pre-register the model, cutoff, features, threshold and staking rule. The configured minimum of eight weeks, 200 qualified bets and 150 races is a gate, not a guarantee of sufficient statistical power. Keep paper operation while repairing the system; do not lower gates just to produce selections. Treat positive CLV as supporting evidence, not proof of profitability.

Monitor freshness, missing fields, source coverage, runner completeness, feature drift, subgroup loss and selected-bet calibration. Preserve frozen model/data hashes and a previous working model. Investigate drift causes before automatic retraining; a feed failure can resemble a sporting change.

**Recommended experiment sequence**

| Order | Deliverable | Evidence required |
|---|---|---|
| 1 | Correct race-grouped splits, canonical WIN records and independent features | No shared races across boundaries; no market-type mismatch; historical features invariant to future appends. |
| 2 | Updated, timestamped racecards/results and dependable settlement | Verified current declarations; recent feature coverage; correct race/horse results; auditable decision records. |
| 3 | Rebuilt market, CatBoost and LightGBM baselines | Same untouched races, race-level metrics and uncertainty, with the strongest market benchmark. |
| 4 | Speed/sectionals, pace and missing-data repairs | Feature-group improvement across several later windows. |
| 5 | Small complementary ensemble and learned market blend | Incremental gain beyond the best component and market-only model. |
| 6 | Qwen versus Hermes extraction benchmark, then text integration | Better extraction and better future race forecasts; both stages must pass. |
| 7 | Frozen prospective paper run | Actual captured decisions and settled results under unchanged rules. |

The highest-return engineering work is likely the first two rows. That is a judgment from the defects and missing data found here, not a promise of a particular accuracy or profit increase.

Source captures are saved in [the research folder](<C:/Users/mshr/Documents/Race Predictor v4/.firecrawl>).
