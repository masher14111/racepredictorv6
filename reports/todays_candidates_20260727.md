# Today's candidates — 2026-07-27

_Generated 2026-07-27T21:41:12+00:00._

| gate | state |
|---|---|
| Model | **NO-GO** |
| Forward release | **FORWARD GATE NOT MET (9 of 9 criteria failed)** |
| Deployment | **PAPER-ONLY** |

## No real-money recommendations

This report contains **no real-money recommendations**, because at least one release gate is not met. The criteria that failed:

- **Model gate: NO-GO.** The model has not been shown to beat the de-vigged market on the headline line.
  - logloss_edge_not_significant:delta=-0.18621 ci95=[-0.22507,-0.14620]
  - loses_to_best_devig:shin=1.68568
  - clv_not_positive:mean=-0.13417 ci95_lower=-0.13888
  - drift_gate_failed:going_speed=4.916, horse_career_runs=0.301
- **Forward-release gate not met.**
  - `model_go`: observed NO-GO, required Stage-4 model verdict == GO
  - `min_weeks`: observed 0.0, required >= 8 weeks of forward tracking
  - `min_qualified_bets`: observed 0, required >= 200 qualified bets
  - `min_qualified_races`: observed 0, required >= 150 qualified races
  - `positive_mean_clv`: observed n/a, required mean CLV > 0
  - `clv_ci_lower`: observed n/a, required 95% race-clustered CI lower bound > 0
  - `ae_stable`: observed n/a, required 0.9 <= A/E <= 1.1
  - `calibration`: observed n/a, required ECE <= 0.03
  - `drawdown`: observed n/a, required max drawdown <= 0.1 of bankroll

Runners are listed below **for paper tracking only**. A row in this table is a shadow ticket: it records what the system would have done so that forward evidence can accumulate. It is not advice and no money is staked.

## No candidates

Nothing cleared the candidate gate today. **"No bet" is a valid and expected output** — the gate passes by default and only produces a candidate when every condition is affirmatively met.

## PASS — why each runner was declined

356 runners considered, 356 declined. A missing input is a PASS reason, never a waiver.

### By condition

| condition | runners stopped | share |
|---|---|---|
| min_expected_value | 356 | 100.0% |
| model_validation | 356 | 100.0% |
| calibrated_probability | 356 | 100.0% |
| executable_price | 356 | 100.0% |
| min_edge | 356 | 100.0% |
| source_health | 356 | 100.0% |
| overround | 243 | 68.3% |
| runner_history | 50 | 14.0% |

### Distinct reasons

| reason | runners |
|---|---|
| executable_price:age_unknown | 356 |
| source_health:livescorebet:stale=50465s>900s | 356 |
| model_validation:NO-GO | 356 |
| model_validation:logloss_edge_not_significant:delta=-0.18621 ci95=[-0.22507,-0.14620] | 356 |
| model_validation:loses_to_best_devig:shin=1.68568 | 356 |
| model_validation:clv_not_positive:mean=-0.13417 ci95_lower=-0.13888 | 356 |
| model_validation:drift_gate_failed:going_speed=4.916, horse_career_runs=0.301 | 356 |
| min_edge:unavailable | 356 |
| min_expected_value:unavailable | 356 |
| calibrated_probability:market_adjusted_only | 336 |
| source_health:paddy_power:stale=50441s>900s | 259 |
| runner_history:no_prior_form | 50 |
| overround:1.6799>1.2500 | 23 |
| overround:1.7984>1.2500 | 23 |
| calibrated_probability:missing | 20 |
| overround:1.4094>1.2500 | 15 |
| overround:1.4493>1.2500 | 15 |
| overround:1.3321>1.2500 | 14 |
| overround:1.3528>1.2500 | 14 |
| overround:1.3953>1.2500 | 14 |
| overround:1.3721>1.2500 | 13 |
| overround:1.3395>1.2500 | 13 |
| overround:1.3509>1.2500 | 13 |
| overround:1.3718>1.2500 | 12 |
| overround:1.3408>1.2500 | 12 |
| overround:1.3174>1.2500 | 12 |
| overround:1.2890>1.2500 | 11 |
| overround:1.3673>1.2500 | 11 |
| overround:1.3109>1.2500 | 10 |
| overround:1.2569>1.2500 | 9 |
| overround:1.2672>1.2500 | 9 |

### Per runner

Conditions failed, one row per runner. Full reason text for every runner is in `reports\candidate_decisions_20260727.csv`.

| horse | race | conditions failed |
|---|---|---|
| Little Lady Karen | 2026-07-27T13:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Bymiddaytomorrow | 2026-07-27T13:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| March Lilly | 2026-07-27T13:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Kanzi | 2026-07-27T13:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Tegernsee | 2026-07-27T13:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Glenna | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Desert Belle | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Flash Kozo | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Tenison | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Easwrith Destiny | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| It's Only Fun | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Further Measure | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Wave Power | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Marbuzet | 2026-07-27T13:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Mr Cool | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Kelpie Grey | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Cotai Starlight | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Water Of Leith | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Tap Dancer | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Great Profit | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Lanarra | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Joshuas Dream | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Falcon Queen | 2026-07-27T13:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Humphrey | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Kokbastau | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sharp Move | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Hello Friend | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Turret | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Centrum | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Alfred Wincham | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Marianita | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Bankatary | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Eshowe | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Pic N Mix | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sandpyper | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Crack Of Thunder | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Argenteus | 2026-07-27T14:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Imperial Guard | 2026-07-27T14:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Native Honey | 2026-07-27T14:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Forbidden Colours | 2026-07-27T14:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Abduction | 2026-07-27T14:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Carmel Valley | 2026-07-27T14:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Roc De Fer | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Fans Favourite | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| James Choice | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sun Lord | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Seraglio Point | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Prince Of Calypso | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Brave Leader | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Ziggy's Avenger | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Five Moons | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Tinsel | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Alkumatic Jo Jo | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Miss Pretty | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Corallience | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Carwyn | 2026-07-27T14:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Runninsonofagun | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Union Island | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Approaching Dawn | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Novak | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Keats House | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Spirit Of Nature | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Samra Star | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Mayor Of Maghera | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Eeetee | 2026-07-27T14:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Venetian Lion | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sagremor | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Kiani King | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Back At One | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Cobalt Comet | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Hatamoto | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Thisonesforyou | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Topathemorning | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Telfy Boy | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Medyg | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Filly Eilish | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Alkaios | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Bosom Pals | 2026-07-27T15:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Empirical | 2026-07-27T15:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Pearl Eye | 2026-07-27T15:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Colourband | 2026-07-27T15:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Geo | 2026-07-27T15:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Alpine Sierra | 2026-07-27T15:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Rising Tiger | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Bullrider | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Roosike | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Zambezi Shark | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Hiram Bingham | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Real Edition | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Peaberry | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Astral Calling | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Lopefernza | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Brazilian Blitz | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Big Bad Storm | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Brondesbury | 2026-07-27T15:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Doon The Glen | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Wee Mary | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Supremissy | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Summerstorms Dream | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Startling | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Ski Angel | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Royal Duke | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Henery Hawk | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Sixcor | 2026-07-27T15:45:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Storm Free | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Spaceman | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Breakdancer | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Chalk Mountain | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Asteverdi | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Ash Wednesday | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Lucky Luna | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Exotic Baby | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Peter The Wolf | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Tropez Power | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| City Of Kings | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Bloodsweatandtyres | 2026-07-27T16:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Witches Familiar | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Carmel's Phoenix | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Arouet | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Mino Des Mottes | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Munsif | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Dawn Coming | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Mento | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Tex Amare | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Like An Ocean | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| William F Browne | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| The Chancie Grey | 2026-07-27T16:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| London Is Blue | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Kiss And Run | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Hint Of Humour | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Merrimack | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Moe's Legacy | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Beaumadier | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Alkuwarrior | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Too Darn Good | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Magna | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Angel Summer | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Dark Alley | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| In The City | 2026-07-27T16:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Recobella | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Royal Blaze | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Falcon Nine | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Penelope's Sister | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Daring Leader | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Ravenscraig Castle | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Just Dottie | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Zebra Star | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Golden Valour | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Betty Bassett | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| One More Bottle | 2026-07-27T16:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Cogitate | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Star Of Mali | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Padua | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Shaman Champion | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Eminency | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Korbut | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Feel The Need | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| King Of Fury | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Chale Chalo | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| White Crown Star | 2026-07-27T16:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Hopeful Hero | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Westoftignes | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Drombane | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Mart Lane | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Can Happen | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Dairy Force | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Whats New | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Allo Al Khawaneej | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Let's Go La Fichad | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Timurshah | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Soldante | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Metamorpheus | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Searcog | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sestini | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Quint Major | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Bruant | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Inchiquin Star | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Figero | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| In The Minus | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Brosna Town | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Claude | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Woodstream Lad | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Desert Friend | 2026-07-27T16:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Lucy The Wire | 2026-07-27T16:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Sweet Horizon | 2026-07-27T16:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Ziata | 2026-07-27T16:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Hotel California | 2026-07-27T16:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Star Suepreme | 2026-07-27T16:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Harswell River | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Sports Day | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Baldosa | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Riddikulus | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Diamond Aura | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Balmoral Boy | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Lc Tiffen | 2026-07-27T17:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Modern Times | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Irish Nectar | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Fifty Nifty | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Belsito | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Bright | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Monsieur Kodi | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Secret Guest | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Cairdeas | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Hierarchy | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| D Flawless | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Pickersgill | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Beyond Borders | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Not Me | 2026-07-27T17:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| One Number | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| City Of Gold | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Our Boy Bailey | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Trean | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Yellow Sky | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Hmas Perth | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Moliere | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Dakota Jack | 2026-07-27T17:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Harry Knows | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Down To You Kid | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Boysofwallstreet | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Angel Ang | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Rex Regum | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Bollengo Boy | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Q T Boy | 2026-07-27T17:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Pure Mint | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Towelontheterrace | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Reflect On Glory | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Akaraka | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Green Bay Dream | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Peacock's Way | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Khazamh | 2026-07-27T17:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Daddy Long Legs | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Tounsivator | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Too Bossy For Us | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Light Up The Dark | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Eagle Fang | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Teed Up | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Granite Bay | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Galileo Dame | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Filey Bay | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Lark In The Mornin | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Highwind | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Immutable | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Comfort Zone | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Glenroyal | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sirius | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Clear Quartz | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Smooth Tom | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Westminster Moon | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Holy See | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Plontier | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Jabbar | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Yashin | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Tyson Fury | 2026-07-27T17:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Forever Glamorous | 2026-07-27T17:53:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Guernsey Angel | 2026-07-27T17:53:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Solar Invincible | 2026-07-27T17:53:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Pull The Rug | 2026-07-27T17:53:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Madman | 2026-07-27T17:53:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Exhibitioning | 2026-07-27T17:53:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Fractional | 2026-07-27T18:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Prosperity | 2026-07-27T18:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Princess Honey Bee | 2026-07-27T18:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Flawless Fusion | 2026-07-27T18:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Roccapina | 2026-07-27T18:05:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Dancing Saxon | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Expert Dancer | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sagasti | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Motta Alta | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Coincidental Glory | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Nazario | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Girl Bear | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Unfamiliar | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Caitouna | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Templenoe | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Dervo Annie | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Slight Of Foot | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Imnotleavinyou | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Final Boss | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Coolshine | 2026-07-27T18:15:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Regional Rock | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Sudden Flight | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Amazonian Dream | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Another Abbot | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Uncle Don | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Brosay | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Ancient State | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Mesaafi | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Hucklesbrook | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Nad Alshiba Green | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Dandana | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Willem Twee | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Maelstrom | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Star Chorus | 2026-07-27T18:28:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Harley | 2026-07-27T18:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Paper View | 2026-07-27T18:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Eleven Eighty Two | 2026-07-27T18:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Empire Rising | 2026-07-27T18:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Moriarty Moon | 2026-07-27T18:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Aura Of Melania | 2026-07-27T18:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Vantage Code | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Darius Dark | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Madbadanddangerous | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sharkeyboy | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Themis | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Nod Of Approval | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sir Benji | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Nermal | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Confused | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Fits Perfect | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Admiral Will Brown | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| So Must I | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Misty Cove | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Ryefield Dasher | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Solas Na Gealai | 2026-07-27T18:50:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Berlinetta | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Eazy On The Eye | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Khuskhas | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Star Of Dubai | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Box Clever | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Denby's Dream | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Cixi | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Mimi's Magic | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Toralou | 2026-07-27T19:00:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Masked Warrior | 2026-07-27T19:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Noelan Star | 2026-07-27T19:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Enchant | 2026-07-27T19:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, runner_history, source_health |
| Beyond The Bar | 2026-07-27T19:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Daydreama | 2026-07-27T19:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Command The Stars | 2026-07-27T19:10:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Power Of Knowledge | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Teofil | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Which Flannerys | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Prime Contender | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Lanespark | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Ddakji | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Dreams Are Forever | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Walkineezy | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Sharbel | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Witness D'Fitness | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Hill Silver | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| Sunkist Orange | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, runner_history, source_health |
| Society Gent | 2026-07-27T19:20:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, overround, source_health |
| My Old Mate | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Takeitorleaveit | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Time To Sparkle | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Katalyst | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Rugby Union | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Sweep In Time | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Give Me Sun | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Thomas Picton | 2026-07-27T19:30:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| What A Tahoo | 2026-07-27T19:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Cheerleader | 2026-07-27T19:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Zuffolo | 2026-07-27T19:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Doralee | 2026-07-27T19:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Brain Freeze | 2026-07-27T19:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |
| Dunnington Lad | 2026-07-27T19:40:00+00:00 | calibrated_probability, executable_price, min_edge, min_expected_value, model_validation, source_health |

---

_This system is paper-only. No real money is ever staked. Staking figures shown are risk ceilings under a conservative fractional-Kelly policy, not profitability claims._
