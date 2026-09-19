# Favourite-longshot recalibration — OOS report

Source frame: `data/backtests/20260616_195957/scored.parquet` (walk-forward, leak-free; bets at pre-off `ppwap`).
Train slice: `race_date < 2026-03-26` (n=64440). Held-out tail:
`race_date >= 2026-03-26` (n=12883, base rate 0.1121).
Recalibrator: `OddsBandCalibrator` (9 fixed odds bands, isotonic per band, log-odds
blended). Artifact: `models/fl_oddsband_v3nf_calib.pkl`.

## Held-out tail — overall quality (before → after)

- **RAW price-free prob** — Brier 0.09501, ECE10 0.00507, AUC 0.6749
- **F-L recalibrated   ** — Brier 0.08515, ECE10 0.00429, AUC 0.7927

(AUC rises because the pre-off price — legitimately available before the off,
unlike the `odds_finish` close used only for CLV — is a strong predictor the
price-free model cannot see; the recalibrator folds it in *per band* without
discarding the model's within-price ranking.)

## A/E by odds band — full OOS frame (n=77323)

| odds band | n | A/E before | A/E after |
| --------- | - | ---------- | --------- |
| odds-on (<2.0) | 1476 | 2.803 | 1.045 |
| 2.0-4.0 | 7928 | 1.884 | 0.968 |
| 4.0-8.0 | 17823 | 1.239 | 0.983 |
| 8.0-16.0 | 18643 | 0.739 | 0.971 |
| 16.0-34.0 | 14364 | 0.437 | 0.958 |
| longshot (>34) | 17089 | 0.166 | 0.801 |

## A/E by odds band — held-out tail only (n=12883)

| odds band | n | A/E before | A/E after |
| --------- | - | ---------- | --------- |
| odds-on (<2.0) | 226 | 2.524 | 1.006 |
| 2.0-4.0 | 1239 | 1.884 | 0.979 |
| 4.0-8.0 | 2780 | 1.247 | 0.999 |
| 8.0-16.0 | 3214 | 0.723 | 0.948 |
| 16.0-34.0 | 2512 | 0.434 | 0.948 |
| longshot (>34) | 2912 | 0.175 | 0.788 |

## Validation command

```
python -m docs.calibration.fit_fl_recalibrator   # re-fits + regenerates this report
python -m docs.calibration.fl_prototype          # full method comparison (logistic2 vs OBC)
```
