# Submitting / ranking diagnostic — frozen E16-A

**Not a leaderboard submission.** Ranking DEP has no taxi labels; RMSE on ranking cannot be computed.

## Verify submitting.parquet

- rows: **344,841** (expected 344,841)
- `MVT_ID_mvt` unique: **True** (344,841)
- schema: `['MVT_ID_mvt', 'TAXITIME_SEC_mvt']` — **exact**
- `TAXITIME_SEC_mvt` all null: **True** (344,841 null / 0 non-null)

`submitting.parquet` is a 2-column template. Covariates for those IDs are the ranking DEP rows
(Jan+Jul **2026**). Ranking is used only as the prediction-time information set. No ranking
row entered OLS, `geo_mean`, hour baselines, or LightGBM.

## Model

```
P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)   # fit on training only
y_hat = P_cal + LGB_L2_residual(E12 matrix + E16-A disruption cols)
if unmatched and airport==LIRF: y_hat = MVT - SCHED
```

Submitting/ranking scores use a model fit on **all 12 training months** (2025).
Jan+Jul 2025 comparison predictions use the E16-A holdout fit (train excludes Jan+Jul),
overall RMSE **376.04** (journal 376.04).

## Ranking population

- DEP rows scored: 344,841
- ARR context rows: 344,693
- time: 2026-01-01 00:01:55+00:00 → 2026-07-31 23:56:19+00:00
- unmatched: 5,290 (1.53%)
- LIRF unmatched fallback candidates: 383

Months are January and July 2026 only — the same calendar months as the primary 2025 holdout.

## Current OpenAir model predictions on submitting.parquet

**n=344,841  mean=1031.6s  median=960.1s  std=625.2s  min=-3916.6  max=111654.0  fallback=383 (0.11%)**

| | Ranking 2026 (scored) | Jan+Jul 2025 val preds |
|---|---:|---:|
| N | 344,841 | 344,419 |
| min | -3916.55 | -26577.00 |
| P1 | 394.45 | 365.29 |
| P5 | 553.26 | 534.54 |
| P25 | 771.45 | 764.22 |
| P50 / median | 960.14 | 958.72 |
| P75 | 1199.48 | 1197.40 |
| P95 | 1677.13 | 1650.23 |
| P99 | 2382.72 | 2252.85 |
| max | 111654.00 | 93535.00 |
| mean | 1031.62 | 1021.36 |
| std | 625.22 | 606.70 |

| Bin | Ranking n | Ranking % | Jan+Jul 2025 val % |
|---|---:|---:|---:|
| <5 min | 1,231 | 0.36% | 0.48% |
| <10 min | 26,129 | 7.58% | 8.67% |
| 10–20 min | 232,643 | 67.46% | 66.54% |
| 20–30 min | 74,087 | 21.48% | 21.75% |
| 30–60 min | 11,338 | 3.29% | 2.86% |
| >60 min | 644 | 0.19% | 0.18% |

LIRF unmatched fallback used on **383** rows (0.11%).
Jan+Jul 2025 val fallback: 397 (0.12%).

## Sanity

- negative predictions: ranking 51 / val 99
- predictions >4 h: ranking 44 / val 47
- predictions >24 h: ranking 2 / val 1

Ranking predictions are +10.3 s mean / +1.4 s median vs Jan+Jul 2025 val preds. Share predicted ≥30 min differs by +0.43 pp. Largest airport mean shift: EHAM 72.3 s.

The 51 negatives and 2 × >24 h predictions are **not new failure modes**:

- **>24 h (2 rows):** both LIRF unmatched, `y_hat = MVT−SCHED` (111,654 s and 87,484 s). Same E11 rule as validation (val also has a 93,535 s max).
- **>4 h (44 rows):** 43/44 are that same LIRF unmatched fallback.
- **Negatives (51):** 27 are unmatched LSZH (no LIRF rule); 5 are LIRF fallback with takeoff-before-schedule; the rest are residual-tree overshoot. Validation had **more** negatives (99), including a −26,577 s LIRF-fallback value.

The model was not clipped or changed.

## Airport means

| Airport | Rank n | Rank mean | Rank median | Rank >30% | Val n | Val mean | Val median | Val >30% | Δ mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 36,317 | 925.9 | 882.8 | 3.09% | 36,830 | 880.1 | 864.3 | 0.62% | +45.8 |
| EDDM | 25,966 | 846.5 | 790.1 | 1.96% | 27,091 | 814.6 | 779.8 | 0.72% | +31.9 |
| EGLL | 39,840 | 1363.9 | 1330.1 | 7.76% | 40,210 | 1401.5 | 1355.1 | 9.99% | -37.6 |
| EHAM | 38,182 | 868.4 | 796.2 | 2.49% | 40,949 | 796.2 | 737.1 | 1.16% | +72.3 |
| LEBL | 30,080 | 956.2 | 906.4 | 0.54% | 28,986 | 967.4 | 916.2 | 0.48% | -11.2 |
| LEMD | 36,954 | 1027.4 | 1007.0 | 0.65% | 35,328 | 1024.0 | 1004.3 | 0.58% | +3.4 |
| LFPG | 39,872 | 996.6 | 918.8 | 3.49% | 39,612 | 1057.6 | 988.8 | 3.95% | -61.0 |
| LIRF | 26,899 | 1293.5 | 1085.0 | 8.13% | 26,528 | 1333.6 | 1111.4 | 9.37% | -40.1 |
| LSZH | 23,150 | 793.7 | 748.5 | 1.48% | 22,346 | 755.0 | 743.3 | 0.48% | +38.8 |
| LTFM | 47,581 | 1114.0 | 1050.4 | 4.17% | 46,539 | 1073.8 | 1032.8 | 2.29% | +40.2 |

## Disruption-state quantiles (ranking predictions)

| Q | N | Mean dis | Mean pred | Median pred | >30% | >60% |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 43,025 | -143 | 992.2 | 949.0 | 1.48% | 0.08% |
| Q2 | 42,898 | 95 | 971.6 | 929.8 | 1.42% | 0.06% |
| Q3 | 42,959 | 194 | 968.9 | 921.0 | 1.52% | 0.08% |
| Q4 | 43,062 | 280 | 971.9 | 921.3 | 1.65% | 0.12% |
| Q5 | 42,924 | 368 | 985.0 | 923.6 | 1.89% | 0.12% |
| Q6 | 42,972 | 471 | 1006.2 | 943.9 | 2.42% | 0.15% |
| Q7 | 42,917 | 621 | 1067.1 | 989.7 | 3.74% | 0.21% |
| Q8 | 42,929 | 1062 | 1285.7 | 1148.4 | 13.58% | 0.61% |
| nan | 1,155 | nan | 1206.5 | 962.6 | 7.19% | 2.16% |

## Verdict

**The bulk prediction distribution looks reasonable relative to Jan+Jul 2025 validation.** Median 960 vs 959 s; P25/P75 within 10 s; 10–20 min bin 67.5% vs 66.5%; LIRF unmatched fallback 383 vs 397 (0.11% vs 0.12%). Ranking RMSE **cannot** be computed (DEP `TAXITIME` is blank). Extremes are the known unmatched-LIRF `MVT−SCHED` tail, not a 2026-only explosion. Airport mean shifts of ~30–70 s (EHAM +, LFPG −) are within year-to-year mix, not a broken pipeline.

This is **not** a leaderboard score and was **not** submitted. The model was not modified.

## Artifacts

- `analysis/submitting_check/tables/submitting_predictions.parquet`
- `analysis/submitting_check/tables/submitting_predictions.csv` (`MVT_ID_mvt`, `TAXITIME_SEC_mvt`)
- `analysis/submitting_check/tables/janjul_val_predictions.parquet`
- `analysis/submitting_check/figures/`

