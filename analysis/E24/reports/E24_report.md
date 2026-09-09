# E24 — Quantile residual LightGBM

**Project:** OpenAir
**Experiment:** E24
**Validation:** Jan+Jul 2025 training holdout
**Stress:** December 2025
**Data:** 12 `training_*.parquet` only. No METAR, no ranking/submitting.
Script: `experiments/run_e24_quantile.py`.

## Question

E15 closed Huber / tail-weighted L2 / two-stage P(y>30) — those still
fit a **conditional mean** with a different loss. Quantile regression
fits the **conditional distribution** (pinball). Median (α=0.5) and a
weighted quantile blend are the point estimates. Also tested as a
fourth NNLS expert next to E20's CatBoost / airport LGB / XGB.

## Setup

- Frozen: P_cal, matched-only geo_mean, LIRF-override rows dropped,
  always-on LIRF `MVT−SCHED`, E16-A disruption cols, LightGBM capacity.
- L2 reproduction must match E18-H 372.36 / 250.98.
- Quantiles α ∈ {0.1, 0.5, 0.9} on `y − P_cal` (lower / median / upper).
- Point estimates: Q50 (median residual) and NNLS of {L2, Q10, Q50, Q90}.
- Expert test: NNLS of E20 C/D/E plus Q50 (does pinball carry weight?).

## Jan+Jul 2025

| Model | Overall | Matched | <20 | >30 | >60 | SSE>30 | mean resid |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2 | 372.36 | 250.98 | 174.0 | 798.6 | 2402.4 | 45.2% | -6.4 |
| Q10 | 467.25 | 376.26 | 193.4 | 1313.5 | 3500.3 | 54.4% | 217.9 |
| Q50 | 380.90 | 263.01 | 162.4 | 910.0 | 2771.0 | 53.5% | 16.9 |
| Q90 | 456.33 | 363.76 | 339.6 | 686.4 | 2105.3 | 15.9% | -236.9 |
| E20 | 368.03 | 244.76 | 170.9 | 774.8 | 2361.4 | 44.7% | -7.5 |
| Qblend | 378.92 | 260.79 | 207.5 | 738.0 | 2299.3 | 35.8% | -68.2 |
| E20+Q50 | 368.03 | 244.76 | 170.9 | 774.8 | 2361.4 | 44.7% | -7.5 |
| E20 ensemble (ref) | 368.03 | 244.76 | — | — | — | — | — |

## December 2025

| Model | Overall | Matched | <20 | >30 | >60 | SSE>30 | mean resid |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2 | 238.01 | 223.95 | 154.1 | 716.2 | 2114.6 | 39.6% | 20.6 |
| Q10 | 351.25 | 340.21 | 198.9 | 1126.9 | 3069.9 | 42.4% | 217.8 |
| Q50 | 246.79 | 232.68 | 147.5 | 797.4 | 2433.0 | 45.4% | 38.1 |
| Q90 | 317.30 | 307.75 | 292.8 | 549.3 | 1736.1 | 12.3% | -193.6 |
| E20 | 228.69 | 214.78 | 151.5 | 672.6 | 2031.7 | 37.9% | 14.9 |
| Qblend | 238.43 | 224.86 | 178.1 | 642.6 | 1990.0 | 31.6% | -36.8 |
| E20+Q50 | 228.69 | 214.78 | 151.5 | 672.6 | 2031.8 | 37.9% | 14.9 |
| E20 ensemble (ref) | 228.45 | 215.90 | — | — | — | — | — |

Quantile-family NNLS (Jan+Jul): L2=0.732, Q10=0.000, Q50=0.000, Q90=0.268

E20+Q50 NNLS (Jan+Jul): C=0.456, D=0.053, E=0.491, Q50=0.000

## Reading

L2 reproduced E18-H exactly (372.36 / 250.98). The three pinball models do
what the conditional distribution says they should, and **contest RMSE
does not want that**:

- **Q10** underpredicts (mean residual +218 s). Overall 467.
- **Q50 (median)** is the MAE analogue: `<20` 174→162, `>30` 799→910, overall 372→381. Same shape as E15-A Huber.
- **Q90** is the tail analogue: `>30` 799→686 and SSE>30 45%→16%, but `<20` 174→340 and overall 456. Same shape as E15-B tail-weighted L2.

NNLS on {L2, Q10, Q50, Q90} keeps L2 (0.73) + Q90 (0.27) and **zeros the median**. That mix is still worse than L2 (378.92 vs 372.36). As a fourth E20 expert, Q50 gets **weight 0** — CatBoost / airport LGB / XGB already dominate a median LightGBM residual.

Pinball is a different *objective* from E15. On this feature set it is not a different *RMSE solution*. Do not add it to the ensemble.

## Decision

**Verdict:** REJECTED

Q50 (median) Jan+Jul 380.90/263.01 vs L2 372.36/250.98; >30 798.6→910.0; <20 174.0→162.4. E20+Q50 NNLS Q50-weight=0.000: Jan+Jul 368.03 vs E20 368.03; Dec 228.69 vs E20 228.69. Qblend 378.92/238.43.

Does not have to beat E20 as a standalone to be useful. If Q50 takes
NNLS weight and does not hurt December, keep it as an expert candidate.
Readme current-model is updated only if E20+Q50 beats E20 on both splits.

Artifacts: `analysis/E24/`.

