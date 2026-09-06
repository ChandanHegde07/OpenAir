# OpenAir

OpenAir is an entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/), organised by the EUROCONTROL Performance Review Commission. The goal is to predict the **taxi-out time** of flights at 10 major European airports, ranked by RMSE.

Target: `TAXITIME_SEC_mvt` for departures (`MVT_TIME − BLOCK_TIME`). Airports: LTFM, EHAM, LFPG, EGLL, EDDF, LEMD, LEBL, EDDM, LIRF, LSZH.

## Current model

```
P_cal = per-airport OLS(MVT−AOBT, AOBT−EOBT, geo_mean)
if unmatched and airport == LIRF:
    y_hat = MVT − SCHED
else:
    y_hat = P_cal + LightGBM_L2_residual(clocks, geometry, stand, dest, airline,
                                         dis_state_30m, ...)
```

What is in the model, and why:

- **Two clocks:** `MVT−AOBT` (NM off-block to takeoff) and `AOBT−EOBT` (lateness vs plan). These are not interchangeable with `MVT−EOBT` / `MVT−SCHED` as taxi substitutes.
- **Stand×runway geometry:** train-split `geo_mean` hierarchy.
- **LIRF unmatched override:** joint NM miss at LIRF is almost always type-null and is scored with `MVT−SCHED` (always on; gating failed in E13).
- **Residual LightGBM (L2):** trees on `y − P_cal`. Direct trees are weaker on the ranking analogue. Huber / tail-weighted / two-stage losses lost to L2 (E15).
- **Airport disruption state (E16-A):** mean of *other* flights’ `AOBT−EOBT` in the previous 30 min (AOBT already known at scored takeoff; self excluded). Not redundant with own `AOBT−EOBT`.

Linear traffic/queue, airport additive bias, and `FLIGHT_ID` as a tail number were tested and rejected.

## Status

Research is logged in [`status.md`](status.md). Numbers below are **January + July 2025** holdouts from `training_*.parquet` (December in parentheses).

| Model | Overall RMSE | Matched RMSE | Notes |
|---|---:|---:|---|
| Airport mean | 660 | 440 | |
| E3 linear (clocks + geometry) | 577 | 304 | |
| E3 + LIRF unmatched `MVT−SCHED` | 410 | 304 | unmatched rule, not matched |
| E9 residual LGB + override | 378.28 | 256.46 | L2 residual; E15 did not beat this |
| **E16-A + disruption state (current)** | **376.04** (Dec **241.27**) | **253.82** (Dec **226.95**) | `dis_state_30m`; >30 min −15 s; >60 min not fixed |

Remaining matched error is still **tail compression**: flights >30 min are ~4.5% of matched rows and ~45% of matched SSE. E16-A unbiases high-disruption residuals (Q8 mean residual +60 → −3.5 s) but does not fix 1–24 h bombs or LFPG.

## Data policy

**Train, fit, calibrate, encode, and select using `data/training_*.parquet` only.**  
`ranking.parquet` / `submitting.parquet` must not enter fits, statistics, or feature engineering. `experiments/common.py` refuses to load them.

Validation: train all 2025 months except January and July; hold out Jan+Jul. Stress: train Jan–Nov, score December. Rolling features use previous flights only (`shift 1`). Geometry and airport×hour tables are fit on the training split only. Always report **overall** RMSE (dominated by unmatched LIRF) and **matched** RMSE.

## Repository layout

```
data/                # training_*.parquet plus ranking/submitting (not used in research fits)
analysis/            # Discovery (01–05, DISCOVERY_REPORT.md) and experiment packs
  E14/ E15/ E16A/    # figures, tables, reports for those experiments
experiments/         # Runners E0–E16A and shared common.py
  results/           # JSON/txt per experiment
status.md            # Research journal: conclusions, failed approaches, current best
```

## License

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for more details.

Copyright (C) 2026 Chandan Hegde
