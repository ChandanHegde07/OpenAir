# OpenAir

OpenAir is an entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/), organised by the EUROCONTROL Performance Review Commission. The goal is to predict the **taxi-out time** of flights at 10 major European airports, ranked by RMSE.

Target: `TAXITIME_SEC_mvt` for departures (`MVT_TIME − BLOCK_TIME`). Airports: LTFM, EHAM, LFPG, EGLL, EDDF, LEMD, LEBL, EDDM, LIRF, LSZH.

## Current model

Two stages. Stage 1 is the E20 matched/ensemble model; stage 2 is the LIRF gate-delay decomposition (E33/E34).

```
# Stage 1 — E20 (all rows)
geo_mean tables fit on matched train rows only
P_cal = per-airport OLS(MVT−AOBT, AOBT−EOBT, geo_mean)
residual LightGBM trained with LIRF-override rows dropped
y_hat = P_cal + LightGBM_L2_residual(clocks, geometry, stand, dest, airline, dis_state_30m, ...)

# Stage 2 — LIRF unmatched gate-delay decomposition (E33/E34)
D = MVT − SCHED
G_hat = CatBoost | LightGBM trained on 2025 LIRF departures  (G = BLOCK − SCHED)
if unmatched and airport == LIRF:
    y_hat = D − G_hat          # v8: enriched CatBoost, α = 1.0
else:
    y_hat = Stage 1
```

What is in the model, and why:

- **Two clocks:** `MVT−AOBT` (NM off-block to takeoff) and `AOBT−EOBT` (lateness vs plan). These are not interchangeable with `MVT−EOBT` / `MVT−SCHED` as taxi substitutes.
- **Stand×runway geometry:** train-split `geo_mean` hierarchy (matched-only, E18-H).
- **Residual LightGBM (L2)** on `y − P_cal`; Huber / tail-weighted / two-stage lost to L2 (E15); LIRF-override rows are dropped from residual training (E18-H).
- **Airport disruption state (E16-A):** mean of *other* flights' `AOBT−EOBT` in the previous 30 min (self excluded).
- **E20 ensemble (E20):** NNLS blend of CatBoost residual (0.456) + airport-specific LGB experts (0.491) + XGB residual (0.053).
- **LIRF gate-delay decomposition (E33/E34):** `TAXITIME = D − G` with `D = MVT−SCHED` observed and `G = BLOCK−SCHED` the latent gate delay. `G` is predicted from prediction-time movement/surface features on the LIRF slice; only the 383 LIRF-unmatched ranking rows are changed. This is where the recent leaderboard gains come from.

Linear traffic/queue, airport additive bias, and `FLIGHT_ID` as a tail number were tested and rejected.

## Status

Research is logged in [`status.md`](status.md). Numbers below are **January + July 2025** holdouts from `training_*.parquet` (December in parentheses).

| Model | Overall RMSE | Matched RMSE | Notes |
|---|---:|---:|---|
| Airport mean | 660 | 440 | |
| E3 linear (clocks + geometry) | 577 | 304 | |
| E3 + LIRF unmatched `MVT−SCHED` | 410 | 304 | unmatched rule, not matched |
| E9 residual LGB + override | 378.28 | 256.46 | L2 residual; E15 did not beat this |
| E16-A + disruption state | 376.04 (Dec 241.27) | 253.82 (Dec 226.95) | `dis_state_30m`; >30 min −15 s; >60 min not fixed |
| E18-H hygiene | 372.36 (Dec 238.01) | 250.98 (Dec 223.95) | matched-only `geo_mean` + drop LIRF-override rows from residual train |
| E20-nnls ensemble | 368.03 (Dec 228.45) | 244.76 (Dec 215.90) | NNLS 0.456 CatBoost + 0.491 airport-LGB + 0.053 XGB; submission `likable-eagle_v4.parquet`, **LB 316.97** |
| E33 v7 gate-delay (α=0.6) | 345.2 (Dec 226.8) | — | LIRF-unmatched `T = D − 0.6·G`; submission `likable-eagle_v7.parquet`, **LB 292.99** |
| **E34 v8 enriched gate-delay (current)** | **333.7** (Dec 228.6) | — | CatBoost `G` + enriched D/surface features, `T = D − G` (α=1.0); only 383 LIRF rows changed; submission `likable-eagle_v8.parquet`, **est. LB ≈ 281.5** |
| E14-surface-state (research) | 371.57 (Dec 231.09) | 248.05 (Dec 218.40) | 31 strictly-causal features in residual LGB; **not ranking-safe** |
| E17-A2 ranking-safe memory (INCONCLUSIVE) | 374.88 (Dec 238.44) | 251.98 (Dec 224.62) | airport + runway operational memory; small gain, unstable |
| E19-F local queue state (WEAK) | 371.02 (Dec 233.67) | 249.22 (Dec 220.00) | causal queue state; tail not reduced; not submitted |

**Current best = E34 (v8).** The matched model is E20; the leaderboard breakthrough comes from modelling the latent gate delay `G = BLOCK−SCHED` on the LIRF-unmatched slice and reconstructing `TAXITIME = D − G`. v7 reached **LB 292.99**; v8 (enriched CatBoost `G`) lowers Jan+Jul internal to 333.7 (LIRF-unmatched 6033 → 3937), **estimated LB ≈ 281.5**. Current submission: `likable-eagle_v8.parquet`. Remaining error is concentrated in the LIRF gate-delay tail and the matched >30 min tail.

**Ranking-safe variant of the E14 signal (E17-A):** E14's gain came from rolling taxi behaviour, which is not computable at submission time (ranking blanks other DEPs' TAXITIME). E17-A reproduced E16-A exactly and re-added the memory using only ranking-safe clocks (`MVT−AOBT`, `AOBT−EOBT`, `MVT−SCHED`) plus runway-local history: airport memory alone −0.97 s Jan+Jul / −4.38 s Dec, but the runway-local family is unstable and E14's 371.57 is not recovered (gap +3.31). Held as INCONCLUSIVE; the airport-memory family is the keep candidate.

**E19 local queue state:** causal (strictly `< t`, zero TAXITIME) neighbour/same-runway/delay-shock/pressure features on the E18-H residual improved Jan+Jul by only −1.34 s (Dec −4.34 s) and **did not reduce the large-positive tail** (top-1% SSE share unchanged). Core hypothesis falsified: rolling queue-state representations do not identify the >30 min tail rows. Next candidate must be a genuine temporal/sequence queue representation, not more rolling statistics.

## Data policy

**Train, fit, calibrate, encode, and select using `data/training_*.parquet` only.**  
`ranking.parquet` / `submitting.parquet` must not enter fits, statistics, or feature engineering. `experiments/common.py` refuses to load them.

Validation: train all 2025 months except January and July; hold out Jan+Jul. Stress: train Jan–Nov, score December. Rolling features use previous flights only (`shift 1`). Geometry and airport×hour tables are fit on the training split only. Always report **overall** RMSE (dominated by unmatched LIRF) and **matched** RMSE.

## Repository layout

```
data/                # training_*.parquet plus ranking/submitting (not used in research fits)
analysis/            # Discovery (01–05, DISCOVERY_REPORT.md) and experiment packs
  E14/ E15/ E16A/ E18/  # figures, tables, reports for those experiments
experiments/         # Runners E0–E20 and shared common.py
  e14_features.py    # strictly-causal surface-state features (E14-surface-state, unsafe)
  e17a_features.py   # ranking-safe operational memory (E17-A)
  e19_features.py    # causal local queue-state features (E19, weak)
  run_e20_ensemble.py      # E20 experts (LGB/CatBoost/XGB/airport) + blends
  make_submission_e20.py   # full-train ensemble fit -> likable-eagle_v4.parquet
  run_e33_delay_decomposition.py / make_submission_e33.py   # v7: T = D - 0.6*G
  run_e34d_enriched.py / make_submission_e34.py             # v8: enriched CatBoost G
  results/           # JSON/txt per experiment
  results/E14/       # E14-surface-state: metrics.json, summary.txt, feature_importance.csv, plots/
  results/E17-A/     # E17-A: metrics.json, summary.txt, feature_importance.csv, plots/
  results/E19/       # E19: summary.md + metrics/ablation/per_airport/tail/feature CSVs, plots/
  results/E20/       # E20: summary.md, model/blend/error_corr/regime/feature CSVs, oof parquet, plots/
  results/E33/       # E33: v7 report + decomposition JSONs
  results/E34/       # E34: v8 report + G-model JSONs, likable-eagle_v8.parquet
status.md            # Research journal: conclusions, failed approaches, current best
```

## License

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for more details.

Copyright (C) 2026 Chandan Hegde
