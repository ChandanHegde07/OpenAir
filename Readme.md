# OpenAir

OpenAir is an entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/), organised by the EUROCONTROL Performance Review Commission. The goal is to predict the **taxi-out time** of flights at 10 major European airports, ranked by RMSE.

Target: `TAXITIME_SEC_mvt` for departures (`MVT_TIME − BLOCK_TIME`). Airports: LTFM, EHAM, LFPG, EGLL, EDDF, LEMD, LEBL, EDDM, LIRF, LSZH.

## Current model

Two stages. Stage 1 is the E20 matched/ensemble model; stage 2 is the LIRF gate-delay decomposition (E33–E35).

```
# Stage 1 — E20 (all rows)
geo_mean tables fit on matched train rows only
P_cal = per-airport OLS(MVT−AOBT, AOBT−EOBT, geo_mean)
residual LightGBM trained with LIRF-override rows dropped
y_hat = P_cal + LightGBM_L2_residual(clocks, geometry, stand, dest, airline, dis_state_30m, ...)

# Stage 2 — LIRF unmatched gate-delay (E33–E34 / v8). Do not use v9 unmatched-only G (LB 300.95).
D = MVT − SCHED
G_hat = CatBoost G on the LIRF slice (v8: all LIRF, enriched)
if unmatched and airport == LIRF:
    y_hat = D − G_hat
else:
    y_hat = Stage 1

# Stage 3 — matched tail gate (E48 / v11)
# ranking-safe Δ = y − (MVT−AOBT) LightGBM; apply only if v8>1460 or (P+Δ−v8)>200
if matched and gate:
    y_hat = 0.5 * Stage12 + 0.5 * clip(MVT−AOBT + Δ_hat, 0)
```

What is in the model, and why:

- **Two clocks:** `MVT−AOBT` (NM off-block to takeoff) and `AOBT−EOBT` (lateness vs plan). These are not interchangeable with `MVT−EOBT` / `MVT−SCHED` as taxi substitutes.
- **Stand×runway geometry:** train-split `geo_mean` hierarchy (matched-only, E18-H).
- **Residual LightGBM (L2)** on `y − P_cal`; Huber / tail-weighted / two-stage lost to L2 (E15); LIRF-override rows are dropped from residual training (E18-H).
- **Airport disruption state (E16-A):** mean of *other* flights' `AOBT−EOBT` in the previous 30 min (self excluded).
- **E20 ensemble (E20):** NNLS blend of CatBoost residual (0.456) + airport-specific LGB experts (0.491) + XGB residual (0.053).
- **LIRF gate-delay (E33–E34 / v8):** `TAXITIME = D − G`. v8 trains G on all LIRF. v9 unmatched-only G **hurt LB (300.95)**.
- **Matched tail gate (E48 / v11):** blend v8 toward `P+Δ` (`P = MVT−AOBT`) on ~12% of matched rows. **LB 287.71.**

Linear traffic/queue, airport additive bias, and `FLIGHT_ID` as a tail number were tested and rejected. E36–E41 (matched residual, coordinate-free ground state, clock fusion, Δ reconstruction, tail calibration, exact AOBT surface queue) did **not** beat v9 and are recorded as rejected.

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
| E34 v8 enriched gate-delay | 333.7 (Dec 228.6) | — | CatBoost `G` + enriched features, `T = D − G`; submission `likable-eagle_v8.parquet`, **LB 288.90** |
| E35 v9 selection-aware gate-delay (REJECT) | 331.3 (Dec 227.3) | — | unmatched-only G; **LB 300.95** (worse than v8) |
| E14-surface-state (research) | 371.57 (Dec 231.09) | 248.05 (Dec 218.40) | 31 strictly-causal features in residual LGB; **not ranking-safe** |
| E17-A2 ranking-safe memory (INCONCLUSIVE) | 374.88 (Dec 238.44) | 251.98 (Dec 224.62) | airport + runway operational memory; small gain, unstable |
| E19-F local queue state (WEAK) | 371.02 (Dec 233.67) | 249.22 (Dec 220.00) | causal queue state; tail not reduced; not submitted |
| E38 off-block clock proxies (small) | — | 243.5 (Jan+Jul, expert) | `MVT−LOBT/IOBT/EOBT` add signal but split-unstable; not shipped |
| E41 exact AOBT surface queue (REJECT) | — | queue model 309 | interval-overlap occupancy at AOBT; optimal blend weight 0; no top-5% SSE cut |
| E42 flight-identity residual prior (REJECT) | — | 249.64 | empirical-Bayes identity priors degrade matched and tail on both splits |
| E43 non-LIRF unmatched D−G (REJECT) | — | LFPG 3487 vs 3441 | LFPG unmatched is extreme true-taxi rows, `corr(T,D)≈0`; D−G does not transfer |
| E44 matched tuning/ensembling depth (REJECT) | — | 244.33 | no HP search/seed-avg in E20; deeper LGB + NNLS = −0.43 s, no tail gain |
| E45 LIRF stand-release G-regime (REJECT) | — | LIRF_u 3753→3683 | delay-window stand reuse is a real G split (2 s vs 3844 s) but redundant with v9 G; overall −0.91 s, Dec ~0 |
| E46 OpenSky/OPDI unmatched AOBT (REJECT) | — | — | unmatched flights are in ADS-B (LIRF 79%) but `first_seen` is lift-off (MVT−fs ≈ −26 s); FCO origin parking events = 0 |
| E47 OPDI-in + stand-release G (v10, REJECT) | 329.78 (Dec 226.73) | — | **LB 296.32**; v9 scored **300.95**. Production is **v8 288.90** |
| E48 v11 matched tail-gated Δ | overall 366.55 (Dec 227.20) | 242.49 (Dec 213.17) | v8 + gated `P+Δ`; **LB 287.71** |
| E49 v12 grec P+Δ with e20 feature | 364.50 (Dec 226.58) | 239.32 (Dec 212.50) | **LB 284.97** |
| E50 v13 grec + leftover hat | 363.98 (Dec 226.19) | 238.52 (Dec 212.08) | **LB 284.10** |
| E51 v14 L2+quantile Δ mix (REJECT) | 364.01 (Dec 225.83) | 238.58 (Dec 211.70) | **LB 284.13** (worse than v13 284.10) |
| E52 v15 op + pos leftover (REJECT) | 363.95 (Dec 226.04) | 238.48 (Dec 211.92) | **LB 284.37** (worse than v13 284.10) |
| E53 v16 prev ARR taxi-in in Δ | 362.91 (Dec 224.76) | 236.86 (Dec 210.54) | **LB 282.09** |
| E54 v17 richer inbound | 362.87 (Dec 224.53) | 236.80 (Dec 210.29) | **LB 281.88** |
| E55 v18 2nd taxi-in + type + rwy + hat 0.28 | **362.74** (Dec 224.46) | **236.59** (Dec **210.22**) | `submit.py --lam-hat 0.28 --arr-rich` |
| E66–E70 v19 matched cap+seed-avg | 362.27 (Dec 224.22) | **235.86** (Dec 209.96) | cap900 3-seed Δ + 2-seed hat, λ=0.35 pos; `submit.py --cap900 --seeds 3 --lam-hat 0.35 --arr-rich --hat-mode pos`; submission `likable-eagle_v19.parquet`, **LB pending** |
| E71–E74 v20 LIRF structural inbound-G | matched preserved (235.86 / 209.96) | LIRF-u RMSE 3834→**3732** (Dec 2919→2625); overall ≈326→325 | E33 `T=D−G` with G trained on all LIRF + causal ARR turnaround features; replaces only the 383 LIRF-unmatched rows; submission `likable-eagle_v20.parquet`, **LB pending** |

**Current best LB = v17 (281.88).** v18, v19 and v20 are pending leaderboard tests. v19 is the best internal matched model (235.86 / 209.96). Ladder: v16 282.09 → v17 **281.88**.

**v20 (E71–E74):** the previous-departure-BLOCK/rotation oracle was falsified (corr(T, prev gap_block) ≈ 0.008) and the stand-release cap `T ≤ since_arr` fails (stand reuse). The surviving structural move enriches the E33 LIRF `D−G` gate component with same-stand ARR turnaround features (`arr_taxiin`, `arr_taxiin_2`, `since_arr`, `arr_delay`) under the robust all-LIRF training population: LIRF-unmatched SSE −5.3% (Jan+Jul) / −19.1% (Dec). Matched v19 and non-LIRF unmatched are untouched.

**260 assessment (E66–E70):** matched is at its practical floor (~236) given the 30 fields — the residual is unbiased gate-hold/clock-disagreement variance, and v19 beats every single-clock proxy on every airport. v13→v16 showed LB tracks matched ~1:1, so LB 260 implies matched ≈218–220, ~16 s below the floor. Reaching it requires the missing LIRF/unmatched gate-hold (`BLOCK`) signal, consistent with E29/E30.

**E19 local queue state:** causal (strictly `< t`, zero TAXITIME) neighbour/same-runway/delay-shock/pressure features on the E18-H residual improved Jan+Jul by only −1.34 s (Dec −4.34 s) and **did not reduce the large-positive tail** (top-1% SSE share unchanged). Core hypothesis falsified: rolling queue-state representations do not identify the >30 min tail rows.

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
