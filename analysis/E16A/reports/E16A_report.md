# E16-A — Airport / network disruption-state features

**Project:** OpenAir  
**Target:** `TAXITIME_SEC_mvt`  
**Experiment:** E16-A  
**Validation:** Jan+Jul 2025 training holdout  
**Stress check:** December 2025  
**Data:** 12 `training_*.parquet` files only. No ranking/submission.

Script: `experiments/run_e16a.py` (copied to `analysis/E16A/run_e16a.py`).

---

## Objective

E14/E15 left a matched RMSE floor of **256.46 s**. The frozen L2 residual overpredicts short taxis and underpredicts long ones. Huber, tail-weighted L2, and a two-stage mixture all lost to L2. Remaining error was diagnosed as tail compression plus LIRF/LFPG disruption tails, not leftover traffic or geometry.

**Hypothesis:** own `AOBT−EOBT` says *this flight* pushed late. It does not say whether the *airport* is in a broader push-delay disruption, where very long taxi becomes more likely. A strictly causal airport disruption state might separate a normal late push from a systemic event.

This is one feature-family test. No loss change, no hyperparameter search, no callsign/rotation features.

---

## Frozen baseline

```
P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)
y_hat = P_cal + LightGBM_L2_residual(frozen E12 matrix)
if unmatched and airport==LIRF:
    y_hat = MVT - SCHED
```

Reproduction: Jan+Jul overall **378.28**, matched **256.46** (E14 delta **−0.00 s**), matched MAE 157.83. December overall 245.38, matched 230.04.

---

## Phase 1 — causal feature family

Prediction time for a scored departure is `MVT_TIME_UTC_mvt`.

A neighbouring departure contributes `AOBT−EOBT` only if its **AOBT ≤ scored MVT** (the push delay is then known). The scored flight is subtracted from the window. No BLOCK, no TAXITIME, no future `ARVT_3`, no other flight’s taxi/block outcome. Arrivals contribute landing-vs-schedule delay only if they have **already landed**.

| Feature | Definition |
|---|---|
| `dis_state_30m` (`airport_recent_push_delay_state`) | Mean `AOBT−EOBT` of **other** pushes in the previous 30 min |
| `dis_frac20_30m` / `dis_frac30_30m` | Fraction of those pushes with delay >20 / >30 min |
| `dis_n_dly20_60m` | Count of other pushes with delay >20 min in the previous 60 min |
| `dis_n_15/30/60m` | Count of other pushes (AOBT-indexed, not takeoff-indexed `dep_*m`) |
| `dis_med_ae_30m` / `dis_p90_ae_30m` | Event-time rolling median/p90, asof last push ≤ MVT (diagnostic; ~1-pt self contamination) |
| `dis_state_vs_hour` | `dis_state_30m` minus train-split airport×hour mean |
| `arr_delay_mean_30m` | Mean `MVT−SCHED` of arrivals already landed in the previous 30 min |

Airport×hour climatology is fit on the **training split only**, same protocol as `geo_mean`.

Model columns if Phase 2 passed: `dis_state_30m`, `dis_frac20_30m`, `dis_frac30_30m`, `dis_n_dly20_60m`, `dis_state_vs_hour`, `arr_delay_mean_30m`.

---

## Phase 2 — does it identify the E14 tail?

Jan+Jul matched, **frozen** L2 residual (disruption not yet in the model). Quantiles of `dis_state_30m`.

| Q | N | Mean y | Mean AOBT−EOBT | Mean dis | >30m rate | >60m rate | Resid mean | Resid RMSE | SSE share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 42,353 | 983 | −68 | −161 | 2.37% | 0.09% | −27.7 | 213 | 8.6% |
| Q2 | 42,200 | 959 | 171 | 111 | 2.76% | 0.08% | −17.1 | 196 | 7.2% |
| Q3 | 42,247 | 951 | 268 | 232 | 2.93% | 0.09% | −11.8 | 248 | 11.7% |
| Q4 | 42,307 | 944 | 344 | 334 | 2.96% | 0.11% | −10.2 | 218 | 9.0% |
| Q5 | 42,208 | 955 | 428 | 435 | 3.12% | 0.14% | −4.9 | 227 | 9.8% |
| Q6 | 42,302 | 983 | 528 | 553 | 3.80% | 0.16% | −3.4 | 234 | 10.4% |
| Q7 | 42,260 | 1,047 | 653 | 713 | 5.39% | 0.27% | +13.7 | 274 | 14.2% |
| Q8 | 42,227 | 1,219 | 993 | 1,124 | **12.34%** | **0.88%** | **+59.6** | **389** | **28.6%** |

Q8 vs Q1: >30 min rate **2.37% → 12.34%** (ratio 5.22); residual RMSE 213 → 389 (gap **+175 s**); mean residual −28 → **+60** (the frozen model underpredicts high-disruption flights). Q8 is 12.5% of matched rows and 28.6% of matched SSE.

Correlations (matched Jan+Jul):

| Pair | corr |
|---|---:|
| `dis_state_30m` vs own `AOBT−EOBT` | 0.453 |
| `dis_state_30m` vs y | 0.178 |
| `dis_state_30m` vs frozen residual | **0.114** |
| `dis_state_30m` vs \|residual\| | 0.160 |
| own `AOBT−EOBT` vs frozen residual | 0.022 |
| `dis_frac20_30m` vs residual | 0.114 |
| `arr_delay_mean_30m` vs residual | 0.007 |

Own `AOBT−EOBT` is already in `P_cal` and the tree, so it is nearly orthogonal to the residual. The airport state is **not**: corr 0.45 with own lateness, but still 0.11 with the leftover residual.

### Controlled for own AOBT−EOBT

If disruption only restates this flight’s lateness, the Q8 vs Q1 tail gap should vanish inside an `AOBT−EOBT` quartile. It does not.

| Own AOBT−EOBT Q | N low dis | N high dis | >30m low | >30m high | ratio | Resid RMSE low | Resid RMSE high | Resid mean low | Resid mean high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AE1 (earliest) | 25,656 | 4,007 | 0.92% | **7.46%** | **8.15** | 180 | 319 | −14 | +23 |
| AE2 | 8,883 | 5,330 | 2.36% | 3.83% | 1.62 | 200 | 217 | −24 | +19 |
| AE3 | 4,943 | 9,883 | 4.05% | 6.45% | 1.59 | 250 | 287 | −60 | +50 |
| AE4 (latest) | 2,871 | 23,007 | 12.43% | **17.70%** | 1.42 | 389 | 462 | −105 | +80 |

Two flights with similar own push delay do **not** have the same tail risk. An on-time push (AE1) at a disrupted airport has a 7.5% chance of y>30 min vs 0.9% in a quiet airport. A late push (AE4) goes 12% → 18%. The frozen residual also flips from negative (overpredict) in low disruption to positive (underpredict) in high disruption, in every AOBT−EOBT quartile.

**Phase 2 decision:** signal is real and not just `AOBT−EOBT`. Train Phase 3.

Figures: `e16a_gt30_rate_by_dis_q.png`, `e16a_resid_rmse_by_dis_q.png`, `e16a_twoway_gt30.png`, `e16a_twoway_resid_mean.png`.

---

## Phase 3 — same L2 residual, disruption columns added

Everything else frozen: `P_cal`, split, preprocessing, LIRF unmatched override, LightGBM capacity (400 / 63 leaves / ES 40 / seed 1).

### Jan+Jul 2025 (primary)

| Metric | Baseline | E16-A | Δ |
|---|---:|---:|---:|
| Overall RMSE | 378.28 | **376.04** | **−2.23** |
| Matched RMSE | 256.46 | **253.82** | **−2.64** |
| Matched MAE | 157.83 | 157.46 | −0.38 |
| <20 min RMSE | 174.51 | 176.00 | +1.49 |
| >30 min RMSE | 821.37 | **806.79** | **−14.58** |
| >60 min RMSE | 2375.85 | 2382.43 | +6.58 |
| SSE share >30 min | 45.80% | 45.12% | −0.69 pp |
| Mean residual | −0.27 | −5.83 | −5.56 |

### December 2025 (stress)

| Metric | Baseline | E16-A | Δ |
|---|---:|---:|---:|
| Overall RMSE | 245.38 | **241.27** | **−4.11** |
| Matched RMSE | 230.04 | **226.95** | **−3.09** |
| Matched MAE | 151.04 | 148.10 | −2.93 |
| <20 min RMSE | 162.13 | 156.13 | −5.99 |
| >30 min RMSE | 733.59 | 726.70 | −6.90 |
| >60 min RMSE | 2153.91 | 2100.69 | −53.21 |
| SSE share >30 min | 39.34% | 39.66% | +0.32 pp |
| Mean residual | +11.87 | +21.11 | +9.24 |

December agrees on matched/overall RMSE. The >60 min improvement is **not** stable (Jan+Jul slightly worse, December better). Do not claim a >60 min fix.

### What the model did to the disruption tail

Frozen vs E16-A residual **mean** by `dis_state_30m` quantile (Jan+Jul matched):

| Q | Frozen mean resid | E16-A mean resid | Frozen RMSE | E16-A RMSE |
|---|---:|---:|---:|---:|
| Q1 | −27.7 | −13.3 | 213 | 212 |
| Q7 | +13.7 | −7.2 | 274 | 273 |
| Q8 | **+59.6** | **−3.5** | 389 | **373** |

The hypothesis mechanism is visible: high-disruption flights were underpredicted by +60 s; after adding airport state the Q8 mean residual is ~0. Residual RMSE in Q8 falls 389 → 373. That is a real uncompression of the *mean*, not a large RMSE win on the 24 h bombs.

### Feature gain (Jan+Jul residual model, 30 features)

| Feature | Gain | Rank |
|---|---:|---:|
| `dis_state_30m` | 6.74e10 | **7** (just below `ADES`, above `mvt_aobt`) |
| `arr_delay_mean_30m` | 4.37e10 | 11 |
| `dis_frac20_30m` | 1.75e10 | 15 |
| `dis_state_vs_hour` | 1.16e10 | 18 |
| `dis_frac30_30m` | 4.28e9 | 23 |
| `dis_n_dly20_60m` | 3.36e9 | 24 |

The useful core is **`dis_state_30m`**. `dis_frac20_30m` is a weaker intensity companion. `dis_frac30_30m` and delayed counts add little. `arr_delay_mean_30m` has tree gain but **Phase 2 residual corr = 0.007** — it is not the tail identifier; treat it as a secondary, possibly bulk/season, covariate.

Top frozen features remain `mvt_sched`, `type_null`, `aobt_eobt`, runway, stand.

### Airport matched RMSE (Jan+Jul)

| Airport | N | Baseline | E16-A | Δ |
|---|---:|---:|---:|---:|
| **LIRF** | 26,131 | 463.03 | **451.86** | **−11.17** |
| LFPG | 38,732 | 283.48 | 283.88 | +0.40 |
| LTFM | 45,882 | 268.55 | 264.72 | −3.83 |
| EGLL | 39,744 | 258.55 | 258.55 | −0.00 |
| LEBL | 28,490 | 233.32 | 228.90 | −4.42 |
| EDDF | 36,231 | 205.13 | 202.05 | −3.08 |
| LSZH | 21,822 | 202.18 | 200.68 | −1.51 |
| EDDM | 26,854 | 195.85 | 195.68 | −0.18 |
| LEMD | 35,019 | 188.02 | 186.22 | −1.80 |
| EHAM | 40,141 | 185.62 | 188.15 | +2.53 |

LIRF matched is the main airport win. **LFPG does not move.** December: LTFM −14.6, LEBL −13.1, LIRF −6.9, LFPG still ~0.

---

## Answers

**Does airport/network disruption state contain information not already captured by AOBT−EOBT?**  
**Yes.** corr(state, own AOBT−EOBT) = 0.45, but corr(state, frozen residual) = 0.11 while corr(own AOBT−EOBT, residual) = 0.02. Inside every own-lateness quartile, high disruption still raises the >30 min rate (ratios 8.1 / 1.6 / 1.6 / 1.4) and flips residual mean from negative to positive.

**Does it reduce matched tail error?**  
**Partially, and only the operational >30 min tail.** Jan+Jul >30 min RMSE 821 → 807 (−15 s). >60 min 2376 → 2382 (**not improved**). Q8 mean residual +60 → −3.5: the model stops systematically underpredicting high-disruption flights, but the 1–24 h bombs remain. December >30 −7 s, >60 −53 s (directionally helpful, not a Jan+Jul replica).

**Does it improve overall RMSE?**  
**Yes, small and consistent.** Jan+Jul 378.28 → 376.04 (−2.23); matched 256.46 → 253.82 (−2.64). December 245.38 → 241.27 (−4.11); matched 230.04 → 226.95 (−3.09). First matched-RMSE move in the right direction since E9/E12.

**Does it damage the <20 min bulk?**  
**Not materially.** Jan+Jul +1.5 s (174.5 → 176.0). December −6.0 s (improves). Not the E15-B/C tradeoff.

**Which specific disruption feature is actually useful?**  
**`dis_state_30m`** — mean of other flights’ AOBT−EOBT in the last 30 minutes. Rank 7. `dis_frac20_30m` is a weaker second. Counts and >30 min fraction are near-noise. Arrival delay has gain but was not the Phase 2 tail signal.

**Is the evidence strong enough to keep this feature family?**  
**Yes, keep `dis_state_30m` (+ `dis_frac20_30m`).** It is causal, it is not a restatement of own AOBT−EOBT, it improves matched and overall RMSE on both splits, and it unbiases the high-disruption residual mean.  

Do **not** oversell it as a solution to E14. Matched RMSE moved ~1%. >60 min on the primary split did not improve. LFPG did not improve. Do not open a disruption-window sweep (15 vs 45 vs 90 min, p90 vs mean, runway-level, etc.) unless a new hypothesis says why the 30 min airport mean is the wrong aggregation.

---

## Decision

- **KEEP** `dis_state_30m` (`airport_recent_push_delay_state`) and `dis_frac20_30m` in the residual LightGBM.
- **DEFER / optional** `arr_delay_mean_30m` (gain without Phase 2 residual correlation).
- **REJECT as incremental** `dis_frac30_30m`, `dis_n_dly20_60m` (bottom gain).
- Current-best pipeline becomes E9 L2 residual + E11 LIRF override + these disruption columns. `P_cal` unchanged.
- Next experiment is **not** more disruption variants, **not** another loss function.

---

## Artifacts

- `analysis/E16A/figures/`
- `analysis/E16A/tables/e16a_metrics.csv`
- `analysis/E16A/tables/e16a_disruption_quantile_janjul.csv`
- `analysis/E16A/tables/e16a_twoway_ae_dis_janjul.csv`
- `analysis/E16A/tables/e16a_within_ae_janjul.csv`
- `analysis/E16A/tables/e16a_findings.json`
- `experiments/results/E16A.json`
