# E14 — Matched residual diagnosis

**Project:** OpenAir  
**Target:** `TAXITIME_SEC_mvt`  
**Experiment:** E14  
**Validation:** Jan+Jul 2025 training holdout  
**Stress check:** December 2025  
**Architecture:** per-airport OLS `P_cal` + LightGBM residual; unmatched LIRF/type_null → `MVT−SCHED`  
**Data:** 12 `training_*.parquet` files only. No ranking/submission.

Script: `analysis/E14/run_e14.py` (copy of `experiments/run_e14.py`).

---

## Objective

Do not add features. Locate the remaining **~256 s matched RMSE** so E15 has one evidence-based target.

---

## Model reproduced

Residual LightGBM (seed=1) on `y − P_cal`, with

`P_cal` = per-airport OLS(`MVT−AOBT`, `AOBT−EOBT`, `geo_mean`).

| Split | Overall RMSE | Matched RMSE | Matched MAE |
|---|---:|---:|---:|
| Jan+Jul (this run) | 378.28 | **256.46** | 157.83 |
| E9/E12 journal | 378.28 | 256.46 | 165.09 (overall MAE) |
| December (this run) | 245.38 | 230.04 | — |

Reproduction matched RMSE delta: **0.00 s**. Proceed.

Matched N = 339,046 (Jan 152,271 / Jul 186,775).

Row-level outputs: `tables/e14_matched_predictions.parquet`.

---

## Overall residual diagnostics

`tables/e14_overall_metrics.csv`

| Split | N | Mean resid | Median resid | Std | MAE | RMSE | P5 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan+Jul | 339,046 | −0.3 | −17.1 | 256.5 | 157.8 | 256.5 | −314 | +354 | +778 |
| Jan | 152,271 | −3.1 | −15.0 | 228.3 | 149.9 | 228.3 | −308 | +335 | +687 |
| Jul | 186,775 | +2.1 | −18.9 | 277.3 | 164.3 | 277.3 | −319 | +369 | +863 |

**Not a global bias problem.** Mean residual ≈ 0. July is harder than January (+49 s RMSE). Remaining error is spread and tail.

Figures: `e14_residual_distribution.png`.

---

## Airport

`tables/e14_airport_metrics.csv` · `figures/e14_airport_rmse.png`

| Airport | N | Mean resid | MAE | RMSE | SSE share |
|---|---:|---:|---:|---:|---:|
| LIRF | 26,131 | +4.6 | 248 | **463** | **25.1%** |
| LTFM | 45,882 | +2.1 | 184 | 269 | 14.8% |
| LFPG | 38,732 | −3.0 | 174 | 283 | 14.0% |
| EGLL | 39,744 | −20.8 | 175 | 259 | 11.9% |
| others | | ~0 | 119–150 | 186–233 | 4–7% each |

LIRF matched is **unbiased** (mean residual +5 s) but **high-variance**. η² of residual by airport = **0.002**. The airport gap is noise/tails, not a missing intercept.

One LIRF flight (stand 234 / rwy 25, July, actual **87,002 s**) produces ~3.7% of **all** matched SSE by itself.

---

## Runway / stand / stand×runway

- `e14_runway_metrics.csv` (n≥100)
- `e14_stand_metrics.csv` (n≥50)
- `e14_stand_runway_metrics.csv` (n≥100)
- `e14_stand_runway_rmse.png`

η² of residual by stand×runway = **0.040**. After `geo_mean` plus stand as a LightGBM categorical, leftover geometry is modest.

LIRF 25 accounts for **21.9%** of matched SSE (it is also LIRF’s main runway). LIRF 234/25 RMSE 2,030 is the 87k-second outlier’s group.

**Not the main E15 lever** unless we are specifically hunting LIRF data-quality tails.

---

## Hour

`e14_hour_metrics.csv` · `e14_airport_hour_metrics.csv` · `e14_hour_rmse.png`

RMSE is slightly higher at night (hour 0: 373 s, n=920; hour 20: 309 s) than midday (~230–270 s). Mean residuals stay near 0. η² hour = **0.0004**; airport×hour = **0.006**.

No strong unused time-of-day intercept.

---

## WTC / aircraft type

`e14_wtc_metrics.csv` · `e14_aircraft_metrics.csv` (n≥200)

| WTC | N | RMSE | Mean resid |
|---|---:|---:|---:|
| L | 1,587 | 233 | +0.0 |
| M | 264,197 | 248 | +0.9 |
| H | 70,811 | 285 | −4.3 |
| J | 2,444 | 307 | −10.8 |

η² WTC = **0.0001**; aircraft type = **0.002**; operator = **0.004**. Heavies are a bit noisier, not a missing bias term.

---

## Target magnitude (main finding)

`e14_target_bin_metrics.csv` · `e14_target_bin_rmse.png` · `e14_actual_vs_predicted.png` · `e14_residual_vs_actual.png`

| Bin | N | Mean actual | Mean pred | Mean resid | MAE | RMSE | SSE share |
|---|---:|---:|---:|---:|---:|---:|---:|
| <10 min | 42,436 | 479 | 592 | **−113** | 133 | 190 | 6.9% |
| 10–15 min | 117,120 | 758 | 824 | **−66** | 118 | 164 | 14.2% |
| 15–20 min | 96,809 | 1037 | 1051 | −14 | 132 | 180 | 14.0% |
| 20–30 min | 67,541 | 1414 | 1321 | **+93** | 191 | 252 | 19.2% |
| 30–45 min | 12,354 | 2095 | 1724 | **+370** | 432 | 543 | 16.3% |
| 45–60 min | 2,011 | 3033 | 2216 | **+817** | 872 | 1046 | 9.9% |
| >60 min | 775 | 4793 | 3162 | **+1631** | 1734 | 2376 | **19.6%** |

The residual model **over-predicts short taxis and under-predicts long ones**. 775 flights >60 min (0.23% of matched) hold **19.6% of SSE**. 15,140 flights >30 min (4.5%) hold **45.8% of SSE**.

This is shrinkage from squared-error training, on top of an OLS `P_cal` whose AOBT slope is already < 1.

---

## Clocks

`e14_clock_residual_metrics.csv` · quantile tables for `MVT−AOBT` and `AOBT−EOBT`

| Variable | corr(resid) | corr(\|err\|) |
|---|---:|---:|
| MVT−AOBT | 0.061 | 0.199 |
| AOBT−EOBT | 0.022 | 0.268 |
| MVT−EOBT | 0.055 | **0.363** |
| MVT−SCHED | 0.028 | 0.204 |
| P_cal | 0.043 | 0.341 |

Mean residual vs `MVT−AOBT` quantile is nearly flat (clocks are calibrated). **Absolute error** grows in the top quantile of `AOBT−EOBT`: Q8 RMSE **452 s** and **36% of matched SSE**. Large plan-vs-actual off-block disagreement marks hard, disrupted flights the tree still shrinks.

A new linear clock term will not help. Any clock work in E15 must be **tail/disruption-aware**.

---

## Traffic / queue

`e14_traffic_residual_metrics.csv` · `e14_queue_quantile_residual.csv` · figures `e14_queue_residual.png`, `e14_traffic_residual.png`

| Variable | corr(resid) | corr(\|err\|) |
|---|---:|---:|
| dep 1/5/10/15/30/60 m | −0.01 to +0.02 | ≤0.04 |
| same-runway 15/60 m | −0.01 to −0.02 | 0.03–0.04 |
| queue / queue_rwy | ~0 | 0.06–0.07 |

Queue Q1 vs Q8: mean residual **0.0 vs +1.8 s**. The model already matches average congestion. RMSE only slightly higher in high queue (281 vs 246). **Confirms E4/E5:** do not add linear traffic/queue.

---

## Worst 50 matched errors

`e14_worst_errors.csv`

Almost all are **July**, actual taxi 1.5–24 h, prediction ~15–90 min (except the 87k-s LIRF case, which the model partly followed via a huge `MVT−AOBT`). Recurring pattern: **LIRF rwy 25** and **LFPG** long-haul heavies, `AOBT−EOBT` often 20–100 min (disrupted push), queue not extreme.

These are operational disruption / possible data-quality tails, not a missing 15-min traffic count.

---

## Error concentration

`e14_error_concentration.csv` · `e14_sse_concentration.png`

| Slice | N | SSE share |
|---|---:|---:|
| worst 0.1% | 339 | 19.4% |
| worst 1% | 3,390 | 40.9% |
| worst 5% | 16,952 | 62.2% |
| worst 10% | 33,905 | 73.3% |
| y > 30 min | 15,140 | 45.8% |
| y > 60 min | 775 | 19.6% |
| y ≤ 20 min | 256,365 | 35.0% |

Matched RMSE is **still a tail problem**, even after residual LightGBM.

---

## Missingness

`e14_missingness_metrics.csv`

Matched ⇒ AOBT/EOBT/IOBT/LOBT/type/operator/market are present. WTC missing n=7. Missingness is not the matched-error story.

---

## Residual structure (η² of residual, not causal)

`e14_residual_decomposition.csv`

| Factor | η² / share |
|---|---:|
| Airport | 0.002 |
| Stand × runway | 0.040 |
| Hour | 0.000 |
| Airport × hour | 0.006 |
| WTC | 0.000 |
| Aircraft type | 0.002 |
| Operator | 0.004 |
| SSE share y>30 min | **0.458** |
| SSE share y>60 min | **0.196** |

Observable group means explain little of what is left. The tail of **actual taxi** explains a large share of SSE.

---

## Evidence-supported failure modes

1. **Tail compression** — systematic under-prediction that grows with actual taxi (mean residual −113 s below 10 min, +1631 s above 60 min).
2. **Disruption / clock-disagreement tails** — top quantile of `AOBT−EOBT` holds 36% of SSE; worst errors have large push delay and multi-hour taxi, especially LIRF 25 and LFPG.
3. **July harder than January** (+49 s matched RMSE) — more of those tails, not a calendar dummy.
4. **LIRF matched variance** — 25% of SSE, unbiased, includes at least one 24 h data-quality point.

## What does not appear important for E15

- Linear traffic or queue
- Another stand×runway mean table
- Additive airport/WTC intercepts
- Hour-of-day as a new main effect
- NM missingness among matched
- LIRF unmatched gating (E13)

## Likely irreducible component

Even a perfect mean predictor in the 10–20 min bulk leaves a ~150–180 s RMSE there (see those bins). That looks like operational noise at a scale the data do not resolve (ATC holds, unrecorded crossings). E15 should not chase that bulk RMSE first.

---

## Recommended E15 (one experiment)

**Observed failure:** matched residuals increase with actual TAXITIME; >30 min bins and high `AOBT−EOBT` flights are systematically under-predicted and dominate SSE.

**Hypothesis:** squared-error LightGBM on `y − P_cal` (and the already-shrunk OLS) compresses the tail. A **tail-aware residual objective** will lift long-taxi predictions without needing new feature families.

**Why the current model cannot capture it:** MSE + early stopping fit the dense 12–20 min mass. `P_cal` AOBT slope is already < 1. The tree residual is also shrunk.

**Specific change (features frozen):** same inputs as E9. Change only residual training, e.g. compare:

1. `log1p(y) − log1p(P_cal)` residual, invert
2. Huber / quantile (e.g. 0.6–0.7) loss
3. Two-stage: classifier P(y>30 min) + residual experts

**Metrics:** Jan+Jul overall, matched, MAE, **RMSE on y>30 min and y>60 min**, and RMSE on y<20 min (must not inflate). December stress check. Training files only.

**Expected improvement:** matched RMSE falls if tail SSE is the bottleneck; if tail RMSE barely moves, remaining error is closer to irreducible / missing ops data.

Do not turn E14 into E15 in this folder. E15 is a new experiment ID.
