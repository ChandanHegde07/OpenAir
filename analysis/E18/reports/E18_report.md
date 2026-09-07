# E18 — Non-LIRF unmatched specialist

**Project:** OpenAir  
**Target:** `TAXITIME_SEC_mvt`  
**Experiment:** E18  
**Validation:** Jan+Jul 2025 training holdout  
**Stress check:** December 2025  
**Data:** 12 `training_*.parquet` files only. No ranking/submission.

Script: `experiments/run_e18_unmatched_specialist.py` (copied to `analysis/E18/run_e18.py`).

---

## Objective

After the LIRF unmatched `MVT−SCHED` override, unmatched RMSE stays ~2240.
E12 showed residual LightGBM slightly *hurts* that slice vs E3+override
(2241 vs 2228). Non-LIRF unmatched (~25% of holdout SSE) are scored with
matched `geo_mean` plus a tree trained on 99% matched rows, including the
LIRF-override rows whose residuals are then discarded.

**Hypothesis:** a dedicated non-LIRF unmatched head — even a train airport
unmatched mean — moves **overall** RMSE.

Matched path and LIRF override stay frozen for A/B. Hygiene (H) is the
one variant allowed to refit the residual tree.

---

## Frozen baseline

```
P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)
y_hat = P_cal + LGB_L2_residual(E12 matrix + E16-A disruption cols)
if unmatched and airport==LIRF: y_hat = MVT - SCHED
```

Reproduction: Jan+Jul overall **376.04**, matched **253.82** (E16-A journal
376.04 / 253.82, delta **0.00**). December **241.27 / 226.95**. Exact.

n unmatched = 5,373 (LIRF 397, non-LIRF 4,976). Non-LIRF unmatched RMSE
**1579.38** (~25% of holdout SSE). LIRF unmatched RMSE **6032.70** (~30%).

---

## Variants

| ID | What changes | Matched path |
|---|---|---|
| E18-0 | E16-A reproduction | frozen |
| E18-A-geo | non-LIRF unmatched → `P_cal` (drop residual) | frozen |
| E18-A-mean | non-LIRF unmatched → train airport-unmatched mean | frozen |
| E18-A-med | non-LIRF unmatched → train airport-unmatched median | frozen |
| E18-H | `geo_mean` fit on matched-only; LIRF-override rows dropped from residual train | refit |
| E18-B | non-LIRF unmatched → unmatched-mean + small L2 residual tree on that slice only | frozen |

Specialist features (ranking-safe, present without AOBT): `mvt_sched`,
`geo_mean`, roll10 `MVT−AOBT`, hour/dow/month, `dis_state_30m`,
`dis_frac20_30m`, dep/queue counts, airport/stand/runway/type/ADES/prefix.

---

## Jan+Jul 2025 (primary)

| Variant | Overall | Matched | Unmatched | Non-LIRF unmatched | LIRF unmatched | Matched MAE |
|---|---:|---:|---:|---:|---:|---:|
| E18-0 | **376.04** | **253.82** | 2235.88 | 1579.38 | 6032.70 | 157.46 |
| E18-A-geo | 375.29 | 253.82 | 2227.72 | 1566.87 | 6032.70 | 157.46 |
| E18-A-mean | 376.22 | 253.82 | 2237.76 | 1582.25 | 6032.70 | 157.46 |
| E18-A-med | 376.58 | 253.82 | 2241.63 | 1588.15 | 6032.70 | 157.46 |
| **E18-H** | **372.36** | **250.98** | **2216.59** | **1549.74** | 6032.70 | **155.03** |
| E18-B | 374.45 | 253.82 | 2218.62 | 1552.88 | 6032.70 | 157.46 |

Deltas vs E18-0 (negative = better):

| Variant | Δ overall | Δ matched | Δ unmatched | Δ non-LIRF unmatched |
|---|---:|---:|---:|---:|
| E18-A-geo | −0.76 | 0.00 | −8.17 | −12.51 |
| E18-A-mean | +0.17 | 0.00 | +1.88 | +2.87 |
| E18-A-med | +0.53 | 0.00 | +5.75 | +8.77 |
| **E18-H** | **−3.68** | **−2.84** | **−19.29** | **−29.64** |
| E18-B | −1.60 | 0.00 | −17.26 | −26.50 |

---

## December 2025 (stress)

| Variant | Overall | Matched | Unmatched | Non-LIRF unmatched | LIRF unmatched | Matched MAE |
|---|---:|---:|---:|---:|---:|---:|
| E18-0 | **241.27** | **226.95** | 851.64 | 573.21 | 2786.14 | 148.10 |
| E18-A-geo | 242.67 | 226.95 | 890.80 | 632.97 | 2786.14 | 148.10 |
| E18-A-mean | 244.00 | 226.95 | 926.37 | 684.79 | 2786.14 | 148.10 |
| E18-A-med | 244.05 | 226.95 | 927.69 | 686.67 | 2786.14 | 148.10 |
| **E18-H** | **238.01** | **223.95** | **838.40** | **552.21** | 2786.14 | **145.83** |
| E18-B | 241.80 | 226.95 | 866.51 | 596.28 | 2786.14 | 148.10 |

| Variant | Δ overall | Δ matched | Δ unmatched | Δ non-LIRF unmatched |
|---|---:|---:|---:|---:|
| E18-A-geo | +1.40 | 0.00 | +39.16 | +59.76 |
| E18-A-mean | +2.73 | 0.00 | +74.73 | +111.58 |
| E18-A-med | +2.78 | 0.00 | +76.05 | +113.47 |
| **E18-H** | **−3.26** | **−3.00** | **−13.24** | **−21.00** |
| E18-B | +0.53 | 0.00 | +14.87 | +23.07 |

December is the kill rule. A-geo, A-mean, A-med, and B all raise December overall.
Only H improves both splits.

---

## Why the simple unmatched constants fail

Full-year EHAM unmatched median is 538 vs matched 744 (E11). That fact does
**not** transfer to the ranking analogue as a train-mean splice.

Jan+Jul val EHAM unmatched: n=808, **mean y = 742**, median 659. Train
airport-unmatched mean predicts **565**. RMSE 584 → **748**. The holdout
EHAM unmatched population is not the full-year short one.

December EHAM unmatched mean y = 637 (closer to the full-year story), but
A-mean still hurts (268 → 507) because other airports' tails dominate:
EDDM unmatched mean y = **1687** in December vs ~1006 in Jan+Jul. An
airport unmatched mean estimated on the rest of the year is not a stable
prior.

---

## Unmatched by airport (Jan+Jul)

| Airport | n | mean y | RMSE E18-0 | RMSE A-geo | RMSE A-mean | RMSE H | RMSE B |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 599 | 1005 | 377 | **339** | 420 | 349 | 375 |
| EDDM | 237 | 1006 | 633 | 692 | 698 | **614** | 642 |
| EGLL | 466 | 1705 | 1156 | 1201 | 1192 | 1139 | **1109** |
| EHAM | 808 | 742 | 584 | 636 | 748 | **566** | 573 |
| LEBL | 496 | 1056 | 382 | **308** | 374 | 339 | 353 |
| LEMD | 309 | 1150 | 267 | 256 | 338 | **249** | 251 |
| LFPG | 880 | 1295 | 3445 | 3452 | 3447 | **3439** | 3445 |
| LIRF | 397 | 6272 | 6033 | 6033 | 6033 | 6033 | 6033 |
| LSZH | 524 | 633 | 1017 | **640** | 703 | 649 | 632 |
| LTFM | 657 | 1146 | 624 | 640 | 675 | **578** | 607 |

LSZH is the smoking gun for “the residual tree hurts unmatched”: E18-0
RMSE 1017, mean pred 742 vs mean y 633. Dropping the residual (A-geo)
cuts RMSE to 640. H and B recover most of that without wrecking EHAM.

LFPG unmatched RMSE ~3444 on all variants. A handful of bombs; no head
moved this slice. LIRF unmatched is frozen at 6032.70 by construction.

---

## E18-B specialist

- train non-LIRF unmatched n = 16,006 (Jan+Jul split) / 19,421 (Dec split)
- trees = 113 / 36
- top gain (Jan+Jul): `RUNWAY_mvt`, `geo_mean`, `mvt_sched`, `ADES_mvt`,
  roll10 `MVT−AOBT`, `dis_state_30m`, stand, `flt_prefix`

Jan+Jul overall −1.60 s looks real (LSZH, EGLL, LEBL). December overall
+0.53 s, non-LIRF unmatched 573 → 596. The small tree overfits the
Jan+Jul unmatched mix (especially EDDM, whose December unmatched mean
jumps to 1687 s). **REJECT as a production head.**

---

## E18-H mechanism

Two changes, bundled:

1. Fit `geo_mean` on **matched** train rows only, so unmatched LIRF
   extremes (median 3922 s) do not inflate stand×runway means used as
   `P_cal` fallback.
2. Drop LIRF-override rows from residual training (1,091 on the Jan+Jul
   train side, 1,400 on the Dec train side). Those rows have
   `y − geo_mean` of many thousands of seconds, then get overwritten at
   predict time. They were contaminating `unmatched=1` leaves and, via
   shared splits, the matched model.

H improves **matched** RMSE on both splits (−2.84 / −3.00 s), more than
the unmatched-only splices. That is the LIRF-residual contamination
story, not just an unmatched offset.

This experiment did **not** split (1) vs (2). Treat H as a bundle until
an ablation says otherwise.

---

## Decision

**Verdict: ACCEPTED (E18-H only).**

- **KEEP E18-H.** Matched-only `geo_mean` + drop LIRF-override rows from
  residual training. Jan+Jul overall 376.04 → **372.36**, matched 253.82 →
  **250.98**; December 241.27 → **238.01**, matched 226.95 → **223.95**.
- **REJECT** airport unmatched mean/median as a splice. The E11 “EHAM is
  shorter” fact is real on the full year and false as a train-mean prior
  on Jan+Jul.
- **REJECT** the unmatched specialist head. Helps Jan+Jul, fails December.
- **REJECT** dropping the residual on unmatched (A-geo) as a standalone
  rule. Helps LSZH, hurts December unmatched.
- LIRF override stays `unmatched and airport==LIRF` (code). Do not apply
  `MVT−SCHED` to non-LIRF unmatched.
- LIRF unmatched RMSE is unchanged (6032.70). Next unmatched lever is E19
  (neighbor delay decomposition), not another constant.

Do not stack E17-A1 airport memory on H without a new experiment.

Artifacts: `analysis/E18/`.
