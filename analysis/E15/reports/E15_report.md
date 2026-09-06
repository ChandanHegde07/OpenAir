# E15 — Tail-aware residual objective

**Project:** OpenAir  
**Target:** `TAXITIME_SEC_mvt`  
**Experiment:** E15  
**Validation:** Jan+Jul 2025 training holdout  
**Stress check:** December 2025  
**Architecture:** frozen E14 `P_cal` + residual LightGBM; unmatched LIRF → `MVT−SCHED`  
**Data:** 12 `training_*.parquet` files only. No ranking/submission.

Script: `experiments/run_e15.py` (copy at `analysis/E15/run_e15.py`).

---

## Objective

E14 located the remaining matched error: **tail compression**. The residual model overpredicts short taxis and underpredicts long taxis. Flights >30 min are 4.5% of matched rows and 45.8% of matched SSE. Traffic, queue, finer geometry, airport intercepts, hour, WTC, type, and missingness had little leftover explanatory power.

E15 tests one question: **can a tail-aware residual training objective uncompress that tail without degrading the 10–20 min bulk?** No new feature families. Not a model sweep.

---

## Frozen (not touched)

- `P_cal` = per-airport OLS(`MVT−AOBT`, `AOBT−EOBT`, `geo_mean`), unmatched → `geo_mean` → airport mean
- Feature matrix `NUM_COLS` + `CAT_COLS` from E12
- Train / validation split (Jan+Jul holdout; December stress)
- Unmatched LIRF → `MVT−SCHED` (exact E12/E14 mask)
- Preprocessing: geometry tables on the train split, causal roll10, traffic, queue
- LightGBM capacity: 400 trees, lr 0.05, 63 leaves, min_child 80, colsample 0.8, λ=1, seed=1, early stopping 40 on the last 15% of **train** by time

Only the residual **objective / sample weights / expert mixture** change.

---

## Variants

| ID | Residual objective |
|---|---|
| baseline | Frozen E14 L2 residual LightGBM |
| E15-A | Huber. δ = 1.345 × 1.4826 × MAD(matched **train** `P_cal` residual) = **228.2 s** on Jan+Jul |
| E15-B | L2 with sample weight `1 + relu(y−1800)/1800 + relu(y−P_cal)/τ`, τ = median positive matched train residual = **115.6 s**, clip [1, 10]. Override rows keep weight 1 |
| E15-C | Two-stage, run only because A/B failed the gate: LightGBM `P(y>30 min)` + residual model trained only on `y>30 min`. Mixture `ŷ = P_cal + (1−p)·r_L2 + p·r_tail`, then the frozen LIRF override |

Huber δ and tail-weight τ use **training-split matched rows only**.

### E15-C gate

Run C only if neither A nor B, on Jan+Jul matched, does all of:

- drop >30 min RMSE by at least 10 s
- keep <20 min RMSE from rising more than 5 s
- keep matched RMSE from rising more than 3 s

**A:** tail worse (`Δ>30 = +111 s`). **B:** tail better (`Δ>30 = −168 s`) but `<20` `+94 s` and matched `+44 s`. **C was run.**

---

## Reproduction

Jan+Jul L2 residual matched RMSE **256.46** (E14 journal 256.46, delta **−0.00 s**). December overall 245.38 / matched 230.04. Proceed.

Matched N = 339,046 (Jan+Jul), 15,140 with y>30 min, 775 with y>60 min.

---

## Jan+Jul 2025 (primary)

| Variant | Overall RMSE | Matched RMSE | Matched MAE | <20 min RMSE | >30 min RMSE | >60 min RMSE | SSE share >30 min | Residual mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E14 L2 residual** | **378.28** | **256.46** | **157.83** | 174.51 | 821.37 | 2375.85 | 45.8% | −0.27 |
| E15-A Huber | 384.21 | 267.66 | 156.66 | **162.90** | 932.83 | 2774.53 | 54.2% | +17.63 |
| E15-B tail-weighted | 411.87 | 300.34 | 201.78 | 268.87 | 653.67 | 1946.48 | 21.2% | −119.93 |
| E15-C two-stage | 426.31 | 317.99 | 195.68 | 274.95 | **623.31** | **1872.68** | 17.2% | −84.38 |

Deltas vs frozen L2 (negative = better):

| Variant | Δ overall | Δ matched | Δ MAE | Δ <20 | Δ >30 | Δ >60 | Δ SSE share >30 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E15-A Huber | +5.94 | +11.21 | −1.17 | **−11.61** | +111.46 | +398.68 | +8.43 |
| E15-B tail-weighted | +33.59 | +43.88 | +43.94 | +94.36 | −167.70 | −429.37 | −24.65 |
| E15-C two-stage | +48.03 | +61.53 | +37.85 | +100.44 | **−198.06** | **−503.17** | −28.65 |

A/B vs the C gate:

| Variant | Sufficient? | Δ >30 | Δ >60 | Δ <20 | Δ matched |
|---|---|---:|---:|---:|---:|
| E15-A Huber | no | +111.46 | +398.68 | −11.61 | +11.21 |
| E15-B tail-weighted | no | −167.70 | −429.37 | +94.36 | +43.88 |

---

## December 2025 (stress)

Same pattern. Not a Jan+Jul artifact.

| Variant | Overall RMSE | Matched RMSE | Matched MAE | <20 min RMSE | >30 min RMSE | >60 min RMSE | SSE share >30 min | Residual mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E14 L2 residual** | **245.38** | **230.04** | 151.04 | 162.13 | 733.59 | 2153.91 | 39.3% | +11.87 |
| E15-A Huber | 249.31 | 234.66 | **147.79** | **150.93** | 806.69 | 2442.03 | 45.7% | +27.73 |
| E15-B tail-weighted | 278.19 | 262.48 | 181.43 | 239.35 | 553.77 | 1659.78 | 17.2% | −93.84 |
| E15-C two-stage | 302.65 | 271.65 | 173.58 | 234.78 | **519.97** | **1629.02** | 14.2% | −55.87 |

| Variant | Δ overall | Δ matched | Δ <20 | Δ >30 | Δ >60 |
|---|---:|---:|---:|---:|---:|
| E15-A Huber | +3.93 | +4.62 | −11.20 | +73.09 | +288.13 |
| E15-B tail-weighted | +32.81 | +32.44 | +77.22 | −179.83 | −494.13 |
| E15-C two-stage | +57.27 | +41.61 | +72.65 | −213.62 | −524.89 |

C also inflates December **unmatched** RMSE 886 → 1365. The mixture leaks into the unmatched path before the LIRF override.

---

## Compression by actual-taxi bin (Jan+Jul matched)

Mean residual `y − pred`. Positive = still underpredicting long taxis.

| Bin | L2 | Huber | Tail-weighted | Two-stage |
|---|---:|---:|---:|---:|
| <10m | −113 | −105 | −188 | −129 |
| 10–15m | −66 | −55 | −161 | −102 |
| 15–20m | −14 | +0 | −135 | −87 |
| 20–30m | +93 | +113 | −64 | −78 |
| 30–45m | +370 | +441 | +131 | +55 |
| 45–60m | +817 | +1043 | +479 | +456 |
| >60m | +1631 | +2226 | +1145 | +1155 |

RMSE by bin:

| Bin | L2 | Huber | Tail-weighted | Two-stage |
|---|---:|---:|---:|---:|
| <10m | 190 | **180** | 277 | 240 |
| 10–15m | 164 | **151** | 263 | 257 |
| 15–20m | 180 | **170** | 272 | 309 |
| 20–30m | **252** | 253 | 287 | 365 |
| 30–45m | 543 | 589 | 431 | **410** |
| 45–60m | 1046 | 1206 | 785 | **735** |
| >60m | 2376 | 2775 | 1946 | **1873** |

Figures: `e15_target_bin_mean_resid_janjul.png`, `e15_target_bin_rmse_janjul.png`, `e15_residual_vs_actual.png`.

---

## What each variant did

### E15-A Huber — REJECT as a tail fix

Canonical robust regression: quadratic inside ±δ, linear outside. δ = **228 s** from matched-train `P_cal` residual MAD (P90 |r| = 344 s, P99 = 848 s). Custom LightGBM objective matches the built-in Huber gradient (clip `pred−true` at ±δ, hessian 1). Early-stop on Huber loss on the train time holdout.

Huber **improves the bulk** (<20 RMSE 174.5 → 162.9; matched MAE 157.8 → 156.7) and **worsens the tail** (>30 821 → 933; >60 2376 → 2775). Mean residual goes from 0 to +18 s (systematic underprediction). >30 min SSE share rises 45.8% → 54.2%.

This is the textbook Huber trade: down-weight the observations E14 said we must fit. The 24 h bombs are not “distorting L2 away from the operational tail.” L2 was already the tail-aware loss. Huber is the wrong direction for RMSE tails.

### E15-B tail-weighted L2 — REJECT as a replacement

Same trees, higher weight on long taxi and large positive `y−P_cal`. Mean train weight 1.68, 45% of rows >1, cap 10. Unmatched LIRF override rows stay at weight 1.

B **does** move the tail: >30 RMSE 821 → 654, >60 2376 → 1946, 30–45 min mean residual +370 → +131. It pays for that by overpredicting the bulk (mean residual **−120 s**; <20 RMSE 175 → 269). Matched RMSE 256 → 300, overall 378 → 412.

The drop in >30 SSE share (46% → 21%) is mostly **bulk SSE exploding**, not the tail being solved. Mean prediction for y<10 min goes 592 → 667. The model cannot tell a true 12-minute taxi from a flight that will become a 40-minute taxi, so upweighting the tail just lifts everyone.

### E15-C two-stage — REJECT

A/B failed the gate, so C added a probability gate and a tail expert on the **same frozen features**.

- Classifier: 47,771 train positives / 1,431,762 negatives, `class_weight='balanced'`, 400 trees.
- Tail residual: 278 trees on `y>30 min` only.
- Mean `p_hat` on Jan+Jul val = **0.136** vs true rate **0.045**. Balanced training makes the gate fire far too often, so the tail expert leaks into the bulk.

C has the best tail RMSE (>30 623, >60 1873) and almost-unbiased 30–45 min residuals (+55 s vs L2 +370). It also has the **worst** matched RMSE (318) and overall RMSE (426). <20 RMSE 175 → 275. December unmatched RMSE 886 → 1365.

The tail expert *can* uncompress 30–45 min taxis when it is used. The frozen features do not say **when** to use it. That is E14’s identifiability result, not a missing loss function.

---

## Conclusion

**Winner: frozen E14 L2 residual.** Keep the current best pipeline. Do not replace L2 with Huber, tail weights, or the two-stage mixture.

| Question | Answer |
|---|---|
| Which variant wins? | **L2 residual (baseline).** Lowest overall and matched RMSE on Jan+Jul and December. |
| Did >30 min error improve? | Only under B (−168 s) and C (−198 s), which **lose** on matched RMSE. Huber **worsens** >30 by +111 s. Under the winner: **no**. |
| Did >60 min error improve? | Same: B/C yes, Huber no, winner **no**. |
| Did <20 min degrade? | Huber: no, it improved (−12 s). B: **yes** (+94 s). C: **yes** (+100 s). Winner: not degraded. |
| Does this justify another experiment? | **No further residual-objective experiment.** The loss can reallocate error between bulk and tail. It cannot identify E14’s disruption tails from the frozen features. A Huber-δ grid, a weight-grid, or a calibrated-p rerun of C would measure the same tradeoff again. |

Current best remains:

```
P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)
if unmatched and airport==LIRF:
    y_hat = MVT - SCHED
else:
    y_hat = P_cal + LightGBM_residual_L2(...)
```

Jan+Jul overall **378.28**, matched **256.46**, matched MAE **157.83**. December overall **245.38**, matched **230.04**.

---

## Artifacts

- `analysis/E15/figures/` — RMSE bars, bin RMSE, bin mean residual, SSE share, scatter, residual vs actual
- `analysis/E15/tables/e15_metrics.csv`
- `analysis/E15/tables/e15_target_bin_metrics.csv`
- `analysis/E15/tables/e15_findings.json`
- `analysis/E15/tables/e15_matched_predictions_janjul.parquet`
- `experiments/results/E15.json`
