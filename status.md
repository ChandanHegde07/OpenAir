# OpenAir Taxi-Out Research Journal

Permanent research memory for PRC 2026 taxi-out prediction.  
Do not delete prior conclusions. If a conclusion changes, record: OLD → EVIDENCE → NEW.

---

## Project objective

Predict **airport-reported taxi-out time** (`TAXITIME_SEC_mvt`, seconds) for departures at 10 European airports.

Primary research goals:

1. Engineer ranking-safe features.
2. Run controlled, leak-checked experiments.
3. Understand why taxi-out behaves as it does.
4. Build a strong predictor **and** a cumulative explanation of what works.

This journal is not a leaderboard tracker. Submission files are out of scope until research justifies them.

---

## Dataset / column knowledge

### Training corpus (the ONLY data used for research fits)

Identified by filename prefix `training_*.parquet` under `data/`. Twelve monthly files, calendar year 2025:

| File | Rows |
|---|---:|
| `training_2025-01-01_2025-02-01.parquet` | 307,257 |
| `training_2025-02-01_2025-03-01.parquet` | 287,411 |
| `training_2025-03-01_2025-04-01.parquet` | 328,622 |
| `training_2025-04-01_2025-05-01.parquet` | 350,288 |
| `training_2025-05-01_2025-06-01.parquet` | 370,105 |
| `training_2025-06-01_2025-07-01.parquet` | 365,986 |
| `training_2025-07-01_2025-08-01.parquet` | 381,161 |
| `training_2025-08-01_2025-09-01.parquet` | 382,047 |
| `training_2025-09-01_2025-10-01.parquet` | 367,679 |
| `training_2025-10-01_2025-11-01.parquet` | 371,163 |
| `training_2025-11-01_2025-12-01.parquet` | 324,530 |
| `training_2025-12-01_2026-01-01.parquet` | 331,548 |
| **Total** | **4,167,797** (2,085,047 DEP / 2,082,750 ARR), 30 columns |

Jan+Jul **validation is a holdout of these training months**, not an external file.

### Non-training files (ignored in the current research phase)

| File | Rows | Role |
|---|---:|---|
| `ranking.parquet` | 689,534 | Downstream evaluation only. Must not enter training or feature fits. |
| `submitting.parquet` | 344,841 | Submission template only. |

---

## Data usage policy (binding)

**Train, fit, calibrate, aggregate, encode, and select using `training_*.parquet` only.**

Do **not** train, fit, calibrate, aggregate, or derive features from:

- `ranking.parquet`
- `submitting.parquet`
- any file whose `TAXITIME` is not a training target

Forbidden uses of those files include: group means, rolling statistics, historical averages, target encodings, clock calibration, stand/runway/airport/traffic statistics, normalization, feature selection, and hyperparameters.

`experiments/common.py` refuses to load any parquet whose name does not start with `training_`.

**Audit of E0–E12:** OLS, LightGBM, geometry tables, rolling clocks, traffic/queue, unmatched medians, and all reported RMSE/MAE used **training files and training holdouts only**. The one exception was E11’s *descriptive* peek at ranking unmatched counts; that did not enter any fitted parameter. Ranking reads have been removed from experiment code. Ranking-derived counts (e.g. “383 ranking LIRF unmatched”) must not be used as training evidence. The LIRF `type_null` fact used in the model is from **training** (1,487 / 1,488 unmatched LIRF).

`*_mvt` = airport movement. `*_flt` = Network Manager flight list (left-joined).  
No coordinates, taxiways, trajectories, or weather in the provided files.

Airports: LTFM, EHAM, LFPG, EGLL, EDDF, LEMD, LEBL, EDDM, LIRF, LSZH.

See `analysis/DISCOVERY_REPORT.md` for the full column dictionary.

---

## Target definition

For `PHASE_mvt == "DEP"`:

```
TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt
```

Verified 100% exact on training DEP.

Unit: seconds. Mean 991, median 912, std 546, P99 2339, max 131167.  
Right-skewed, heavy-tailed. RMSE is dominated by the tail and by unmatched/LIRF cases.

Arrivals are not scored. Their taxi-in and timestamps are context.

---

## Prediction-time availability

Ranking blanks **only** `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` for departures.

Present for ranking DEP: takeoff (`MVT_TIME`), schedule, stand, runway, aircraft type, and NM fields including `AOBT_3_flt` for 98.47% of rows. Ranking ARR is complete, including taxi-in.

**Information set:** anything non-null on the scored row, plus other flights’ fields except other DEP `BLOCK_TIME` / `TAXITIME`.

This is post-operations reconstruction, not a pushback-time forecast.

---

## Discoveries

### A. Discovery-phase findings (pre E0; re-verified where noted)

**AOBT proxy** `P = MVT − AOBT` (matched rows; NaNs dropped):

| Split | Raw proxy RMSE | OLS(P) RMSE | OLS(P)+roll10 RMSE |
|---|---:|---:|---:|
| Jan+Jul | 428.17 | 396.35 | 378.85 |
| December | 374.60 | 339.64 | 327.52 |
| July | 465.76 | 432.60 | 411.88 |
| January | 376.98 | 346.95 | 336.76 |

E0 reproduced these numbers exactly.

AOBT is a strong but noisy/miscalibrated operational proxy (minute resolution; airport BLOCK is second-level). MAE(BLOCK − AOBT) ≈ 238 s. Exact match 0.65%.

**Airport clock bias** — mean(TAXITIME − (MVT−AOBT)):

| Airport | Residual mean |
|---|---:|
| LTFM | −236 s |
| EHAM | −152 s |
| EDDF | −53 s |
| EGLL | −23 s |
| LEMD | +34 s |
| LFPG | +35 s |
| LSZH | +59 s |
| LEBL | +68 s |
| EDDM | +82 s |
| LIRF | +214 s |

OLS slope on P is ~0.59 (errors-in-variables shrinkage).

**Stand × runway geometry** (discovery, in-sample descriptive):

- ~12,053 observed stand–runway pairs
- ~6,436 with n≥30
- ~1,837 stands have multiple well-sampled runways
- same-stand / different-runway taxi spread ≈ 302 s mean
- stand×runway η² ≈ 0.2525
- airport η² ≈ 0.1121

**Traffic / queue vs raw TAXITIME** (Pearson, discovery):

- queue at takeoff ≈ 0.264
- same-runway departures 15 min ≈ 0.141
- same-runway departures 60 min ≈ 0.167
- airport-wide 15 min departures ≈ 0.051

**Extremes:** >1 h = 4,126 cases (LIRF, LTFM, EGLL, LFPG). Unmatched NM = 1.08% of DEP and 42% of mean-model SSE. Unmatched LIRF median 3,922 s. Do not discard.

### B. Conclusion changes after E0–E5

**C1. EOBT as a taxi estimate**

- OLD: `MVT−EOBT` is useless (raw RMSE 746 vs mean 687; discovery).
- EVIDENCE (E1): raw `MVT−EOBT` is still a bad *substitute* for taxi. OLS-calibrated `MVT−EOBT` matched RMSE 386.9 vs OLS `MVT−AOBT` 396.4 (MAE worse: 267 vs 252). Adding `AOBT−EOBT` to `MVT−AOBT` drops Jan+Jul **matched RMSE 396.4 → 347.6**.
- NEW: EOBT is not a taxi clock. **`AOBT−EOBT` (lateness vs plan) is a second clock** that linearly explains residual taxi after `MVT−AOBT`. Equivalent to OLS on `(MVT−AOBT, MVT−EOBT)`.

**C2. Rolling neighbour `MVT−AOBT`**

- OLD: roll10 is an important surface-state signal (396 → 379).
- EVIDENCE (E2b): after `AOBT−EOBT` is in the model, chaining roll10 changes per-airport matched RMSE 318.0 → 316.2 (Jan+Jul). December 276.9 → 274.2.
- NEW: roll10 is real but **second-order** once the EOBT lateness clock is present. Keep as a small chained fallback, not as the main idea.

**C3. Traffic and queue**

- OLD: queue/runway volume are promising dynamic surface-state features (corr 0.26 / 0.17 with raw taxi).
- EVIDENCE (E4, E5): on top of per-airport `AOBT + (AOBT−EOBT) + geo_mean`, adding traffic or queue changes Jan+Jul matched RMSE **303.53 → 303.64 / 303.56**. Linear incremental value ≈ 0.
- NEW: those raw correlations were largely **proxies for the AOBT duration clock and geometry**. Linear traffic/queue are redundant given clocks+geometry. Trees (E12) may still find interactions; not justified in linear models.

**C4. Additive airport bias**

- OLD: airport residual means (LTFM −236, LIRF +214) suggest adding a per-airport constant to the proxy.
- EVIDENCE (E2): airport *additive* bias matched RMSE 409 vs global OLS 396 (worse). Airport *slope+intercept* OLS 370.
- NEW: calibration must include **slope**, not only intercept. Errors-in-variables + different AOBT quality by airport.

**C5. Unmatched rows vs geometry**

- OLD: unmatched need a dedicated model; stand/runway might help because they are present without AOBT.
- EVIDENCE (E3): geometry on unmatched Jan+Jul RMSE 3937 vs airport mean 3961. LIRF unmatched still ~13,379.
- NEW: stand×runway does **not** explain unmatched LIRF extremes. Those rows are a different generating process (or data errors), not “wrong gate.”

**C6. Unmatched is not one population (E11)**

- OLD: unmatched (~1%) are “disproportionately difficult, especially LIRF.”
- EVIDENCE: unmatched = joint NM-join miss. EHAM unmatched median 538 (shorter than matched). EDDF/LEBL/LEMD unmatched ≈ matched. **LIRF unmatched: median 3922 s, 50% >1 h, aircraft type null 1487/1488, corr(y, MVT−SCHED)=0.92 on Jan+Jul val.** Ranking LIRF unmatched (383) are also 100% type-null.
- NEW: only LIRF unmatched (and mildly LTFM/LFPG tails) are the overall-RMSE bomb. Other unmatched airports can keep the geometry fallback. LIRF unmatched taxi is approximately takeoff-minus-schedule when BLOCK≈SCHED.

**C7. FLIGHT_ID is not a tail number (E7)**

- OLD (open): maybe FLIGHT_ID links aircraft/turnaround.
- EVIDENCE: 315,272 IDs appear twice; 315,262 are DEP then ARR of the **same city-pair** at two challenge airports (median 84 min later = airborne). Same-FLIGHT_ID ARR is after this DEP → leakage if used for taxi-out. Stand-previous-ARR same type is common (83%) but corr(taxi-out, prev taxi-in)=0.12. Same-callsign previous DEP is typically **yesterday’s rotation** (median 24.1 h), corr 0.41 with actual previous taxi — but previous DEP TAXITIME is blank for other ranking DEPs.
- NEW: REJECT FLIGHT_ID as aircraft ID. Stand turnaround taxi-in is weak. Callsign lag of actual TAXITIME is not fully ranking-safe; lag of `MVT−AOBT` remains optional.

**C10. Remaining matched error is tail compression, not leftover traffic/geometry (E14)**

- OLD: leftover matched RMSE ~256 s might be congestion, finer geometry, airport regimes, or clocks the tree missed.
- EVIDENCE: reproduced matched RMSE 256.46. Mean residual −0.3 s. Residual vs actual taxi: −113 s (<10 min) to +1631 s (>60 min). y>30 min = 45.8% of SSE. Traffic/queue corr(resid)≈0. Stand×runway η²=0.040. Airport η²=0.002. `AOBT−EOBT` top quantile = 36% of SSE (disruption tails).
- NEW: E15 should change the **residual objective / tail handling**, not add linear traffic, queue, or geometry tables.

**C11. Tail compression is not a residual-objective artifact (E15)**

- OLD (E14): a Huber / tail-weighted / two-stage residual might uncompress y>30 min without hurting the 10–20 min bulk.
- EVIDENCE (frozen P_cal, features, split, LIRF fallback): L2 reproduced matched 256.46. Huber δ=228 s improves <20 RMSE 174.5→162.9 but **worsens** >30 821→933 and matched 256→268. Tail-weighted L2 improves >30 821→654 and >60 2376→1946 but **destroys** the bulk (<20 175→269, mean residual −120 s, matched 300). Two-stage P(y>30)+tail expert is the same trade, worse (matched 318, <20 275, mean p_hat 0.136 vs true 0.045). December copies the pattern.
- NEW: **KEEP L2 residual.** REJECT Huber, tail weights, and two-stage mixture as replacements. The loss reallocates bulk vs tail error; it cannot identify E14 disruption tails from the frozen features. Do **not** run another residual-objective experiment.

**C12. Causal airport push-delay state is not just own AOBT−EOBT (E16-A)**

- OLD: leftover tail after E15 is unidentified given frozen features; airport disruption might already be in `AOBT−EOBT`.
- EVIDENCE: `dis_state_30m` = mean of *other* flights' AOBT−EOBT in the previous 30 min (self excluded, AOBT ≤ scored MVT). corr vs own AOBT−EOBT = 0.45, vs frozen residual = 0.11 (own AOBT−EOBT vs residual = 0.02). Inside every own-lateness quartile, high vs low disruption still raises >30 min rate (8.1× / 1.6× / 1.6× / 1.4×) and flips residual mean from over- to under-prediction. Adding it to the frozen L2 residual: Jan+Jul matched 256.46→253.82, >30 821→807, <20 +1.5 s; December matched 230.04→226.95. Q8 mean residual +60→−3.5. >60 min Jan+Jul not improved. LIRF matched −11 s; LFPG ~0.
- NEW: **KEEP `dis_state_30m` (+ `dis_frac20_30m`)** in the residual LightGBM. It is a real but small (~1% matched RMSE) uncompression of the operational disruption tail, not a fix for 1–24 h bombs. Do not sweep extra windows.

**C9. LIRF unmatched has two TARGET regimes, but they are not separable at prediction time (E13)**

- OLD (E11 / queue item 1): LIRF unmatched is bimodal (~15 min vs multi-hour). Gate `MVT−SCHED` so it is not applied to the normal half.
- EVIDENCE (E13, training holdouts): Oracle split is real (Jan+Jul 228 with y≤30 min vs 169 with y>30 min). On the **normal** half, `MVT−SCHED` RMSE=7081 vs P_cal/geo **379** — the current rule *does* wreck those flights. On the **extreme** half, `MVT−SCHED` RMSE=4224 vs P_cal 20502 (Dec extreme MAE **28 s**). But `MVT−SCHED` itself is ~6000 s in *both* y-bins (short-taxi unmatched still took off ~1.5 h late; delay was at the gate, not on the taxiway). Threshold `mvt_sched>1800` flags 383/397 rows (precision 0.43). Stand/hour/dest/prefix flags are too sparse. Train-best T>7200: Jan+Jul overall 410→398, December 274→278 (does not generalize).
- NEW: Keep **always** `MVT−SCHED` for unmatched LIRF/type_null. The two regimes exist in TAXITIME; they are **not** identifiable from takeoff-vs-schedule or simple stand/time/airline flags. Gating would require a feature that says “late takeoff was gate hold, not taxi” — AOBT/BLOCK, which unmatched rows do not have.

**C8. Residual trees beat direct trees on the ranking analogue (E9/E12)**

- OLD: residual vs direct “may be equivalent linearly; matters for trees.”
- EVIDENCE: Jan+Jul matched RMSE E3 304 → LGB direct 288 → LGB residual 256. December: direct 225 vs residual 230 (direct slightly better). Overall still needs the LIRF unmatched override; trees did not fully learn it.
- NEW: prefer residual LightGBM on the ranking analogue. Always overlay LIRF unmatched `MVT−SCHED`. Traffic/queue still absent from top importances.

---

## Feature definitions

| Feature | Definition | Ranking-safe? | Status |
|---|---|---|---|
| `mvt_aobt` | `MVT_TIME − AOBT_3` (s) | Yes if AOBT present | KEEP — primary clock |
| `aobt_eobt` | `AOBT_3 − EOBT_1` (s) | Yes | KEEP — lateness vs plan |
| `mvt_eobt` | `MVT_TIME − EOBT_1` | Yes | KEEP only as linear combo with `mvt_aobt` (equivalent to pair above) |
| `mvt_iobt` / `mvt_lobt` / `mvt_sched` | other takeoff−clock diffs | Yes | REJECT as taxi substitutes |
| `clock_std` / `clock_range` | spread of {AOBT,EOBT,IOBT,LOBT} | Yes | weak incremental; DEFER |
| `unmatched` | `AOBT_3` is null | Yes | KEEP as gate for fallback path |
| `roll10_mean_mvt_aobt` | mean of previous 10 completed `mvt_aobt` at airport (shift 1) | Yes | KEEP small; chain when finite |
| `geo_mean` | train-only stand×rwy mean if n≥30, else stand, else rwy, else airport | Yes | KEEP |
| `geo_med` / `geo_p90` / `sr_minus_*` | geometry variants | Yes | p90/deltas ≈ 0 once `geo_mean` in; REJECT for linear |
| `ac_family` | first 3 chars of `AIRCRAFT_TYPE_mvt` | Yes | mild in E2; DEFER to trees |
| `dep_*m` / `arr_*m` / `dep_rwy_*m` | causal movement counts | Yes | REJECT linear incremental |
| `queue` / `queue_rwy` / `push_*` | AOBT/MVT overlap queue | Yes (batch) | REJECT linear; unused by LGB top-gain |
| `type_null` | `AIRCRAFT_TYPE_mvt` is null | Yes | KEEP — nearly = LIRF unmatched |
| `mvt_sched` as unmatched LIRF predictor | `MVT_TIME − SCHED` | Yes | KEEP for LIRF unmatched / type-null |
| `dis_state_30m` | mean other-flight `AOBT−EOBT`, AOBT in previous 30 min, self excluded | Yes (AOBT ≤ scored MVT) | KEEP — airport push-delay state (E16-A) |
| `dis_frac20_30m` | fraction of those others with delay >20 min | Yes | KEEP small companion (E16-A) |
| `arr_delay_mean_30m` | mean landing−schedule of already-landed ARR, 30 min | Yes | optional; gain without residual corr |

---

## Validation protocol

Primary: train all 2025 months except January and July; validate January + July 2025.  
Sanity: train Jan–Nov, validate December.  
Also report January-only and July-only in E0.

Rolling features: strictly previous flights (shift 1).  
Split-fit tables (geometry, calibration): fit on the training split only.

When a feature is missing (unmatched / no roll10), **chain** to the next-best model that can score the row. Do not dump missing-feature rows onto the airport mean if a weaker clock/geometry predictor exists.

Metrics: RMSE and MAE overall; matched; unmatched; >30 min; >1 h; LIRF / EGLL / LFPG / LTFM.

**Reporting trap:** overall RMSE is dominated by unmatched (~3960). Always publish **matched RMSE** when judging clock/geometry features. Overall RMSE is the unmatched problem.

---

## Experiment table

Jan+Jul 2025 unless noted. “Matched RMSE” drops rows with NaN prediction; “Overall RMSE” uses documented fallback.

| ID | Experiment | Features | Model | Validation | Overall RMSE | Matched RMSE | MAE | Finding | Decision |
|---|---|---|---|---|---:|---:|---:|---|---|
| E0 | Baseline verification | mean, AOBT proxy, OLS, roll10 | mean / OLS | Jan+Jul | 632 (OLS+fb) | 396 / 379 | 252 | Discovery numbers reproduced | KEEP baseline |
| E1 | Clock relationships | NM clock diffs | raw / OLS | Jan+Jul | 603 | **348** | 227 | `AOBT−EOBT` is the second clock | KEEP `mvt_aobt+aobt_eobt` |
| E2 | AOBT calibration | P by airport/WTC/family/rwy | grouped OLS | Jan+Jul | 611–616 | 362–370 | 228–235 | Slope+intercept per airport; additive bias fails | KEEP airport OLS; REJECT additive-only |
| E2b | Clocks + airport cal | `mvt_aobt+aobt_eobt` ± roll10 | per-airport OLS | Jan+Jul | **586** | **316** | **207** | Airport × two-clocks; roll10 chain +2s | KEEP |
| E3 | Stand/runway geometry | geo hierarchy + clocks | OLS / per-airport | Jan+Jul | **577** | **304** | **191** | Geometry adds ~14s matched on top of E2b | KEEP `geo_mean` |
| E4 | Traffic/congestion | counts 5–60m, ratios, vs-hour | OLS on E3 | Jan+Jul | 577 | 304 | 190 | ≈0 incremental | REJECT linear |
| E5 | Queue dynamics | queue, rwy queue, push, growth | OLS on E3 | Jan+Jul | 577 | 304 | 189 | ≈0 incremental | REJECT linear |
| E11 | Unmatched / LIRF | missingness, type-null, MVT−SCHED | investigation + rule | Jan+Jul | **410** (E3+rule) | 304 | 190 | LIRF unmatched ≈ MVT−SCHED (corr 0.92) | KEEP rule; do not drop rows |
| E7 | FLIGHT_ID / turnaround | ID pairs, stand ARR, callsign lag | semantics | full 2025 | — | — | — | FLIGHT_ID = city-pair flight | REJECT ID as tail; DEFER callsign lag |
| E12 | LightGBM frozen repr. | clocks, geo, cats, traffic, queue | LGB 400/ES | Jan+Jul | 468 direct | **288** | **165** | Trees help matched; not unmatched LIRF | KEEP trees for matched |
| E9 | Residual vs direct | E3 P_cal + LGB residual | LGB | Jan+Jul | 448 / **378** w/ rule | **256** | 165 | Residual wins Jan+Jul matched | KEEP residual + LIRF override |
| E13 | LIRF unmatched regimes | MVT−SCHED threshold, stand/hour/dest/prefix | E3 + gated rule | Jan+Jul | 398 (T>2h) / 410 (always) | 304 | 189 | Two y-regimes; not separable at prediction time | KEEP always-on `MVT−SCHED`; REJECT gating |
| E14 | Matched residual diagnosis | current best, no new features | residual LGB reproduced | Jan+Jul | 378 | **256.46** | 158 matched | Tail compression + LIRF/LFPG disruption tails | KEEP diagnosis; E15 = tail-aware residual |
| E15 | Tail-aware residual objective | frozen E14 features | Huber / tail-weight / two-stage | Jan+Jul | 378 / 384 / 412 / 426 | **256** / 268 / 300 / 318 | 158 / 157 / 202 / 196 matched | Huber helps bulk, hurts tail; B/C help tail, wreck bulk | KEEP L2; REJECT A/B/C; no more loss experiments |
| E16-A | Airport disruption state | causal other-flight AOBT−EOBT 30m | E14 L2 + dis_state | Jan+Jul | **376.04** | **253.82** | 157.46 matched | Not just own AOBT−EOBT; Q8 resid +60→−3.5; >30 −15s; >60 no | KEEP `dis_state_30m`; small gain |

December of current best (E9 residual LGB + LIRF unmatched `MVT−SCHED`): overall **245**, matched **230**, MAE **153**. Direct LGB + override: overall 241, matched 225.

---

## Experiment log

### E0 — Baseline verification

- **Objective:** Reproduce discovery-phase AOBT numbers on shared split/metrics code.
- **Hypothesis:** Raw Jan+Jul proxy RMSE ≈ 428; OLS ≈ 396; OLS+roll10 ≈ 379.
- **Features:** `mvt_aobt`, `roll10_mean_mvt_aobt`; airport-mean fallback for unmatched.
- **Leakage:** none. Proxy uses ranking-present fields. Roll uses shift(1).
- **Validation:** Jan+Jul, Dec, Jan, Jul.
- **Model:** mean, raw proxy, OLS.
- **Result:** Hypothesis confirmed to 0.01 s on matched RMSE.

| Split | Global mean | Airport mean | Raw proxy (matched) | OLS AOBT (matched) | OLS AOBT+roll10 (matched, n with roll10) |
|---|---:|---:|---:|---:|---:|
| Jan+Jul | 686.62 | 659.96 | 428.17 | 396.35 | 378.85 |
| Dec | 514.74 | 488.42 | 374.60 | 339.64 | 327.52 |
| Jan | 604.85 | 584.73 | 376.98 | 346.95 | 336.76 |
| Jul | 745.98 | 715.32 | 465.76 | 432.60 | 411.88 |

OLS coef Jan+Jul: intercept 396.3, slope 0.586. Roll10 coefs: intercept 262, slope_P 0.472, slope_roll 0.249.

**Interpretation:** Unmatched RMSE ≈ 3960 (Jan+Jul) / 3139 (Dec). Overall OLS+fallback RMSE 632 vs matched 396. **Matched and overall are different problems.** Clipping proxy to [0,7200] slightly *hurts* matched RMSE on Jan+Jul (432 vs 428) because some true tails exceed 2 h.

**Decision:** KEEP as official baseline. **Next:** E1 clocks.

---

### E1 — Clock relationships

- **Objective:** Which clock differences predict taxi beyond (or instead of) `MVT−AOBT`?
- **Hypothesis:** Only `MVT−AOBT` is a taxi-like quantity; other diffs are delay/plan noise.
- **Features:** `mvt_aobt/eobt/iobt/lobt/sched`, `aobt_eobt/iobt/lobt/sched`, pairwise EOBT/IOBT/LOBT, `clock_std/range`, absolute diffs.
- **Leakage:** all NM clocks are in ranking. `MVT` is in ranking. Safe.
- **Validation:** Jan+Jul primary; Dec for winners.
- **Model:** raw value as prediction; univariate OLS; small OLS combinations. Unmatched → airport mean.

**Result (Jan+Jul matched RMSE unless noted):**

Raw-as-taxi: only `mvt_aobt` (428) is in the same world as the target. Everything else is 746–2591.

Univariate OLS: `mvt_aobt` 396.4; `mvt_eobt` 386.9 (better RMSE, worse MAE 267 vs 252); IOBT/LOBT ~398; all `aobt_*` and spreads ~456–477 (≈ airport mean 440).

Combinations:

| Combo | Overall RMSE | Matched RMSE | MAE |
|---|---:|---:|---:|
| OLS `mvt_aobt` | 631.94 | 396.35 | 252.21 |
| `mvt_aobt + aobt_eobt` | **603.02** | **347.57** | **226.77** |
| `mvt_aobt + mvt_eobt` | 603.02 | 347.57 | 226.77 |
| `mvt_aobt + clock_std` | 616.23 | 370.37 | 244.72 |
| `mvt_aobt + roll10` | 632.93 | 397.95* | 255.65 |
| `mvt_aobt + roll10 + aobt_eobt` | 609.59 | 359.02* | 233.32 |

\*roll10 nulls on some matched rows were filled with airport mean in the “else ap_mean” metric, which inflates matched RMSE vs the matched-only 378.85 from E0.

`mvt_aobt + aobt_eobt` ≡ `mvt_aobt + mvt_eobt` (linear dependence). Coefs Jan+Jul: intercept 205.5, `mvt_aobt` 0.687, `aobt_eobt` 0.257.

December: `aobt+eobt` matched **294.4** vs OLS AOBT 339.6. Stable.

EGLL matched-path RMSE 489 → 407; LTFM 436 → 362. LIRF overall still ~1750 (unmatched).

**Interpretation:** If NM off-block is later than EOBT, taxi is longer (positive 0.26 s taxi per 1 s lateness), even after accounting for `MVT−AOBT`. Operational reading: delayed pushback co-occurs with slower surface / longer hold, **or** AOBT and EOBT together triangulate airport BLOCK better than AOBT alone (two noisy clocks). Not resolved yet; both are ranking-safe.

**Decision:** KEEP `mvt_aobt + aobt_eobt`. REJECT other clocks as taxi substitutes. REJECT clock_std as a primary feature (small vs EOBT lateness). **Next:** E2 calibration.

---

### E2 — AOBT calibration

- **Objective:** How to calibrate `MVT−AOBT` — global, airport, airport×WTC/family/runway?
- **Hypothesis:** Airport slope+intercept beats global OLS and beats additive bias.
- **Leakage:** coefficients fit on train split only.
- **Validation:** Jan+Jul, Dec.
- **Model:** grouped OLS, min_n 100–200, shrinkage k=200–300 toward parent.

**Result (Jan+Jul, unmatched → airport mean):**

| Calibrator | Overall RMSE | Matched RMSE | MAE |
|---|---:|---:|---:|
| Global OLS AOBT | 631.94 | 396.35 | 252.21 |
| Airport additive bias | 639.92 | 409.14 | 245.49 |
| Airport OLS (slope+int) | 615.96 | 369.90 | 235.11 |
| Airport × WTC | 613.53 | 365.79 | 231.89 |
| Airport × ac_family | 611.09 | 361.61 | 228.36 |
| Airport × runway | 612.70 | 364.37 | 229.30 |
| Per-airport AOBT+roll10 | 617.07 | 371.79 | 239.00 |

December airport OLS matched 324.1 vs global 339.6. Shrinkage k=200 ≈ no-shrink (airports are large).

**Interpretation:** Additive bias over-corrects a noisy clock (classic errors-in-variables). Family/WTC/runway add a few seconds on top of airport OLS, much less than adding `AOBT−EOBT` (E1: 348). Fine cells are optional for trees, not the linear core.

**Decision:** KEEP per-airport slope+intercept. REJECT additive-only. DEFER WTC/family/runway cells to tree models. **Next:** E2b combine with E1 clocks.

---

### E2b — Clocks + airport calibration

- **Objective:** Put E1’s two-clock model inside per-airport OLS; chain roll10 only when finite.
- **Hypothesis:** Airport × (`mvt_aobt`,`aobt_eobt`) beats both parents; roll10 still helps if not naively filled with airport mean.
- **Leakage:** same as E1/E2.
- **Validation:** Jan+Jul, Dec.

**Result:**

| Model | Jan+Jul overall / matched / MAE | Dec overall / matched / MAE |
|---|---|---|
| Global AOBT | 632 / 396 / 252 | 461 / 340 / 233 |
| Per-airport AOBT | 616 / 370 / 235 | 450 / 324 / 221 |
| Global aobt+eobt | 603 / 348 / 227 | 429 / 294 / 203 |
| Per-airport aobt+eobt | **587 / 318 / 208** | **417 / 277 / 190** |
| Per-airport aobt+eobt+roll10 (NaNs hurt) | 595 / 334 / 216 | 424 / 288 / 195 |
| **Chain: 3feat else 2feat else ap mean** | **586 / 316 / 207** | **415 / 274 / 188** |

EGLL Jan+Jul overall 373 vs 489 at E0 OLS AOBT. LTFM 341 vs 436. LIRF overall 1721 — still unmatched.

**Decision:** KEEP per-airport two-clock OLS with roll10 chain. Current linear core before geometry. **Next:** E3.

---

### E3 — Stand/runway geometry

- **Objective:** Do historical stand×runway taxi tables add anything after clocks+airport calibration?
- **Hypothesis:** Geometry is the unimpeded component; AOBT clocks are the operational component; they should stack.
- **Features:** train-only mean/median/p25/p75/p90/std/n for airport×stand×runway, stand, runway, airport. Hierarchy n≥30. Contrasts `sr − stand/rwy/airport`. `mvt_aobt − geo_mean`.
- **Leakage:** tables from train split only. Stand and runway are in ranking.
- **Validation:** Jan+Jul, Dec. Unmatched scored with `geo_mean` (available without AOBT).

**Result (Jan+Jul):**

| Model | Overall | Matched | MAE | Unmatched |
|---|---:|---:|---:|---:|
| geo_mean hierarchy alone | 629.20 | 395.64 | 228.35 | 3937 |
| OLS AOBT (E0) | 631.94 | 396.35 | 252.21 | 3961 |
| aobt+eobt+geo (global) | 586.65 | 322.45 | 197.93 | 3937 |
| per-airport aobt+eobt (E2b) | 586.72 | 317.98 | 208.19 | 3961 |
| **per-airport aobt+eobt+geo** | **576.63** | **303.53** | **191.04** | **3937** |

December per-airport aobt+eobt+geo: overall 403.37, matched 260.93, MAE 174.06.

geo_p90, `sr_minus_rwy`, `sr_minus_ap` ≈ 0 once `geo_mean` is in (linear collinearity). Median hierarchy slightly worse RMSE than mean (tail-unfriendly).

Unmatched LIRF RMSE with geometry still 13,379. Geometry does not fix unmatched extremes.

Coefs global aobt+eobt+geo Jan+Jul: intercept −153, `mvt_aobt` 0.38, `aobt_eobt` 0.22, `geo_mean` 0.69. Geometry gets the largest weight — unimpeded time is first-class, clocks adjust it.

**Decision:** KEEP `geo_mean` hierarchy. REJECT extra geometry percentiles/contrasts in linear models. **Next:** E4 traffic on this baseline.

---

### E4 — Traffic / congestion

- **Objective:** After clocks+geometry, do causal movement counts still help?
- **Hypothesis:** Discovery correlations will shrink a lot; some runway-specific volume may remain.
- **Features:** dep 5/15/30/60m, same-runway 15/60, arr 15/30/60, ratios, acceleration, dep15 vs airport-hour mean. Causal rolling on `MVT_TIME`, current flight excluded.
- **Leakage:** counts use only past takeoffs/landings. Safe.
- **Validation:** Jan+Jul, Dec. Baseline = E3 per-airport aobt+eobt+geo.

**Result:** Every single traffic feature added to global aobt+eobt+geo changes matched RMSE by < 1 s (322.45 → 321.8–322.7). Per-airport + traffic combo: **303.64 vs baseline 303.53**. December 260.12 vs 260.93.

**Interpretation:** Linear traffic is redundant given `MVT−AOBT` (which already stretches when the surface is slow) and stand–runway means (which already encode typical peak-hour gates). Discovery corr(queue, taxi)=0.26 was mostly that.

**Decision:** REJECT as linear features. DEFER interactions to E12 trees. **Next:** E5 queue (same test).

---

### E5 — Queue dynamics

- **Objective:** Does explicit surface-queue (AOBT/MVT overlap) beat raw counts?
- **Hypothesis:** Queue is closer to “aircraft ahead” than a 15-min count, so it might survive after clocks.
- **Features:** `queue` at takeoff, same-runway queue, pushbacks 5/15m, queue growth over last 5/10 flights, queue/dep15. Vectorized; uses other flights’ AOBT and MVT (batch-available in ranking).
- **Leakage:** uses other flights’ future takeoff times relative to *i* to know they have not taken off yet. Allowed in this batch post-ops setup; would be illegal in a strictly causal live system. Flagged.
- **Validation:** Jan+Jul, Dec.

**Result:** +queue on global aobt+eobt+geo: matched 323.14 vs 322.45 (slightly worse). Per-airport + rwy15 + queue: **303.56 vs 303.53**. MAE 189.17 vs 191.04 (tiny). December matched 259.53 vs 260.93.

**Interpretation:** Queue is almost a restatement of how many aircraft have AOBT < t < MVT — information already baked into the distribution of `MVT−AOBT` around *i*. Linear model cannot use it twice.

**Decision:** REJECT linear. Keep the batch-leakage flag if a tree ever wants queue. **Next:** E6 remaining rolling stats, or skip to E7/E9 given roll10 already tested.

---

### E11 — Unmatched / LIRF investigation

- **Objective:** What unmatched rows actually are. Do not drop them; do not jump to a specialist black box.
- **Hypothesis:** Unmatched ≈ NM join failure; LIRF unmatched are a different generating process (data error or extra-long hold), not “slightly longer taxi.”
- **Leakage:** investigation only, plus a ranking-safe rule using `MVT−SCHED` (both present in ranking).
- **Validation:** full-year description; Jan+Jul / Dec for the diagnostic rule.

**What unmatched is**

- 22,470 DEP (1.08%). `AOBT/EOBT/IOBT/LOBT/MARKET/ARVT` are jointly null. 46 rows have `FLIGHT_ID` without AOBT.
- All unmatched rates and LIRF diagnostics below are from **training** (full year or Jan+Jul/Dec holdouts). Ranking files were not used for these fits.

**Not one population**

| Airport | Unmatched n | u median | matched median | u >1 h |
|---|---:|---:|---:|---:|
| LIRF | 1,488 | **3922** | 1024 | **50.1%** |
| EGLL | 1,414 | 1384 | 1319 | 3.2% |
| LTFM | 3,645 | 1009 | 963 | 6.1% |
| LFPG | 3,773 | 1023 | 954 | 1.2% |
| EHAM | 4,084 | **538** | 744 | 0.3% |
| others | | ≈ matched | | ~0–1% |

LIRF unmatched: **aircraft type null 1,487/1,488** (training). `type_null` is a training-observed alias of this slice.

LIRF unmatched BLOCK−SCHED median **+5 s** (on-time off-block vs schedule) but MVT−SCHED median **6657 s**. Taxi is huge because takeoff is hours after a schedule-like BLOCK. Only 4.7% cross midnight — not mostly overnight mis-tags. Previous-stand BLOCK copy is not the story (11/1488).

Bimodal: ~half look like normal ~15 min taxis (RYR/WMT/ITY also exist as matched flight numbers: 560/712); ~half are multi-hour (positioning-like callsigns `NOSOS*`, `CHHHH*`, dest LLBG, remote 8xx stands). A single median cannot RMSE-fix a two-component mixture (diagnostic: unmatched LIRF → train median moved airport-mean overall 660→638 only).

**The ranking-safe clock that tracks the extreme component**

Jan+Jul val LIRF unmatched n=397: **corr(y, MVT−SCHED)=0.916**. RMSE of raw `MVT−SCHED` 6033 vs airport mean 13443. Clip-to-4h **hurts**. Dec corr **0.976**.

Applied on top of E3 (matched unchanged):

| Split | E3 overall | E3 + LIRF unmatched=`MVT−SCHED` | unmatched RMSE | LIRF RMSE |
|---|---:|---:|---:|---:|
| Jan+Jul | 576.63 | **410.02** (405.17 if clip 24h) | 2228 | 909 |
| Dec | 403.37 | **274.41** | 891 | 442 |

Jan+Jul val: unmatched 1.56% of rows, **52.6% of mean-model SSE**; LIRF unmatched 0.12% of rows, **44.7% of SSE**.

**Decision:** KEEP unmatched rows. KEEP rule: if LIRF (or `type_null`) and unmatched, predict `MVT−SCHED` (optional clip 24h). REJECT treating all unmatched alike. REJECT a global unmatched constant. **Next:** trees still needed for matched; this rule is for the overall metric.

---

### E7 — FLIGHT_ID / turnaround

- **Objective:** Does `FLIGHT_ID` identify an aircraft / enable a leak-free chain?
- **Hypothesis:** It might be a tail or rotation key. Do not assume that.
- **Leakage check:** same-ID ARR is after DEP; DEP `BLOCK` as asof key is illegal at test.

**Result**

- Multiplicity: 3,506,323 IDs once; **315,272 twice**. Of pairs: 315,262 DEP then ARR, 312,568 with identical ADEP/ADES, two airports, median Δt 5065 s (~84 min). This is **the same city-pair flight observed as departure then arrival** inside the 10-airport set.
- Using that ARR for the DEP’s taxi is **future information**.
- Stand asof previous ARR: 99.9% of DEPs match someone (vacuous). Same type 94%. Same callsign 2%. Same FLIGHT_ID ~0. Plausible 20 min–8 h same-type: 83% of DEP. corr(taxi-out, prev taxi-in)=**0.12**; corr vs turnaround duration=0.09. Weak.
- Same callsign previous DEP at airport: median gap **24.1 h**, different FLIGHT_ID ~100%, corr with previous **actual** taxi **0.41**. That previous TAXITIME is blank for other ranking DEPs in the same month. Ranking-safe substitute would be previous `MVT−AOBT`, not tested here.

**Decision:** REJECT `FLIGHT_ID` as tail/turnaround key. REJECT same-ID ARR features. DEFER stand taxi-in (weak). DEFER callsign lag until implemented with `MVT−AOBT` only. **Next:** E12.

---

### E12 — LightGBM on the frozen representation

- **Objective:** Do nonlinear interactions beat the 304/577 linear core?
- **Hypothesis:** Trees help matched via stand/airline/hour interactions; traffic/queue may appear; unmatched LIRF still needs the E11 rule.
- **Features:** `mvt_aobt`, `aobt_eobt`, `mvt_sched`, `geo_mean`, `roll10`, dep/arr/queue counts, hour/dow/month, unmatched, type_null; categoricals airport/runway/stand/type/WTC/segment/operator/ADES. Train-only geometry. Causal rolling.
- **Model:** LGBMRegressor, 400 trees, lr 0.05, 63 leaves, early stopping 40 on last 15% of train by time. No HP search.
- **Validation:** Jan+Jul, Dec.

**Result**

| Model | Jan+Jul overall / matched / MAE | Dec overall / matched / MAE |
|---|---|---|
| E3 linear | 577 / 304 / 191 | 403 / 261 / 174 |
| E3 + LIRF `MVT−SCHED` | 410 / 304 / 190 | 274 / 261 / 171 |
| LGB **direct** | 468 / **288** / **165** | 294 / **225** / **151** |
| LGB residual (E9) | 448 / **256** / 166 | 296 / 230 / 154 |
| Direct + LIRF override | 399 / 288 / 164 | **241 / 225 / 150** |
| Residual + LIRF override | **378 / 256 / 165** | 245 / 230 / 153 |

Top gain (Jan+Jul residual): STAND, ADES, `mvt_sched`, operator, `aobt_eobt`, RUNWAY, `mvt_aobt`, roll10, geo_mean, airport, hour, type. **Traffic and queue not in the top 12.**

**Interpretation:** Trees improve matched RMSE ~48 s vs E3 (304→256) and MAE 191→165. They do **not** replace the LIRF unmatched rule (direct overall 468 vs 410 with the rule alone). Stand and destination carry extra geometry/route beyond `geo_mean`. `mvt_sched` is used globally by the tree, not only on unmatched.

**Decision:** KEEP LightGBM for matched residuals. KEEP E11 override. DEFER traffic/queue even in trees (unused). No HP search warranted until unmatched LIRF is handled structurally.

---

### E9 — Residual vs direct

- **Objective:** `LGB → y` vs `P_cal = E3`, `LGB → y − P_cal`.
- **Hypothesis:** Residual lets the tree spend capacity on what clocks/geometry miss.
- **Same features/splits as E12.**

**Result:** Residual wins Jan+Jul matched **256 vs 288** and overall with override **378 vs 399**. December: direct slightly better matched 225 vs 230. Ranking analogue is Jan+Jul → residual preferred.

**Decision:** KEEP residual formulation as the default tree. Direct remains a sanity check. **Next:** unmatched-LIRF mixture (E13).

---

### E13 — LIRF unmatched regimes

- **Question:** Can we identify normal vs extreme LIRF unmatched/type_null at prediction time, so `MVT−SCHED` is not applied to ~15 min taxis?
- **Hypothesis:** Two TAXITIME regimes exist; `MVT−SCHED` (and/or stand, hour, dest, prefix) can gate the current rule.
- **Features (prediction-time only):** `MVT−SCHED`, hour, weekday, month, runway, stand prefix, ADES, flight-number prefix. No TAXITIME as a feature. No AOBT/EOBT/WTC (null on this slice). No ranking files.
- **Validation:** Jan+Jul 2025 and December 2025 training holdouts. Thresholds chosen on the **train** side of each split. Compared to E3 + always `MVT−SCHED` (unmatched branch of current best; matched RMSE unchanged vs E3).
- **Leakage:** `MVT` and `SCHED` are on the row. Geometry/P_cal fit on train split only.

**The two target regimes are real**

Full-year training LIRF unmatched n=1,488 (type_null 1,487). corr(y, `MVT−SCHED`)=0.90.

| y bin | n | mean y | mean `MVT−SCHED` | RMSE(`MVT−SCHED`) |
|---|---:|---:|---:|---:|
| <15 min | 278 | 703 | **5844** | 5930 |
| 15–30 min | 418 | 1201 | **6677** | 6848 |
| 30–60 min | 47 | 2300 | 6132 | 5272 |
| 1–2 h | 306 | 5832 | 5729 | **949** |
| >2 h | 439 | 16236 | 15121 | 5645 |

Short-taxi unmatched flights still have takeoff ~1.5 h after schedule. Delay was **before off-block**. `MVT−SCHED` = gate delay + taxi; without BLOCK/AOBT we cannot split those.

Jan+Jul val LIRF unmatched n=397 (228 y≤30 min, 169 y>30 min, 148 y>1 h):

| Predictor | Oracle-normal RMSE / MAE | Oracle-extreme>30m RMSE / MAE |
|---|---|---|
| P_cal / geo | **379 / 294** | 20502 / 11980 |
| Always `MVT−SCHED` | 7081 / 6236 | **4224 / 1521** |

December extreme half: `MVT−SCHED` MAE **28 s**. The rule is nearly exact when taxi really is takeoff−schedule (BLOCK≈SCHED). It is the wrong clock when the aircraft left the gate late.

**Prediction-time separation fails**

`mvt_sched>1800` vs oracle y>1800 (Jan+Jul): TP 163, FP 220, FN 6, TN 8. Precision **0.43**, recall 0.96. Flags 383/397 rows — essentially the current always-on rule.

Train-chosen T>7200 (best LIRF unmatched RMSE on train):

| Split | Always SCHED overall | T>7200 overall | Notes |
|---|---:|---:|---|
| Jan+Jul | 410.02 | **397.69** | oracle-normal still 5561 vs P_cal 379 |
| Dec | **274.41** | 277.86 | does not generalize |

Stand 8xx, night hours, LLBG, prefixes NOS/ISR/ETH raise P(y>30 min) but as the *only* trigger miss most extremes (overall 529–574 vs 410). OR-ing them with `ms>1800` collapses back to always-on.

EOBT/WTC/aircraft type cannot be used: they are null on this slice.

**Oracle ceiling (not a deployable rule):** route normal→P_cal and extreme→`MVT−SCHED` would recover ~43 s overall vs always-on (≈367 vs 410) if we magically knew y. We do not.

- **Interpretation:** The current rule trades a large error on ~half of LIRF unmatched (the 15 min flights) for a much larger save on the multi-hour half. RMSE prefers that trade. We cannot make the trade only on the extreme half with available prediction-time fields, because late-vs-schedule is common to both.
- **Decision:** KEEP always-on `MVT−SCHED` for unmatched LIRF / type_null. REJECT MVT−SCHED thresholds, stand/hour/dest/prefix gates. Do not claim a regime classifier. **Next:** E14 matched residual-error analysis.

---

### E14 — Matched residual diagnosis

- **Question:** Where does the remaining ~256 s matched RMSE come from?
- **Hypothesis:** leftover geometry, congestion, airport intercepts, clocks, or tail compression / irreducible noise.
- **Features:** none new. Reproduce E9 residual LightGBM + E11 LIRF override.
- **Validation:** train 2025 except Jan+Jul; val Jan+Jul 2025. December stress. `training_*.parquet` only.
- **Reproduction:** overall 378.28, matched **256.46** (delta 0.00 vs E9), matched MAE 157.83. December matched 230.04.

**Findings (matched Jan+Jul, N=339,046)**

- Mean residual −0.3 s: **not globally biased**. July RMSE 277 vs January 228.
- **Tail compression is the main structure:** mean residual −113 s if y<10 min, +370 s at 30–45 min, +1631 s if y>60 min. 775 flights >60 min = **19.6% of SSE**. 4.5% of flights >30 min = **45.8% of SSE**. Worst 1% of errors = 40.9% of SSE.
- LIRF matched RMSE 463 and **25% of SSE**, but mean residual only +5 s (variance/tails, including one 87,002 s point). η² airport = 0.002.
- Stand×runway η² of residual = 0.040. Geometry leftovers modest.
- Traffic/queue corr(resid) ≈ 0. Queue Q1 vs Q8 mean residual 0 vs +2 s. Confirms E4/E5.
- `AOBT−EOBT` Q8 (largest push delay) RMSE 452 and **36% of SSE**. Hard disrupted flights, still shrunk.
- Hour / WTC / type η² ≤ 0.006. Missingness among matched is negligible.
- Worst 50: almost all July, actual 1.5–24 h, LIRF 25 and LFPG, large `AOBT−EOBT`, queue not extreme.

Artifacts: `analysis/E14/figures/`, `tables/`, `reports/E14_report.md`.

- **Decision:** remaining matched error is **tail compression + disruption tails**, not unused congestion or missing stand–runway means. **E15 = tail-aware residual objective on frozen features.** Do not add feature families first.

---

### E15 — Tail-aware residual objective (frozen features)

- **Question:** Can Huber, tail-weighted L2, or (if those fail) a two-stage `P(y>30 min)` + tail residual uncompress the E14 tail without degrading y<20 min?
- **Frozen:** `P_cal`, E12 feature matrix, Jan+Jul / December splits, LIRF unmatched `MVT−SCHED`, preprocessing, LGB capacity (400 / 63 leaves / ES 40 / seed 1).
- **Validation:** train 2025 except Jan+Jul; val Jan+Jul 2025. December stress. `training_*.parquet` only.
- **Reproduction:** L2 matched RMSE **256.46** (delta −0.00 vs E14).

**Jan+Jul matched**

| Variant | Overall | Matched | MAE | <20 | >30 | >60 | SSE share >30 | Mean resid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 (E14) | **378.28** | **256.46** | **157.83** | 174.51 | 821.37 | 2375.85 | 45.8% | −0.3 |
| A Huber δ=228 s | 384.21 | 267.66 | 156.66 | **162.90** | 932.83 | 2774.53 | 54.2% | +17.6 |
| B tail-weighted | 411.87 | 300.34 | 201.78 | 268.87 | 653.67 | 1946.48 | 21.2% | −119.9 |
| C two-stage | 426.31 | 317.99 | 195.68 | 274.95 | **623.31** | **1872.68** | 17.2% | −84.4 |

December: L2 245.38 / 230.04 still best; A/B/C repeat the same bulk-vs-tail trade.

- A: Huber is the wrong direction for RMSE tails. Bulk improves, tail and matched RMSE worsen.
- B: tail RMSE does fall, but by overpredicting everyone (mean residual −120 s). Matched RMSE +44 s.
- C run because A/B failed the “tail without bulk damage” gate. Best tail RMSE, worst contest RMSE. Gate `p_hat` mean 0.136 vs true 4.5%.
- Winner = **L2**. >30 / >60 did **not** improve under the winner. <20 did **not** degrade under the winner. B/C improve the tail only by degrading the bulk.

Artifacts: `analysis/E15/figures/`, `tables/`, `reports/E15_report.md`.

- **Decision:** KEEP L2 residual. REJECT Huber, tail-weighted L2, and two-stage mixture. **Do not run another residual-objective experiment.** Remaining tail is unidentified given frozen features (E14 disruption tails), not an L2 artifact.

---

### E16-A — Airport / network disruption state

- **Question:** Does a causal airport push-delay state distinguish a normal late push from a systemic disruption, beyond own `AOBT−EOBT`?
- **Features:** `dis_state_30m` = mean of *other* flights' `AOBT−EOBT` with AOBT in the previous 30 min (self excluded). Companions: frac>20/30, delayed-push counts, airport×hour anomaly, already-landed arrival delay. No new clocks of the scored flight.
- **Frozen:** `P_cal`, split, LIRF unmatched override, L2 residual capacity.
- **Validation:** Jan+Jul 2025 holdout; December stress. `training_*.parquet` only.

**Phase 2 (frozen model, Jan+Jul matched):** Q8 vs Q1 of `dis_state_30m`: >30 min rate 2.37%→12.34% (×5.22), residual RMSE 213→389, mean residual −28→+60, 28.6% of SSE. corr(state, own AOBT−EOBT)=0.45; corr(state, residual)=0.11. Inside all four own-lateness quartiles the >30 min rate still rises with disruption (×8.1 / 1.6 / 1.6 / 1.4). **Not redundant with AOBT−EOBT.**

**Phase 3 (same L2 residual + disruption columns)**

| Split | Overall | Matched | MAE | <20 | >30 | >60 |
|---|---:|---:|---:|---:|---:|---:|
| Jan+Jul baseline | 378.28 | 256.46 | 157.83 | 174.51 | 821.37 | 2375.85 |
| Jan+Jul E16-A | **376.04** | **253.82** | 157.46 | 176.00 | 806.79 | 2382.43 |
| Dec baseline | 245.38 | 230.04 | 151.04 | 162.13 | 733.59 | 2153.91 |
| Dec E16-A | **241.27** | **226.95** | 148.10 | 156.13 | 726.70 | 2100.69 |

Q8 mean residual +60 → −3.5. `dis_state_30m` rank 7. LIRF matched −11 s; LFPG ~0. >60 min not improved on the primary split.

Artifacts: `analysis/E16A/`.

- **Decision:** KEEP `dis_state_30m` (+ `dis_frac20_30m`). Small consistent matched-RMSE gain, not an E14 solution. Do not sweep extra disruption windows.

---

## Failed approaches

| Approach | Why it failed |
|---|---|
| Raw `MVT−EOBT` / `MVT−LOBT` / `MVT−SCHED` as taxi | Wrong quantity; RMSE worse than the mean (E1). |
| Airport additive bias on AOBT | Ignores slope; matched RMSE 409 vs 396 global OLS (E2). |
| Filling missing roll10 with airport mean | Turns a 379 matched model into 398 on the full matched set (E0/E1). Must chain to AOBT OLS. |
| Linear traffic counts after clocks+geometry | ≈0 ΔRMSE (E4). |
| Linear queue after clocks+geometry | ≈0 ΔRMSE (E5). |
| Geometry percentiles / stand−runway deltas in OLS | Collinear with `geo_mean` (E3). |
| Using geometry to fix unmatched LIRF | Unmatched LIRF RMSE still ~13k (E3). |
| Treating all unmatched airports like LIRF | EHAM unmatched are *shorter* than matched (E11). |
| Unmatched LIRF → train median/mean constant | Bimodal; overall RMSE barely moves (660→638) (E11). |
| Clip `MVT−SCHED` to 4 h for LIRF unmatched | True tails exceed 4 h; RMSE worse than unclipped (E11b). |
| `FLIGHT_ID` as aircraft/turnaround | It is a city-pair NM flight; paired ARR is after DEP (E7). |
| LightGBM without LIRF unmatched override | Matched improves; overall still ~448–468 vs 410 with the simple rule (E12). |
| Gate `MVT−SCHED` on LIRF unmatched via threshold/stand/hour/dest/prefix | Two y-regimes exist, but `MVT−SCHED` is large in both; gates do not generalize (E13). |
| Huber residual (E15-A) | Improves <20 min RMSE, worsens >30/>60 and matched RMSE. Robust loss down-weights the tail we need (E15). |
| Tail-weighted residual (E15-B) | >30/>60 RMSE fall; bulk overpredicted (mean residual −120 s); matched RMSE 256→300 (E15). |
| Two-stage P(y>30)+tail residual (E15-C) | Best tail RMSE, worst matched/overall; gate fires at 14% vs true 4.5%; December unmatched 886→1365 (E15). |

---

## Assumptions

1. Research-phase data = the 12 `training_*.parquet` files only. Ranking/submitting are out of scope until a later downstream step.
2. `AOBT_3` is NM actual off-block, not airport BLOCK. Using it as a feature is a training-column choice, not a ranking-file statistic.
3. Jan+Jul 2025 holdout (from training months) is the primary validation split.
4. Static tables fit on “all training months except Jan+Jul” use Aug–Dec when scoring July. Slightly optimistic for July; December split is the fully causal sanity check.
5. Unmatched rows need a fallback. Airport mean and `geo_mean` are almost equally useless on LIRF unmatched.
6. Queue at takeoff using `MVT_j ≥ MVT_i` uses other **training** rows’ takeoff times (batch within the training stream).

---

## Leakage risks

- Other DEP `TAXITIME` / `BLOCK` at test time.
- Rolling windows that include the current flight or future flights.
- This flight’s `ARVT_3` (arrival after takeoff).
- Treating `FLIGHT_ID` as an aircraft tail number without evidence (E7 not done).
- Stand–runway means computed with future months when interpreting July strictly.
- Queue feature uses later takeoffs to mark “still taxiing” (batch-only).

---

## Decisions

- Primary matched metric for clock/geometry research; always also report overall (unmatched).
- Linear core is **per-airport OLS** `y ~ mvt_aobt + aobt_eobt + geo_mean`, unmatched → `geo_mean` → airport mean. Optional roll10 chain.
- Do not add linear traffic/queue to that core.
- Do not start a large hyperparameter search until E12.
- Do not discard unmatched / >1 h rows; they need their own experiment (E11 / unmatched model).
- Linear core remains E3. Unmatched LIRF (type-null) uses **always** `MVT−SCHED` (E13: do not gate). Trees predict the residual of E3 for matched rows.
- Do not use `FLIGHT_ID` as an aircraft key.
- Traffic/queue stay out of the linear core and were not used by LGB top-gain; do not re-open without a new hypothesis.
- Do not start a large hyperparameter search.
- Residual trainer stays **L2**. Do not replace it with Huber, tail weights, or a P(y>30) mixture (E15). Do not start a residual-objective grid.
- Keep causal `dis_state_30m` (other-flight AOBT−EOBT, 30 min, self excluded) in the residual model (E16-A). Do not sweep extra disruption aggregations without a new hypothesis.

---

## Current best model

**Name:** E9 residual LightGBM + E11 LIRF unmatched override + E16-A disruption state.

```
P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)   # unmatched → geo_mean
if unmatched and (airport==LIRF or type_null):
    y_hat = MVT - SCHED
else:
    y_hat = P_cal + LightGBM_residual_L2(clocks, geo, stand, dest, airline,
                                         dis_state_30m, dis_frac20_30m, ...)
```

| Split | Overall RMSE | MAE | Matched RMSE | Unmatched RMSE | LIRF | EGLL | LFPG | LTFM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan+Jul E16-A residual+override | **376.04** | 157.46 matched | **253.82** | 2236 | — | — | — | — |
| Jan+Jul residual+override (no disruption) | 378.28 | 165.09 | 256.46 | 2241 | 869 | 285 | 585 | 277 |
| Jan+Jul E3+override (no tree) | 410.02 | 189.85 | 303.53 | 2228 | 909 | 360 | 612 | 319 |
| Jan+Jul E3 only | 576.63 | 191.04 | 303.53 | 3937 | 1721 | 360 | 612 | 319 |
| Dec E16-A residual+override | **241.27** | 148.10 matched | **226.95** | 852 | — | — | — | — |
| Dec residual+override (no disruption) | 245.38 | 153.32 | 230.04 | 886 | 405 | 239 | 285 | 252 |
| Dec direct LGB+override | 240.79 | 150.09 | 225.29 | 881 | 403 | 244 | 281 | 245 |

Vs E3-only: Jan+Jul overall 577 → 378, matched 304 → 256, MAE 191 → 165.  
Most of the overall drop is the LIRF unmatched rule (577 → 410) not the tree.

E15 left the objective unchanged. E16-A adds causal airport push-delay state (`dis_state_30m`) on top of the same L2 residual: Jan+Jul matched 256.46→253.82, December 230.04→226.95.

---

## Open questions / research queue

1. **E15 closed.** Huber, tail-weighted L2, and two-stage P(y>30)+tail expert all lose to L2 on contest RMSE. Do not reopen residual-objective work.
1b. **E16-A kept (small).** `dis_state_30m` is not own AOBT−EOBT. Do not sweep extra disruption windows. Remaining >60 min / LFPG tails are still open.
2. **LIRF unmatched (closed as a gate problem):** always `MVT−SCHED`. Further gains need a gate-vs-taxi delay split, which unmatched rows do not provide (no AOBT/BLOCK).
3. **Callsign lag of `MVT−AOBT`** (not actual TAXITIME) — E7 corr 0.41 used leaky labels.
4. **E8** calendar/route: ADES already high LGB gain; explicit airport×hour maybe redundant.
5. **E10** hidden BLOCK: for LIRF unmatched, `BLOCK_hat ≈ SCHED` then `TAXI_hat = MVT − SCHED` is exactly the always-on E11 rule. Does not fix the normal unmatched half (their BLOCK is also late).
6. CatBoost vs LightGBM, still no large HP search.
7. Why `AOBT−EOBT` works — triangulation vs delay→taxi.
8. E6 rolling quantiles remain low priority.
9. Remaining unmatched RMSE (~2200 Jan+Jul) = non-LIRF unmatched + LIRF normal-half damage from `MVT−SCHED`.

---

## Changelog

- **2026-09-05 (discovery):** Imported dataset structure, target identity, AOBT proxy, airport bias, geometry η², traffic correlations, unmatched/LIRF tail facts. See `analysis/DISCOVERY_REPORT.md`.
- **2026-09-05 (session):** Created `status.md`, `experiments/common.py`.
- **2026-09-05 E0:** Verified AOBT baselines exactly. Documented unmatched vs matched RMSE split.
- **2026-09-05 E1:** Found `AOBT−EOBT` as second clock (matched 396 → 348). Updated C1 vs discovery “EOBT is useless.”
- **2026-09-05 E2:** Airport slope+intercept KEEP; additive bias REJECT.
- **2026-09-05 E2b:** Per-airport two-clock OLS matched 316. Roll10 demoted to chained extra (C2).
- **2026-09-05 E3:** `geo_mean` KEEP. Current best matched 304 / overall 577. Geometry does not fix unmatched LIRF (C5).
- **2026-09-05 E4, E5:** Linear traffic and queue REJECT on top of best model (C3). Queue batch-leakage flagged.
- **2026-09-05 E11:** Unmatched is joint NM miss, heterogeneous by airport. LIRF unmatched is type-null and bimodal; `MVT−SCHED` corr 0.92. E3+rule overall 577→410. C6 recorded.
- **2026-09-05 E7:** `FLIGHT_ID` is a city-pair flight, not a tail. REJECT. Callsign daily lag DEFER with ranking-safe clocks only. C7 recorded.
- **2026-09-05 E12/E9:** Residual LGB matched 304→256. Direct weaker on Jan+Jul. Trees do not replace the LIRF rule. Combined overall **378**. Traffic/queue unused. Current best updated. C8 recorded.
- **2026-09-05 data policy:** Binding training-only rule. Training corpus = 12 `training_*.parquet` files. `ranking.parquet` / `submitting.parquet` must not enter fits, stats, encodings, or HP. Loaders in `experiments/common.py` now refuse non-`training_` files. E11 ranking descriptive peek removed from experiment code; it never entered fitted parameters. E0–E12 RMSE remains training-holdout only.
- **2026-09-05 E13:** Two LIRF unmatched TAXITIME regimes confirmed. `MVT−SCHED` is ~6000 s in both (gate delay vs taxi). Thresholds/flags cannot gate the rule stably. KEEP always-on `MVT−SCHED`. C9 recorded. Current best pipeline unchanged.
- **2026-09-05 E14:** Reproduced matched RMSE 256.46. Remaining error is tail compression (y>30 min = 46% SSE) plus LIRF/LFPG disruption tails. Traffic/geometry leftovers small. C10 recorded. Recommended E15: tail-aware residual, no new feature family. Artifacts in `analysis/E14/`.
- **2026-09-06 E15:** Frozen-feature residual-objective test. L2 reproduced 256.46. Huber helps bulk, hurts tail. Tail-weight and two-stage help >30/>60 by overpredicting the bulk (matched 300 / 318). Winner = L2. C11 recorded. No further loss-function experiment. Artifacts in `analysis/E15/`.
- **2026-09-06 E16-A:** Causal airport push-delay state (other flights' AOBT−EOBT, 30 min, self excluded) is not redundant with own AOBT−EOBT. L2 + `dis_state_30m`: Jan+Jul matched 256.46→253.82, >30 821→807, >60 not improved; December matched 230.04→226.95. KEEP the family, small gain. C12 recorded. Artifacts in `analysis/E16A/`.
- **2026-09-06 E14-surface-state (spec, `experiments/`):** Leakage-safe dynamic surface-state features (strictly `< t`) added to the residual LGB of the E13 architecture (P_cal unchanged; LIRF override unchanged). Refit E13 reproduces 378.28/256.46 exactly. E14 (31 new cols: dep/arr 5-60m counts, same-runway 10/30m, time-since, rates, rolling taxi mean/med/p90/std, pressure/accel/burst): Jan+Jul overall 378.28→**371.57** (matched 256.46→248.05, MAE 165→159); December 245.38→**231.09** (matched 230.04→218.40). Gain concentrated in rolling-taxi stats + `s_rwy_dep_30m`; traffic counts ≈0. All airports improve except EHAM (Jan+Jul). **Caveat:** the winning features consume other DEPs' `TAXITIME` — NOT ranking-safe (ranking blanks DEP TAXITIME); a ranking-safe variant (rolling `MVT−AOBT`) must be tested before transfer. Artifacts: `experiments/run_e14_dynamic_surface_state.py`, `experiments/e14_features.py`, `experiments/results/E14/` (metrics.json, summary.txt, feature_importance.csv, 8 plots ×2 splits).

---

## E17-A — Ranking-Safe Operational Memory (2026-09-06)

**Hypothesis:** E14 showed recent taxi behaviour predicts taxi-out, but TAXITIME is not ranking-safe. Can the same short-term operational-memory signal be recovered from strictly-causal rolling stats of `MVT−AOBT`, `AOBT−EOBT`, `MVT−SCHED`, and runway-local history?

**Feature groups (all RANKING_SAFE, strictly `< t`, ZERO TAXITIME anywhere):**
- Airport memory (21): rolling mean/median/P90/std of `MVT−AOBT` (5/10/15/30/60m), `AOBT−EOBT` (same windows), `MVT−SCHED` (10/30m).
- Runway memory (14): same-runway dep counts 5–60m, time-since-previous, rate, burst, acceleration, and rolling `MVT−AOBT` stats (10/30m mean, 30m med/P90/std).

**Leakage methodology:** counts/gaps via `searchsorted` (`< t` exact); rolling stats via pandas time-windows with `shift(1)` (previous rows only), same convention as `add_causal_rolling`. Training files only; no ranking/submitting.

**Baseline reproduction:** E17-A0 = E16-A refit → Jan+Jul **376.04 / 253.82** (Δ 0.00), Dec 241.27 / 226.95. Exact.

**Results (overall / matched):**

| Model | Jan+Jul | Dec |
|---|---:|---:|
| E17-A0 (E16-A) | 376.04 / 253.82 | 241.27 / 226.95 |
| E17-A1 (+ airport memory) | **375.07** / **251.95** | **236.89** / **222.29** |
| E17-A2 (+ runway memory) | 374.88 / 251.98 | 238.44 / 224.62 |

**RMSE improvement:** A1 vs E16-A: Jan+Jul −0.97 (matched −1.87), Dec −4.38. A2 vs A1: Jan+Jul −0.19 (matched +0.03), Dec **+1.55 (matched +2.32 — WORSE)**. E14 reference (unsafe) 371.57 — **not recovered** (gap +3.31).

**Airport-level effects (A2−A0, Jan+Jul):** biggest wins LTFM −6.5, EDDM −5.4, LEBL −2.1; hurt LSZH +5.9, EHAM +1.9. On Dec wins EDDM −8.6, LTFM −6.4, LFPG −5.9; hurt EGLL +2.2, LIRF +1.7.

**Error-tail effects:** matched SSE share >600 s effectively unchanged; gains are spread across the 180–600 s regime, not the extreme tail (same pattern as E16-A — tail compression not fixed by operational memory).

**Feature importance (E17-A2, top-30):** 10 memory features — 7 airport (top: `rm_ae_med_30m`, `rm_sched_mean_30m`, `rm_aobt_med_30m`) + 3 runway (`rm_rwy_aobt_std_30m`, `rm_rwy_ts_dep`, …). `dis_state_30m` (E16-A) still ranks above all of them.

**Ranking-safety conclusion:** E17-A is 100% ranking-safe — no feature depends on any departure's TAXITIME; all inputs are clocks known at scored MVT.

**Decision:** **INCONCLUSIVE.** E16-A baseline reproduced exactly; airport-level memory gives a small consistent gain on both splits, but the runway-local family is unstable (helps Jan+Jul marginally, hurts December), the total Jan+Jul gain is only −1.16 s (−1.84 matched), and E14's level (371.57) is not recovered. Per the acceptance rule, this is a tiny, partially-consistent improvement — do not auto-accept.

**Recommendation for next experiment:** keep the airport-memory family (especially `AOBT−EOBT`/`MVT−AOBT` median memory), drop or re-parameterize the runway-local family, and focus on the matched tail (>30 min = ~45% of matched SSE), which operational memory does not touch. Reconsider after testing a ranking-safe rolling `MVT−AOBT` (E14-unsafe taxi variant) to quantify how much of the remaining E14 gap is recoverable. Artifacts: `experiments/e17a_features.py`, `experiments/run_e17a_ranking_safe_memory.py`, `experiments/results/E17-A/`.
