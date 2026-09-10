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

**C13. Non-LIRF unmatched is not a mean-offset problem; LIRF residuals were contaminating the tree (E18)**

- OLD: remaining unmatched RMSE ~2200 is non-LIRF unmatched (EHAM shorter, LTFM/EGLL tails) scored with matched `geo_mean` + residual; a dedicated unmatched head or airport unmatched mean should move overall RMSE.
- EVIDENCE (E16-A reproduced 376.04 / 253.82): non-LIRF unmatched n=4,976 RMSE **1579**. Train airport-unmatched mean/median **raise** overall RMSE. Jan+Jul EHAM unmatched mean y=742 vs train unmatched mean 565 — the full-year “EHAM shorter” fact is not a stable prior. Dropping the residual on unmatched (A-geo) helps LSZH 1017→640 but fails December (+1.40 overall). A small unmatched-only tree helps Jan+Jul (−1.60) and fails December (+0.53). **Hygiene** (matched-only `geo_mean` + drop 1,091 LIRF-override rows from residual train): Jan+Jul 376.04→**372.36**, matched 253.82→**250.98**; December 241.27→**238.01**, matched 226.95→**223.95**. LIRF unmatched RMSE unchanged at 6032.70.
- NEW: **KEEP E18-H.** REJECT unmatched airport mean/median and the unmatched specialist as production heads. The residual tree was being poisoned by LIRF-override rows it never scores, and unmatched LIRF y was leaking into `geo_mean`. Do not apply `MVT−SCHED` to non-LIRF unmatched. Remaining unmatched bomb is still LIRF (E19).

**C14. After E18-H, overall SSE is two unmatched generating processes plus matched tail (E19-A)**

- OLD: remaining unmatched RMSE ~2217 = LIRF unmatched 6033 (~30% SSE) + non-LIRF unmatched ~1550 (~25%), with EHAM/LTFM/EGLL as the non-LIRF story.
- EVIDENCE (E18-H metrics + training holdout, no new fit): unmatched is 55.3% of Jan+Jul SSE. LIRF unmatched 30.3%. **LFPG unmatched 21.8%** — but 97% of that LFPG unmatched SSE is **two easyJet rows** (EJU983W y=84,240 with MVT−SCHED=1,740; EJU42AY y=58,206 with MVT−SCHED=2,043). Implied BLOCK is 16–23 h *before* schedule (wrong-day BLOCK). corr(y, MVT−SCHED) on LFPG unmatched = **−0.02** (full year +0.02). A LIRF-style MVT−SCHED override on LFPG raises overall 372→518. Other unmatched airports together ≈ 3% of all SSE. Matched 251→240 would move overall by only −7 s. LIRF oracle gate 6033→2771: overall −47 s. Fixing the two LFPG bombs: overall −42 s.
- NEW: Structural leftover is (1) LIRF unmatched regimes, (2) a handful of wrong-day BLOCK bombs (LFPG, not a CDG unmatched model), (3) matched y>30 min tail. Do not build a non-LIRF unmatched mean/tree. Do not apply MVT−SCHED outside LIRF. Next experiments: ranking-safe “BLOCK yesterday” gate; LIRF neighbor gate-vs-taxi split.

**C15. Wrong-day BLOCK wrap is an oracle, not a ranking-safe gate (E19-B)**

- OLD (E19-A): two CDG easyJet unmatched rows are 97% of LFPG unmatched SSE; a 24 h wrap `MVT−SCHED+86400` would recover ~42 overall RMSE if gated.
- EVIDENCE: full-year unmatched wrong-day BLOCK = 3 real bombs (EJU×2 Jan LFPG, NJE May LSZH) plus LIRF ITY already under `MVT−SCHED`. Jan+Jul *train* contains none of the CDG bombs. Every train-selectable gate has precision 0. The val-perfect `LFPG∩EJU∩ms<2400` fires on 2 December false positives. Oracle wrap on the two val bombs: Jan+Jul 372.36→**334.39**; December 238.01→238.01 (zero events).
- NEW: **REJECT** a deployable wrap. KEEP the oracle fact (those labels are yesterday’s BLOCK). Do not reopen LFPG unmatched as a model class. Next structural experiment is LIRF neighbour gate-vs-taxi (E19-C).

**C16. LIRF unmatched is not the airport’s surface-delay regime (E19-C)**

- OLD: matched neighbours’ `(MVT−AOBT)/(MVT−SCHED)` should say whether a late unmatched takeoff was a gate hold or a long taxi, gating `geo_mean` vs `MVT−SCHED`.
- EVIDENCE: on LIRF unmatched, corr(neigh_taxi_frac, y>30 min) = −0.02 train / −0.12 val. Q8/Q1 P(y>30) = 0.86 train / 0.66 val (wrong sign, weak). No train threshold on taxi_frac / push_mean / taxi_mean beats always-`MVT−SCHED` by 1 s. Oracle mix still 372→325 overall.
- NEW: **REJECT** neighbour gate-vs-taxi for LIRF unmatched. Keep always-on `MVT−SCHED`. That slice is a join miss, not congestion. Do not sweep neighbour windows.

**C17. LIRF prefix/ADES rates are descriptive, not a contest-RMSE rule (E20)**

- OLD: E13 rejected those keys as exclusive *extreme* triggers; a train-only P(y>30) mixture or reverse (low-p → geo) gate might still recover part of the oracle −47 overall.
- EVIDENCE: soft mix of geo and `MVT−SCHED` raises Jan+Jul overall +11 to +33 (December +15 to +31). A convex combination of a 400 s expert and a 6000 s expert is still thousands of seconds on both halves. Low-p prefix → geo: Jan+Jul −1.74, December −0.87 but extreme RMSE 194→1131.
- NEW: **REJECT** history-prior mix/gate. Keep always-on `MVT−SCHED`. Do not reopen airline/dest/stand as LIRF unmatched splitters. The E13 oracle remains unreachable from prediction-time keys on the row.

**C18. A location shift on MVT−SCHED cannot serve both LIRF unmatched regimes (E21)**

- OLD: 6033 might be a clock bias (mean residual, hour, season) that a train-only additive/slope on the override could cut without gating.
- EVIDENCE: Jan+Jul val n=397, mean r=−3315, median r=−3963, 75% negative. 79% of SSE is y≤30 min (SCHED overpredicts). Worst-1 is only 5.7% of SSE — uniform regime error, not bombs. December median r=**0**, 60/88 rows are the extreme half. Mean/winsor/OLS: Jan+Jul overall −14 to −15 s, December LIRF_u **up**. Median shift −4 s on train, −0.04 overall. Hour/weekday medians overfit and raise December overall +3 to +4 s.
- NEW: **REJECT** all E21 calibrations. Keep raw `MVT−SCHED`. Do not add intercepts on this slice. Same December-kill as E18-A-mean.

**C19. Hourly METAR is a winter residual, not a contest-RMSE replacement (E22)**

- OLD: with no ADS-B ground trajectories in the bundle, public METAR at pushback (fog / precip / low vis / wind) might identify the >30 min tail that clock-based queue proxies (E16-A, E19) missed.
- EVIDENCE: STEP 0 — `data/` is 12 `training_*.parquet` + ranking/submitting; **`air-data/` does not exist**; training schema is 30 flight-list columns (clocks, stand, runway, type, WTC, operator, ADES); no lat/lon or 1-second surface tracks. 2026 eligibility allows extra data if openly accessible/usable, documented, and open-license. Iowa Mesonet ASOS 2025 joined ranking-safe (`AOBT` else `MVT`); miss ≈ 0, median age 28–30 min. Fog matched residual +47 s vs −7 s. E18-H reproduced **372.36 / 250.98**. E22: Jan+Jul **371.81 / 250.11** (−0.56 / −0.87); >30 798.6→796.5; >60 2402→2398; SSE>30 **45.2%→45.3%** (tail not closed). Dec **238.01→229.04** (−8.98), matched 223.95→215.27, >30 716→651. Does not beat E20 **368.03 / 228.45**.
- NEW: **INCONCLUSIVE** as a current-model replacement. METAR is legal and has a real December winter effect; it does not close the Jan+Jul tail and does not beat the ensemble. Readme unchanged. Do not rerun E18-H+METAR. Optional later: stack the same columns into E20 CatBoost / airport experts. Branch A (true queue from trajectories) is unavailable in this data.

**C20. Stacking METAR into E20 experts is a winter-only gain (E23)**

- OLD: E22's December −9 s / >30 716→651 might survive if the same columns go into CatBoost (0.456) and per-airport LGB (0.491), which can learn airport-specific fog; the risk is Jan+Jul overfit on 12 extra columns.
- EVIDENCE: METAR C/D/E vs E20 OOF. Jan+Jul ensemble **368.03→368.07** (+0.04), matched +0.20, >30 **774.8→781.0** (hurt). CatBoost itself worse (+0.82). Airport E −0.36 overall but >30 worse. December **228.69→221.71** (−7.0), >30 672.6→618.5 — the winter tail **survives in the blend**. Airport E: EDDM −22, LFPG −15, EHAM −13, LSZH −12; LEBL/LEMD/LTFM/LIRF ≈ 0. NNLS C/D/E = 0.368/0.075/0.557 (E takes more weight). Frozen E20 weights on METAR experts ≈ same as refit.
- NEW: **INCONCLUSIVE.** Do not replace E20. Ranking is Jan+Jul-like; a December-only win is the wrong split. METAR is a real winter residual, not a contest upgrade. Do not stack it into production. Readme unchanged.

**C22. Floor + log-excess + per-airport is a worse residualisation than P_cal (E25)**

- OLD: E20 plateaued; a one-shot rebuild — hierarchical p10 floor, `log(y − floor)`, Duan smearing, OOF James-Stein encodings, 10–15 min same-runway DEP+ARR congestion, one LightGBM per airport — might beat residual-on-P_cal.
- EVIDENCE: Schema allows full per-airport models (smallest Jan+Jul-train airport LSZH n=112,561). Leakage tests 10/10. Combined architecture Jan+Jul **473.61 / 386.42**, December **311.23 / 300.26** vs E20 **368.03 / 244.76** and **228.45 / 215.90**. Ablation (after the combined result): no_floor 398.54 / 235.02; raw_target **394.01 / 232.83**; pooled 479.79; no_enc 484.32. raw_target matches E20 expert B (direct LGB 394.63). Log-excess smear 1.64–2.44 vs `log(y)` smear ≈ 1.03. `all_sched` unmatched 920. Arrival-crossing counts sit at the bottom of gain.
- NEW: **REJECT** as a production replacement. p10 is not the unimpeded taxi time given `MVT−AOBT`. Do not iterate on `log(y − p10)`. Keep E20. Keep LIRF `MVT−SCHED`. Per-airport experts already exist as E20-E.

**C21. Pinball residual is a different objective, not a different RMSE solution (E24)**

- OLD: E15 closed mean-loss reweighting (Huber / tail-weighted L2 / P(y>30)). Quantile regression learns the conditional distribution (α=0.1/0.5/0.9) and might still help as a point estimate (median / quantile blend) or as a fourth NNLS expert next to E20 C/D/E.
- EVIDENCE: L2 reproduced E18-H 372.36/250.98. Q10 overall 467 (underpredict, mean r=+218). Q50 `<20` 174→162 but `>30` 799→910, overall 381 (Huber shape). Q90 `>30` 799→686 / SSE>30 45%→16% but `<20` 174→340, overall 456 (E15-B shape). Quantile-NNLS: L2 0.73 + Q90 0.27, Q10=Q50=0, blend 378.92 **worse than L2**. E20+Q50 NNLS: **Q50 weight 0**.
- NEW: **REJECT.** Pinball is not E15, but on this feature set it is the same RMSE tradeoff. Do not add a quantile LightGBM to the ensemble. Median is not a better point estimate than L2. Readme unchanged.

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
| E18 | Non-LIRF unmatched specialist | unmatched mean/median/geo splice; unmatched-only tree; matched-only geo + drop LIRF-override from residual train | E16-A + splice / hygiene | Jan+Jul | **372.36** (H) | **250.98** | 155.03 matched | Mean/median/specialist fail December; H wins both splits; LIRF_u still 6033 | KEEP H; REJECT A/B |
| E25 | One-shot structural rebuild | p10 floor, log-excess, OOF JS encodings, 10–15 min rwy congestion, per-airport LGB, Duan smear | one architecture, then ablation | Jan+Jul | **473.61** (raw_target 394.01) | **386.42** (281.36) | 222 / 160 | Floor+log-excess poisons; best ablation = E20 expert B; Dec also loses | REJECT; keep E20 |

December of current best (E18-H): overall **238.01**, matched **223.95**, MAE **145.83** matched. Direct LGB + override remains a Dec sanity check (241 / 225 on the pre-H pipeline).

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

### E18 — Non-LIRF unmatched specialist

- **Question:** After the LIRF override, can a dedicated non-LIRF unmatched head (airport unmatched mean, or a small tree on ranking-present fields) move overall RMSE? Does residual-train hygiene (matched-only `geo_mean`, drop LIRF-override rows) help?
- **Frozen:** E16-A path for A/B splices. H is allowed to refit.
- **Validation:** Jan+Jul 2025 holdout; December stress. `training_*.parquet` only.
- **Reproduction:** E18-0 = E16-A: Jan+Jul **376.04 / 253.82** (delta 0.00).

**Jan+Jul overall / matched / unmatched / non-LIRF unmatched**

| Variant | Overall | Matched | Unmatched | Non-LIRF u |
|---|---:|---:|---:|---:|
| E18-0 | 376.04 | 253.82 | 2235.88 | 1579.38 |
| A-geo | 375.29 | 253.82 | 2227.72 | 1566.87 |
| A-mean | 376.22 | 253.82 | 2237.76 | 1582.25 |
| A-med | 376.58 | 253.82 | 2241.63 | 1588.15 |
| **H** | **372.36** | **250.98** | **2216.59** | **1549.74** |
| B specialist | 374.45 | 253.82 | 2218.62 | 1552.88 |

December: only H improves (241.27→**238.01**, matched 226.95→**223.95**). A/B raise December overall.

Jan+Jul EHAM unmatched mean y=742 vs train unmatched mean 565 — full-year “EHAM shorter” is not a stable prior. LSZH unmatched: dropping the residual 1017→640 (tree was overpredicting). LFPG unmatched ~3444 on every variant. LIRF unmatched frozen at 6032.70.

- **Decision:** KEEP H. REJECT A-mean/A-med/A-geo/B. C13 recorded. Artifacts: `analysis/E18/`.

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
| Additive/OLS calibration of LIRF `MVT−SCHED` (E21) | Mean/winsor/OLS −14 s Jan+Jul overall, December LIRF_u up. Median shift ≈ 0. Hour/dow overfit (E21). |
| E18-H + METAR as current-model replacement (E22) | Jan+Jul −0.56 vs E18-H, 371.81 vs E20 368.03; Jan+Jul tail SSE share unchanged. December −9 s is real but not enough to replace the ensemble. |
| METAR stacked into E20 C/D/E (E23) | Jan+Jul 368.03→368.07, >30 775→781 (hurt). December −7 s and >30 673→619 survive in the blend. Winter-only; ranking analogue is Jan+Jul. |
| Quantile residual LGB as point estimate or 4th expert (E24) | Q50 381 vs L2 372 (bulk yes, tail no). Q90 tail yes, bulk no. NNLS zeros Q50 next to E20 C/D/E. Same RMSE tradeoff as E15. |
| `FLIGHT_ID` as aircraft/turnaround | It is a city-pair NM flight; paired ARR is after DEP (E7). |
| LightGBM without LIRF unmatched override | Matched improves; overall still ~448–468 vs 410 with the simple rule (E12). |
| Gate `MVT−SCHED` on LIRF unmatched via threshold/stand/hour/dest/prefix | Two y-regimes exist, but `MVT−SCHED` is large in both; gates do not generalize (E13). |
| Huber residual (E15-A) | Improves <20 min RMSE, worsens >30/>60 and matched RMSE. Robust loss down-weights the tail we need (E15). |
| Tail-weighted residual (E15-B) | >30/>60 RMSE fall; bulk overpredicted (mean residual −120 s); matched RMSE 256→300 (E15). |
| Two-stage P(y>30)+tail residual (E15-C) | Best tail RMSE, worst matched/overall; gate fires at 14% vs true 4.5%; December unmatched 886→1365 (E15). |
| Non-LIRF unmatched → train airport mean/median (E18-A) | Jan+Jul EHAM unmatched mean y=742 vs train unmatched mean 565; overall RMSE up on both splits (E18). |
| Non-LIRF unmatched specialist tree (E18-B) | Jan+Jul overall −1.60; December +0.53 (EDDM unmatched mean jumps to 1687). Does not generalize (E18). |
| Drop residual on non-LIRF unmatched (E18-A-geo) | Helps LSZH 1017→640; December unmatched 573→633, overall +1.40 (E18). |

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
- Fit `geo_mean` on **matched** train rows only. Drop LIRF-override rows from residual training (E18-H). Do not splice airport unmatched means. LIRF override remains `unmatched and airport==LIRF` (not `type_null`).

---

## Current best model

**Name:** E9 residual LightGBM + E11 LIRF unmatched override + E16-A disruption state + E18-H hygiene.

```
geo_mean tables fit on matched train rows only
P_cal = per-airport OLS(mvt_aobt, aobt_eobt, geo_mean)   # unmatched → geo_mean
residual LightGBM trained with LIRF-override rows dropped
if unmatched and airport==LIRF:
    y_hat = MVT - SCHED
else:
    y_hat = P_cal + LightGBM_residual_L2(clocks, geo, stand, dest, airline,
                                         dis_state_30m, dis_frac20_30m, ...)
```

| Split | Overall RMSE | MAE | Matched RMSE | Unmatched RMSE | LIRF | EGLL | LFPG | LTFM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan+Jul **E18-H** | **372.36** | 155.03 matched | **250.98** | 2217 | — | — | — | — |
| Jan+Jul E16-A residual+override | 376.04 | 157.46 matched | 253.82 | 2236 | — | — | — | — |
| Jan+Jul residual+override (no disruption) | 378.28 | 165.09 | 256.46 | 2241 | 869 | 285 | 585 | 277 |
| Jan+Jul E3+override (no tree) | 410.02 | 189.85 | 303.53 | 2228 | 909 | 360 | 612 | 319 |
| Jan+Jul E3 only | 576.63 | 191.04 | 303.53 | 3937 | 1721 | 360 | 612 | 319 |
| Dec **E18-H** | **238.01** | 145.83 matched | **223.95** | 838 | — | — | — | — |
| Dec E16-A residual+override | 241.27 | 148.10 matched | 226.95 | 852 | — | — | — | — |
| Dec residual+override (no disruption) | 245.38 | 153.32 | 230.04 | 886 | 405 | 239 | 285 | 252 |
| Dec direct LGB+override | 240.79 | 150.09 | 225.29 | 881 | 403 | 244 | 281 | 245 |

Vs E3-only: Jan+Jul overall 577 → 372, matched 304 → 251.  
Most of the overall drop is still the LIRF unmatched rule (577 → 410), then residual trees, then E16-A + E18-H.

E18-H does not change the LIRF override (RMSE 6032.70 on 397 Jan+Jul rows). Non-LIRF unmatched 1579 → 1550.

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
9. **E19-A located remaining SSE.** After E18-H: LIRF unmatched 30.3% of all SSE; LFPG unmatched 21.8% of all SSE but **2 rows = 97% of that LFPG slice** (wrong-day BLOCK, not a CDG unmatched DGP); all other unmatched airports ~3%; matched 44.7%.
9d. **E19-B closed the wrap.** Oracle 24 h wrap on the two CDG bombs: Jan+Jul 372→334. No train-safe gate; December FPs. REJECT deployable wrap.
9e. **E19-C closed neighbour gate-vs-taxi.** LIRF unmatched corr with neighbour taxi fraction ≈ 0. No train threshold beats always-SCHED. Oracle 372→325 still unreachable. Keep always-on `MVT−SCHED`.
9f. **E20 closed LIRF history priors.** Soft mix of geo and SCHED wrecks RMSE. Low-p→geo is ~2 s and hurts December extremes. Always-on `MVT−SCHED` stays.
9g. **E21 closed LIRF override calibration.** 397 rows = 0.115% of holdout and 31% of E20 SSE. Mean/OLS look like −14 s Jan+Jul and fail December (different regime mix). Median does nothing. Keep raw `MVT−SCHED`.
9h. **E22 closed METAR as a standalone replacement.** No ADS-B trajectories in the bundle (Branch B). Hourly Iowa Mesonet ASOS is ranking-safe and legal. Helps December (−9 s), not the Jan+Jul >30 min tail. Does not beat E20. Optional: stack the same columns into E20 experts, not another E18-H residual-only run.
9i. **E23 closed METAR-into-E20 stacking.** Airport experts absorb winter weather (EDDM/LFPG/EHAM/LSZH). Jan+Jul ensemble is flat-to-worse and the >30 tail **hurts**. Do not ship. Ranking-analogue split is Jan+Jul.
9j. **E24 closed pinball/quantile residual.** Mechanistically not E15 (conditional distribution vs reweighted mean). RMSE tradeoff is the same. Q50 gets zero NNLS weight next to E20. Do not add.
9i. **Temporal v2 TCN closed as an E20 replacement.** Ordered 64-step TCN on ranking-safe movement history is a real (small) matched-holdout signal (−3.1 s) but full residual overshoots, the 0.9 blend cheated via LIRF unmatched, and **leaderboard 317.47 lost to E20 316.97**. Keep E20. Do not add more rolling features. Next sequence work must fix OOF-E20 strength mismatch and not select shrink on the same Jan+Jul val.
9b. **E18-H kept as a bundle.** Matched-only `geo_mean` and dropping LIRF-override rows from residual train were not ablated separately. Optional micro-ablation before treating them as atomic.
9c. Do not auto-stack E17-A1 airport memory on E18-H without a new experiment.

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
- **2026-09-07 E18:** Non-LIRF unmatched specialist. E16-A reproduced 376.04/253.82. Airport unmatched mean/median and an unmatched-only tree fail December. Hygiene (matched-only `geo_mean` + drop LIRF-override rows from residual train): Jan+Jul **372.36 / 250.98**, December **238.01 / 223.95**. KEEP H. REJECT A/B. C13 recorded. Override formula in the journal corrected to `unmatched and airport==LIRF` (matches code). Artifacts in `analysis/E18/`.
- **2026-09-07 E19-A:** Structural SSE after E18-H. Unmatched 55% of overall SSE. LIRF unmatched 30.3%. LFPG unmatched 21.8% is two easyJet wrong-day BLOCK rows (97% of that slice SSE); corr(y, MVT−SCHED)=0. Do not LIRF-rule LFPG. Matched 1% features move overall ~7 s. C14 recorded. Artifacts in `analysis/E19A/`.
- **2026-09-07 E19-B:** Wrong-day BLOCK wrap. Full year: 3 real non-LIRF unmatched bombs. No train-safe gate (CDG bombs are in Jan val; Dec gate FPs). Oracle wrap Jan+Jul 372.36→334.39, Dec unchanged. REJECT rule. C15 recorded. Artifacts in `analysis/E19B/`.
- **2026-09-07 E19-C:** LIRF neighbour gate-vs-taxi. Neighbour taxi fraction does not identify y>30 on unmatched LIRF (corr −0.12 val). No train threshold beats always-SCHED. Oracle mix 372→325. REJECT. C16 recorded. Artifacts in `analysis/E19C/`.
- **2026-09-07 E20:** LIRF prefix/ADES/flight-number prior. Soft mix +11 to +33 overall (wrong functional form). Low-p→geo −1.74 / −0.87 with December extreme damage. REJECT. C17 recorded. Artifacts in `analysis/E20/`.
- **2026-09-07 leaderboard v4:** E20 NNLS ensemble (`likable-eagle_v4.parquet`) scored **316.9654** RMSE on ranking (`truthing.parquet`, 344,841 pairs). Previous ≈ 323. Confirms the ensemble moved the official metric, not only the holdout. Journal PENDING cleared.
- **2026-09-07 E21:** LIRF unmatched override calibration. Diagnosis: 6033 is the normal half (79% of SSE), not bombs; December median residual is 0. Mean/OLS −14 s Jan+Jul, December worse. Median ≈ 0. REJECT. Keep raw `MVT−SCHED`. C18 recorded. Artifacts in `analysis/E21/`. Current model unchanged.
- **2026-09-07 E22:** STEP 0: no `air-data/`, 30 flight-list columns, no ADS-B trajectories; 2026 rules allow documented open extra data. Branch B METAR (Iowa Mesonet ASOS). E18-H reproduced 372.36/250.98. E22 Jan+Jul **371.81 / 250.11** (tail SSE share unchanged); Dec **229.04 / 215.27**. Does not beat E20 368.03/228.45. **INCONCLUSIVE.** C19 recorded. Readme unchanged. Artifacts in `analysis/E22/`.
- **2026-09-08 E23:** METAR stacked into E20 C/D/E (not a fourth model). Jan+Jul **368.03→368.07**, >30 775→781 (hurt). Dec **228.69→221.71**, >30 673→619 (winter tail survives in the blend). NNLS 0.368/0.075/0.557. **INCONCLUSIVE.** C20 recorded. Production stays E20. Readme unchanged. Artifacts in `analysis/E23/`.
- **2026-09-08 E24:** Quantile residual LGB α=0.1/0.5/0.9 on frozen E18-H features. L2 reproduced 372.36. Q50 380.90 (bulk better, tail worse). Q90 tail better, bulk wrecked. E20+Q50 NNLS **Q50 weight 0**. **REJECT.** C21 recorded. Readme unchanged. Artifacts in `analysis/E24/`.
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

---

## E19 — RMSE tail reduction (causal local queue state) (2026-09-07)

**Hypothesis:** the residual error is dominated by large positive residuals (~45% of matched SSE >30 min); the model under-estimates the local operational queue. Reconstruct a causal local queue/state from flights strictly before t (no TAXITIME, ranking-safe) and add it to the E18-H residual (identical L2, seed, early stopping).

**Feature families (all strictly `< t`):** A neighbour state (prev-1/2/3/5/10 dep gaps; rolling AOBT−EOBT/MVT−AOBT/MVT−SCHED; same-runway/airline/type flags; delayed counts; consec runs); B same-runway queue (gaps, counts, rolling delay stats); C delay shock (`recent − expected`, climatology fit on train split: airport×hour / airport×runway); D queue/arrival pressure (2–30m dep/rwy/arr counts, delayed-arrival counts, inverse gaps, gap compression); E interactions (queue-pressure × shock × disruption); F tail-aware products.

**Baseline reproduction:** E19-0 = E18-H refit → Jan+Jul **372.36 / 250.98** (Δ 0.00), Dec 238.01 / 223.95. Exact.

**Results (Jan+Jul overall / matched; Dec overall):**

| Variant | Jan+Jul | matched | Dec |
|---|---:|---:|---:|
| E19-0 (E18-H) | 372.36 | 250.98 | 238.01 |
| E19-A neighbour | 371.99 | 250.18 | 235.39 |
| E19-B same-runway queue | 371.14 | 249.35 | 233.67 |
| E19-C shock | 372.22 | 250.64 | 236.40 |
| E19-D pressure | 371.95 | 250.24 | 236.12 |
| E19-E full state | 371.05 | 249.09 | 233.69 |
| **E19-F + tail-aware (best)** | **371.02** | 249.22 | **233.67** |

**Tail verdict: E19 did NOT reduce the extreme tail.** Positive >300 s SSE share and top-1%/top-5% SSE shares are unchanged or marginally *worse* (top-1% 72.25%→72.73%); the Jan+Jul gain (only −1.34 s overall, −1.76 matched) comes from the mid-range/bulk. Gains are consistent across airports except EGLL (tail +8 s). December improvement is larger (−4.34 s) than Jan+Jul.

**Decision: WEAK — not accepted for submission.** The 1–3 s Jan+Jul band is below the meaningful bar, and the core hypothesis (local queue state reduces the large-positive tail) is falsified: better rolling state representations do not touch the >30 min flights (same conclusion as E16-A/E17-A — tail rows are not identified by any causal rolling statistic of delay/clocks). Do not keep adding rolling-statistic features.

**Recommendation for next experiment:** the next candidate must be a genuine temporal/sequence representation of the queue (e.g., position-in-queue / departure-burst decomposition with explicit service-rate structure), not more rolling statistics; or attack the >60 min rows as a rare-event regime via the E11 gate idea revisited with the E18-H trainer. Artifacts: `experiments/e19_features.py`, `experiments/run_e19_rmse_tail.py`, `experiments/results/E19/` (summary.md, metrics.csv, ablation.csv, per_airport.csv, tail_metrics.csv, feature_importance.csv, E19.json, plots/). No submission generated.

---

## E20 — Stacked residual ensemble (2026-09-07)

**Objective:** break the single-model ceiling by blending genuinely different experts. E18-H hygiene everywhere (matched-only geo_mean, LIRF-override rows excluded from every expert fit, always-on LIRF `MVT−SCHED`).

**Experts (ranking-safe causal matrix; Jan+Jul / Dec overall, matched):**

| Expert | Jan+Jul | matched | Dec | matched |
|---|---:|---:|---:|---:|
| A LGB residual (E18-H, reproduced Δ0.00) | 372.36 | 250.98 | 238.01 | 223.95 |
| A2 A + E19-B same-runway | 371.14 | 249.35 | 233.67 | 220.11 |
| B LGB direct TAXITIME (P_cal as feature) | 394.63 | 283.08 | 237.51 | 223.62 |
| C CatBoost residual | 370.37 | 248.25 | 232.37 | 218.56 |
| D XGBoost residual (numeric) | 385.50 | 266.24 | 246.17 | 232.70 |
| E airport-specific residual LGB | 370.52 | 247.76 | 229.94 | 215.90 |

CatBoost and airport experts both beat E18-H alone; the direct-LGB and numeric-XGB experts are individually weaker but add diversity.

**Blends (weights fit on Jan+Jul holdout, NNLS/Ridge constrained ≥0, sum=1):**

| Blend | Jan+Jul | Δ vs E18-H | matched | Dec | Δ Dec |
|---|---:|---:|---:|---:|---:|
| equal5 | 371.99 | −0.38 | 250.86 | 232.00 | −6.01 |
| A_E | 369.99 | −2.37 | 247.49 | 233.50 | −4.51 |
| **nnls** | **368.03** | **−4.33** | **244.76** | **228.45** | **−9.56** |
| ridge1 / ridge10 | 368.17 | −4.19 | 244.93 | 229.60 | −8.41 |

**NNLS weights: A=0, B=0, C=0.456, D=0.053, E=0.491** — the optimizer drops the LightGBM E18-H entirely in favour of CatBoost (C) + airport experts (E); ridge gives the same picture. Improvement is not a Jan+Jul artifact: December improves by −9.56 s (weights were fit only on Jan+Jul). Matched RMSE drops to 244.76 (Jan+Jul) / 215.90 (Dec) — the best matched result so far.

**Decision: ACCEPT.** Meaningful (4.3 s Jan+Jul, 9.6 s Dec), consistent, no December regression.

**Leaderboard (2026 ranking, `truthing.parquet`, 344,841 pairs):** `likable-eagle_v4.parquet` RMSE **316.9654**. Previous benchmark ≈ 323. Internal Jan+Jul 368.03 vs LB 317 is the same ~50 s “ranking is easier than Jan+Jul holdout” offset seen for E16-A/E18-H (~372 internal vs ~323 LB). December holdout 228.45 is *optimistic* relative to the leaderboard. Gap to reported best 265: **52 points**.

**Residual correlation (Jan+Jul matched):** expert residuals are positively but not perfectly correlated (A–C ~0.97, A–E ~0.96, C–E ~0.98) — the gain comes from C and E each being individually stronger and slightly different, not from low-correlation averaging of equals. See `experiments/results/E20/error_correlation.csv`.

**Artifacts:** `experiments/run_e20_ensemble.py`, `experiments/make_submission_e20.py`, `experiments/results/E20/` (summary.md, model_metrics.csv, blend_metrics.csv, error_correlation.csv, regime_metrics.csv, feature_importance.csv, E20.json, oof_predictions_{janjul,dec}.parquet, plots/). Submission: `likable-eagle_v4.parquet` (fit on all 2025 training months). **Leaderboard RMSE: 316.9654** (`used_pairs` 344,841; scorer `truthing.parquet`).

---

## E22 — METAR weather, Branch B (2026-09-07)

**STEP 0.** `data/` has 12 `training_*.parquet`, `ranking.parquet`, `submitting.parquet`. **`air-data/` does not exist.** Training schema is 30 columns: movement clocks (`MVT`/`BLOCK`/`SCHED`), stand, runway, aircraft type, plus NM (`AOBT`/`EOBT`/`IOBT`/`LOBT`, WTC, operator, ADES, callsign). No lat/lon, taxiway, or 1-second ADS-B surface trajectories. 2026 eligibility (prc-data-challenge-2026.netlify.app/eligibility.html): extra data allowed if openly accessible/usable and documented, and additional datasets under an open license. **Branch B.**

**Join.** Iowa Mesonet ASOS 2025, nearest report with `valid ≤ AOBT` (else `MVT`), stale >3 h nulled. Features: vis/wind/gust/temp/RH/precip/age + fog/precip/low-vis/strong-wind flags. Frozen E18-H residual LightGBM. Training files only.

**Coverage.** 87,589 hourly reports, 10/10 airports. Miss ≈ 0. Median age 28–30 min. Fog 0.3–4.9% of DEPs; precip 5–17%.

**Holdout (E18-H reproduced exactly):**

| Split | Model | Overall | Matched | >30 RMSE | >60 RMSE | SSE>30 | SSE>60 |
|---|---|---:|---:|---:|---:|---:|---:|
| Jan+Jul | E18-H | 372.36 | 250.98 | 798.6 | 2402.4 | 45.2% | 20.9% |
| Jan+Jul | E22 METAR | 371.81 | 250.11 | 796.5 | 2398.3 | 45.3% | 21.0% |
| Jan+Jul | E20 ensemble | **368.03** | **244.76** | — | — | — | — |
| Dec | E18-H | 238.01 | 223.95 | 716.2 | 2114.6 | 39.6% | 10.3% |
| Dec | E22 METAR | 229.04 | 215.27 | 651.1 | 2038.3 | 35.4% | 10.4% |
| Dec | E20 ensemble | **228.45** | 215.90 | — | — | — | — |

Fog matched residual +47 s (signal exists). Trees use temp/RH/precip, not the fog flag. Jan+Jul tail **not** reduced. December winter −9 s is real.

**Decision: INCONCLUSIVE.** Does not beat E20 on both splits. Readme current-model unchanged. Optional later stack into E20 experts; do not reopen E15/E19. Artifacts: `experiments/run_e22_metar.py`, `experiments/download_metar.py`, `analysis/E22/`, `data/external/metar/`.

---

## E23 — METAR stacked into E20 experts (2026-09-08)

**Question.** E22's December winter gain might survive if the same METAR columns go into the three experts that actually carry E20 weight (CatBoost 0.456, airport LGB 0.491, XGB 0.053), not as a fourth model. Risk: Jan+Jul overfit.

**Setup.** E22 join. E18-H hygiene. No-wx C/D/E = E20 OOF (not retrained). METAR C/D/E fit here. Frozen weights vs NNLS refit on Jan+Jul. CatBoost on GPU; D/E on CPU.

**Experts (overall / >30 min):**

| Expert | Jan+Jul no-wx | Jan+Jul +wx | Dec no-wx | Dec +wx |
|---|---:|---:|---:|---:|
| C CatBoost | 370.37 / 782.6 | 371.19 / 792.5 | 232.37 / 689.7 | 225.49 / 635.6 |
| D XGB | 385.50 / 852.9 | 385.00 / 846.8 | 246.17 / 738.2 | 236.86 / 682.4 |
| E airport | 370.52 / 781.7 | 370.16 / 786.0 | 229.94 / 665.2 | 224.03 / 615.2 |

Airport E December matched: EDDM −22, LFPG −15, EHAM −13, LSZH −12; LEBL/LEMD/LTFM/LIRF ≈ 0.

**Blends:**

| Split | Model | Overall | Matched | >30 | SSE>30 |
|---|---|---:|---:|---:|---:|
| Jan+Jul | E20 frozen OOF | **368.03** | **244.76** | **774.8** | 44.7% |
| Jan+Jul | METAR + frozen w | 368.12 | 244.99 | 781.3 | 45.4% |
| Jan+Jul | METAR + NNLS 0.368/0.075/0.557 | 368.07 | 244.96 | 781.0 | 45.4% |
| Dec | E20 frozen OOF | 228.69 | 214.78 | 672.6 | 37.9% |
| Dec | METAR + frozen w | 221.78 | 208.00 | 619.4 | 34.3% |
| Dec | METAR + NNLS | **221.71** | **207.93** | **618.5** | 34.2% |

December >30 gain **survives in the ensemble**. January+July tail **hurts**. Ranking is the Jan+Jul analogue (LB 317 vs holdout 368).

**Decision: INCONCLUSIVE.** Do not replace E20. Readme unchanged. C20 recorded. Artifacts: `experiments/run_e23_metar_ensemble.py`, `analysis/E23/`.

---

## E24 — Quantile residual LightGBM (2026-09-08)

**Question.** E15 closed mean-loss reweighting. Pinball (α=0.1/0.5/0.9) learns the conditional distribution. Median or a quantile blend might be a better point estimate, or Q50 might take NNLS weight as a fourth E20 expert.

**Setup.** Frozen E18-H features. No extra data. L2 must reproduce 372.36. Quantile LGB on `y − P_cal`. NNLS of {L2,Q10,Q50,Q90} and of {C,D,E,Q50}.

| Model | Jan+Jul | matched | <20 | >30 | Dec |
|---|---:|---:|---:|---:|---:|
| L2 (E18-H) | **372.36** | **250.98** | 174.0 | 798.6 | 238.01 |
| Q10 | 467.25 | 376.26 | 193.4 | 1313.5 | 351.25 |
| Q50 median | 380.90 | 263.01 | **162.4** | 910.0 | 246.79 |
| Q90 | 456.33 | 363.76 | 339.6 | **686.4** | 317.30 |
| Qblend L2+Q90 | 378.92 | 260.79 | 207.5 | 738.0 | 238.43 |
| E20 | **368.03** | **244.76** | 170.9 | 774.8 | 228.69 |
| E20+Q50 | 368.03 | 244.76 | 170.9 | 774.8 | 228.69 |

E20+Q50 NNLS: **Q50 weight 0**. Quantile-family NNLS: L2 0.73 + Q90 0.27, zeros Q10/Q50.

**Decision: REJECT.** Same RMSE tradeoff as E15 (median=Huber shape, Q90=tail-weight shape). Not a useful fourth expert. C21 recorded. Readme unchanged. Artifacts: `experiments/run_e24_quantile.py`, `analysis/E24/`.

---

## Temporal v2 — learned airport-state TCN (2026-09-08)

**Objective:** replace rolling-stat stacking with an ordered movement-sequence encoder. E20 stays the incumbent. TCN predicts a residual correction from the previous 64 same-airport DEP/ARR movements (ranking-safe clocks only: no TAXITIME, no BLOCK).

**Pipeline (leakage-checked):** per-airport packed stream, strictly `event_time < t`, train-only vocabs/scalers. Causal expanding-window E20 OOF on train months excluding Jan+Jul (lite C+E, 1,257,159 rows; fold RMSE 330 / 285 / 242). Jan+Jul val uses the frozen E20 OOF parquet (368.03 reproduced). LIRF unmatched frozen to `MVT−SCHED`.

**Jan+Jul 2025 holdout:**

| Model | Overall | Matched | MAE | >30m matched | >60m matched | LIRF |
|---|---:|---:|---:|---:|---:|---:|
| E20 | **368.03** | 244.76 | 157.05 | 774.8 | 2361.4 | 857.42 |
| temporal-alone | 539.86 | 275.69 | 155.74 | 992.7 | 3557.6 | 1661.37 |
| 0.9 E20 + 0.1 temporal | 365.20 | 244.58 | 155.48 | 786.0 | 2448.0 | 849.06 |
| E20 + 1.0 TCN corr | 369.71 | 247.15 | 156.31 | 797.2 | 2335.6 | 860.27 |
| **E20 + 0.5 TCN corr** | **365.84** | **241.65** | 154.43 | 773.3 | 2326.3 | 856.47 |

January −1.77 / July −2.52 for 0.5-corr (not a one-month artifact). Matched gains: LSZH −7.4, LTFM −8.3, EDDF −6.0; EHAM +2.2. Tail SSE share not reduced.

**Do not use the 0.9 blend.** Freezing LIRF unmatched back to E20 erases most of its overall gain (365.20 → 367.83). That blend was quietly editing the 397 LIRF-override rows.

**Leaderboard:** `likable-eagle_v5.parquet` = E20 v4 + 0.5 TCN correction, LIRF fallback preserved, 344,841 pairs. **RMSE 317.4747** vs E20 v4 **316.9654** (Δ **+0.51**, worse). Local −2.2 s did not transfer.

**Decision: REJECT as a replacement.** Sequence model has a small matched-holdout signal and a negative leaderboard delta. Production stays E20 (`likable-eagle_v4.parquet`, 316.9654). Full residual overshoots because train OOF E20 (lite, easier months) is weaker than the val/ranking E20 — TCN learns oversized corrections.

**Artifacts:** `openair/models/temporal_v2/`, `openair/models/e20_baseline/fit.py`, `experiments/temporal_v2/`, `experiments/results/temporal_v2/`, `likable-eagle_v5.parquet`.

---

## E27 — AOBT off-block anchor audit (2026-09-09)

**Question.** AOBT_3_flt (NM actual off-block) is a same-flight timestamp, present at prediction time. Does `MVT − AOBT` reconstruct the hidden BLOCK_TIME/TAXITIME well enough to replace E20? No submission; E20 frozen.

**Ranking coverage (descriptive peek only):** AOBT/LOBT/IOBT/EOBT present for **98.47%** of ranking DEP (missing = exactly the 5,290 unmatched rows; LIRF DEP 98.6%). SCHED 100%.

**Phase 1 — delta = BLOCK − AOBT (matched, all training):** exact 0.65%; median Δ +51 s; MAE 238 s; RMSE 385 s; p90 abs 513 s; p99 1321 s; max 20,657 s. Airport-dependent bias (LTFM +296 s, LIRF −118 s). AOBT and BLOCK are genuinely different off-block sources (NM vs airport), not interchangeable.

**Phase 2 — raw anchors (Jan+Jul matched):** AOBT **428.17** (best); LOBT 816.96; IOBT 816.80; EOBT 745.66; SCHED 2472. AOBT dominates every other anchor.

**Phase 3 — calibrated delta (small LGB):** matched Jan+Jul **254.03**, Dec 226.85; gt30 780.6. Close to but NOT better than E20 (244.76 / 214.78). E20's tree stack already consumes `mvt_aobt` + geometry + disruption state; the calibration adds nothing beyond what E20 extracts.

**Phase 5 — hybrid (AOBT_cal matched + E20 unmatched):** Jan+Jul overall 374.17 vs E20 368.03 — E20 wins.

**Phase 4 — NEW tail signal:** |AOBT−EOBT/IOBT/LOBT| (same-flight source disagreement) correlates 0.29 with E20 |residual|; >30 min tail rate rises 3.1% → 16.8% from below to above p90 disagreement. Off-block source disagreement is a **tail-indicator** (usable at prediction time; AOBT/EOBT/IOBT/LOBT all known before MVT).

**Phase 6 — leakage:** AOBT < MVT for 99.95% of matched rows (1,012/2.06M violations, ~glitches); strictly causal otherwise.

**Decision: hypothesis RESOLVED — AOBT is a near-universal strong anchor, not a replacement.** Raw AOBT matched 428, calibrated 254; E20 244.76 stays. Do NOT rebuild around AOBT alone. Actionable leftover: Phase-4 source-disagreement as a tail/regime feature is new information for a future tail model. C23 recorded. Artifacts: `experiments/run_e27_aobt_audit.py`, `experiments/results/E27/` (summary.md, E27.json).

---

## E28 STEP 1 — Unmatched error forensics (C24, 2026-09-10)

**Mechanism.** Every unmatched DEP (5,373 Jan+Jul / 1,649 Dec; 22,470 in training) has **all NM flight-table fields null** — `AOBT_3_flt`, `AIRCRAFT_OPERATOR_flt`, `ADES_FILED_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`. Unmatched = a **total NM flight-record miss**, not a fuzzy/route/schedule mismatch (`partial_NM_match` = 0). Movement-side fields (ADEP/ADES/FLIGHT/SCHED/MVT/runway/stand/type) are present.

**SSE decomposition (Jan+Jul, E20 OOF):** total 4.665e10; matched 2.031e10 (43.5%); **unmatched 2.634e10 (56.5%) from 1.56% of rows**. Unmatched RMSE 2214 = LIRF 6033 (n=397, ~30% of total SSE) + non-LIRF 1546 (n=4976, ~25%). Dec: unmatched only 12.7% of SSE (LIRF 2786, non-LIRF 516).

**Concentration:** top-10 SSE rows = **32% of total SSE** (7 of 10 are LIRF unmatched); top-50 = 42%; top-100 = 48.5%. A few dozen LIRF gate-delay bombs dominate the leaderboard metric.

**Ceilings (Jan+Jul):** perfect non-LIRF unmatched → 317.7; perfect LIRF unmatched → 305.8; perfect all unmatched → **242.8**. Unmatched is the single largest lever; LIRF unmatched is the crux.

**Recovery preview (causal movement/service profiles, train-only):** non-LIRF unmatched service-median 1583 (vs E20 1546) — no gain, Dec worse (667 vs 516); LIRF service-median 13391 vs E20 `MVT−SCHED` 6033 — far worse. Naive service recovery is NOT the answer; LIRF rows lack all off-block anchors so E27 source-disagreement is unavailable for them.

**Implication for STEP 2/3:** non-LIRF unmatched is already near its practical floor with E20; the only material unmatched lever is the LIRF gate-delay bombs, which require separating "normal taxi" from "multi-hour gate delay" using movement-only signals at prediction time (E13/E21 showed naive gating/calibration fails — must find a new causal discriminator). C24 recorded. Artifacts: `experiments/run_e28_unmatched_audit.py`, `experiments/results/E28/`.

## C25 — Unmatched Recovery (pending STEP 2)
## C26 — Tail Regime (pending STEP 3)
## C27 — Matched Surface Interaction (pending STEP 4)
## C28 — Analog/Service Correction (pending STEP 5)
## C29 — Final Gated E28 (pending STEP 6)

---

## E28 STEP 3 — Matched tail regime (C25, 2026-09-10)

**Setup.** Matched rows only (LIRF unmatched excluded, separate mission). Model E20 residual = y − e20; add off-block source disagreement (E27) + compact causal surface pressure (dep/rwy/arr counts 5/10/30m, time-since, burstiness, ratio); risk classifier P(|resid|>30m); residual correction; gate threshold on P(tail). E20 immutable. **Caveat:** E20 OOF exists only for holdout months, so tail models were fit by temporal cross-fit (Jan↔Jul; Jan+Jul→Dec; Dec→Jan+Jul) — row-level out-of-sample, but fit periods are holdout months (a fully clean train→val fit needs a train-month E20 OOF).

**Matched tail scale (Jan+Jul):** |resid|>15m 2,592 rows; >20m 1,284; >30m **464**; >45m 143; >60m 58. Matched >30m SSE = 4.83e9 = **10.4% of total SSE** (matched SSE 2.03e10).

**Diagnostic (STEP 3, valuable):** source disagreement strongly stratifies E20 error. Max-disagreement q5: RMSE 906, tail>30m rate 5.3% vs q0 RMSE 188, 0.0% (E27 confirmed). Disagreement × surface interaction exists but is not exploitable safely.

**Ablations (Jan+Jul matched RMSE):** A E20 **244.76**; B +disagreement (ungated) **689**; C +surface (ungated) **698**; D +both (ungated) **691**; E gated **244.65** (gate 0.55). December: A 214.78, E gated 214.92 (slightly worse), extreme regime 1645→1822.

**Regimes (Jan+Jul):** normal 339,012 rows unchanged (244.1→244.1); hard (27 rows) 1720→1630; extreme (7 rows) 2063→1698. December extreme reversed (1645→1822).

**Decision: REJECT.** Ungated residual corrections destroy normal flights (matched RMSE ~690); the risk gate is inert (selects ~34/38 rows, ΔSSE ≈ 0.1%) because P(tail) rarely exceeds 0.55 at the useful operating point; and the only regime that improves on Jan+Jul (extreme) regresses on December. No meaningful total-SSE reduction. Matched tail is only ~10% of total SSE, so even a perfect fix caps at ~348 overall — the dominant lever remains unmatched (56%), specifically the LIRF gate-delay bombs. Keep E20. C25 recorded. Artifacts: `experiments/run_e28_tail_regime.py`, `experiments/results/E28/summary_tail_regime.md`, `E28_tail_regime.json`.

---

## E29 — 260 RMSE attack (C26–C30, 2026-09-10)

**Budget/calibration:** internal Jan+Jul 368.03 ↔ LB 316.9654 ⇒ **offset 51.1 s**, so LB 260 ⇒ internal ≈311. Required total SSE 3.331e10 (cut 29%). With matched 245 that needs **LIRF unmatched RMSE 6033 → ~1643** (73% cut).

**C26 Information mask (movement-only LGB, train months):** overall 523.8 vs E20 368.0; matched 360.7 vs 244.8; unmatched 3061.7 vs 2214.1; LIRF-u 9814.8 vs 6032.7; non-LIRF-u 1560.9 vs 1545.9. NM information is essential; movement-only is weaker everywhere and never beats E20.

**C27 Movement-only (LightGBM/XGB, movement fields only):** no unmatched improvement; LIRF-u 9815 (worse); non-LIRF-u 1561 ≈ E20 1546.

**C28 LIRF forensics:** true TAXITIME median 1447 s vs MVT−SCHED median 6781 s — E20's override predicts schedule displacement (gate delay), not taxi. Largest SSE rows = normal taxi + huge MVT−SCHED; corr(|resid|, MVT−SCHED)=0.04 ⇒ the gate-delay component is unobservable from movement-only data. 35% of LIRF-u rows have |resid|<900 (override good); 64.5% >30m error.

**C29 LIRF regime/calibration probes (all REJECT):** constant median 13344; clip(MVT−SCHED≤C) worse (best 11844); oracle blend w·move+(1−w)·MVT−SCHED w=0.9 → 5850 Jan+Jul but 2997 Dec (worse than 2786); LIRF-specific movement LGB 9492/9079. `MVT−SCHED` is at the information floor.

**C30 Final:** no E29 candidate reduces total SSE; reduction achieved 0% of the 260 budget. **260 is not reachable via causal movement-only unmatched modeling.** Closes service recovery (C24), matched-tail correction (C25), and movement-only unmatched (E29). E20 stays production (LB 316.9654). Next viable direction (if pursued) must add *new same-flight information* absent from the movement record (e.g., external ground/pushback data), not more modeling of existing fields. Artifacts: `experiments/run_e29_movement_only.py`, `experiments/run_e29_lirf_forensics.py`, `experiments/results/E29/` (E29_final_report.md, JSONs).

---

## E30 — 246 RMSE attack: distribution/schema/gap audit (C31–C33, 2026-09-10)

**C32 Schema:** training and ranking share the identical 30 columns; no unused/hidden field. Ranking DEP 344,841: NM fields present 98.47%, SCHED 100%; unmatched 5,290 (1.53%) null every flt field incl. `FLIGHT_ID_mvt`. `ARVT_3_flt` occurs after MVT for 100% of rows (forbidden future info).

**C31 Population shift:** MVT−SCHED train median 1402 / Dec 1385 / Jan 1277 / Jul 1715 / **ranking 1501**; **PSI 0.0145** (negligible); unseen runway 0.0%, stand 0.1%, type 0.0%. LIRF-unmatched MVT−SCHED ranking median 6955 vs 2025 6781 (p90 15902 vs 14939). **Ranking is not materially shifted from the 2025 holdout.**

**C33 Gap arithmetic:** internal 368.03 ↔ LB 316.9654 (offset 51.1). LB-246 SSE target 2.087e10 vs LB-317 SSE 3.464e10 → the entire gap is unmatched/LIRF SSE; matched internal 244.76 RMSE (SSE 2.031e10) already sits at the ~246 level.

**Verdict:** No legitimate observable signal in the provided data explains a ~246 leaderboard result. With (a) population aligned, (b) all columns audited, (c) LIRF gate delay proven unobservable from movement-only fields (E29/C28), reaching 246 requires information absent from the dataset, forbidden post-takeoff fields (`ARVT_3_flt`/future), or a different ranking ground-truth definition. **No submission; E20 remains production (LB 316.9654).** Artifacts: `experiments/run_e30_distribution_audit.py`, `experiments/results/E30/E30_report.md`, `E30_distribution_audit.json`.

---

## E31 — leaderboard attack (<300) (2026-09-10)

**Goal:** LB <300 (need ≈ −17 s internal). **Result: no candidate expected to beat E20; no v7 submission.**

**Tested (all worse):** causal service-identity features (airport×FLIGHT×ADES / ×hour / ×weekday / ×month / ×runway, shrinkage m=20, mean/median/std/count) added to the matched residual expert → **matched 250.98 → 252.08** (worse). Movement-only unmatched (E29) 1561 vs E20 1546. LIRF movement-only/clip/blend all worse (E29/C30). Service-median recovery worse (C24). Matched-tail correction rejected (C25).

**SSE arithmetic:** LB 300 ⇔ internal ≈351 ⇔ cut ~11% total SSE. Matched would need ~25% cut; unmatched ~19%; LIRF RMSE 6033→~4500 with a causal model — none exists. Blocker remains unmatched/LIRF SSE, unobservable from the 30 provided columns.

**Decision:** E20 stays production (LB 316.9654). No `likable-eagle_v7.parquet`. Artifacts: `experiments/run_e31_service_expert.py`, `experiments/results/E31/E31_report.md`.

---

## E32 — surface-state reconstruction (2026-09-10)

**Built:** unified DEP+ARR causal event stream; airport/runway/stand windows (1–60 min), arrival→departure interaction (arrivals 1/2/3/5/10/15m, time-since, arr-since-dep), time-since features, temporal derivatives, dep/arr ratios, sequence signature; LightGBM matched/unmatched/LIRF models; conditional NNLS blend with E20. 2025 targets never enter features.

**Result (Jan+Jul):** E20 368.0/244.8/2214.1/6033 (overall/matched/unmatched/LIRF_u). Surface standalone 490.1/343.0/2823.5. **E20+surface NNLS → 359.1 / 244.8 / 2118.4 / 5548** — first genuine unmatched/LIRF gain (LIRF 6033→5548, total SSE −4.8%).

**December:** E20 228.7/214.8/816.2/2786; blend = E20 exactly (surface unmatched 1796 vs E20 816 ⇒ NNLS weight → 0). The gain is **Jan+Jul-specific and does not transfer**.

**Decision:** expected LB ≈308 best-case (not <300) and non-transferring ⇒ **no `likable-eagle_v7.parquet`**; E20 stays production. This is the first surface-state signal that touches LIRF (contra the E29 blanket conclusion), but it is not robust enough to ship. Artifacts: `experiments/run_e32_surface_state.py`, `experiments/results/E32/`.

---

## E33 — Latent delay decomposition (2026-09-10) — BREAKTHROUGH, v7 submitted

**Identity:** `TAXITIME = D − G` with `D = MVT−SCHED` (observed) and `G = BLOCK−SCHED`. Instead of predicting TAXITIME, model the latent gate delay **G** on the LIRF slice, reconstruct `T_hat = D − G_hat`, and α-blend with E20 on LIRF-unmatched only.

**Why it works where E29 failed:** E29 predicted T (or D−G implicitly) with a pooled movement-only model; here the target is G, the surface/regime features are LIRF-specific, and the reconstruction uses the observed D. The gate-delay target regularises the huge-D rows that dominate LIRF SSE.

**Results (LIRF unmatched, n=397 Jan+Jul / 88 Dec):**
| split | E20 | E33 (α=0.6) |
|---|---:|---:|
| Jan+Jul LIRF_u RMSE | 6033 | **4717** |
| Jan+Jul overall | 368.0 | **345.2** |
| Dec LIRF_u RMSE | 2786 | **2474** |
| Dec overall | 228.7 | **226.8** |

α=0.6 chosen as robust across both splits (pure G model α=1.0 is best on Jan+Jul 4180 but regresses Dec 3103; α=0.6 improves both). Improvement is consistent → first transferable unmatched gain found.

**Estimated LB ≈ 294 (<300)** using the E20 internal→LB offset (51.1). **Submission created: `experiments/results/E33/likable-eagle_v7.parquet`** (= `likable-eagle_v4` with 383 LIRF-unmatched rows replaced by the decomposition blend; 344,841 rows; IDs match; no nulls). **Leaderboard RMSE pending upload.** Artifacts: `experiments/run_e33_delay_decomposition.py`, `run_e33b_lirf_g.py`, `make_submission_e33.py`, `experiments/results/E33/`.

---

## E34 — v7+ latent gate-delay model (2026-09-10)

**Baseline:** v7 `likable-eagle_v7.parquet`, **LB 292.9926**. Key simplification: v7 ≡ `T = D − 0.60·G` (E20's LIRF-unmatched output is `D`).

**Model:** CatBoost predicting `G = BLOCK−SCHED` on LIRF departures with enriched features (`d_log`, LIRF-relative `d_rank`, `d×arrival/departure`, hour sin/cos, 5–30 min surface windows); reconstruct `T = D − G_hat`. Global-vs-conditional α tested; α=1.0 best for the Jan–Jul (ranking) analogue. Q=T/D, gate-ratio, XGB and LGB+Cat+XGB blends were all worse; non-LIRF unmatched decomposition is far worse (1824 vs 1546) so E20 is retained there.

**Results (LIRF unmatched n=397 / 88):** E20 6033 → v7 4717 → **E34 3937** (Jan+Jul); Dec E20 2786 → v7 2474 → E34 2766. Overall Jan+Jul 368.0 → v7 345.2 → **E34 333.7**; Dec 228.7 → v7 226.8 → E34 228.6.

**SSE:** Jan+Jul total 4.665e10 → 3.908e10 (−1.6e9, −3.4%). **Estimated LB ≈ 281.5** (offset 52.2) — large gain vs 292.99, ~1.5 s short of the 280 target.

**Submission:** `experiments/results/E34/likable-eagle_v8.parquet` (= v7 with 383 LIRF-unmatched rows replaced; 344,841 rows; IDs match; no nulls). **Leaderboard RMSE pending upload.** Artifacts: `run_e34_lirf_improve.py`, `run_e34b_nonlirf.py`, `run_e34c_gens.py`, `run_e34d_enriched.py`, `make_submission_e34.py`, `experiments/results/E34/`.

---

## E35 — Selection-aware latent gate-delay model (2026-09-10)

**Baseline:** v8 `likable-eagle_v8.parquet`, **LB 288.9003** (LIRF_u 3937, Jan+Jul overall 333.7).

**Idea:** G trained on matched flights but deployed on unmatched → selection bias. Tested (a) propensity weighting `P(M=0|x)/P(M=1|x)` (95/99 pct clipped), (b) G trained on unmatched LIRF rows only, (c) enriched features.

**Results (LIRF-unmatched RMSE):** v8 3937 / 2766 (Jan+Jul/Dec); weighted 3894 / 2641; **unmatched-only 3752 / 2557** — best on both splits. Jan+Jul overall 333.7 → 331.3; Dec 228.6 → 227.3; total Jan+Jul SSE 3.835e10 → 3.779e10.

**Decision:** `unm_only` α=1.0 produces `likable-eagle_v9.parquet` (= v8 with 383 LIRF-unmatched rows replaced by `D − G_unm`; 344,841 rows; IDs match; no nulls). **Estimated LB ≈ 286.5** (offset 44.8) — consistent gain vs 288.90 but not yet <280. Artifacts: `run_e35_selection_g.py`, `make_submission_e35.py`, `experiments/results/E35/`.

---

## E36 — matched-population attack (2026-09-10)

**Anchor:** v9 `likable-eagle_v9.parquet`, **LB 288.9003** (= v8; E35 LIRF change did not transfer ⇒ LIRF lever saturated).

**STEP 1 (matched SSE map, Jan+Jul):** matched n=339,046, RMSE 244.8, SSE 2.031e10; top 1% = 41.8% SSE. Dominated by **high-taxi rows** (T top-5%: RMSE 1379, 32% SSE) and LIRF/LFPG/LTFM/EGLL; largest-D bin only ~10% ⇒ schedule delay is not the matched cost driver. Source-disagreement top 1% = 11% SSE.

**STEP 5 (direct v9 residual, cross-month OOF):** matched RMSE 244.76 → 242.91 (λ=0.5) Jan+Jul, 214.78 → 213.32 Dec; SSE −0.31e9 / −0.10e9; overall ~331.3 → 329.9 (**−1.4 internal**), **est. LB ≈ 287.5**. Gating by correction magnitude did not help.

**Decision: no v10.** Best candidate ≈ 287.5 LB, below E36's minimum bar (<285). The matched tail (42% of matched SSE in top 1%) is not further reducible from available causal fields beyond ~1.5 s shrinkage. Artifacts: `experiments/run_e36_matched_sse.py`, `run_e36_matched_residual.py`, `experiments/results/E36/`.

---

## E37 — Ground-state representation (2026-09-10)

**Anchor:** v9 LB 288.9003. **Mission:** build an airport ground-movement/route representation to attack the matched high-taxi tail (top 5% T = 32% of matched SSE).

**Geometry blocker:** provided stand IDs (`811`,`R07`,`K7L`,`B10R`,`A52`) have no openly licensed, reproducible stand→coordinate mapping; the requested taxiway graph would require fabricated coordinates (forbidden: "do not fake precision").

**Coordinate-free subset tested:** two-stage `T_phys` (shrunk median by airport×runway×stand-zone, hierarchical to airport×runway then airport) + `T_excess` (LightGBM on runway-configuration state, route(runway×zone) pressure, surface counts); soft blend with E20.

**Result (matched RMSE / top-5% SSE):** E20 244.8 / 1.264e10; ground-state 353.5 / 3.018e10 (Jan+Jul); Dec E20 214.8 / 4.240e9 vs 282.6 / 7.682e9. Optimal blend weight **g=0.0** (E20) on both splits. **Top-5% SSE unchanged.**

**Decision:** no v10. The coordinate-free representation adds nothing over E20's existing stand×runway `geo_mean`; the true geometry engine is not reproducible from the available stand IDs. Per E37 §15/§18, stopped. Artifacts: `experiments/run_e37_ground_state.py`, `experiments/results/E37/`.

---

## E38 — Latent off-block clock fusion (2026-09-10)

**Anchor:** v9 LB 288.9003. **Idea:** fuse the multiple pre-MVT off-block clocks (AOBT/LOBT/IOBT/EOBT) as noisy measurements of a latent off-block B*.

**STEP 1:** the four clocks are mutually identical (`clk_range` median 0 in normal and extreme groups) ⇒ no disagreement signal. The information is the **absolute time-to-MVT of each clock** (extra taxi proxies).

**STEP 15 ablation (matched expert A):** base 250.98 → +all clocks **243.51** (Jan+Jul; beats E20 ensemble matched 244.76); top-5% SSE 1.316e10 → 1.204e10 (−8.5%); Dec 223.95 → 221.81.

**Ensemble:** clock expert + cached E20 experts NNLS → Jan+Jul matched **240.69**, overall 368.03→365.38; Dec 214.78→214.38 with very different weights (A 0.574 vs 0.166) ⇒ split-unstable. 2-component grid: robust w=0.2 → Jan+Jul −1.5, Dec neutral; w≥0.4 regresses Dec.

**Decision: no v10.** Robust gain ~−1.5 s internal (est. LB ~287), below bar; the stronger Jan+Jul setting is overfit and degrades December. Clocks are a real but small, split-sensitive signal. Artifacts: `experiments/run_e38_clock_fusion.py`, `run_e38b_ensemble.py`, `experiments/results/E38/`.

---

## E39 — Off-block delta (Δ = AOBT − BLOCK) (2026-09-10)

**Anchor:** v9 LB 288.9003. Identity `T = P + Δ`, `P = MVT−AOBT`.

**Baselines (matched, cross-month):** E20 244.76 / 214.78; P 428.17 / 374.60; P+global-meanΔ 428.18 / 374.14; P+airport-meanΔ 409.14 / 356.29; **P+LightGBM Δ 248.66 / 219.94**. Δ: mean −19 s, sd 375 s (airport reporting bias + noise).

**Decision:** no v10. `P + Δ_hat` does not beat E20 (248.66 vs 244.76 Jan+Jul; 219.94 vs 214.78 Dec) ⇒ Δ is not predictable enough to convert E38's clock signal into a leaderboard gain. Δ exhausted. Artifacts: `experiments/run_e39_delta.py`, `experiments/results/E39/`.

---

## E40 — Regime break / tail calibration (2026-09-10)

**Anchor:** v9 LB 288.9003. **Hypothesis:** v9 ranks extreme rows correctly but compresses their magnitude → calibrate the tail.

**Diagnostics:** Spearman(v9,true)=0.86; actual/predicted quantile ratios p50 0.96 → p99 **1.15** (Jan+Jul), Dec p99 1.17 ⇒ mild tail compression only.

**Calibration (cross-month OOF):** base matched 244.76; isotonic 271.85 (overfits); LGB residual (e20,P) 243.88; P-scaled residual 243.69. **Top-5% SSE 1.264e10 → 1.257e10 (−0.6%, unchanged)**; December regresses (214.78 → 215.4).

**Decision:** no v10. The tail-compression hypothesis is falsified — ranking is already good, compression is modest, and calibration gives <1.1 s with no top-5% SSE reduction plus a December regression. Artifacts: `experiments/run_e40_tail_calibration.py`, `experiments/results/E40/`.

---

## E41 — Actual surface queue at AOBT (2026-09-10)

**Anchor:** v9 LB 288.9003. **Representation:** exact surface occupancy at `t=AOBT_i` via interval overlap — active deps `AOBT_j≤t<MVT_j` (all/same-rwy/by WTC), active arrivals `MVT_j≤t<BLOCK_j`, runway takeoff service rates and previous-WTC; LGB on TAXITIME with P and queue state; OOF blend with E20.

**Results (matched):** Jan+Jul E20 244.76 / top5 1.264e10; queue model 309.41 / 2.316e10; optimal blend **a=0.0 → E20**. Dec E20 214.78 / 4.240e9; blend a=0.2 → 213.09 / 4.154e9 (−1.7 s, top5 −2%).

**Decision:** no v10. The exact AOBT queue does not reduce top-5% matched SSE on the primary split; E20 already captures the surface signal. Artifacts: `experiments/run_e41_actual_queue.py`, `experiments/results/E41/`.

---

## E42 — Flight identity residual prior (2026-09-10)

**Hypothesis:** some recurring flights have a stable `residual = TAXITIME − base` that a shrunk historical identity prior can capture.

**Method (chronological):** identity keys CALLSIGN / CALLSIGN×airport / ×ADES / FLIGHT_mvt variants / ×weekday / ×hour-bucket / operator×route; empirical-Bayes shrinkage `n/(n+λ)·mean_resid` (λ grid); applied as `pred = E20 + α·prior`. Key/λ/α auto-selected on a train-internal temporal holdout (tr_es residuals from expert A), then frozen and evaluated on Jan+Jul and Dec (E20 OOF).

**Result:** selected CALLSIGN×weekday×hour-bucket, λ=1, α=1, coverage 35.8% (Jan+Jul) / 88.5% (Dec). E20 matched 244.76 → **249.64** (worse); top-5% SSE 1.264e10 → 1.289e10 (worse); Dec 214.78 → **223.58** (worse).

**Decision: NO-GO — no v10.** The identity residual is not persistent/transferable: even shrunk, the prior adds noise and degrades both splits and the tail. Confirms E26/E35 findings on identity/service medians. Artifacts: `experiments/run_e42_flight_identity.py`, `experiments/results/E42/E42_report.md`, `E42_results.json`.

---

## E43 — Non-LIRF unmatched D−G opportunity (2026-09-10) — REJECT

**Diagnostic (Jan+Jul, E20/v9 baseline):** per-airport unmatched RMSE / %dataset SSE — **LFPG 880 rows, RMSE 3440.8, 12.5× matched, 22.33% SSE**; LIRF (already handled) 397, 6032.7, 30.97%; EHAM 571.6/0.57%; LSZH 611.8/0.42%; LTFM 593.1/0.50%; all others ≤0.09%. Ranking unmatched: EHAM 3.16%, LSZH 2.85%, LFPG 1.84%.

**Step 2 (LFPG D−G, E34 CatBoost G, per-airport blend):** Jan+Jul E20 3440.8 → best blend **3487.3 (worse)**; Dec 743.4 → **1873.6 (worse)**.

**Mechanism:** LFPG unmatched truth is normal taxi (median 1026 s, p90 1685, 1% >1 h) while D=MVT−SCHED is huge (median 5884) with corr(T,D)≈0. E20 already predicts normal values (mean 1139, max 2582); its SSE is a few extreme true-taxi rows with no causal signal. Priors ≈ E20; D−G worse.

**Decision: REJECT, no v10.** LIRF's D−G success required a real corr(T,D); LFPG's extreme rows have none. Branch closed. Artifacts: `experiments/run_e43_unmatched_diagnostic.py`, `run_e43_lfpg_g.py`, `experiments/results/E43/`, `analysis/E43/`.

---

## E44 — Matched-tail re-audit (2026-09-10) — REJECT

**Step 0:** ranking metric is pooled RMSE; no stratification documented (undeterminable beyond that).

**Step 1 (E20 tuning/ensembling depth):** audit found E20 used fixed HPs, one seed, **no HP search / seed averaging / bagging**. Ran LGB variants (deep 127/0.04, shallow 31/0.03, reg λ=10) + 3-seed average + NNLS with E20 experts: Jan+Jul matched 244.76 → **244.33 (−0.43)**, Dec 214.78 → 214.78 (**0.00**, weight E20=1.0); top-5% SSE not materially reduced ⇒ **no headroom**. (CatBoost/XGB sweeps infeasible at 1.4M rows.)

**Step 2 coverage:** E40's calibration rejection was **global-only** (per-airport isotonic genuinely new); E42's identity rejection was **CALLSIGN/route scope** (operator×type×hour genuinely new); segment stratification and log-space refit genuinely new. Segment diagnostic: Jan+Jul matched RMSE Non-Scheduled 333 / Cargo 279 / Lowcost 286 / Mainline 238 / Regional 182; WK_TBL H 273 vs M 236 — segments differ, but E20 already uses both as categorical features and mean residuals are small. None advanced.

**Decision: REJECT, no v10.** No exploitable matched-tail headroom from deeper tuning; untested Step 2 angles are low-yield given existing features and E40/E42's measured mild structure. Artifacts: `experiments/run_e44_matched_tuning.py`, `run_e44b_matched_lgb.py`, `experiments/results/E44/`, `analysis/E44/`.
