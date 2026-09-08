# E21 — LIRF unmatched override calibration

**Project:** OpenAir  
**Experiment:** E21  
**Validation:** Jan+Jul 2025 training holdout  
**Stress:** December 2025  
**Data:** 12 `training_*.parquet` only. No ranking/submission.  
**Scope:** a correction *on top of* `y_hat = MVT−SCHED` for `unmatched and airport==LIRF`. Ensemble weights frozen. Matched path frozen.

Script: `experiments/run_e21_lirf_calibration.py` (copied to `analysis/E21/run_e21.py`).

---

## 1. Diagnosis (before any correction)

The 6033 number is **not** a few bombs on an otherwise-calibrated clock. Residual
`r = y − (MVT−SCHED)` on Jan+Jul **val** LIRF unmatched (n=397):

| | Jan+Jul val | December val |
|---|---:|---:|
| n | **397** (0.115% of 344,419) | **88** |
| RMSE | 6032.70 | 2786.14 |
| MAE | 4228.75 | 1229.64 |
| mean r | **−3315** | −1145 |
| median r | **−3963** | **0** |
| skew | +1.97 | −2.13 |
| % r < 0 | 75% | 48% |
| n \|r\| > 1 h | 253 / 397 | 18 / 88 |
| n \|r\| > 2 h | 73 | 3 |
| mean y / mean MVT−SCHED | 6272 / 9587 | 7706 / 8851 |
| corr(y, MVT−SCHED) | 0.916 | 0.976 |

SSE concentration, Jan+Jul val:

| Slice | k | SSE share |
|---|---:|---:|
| worst 1 | 1 | 5.7% |
| worst 5 | 5 | 22.5% |
| worst 10 | 10 | 29.5% |
| worst 10% of rows | 39 | 53.4% |
| \|r\| > 1 h | 253 | 99.9% |
| **y ≤ 30 min (normal half)** | **228** | **79.1%** |
| y > 30 min (extreme half) | 169 | 20.9% |

**Reading:** 6033 is the E13 normal half. `MVT−SCHED` overpredicts ~75% of Jan+Jul
LIRF unmatched by ~1 h (median residual −66 min). The extreme half, where
BLOCK≈SCHED, is already well fit. Worst-1 is only 5.7% of SSE — this is
**uniform regime error**, not two CDG-style bombs.

December is a **different mixture**: median residual **0**, 60/88 rows are
y>30 min, and that extreme half is **0.3% of December LIRF unmatched SSE**.
Almost all December LIRF unmatched RMSE is the 28 short-taxi rows. Subtracting
a large constant (the Jan+Jul mean residual) will wreck the 60 well-fit
extremes.

Hour-of-day: daytime hours (10–19 UTC) have median residual ≈ −4000 s (normal
half dominates). Night hours 0–4 and 20, 23 have median residual ≈ 0 (extreme
half). That is the same bimodality, not a clock-bias that a per-hour intercept
can fix without knowing the regime.

MVT−SCHED bins: 1–2 h and 2–4 h after schedule (330 of 397 rows) have median
residual −4080 / −6776. The >4 h bins have median residual ≈ 0. “SCHED looks
late” is exactly when the override is already correct.

---

## Row-count vs overall RMSE (why this slot is expensive)

Against E20 (overall 368.03, n=344,419):

- LIRF unmatched: **397 rows, 0.115% of the holdout, 31.0% of overall SSE**.
- A 10% LIRF-u RMSE cut (6033→5430) ≈ **−3 to −4 s overall**.
- Halving LIRF-u RMSE (6033→3016) ≈ **−20 s overall**.
- The E13 oracle (normal→geo, extreme→SCHED) is still ≈ **−47 s overall**.

Worth the experiment only if a correction is large *and* survives December.
E20’s ensemble was −4.3 s overall from the matched path. A failed LIRF
calibration that helps Jan+Jul and hurts December is a net loss.

---

## 2. Corrections (train LIRF unmatched only)

`c` estimated on the **train** side of each split (n=1,091 Jan+Jul train,
n=1,400 Dec train). Applied as `y_hat = (MVT−SCHED) + c`, or OLS
`a + b·(MVT−SCHED)`, or clip MVT−SCHED to train 1–99%.

Train median residual is **−4 s** (the two regimes straddle zero). Train mean
residual is **−1890 s** (pulled by the overpredicted normal half).

---

## 3. Holdout

Matched RMSE is **244.76 / 215.90** on both splits for every variant
(this slice is disjoint from matched).

### Jan+Jul 2025 (E20 overall 368.03)

| Rule | LIRF unmatched | Δ | Overall | Δ |
|---|---:|---:|---:|---:|
| raw MVT−SCHED | 6032.7 | 0 | 368.03 | 0 |
| add mean (−1890) | 5238.0 | −795 | **353.73** | **−14.30** |
| add median (−4) | 6030.5 | −2 | 367.99 | −0.04 |
| add winsor mean (−2156) | 5171.9 | −861 | **352.60** | **−15.43** |
| add hour-median | 5666.7 | −366 | 361.26 | −6.77 |
| add month-median | 6030.5 | −2 | 367.99 | −0.04 |
| add weekday-median | 6030.8 | −2 | 367.99 | −0.04 |
| OLS a+b·ms | 5217.7 | −815 | 353.38 | −14.65 |
| clip ms p01–p99 | 6737.4 | +705 | 381.86 | +13.83 |

### December 2025 (E20 overall 228.45)

| Rule | LIRF unmatched | Δ | Overall | Δ |
|---|---:|---:|---:|---:|
| raw MVT−SCHED | 2786.1 | 0 | 228.45 | 0 |
| add mean (−2341) | 2807.3 | **+21** | 228.59 | **+0.14** |
| add median (−5) | 2784.1 | −2 | 228.44 | −0.01 |
| add winsor mean | 2953.6 | **+167** | 229.56 | **+1.11** |
| add hour-median | 3376.9 | **+591** | 232.64 | **+4.19** |
| add month-median | 2784.1 | −2 | 228.44 | −0.01 |
| add weekday-median | 3312.0 | **+526** | 232.15 | **+3.70** |
| OLS a+b·ms | 2831.8 | **+46** | 228.75 | **+0.30** |
| clip ms p01–p99 | 4458.1 | **+1672** | 242.12 | **+13.67** |

Mean / winsor / OLS look like a 14–15 s overall win on Jan+Jul because they
shrink the overpredicted normal half. December’s mixture is mostly the
extreme half, where `MVT−SCHED` is already exact; the same shift **raises**
LIRF unmatched RMSE. Hour and weekday medians overfit Jan+Jul train hours
and blow up December.

Median / month-median survive both splits and do **nothing** (−0.04 s).

---

## 4. Decision

**Verdict: REJECT.** Keep always-on uncalibrated `MVT−SCHED`.

- OLD: 6033 might be a systematic clock bias (hour/season/mean residual)
  that a train-only additive or slope on `MVT−SCHED` could shave without
  reopening E13 gates.
- EVIDENCE: 79% of Jan+Jul LIRF unmatched SSE is the *normal* half
  (`MVT−SCHED` too big). Mean residual −3315 s. A mean/OLS/winsor shift
  cuts Jan+Jul overall by 14–15 s and **fails December** (median residual
  there is 0; extreme half is 60/88 rows). Median shift is −4 s on train
  and −0.04 overall. Same bimodality E13 already named; a location shift
  cannot serve both regimes.
- NEW: Do not calibrate `MVT−SCHED`. Do not add hour/weekday intercepts
  on this slice. The override stays raw `MVT − SCHED`. Ensemble unchanged.
  Readme current-model block unchanged.

This is the same rejection pattern as E18-A-mean: a large, real Jan+Jul
move that does not survive December.

Artifacts: `analysis/E21/`.
