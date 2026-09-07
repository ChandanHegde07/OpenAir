# E20 — LIRF unmatched history prior

**Project:** OpenAir  
**Experiment:** E20  
**Data:** 12 `training_*.parquet` only. No ranking/submission. No LightGBM.  
**Script:** `experiments/run_e20_lirf_history_prior.py`

E13 rejected prefix / dest / stand as *exclusive triggers* for the extreme
half (low recall; OR with `mvt_sched>1800` collapses to always-on). This
reuses those ranking-safe keys as train-only rates:

- `p = P(y>30 min | prefix or ADES)` on train LIRF unmatched, shrunk to p0
- mixture `ŷ = (1−p)·geo_mean + p·(MVT−SCHED)`
- and a reverse hard rule: low p → geo (identify the *normal* half), else keep `MVT−SCHED`

Matched path and E18-H are untouched. Overall RMSE is SSE substitution on
the LIRF unmatched slice only.

---

## Why a soft mixture cannot work here

Jan+Jul LIRF unmatched, always `MVT−SCHED`: normal half RMSE **7081**, extreme
half **4224**. Oracle (normal→geo **339**, extreme→SCHED **4224**) overall
**324.86** (−47.5).

A convex combination of geo (~400) and SCHED (~6000–7000) is still thousands
of seconds on *both* halves. Mixing cannot RMSE-fix a bimodal target when
the two experts are an order of magnitude apart.

| Rule | Overall | Δ | LIRF_u | Normal | Extreme |
|---|---:|---:|---:|---:|---:|
| always SCHED | 372.36 | 0 | 6033 | 7081 | 4224 |
| mix prefix | 405.34 | +33 | 7658 | 3438 | 11036 |
| mix ADES | 393.98 | +22 | 7125 | 3797 | 9990 |
| mix max(prefix,ADES) | 383.02 | +11 | 6586 | 4067 | 8921 |
| **oracle** | **324.86** | **−47.5** | 2768 | 339 | 4224 |
| low-p prefix → geo (T=0.25) | 370.63 | −1.74 | 5939 | 6942 | 4224 |

December: every mixture **raises** overall +15 to +31 s (extreme half is
almost exact under always-SCHED, MAE ~28 s in E13; any geo bleed is fatal).
Low-p→geo: overall **−0.87**, but extreme RMSE 194→**1131**.

The two hard rules (low-p→geo vs high-p→sched) are the same threshold
T=0.25 written two ways. Train LIRF_u RMSE 5287→5234 (−53 s only): the
identifiable normal prefixes are a thin slice.

---

## December 2025

| Rule | Overall | Δ | LIRF_u | Normal | Extreme |
|---|---:|---:|---:|---:|---:|
| always SCHED | 238.01 | 0 | 2786 | 4931 | 194 |
| mix prefix | 257.05 | +19 | 5050 | 2461 | 5881 |
| low-p → geo | 237.14 | −0.87 | 2643 | 4383 | 1131 |
| oracle | 229.33 | −8.68 | 359 | 569 | 194 |

---

## Decision

**Verdict: REJECT as a current-best change.** (Auto-picker said INCONCLUSIVE
on the −1.74 / −0.87 pair; December extreme damage and mixture blow-ups
override that.)

- OLD: train-only P(y>30 | prefix/ADES/flight) as a soft mix or reverse
  gate might recover part of the E13 oracle (−47 overall) that hard
  exclusive triggers missed.
- EVIDENCE: soft mix is the wrong functional form for this bimodal RMSE
  problem (Jan+Jul +11 to +33 overall; December worse). Low-p→geo is a
  ~2 s overall move and sends December extremes to geo (194→1131).
- NEW: Keep always-on `MVT−SCHED` for LIRF unmatched. Do not mix geo and
  SCHED. Prefix/ADES rates are real descriptively (ISR/ETH vs EJU/EZY) and
  still do not yield a stable contest-RMSE rule. E13+E20 together close
  “use airline/dest to split LIRF unmatched.”

Artifacts: `analysis/E20/`.
