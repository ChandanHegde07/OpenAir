# E19-C — LIRF unmatched neighbour gate-vs-taxi

**Project:** OpenAir  
**Experiment:** E19-C  
**Data:** 12 `training_*.parquet` only. No ranking/submission. No LightGBM.  
**Script:** `experiments/run_e19c_lirf_neighbor_split.py`

E13 closed hard `MVT−SCHED` thresholds. The missing feature was “gate hold vs
long taxi,” which unmatched rows cannot observe (no AOBT/BLOCK). Matched
neighbours at the same airport *do* have AOBT by the scored takeoff.

`neigh_taxi_frac_30m` = mean of other flights’ `(MVT−AOBT)/(MVT−SCHED)` with
AOBT in the previous 30 min (self excluded). High fraction ⇒ delay is on the
taxiway; low ⇒ delay is at the gate.

---

## Phase 2 — does neighbour mix identify the y>30 min regime?

| Split | n LIRF unmatched | n y>30 | corr(frac, y) | corr(frac, y>30) | Q8/Q1 P(y>30) |
|---|---:|---:|---:|---:|---:|
| Jan+Jul train | 1,091 | 622 | 0.049 | −0.021 | 0.86 |
| Jan+Jul val | 397 | 169 | 0.017 | −0.121 | 0.66 |
| Dec train | 1,400 | 731 | 0.048 | −0.015 | 0.88 |
| Dec val | 88 | 60 | −0.092 | −0.059 | 1.00 |

Signal is **absent or backwards**. High neighbour taxi fraction does *not*
raise P(y>30) on LIRF unmatched (if anything the opposite, weakly).
`neigh_push_mean` and `neigh_taxi_mean` are the same.

Jan+Jul val by taxi-frac quantile: see `tables/e19c_val_frac_q_janjul.csv`.
No monotone y>30 rate.

---

## Oracle vs deployable

LIRF unmatched still uses always-on `MVT−SCHED` (E18-H). Magically routing
y≤30 min → `geo_mean` and y>30 → `MVT−SCHED`:

| Split | Always SCHED (LIRF_u RMSE) | Always geo | Oracle mix | Overall if oracle |
|---|---:|---:|---:|---:|
| Jan+Jul | 6,033 | ~13,460 | ~2,771 | **324.86** (−47 vs 372) |
| December | 2,786 | — | — | **229.33** (−9 vs 238) |

Train-best threshold on any neighbour column: **no T beats always-SCHED by 1 s**
on the train LIRF unmatched slice (`T=None`). There is nothing to apply on val.

---

## Decision

**Verdict: REJECT.**

- OLD: neighbour push-vs-taxi mix is the information E13 said unmatched rows
  lack, and should gate `MVT−SCHED` vs `geo_mean`.
- EVIDENCE: corr with the extreme regime is ~0 (val −0.12, train −0.02).
  No train threshold improves LIRF unmatched RMSE. Oracle ceiling is real
  (−47 overall) and still unreachable from prediction-time neighbour clocks.
- NEW: LIRF unmatched is **not** the airport’s current surface-delay regime.
  It is a join-failure / type-null DGP. Neighbour AOBT does not identify
  gate-hold vs long-taxi on that slice. Keep always-on `MVT−SCHED`. Do not
  reopen neighbour-window sweeps. Remaining LIRF headroom is still the
  unidentified regime split (callsign/history prior, or accept the floor).

Artifacts: `analysis/E19C/`.
