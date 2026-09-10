# E41 — actual surface queue at AOBT: report

## Baseline
v9 = `likable-eagle_v9.parquet`, **LB 288.9003**.

## The representation built
Exact surface occupancy at the target's actual off-block time `t = AOBT_i`,
via interval overlap (prefix counts over sorted AOBT/MVT):
- active departures: `AOBT_j ≤ t < MVT_j` (all / same-runway / other-runway / by WTC)
- active arrivals: `MVT_j ≤ t < BLOCK_j` (arrival BLOCK is visible in ranking —
  only departure BLOCK is blanked)
- runway takeoffs in last 5/10/20/30 min before AOBT, time since last takeoff,
  previous takeoff WTC (heavy/super), queue/service ratio.
Model: LightGBM on `TAXITIME` with `P = MVT−AOBT` + exact queue state; blend
with E20 by OOF-selected weight.

## Results (matched)
| split | E20 RMSE / top-5% SSE | queue model | best blend |
|---|---:|---:|---:|
| Jan+Jul | 244.76 / 1.264e10 | 309.41 / 2.316e10 | **a=0.0 → E20** |
| Dec | 214.78 / 4.240e9 | 236.62 / 5.143e9 | a=0.2 → 213.09 / 4.154e9 (−1.7) |

On the primary split the optimal blend weight is **0.0** and top-5% SSE does
**not** fall (the standalone queue model raises it substantially). December
shows only a ~1.7 s matched gain with −2% top-5% SSE.

## Decision
**No `likable-eagle_v10.parquet`.** The exact AOBT surface queue — the central
novelty of E41 — does not reduce top-5% matched SSE on Jan+Jul and adds little
on Dec. E20 already captures the useful surface information. This closes the
"actual queue" hypothesis as a route past 288.9003. Artifacts:
`run_e41_actual_queue.py`, `experiments/results/E41/`.
