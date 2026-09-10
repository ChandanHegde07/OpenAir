# E29 — 260 RMSE attack: final report (no candidate accepted)

E20 untouched and remains production.

## Leaderboard calibration
Internal Jan+Jul 2025 overall 368.03 ↔ official leaderboard 316.9654 → **offset ≈ 51.1 s**.
So **LB 260 ⇒ internal ≈ 311**, NOT internal 260.

## Phase 0 — the 260 SSE budget (Jan+Jul, N=344,419)
Current: total SSE 4.665e10 | matched 2.031e10 | unmatched 2.634e10 | LIRF 1.445e10 | non-LIRF 1.189e10.

| matched RMSE | allowable unmatched RMSE | allowable LIRF RMSE (non-LIRF fixed) |
|---|---:|---:|
| 245 | 1553 | 1643 |
| 235 | 1648 | 2607 |
| 225 | 1734 | 3275 |
| 220 | 1774 | 3553 |

Required for internal 311: total SSE 3.331e10 → **cut 1.334e10 (29%)**. The only way to get there is **LIRF unmatched RMSE 6033 → ~1600–2600** (or a mix with large non-LIRF gains).

## E20 vs best E29 candidate (identical — nothing improved)

| Metric (Jan+Jul) | E20 | E29 best |
|---|---:|---:|
| overall RMSE | 368.03 | 368.03 |
| matched RMSE | 244.76 | 244.76 |
| unmatched RMSE | 2214.1 | 2214.1 |
| LIRF unmatched RMSE | 6032.7 (n=397) | 6032.7 |
| non-LIRF unmatched RMSE | 1545.9 (n=4976) | 1545.9 |
| total SSE | 4.665e10 | 4.665e10 |
| matched SSE | 2.031e10 | 2.031e10 |
| unmatched SSE | 2.634e10 | 2.634e10 |
| LIRF SSE | 1.445e10 | 1.445e10 |
| non-LIRF SSE | 1.189e10 | 1.189e10 |

Matched tail (Jan+Jul): >30 min n=464, >45 min 143, >60 min 58 (matched >30m SSE = 10.4% of total).
SSE reduction achieved toward 260 budget: **0.0%**; still required: **1.334e10**.

## Phase 1/2 — information masking (movement-only model)

| Rows | E20 RMSE | movement-only RMSE |
|---|---:|---:|
| overall | 368.03 | 523.8 |
| matched | 244.76 | 360.7 |
| unmatched | 2214.1 | 3061.7 |
| LIRF unmatched | 6032.7 | 9814.8 |
| non-LIRF unmatched | 1545.9 | 1560.9 |

Movement-only information is far weaker than NM-derived information everywhere, and does **not** beat E20 on any unmatched subset.

## Phase 5–9 — LIRF unmatched probes (all rejected)
- Constant (median) 13,344; clip(MVT−SCHED ≤ C) worse (best 11,844 on Jan+Jul, 10,588 Dec); oracle blend w·move + (1−w)·MVT−SCHED: w=0.9 → 5,850 Jan+Jul but 2,997 Dec (worse than E20 2,786).
- LIRF-specific movement-only LGB: 9,492 (Jan+Jul) / 9,079 (Dec); best blend with MVT−SCHED −127 s Jan+Jul, −1 s Dec.
- Mechanism: TAXITIME true median 1447 s vs MVT−SCHED median 6781 s — the schedule displacement is gate delay, not taxi; the largest SSE rows have **normal taxi with huge MVT−SCHED**, and |residual| does not correlate with MVT−SCHED (0.04). The gate-delay component is **not observable** without the NM off-block time.

## Verdict
**REJECT / 260 not reachable with causal movement-only modeling.** LIRF unmatched is already at its information floor with `MVT−SCHED` (6,033); non-LIRF unmatched movement-only ≈ E20 (no gain); matched tail is only ~10% of total SSE and C25's correction failed. E20 remains production (LB 316.9654). C26–C30 recorded.
