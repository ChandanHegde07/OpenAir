# E19-A — Remaining structural SSE after E18-H

Diagnosis only. No new model. `training_*.parquet` only.
Script: `experiments/run_e19a_structural_sse.py`.

E18-H was training hygiene (matched-only `geo_mean`, drop LIRF-override
rows from residual train). This note locates the error that hygiene cannot
reach.

## E18-H Jan+Jul SSE budget

| Slice | n | RMSE | SSE share |
|---|---:|---:|---:|
| Overall | 344,419 | 372.36 | 100% |
| Matched | 339,046 | 250.98 | 44.7% |
| Unmatched | 5,373 | 2216.59 | 55.3% |
| LIRF unmatched | 397 | 6032.70 | 30.3% |
| Non-LIRF unmatched | 4,976 | 1549.74 | 25.0% |

Unmatched by airport (E18-H predictions):

| Airport | n | mean y | RMSE H | % of all SSE | % of unmatched SSE |
|---|---:|---:|---:|---:|---:|
| LIRF | 397 | 6272 | 6032.7 | 30.3% | 54.7% |
| LFPG | 880 | 1295 | 3439.3 | 21.8% | 39.4% |
| EGLL | 466 | 1705 | 1139.0 | 1.3% | 2.3% |
| EHAM | 808 | 742 | 565.6 | 0.5% | 1.0% |
| LSZH | 524 | 633 | 649.0 | 0.5% | 0.8% |
| LTFM | 657 | 1146 | 578.2 | 0.5% | 0.8% |
| EDDM | 237 | 1006 | 613.9 | 0.2% | 0.3% |
| EDDF | 599 | 1005 | 348.5 | 0.2% | 0.3% |
| LEBL | 496 | 1056 | 338.6 | 0.1% | 0.2% |
| LEMD | 309 | 1150 | 248.5 | 0.0% | 0.1% |

## Unmatched y vs MVT−SCHED (Jan+Jul val)

Would a LIRF-style `MVT−SCHED` override help anywhere else?

| Airport | n | corr(y, MVT−SCHED) | RMSE geo | RMSE MVT−SCHED | mean y | mean MVT−SCHED |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 599 | 0.134 | 442 | 8388 | 1005 | 7052 |
| EDDM | 237 | 0.240 | 725 | 5818 | 1006 | 6266 |
| EGLL | 466 | 0.150 | 1224 | 6348 | 1705 | 7427 |
| EHAM | 808 | 0.309 | 728 | 7778 | 742 | 5327 |
| LEBL | 496 | 0.132 | 386 | 6970 | 1056 | 7078 |
| LEMD | 309 | 0.301 | 360 | 6523 | 1150 | 7148 |
| LFPG | 880 | -0.020 | 3456 | 7904 | 1295 | 6447 |
| LIRF | 397 | 0.916 | 13460 | 6033 | 6272 | 9587 |
| LSZH | 524 | 0.345 | 708 | 4363 | 633 | 3542 |
| LTFM | 657 | 0.180 | 656 | 3154 | 1146 | 2708 |

## LFPG unmatched

LFPG unmatched is **not** a second LIRF. Full-year corr(y, MVT−SCHED) = 0.020 (LIRF was 0.90). Applying `MVT−SCHED` as taxi is the wrong clock.

y bins (full-year LFPG unmatched) vs MVT−SCHED:

| Bin | n | mean y | mean MVT−SCHED | RMSE(MVT−SCHED) |
|---|---:|---:|---:|---:|
| <15m | 1317 | 708 | 6196 | 7684 |
| 15-30m | 2143 | 1207 | 6504 | 6866 |
| 30-60m | 257 | 2264 | 7887 | 6196 |
| 1-2h | 39 | 4665 | 8014 | 4274 |
| >2h | 7 | 26205 | 7827 | 37772 |

Error concentration vs E18-H mean prediction on Jan+Jul LFPG unmatched:

| Worst k | SSE share |
|---:|---:|
| 1 | 66.0% |
| 2 | **97.1%** |
| 3 | 97.5% |
| 5 | 97.9% |
| 10 | 98.3% |

The 21.8% of *all* holdout SSE from LFPG unmatched is **two easyJet rows**:

| Flight | Dest | Stand | Type | y (s) | MVT−SCHED | Implied BLOCK−SCHED |
|---|---|---|---|---:|---:|---:|
| EJU983W | LFMN | C03 | A319 | 84,240 (23.4 h) | 1,740 | **−82,500** (~−23 h) |
| EJU42AY | GMMX | D06 | A320 | 58,206 (16.2 h) | 2,043 | **−56,163** (~−16 h) |

Takeoff is ~30 min after schedule; reported taxi is 16–23 h. That is a **wrong-day BLOCK**, not a long taxi. `MVT−SCHED` as taxi is the wrong clock (RMSE 7904). A 24 h wrap `y_hat = MVT−SCHED + 86400` would score those two near-correctly (~88k vs 84k) and **destroy** the other 878 LFPG unmatched (typical MVT−SCHED ~6000 s). The wrap has to be gated.

Full-year LFPG unmatched >2 h: only **7** rows. This is not a CDG unmatched population.

## What-if overall RMSE (E18-H base 372.36)

| Change | Overall RMSE | Δ |
|---|---:|---:|
| Identify/fix the 2 LFPG bombs (slice RMSE 3439→~400) | **329.9** | **−42** |
| LIRF unmatched oracle gate (6033→2771, E13) | **324.9** | **−47** |
| Both of the above | ~**275** | ~**−97** |
| LFPG unmatched → raw MVT−SCHED | 517.7 | +145 (do not) |
| Matched 251→240 (E16/E17 class) | 365.2 | −7 |
| Matched 251→200 (fantasy) | 340.6 | −32 |

Matched-only work cannot close a 65-point leaderboard gap. The two unmatched generating processes can.

## Implications

1. Do not spend the next weeks on matched 1% features (E16-A / E17-A class) or more training hygiene.
2. **LIRF unmatched (30% of SSE):** still the E13 problem — two TAXITIME regimes, `MVT−SCHED` large in both. Next information is neighbor gate-vs-taxi delay (matched neighbours have AOBT; the scored row does not).
3. **LFPG unmatched (22% of SSE):** not a model class. It is 2 wrong-day BLOCK labels. Next information is a ranking-safe gate for “BLOCK is yesterday” (easyJet + unmatched + small MVT−SCHED is a candidate, must be precision-checked on train so it does not fire on the 878 normal CDG unmatched).
4. Other unmatched airports together are **~3% of all SSE**. EHAM/LSZH offsets are not the overall metric.
5. Matched tail (y>30 min ≈ 46% of *matched* SSE, ~20% of overall) remains real and unidentified after E15/E16; it is the third structural source, not the first.

