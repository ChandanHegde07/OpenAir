# E27 — AOBT off-block anchor audit

**Ranking DEP AOBT coverage: 98.47%** (missing = the 5,290 unmatched rows). Same null-set for LOBT/IOBT/EOBT. SCHED 100%.

## Phase 1 — delta = BLOCK_TIME − AOBT (matched, full training)
| exact rate | median Δ | MAE | RMSE | p50 abs | p90 abs | p95 abs | p99 abs | max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.65% | 51s | 238s | 385s | 175s | 513s | 664s | 1321s | 20657s |

| Airport | median Δ | MAE | RMSE | p90 abs |
|---|---:|---:|---:|---:|
| EDDF | 87s | 178s | 255s | 339s |
| EDDM | -2s | 207s | 349s | 476s |
| EGLL | 54s | 250s | 419s | 484s |
| EHAM | 107s | 205s | 330s | 433s |
| LEBL | -54s | 193s | 311s | 419s |
| LEMD | 2s | 173s | 276s | 357s |
| LFPG | -4s | 234s | 380s | 536s |
| LIRF | -118s | 325s | 557s | 718s |
| LSZH | -33s | 183s | 268s | 373s |
| LTFM | 296s | 387s | 531s | 664s |

## Phase 2/3/5 — Jan+Jul 2025 (primary)

| Model | overall | matched | >30m | >45m | >60m |
|---|---:|---:|---:|---:|---:|
| E20 | 368.03 | 244.76 | 774.8 | 1484.2 | 2361.4 |
| AOBT | 475.06 | 428.17 | 1285.0 | 2367.3 | 3583.2 |
| LOBT | 842.14 | 816.96 | 1031.4 | 1323.5 | 1571.7 |
| IOBT | 841.99 | 816.80 | 1028.6 | 1317.9 | 1557.6 |
| EOBT | 773.26 | 745.66 | 1019.6 | 1446.8 | 1978.0 |
| SCHED | 2591.40 | 2472.03 | 2697.6 | 2821.8 | 2345.4 |
| AOBT_cal | 327.14 | 254.03 | 780.6 | 1391.0 | 1925.9 |
| AOBT+E20_unmatched | 506.89 | 428.17 | 1285.0 | 2367.3 | 3583.2 |
| AOBT_cal+E20_unmatched | 374.17 | 254.03 | 780.6 | 1391.0 | 1925.9 |

Proxy coverage: {'AOBT': 0.9843998153411978, 'LOBT': 0.9843998153411978, 'IOBT': 0.9843998153411978, 'EOBT': 0.9843998153411978, 'SCHED': 1.0}

## Phase 4 — off-block source disagreement vs E20 residual (matched)

| source pair | corr(abs diff, E20 |resid|) | >30m rate low | >30m rate high(p90+) |
|---|---:|---:|---:|
| d_aobt_eobt | 0.286 | 0.031 | 0.168 |
| d_aobt_iobt | 0.286 | 0.033 | 0.163 |
| d_aobt_lobt | 0.285 | 0.033 | 0.163 |
| d_lobt_iobt | 0.031 | 0.044 | 0.111 |
| d_lobt_eobt | 0.086 | 0.044 | 0.061 |

## Phase 2/3/5 — December 2025 (sanity)

| Model | overall | matched | >30m | >45m | >60m |
|---|---:|---:|---:|---:|---:|
| E20 | 228.69 | 214.78 | 672.6 | 1368.3 | 2031.7 |
| AOBT | 380.02 | 374.60 | 1044.9 | 2031.9 | 3018.4 |
| LOBT | 703.75 | 700.97 | 970.8 | 1522.4 | 1940.4 |
| IOBT | 703.14 | 700.36 | 968.5 | 1517.4 | 1940.6 |
| EOBT | 648.38 | 645.34 | 950.4 | 1542.7 | 2012.9 |
| SCHED | 2480.22 | 2375.50 | 2251.1 | 2625.0 | 2580.9 |
| AOBT_cal | 235.79 | 226.85 | 691.8 | 1448.4 | 2092.5 |
| AOBT+E20_unmatched | 381.52 | 374.60 | 1044.9 | 2031.9 | 3018.4 |
| AOBT_cal+E20_unmatched | 239.96 | 226.85 | 691.8 | 1448.4 | 2092.5 |

Proxy coverage: {'AOBT': 0.9900468984831932, 'LOBT': 0.9900468984831932, 'IOBT': 0.9900468984831932, 'EOBT': 0.9900468984831932, 'SCHED': 1.0}

## Phase 4 — off-block source disagreement vs E20 residual (matched)

| source pair | corr(abs diff, E20 |resid|) | >30m rate low | >30m rate high(p90+) |
|---|---:|---:|---:|
| d_aobt_eobt | 0.267 | 0.027 | 0.145 |
| d_aobt_iobt | 0.243 | 0.028 | 0.142 |
| d_aobt_lobt | 0.242 | 0.028 | 0.142 |
| d_lobt_iobt | 0.030 | 0.039 | 0.076 |
| d_lobt_eobt | 0.041 | 0.039 | 0.042 |

## Phase 6 — leakage audit (AOBT must be < MVT)

violations 1012 / 2,062,577 (AOBT−MVT p1 = -2105s; min = -87181s)

## Bottom line

AOBT raw proxy matched 428.17; calibrated 254.03; hybrid (cal + E20 unmatched) 254.03 vs E20 244.76.
