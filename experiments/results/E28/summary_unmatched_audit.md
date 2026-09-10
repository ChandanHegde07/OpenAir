# E28 — Unmatched error forensics (Mission 2, STEP 1)

## Mechanism (why unmatched)

Every unmatched DEP has **all NM flight-table fields null** (`AOBT_3_flt`, `AIRCRAFT_OPERATOR_flt`, `ADES_FILED_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`). Unmatched = a **total NM flight-record miss**, not a fuzzy-match failure. Movement-side fields (ADEP/ADES/FLIGHT/SCHED/MVT/runway/stand/type) are present.

## Jan+Jul 2025

- matched 339,046 / unmatched 5,373 (1.56%)
- RMSE overall 368.03 / matched 244.76 / unmatched 2214.08
- SSE total 4.665e+10; unmatched 2.634e+10 (**56.5% of total SSE**)
- classes {'no_NM_candidate_all_nm_cols_null': 5373, 'partial_NM_match': 0, 'unmatched_lirf': 397, 'unmatched_non_lirf': 4976}
- non-LIRF unmatched: median y 962s, >30m 6.4%, >60m 1.2%
- LIRF unmatched: median y 1447s, >30m 42.6%, >60m 37.3%
- top-SSE: {'top10': {'unmatched_in_topk': 9, 'lirf_unmatched_in_topk': 7, 'matched_in_topk': 1, 'sse_share_of_total': 0.3197141033712708}, 'top50': {'unmatched_in_topk': 48, 'lirf_unmatched_in_topk': 44, 'matched_in_topk': 2, 'sse_share_of_total': 0.4213815384015405}, 'top100': {'unmatched_in_topk': 92, 'lirf_unmatched_in_topk': 85, 'matched_in_topk': 8, 'sse_share_of_total': 0.4853224864000812}, 'top500': {'unmatched_in_topk': 294, 'lirf_unmatched_in_topk': 254, 'matched_in_topk': 206, 'sse_share_of_total': 0.6295469649577743}}

Per-airport unmatched:

| Airport | n | RMSE | SSE | % total SSE |
|---|---:|---:|---:|---:|
| LIRF | 397 | 6033 | 1.445e+10 | 0.3% |
| LFPG | 880 | 3441 | 1.042e+10 | 0.2% |
| EGLL | 466 | 1103 | 5.673e+08 | 0.0% |
| EHAM | 808 | 572 | 2.640e+08 | 0.0% |
| LTFM | 657 | 593 | 2.311e+08 | 0.0% |
| LSZH | 524 | 612 | 1.961e+08 | 0.0% |
| EDDM | 237 | 609 | 8.780e+07 | 0.0% |
| EDDF | 599 | 335 | 6.738e+07 | 0.0% |
| LEBL | 496 | 293 | 4.252e+07 | 0.0% |
| LEMD | 309 | 231 | 1.644e+07 | 0.0% |

**Recovery preview (causal service profiles, train-only):** L1 coverage 78.5%, service RMSE(all unmatched) **3946** vs E20 unmatched 2214 (non-LIRF 1583, LIRF 13391)

## December 2025

- matched 164,028 / unmatched 1,649 (1.00%)
- RMSE overall 228.69 / matched 214.78 / unmatched 816.18
- SSE total 8.665e+09; unmatched 1.098e+09 (**12.7% of total SSE**)
- classes {'no_NM_candidate_all_nm_cols_null': 1649, 'partial_NM_match': 0, 'unmatched_lirf': 88, 'unmatched_non_lirf': 1561}
- non-LIRF unmatched: median y 958s, >30m 9.5%, >60m 0.9%
- LIRF unmatched: median y 6034s, >30m 68.2%, >60m 64.8%
- top-SSE: {'top10': {'unmatched_in_topk': 9, 'lirf_unmatched_in_topk': 7, 'matched_in_topk': 1, 'sse_share_of_total': 0.06583043884145622}, 'top50': {'unmatched_in_topk': 25, 'lirf_unmatched_in_topk': 18, 'matched_in_topk': 25, 'sse_share_of_total': 0.13774666860233256}, 'top100': {'unmatched_in_topk': 31, 'lirf_unmatched_in_topk': 18, 'matched_in_topk': 69, 'sse_share_of_total': 0.17522046685171075}, 'top500': {'unmatched_in_topk': 68, 'lirf_unmatched_in_topk': 20, 'matched_in_topk': 432, 'sse_share_of_total': 0.28664536402358254}}

Per-airport unmatched:

| Airport | n | RMSE | SSE | % total SSE |
|---|---:|---:|---:|---:|
| LIRF | 88 | 2786 | 6.831e+08 | 0.1% |
| LFPG | 313 | 743 | 1.730e+08 | 0.0% |
| EDDM | 130 | 917 | 1.094e+08 | 0.0% |
| LTFM | 330 | 427 | 6.018e+07 | 0.0% |
| EGLL | 127 | 432 | 2.366e+07 | 0.0% |
| EHAM | 315 | 261 | 2.145e+07 | 0.0% |
| LSZH | 101 | 343 | 1.185e+07 | 0.0% |
| LEMD | 60 | 366 | 8.042e+06 | 0.0% |
| EDDF | 96 | 217 | 4.523e+06 | 0.0% |
| LEBL | 89 | 191 | 3.261e+06 | 0.0% |

**Recovery preview (causal service profiles, train-only):** L1 coverage 68.2%, service RMSE(all unmatched) **3095** vs E20 unmatched 816 (non-LIRF 667, LIRF 13099)


## SSE ceilings (what perfect recovery buys, Jan+Jul)

| scenario | overall RMSE |
|---|---:|
| E20 now | 368.0 |
| perfect non-LIRF unmatched | 317.7 |
| perfect LIRF unmatched | 305.8 |
| perfect ALL unmatched (= matched-only bound) | 242.8 |

The **397 LIRF unmatched rows alone carry ~30% of total SSE**; the
top-10 SSE rows are 32% of total SSE and 7 of them are LIRF unmatched.
The leaderboard metric is dominated by a few dozen LIRF gate-delay bombs.

## Classification (why unmatched)

`no_NM_candidate_all_nm_cols_null` = 100% of unmatched; `partial_NM_match` = 0.
Unmatched is a **total NM flight-record miss**, not a fuzzy match. For these
rows the NM off-block anchors (AOBT/LOBT/IOBT/EOBT) are ALL absent, so E27's
source-disagreement signal is unavailable and no off-block anchor exists.

## Recovery preview verdict

- non-LIRF unmatched: service-median RMSE 1583 vs E20 1546 (Jan+Jul) — no gain;
  Dec 667 vs E20 516 — worse. Movement-only service profiles do not beat E20.
- LIRF unmatched: service-median 13391 vs E20 `MVT−SCHED` 6033 — far worse
  (these are gate-delay bombs, median taxi for the segment is ~1400 s).
