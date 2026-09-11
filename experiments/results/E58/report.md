# E58 — Phase 1 structural forensics

## Candidate identities (audit result)
- T = MVT - BLOCK (target).
- (MVT-SCHED) - (BLOCK-SCHED): closed E33/E56.
- (MVT-AOBT) - (BLOCK-AOBT): closed E48-E51.
- Clock decompositions: all closed E56.
- Turnaround chain (arr BLOCK -> dep AOBT, stand-linked): NEW audit.

## SSE by source-defined population (E20 baseline)

| population | n | frac | RMSE | SSE share |
|---|---:|---:|---:|---:|
| matched | 503,074 | 98.62% | 235.4 | 50.40% |
| unmatched | 7,022 | 1.38% | 1976.7 | 49.60% |
| matched_allclocks | 503,074 | 98.62% | 235.4 | 50.40% |
| matched_arrival_linked | 411,043 | 80.58% | 236.3 | 41.51% |
| matched_arrival_unlinked | 92,031 | 18.04% | 231.2 | 8.89% |

arrival-linked matched: 411,043 rows; corr(arr_taxi, dep_taxi)=0.123; corr(arr_taxi, e20-resid)=-0.008

## Verdict (Phase 1)
No population defined purely by source availability / turnaround linkage shows an E33-scale SSE concentration beyond the already-exploited LIRF unmatched (E33) slice, and arrival-side taxi has ~no correlation with departure taxi/residual. No credible second hidden-process identity -> per E58 stop rule, no Phase-2 model.
