# E43 — Non-LIRF unmatched D−G opportunity — REJECTED

**Scope:** reapply the proven E33/E34 `D − G_hat` machinery to non-LIRF
unmatched airports. Production baseline v9/E34 = 288.9003.

## STEP 1 — per-airport unmatched diagnostic (Jan+Jul 2025, E20/v9 baseline)
| Airport | unmatched n | unmatched RMSE | ratio vs matched | % dataset SSE |
|---|---:|---:|---:|---:|
| **LFPG** | **880** | **3440.8** | **12.5** | **22.33%** |
| LIRF (already handled by E34) | 397 | 6032.7 | 13.7 | 30.97% |
| EHAM | 808 | 571.6 | 3.2 | 0.57% |
| LSZH | 524 | 611.8 | 3.1 | 0.42% |
| LTFM | 657 | 593.1 | 2.3 | 0.50% |
| LEBL/LEMD/EDDF/EDDM/EGLL | 96–496 | 217–918 | 1.1–4.1 | ≤0.09% (EDDM Dec 1.26%) |

Ranking unmatched rates: EHAM 3.16%, LSZH 2.85%, LFPG 1.84% (734 rows), others ≤1.5%.
Only **LFPG** is a material non-LIRF opportunity (22.3% dataset SSE) → Step 2 runs for LFPG.

## STEP 2 — D−G for LFPG unmatched (E34 CatBoost G, per-airport blend)
| split | E20 LFPG unmatched | best D−G blend |
|---|---:|---:|
| Jan+Jul | 3440.8 (SSE 1.042e10) | **3487.3** (worse) |
| Dec | 743.4 | **1873.6** (worse) |

**Mechanism:** LFPG unmatched truth is normal taxi (median **1026 s**, p90 1685,
only 1% >1 h) while `D = MVT−SCHED` is huge (median 5884) with
`corr(T, D) ≈ 0`. E20 already predicts near-normal values (mean 1139, max 2582);
its 22% SSE comes from a handful of **extreme true-taxi rows**, not schedule
displacement. Simple priors (global/runway/stand median) ≈ E20 (3441–3464), and
D−G is worse. No causal observable signal exists for those rows.

## Verdict: REJECT — no `likable-eagle_v10.parquet`
The diagnostic identifies LFPG, but the proven LIRF mechanism does not transfer
(LIRF's success relied on a real `corr(T, D)`; LFPG's extreme rows have none).
Recorded as E43 so it is not re-attempted. Artifacts:
`experiments/run_e43_unmatched_diagnostic.py`, `run_e43_lfpg_g.py`,
`experiments/results/E43/`.
