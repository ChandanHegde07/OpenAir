# E47 — OPDI-in regime + EGLL unmatched

| split | cand | LIRF_u | overall | overall+EGLL med |
|---|---|---:|---:|---:|
| janjul | v9 | 3753.1 | 331.26 | 331.82 |
| janjul | A_opdi_sr | 3637.9 | 329.78 | 330.34 |
| janjul | B_twohead | 4299.8 | 338.83 | 339.38 |
| janjul | C_outD | 4326.9 | 339.23 | 339.78 |
| janjul | D | 6032.7 | 368.03 | 368.53 |
| janjul | EGLL e20/med | 1103.4/1220.9 |  |  |
| dec | v9 | 2556.7 | 227.27 | 227.34 |
| dec | A_opdi_sr | 2464.9 | 226.73 | 226.80 |
| dec | B_twohead | 2548.6 | 227.22 | 227.29 |
| dec | C_outD | 2293.3 | 225.77 | 225.84 |
| dec | D | 2786.1 | 228.69 | 228.77 |
| dec | EGLL e20/med | 431.6/479.3 |  |  |

## GO/NO-GO

- A_opdi_sr:lirf: GO=False Jan+Jul +1.48 Dec +0.54
- A_opdi_sr:plus_egll: GO=False Jan+Jul +0.92 Dec +0.47
- B_twohead:lirf: GO=False Jan+Jul -7.57 Dec +0.05
- B_twohead:plus_egll: GO=False Jan+Jul -8.12 Dec -0.03
- C_outD:lirf: GO=False Jan+Jul -7.97 Dec +1.50
- C_outD:plus_egll: GO=False Jan+Jul -8.52 Dec +1.42

**Holdout:** A (stand-release + in_opdi) improves **both** splits vs v9: Jan+Jul overall 331.26→329.78 (−1.48 s), LIRF_u 3753→3638; Dec 227.27→226.73 (−0.54 s), LIRF_u 2557→2465. Below the old 5–8 s bar, but it is the only leftover-SSE candidate that transfers to December.

EGLL unmatched median and two-head/OUT→D gates fail Jan+Jul. Non-LIRF unmatched medians and LTFM D−G lose to E20. ISR/ETH unmatched LIRF is essentially T=D (RMSE≈4 s) on both holdouts; v10 forces T=D on those prefixes (5 ranking rows).

**Submission:** `likable-eagle_v10.parquet` = v9 with 382 LIRF-unmatched rows replaced. Ranking in_opdi rate 0.679 using OPDI 2026-01/07 open lists.

**Leaderboard: 296.321 (WORSE than v9 288.9003, +7.42 s).** Extra SSE ≈ 1.50e9. v10 moved **away from D** (RMSE vs D 2714 vs v9 2444) and **down** vs v9 on WMT/RYR (median 5570/4854 vs v9 6834/6113 vs D 7022/6661). Ranking unmatched LIRF is more bomb-like than 2025 Jan+Jul IN (med T 1370); v10 over-shrunk taxi. ISR/ETH force-D was only 5 rows, not the damage.

**Verdict: REJECT v10. Production remains v9 (LB 288.90). Do not stack more LIRF unmatched G features.**
