# E36 — matched-population attack: report

## Baseline
- v9 `likable-eagle_v9.parquet`, **LB 288.9003** (= v8's LB; the E35 LIRF-only change did **not** transfer).

## STEP 1 — where v9 loses SSE (matched rows; v9 == E20 on matched)
Jan+Jul matched n=339,046, RMSE 244.8, total SSE 2.031e10. Top 1% of rows = **41.8%** of matched SSE.
- **T quantiles:** T_q5 (highest-taxi) RMSE **1379**, **32%** of matched SSE; T_q0 24%.
- **D quantiles:** largest-D bin only ~10% — high schedule delay is *not* the main matched cost.
- **Airports:** LIRF 25%, LFPG 15%, LTFM 15%, EGLL 13%.
- **Source disagreement:** top 1% RMSE 795, 11% SSE.
Dec: T_q5 25% SSE; LFPG 17%, LIRF 15%.

**Top-SSE regime = high-taxi-time matched rows** (tail compression), concentrated in LIRF/LFPG/LTFM/EGLL.

## STEP 5 — direct v9 residual specialist (strict cross-month OOF)
Target `T_error = y − e20`; LightGBM on causal features; cross-month folds
(Jan→Jul, Jul→Jan; Jan+Jul→Dec); shrinkage λ.

| split | E20/v9 matched | λ=0.3 | λ=0.5 |
|---|---:|---:|---:|
| Jan+Jul | 244.76 | 243.07 | **242.91** |
| Dec | 214.78 | 213.56 | **213.32** |

SSE: Jan+Jul 2.031e10 → 2.000e10 (−0.31e9); Dec 7.567e9 → 7.464e9 (−0.10e9).
Consistent sign on both splits but small: overall ~331.3 → 329.9 (**−1.4 internal**),
**estimated LB ≈ 287.5**.

## Decision
**No `likable-eagle_v10.parquet`.** The best E36 candidate improves the est.
leaderboard by ~1.4 s (288.90 → ~287.5), below E36's minimum meaningful bar
(<285). Hard-case gating by predicted-correction magnitude did not beat plain
shrinkage (Jan+Jul gated 244.95). The matched tail is highly concentrated (top
1% = 42% SSE) but its remaining error is not predictable from the available
causal fields beyond a ~1.5 s shrinkage gain. Artifacts:
`run_e36_matched_sse.py`, `run_e36_matched_residual.py`, `E36_matched_sse.json`,
`E36_matched_residual.json`.
