# E34 — v7+ latent gate-delay model: report

## Baseline
- v7 = `likable-eagle_v7.parquet`, **LB 292.9926** (Jan+Jul internal 345.2, LIRF_u 4717; Dec 226.8).
- v7 reduces to `T = D − 0.60·G` (E20's LIRF-unmatched prediction is `D = MVT−SCHED`).

## Best E34 candidate
Richer gate-delay model: CatBoost predicting `G = BLOCK−SCHED` on the LIRF
slice with enriched features (`d_log`, `d_rank` within LIRF, `d×arrival`,
`d×departure`, hour sin/cos, surface windows 5–30 min), reconstruct
`T = D − 1.0·G` (clamped ≥0). Applied to the LIRF-unmatched slice only; all
other rows keep their v7 prediction.

| month | v7 | E34 (this) |
|---|---:|---:|
| Jan+Jul LIRF-unmatched RMSE | 4717 | **3937** |
| Jan+Jul overall RMSE | 345.2 | **333.7** |
| Dec overall RMSE | 226.8 | 228.6 (≈ E20 228.7) |

Total SSE reduction (Jan+Jul): 4.665e10 → 3.908e10 (**−1.6e9, −3.4%**).

## Final blend
`final = D − G_hat` for the 383 LIRF-unmatched ranking rows; `= v7` elsewhere.

## Submission
`experiments/results/E34/likable-eagle_v8.parquet` (copy at repo root).
v7 with exactly **383 rows changed**; 344,841 rows; IDs match `submitting.parquet`;
no nulls; causal features only.

**Estimated leaderboard ≈ 281.5** (v7 internal→LB offset 52.2). This is a large
improvement over 292.99 but ~1.5 s above the 280 target; α=0.9 is a slightly more
conservative alternative (Jan+Jul LB est 282.6, Dec 227.6).

**Leaderboard RMSE: pending upload.**
