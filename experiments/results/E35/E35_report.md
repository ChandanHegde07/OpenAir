# E35 — selection-aware latent gate-delay model: report

## Baseline
- v8 = `likable-eagle_v8.parquet`, **LB 288.9003** (Jan+Jul LIRF-unmatched 3937; Jan+Jul overall 333.7; Dec 228.6).

## Core idea
G is trained mostly on **matched** LIRF flights but deployed on **unmatched**
flights. Selection-aware fixes:
- propensity classifier `P(M=1|x)` (prediction-time features) → importance
  weights `P(M=0|x)/P(M=1|x)` (95/99th-pct clipped);
- **G trained on unmatched LIRF training rows only** (`unm_only`);
- enriched features (D-log, LIRF-relative D-rank, D×arrival/departure, hour
  sin/cos, surface 5–30 min windows).

## Results (LIRF-unmatched RMSE)
| model | Jan+Jul | Dec |
|---|---:|---:|
| v8 (all-LIRF CatBoost, α=1.0) | 3937 | 2766 |
| selection-weighted CatBoost | 3894 | 2641 |
| **unmatched-only CatBoost** | **3752** | **2557** |

`unm_only` is best on **both** splits (α=1.0). Jan+Jul overall 333.7 → **331.3**;
Dec 228.6 → **227.3**. Total Jan+Jul SSE 3.835e10 → **3.779e10** (−0.56e9).

## Final model
`T = D − G_hat` on the 383 LIRF-unmatched ranking rows, with `G_hat` from a
CatBoost trained on the 1,488 unmatched LIRF departures of 2025; all other rows
= v8.

## Submission
`experiments/results/E35/likable-eagle_v9.parquet` (copy at repo root).
344,841 rows; IDs match `submitting.parquet`; zero nulls; 383 rows changed.

## Estimated leaderboard
v8 internal→LB offset = 333.7 − 288.9003 = 44.8 → **estimated LB ≈ 286.5**.
This is a consistent improvement over 288.90 but **not yet <280**; the v7→v8
transfer ratio (internal −11.5 → LB −4.1) warns that LIRF-only internal gains
transfer partially. Submit at the team's discretion.

**Leaderboard RMSE: pending upload.**
