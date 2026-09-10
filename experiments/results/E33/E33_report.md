# E33 — latent delay decomposition: report

## Baseline (E20 = likable-eagle_v4, LB 316.9654)
| split | overall | matched | unmatched | LIRF unmatched |
|---|---:|---:|---:|---:|
| Jan+Jul 2025 | 368.0 | 244.8 | 2214.1 | 6033 |
| Dec 2025 | 228.7 | 214.8 | 816.2 | 2786 |

## Model (best candidate)
Identity: `TAXITIME = D − G`, `D = MVT−SCHED` (observed), `G = BLOCK−SCHED`.
A LightGBM regressor predicts **G** (not TAXITIME) on the LIRF slice from
prediction-time movement/surface features (`mvt_sched`, hour, weekday, month,
runway, stand, aircraft type, route, operator-prefix, dep/arr/runway/stand
surface windows). Reconstruction `T_hat = D − G_hat` (clamped ≥0), then a
robust α-blend with E20 on LIRF-unmatched only:

    final = 0.60 · (D − G_hat) + 0.40 · E20

α=0.6 minimises normalized SSE across both splits (robust, not overfit to one).

| split | E20 overall | E33 overall | E20 LIRF_u | E33 LIRF_u |
|---|---:|---:|---:|---:|
| Jan+Jul | 368.0 | **345.2** | 6033 | **4717** |
| Dec | 228.7 | **226.8** | 2786 | **2474** |

## Estimated leaderboard
Internal improvement −22.8 s (Jan+Jul); with the E20 internal→LB offset (51.1),
**estimated LB ≈ 294 < 300.** The LIRF-unmatched slice alone accounts for the
gain (−1.7e9 SSE, ~5.5% of total SSE); all other rows keep the E20 prediction.

## Submission
`experiments/results/E33/likable-eagle_v7.parquet` (copy at repo root).
E20 (likable-eagle_v4) with exactly **383 rows changed** (LIRF unmatched),
344,841 rows, IDs match `submitting.parquet`, no nulls/NaN, no forbidden fields.

Features/model: LightGBM G-regressor (600 trees, lr 0.04, leaves 31), surface
windows 5–30 min from the 2026 ranking DEP+ARR stream (causal), trained on all
2025 LIRF departures. Ranking used for inference only; no ranking targets.

**Leaderboard RMSE: pending upload.**
