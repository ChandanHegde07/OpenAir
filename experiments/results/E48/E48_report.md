# E48 — matched tail (no LIRF unmatched G)

Baseline E20 matched Jan+Jul 244.76 / Dec 214.78. Production LB is **v8 288.90**; v9 scored **300.95**.

| split | cand | matched RMSE | Δ | top1 SSE | overall |
|---|---|---:|---:|---:|---:|
| janjul | e20 | 244.76 | +0.00 | 8.500e+09 | 368.03 |
| janjul | delta_l0.3 | 243.49 | -1.27 | 8.128e+09 | 367.20 |
| janjul | glob_l0.3 | 244.22 | -0.54 | 8.451e+09 | 367.68 |
| janjul | glob_l0.2 | 244.23 | -0.53 | 8.457e+09 | 367.69 |
| janjul | ap_l0.2 | 244.47 | -0.29 | 8.484e+09 | 367.84 |
| janjul | ap_l0.3 | 244.70 | -0.06 | 8.498e+09 | 367.99 |
| janjul | glob_l0.5 | 244.70 | -0.06 | 8.471e+09 | 367.99 |
| janjul | glob_l0.0 | 244.76 | +0.00 | 8.500e+09 | 368.03 |
| janjul | ap_l0.0 | 244.76 | +0.00 | 8.500e+09 | 368.03 |
| dec | e20 | 214.78 | +0.00 | 2.488e+09 | 228.69 |
| dec | delta_l0.3 | 213.07 | -1.70 | 2.458e+09 | 227.11 |
| dec | delta_l0.5 | 213.26 | -1.52 | 2.478e+09 | 227.28 |
| dec | glob_l0.3 | 214.35 | -0.43 | 2.469e+09 | 228.29 |
| dec | glob_l0.2 | 214.39 | -0.39 | 2.471e+09 | 228.33 |
| dec | ap_l0.2 | 214.46 | -0.32 | 2.472e+09 | 228.39 |
| dec | ap_l0.3 | 214.51 | -0.27 | 2.472e+09 | 228.44 |
| dec | iso_l0.3 | 214.51 | -0.26 | 2.478e+09 | 228.45 |
| dec | glob_l0.5 | 214.56 | -0.22 | 2.475e+09 | 228.49 |

## GO

**E48 (per-airport isotonic / residual / Δ):** best `delta_l0.3` matched −1.27 / −1.70 s, below the 1.5 s Jan+Jul bar.

**E48b (E36 features, tail-gated residual):** `allgate_l0.5` **GO** on the stated bar:

| split | E20 matched | allgate λ=0.5 | top-1% SSE | overall |
|---|---:|---:|---:|---:|
| Jan+Jul | 244.76 | **242.49 (−2.27)** | −4.7% | 366.55 |
| Dec | 214.78 | **213.17 (−1.61)** | −4.1% | 227.20 |

Tail-trained-only models exploded (RMSE 400–1100). Gate must sit on a **full-population** residual, applied only when `e20>p90` or `hat>200`.

**v11** = v8 + gated `P+Δ` blend (λ=0.5) on matched ranking rows only; unmatched = v8. 40,308 rows changed (11.7%).

**Leaderboard: 287.7101** (v8 was 288.9003, **−1.19 s**). First score below 288.90. Production is v11. Do not restack LIRF unmatched G.
