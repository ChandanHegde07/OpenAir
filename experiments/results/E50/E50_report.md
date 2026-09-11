# E50 — richer matched P+Δ

grec (v12-like) ref: Jan+Jul 239.02 / Dec 212.38

| split | cand | matched | vs grec | top1 | overall |
|---|---|---:|---:|---:|---:|
| janjul | e20 | 244.76 | +5.74 | +0.0% | 368.03 |
| janjul | grec_v12 | 239.02 | +0.00 | -13.7% | 364.30 |
| janjul | grec_hat_l0.25 | 238.52 | -0.50 | -14.8% | 363.98 |
| janjul | grec_hat_l0.15 | 238.55 | -0.47 | -14.5% | 364.00 |
| janjul | grec_hatg_l0.25 | 238.58 | -0.44 | -15.1% | 364.01 |
| janjul | grec2_l0.5 | 238.64 | -0.38 | -14.4% | 364.05 |
| janjul | grec_hatg_l0.15 | 238.64 | -0.38 | -14.7% | 364.05 |
| janjul | grec_hatg_l0.35 | 238.67 | -0.35 | -15.3% | 364.07 |
| janjul | grecm_l0.5 | 238.70 | -0.32 | -14.2% | 364.09 |
| janjul | grec_hat_l0.35 | 238.72 | -0.30 | -14.8% | 364.11 |
| dec | e20 | 214.78 | +2.40 | +0.0% | 228.69 |
| dec | grec_v12 | 212.38 | +0.00 | -3.9% | 226.46 |
| dec | nnls | 212.05 | -0.33 | -4.4% | 226.16 |
| dec | grec_hat_l0.25 | 212.08 | -0.30 | -4.5% | 226.19 |
| dec | grec_hatg_l0.25 | 212.10 | -0.27 | -4.8% | 226.21 |
| dec | grec_hat_l0.15 | 212.10 | -0.27 | -4.4% | 226.21 |
| dec | grec_hatg_l0.15 | 212.14 | -0.23 | -4.6% | 226.25 |
| dec | grec_hatg_l0.35 | 212.15 | -0.23 | -4.9% | 226.25 |
| dec | recm_l0.5 | 212.17 | -0.20 | -3.5% | 226.27 |
| dec | grec_hat_l0.35 | 212.18 | -0.20 | -4.5% | 226.28 |

**GO vs grec:** grec_hat_l0.15, grec_hat_l0.25, grec_hatg_l0.25

Rich clocks + segment cats make grec 239.02 / 212.38. Then **grec + 0.25 × residual-vs-e20** → 238.52 / 212.08 (overall 363.98 / 226.19). NNLS weights did not transfer.

**v13:** `submissions/likable-eagle_v13.parquet` = v8 unmatched + rich grec λ=0.5 + leftover hat λ=0.25. No LIRF unmatched G.
