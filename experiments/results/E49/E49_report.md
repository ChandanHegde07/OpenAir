# E49 — deeper matched tail

v11-like gate ref: Jan+Jul 242.49 / Dec 213.17

| split | cand | matched | ΔE20 | top1 | overall |
|---|---|---:|---:|---:|---:|
| janjul | e20 | 244.76 | +0.00 | +0.0% | 368.03 |
| janjul | ghat_l0.5_h200 | 242.49 | -2.27 | -4.7% | 366.55 |
| janjul | ghat_l0.6_h200 | 242.43 | -2.33 | -5.2% | 366.51 |
| janjul | ghat_l0.7_h200 | 242.52 | -2.24 | -5.5% | 366.57 |
| janjul | g85_l0.5_h200 | 242.48 | -2.28 | -4.6% | 366.54 |
| janjul | v11_lirf_l0.3 | 242.76 | -2.00 | -4.6% | 366.73 |
| janjul | v11_both_l0.3 | 242.56 | -2.20 | -4.9% | 366.59 |
| janjul | gcb_l0.5 | 241.92 | -2.84 | -5.4% | 366.18 |
| janjul | nnls4_g0.5 | 240.31 | -4.45 | -8.7% | 365.13 |
| janjul | grec_l0.5 | 239.32 | -5.44 | -13.1% | 364.50 |
| janjul | nnls4 | 238.65 | -6.10 | -12.7% | 364.06 |
| janjul | grec_l0.6 | 239.46 | -5.30 | -14.0% | 364.59 |
| janjul | grec_l0.4 | 239.60 | -5.16 | -11.7% | 364.67 |
| janjul | rec_l0.4 | 239.98 | -4.78 | -11.1% | 364.92 |
| janjul | grec_l0.7 | 240.02 | -4.74 | -14.2% | 364.94 |
| dec | e20 | 214.78 | +0.00 | +0.0% | 228.69 |
| dec | ghat_l0.5_h200 | 213.17 | -1.61 | -4.1% | 227.20 |
| dec | ghat_l0.6_h200 | 213.11 | -1.67 | -4.4% | 227.14 |
| dec | ghat_l0.7_h200 | 213.15 | -1.63 | -4.4% | 227.18 |
| dec | g85_l0.5_h200 | 213.17 | -1.61 | -4.0% | 227.20 |
| dec | v11_lirf_l0.3 | 213.47 | -1.31 | -3.2% | 227.48 |
| dec | v11_both_l0.3 | 213.17 | -1.60 | -4.1% | 227.20 |
| dec | gcb_l0.5 | 212.93 | -1.85 | -4.4% | 226.97 |
| dec | nnls4_g0.5 | 212.78 | -2.00 | -4.2% | 226.83 |
| dec | grec_l0.5 | 212.50 | -2.27 | -3.8% | 226.58 |
| dec | nnls4 | 211.89 | -2.88 | -4.9% | 226.01 |
| dec | grec_l0.6 | 212.48 | -2.30 | -3.7% | 226.56 |
| dec | rec_l0.5 | 212.48 | -2.30 | -3.2% | 226.56 |
| dec | rec_l0.4 | 212.51 | -2.27 | -3.2% | 226.58 |
| dec | grec_l0.7 | 212.59 | -2.18 | -3.3% | 226.66 |

**GO vs v11:** rec_l0.4, grec_l0.4, cb_l0.4, rec_l0.5, grec_l0.5, cb_l0.5, gcb_l0.5, rec_l0.6, grec_l0.6, cb_l0.6, gcb_l0.6, rec_l0.7, grec_l0.7, cb_l0.7, gcb_l0.7, nnls4, nnls4_g0.5

The v11-like residual-vs-e20 gate saturates (~−2 s). Predicting **Δ = y − (MVT−AOBT)** with E36 clocks **and e20 as a feature**, then `e20 + λ(P+Δ − e20)` gated, is the larger step (top-1% SSE **−13%**).

**v12** = v8 unmatched + grec λ=0.5 with e20-in-Δ. `submissions/likable-eagle_v12.parquet` (36,366 matched rows). No LIRF unmatched G.
