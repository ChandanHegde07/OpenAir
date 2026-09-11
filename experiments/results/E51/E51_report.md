# E51 — more matched P+Δ

v13 ref: 239.85 / 211.97

| split | cand | matched | vs v13 | top1 | overall |
|---|---|---:|---:|---:|---:|
| janjul | e20 | 244.76 | +4.91 | +0.0% | 368.03 |
| janjul | v13 | 239.85 | +0.00 | -14.3% | 364.84 |
| janjul | g_cb_hfoc25 | 237.48 | -2.37 | -16.3% | 363.31 |
| janjul | g_cb_h20 | 237.57 | -2.28 | -16.2% | 363.37 |
| janjul | g_cb_h25 | 237.62 | -2.23 | -16.3% | 363.40 |
| janjul | g_cb_h30 | 237.74 | -2.11 | -16.4% | 363.47 |
| janjul | g_cb | 238.11 | -1.74 | -15.0% | 363.71 |
| janjul | g_mix_h25 | 238.78 | -1.07 | -15.2% | 364.14 |
| janjul | g_avg_hfoc25 | 239.11 | -0.74 | -14.6% | 364.36 |
| janjul | g_r_hfoc25 | 239.37 | -0.48 | -14.5% | 364.53 |
| dec | e20 | 214.78 | +2.81 | +0.0% | 228.69 |
| dec | v13 | 211.97 | +0.00 | -4.9% | 226.08 |
| dec | g_mix_h25 | 211.90 | -0.07 | -5.1% | 226.02 |
| dec | g_avg_h20 | 211.92 | -0.05 | -4.9% | 226.04 |
| dec | g_avg_h25 | 211.93 | -0.04 | -4.9% | 226.05 |
| dec | g_ravg_h25 | 211.93 | -0.04 | -4.9% | 226.05 |
| dec | g_r_h20 | 211.96 | -0.01 | -4.8% | 226.08 |
| dec | g_avg_h30 | 211.97 | -0.00 | -4.9% | 226.08 |
| dec | g_r_h25 | 211.97 | +0.00 | -4.9% | 226.08 |
| dec | g_r_h30 | 212.01 | +0.04 | -4.9% | 226.12 |

**GO vs v13:** none from seed-avg / CatBoost / focus-airport (CB helps Jan+Jul, hurts Dec).

**E51c quantile mix:** `0.7 L2 + 0.3 quantile(0.65)` Δ, then grec+hat:

| split | v13 | rec_mix_gh25 |
|---|---:|---:|
| Jan+Jul matched | 238.52 | 238.58 (tied) |
| Dec matched | 212.08 | **211.70 (−0.38)** |
| Dec overall | 226.19 | **225.83** |

**v14:** `submissions/likable-eagle_v14.parquet`. Unmatched = v8. No LIRF G. If it does not beat **284.10**, keep v13.
