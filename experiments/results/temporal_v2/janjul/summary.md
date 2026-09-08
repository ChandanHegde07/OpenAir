# Temporal v2 — janjul

E20: rmse=368.03  matched=244.76  mae=157.05  >30m=774.8  >60m=2361.4  sse>1800=0.506

| Model | Overall RMSE | Matched | MAE | >30m matched | >60m matched |
|---|---:|---:|---:|---:|---:|
| E20 | 368.03 | 244.76 | 157.05 | 774.8 | 2361.4 |
| temporal_alone | 539.86 | 275.69 | 155.74 | 992.7 | 3557.6 |
| blend_w0.5 | 405.39 | 251.46 | 152.37 | 857.0 | 2882.1 |
| blend_w0.75 | 370.97 | 245.75 | 153.74 | 808.0 | 2595.9 |
| blend_w0.9 | 365.20 | 244.58 | 155.48 | 786.0 | 2448.0 |
| E20_plus_temporal | 369.71 | 247.15 | 156.31 | 797.2 | 2335.6 |
| E20_plus_0.5corr | 365.84 | 241.65 | 154.43 | 773.3 | 2326.3 |
| E20_plus_0.75corr | 367.02 | 243.33 | 154.79 | 782.2 | 2325.4 |
| E20_plus_1.0corr | 369.71 | 247.15 | 156.31 | 797.2 | 2335.6 |

Honest best on this split: **E20 + 0.5 TCN correction** (365.84 / matched 241.65, Δ −2.19 / −3.11).

`blend_w0.9` is **not** a valid winner: most of its overall gain came from mixing the TCN into LIRF unmatched (freeze those rows → 367.83). Full residual overshoots (369.71).

**Leaderboard** `likable-eagle_v5.parquet` (E20 v4 + 0.5 corr): **317.4747** vs E20 v4 **316.9654**. REJECT. Production remains v4.

