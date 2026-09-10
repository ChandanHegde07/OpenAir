# E28 STEP 3 — matched tail regime (C25)

Matched rows only. E20 residual modelled; correction gated by a causal tail-risk score (off-block source disagreement + compact surface pressure). **Caveat:** E20 OOF exists only for holdout months, so tail models are fit by temporal cross-fit (Jan<->Jul; Jan+Jul->Dec; Dec->Jan+Jul) rather than on 2025 train months. Row-level out-of-sample, but the fit periods are holdout months.

Gate (P(tail) >= g) chosen on combined split SSE: g = 0.55

## Matched ablations

| Split | Model | matched RMSE | SSE | SSE |r|>30m | SSE |r|>60m |
|---|---|---:|---:|---:|---:|
| janjul | A E20 | 244.76 | 2.031e+10 | 4.834e+09 | 2.517e+09 |
| janjul | B +disagreement | 689.18 | 1.610e+11 | 7.199e+09 | 2.109e+09 |
| janjul | C +surface | 698.41 | 1.654e+11 | 7.933e+09 | 2.278e+09 |
| janjul | D +both ungated | 690.59 | 1.617e+11 | 6.876e+09 | 2.089e+09 |
| janjul | E gated tail | 244.65 | 2.029e+10 | 4.815e+09 | 2.490e+09 |
| dec | A E20 | 214.78 | 7.567e+09 | 9.531e+08 | 2.086e+08 |
| dec | B +disagreement | 602.96 | 5.963e+10 | 1.783e+09 | 6.272e+07 |
| dec | C +surface | 631.48 | 6.541e+10 | 3.065e+09 | 1.140e+08 |
| dec | D +both ungated | 589.50 | 5.700e+10 | 1.490e+09 | 8.809e+07 |
| dec | E gated tail | 214.92 | 7.576e+09 | 9.609e+08 | 2.086e+08 |

## Regimes — janjul (E20 vs gated E28)

| Regime | n | E20 RMSE | E28 RMSE | E20 SSE | E28 SSE |
|---|---:|---:|---:|---:|---:|
| normal | 339,012 | 244.1 | 244.1 | 2.020e+10 | 2.020e+10 |
| hard | 27 | 1720.1 | 1630.4 | 7.988e+07 | 7.177e+07 |
| extreme | 7 | 2063.1 | 1698.1 | 2.980e+07 | 2.018e+07 |

## Regimes — dec (E20 vs gated E28)

| Regime | n | E20 RMSE | E28 RMSE | E20 SSE | E28 SSE |
|---|---:|---:|---:|---:|---:|
| normal | 163,990 | 213.8 | 213.8 | 7.499e+09 | 7.499e+09 |
| hard | 22 | 1039.3 | 1039.3 | 2.377e+07 | 2.377e+07 |
| extreme | 16 | 1645.0 | 1822.0 | 4.329e+07 | 5.312e+07 |

## Source-disagreement buckets (Jan+Jul, E20 residual)

### abs_eobt

| bucket | n | tail>30m rate | RMSE | SSE share |
|---|---:|---:|---:|---:|
| q0 | 172,774 | 0.0% | 188.2 | 30.1% |
| q1 | 107,497 | 0.1% | 213.4 | 24.1% |
| q2 | 52,288 | 0.1% | 266.2 | 18.2% |
| q3 | 21,073 | 0.2% | 339.8 | 12.0% |
| q4 | 14,188 | 0.6% | 432.2 | 13.0% |
| q5 | 3,695 | 3.8% | 778.3 | 11.0% |

### max

| bucket | n | tail>30m rate | RMSE | SSE share |
|---|---:|---:|---:|---:|
| q0 | 177,031 | 0.0% | 188.3 | 30.9% |
| q1 | 94,148 | 0.0% | 208.0 | 20.0% |
| q2 | 54,608 | 0.1% | 260.1 | 18.2% |
| q3 | 19,921 | 0.2% | 331.4 | 10.8% |
| q4 | 14,509 | 0.6% | 411.5 | 12.1% |
| q5 | 3,552 | 5.3% | 906.4 | 14.4% |

