# E20 — Stacked residual ensemble

E18-H reproduction: Jan+Jul 372.36 / matched 250.98; Dec 238.01 / 223.95.

## Expert scores

| Expert | Jan+Jul | matched | Dec | matched |
|---|---:|---:|---:|---:|
| A | 372.36 | 250.98 | 238.01 | 223.95 |
| A2 | 371.14 | 249.35 | 233.67 | 220.11 |
| B | 394.63 | 283.08 | 237.51 | 223.62 |
| C | 370.37 | 248.25 | 232.37 | 218.56 |
| D | 385.50 | 266.24 | 246.17 | 232.70 |
| E | 370.52 | 247.76 | 229.94 | 215.90 |

## Blends (base set A,B,C,D,E)

| Blend | Jan+Jul | Δ vs E18-H | matched | Dec | Δ Dec |
|---|---:|---:|---:|---:|---:|
| E18-H | 372.36 | +0.00 | 250.98 | 238.01 | +0.00 |
| equal5 | 371.99 | -0.38 | 250.86 | 232.00 | -6.01 |
| A_centric | 372.54 | +0.17 | 251.57 | 235.02 | -3.00 |
| A_B_C_D | 374.25 | +1.88 | 254.16 | 235.04 | -2.97 |
| A_B_D_E | 374.11 | +1.74 | 253.93 | 234.85 | -3.16 |
| A_E | 369.99 | -2.37 | 247.49 | 233.50 | -4.51 |
| nnls | 368.03 | -4.33 | 244.76 | 228.45 | -9.56 |
| ridge1 | 368.17 | -4.19 | 244.93 | 229.60 | -8.41 |
| ridge10 | 368.17 | -4.19 | 244.93 | 229.60 | -8.41 |

## Decision: ACCEPT  (best=nnls)

best blend nnls: Jan+Jul 372.36 -> 368.03 (Δ+4.33); Dec 238.01 -> 228.45 (Δ+9.56)

Residual correlation (Jan+Jul matched): see error_correlation.csv / E20.json.

Artifacts: model_metrics.csv, blend_metrics.csv, error_correlation.csv, regime_metrics.csv, oof_predictions_{janjul,dec}.parquet, feature_importance.csv, plots/.
