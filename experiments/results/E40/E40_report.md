# E40 — regime break / tail calibration: report

## Baseline
v9 = `likable-eagle_v9.parquet`, **LB 288.9003**.

## STEP 10/13 — is v9 tail-compressed?
Spearman(v9, true) = **0.86** on both splits (v9 ranks well). Predicted-vs-actual
quantile ratios (actual/predicted): p50 0.96, p75 1.00, p90 1.03, p95 1.05,
**p99 1.15** (Jan+Jul); Dec p99 1.17. So v9 is **mildly under-dispersed at the
extreme (p99 ~15% low)** — real, but small.

## Tail calibration (fit cross-month: Jan↔Jul; Jan+Jul→Dec)
| candidate | Jan+Jul matched | top-5% SSE | Dec matched |
|---|---:|---:|---:|
| E20/v9 base | 244.76 | 1.264e10 | 214.78 |
| isotonic calibration | 271.85 | 1.750e10 | 215.17 |
| LightGBM residual correction (e20,P) | 243.88 | 1.260e10 | 215.46 |
| P-scaled residual (P·h(X)) | **243.69** | 1.257e10 | 215.47 |

Best calibration improves matched by ≤1.1 s, **leaves top-5% SSE unchanged
(−0.6%)**, and **regresses December**. Isotonic overfits the tail badly.

## Decision
**No `likable-eagle_v10.parquet`.** E40's central hypothesis — v9 gets the
extreme rows roughly right but compresses magnitude, so calibration yields a
large SSE cut — is **falsified**: ranking is already good (ρ=0.86), compression
is modest (p99 +15%), and correcting it does not reduce top-5% SSE and hurts
December. Regime/scale representation of P cannot produce a major gain.
Artifacts: `run_e40_tail_calibration.py`, `experiments/results/E40/`.
