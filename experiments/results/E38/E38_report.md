# E38 — latent off-block clock fusion: report

## Baseline
v9 = `likable-eagle_v9.parquet`, **LB 288.9003**.

## STEP 1 — clock forensics
The four NM off-block clocks (AOBT/LOBT/IOBT/EOBT) are **mutually identical**
for almost all rows (pairwise `clk_range` median = 0 in both normal and
extreme-taxi groups). So clock *disagreement* carries no new information —
E38's gain, where it exists, comes from the **absolute time-to-MVT of each
clock** (`MVT−LOBT`, `MVT−IOBT`, `MVT−EOBT`), i.e. additional taxi proxies E20
does not use (E20 uses only `MVT−AOBT` and `AOBT−EOBT`).

## STEP 15 — ablation (matched expert A)
| model | Jan+Jul matched | top-5% SSE | Dec matched |
|---|---:|---:|---:|
| base (E20 features) | 250.98 | 1.316e10 | 223.95 |
| +all off-block clocks | **243.51** | **1.204e10** | **221.81** |

Adding the clocks to the residual expert beats E20's ensemble matched (244.76)
in isolation and cuts top-5% SSE by 8.5%.

## Ensemble test (clock expert + cached E20 experts, matched NNLS)
Jan+Jul matched 244.76 → **240.69**, top-5% SSE 1.264e10 → 1.191e10; overall
368.03 → 365.38. Dec 214.78 → 214.38 (essentially flat) with very different
NNLS weights (janjul A=0.574, dec A=0.166) → **split-unstable**.

Two-component blend (E20 vs clock expert) grid: robust weight w=0.2 gives
Jan+Jul overall 366.56 (−1.5) with Dec neutral; w≥0.4 regresses December.

## Decision
**No `likable-eagle_v10.parquet`.** The only robust gain is ~−1.5 s internal
(est. LB ~287), below E38's bar. The stronger Jan+Jul setting (w≈0.5) is
Jan+Jul-overfit and degrades December, exactly the transfer risk E38 warned
about. The off-block-clock signal is real and tail-concentrated but too small
and too split-sensitive to justify a submission.

Artifacts: `run_e38_clock_fusion.py`, `run_e38b_ensemble.py`,
`experiments/results/E38/`.
