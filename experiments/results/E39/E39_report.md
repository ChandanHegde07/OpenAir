# E39 — off-block delta: report

## Baseline
v9 = `likable-eagle_v9.parquet`, **LB 288.9003**. Identity `T = P + Δ`,
`P = MVT−AOBT`, `Δ = AOBT−BLOCK`.

## Delta forensics / baselines (matched, cross-month)
| model | Jan+Jul matched | Dec matched |
|---|---:|---:|
| E20/v9 | 244.76 | 214.78 |
| B0 `P` | 428.17 | 374.60 |
| B1 `P + global mean Δ` | 428.18 | 374.14 |
| B2 `P + airport-mean Δ` | 409.14 | 356.29 |
| B4 `P + LightGBM Δ` | **248.66** | 219.94 |

`Δ` = mean −19 s, sd 375 s → dominated by airport-dependent reporting bias
plus noise. Best learned Δ (B4) still leaves matched RMSE 248.66, **worse than
E20 (244.76)** on Jan+Jul and worse on Dec (219.94 vs 214.78).

## Decision
**No `likable-eagle_v10.parquet`.** Δ is not predictable enough to turn E38's
clock-proxy signal into a leaderboard gain; `P + Δ_hat` does not beat E20. E38's
modest clock-expert gain came from adding the clock proxies *inside* the
E20-feature residual model, not from modelling Δ from P — and that gain was
split-unstable. E39 is exhausted. Artifacts: `run_e39_delta.py`,
`experiments/results/E39/`.
