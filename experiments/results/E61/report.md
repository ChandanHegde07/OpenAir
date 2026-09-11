# E61 — causal historical analogue engine — REJECT

Built cross-month v13 (grec), then a strict-causal analogue DB (earlier months
only) with hierarchical exact-key retrieval (airport×rwy×stand×FLIGHT →
airport×rwy×stand → airport×rwy, EB-shrunk) and confidence = n/(n+2)/(1+sd/600),
applied as `v13 + α·analogue_residual` (α grid, gated/ungated).

| split | v13 | best analogue |
|---|---:|---:|
| Jan+Jul matched | 239.53 | 239.53 (Δ 0.00, top1 +0.1%) |
| Dec matched | 212.38 | 212.44 (Δ +0.05) |

Coverage ~99–100%, but the analogue residual prior is ~0: v13's residual is not
locally predictable from historical analogues of the same operational
configuration. Consistent with E42 (identity residual priors rejected) and
E26/E25-family results. **No v15.** Artifacts: `experiments/run_e61_analogue_engine.py`,
`experiments/results/E61/metrics.json`.
