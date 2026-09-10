# E31 — leaderboard attack (<300): report

**No candidate is expected to beat E20 on the leaderboard; no submission generated.**
Producing `likable-eagle_v7.parquet` would mean submitting a model known to be
worse than E20 on temporal validation, which violates E31's own decision rule.

## Baseline (E20)
| split | overall | matched | unmatched | LIRF-unmatched | non-LIRF-unmatched |
|---|---:|---:|---:|---:|---:|
| Jan+Jul 2025 | 368.03 | 244.76 | 2214.1 | 6032.7 | 1545.9 |
| Dec 2025 | 228.45 | 214.78 | 816.2 | 2786.1 | 515.8 |
| leaderboard | — | — | — | — | **316.9654** |

## E31 candidates tested (all worse)
| candidate | Jan+Jul matched | verdict |
|---|---:|---|
| E20 expert A (baseline) | **250.98** | reference |
| A + causal service-identity features (train-only tables, shrinkage) | 252.08 | **worse** |
| Movement-only dedicated unmatched (E29) | 1561 non-LIRF (vs 1546) | worse |
| LIRF movement-only / clip / blend (E29) | 9815 / worse / inconsistent | worse |
| Service-median recovery (C24) | 1583 non-LIRF, 13391 LIRF | worse |
| Matched-tail source-disagreement correction (C25) | ungated 690; gated inert | rejected |

Features tried this phase: service × {airport, hour, weekday, month, runway, ADES},
shrinkage to parent (m=20), historical mean/median/std/count. None reduced matched SSE.

## SSE arithmetic (why <300 needs ≥ ~17 s internal)
- internal 368.03 ↔ LB 316.9654 (offset 51.1).
- To reach LB 300 ⇔ internal ≈ 351 → cut total SSE ≈ 11% (≈0.5e10).
- Matched (2.031e10) would need ~25% cut (impossible); unmatched (2.634e10) needs
  ~19% cut. LIRF alone (1.445e10) would need RMSE 6033→~4500 with a causal model;
  no tested causal model beats MVT−SCHED (E29), and service features worsen the
  matched expert.

## Conclusion
Under the leakage rules and the provided fields, no validated candidate reduces
total SSE below E20. E20 remains production (LB 316.9654). **No `likable-eagle_v7.parquet`.**
The blocker is unmatched/LIRF SSE (C24/C28), which is unobservable from the 30
provided columns; matched service features are null-or-negative.

Artifacts: `experiments/run_e31_service_expert.py`, `experiments/results/E31/E31_service_expert.json`, this report.
