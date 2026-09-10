# E32 — surface-state reconstruction: report

## Baseline (E20)
| split | overall | matched | unmatched | LIRF unmatched |
|---|---:|---:|---:|---:|
| Jan+Jul 2025 | 368.0 | 244.8 | 2214.1 | 6033 |
| Dec 2025 | 228.7 | 214.8 | 816.2 | 2786 |
| leaderboard | | | | **316.9654** |

## Best E32 model
Unified DEP+ARR causal event stream; airport/runway/stand windows (1–60 min),
arrival→departure interaction, time-since features, temporal derivatives,
arrival/departure ratios, sequence signature; LightGBM models for
matched / unmatched / LIRF-unmatched; conditional NNLS blend with E20.

| split | model | overall | matched | unmatched | LIRF unmatched |
|---|---:|---:|---:|---:|---:|
| Jan+Jul | E20 | 368.0 | 244.8 | 2214.1 | 6033 |
| Jan+Jul | surface (standalone) | 490.1 | 343.0 | 2823.5 | — |
| Jan+Jul | **E20 + surface (NNLS)** | **359.1** | 244.8 | **2118.4** | **5548** |
| Dec | E20 | 228.7 | 214.8 | 816.2 | 2786 |
| Dec | E20 + surface (NNLS) | 228.7 | 214.8 | 816.2 | 2786 |

**Estimated SSE improvement (Jan+Jul):** total 4.665e10 → 4.442e10 (−4.8%).
**Estimated LB if it transfered:** ≈ 308 — still >300.

## Blocker
The gain is **Jan+Jul-specific**. On December the surface unmatched model
degrades (1796 vs 816) and the NNLS weight goes to zero, so the blend is
identical to E20. Weights fit on Jan+Jul do not transfer, and the ranking
analogue is Jan–Jul 2026 — but the December collapse means we cannot claim a
robust improvement.

## Decision
**No `likable-eagle_v7.parquet`.** Expected leaderboard under the best case is
≈308 (not <300), and the December non-transfer means the true expected value
is closer to E20's 316.97. Creating v7 would violate E32's own submission rule.

Artifact: `experiments/run_e32_surface_state.py`, `experiments/results/E32/E32_surface.json`.
