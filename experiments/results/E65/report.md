# E65 — causal airport event-graph transformer — REJECT (no signal)

Minimal end-to-end PyTorch causal event-graph transformer: 1 layer / 2 heads /
64-dim, context = previous 32 same-airport events (strictly earlier MVT),
numeric node features + time-delta + mask, scalar head predicting y − e20;
train Jan (120k matched subsample), eval Jul, alpha blend into e20.

| metric | E20 | E20 + neural (α=0.5) |
|---|---:|---:|
| val matched RMSE | 253.23 | 253.08 (−0.15) |
| top-1% SSE | — | +0.6% |

Improvement ~0.15 s and tail unchanged — well below the ≥1 s bar. Consistent
with the project's full-scale sequence precedent (temporal_v2 TCN, LB v5 worse
than v4). A full MoE/gated variant at scale would require very heavy compute for
the same expected null. **No v15.** Artifacts:
`experiments/run_e65_causal_event_graph.py`, `experiments/results/E65/metrics.json`.
