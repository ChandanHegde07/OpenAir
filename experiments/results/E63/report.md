# E63 — continuous runway workload on v13 — REJECT

Base = v13 (grec). Causal continuous workload: per-runway session backlog_cum
(entering AOBT − served MVT, idle-gap reset) + service/queue features (E62 QF).
Residual LGB (y − v13) cross-month OOF; alpha blend.

| split | v13 | v13+E63 (α=0.2/0.3) | Δ | top-1% |
|---|---:|---:|---:|---:|
| Jan+Jul | 239.2 | 239.17 | −0.21 | −0.4% |
| Dec | 212.5 | 212.34 | −0.15 | −0.3% |

Continuous workload improves v13 by ~0.2 s on both splits — below the 0.5 s
noise threshold, and top-1% SSE essentially unchanged. v13 already absorbs the
workload signal (consistent with E52; E62's −2.17 gain was relative to E20,
which v13 already beats).

**Decision: no v15.** Artifacts: `experiments/run_e63_continuous_workload.py`,
`experiments/results/E63/metrics.json`.
