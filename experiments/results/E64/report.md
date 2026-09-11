# E64 — runway departure service-process model — REJECT

Base v13 (grec). Causal service-process features: predicted headway (rolling
baseline), observed/predicted headway ratio, excess-headway sum/max over last 3,
positional index — plus E62 QF. Residual LGB (y − v13) cross-month OOF.

| split | v13 | v13+E64 (α=0.3) | Δ | top-1% |
|---|---:|---:|---:|---:|
| Jan+Jul | 239.5 | 239.15 | −0.38 | −0.8% |
| Dec | 212.4 | 212.13 | −0.25 | −0.2% |

Improvement is below the 0.5 s bar and top-1% SSE is essentially unchanged.
The service-process signal is almost fully absorbed by v13 (E62's gain was vs
E20; E63/E64 confirm redundancy on v13). **Decision: no v15.** Artifacts:
`experiments/run_e64_runway_service_process.py`, `experiments/results/E64/metrics.json`.
