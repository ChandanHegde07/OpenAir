# E62 — dynamic runway queue / service-rate model

## Model
Strictly-causal same-runway MVT-ordered features: rolling median headway
(5/10/20/50), service_rate = 60/median, service_slowdown, active runway queue
(AOBT≤t<MVT overlap), recent same-runway MVT counts, EOBT queue,
**expected_wait = queue_position × service_headway**, backlog (entering−served),
demand/supply ratio. Residual LGB (target y − base) cross-month OOF, α=0.5 blend.

## Result (baseline = E20; α=0.5)
| split | E20 matched | E20+E62 | Δ | top-1% |
|---|---:|---:|---:|---:|
| Jan+Jul | 244.76 | **242.59** | −2.17 | −5.3% |
| Dec | 214.78 | **212.91** | −1.87 | −4.8% |

**Positive finding:** expected-runway-wait features carry real signal beyond E20
and transfer (first such gain since E48).

## Comparison vs v13
v13 grec+hat matched = 238.5 (Jan+Jul), 212.1 (Dec) — **v13 already beats
E20+E62** (242.59). E52's precedent (congestion features injected into v13's Δ
model) gave ≈0.0 s, so E62 features are expected to be largely redundant with
v13's existing Δ structure.

## Decision
**No `likable-eagle_v15.parquet`** — E62 improves over E20 but not over v13.
Recommended follow-up: inject the QF service/expected-wait features into v13's
grec Δ model (E52-style harness) to confirm; prior suggests ≈0.
Artifacts: `experiments/run_e62_runway_queue.py`, `experiments/results/E62/metrics.json`.
