# E53 — stage-3 hard-tail residual cascade — REJECT (killed quickly)

**Idea (crazy mode):** v13 = e20 + 0.5(P+Δ−e20)·gate + 0.25·hat. Train stage-3 LGB on `r3 = y − v13` with synthetic hard-tail amplification (weights ∝|r3|, top-10% ×3), apply as `v13 + λ·r3hat·gate` (gate = |r3hat|>200 or e20>p90).

**Result (cross-month OOF):** v13 baseline reproduced exactly (Jan+Jul 238.53 / Dec 212.08). Best stage-3 variant: `v13_g0.1` Jan+Jul **239.06 (worse)**, top1 −0.3%; Dec `v13_lam0.1` **212.19 (worse)**, top1 −0.6%. No λ/gate combo improves matched or tail.

**Decision: abandon.** The stage-3 residual-of-residual is not predictable; the top-1% tail (40%+ of matched SSE) does not respond to a third correction stage, matching every prior hard-tail attempt. **No v15.** Artifacts: `experiments/run_e53_hardtail_cascade.py`, `experiments/results/E53/`.
