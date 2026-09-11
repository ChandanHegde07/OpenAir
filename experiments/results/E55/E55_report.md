# E55 — extreme-tail non-linear stacking + magnitude-conditioned blend — REJECT

**Math:** gather diverse predictors (e20, Δ-rec1/rec2, q85-quantile Δ, residual
hat), train a hard-amplified non-linear LightGBM stacker on y, blend into v13
with magnitude-conditioned λ (piecewise in e20 bins) or a tail gate.

**Result (cross-month OOF):** v13 baseline 238.71 matched Jan+Jul / 212.03 Dec.
Every stacker/gated/magnitude blend is **worse** (best-gated variant matched
250.94, top1 +15.9% worse; Dec 215.54). No improvement on matched or the tail.

**Decision: abandon.** Non-linear stacking of all available previous predictors
cannot beat v13; the conditional mean over this information set is already v13's.
Combined with E51 (quantile mixing, rejected on LB) and E40 (calibration), the
mathematical levers available on the 30 columns are exhausted. **No v15.**
Artifacts: `experiments/run_e55_math.py`, `experiments/results/E55/`.
