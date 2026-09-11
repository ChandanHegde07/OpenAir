# E54 — meta-layer corrector on v13 — REJECT

**Idea (beast mode):** turn prior-model history into features — individual E20
experts (pred_C/D/E), E20 blend, P, clocks, Δ-rec, residual hat, **seed-variance
of the Δ model**, P/e20 ratio — train a stage-3 LGB on `r3 = y − v13` with
hard-tail amplification; apply `v13 + λ·r3hat·gate`.

**Result (cross-month OOF):** v13 baseline ~237.8 matched Jan+Jul / 212.7 Dec.
All meta-correctors worse or neutral (best Jan+Jul 238.43, top1 −0.2%; Dec
212.86, top1 −0.5%). No λ/gate improves matched or the top-1% tail.

**Decision: abandon.** Adding the individual experts and model-uncertainty meta
features does not make the stage-3 residual predictable — consistent with E53.
The v13 matched tail is not further reducible from the 30 columns via stacking,
cascades, or meta-layers. **No v15.** Artifacts: `experiments/run_e54_metalayer.py`,
`experiments/results/E54/`.
