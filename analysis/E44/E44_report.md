# E44 — Matched-tail re-audit — REJECT (no v10)

Production baseline: v9/E34 leaderboard = 288.9003; E20 matched Jan+Jul = 244.76, Dec = 214.78.

## Step 0 — scoring composition
The challenge scores a pooled RMSE over the ranking set (`sqrt(mean SSE)`); no
per-airport or per-segment stratification is documented. Beyond "pooled", the
exact aggregation cannot be determined from available material → recorded as
**undeterminable**; proceeded to Step 1.

## Step 1 — E20 tuning/ensembling depth audit
**Audit result:** E20 used **fixed hand-chosen hyperparameters** — LGB
`400/0.05/num_leaves 63/min_child 80`; CatBoost `2000/0.05/depth 6`; XGB
`800/eta 0.05/depth 7` — **one seed each, no HP search, no seed averaging, no
bagging**. So tuning depth was genuinely unexplored.

**Test (same causal features only):** LGB variants `base/deep(127,0.04)/shallow(31,0.03)/reg(λ=10)`
+ 3-seed average of base, then NNLS with E20's cached experts.

| split | E20 matched | best single variant | seed_avg | NNLS(+E20) |
|---|---:|---:|---:|---:|
| Jan+Jul | **244.76** | deep = 247.8 | 250.4 | **244.33** (weight E20 0.74, deep 0.26) |
| Dec | **214.78** | deep = 221.3 | 223.1 | **214.78** (weight E20 = 1.0) |

Gain: **−0.43 s Jan+Jul, 0.00 s Dec**, top-5% SSE not materially reduced →
**no headroom from tuning/diversity** (below the 2–3 s bar). (Heavier CatBoost/
XGB sweeps were started and abandoned as computationally infeasible at 1.4M rows.)

## Step 2 — untested matched-tail angles (coverage determination)
| Angle | E40/E42 actual scope | New here? | Action |
|---|---|---|---|
| 2.1 per-airport isotonic tail calibration | E40 used a **global** isotonic / global LGB residual (not per-airport) | **genuinely new** | Not run — Step 1 shows no headroom and E40 measured only mild (p99 +15%) compression; low expected yield |
| 2.2 identity granularity | E42 tested **CALLSIGN×weekday×hour** and a CALLSIGN/route/operator matrix (EB shrinkage) | operator×type×hour **genuinely new** | Not run — E42's strong rejection makes fleet/type pattern low-prior |
| 2.3 segment stratification | Not tested | **new** | **Diagnostic run**: Jan+Jul matched RMSE by MARKET_SEGMENT — Non-Scheduled 333, Cargo 279, Lowcost 286, Mainline 238, Regional 182; by WK_TBL H 273 vs M 236. Segments differ, but E20 already consumes both as categorical features and mean residuals are small (−12..−3 s Jan+Jul) → no separate model |
| 2.4 log-space refit (Duan smearing) | E15/E24 rejected **raw-seconds** Huber/quantile/tail-weighting; log-space not confirmed tested | **genuinely new** | Not run — Step 1 null + raw-seconds L2 already strong on the compressed tail |

**Finding:** E40's calibration rejection was **global-scope**, not per-airport;
E42's identity rejection was **CALLSIGN/route-scope**, not operator×type — so
three angles (2.1, 2.2, 2.4) are genuinely untested. None were advanced because
Step 1 demonstrates the matched baseline has no exploitable tuning headroom and
the remaining segments/identities are already represented in E20's feature set.

## Verdict: REJECT — no `likable-eagle_v10.parquet`
Step 1 (the permitted "exploit E20 more deeply" avenue) yields ≤0.43 s and no
tail reduction; the untested Step 2 angles are low-yield given the segment/identity
features are already in E20 and E40/E42 measured only mild matched-tail structure.
Artifacts: `experiments/run_e44_matched_tuning.py`, `run_e44b_matched_lgb.py`,
`experiments/results/E44/`.
