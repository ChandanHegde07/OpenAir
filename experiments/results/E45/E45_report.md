# E45 — stand-release G-regime (LIRF unmatched)

Baseline = this run's unmatched-only CatBoost G (E35 / v9 recipe).
Production v9 LB **288.9003**. Identity `T = D − G`.

## Diagnostic (full-year unmatched LIRF, not a fit)

- n=1488, has_tv rate=0.429, corr(T,D)=0.896, corr(G,D)=-0.117
- reused: n=638, med G=2, med T=6965, med D=8283
- not reused: n=850, med G=3844, med T=1262, med D=5970
- always D RMSE=5496; hard mix D-if-reused else 1100 RMSE=5879
- D in (4000,9000): n=969, med G has/no=2/4140, med T has/no=5818/1196, RMSE D=3830, mix=2909

## Holdout results

| split | cand | LIRF_u RMSE | overall RMSE | 4–9h D slice | top-50 SSE share |
|---|---|---:|---:|---:|---:|
| janjul | v9 | 3753.1 | 331.26 | 2067.6 | 0.743 |
| janjul | A | 3683.1 | 330.35 | 1925.6 | 0.753 |
| janjul | B | 13161.3 | 541.44 | 3991.7 | 0.898 |
| janjul | C | 5645.7 | 360.88 | 2660.2 | 0.834 |
| janjul | D | 6032.7 | 368.03 | 4326.9 | 0.593 |
| janjul | tphys | 13393.3 | 547.96 | 2908.4 | 0.949 |
| dec | v9 | 2556.7 | 227.27 | 1927.6 | 0.989 |
| dec | A | 2551.9 | 227.24 | 1858.8 | 0.988 |
| dec | B | 12718.6 | 366.19 | 4624.7 | 0.986 |
| dec | C | 6430.1 | 264.84 | 2616.2 | 1.000 |
| dec | D | 2786.1 | 228.69 | 3178.7 | 1.000 |
| dec | tphys | 13237.8 | 375.84 | 4382.2 | 0.993 |

## Feature importance (A, stand-release only)

- janjul: `{'has_tv': 0.5663647511330349, 'G_ub': 8.590353448722363, 'T_lb': 4.28123930890268, 'n_dep_mvt': 0.4590567033652801, 'n_arr_blk': 0.16729684472337342, 'n_int': 0.6400907291967795, 'tv_hour': 0.9514816940448538, 'stand_remote': 0.18843185363372472}`
- dec: `{'has_tv': 0.2964199684474842, 'G_ub': 5.703534145272852, 'T_lb': 4.4740064495360485, 'n_dep_mvt': 0.4734699077977247, 'n_arr_blk': 0.5190159689739123, 'n_int': 0.9342658709864, 'tv_hour': 1.1060148482162218, 'stand_remote': 0.4835807606836207}`

## GO / NO-GO

- **A: NO-GO** — Jan+Jul overall +0.91 s, LIRF_u +70.0 / Dec +4.8, slice cut 6.9%
- **B: NO-GO** — Jan+Jul overall -210.18 s, LIRF_u -9408.2 / Dec -10161.8, slice cut -93.1%
- **C: NO-GO** — Jan+Jul overall -29.62 s, LIRF_u -1892.6 / Dec -3873.3, slice cut -28.7%

Rule: GO if LIRF_u drops on both splits, Jan+Jul overall drop ≥8 s, and the 4–9h slice RMSE cut vs v9 is ≥10%. NO-GO if December LIRF_u rises or Jan+Jul overall drop <5 s.

**Verdict: NO-GO — do not submit; Direction 2 (OpenSky) is next.**

## Interpretation

The delay-window stand-release **identity is real** (full-year unmatched LIRF: reused med G=2 s vs not-reused 3844 s; D-controlled 4–9 h slice hard-mix 2909 vs always-D 3830). It is **not new enough to beat v9's G model**.

v9 CatBoost already sees `D`, stand, hour, prefix, and surface windows. Adding `has_tv` / `G_ub` / `T_lb` is a 70 s LIRF-unmatched cut (3753→3683) and **0.91 s overall** on Jan+Jul; December LIRF_u 2557→2552. Slice cut 6.9% (bar was 10%). Feature gain is concentrated in `G_ub` and `T_lb` (CatBoost importance 8.6 / 4.3 on Jan+Jul); the binary `has_tv` is almost unused (0.57).

Hard mixture (C) and P(G<300) mixture (B) fail the same way E13 gates failed: incomplete recall of push-to-hold bombs. `has_tv` precision is high; recall is not. Squared error on the missed bombs dominates.

This closes “stand occupancy during `(SCHED, MVT)` as a deployable LIRF override.” Do not retune. The remaining LIRF unmatched error is still the missing off-block clock.

Artifacts: `experiments/run_e45_stand_release.py`, `experiments/results/E45/`.
