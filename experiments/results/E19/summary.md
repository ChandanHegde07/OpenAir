# E19 — RMSE tail reduction (causal local queue state)

Baseline E18-H: Jan+Jul overall 372.36 / matched 250.98; Dec 238.01 / 223.95.

| Variant | Jan+Jul overall | Δ vs E18-H | matched | >300s SSE share | Dec overall | Δ Dec |
|---|---:|---:|---:|---:|---:|---:|
| E19-0 | 372.36 | +0.00 | 250.98 | 89.0% | 238.01 | +0.00 |
| E19-A | 371.99 | -0.38 | 250.18 | 89.0% | 235.39 | -2.62 |
| E19-B | 371.14 | -1.23 | 249.35 | 89.0% | 233.67 | -4.34 |
| E19-C | 372.22 | -0.15 | 250.64 | 88.9% | 236.40 | -1.61 |
| E19-D | 371.95 | -0.41 | 250.24 | 89.1% | 236.12 | -1.89 |
| E19-E | 371.05 | -1.32 | 249.09 | 89.0% | 233.69 | -4.32 |
| E19-F | 371.02 | -1.34 | 249.22 | 89.0% | 233.67 | -4.34 |

## Decision: WEAK (1-3s gain, unstable)

best E19-F: Jan+Jul 372.36 -> 371.02 (Δ-1.34, matched 250.98 -> 249.22 Δ-1.76); Dec 238.01 -> 233.67 (Δ-4.34).

Tail concentration (Jan+Jul, matched>300 s SSE share is the key row above). Full tail table in `tail_metrics.csv`.

Top features per family (E19-F, Jan+Jul):
- e16a: dis_state_30m, dis_frac20_30m, dis_frac30_30m, dis_state_vs_hour
- E19-A: e19_aobt_mean3, e19_aobt_p90_5, e19_sched_mean3, e19_aobt_p1
- E19-B: e19_rwy_ae_med3, e19_rwy_aobt_p90_5, e19_rwy_aobt_mean3, e19_rwy_ae_mean3
- E19-C: e19_shock_rwy, e19_shock_med5, e19_shock_p1, e19_shock_mean3
- E19-D: e19_dep_30m, e19_rwy_dep_30m, e19_inv_rwy_gap1, e19_arr_30m
- E19-E: e19_qp, e19_qp_rwyd, e19_rwy_dis_pressure, e19_arr_rwy_pressure
- E19-F: e19_rwy_ndly_5, e19_tailprod3, e19_tailprod1, e19_max_ae_10
- baseline: aobt_eobt, STAND_mvt, mvt_sched, AIRCRAFT_OPERATOR_flt

Artifacts: metrics.csv, ablation.csv, per_airport.csv, tail_metrics.csv, feature_importance.csv, plots/.
