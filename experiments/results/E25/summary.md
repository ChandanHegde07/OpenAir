# E25 — One-shot structural rebuild

Single architecture: hierarchical p10 floor → log(y − floor) → per-airport LightGBM
→ Duan smearing → unmatched fallback. Not an incremental patch on E20.

## Schema (measured on training DEP, n=2,085,047)

- `STAND_mvt`: 0.0013% null, 1899 unique.
- `RUNWAY_mvt`: 0% null, 53 unique.
- `AIRCRAFT_OPERATOR_flt`: 1.08% null (equals unmatched), 676 unique.
- Smallest Jan+Jul-train airport: LSZH n=112,561.
  Full per-airport models are viable everywhere (E20 used a 20k cutoff; all ten exceed 110k).

## Ranking-safe clocks

Airport `BLOCK_TIME_UTC_mvt` is the target ingredient and is blank in ranking.
AOBT in this pipeline is NM `AOBT_3_flt`. Scored-row `MVT_TIME` is available
(post-ops reconstruction). Floors/encodings use training-split y only.

## Validation vs E20 (current production / leaderboard 316.9654)

| Split | E25 overall | E20 overall | Δ | E25 matched | E20 matched | Δ matched |
|---|---:|---:|---:|---:|---:|---:|
| Jan+Jul | 473.61 | 368.03 | +105.58 | 386.42 | 244.76 | +141.66 |
| December | 311.23 | 228.45 | +82.78 | 300.26 | 215.90 | +84.36 |

Unmatched policy locked: **lirf_sched_else_spec**.

## Per-airport RMSE (Jan+Jul)

| Airport | RMSE |
|---|---:|
| EDDF | 257.12 |
| EDDM | 279.03 |
| EGLL | 383.84 |
| EHAM | 238.30 |
| LEBL | 290.91 |
| LEMD | 259.42 |
| LFPG | 642.11 |
| LIRF | 1139.90 |
| LSZH | 314.90 |
| LTFM | 377.00 |

## Tail (Jan+Jul matched): <20 min 251.28 · >30 min 1242.06 · >60 min 4138.02

## Ablation (after the combined result)

| Variant | Jan+Jul overall | matched | Dec overall | matched |
|---|---:|---:|---:|---:|
| full (floor + log-excess + per-airport + encodings) | 473.61 | 386.42 | 311.23 | 300.26 |
| no_floor (`log(y)` + smearing, floor not in target/features) | 398.54 | 287.76 | 235.02 | 219.33 |
| raw_target (L2 on seconds, floor kept as a feature) | **394.01** | **281.36** | **232.83** | **216.96** |
| pooled (same log-excess target, one model, airport cat) | 479.79 | 394.09 | 313.33 | 302.45 |
| no_enc (log-excess, drop the three JS encodings) | 484.32 | 399.67 | 309.37 | 298.32 |

What actually moved RMSE:

1. **The p10-floor log-excess target is the failure mode, not the features.** Dropping the floor from the target (no_floor) recovers **−75 s** Jan+Jul / **−76 s** Dec. Fitting L2 on raw seconds (raw_target) recovers another ~4 s. Smear factors tell the same story: log-excess smear is 1.64–2.44 per airport (large log-residual bias); `log(y)` smear is 1.00–1.06. Subtracting a geometry p10 and logging the remainder is a worse residualisation than E20's OLS on `(mvt_aobt, aobt_eobt, geo_mean)`. p10 is not the unimpeded taxi time in this information set — `MVT−AOBT` already is.
2. **Best ablation ≈ E20 expert B (direct LGB), which already lost to residual trees.** raw_target Jan+Jul **394.01 / 281.36** matches E20's direct LightGBM expert B at 394.63 / 283.08. C8/E9 already showed residual-on-P_cal beats direct trees on the ranking analogue (matched 256 vs 288). This rebuild never constructs P_cal, so it cannot beat that gap.
3. **Per-airport vs pooled is real but small** on the broken target (−6 s Jan+Jul). Not the loss.
4. **Encodings help ~10 s on the broken target, mixed on December.** Keep as a later add-on to a residual model, not as a replacement for `geo_mean`.
5. **Tight-window congestion and arrival crossings are low-gain.** `dep_rwy_aobt_15m` appears in mid-pack importance; `arr_rwy_*` is at the bottom. Consistent with E4/E5: once clocks are in, 10–15 min counts do not identify the tail (`>30` matched RMSE 1242 vs E20 ~775).
6. **Unmatched: keep LIRF `MVT−SCHED`; do not apply it everywhere.** `all_sched` overall 920 / 809. A non-LIRF unmatched specialist is a small overall help (−1.6 Jan+Jul, −1.6 Dec) with matched unchanged. LIRF unmatched stays 6032.70 — the override is working; the slice is still unsolved.

Leakage tests (synthetic, no parquet): `python -m pytest experiments/test_e25_leakage.py -q` — 10 passed (self-row encodings, time-forward floors, val-y isolation, congestion self/window/runway, smearing vs naive exp, no BLOCK/TAXITIME in feature lists).

## Recommendation

**DO NOT SUBMIT.** Jan+Jul +105.58 (matched +141.66), December +82.78 (matched +84.36) vs E20. Even the best piece of the rebuild (raw L2 per-airport, 394 / 232) is +26 / +4 vs E20 and is a re-discovery of expert B. Production stays the E20 NNLS ensemble (`likable-eagle_v4.parquet`, leaderboard 316.97).

Do not iterate on log(y − p10). If this feature family is reused, put it on a residual-on-P_cal target, not in place of P_cal.
